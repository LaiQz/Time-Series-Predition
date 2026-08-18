from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import *
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from scipy import stats
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.L1Loss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)[:, -96:]

                # decoder input
                if self.args.future:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, self.args.fea_t:]).float()
                    dec_inp = torch.cat([batch_y[:, -self.args.pred_len:, :self.args.fea_t].float(), dec_inp], dim=2).to(self.device)
                    # batch_x = torch.cat((batch_x, dec_inp), dim=1).float().to(self.device)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)[:, -96:]

                # decoder input
                if self.args.future:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, self.args.fea_t:]).float()
                    dec_inp = torch.cat([batch_y[:, -self.args.pred_len:, :self.args.fea_t].float(), dec_inp], dim=2).to(self.device)
                    # batch_x = torch.cat((batch_x, dec_inp), dim=1).float().to(self.device)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)[:, -96:]

                # decoder input
                if self.args.future:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, self.args.fea_t:]).float()
                    dec_inp = torch.cat([batch_y[:, -self.args.pred_len:, :self.args.fea_t].float(), dec_inp], dim=2).to(self.device)
                    # batch_x = torch.cat((batch_x, dec_inp), dim=1).float().to(self.device)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                outputs = outputs[:, :, f_dim]
                batch_y = batch_y[:, :, f_dim]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0).clip(min=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape)
        # cost = time.time() - time_now

        indices = np.arange(0, preds.shape[0], self.args.pred_len)
        slices_p = [preds[i] for i in indices]
        preds = np.array(slices_p)
        slices_t = [trues[i] for i in indices]
        trues = np.array(slices_t)
        print('test shape:', preds.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        # np.savetxt(f"D:\github code\\baka\Time-Series-Library-main\pred\\pred_0{self.args.aaa}.txt", preds, fmt="%.4f", delimiter=",")
        # np.savetxt("D:\github code\\baka\Time-Series-Library-main\pred\\true.txt", trues, fmt="%.4f", delimiter=",")
        mape = MAPE(np.mean(preds, 1), np.mean(trues, 1))
        # preds = preds.T.reshape(-1,24)
        # trues = trues.T.reshape(-1,24)
        trues_mean = np.mean(trues, axis=1)
        preds_mean = np.mean(preds, axis=1)
        acc = Acc(preds_mean, trues_mean, self.args.Cap)
        preds_fla = preds_mean.flatten()
        trues_fla = trues_mean.flatten()
        ccc = np.abs(preds_fla -trues_fla) /self.args.Cap
        ccc = 1 -ccc
        condition_met = np.abs(preds_fla -trues_fla) /self.args.Cap <=0.2
        count = np.sum(condition_met)
        qr = count /len(preds_fla)
        r, _ = stats.pearsonr(preds_fla, trues_fla)
        print('Day -Mean:{}, Max:{}, Min:{}'.format(acc, np.max(ccc), np.min(ccc)))
        print('MAE:{}, RMSE:{}, MAPE:{}, ACC:{}, QR:{}, R:{}'.format(mae, rmse, mape, acc, qr, r))

        rmse_h = np.sqrt(np.mean((preds -trues) **2, axis=1))
        Acc_h = 1 -rmse_h /self.args.Cap
        count = np.sum(Acc_h >=0.80)
        ra = count /preds.shape[0]
        print('24h -Max:{}, Min:{}, mean:{}, ratio:{}'.format(np.max(Acc_h), np.min(Acc_h), np.mean(Acc_h), ra))

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return np.min(ccc)
