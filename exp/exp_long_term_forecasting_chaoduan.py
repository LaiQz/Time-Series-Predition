import matplotlib.pyplot as plt
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import *
import torch
import torch.nn as nn
from torch import optim
import os
import math
import time
import warnings
import numpy as np
from scipy import stats
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        diff = pred - target
        # log(cosh(x)) = log(exp(x) + exp(-x)) - log(2)
        # 用 logaddexp 处理大值情况
        return torch.mean(torch.logaddexp(diff, -diff) - math.log(2))

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        # total_params = sum(p.numel() for p in model.parameters())
        # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(total_params, trainable_params)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = LogCoshLoss()#nn.L1Loss()   #nn.MSELoss() #
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)[:, -self.args.day_len:]

                # decoder input
                if self.args.future:
                    # diff = torch.diff(batch_x[:,-96:, -8:], dim=1)
                    # x = batch_x[:, :, -4:].reshape(32, 96, 4, 4).mean(axis=2)
                    # m12 = x.reshape(32, 8, 12, 4).mean(axis=2).repeat(1, 12, 1)
                    # m24 = x.reshape(32, 4, 24, 4).mean(axis=2).repeat(1, 24, 1)
                    # mean_inp = torch.cat([m12, m24], dim=-1)[:, :, 1:]
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, self.args.fea_t:]).float()
                    dec_inp = torch.cat([batch_y[:, -self.args.pred_len:, :self.args.fea_t].float(), dec_inp], dim=2).to(self.device)
                    # dec_inp = torch.cat((mean_inp, dec_inp), dim=2).float().to(self.device)
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
                outputs = outputs[:, -1, f_dim]
                batch_y = batch_y[:, -1, f_dim].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()
                # pred_diff = torch.diff(pred, dim=1)
                # true_diff = torch.diff(true, dim=1)

                loss = criterion(pred, true) #+criterion(pred_diff, true_diff) *0.2

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
                batch_y_mark = batch_y_mark.float().to(self.device)[:, -self.args.day_len:]

                # decoder input
                if self.args.future:
                    # diff = torch.diff(batch_x[:,-96:, -8:], dim=1)
                    # x = batch_x[:,:, -4:].reshape(32, 96, 4, 4).mean(axis=2)
                    # m12 = x.reshape(32, 8, 12, 4).mean(axis=2).repeat(1, 12, 1)
                    # m24 = x.reshape(32, 4, 24, 4).mean(axis=2).repeat(1, 24, 1)
                    # mean_inp = torch.cat([m12, m24], dim=-1)[:,:,1:]
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, self.args.fea_t:]).float()
                    dec_inp = torch.cat([batch_y[:, -self.args.pred_len:, :self.args.fea_t].float(), dec_inp], dim=2).to(self.device)
                    # dec_inp = torch.cat((mean_inp, dec_inp), dim=2).float().to(self.device)
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
                    outputs = outputs[:, -1, f_dim]
                    batch_y = batch_y[:, -1, f_dim].to(self.device)
                    # pred_diff = torch.diff(outputs, dim=1)
                    # true_diff = torch.diff(batch_y, dim=1)

                    loss = criterion(outputs, batch_y) #+criterion(pred_diff, true_diff) *0.2
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
        time_now = time.time()
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
                batch_y_mark = batch_y_mark.float().to(self.device)[:, -self.args.day_len:]

                # decoder input
                if self.args.future:
                    # diff = torch.diff(batch_x[:,-96:, -8:], dim=1)
                    # x = batch_x[:, :, -4:].reshape(32, 96, 4, 4).mean(axis=2)
                    # m12 = x.reshape(32, 8, 12, 4).mean(axis=2).repeat(1, 12, 1)
                    # m24 = x.reshape(32, 4, 24, 4).mean(axis=2).repeat(1, 24, 1)
                    # mean_inp = torch.cat([m12, m24], dim=-1)[:, :, 1:]
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, self.args.fea_t:]).float()
                    dec_inp = torch.cat([batch_y[:, -self.args.pred_len:, :self.args.fea_t].float(), dec_inp], dim=2).to(self.device)
                    # dec_inp = torch.cat((mean_inp, dec_inp), dim=2).float().to(self.device)
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

                outputs = outputs[:, -1, f_dim]
                batch_y = batch_y[:, -1, f_dim]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        cost = time.time() -time_now
        print('cost time:', cost)
        preds = np.concatenate(preds, axis=0).clip(min=0)[:2880].reshape(-1,96)
        trues = np.concatenate(trues, axis=0)[:2880].reshape(-1,96)
        print('test shape:', preds.shape)

        # cost = time.time() - time_now

        # indices = np.arange(self.args.zzz, preds.shape[0] -1, self.args.pred_len)
        # slices_p = [preds[i] for i in indices]
        # preds = np.flip(np.array(slices_p), axis=0)
        # slices_t = [trues[i] for i in indices]
        # trues = np.flip(np.array(slices_t), axis=0)
        # print('test shape:', preds.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        # np.savetxt(f"D:\github code\\baka\Time-Series-Library-main\pred\pred_{self.args.aaa}.txt", preds, fmt="%.4f", delimiter=",")
        # np.savetxt("D:\github code\\baka\Time-Series-Library-main\pred\\true.txt", trues, fmt="%.4f", delimiter=",")

        kaohes = []
        alpha_counts = []  # 记录每个样本的alpha值
        count_07_total = 0  # 统计0.7的总数
        count_total = 0  # 统计总数

        for m in range(30):
            sample_alphas = []  # 记录当前样本的所有alpha值
            for n in range(96):
                pred = preds[m,n]
                true = trues[m,n]

                p_delta = max(true *0.12, 1)
                zzz = np.abs(true -pred)
                pm = max(0.5* self.args.Cap, true)
                if zzz <= p_delta:
                    alpha = 0
                elif zzz / pm < 0.25:
                    alpha = 0.07
                elif zzz / pm < 0.5:
                    alpha = 0.2
                elif zzz / pm < 0.7:
                    alpha = 0.4
                elif zzz / pm < 0.9:
                    alpha = 0.7
                else:
                    alpha = 1

                kaohe = (np.abs(pred-true) -p_delta) *alpha
                kaohes.append(kaohe)
        kaohes = sum(np.array(kaohes))
        chaoduan = kaohes /sum(sum(trues[:30]))
        print('ultra-short:(<0.08)',chaoduan)

        # for i in range(73):
        #     plt.figure(figsize=(12, 6))
        #     plt.plot(preds[i], label='Predicted Values', color='r')#, alpha=0.5)
        #     plt.plot(trues[i], label='True Values', color='b')
        #     plt.legend()
        #     output_file = f".\pre\plot_{i}.pdf"
        #     plt.savefig(output_file, format='pdf')

        # np.savetxt(f"D:\github code\\baka\Time-Series-Library-main\data\\pred_{self.args.aaa}.txt", preds, fmt="%.4f", delimiter=",")
        # np.savetxt("D:\github code\\baka\Time-Series-Library-main\data\\true.txt", trues, fmt="%.4f", delimiter=",")
        mape = MAPE(np.mean(preds, 1), np.mean(trues, 1))
        # preds = preds.T.reshape(-1,24)
        # trues = trues.T.reshape(-1,24)
        trues_mean = np.mean(trues, axis=1)
        preds_mean = np.mean(preds, axis=1)
        acc = Acc(preds_mean, trues_mean, self.args.Cap)
        preds_fla = preds_mean.flatten()
        trues_fla = trues_mean.flatten()
        ccc = np.abs(preds_fla - trues_fla) / self.args.Cap
        ccc = 1 - ccc
        condition_met = np.abs(preds_fla - trues_fla) / 100 <= 0.2
        count = np.sum(condition_met)
        qr = count / len(preds_fla)
        r, _ = stats.pearsonr(preds_fla, trues_fla)
        print('Day -Mean:{}, Max:{}, Min:{}'.format(acc, np.max(ccc), np.min(ccc)))
        np.savetxt(f"D:\github code\\baka\Time-Series-Library-main\pred\\acc\\acc.txt", ccc, fmt="%.4f", delimiter=",")
        np.savetxt(f"D:\github code\\baka\Time-Series-Library-main\pred\\acc\\pred.txt", preds_fla, fmt="%.4f", delimiter=",")
        np.savetxt(f"D:\github code\\baka\Time-Series-Library-main\pred\\acc\\true.txt", trues_fla, fmt="%.4f", delimiter=",")
        print('MAE:{}, RMSE:{}, MAPE:{}, ACC:{}, QR:{}, R:{}'.format(mae, rmse, mape, acc, qr, r))

        return chaoduan
