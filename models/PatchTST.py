import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding, DataEmbedding_wo_pos
import numpy

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # 特征维度的缩放参数

    def forward(self, x):
        # x形状应为 (batch_size, seq_len, d_model) 或 (batch_size * nvars, seq_len, d_model)
        # 计算最后一维的均方根
        rms = torch.sqrt(torch.mean(x **2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight  # 归一化后缩放

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=4, stride=2):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.configs = configs
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.patch_dembedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.data_embedding = DataEmbedding_wo_pos(
            configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.revin = configs.revin
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        self.revin_dlayer = RevIN(configs.dec_in, affine=True, subtract_last=False)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=RMSNorm(configs.d_model)
        )

        # self.conv = nn.Conv1d(in_channels=configs.dec_in, out_channels=configs.enc_in, kernel_size=1)
        # x_dec = self.conv(x_dec)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=RMSNorm(configs.d_model),
            # projection=nn.Linear(configs.d_model, configs.pred_len, bias=True)
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.pred_len -patch_len) /stride +2)    # +configs.label_len
        self.head = FlattenHead(configs.dec_in, self.head_nf, configs.pred_len,    # self.head_nf
                                    head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)


        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)

        # Decoder
        x_dec = x_dec.permute(0, 2, 1)  # (bs, pred_len, nvars_t) -> (bs, nvars_t, pred_len)
        dec_out, _ = self.patch_dembedding(x_dec)  # (bs * nvars, pred_patch_num, d_model)
        dec_out = self.decoder(dec_out, enc_out)

        # z: [bs x nvars x patch_num x d_model]
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        dec_out = dec_out.permute(0, 1, 3, 2)
        dec_out = self.head(dec_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')
            x_dec = self.revin_dlayer(x_dec, 'norm')

        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.revin:
            dec_out = self.revin_layer(dec_out, 'denorm')
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
