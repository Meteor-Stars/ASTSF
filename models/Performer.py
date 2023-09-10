# Copyright (c) 2019-present, Royal Bank of Canada.
# Copyright (c) 2021 THUML @ Tsinghua University
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the Autoformer (https://arxiv.org/pdf/2106.13008.pdf) implementation
# from https://github.com/thuml/Autoformer by THUML @ Tsinghua University
####################################################################################
import time

import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import PerformerLayer
from layers.Embed import DataEmbedding

import torch.nn.functional as F
import torch.nn.functional as f


def sample_gumbel(shape, eps=1e-20,device='cuda:2'):

    U = torch.rand(shape).to(device)
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=1e-10,device='cuda:2'):
    sample = sample_gumbel(logits.size(), eps=eps,device=device)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)
def gumbel_softmax(logits, temperature=1.0, hard=True, eps=1e-11,device=None):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  # device='cuda:2'
  y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps,device=device)
  if hard:
      shape = logits.size()
      values, k = y_soft.data.max(-1)
      y=k

  else:
      y = y_soft
  return y


class ScaleForecast(nn.Module):

    def __init__(self, thetas_dim, configs=None):
        super().__init__()
        self.configs = configs
        self.thetas_dim=thetas_dim
        hid_dim=thetas_dim
        units=thetas_dim//2
        self.fc1 = nn.Linear(thetas_dim, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.theta_b_fc = nn.Linear(units, hid_dim, bias=False)
        self.theta_f_fc = nn.Linear(units, hid_dim, bias=False)
        self.forecast_fc = nn.Linear(hid_dim, self.configs.enc_in)
        # self.forecast_fc = nn.Linear(hid_dim, 1)
        self.flatten = nn.Flatten(start_dim=-2)

        self.actv={'relu':F.relu,'sft':F.softplus}
    def forward(self, x):

        act = 'relu'
        x=self.actv[act](self.fc1(x))
        x=self.actv[act](self.fc2(x))
        x = self.actv[act](self.fc3(x))
        theta_f = self.actv[act](self.theta_f_fc(x))
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.
        return forecast
class Model(nn.Module):
    """
    Reformer with O(LlogL) complexity
    - It is notable that Reformer is not proposed for time series forecasting, in that it cannot accomplish the cross attention.
    - Here is only one adaption in BERT-style, other possible implementations can also be acceptable.
    - The hyper-parameters, such as bucket_size and n_hashes, need to be further tuned.
    The official repo of Reformer (https://github.com/lucidrains/reformer-pytorch) can be very helpful, if you have any questiones.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    PerformerLayer(None, configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

        self.hidden_sizes = [128, 256, 128]
        self.kernel_sizes = [9, 5, 3]
        self._build_model(self.configs.enc_in, self.hidden_sizes, self.kernel_sizes)
        self.flat = nn.Flatten(start_dim=-2)
        self.scale_fore_x = ScaleForecast(self.configs.seq_len, configs=self.configs)
        self.scale_sel = ScaleForecast(self.configs.seq_len, configs=self.configs)
        self.scale_sel_prob = nn.Linear(1, 2)
        self.sigmoid = nn.Sigmoid()
        self.sftp = nn.Softplus()
    def _build_model(self,input_size, hidden_sizes, kernel_sizes):

        self.conv1 = nn.Conv1d(in_channels=input_size,
                               out_channels=hidden_sizes[0],
                               kernel_size=kernel_sizes[0])

        self.conv2 = nn.Conv1d(in_channels=hidden_sizes[0],
                               out_channels=hidden_sizes[1],
                               kernel_size=kernel_sizes[1])

        self.conv3 = nn.Conv1d(in_channels=hidden_sizes[1],
                               out_channels=hidden_sizes[2],
                               kernel_size=kernel_sizes[2])

        self.norm1 = nn.BatchNorm1d(num_features=hidden_sizes[0])
        self.norm2 = nn.BatchNorm1d(num_features=hidden_sizes[1])
        self.norm3 = nn.BatchNorm1d(num_features=hidden_sizes[2])
    def get_htensor(self, x):
        h = x
        h = f.pad(h, (int(self.kernel_sizes[0]/2), int(self.kernel_sizes[0]/2)), "constant", 0)
        h = f.relu(self.norm1(self.conv1(h)))
        h = f.pad(h, (int(self.kernel_sizes[1]/2), int(self.kernel_sizes[1]/2)), "constant", 0)
        h = f.relu(self.norm2(self.conv2(h)))
        h = f.pad(h, (int(self.kernel_sizes[2]/2), int(self.kernel_sizes[2]/2)), "constant", 0)
        h = f.relu(self.norm3(self.conv3(h)))
        return h
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                scale_factors=None, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,pretrain=False):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        local_fea = self.get_htensor(x_enc.transpose(1, 2))

        local_fea = F.max_pool1d(
            local_fea.transpose(1, 2),
            kernel_size=128,
        ).transpose(1, 2)

        local_fea = self.flat(local_fea)

        scales_x_or_ = self.scale_fore_x(local_fea)

        scales_x_or = self.sigmoid(scales_x_or_)


        scales_x_or=torch.mean(scales_x_or) ##

        scales_x_sel_prob=scales_x_sel = scales_x_or_
        scales_x_sel_prob = self.scale_sel_prob(scales_x_sel.unsqueeze(-1))

        scales_x_sel_prob = gumbel_softmax(scales_x_sel_prob, device=self.configs.device)

        _,scales_x_sel_prob=scales_x_sel_prob.max(-1) ##
        scales_x_sel_prob=scales_x_sel_prob.unsqueeze(-1)##

        scales_false = scales_x_sel_prob == 0

        scales_x_ones = torch.ones_like(scales_x_or)

        if self.configs.sel_scale:
            scales_x_or = torch.where(scales_false, scales_x_ones, scales_x_or)

        scale_factors_ = scale_factors

        scales_x = scale_factors_ * scales_x_or

        if self.configs.sel_scale:
            scales_x = torch.where(scales_false, scale_factors, scales_x)
        scales_x = scales_x.unsqueeze(1).repeat(1, self.configs.seq_len, 1)
        if self.configs.scale_adaptive:

            x_enc=x_enc/scales_x

        seq_last = x_enc[:, -1:, :].detach()
        x_enc=x_enc-seq_last

        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)

        x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)

        # Performer: encoder only
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        enc_out, attns = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out[:, -self.pred_len:, :], attns
        else:
            return enc_out[:, -self.pred_len:, :]+seq_last,scales_x_or   # [B, L, D]
