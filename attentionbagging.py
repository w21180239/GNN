import random

import torch
import torch.nn as nn


class AttentionBagging(nn.Module):
    def __init__(self, input_dim, output_dim, bagging_num, bagging_rate, prob_mode=False, attention_mode=True,
                 hidden_width=None):
        super(AttentionBagging, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bagging_num = bagging_num
        self.bagging_rate = bagging_rate
        self.use_attention = attention_mode
        if hidden_width is None:
            hidden_width = [32 for i in range(1, self.bagging_num + 1)]
        if prob_mode:
            self.bagging_decoder_list = nn.ModuleList(
                [nn.Sequential(nn.Linear(int(self.input_dim * self.bagging_rate), h),
                               nn.ReLU6(), nn.Linear(h, self.output_dim), nn.Softmax()).cuda() for h
                 in hidden_width])
        else:
            self.bagging_decoder_list = nn.ModuleList(
                [nn.Sequential(nn.Linear(int(self.input_dim * self.bagging_rate), h),
                               nn.ReLU6(), nn.Linear(h, self.output_dim)).cuda() for h
                 in hidden_width])
        self.index_list = self.random_select(self.bagging_rate, self.input_dim, self.bagging_num)
        self.attention_weight = nn.Sequential(nn.Linear(self.input_dim, self.bagging_num), nn.Softmax())

    @staticmethod
    def random_select(rate, input_dim, times):
        total = [i for i in range(input_dim)]
        re = []
        out_num = int(rate * input_dim)

        for _ in range(times):
            random.shuffle(total)
            re.append(total[:out_num])
        return re

    def forward(self, _input):
        bagging = [
            list(self.bagging_decoder_list)[i](
                torch.index_select(_input, 1, torch.LongTensor(self.index_list[i]).cuda()))
            for
            i in range(self.bagging_num)]

        if self.use_attention:
            weight = self.attention_weight(_input)
            weight = torch.unsqueeze(weight, -1)
            weight = weight.expand(weight.size(0), weight.size(1), self.output_dim)
            for i in range(len(bagging)):
                bagging[i] = bagging[i] * weight[:, i, :]
        output = torch.squeeze(sum(bagging))
        return bagging, output
