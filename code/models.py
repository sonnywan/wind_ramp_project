# -*- coding: utf-8 -*-
"""
时序模型：LSTM 与 TCN（PyTorch，支持 GPU）
=========================================================
输入: (B, T, F)  输出: (B, 3) 三分类 logits (-1/0/+1 -> 0/1/2)
"""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, n_feat, hidden=128, n_layers=2, n_class=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat, hidden_size=hidden, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_class)

    def forward(self, x):
        out, (h, c) = self.lstm(x)       # out: (B, T, H)
        last = out[:, -1, :]             # 取最后一步
        return self.fc(self.dropout(last))


class Chomp1d(nn.Module):
    """裁剪右侧 padding，保证因果（只看过去）。"""
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp].contiguous() if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(n_in, n_out, kernel, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.drop1,
                                 self.conv2, self.chomp2, self.relu2, self.drop2)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNClassifier(nn.Module):
    def __init__(self, n_feat, channels=(64, 64, 128), kernel=3, n_class=3, dropout=0.2):
        super().__init__()
        blocks = []
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            in_ch = n_feat if i == 0 else channels[i - 1]
            blocks.append(TemporalBlock(in_ch, ch, kernel, dilation, dropout))
        self.tcn = nn.Sequential(*blocks)
        self.fc = nn.Linear(channels[-1], n_class)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        y = self.tcn(x)          # (B, C, T)
        y = y[:, :, -1]          # 取最后时间步
        return self.fc(y)


def build_model(name, n_feat, n_class=3):
    if name == "lstm":
        return LSTMClassifier(n_feat, n_class=n_class)
    if name == "tcn":
        return TCNClassifier(n_feat, n_class=n_class)
    raise ValueError(name)