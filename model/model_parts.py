import torch
import torch.nn as nn


# class ConvReluBlock1d(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size, dilation):
#         super(ConvReluBlock1d, self).__init__()
#         self.convrelu = nn.Sequential(
#             nn.ZeroPad2d((kernel_size//2, kernel_size//2, 0, 0)),
#             nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, kernel_size), stride=1, dilation=dilation),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU()
#             )

#     def forward(self, x):
#         return self.convrelu(x)


class ConvReluBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super(ConvReluBlock1d, self).__init__()
        self.convrelu = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=kernel_size//2, stride=1, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
            )

    def forward(self, x):
        return self.convrelu(x)


class LinReluBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LinReluBlock, self).__init__()
        self.linrelu = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=in_ch, out_features=out_ch),
            nn.ReLU(),
            nn.Linear(in_features=out_ch, out_features=out_ch),
            nn.Sigmoid()
            )

    def forward(self, x):
        return self.linrelu(x)