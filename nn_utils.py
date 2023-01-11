import torch
import torch.nn as nn


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


LRELU_SLOPE = 0.1


class AddSkipConn(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.add.add(x, self.net(x))


class ConcatSkipConn(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.cat([x, self.net(x)], 1)
