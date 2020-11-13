import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

def grad_scale(x, scale):
    yOut = x
    yGrad = x*scale
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def round_pass(x):
    yOut = x.round()
    yGrad = x
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def quantizeLSQ(v, s, p, isActivation=False):
    if isActivation:
        Qn = 0
        Qp = 2**p - 1
        gradScaleFactor = 1.0 / math.sqrt(v.numel() * Qp)
    else: # is weight
        Qn = -2**(p-1)
        Qp = 2**(p-1) - 1
        gradScaleFactor = 1.0 / math.sqrt(v.numel() * Qp)

    #quantize
    s = grad_scale(s, gradScaleFactor)
    vbar = round_pass((v/s).clamp(Qn, Qp))
    vhat = vbar*s
    return vhat

class _Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(_Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.nbits = kwargs['nbits']
        self.step_size = Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

class _LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(_LinearQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.nbits = kwargs['nbits']
        self.step_size= Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

class _ActQ(nn.Module):
    def __init__(self, **kwargs_q):
        super(_ActQ, self).__init__()
        self.nbits = kwargs_q['nbits']
        self.step_size = Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

class Conv2dLSQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=8):
        super(Conv2dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits)

    def forward(self, x):
        if self.init_state == 0:
            self.step_size.data.copy_(2 * self.weight.abs().mean() / math.sqrt(2 ** (self.nbits - 1) - 1))
            self.init_state.fill_(1)

        w_q = quantizeLSQ(self.weight, self.step_size, self.nbits)

        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class LinearLSQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=8):
        super(LinearLSQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits)

    def forward(self, x):
        if self.init_state == 0:
            self.step_size.data.copy_(2 * self.weight.abs().mean() / math.sqrt(2 ** (self.nbits - 1) - 1))
            self.init_state.fill_(1)

        w_q = quantizeLSQ(self.weight, self.step_size, self.nbits)

        return F.linear(x, w_q, self.bias)

class ActLSQ(_ActQ):
    def __init__(self, nbits=8):
        super(ActLSQ, self).__init__(nbits=nbits)

    def forward(self, x):
        if self.init_state == 0:
            self.step_size.data.copy_(2 * x.abs().mean() / math.sqrt(2 ** self.nbits - 1))
            self.init_state.fill_(1)

        x_q = quantizeLSQ(x, self.step_size, self.nbits, isActivation=True)

        return x_q
