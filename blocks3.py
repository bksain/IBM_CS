import torch
import torch.nn as nn
import scipy as misc
from collections import OrderedDict
import sys


################
# Basic blocks
################

def activation(act_type='relu', inplace=True, slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    layer = None
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(slope, inplace)
    elif act_type == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    elif act_type =='tanh':
        layer = nn.Tanh()
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!' % act_type)
    return layer


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    layer = None
    if norm_type == 'bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError('[ERROR] Normalization layer [%s] is not implemented!' % norm_type)
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None

    layer = None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('[ERROR] Padding layer [%s] is not implemented!' % pad_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict' % sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0,
              act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!' % sys.modules[__name__]

    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, conv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, conv)


def ConvBlock2(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0, pad_type='zero'):
    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    return sequential(p, conv)


def ConvrelBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0, pad_type='zero', act_type='relu'):
    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    act = activation(act_type) if act_type else None

    return sequential(p, conv, act)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * 255. * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        for p in self.parameters():
            p.requires_grad = False


################
# Advanced blocks
################

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channle, mid_channel, kernel_size, stride=1, valid_padding=True, padding=0, dilation=1, bias=True, \
                 pad_type='zero', norm_type='bn', act_type='relu', mode='CNA', res_scale=1):
        super(ResBlock, self).__init__()
        conv0 = ConvBlock(in_channel, mid_channel, kernel_size, stride, dilation, bias, valid_padding, padding, act_type, norm_type, pad_type, mode)
        act_type = None
        norm_type = None
        conv1 = ConvBlock(mid_channel, out_channle, kernel_size, stride, dilation, bias, valid_padding, padding, act_type, norm_type, pad_type, mode)
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class UpprojBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, valid_padding=False, padding=0, bias=True, \
                 pad_type='zero', norm_type=None, act_type='prelu'):
        super(UpprojBlock, self).__init__()

        self.deconv_1 = DeconvBlock(in_channel, out_channel, kernel_size, stride=stride, \
                                    padding=padding, norm_type=norm_type, act_type=act_type)

        self.conv_1 = ConvBlock(out_channel, out_channel, kernel_size, stride=stride, padding=padding, \
                                valid_padding=valid_padding, norm_type=norm_type, act_type=act_type)

        self.deconv_2 = DeconvBlock(out_channel, out_channel, kernel_size, stride=stride, \
                                    padding=padding, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        H_0_t = self.deconv_1(x)
        L_0_t = self.conv_1(H_0_t)
        H_1_t = self.deconv_2(L_0_t - x)

        return H_0_t + H_1_t


class D_UpprojBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, valid_padding=False, padding=0, bias=True, \
                 pad_type='zero', norm_type=None, act_type='prelu'):
        super(D_UpprojBlock, self).__init__()
        self.conv_1 = ConvBlock(in_channel, out_channel, kernel_size=1, norm_type=norm_type, act_type=act_type)
        self.deconv_1 = DeconvBlock(out_channel, out_channel, kernel_size, stride=stride, \
                                    padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv_2 = ConvBlock(out_channel, out_channel, kernel_size, stride=stride, padding=padding, \
                                valid_padding=valid_padding, norm_type=norm_type, act_type=act_type)
        self.deconv_2 = DeconvBlock(out_channel, out_channel, kernel_size, stride=stride, \
                                    padding=padding, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        x = self.conv_1(x)
        H_0_t = self.deconv_1(x)
        L_0_t = self.conv_2(H_0_t)
        H_1_t = self.deconv_2(L_0_t - x)

        return H_1_t + H_0_t


class DownprojBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, valid_padding=True,
                 padding=0, dilation=1, bias=True, \
                 pad_type='zero', norm_type=None, act_type='prelu', mode='CNA', res_scale=1):
        super(DownprojBlock, self).__init__()

        self.conv_1 = ConvBlock(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                valid_padding=valid_padding, norm_type=norm_type, act_type=act_type)

        self.deconv_1 = DeconvBlock(out_channel, out_channel, kernel_size, stride=stride,
                                    padding=padding, norm_type=norm_type, act_type=act_type)

        self.conv_2 = ConvBlock(out_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                valid_padding=valid_padding, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        L_0_t = self.conv_1(x)
        H_0_t = self.deconv_1(L_0_t)
        L_1_t = self.conv_2(H_0_t - x)

        return L_0_t + L_1_t


class D_DownprojBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, valid_padding=False, padding=0, bias=True, \
                 pad_type='zero', norm_type=None, act_type='prelu'):
        super(D_DownprojBlock, self).__init__()
        self.conv_1 = ConvBlock(in_channel, out_channel, kernel_size=1, norm_type=norm_type, act_type=act_type)

        self.conv_2 = ConvBlock(out_channel, out_channel, kernel_size, stride=stride, padding=padding, \
                                valid_padding=valid_padding, norm_type=norm_type, act_type=act_type)
        self.deconv_1 = DeconvBlock(out_channel, out_channel, kernel_size, stride=stride, \
                                    padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv_3 = ConvBlock(out_channel, out_channel, kernel_size, stride=stride, padding=padding, \
                                valid_padding=valid_padding, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        x = self.conv_1(x)
        L_0_t = self.conv_2(x)
        H_0_t = self.deconv_1(L_0_t)
        L_1_t = self.conv_3(H_0_t - x)

        return L_1_t + L_0_t


class DensebackprojBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, bp_stages, stride=1, valid_padding=True,
                 padding=0, dilation=1, bias=True, \
                 pad_type='zero', norm_type=None, act_type='prelu', mode='CNA', res_scale=1):
        super(DensebackprojBlock, self).__init__()

        # This is an example that I have to create nn.ModuleList() to append a sequence of models instead of list()
        self.upproj = nn.ModuleList()
        self.downproj = nn.ModuleList()
        self.bp_stages = bp_stages
        self.upproj.append(UpprojBlock(in_channel, out_channel, kernel_size, stride=stride, valid_padding=False,
                                       padding=padding, norm_type=norm_type, act_type=act_type))

        for index in range(self.bp_stages - 1):
            if index < 1:
                self.upproj.append(UpprojBlock(out_channel, out_channel, kernel_size, stride=stride, valid_padding=False,
                                               padding=padding, norm_type=norm_type, act_type=act_type))
            else:
                uc = ConvBlock(out_channel * (index + 1), out_channel, kernel_size=1, norm_type=norm_type, act_type=act_type)
                u = UpprojBlock(out_channel, out_channel, kernel_size, stride=stride, valid_padding=False,
                                padding=padding, norm_type=norm_type, act_type=act_type)
                self.upproj.append(sequential(uc, u))

            if index < 1:
                self.downproj.append(DownprojBlock(out_channel, out_channel, kernel_size, stride=stride, valid_padding=False,
                                                   padding=padding, norm_type=norm_type, act_type=act_type))
            else:
                dc = ConvBlock(out_channel * (index + 1), out_channel, kernel_size=1, norm_type=norm_type, act_type=act_type)
                d = DownprojBlock(out_channel, out_channel, kernel_size, stride=stride, valid_padding=False,
                                  padding=padding, norm_type=norm_type, act_type=act_type)
                self.downproj.append(sequential(dc, d))

    def forward(self, x):
        low_features = []
        high_features = []

        H = self.upproj[0](x)
        high_features.append(H)

        for index in range(self.bp_stages - 1):
            if index < 1:
                L = self.downproj[index](H)
                low_features.append(L)
                H = self.upproj[index + 1](L)
                high_features.append(H)
            else:
                H_concat = torch.cat(tuple(high_features), 1)
                L = self.downproj[index](H_concat)
                low_features.append(L)
                L_concat = torch.cat(tuple(low_features), 1)
                H = self.upproj[index + 1](L_concat)
                high_features.append(H)

        output = torch.cat(tuple(high_features), 1)
        return output


class ResidualDenseBlock_8C(nn.Module):
    '''
    Residual Dense Block
    style: 8 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
        super(ResidualDenseBlock_8C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = ConvBlock(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = ConvBlock(nc + gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = ConvBlock(nc + 2 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = ConvBlock(nc + 3 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv5 = ConvBlock(nc + 4 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv6 = ConvBlock(nc + 5 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv7 = ConvBlock(nc + 6 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv8 = ConvBlock(nc + 7 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv9 = ConvBlock(nc + 8 * gc, nc, 1, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x6 = self.conv6(torch.cat((x, x1, x2, x3, x4, x5), 1))
        x7 = self.conv7(torch.cat((x, x1, x2, x3, x4, x5, x6), 1))
        x8 = self.conv8(torch.cat((x, x1, x2, x3, x4, x5, x6, x7), 1))
        x9 = self.conv9(torch.cat((x, x1, x2, x3, x4, x5, x6, x7, x8), 1))
        return x9.mul(0.2) + x


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


class ConcatBlock(nn.Module):
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), 1)
        return output


################
# Upsampler
################
def UpsampleConvBlock(upscale_factor, in_channels, out_channels, kernel_size, stride, valid_padding=True, padding=0, bias=True, \
                      pad_type='zero', act_type='relu', norm_type=None, mode='nearest'):
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = ConvBlock(in_channels, out_channels, kernel_size, stride, bias=bias, valid_padding=valid_padding, padding=padding, \
                     pad_type=pad_type, act_type=act_type, norm_type=norm_type)
    return sequential(upsample, conv)


def PixelShuffleBlock():
    pass


def DeconvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, padding=0, \
                act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!' % sys.modules[__name__]

    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, deconv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, deconv)


################
# ADMM module
################
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


'''
def A(input, scale_factor):

    out = nn.functional.interpolate(input, scale_factor=1/scale_factor, mode='bicubic', align_corners=False)
    #out = misc.imresize(HR_img.numpy(), 1 / scale_factor, interp='bicubic')

    return out
'''


def downs(in_channels, out_channels, kernel_size, scale,bias=True, valid_padding=True, padding=0, pad_type='zero', act_type='relu'):
    if scale == 2:
        down1 = ConvrelBlock(in_channels, out_channels, 6, stride=2, bias=bias, valid_padding=False, padding=2, pad_type=pad_type)
        #sequential(down1) # O = (I - K + 2p)/S +1
    elif scale == 5:
        down1 = ConvrelBlock(in_channels, out_channels, 9, stride=5, bias=bias, valid_padding=False, padding=2, pad_type=pad_type)
    elif scale == 3:
        down1 = ConvrelBlock(in_channels, out_channels, 7, stride=3, bias=bias, valid_padding=False, padding=2, pad_type=pad_type)
    elif scale == 11:
        down1 = ConvrelBlock(in_channels, out_channels, 11, stride=1, bias=bias, valid_padding=False, padding=0, pad_type=pad_type)
    elif scale == 33:
        down1 = ConvrelBlock(in_channels, out_channels, 33, stride=1, bias=bias, valid_padding=False, padding=0, pad_type=pad_type)
    return sequential(down1)

def downs_con(in_channels, out_channels, kernel_size, scale,bias=True, valid_padding=True, padding=0, pad_type='zero', act_type='relu'):
    if scale == 2:
        down1 = ConvBlock2(in_channels, out_channels, 6, stride=2, bias=bias, valid_padding=False, padding=2, pad_type=pad_type)
        #sequential(down1) # O = (I - K + 2p)/S +1
    elif scale == 5:
        down1 = ConvBlock2(in_channels, out_channels, 9, stride=5, bias=bias, valid_padding=False, padding=2, pad_type=pad_type)
    elif scale == 3:
        down1 = ConvBlock2(in_channels, out_channels, 7, stride=3, bias=bias, valid_padding=False, padding=2, pad_type=pad_type)
    elif scale == 11:
        down1 = ConvBlock2(in_channels, out_channels, 15, stride=11, bias=bias, valid_padding=False, padding=2, pad_type=pad_type)
    elif scale == 33:
        down1 = ConvBlock2(in_channels, out_channels, 33, stride=1, bias=bias, valid_padding=False, padding=0, pad_type=pad_type)
    return sequential(down1)


def ups(in_channels, out_channels, scale, stride=1, dilation=1, bias=True, valid_padding=True, padding=0, pad_type='zero', act_type='relu'):
    if scale == 2:
        down1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=6 , stride=2, padding=2, bias=bias) # -stride -2padding + k = 0
        act = activation(act_type) if act_type else None
        return sequential(down1, act)
    elif scale == 5:
        down1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=9 , stride=5, padding=2, bias=bias)
        act = activation(act_type) if act_type else None
        return sequential(down1, act)
    elif scale == 3:
        down1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=7 , stride=3, padding=2, bias=bias)
        act = activation(act_type) if act_type else None
        return sequential(down1, act)
    elif scale == 11:
        down1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=15 , stride=11, padding=2, bias=bias)
        act = activation(act_type) if act_type else None
        return sequential(down1, act)



class A(nn.Module):
    def __init__(self, in_channels, out_channels, F, kernel_size1, kernel_size2, stride=1, dilation=1, bias=True, padding=0, scale=4, act_type='relu', pad_type='zero'):
        super(A, self).__init__()
        self.scale = scale

    def forward(self, x):
        inter_res = nn.functional.interpolate(x, scale_factor=1 / self.scale, mode='bicubic', align_corners=False)
        # inter_res = x

        return inter_res


class AT(nn.Module):
    def __init__(self, in_channels, out_channels, F, kernel_size1, kernel_size2, stride=1, dilation=1, bias=True, padding=0, scale=4, act_type='relu', pad_type='zero'):
        super(AT, self).__init__()
        self.scale = scale

    def forward(self, x):
        inter_res = nn.functional.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)

        return inter_res


class F_(nn.Module):
    def __init__(self, in_channels, out_channels, F, kernel_size1, kernel_size2, stride=1, dilation=1, bias=True, padding=0, scale=4, act_type='relu', pad_type='zero'):
        super(F_, self).__init__()
        self.scale = scale

        self.conv0 = ConvrelBlock(F, F, 1, stride, bias=bias, pad_type=pad_type)
        self.conv1 = ConvrelBlock(F, F, kernel_size1, stride, bias=bias, pad_type=pad_type)
        self.conv2 = ConvrelBlock(F, out_channels, kernel_size1, stride, bias=bias, pad_type=pad_type)

    def forward(self, x):
        x1 = self.conv0(x)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)

        return x1


class FT(nn.Module):
    def __init__(self, in_channels, out_channels, F, kernel_size1, kernel_size2, stride=1, dilation=1, bias=True, padding=0, scale=4, act_type='relu', pad_type='zero'):
        super(FT, self).__init__()
        self.scale = scale

        self.conv0 = ConvrelBlock(in_channels, F, kernel_size1, stride, bias=bias, pad_type=pad_type)
        self.conv1 = ConvrelBlock(F, F, kernel_size1, stride, bias=bias, pad_type=pad_type)
        self.conv2 = ConvrelBlock(F, F, 1, stride, bias=bias, pad_type=pad_type)

    def forward(self, x):
        x1 = self.conv0(x)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)

        return x1


class D_H(nn.Module):
    def __init__(self, in_channels, out_channels, F, kernel_size1, kernel_size2, stride=1, dilation=1, bias=True, padding=0, scale=4, act_type='relu', pad_type='zero'):
        super(D_H, self).__init__()
        self.scale = scale

        # alpha input
        self.conv0 = ConvrelBlock(F, F, kernel_size1, bias=bias, pad_type=pad_type)

        # # SPADE block without batch normalization (y input)
        # self.conv1 = ConvrelBlock(in_channels, F, kernel_size1, bias=bias, pad_type=pad_type)
        # self.conv2 = ConvrelBlock(F, F, 1, bias=bias, pad_type=pad_type)
        # self.conv3 = ConvrelBlock(F, F*2, kernel_size1, bias=bias, pad_type=pad_type) # number of feature ?
        ## multiply
        # self.conv4 = ConvrelBlock(F*2, F, kernel_size1, bias=bias, pad_type=pad_type)
        ## add
        # self.conv5 = ConvrelBlock(F*2, F, kernel_size1, bias=bias, pad_type=pad_type)

        # to image
        self.ups2 = ups(F, F, kernel_size2, scale=self.scale, stride=stride, padding=padding)
        self.conv6 = ConvrelBlock(F, F, 1, bias=bias, pad_type=pad_type)
        self.conv7 = ConvrelBlock(F, out_channels, kernel_size1, bias=bias, pad_type=pad_type)

    def forward(self, a, DTy):
        a = a + DTy
        x1 = self.conv0(a)

        # SPADE
        # x2 = self.conv1(y)
        # x2 = self.conv2(x2)
        # x2 = self.conv3(x2)
        # mul = self.conv4(x2)
        # add = self.conv5(x2)
        #
        # x2 = x1 * mul + add
        # # SPADE
        # x1 = x1 + x2

        x1 = self.ups2(x1)
        x1 = self.conv6(x1)
        x1 = self.conv7(x1)

        return x1


def con_rel_con(in_channels, out_channels, F, kernel_size, stride=1, dilation=1, bias=True, padding=0, act_type='relu', pad_type='zero'):
    conv1 = ConvBlock2(F, F, kernel_size, stride, bias=bias, valid_padding=True, padding=padding, pad_type=pad_type)
    act1 = activation(act_type, inplace=False) if act_type else None
    conv2 = ConvBlock2(F, F, kernel_size, stride, bias=bias, valid_padding=True, padding=padding, pad_type=pad_type)

    return sequential(conv1, act1, conv2)


class V(nn.Module):
    def __init__(self, in_channels, out_channels, F, kernel_size1,  stride=1, dilation=1, bias=True, padding=0, scale=4, act_type='relu', pad_type='zero'):
        super(V, self).__init__()
        self.scale = scale

        self.iter = 4
        self.DenseL = nn.ModuleList()

        self.convI = ConvrelBlock(in_channels, F, kernel_size1, bias=bias, pad_type=pad_type)

        for i in range(self.iter):
            self.DenseL.append(ConvrelBlock( F * (i+1), F, kernel_size1, bias=bias, pad_type=pad_type))

        self.conv1x1 = ConvrelBlock(F * (self.iter + 1), F, kernel_size1, bias=bias, pad_type=pad_type)

        # self.up = ups(F,F,kernel_size2,stride)

        self.convE = ConvrelBlock(F, F, kernel_size1, bias=bias, pad_type=pad_type)
        self.convO = ConvrelBlock(F, out_channels, kernel_size1, bias=bias, pad_type=pad_type)

    def forward(self, x):

        x = self.convI(x)
        x1 = x
        DenseV = []

        for i in range(self.iter):
            DenseV.append(self.DenseL[i](x1))
            x1 = torch.cat((x1, DenseV[i]), 1)

        x1 = self.conv1x1(x1)
        x1 = x1 + x

        # x1 = self.up(x1)

        x1 = self.convE(x1)
        x1 = self.convO(x1)

        return x1


class ResidualDenseBlock_(nn.Module):
    '''
    Residual Dense Block
    style: 8 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=3, F=64, stride=1, bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
        super(ResidualDenseBlock_, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = ConvrelBlock(nc, F, kernel_size, stride, bias=bias, pad_type=pad_type)
        self.conv2 = ConvrelBlock(nc + F, F, kernel_size, stride, bias=bias, pad_type=pad_type)
        self.conv3 = ConvrelBlock(nc + 2 * F, F, kernel_size, stride, bias=bias, pad_type=pad_type)
        self.conv4 = ConvrelBlock(nc + 3 * F, F, kernel_size, stride, bias=bias, pad_type=pad_type)
        self.conv5 = ConvrelBlock(nc + 4 * F, F, kernel_size, stride, bias=bias, pad_type=pad_type)
        self.conv6 = ConvrelBlock(nc + 5 * F, F, kernel_size, stride, bias=bias, pad_type=pad_type)
        self.conv7 = ConvrelBlock(nc + 6 * F, F, kernel_size, stride, bias=bias, pad_type=pad_type)
        self.conv8 = ConvrelBlock(nc + 7 * F, F, kernel_size, stride, bias=bias, pad_type=pad_type)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv9 = ConvrelBlock(nc + 8 * F, nc, 1, stride, bias=bias, pad_type=pad_type)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x6 = self.conv6(torch.cat((x, x1, x2, x3, x4, x5), 1))
        x7 = self.conv7(torch.cat((x, x1, x2, x3, x4, x5, x6), 1))
        x8 = self.conv8(torch.cat((x, x1, x2, x3, x4, x5, x6, x7), 1))
        x9 = self.conv9(torch.cat((x, x1, x2, x3, x4, x5, x6, x7, x8), 1))
        return x9.mul(0.2) + x


class DilatedResidual_(nn.Module):
    '''
    Residual Dense Block
    style: 8 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=3, F=64, stride=1, bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
        super(DilatedResidual_, self).__init__()
        # gc: growth channel, i.e. intermediate channels

        self.conv0 = ConvrelBlock(F, F, kernel_size, stride, bias=bias, pad_type=pad_type)
        # self.conv1 = ConvrelBlock(F, F, 1, stride, bias=bias, pad_type=pad_type)
        self.conv1 = ConvBlock2(F, F, 1, stride, bias=bias, pad_type=pad_type)

        self.iter = 5

        self.dilconv = nn.ModuleList()

        for i in range(self.iter):
            self.dilconv.append(ConvrelBlock(F, F, kernel_size, stride, bias=bias, pad_type=pad_type, dilation=(i + 1)))

        self.conv2 = ConvrelBlock(F, F, kernel_size, stride, bias=bias, pad_type=pad_type)

        self.conv3 = ConvrelBlock(F, F, 1, stride, bias=bias, pad_type=pad_type)
        self.conv4 = ConvrelBlock(F, F, kernel_size, stride, bias=bias, pad_type=pad_type)

    def forward(self, x):
        x1 = self.conv0(x)
        x1 = self.conv1(x1)

        x2 = x1
        x_ = []

        for i in range(self.iter):
            x2 = self.dilconv[i](x2)
            x_.append(x2)

        x2 = self.conv2(x2)

        for i in range(self.iter):
            x2 = x_[self.iter - 1 - i] + x2
            x2 = self.dilconv[self.iter - 1 - i](x2)

        x2 = x2 + x1
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)

        return x2 + x


class EAM(nn.Module):
    '''
    Residual Dense Block
    style: 8 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=3, F=64, stride=1, scale=4, kernel_size2=8, bias=True, padding=2, pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
        super(EAM, self).__init__()
        # gc: growth channel, i.e. intermediate channels

        # dilation
        self.conv00 = ConvrelBlock(nc, F, kernel_size, bias=bias, pad_type=pad_type)

        self.conv0_0 = ConvrelBlock(F, F, kernel_size, bias=bias, pad_type=pad_type)
        self.conv0_1 = ConvrelBlock(F, F, kernel_size, bias=bias, pad_type=pad_type)

        self.conv1_0 = ConvrelBlock(F, F, kernel_size, bias=bias, pad_type=pad_type, dilation=5)
        self.conv1_1 = ConvrelBlock(F, F, kernel_size, bias=bias, pad_type=pad_type, dilation=7)

        self.conv2 = ConvrelBlock(F * 2, F, kernel_size, bias=bias, pad_type=pad_type)

        # self.down = downs(F,F,kernel_size,scale=scale,stride=stride)
        self.conv3_1 = ConvrelBlock(F, F, kernel_size, bias=bias, pad_type=pad_type)
        self.conv3_2 = ConvrelBlock(F, F // 2, 1, bias=bias, pad_type=pad_type)
        self.conv3_3 = ConvrelBlock(F // 2, F, kernel_size, bias=bias, pad_type=pad_type)
        # self.up = ups(F,F,kernel_size2,scale,stride, padding=padding)

        self.Att = SELayer(F, scale)

        self.to_image = ConvrelBlock(F, gc, kernel_size, bias=bias, pad_type=pad_type)

    def forward(self, x):
        x0 = self.conv00(x)

        x1 = self.conv0_0(x0)
        x1 = self.conv0_1(x1)

        x2 = self.conv1_0(x0)
        x2 = self.conv1_1(x2)

        x2 = torch.cat((x1, x2), 1)

        x2 = self.conv2(x2)
        x2 = x2 + x0

        # x1 = self.down(x2)
        x1 = self.conv3_1(x2)
        x1 = self.conv3_2(x1)
        x1 = self.conv3_3(x1)
        # x1 = self.up(x1)
        x1 = x1 + x2

        x2 = self.Att(x1)
        x2 = x1 + x2 + x0

        x2 = self.to_image(x2)

        x2 = x2 + x

        return x2



################
# CS_recon
################
'''
class PHI(nn.Module):
    def __init__(self, in_channels, out_channels, F, kernel_size1,  stride=1, bias=True,  CS_ratio=99, act_type='relu', pad_type='zero'):
        super(PHI, self).__init__()

        self.conv0 = ConvrelBlock(in_channels, F, kernel_size1, stride, bias=bias, pad_type=pad_type)
        self.down0 = downs(F, F, kernel_size1, scale=2)
        self.conv1 = ConvrelBlock(F, F, kernel_size1, stride, bias=bias, pad_type=pad_type)
        self.down1 = downs_con(F,100, kernel_size1, scale=5)
        self.conv2 = ConvBlock2(100, out_channels, 1, stride, bias=bias, pad_type=pad_type)

    def forward(self, x):

        x1 = self.conv0(x)
        x1 = self.down0(x1)
        x1 = self.conv1(x1)
        x1 = self.down1(x1)
        x1 = self.conv2(x1)

        return x1


class PHI_inv(nn.Module):
    def __init__(self, in_channels, out_channels, F, kernel_size1,  stride=1, bias=True, CS_ratio=99, act_type='relu', pad_type='zero'):
        super(PHI_inv, self).__init__()


        self.conv0 = ConvBlock2(in_channels, 100, 1, stride, bias=bias, pad_type=pad_type)
        self.up0 = ups(100,F,scale=5)
        self.conv1 = ConvrelBlock(F, F, kernel_size1, stride, bias=bias, pad_type=pad_type)
        self.up1 = ups(F, F, scale=2)
        self.conv2 = ConvrelBlock(F, out_channels, kernel_size1, stride, bias=bias, pad_type=pad_type)

    def forward(self, x):

        x1 = self.conv0(x)
        x1 = self.up0(x1)
        x1 = self.conv1(x1)
        x1 = self.up1(x1)
        x1 = self.conv2(x1)

        return x1
'''

class PHI(nn.Module):
    def __init__(self, in_channels, out_channels, F, kernel_size1,  stride=1, bias=True,  CS_ratio=99, act_type='relu', pad_type='zero'):
        super(PHI, self).__init__()

        ratio = 1.2


        self.conv4 = nn.Conv2d(1,1,kernel_size=33,stride=33,padding=0)


        if act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'relu6':
            self.act = nn.ReLU6(inplace=True)
        #self.conv2 = ConvBlock2(in_channels, out_channels, 1, stride, bias=False, pad_type=pad_type)

    def forward(self, x):

        x1 = self.conv0(x)
        x1 = self.conv1_0(x1)
        x1 = self.conv1(x1)
        x1 = self.act(x1)
        x1 = self.conv2(x1)
        x1 = self.act(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)

        return x1

class PHI_inv(nn.Module):
    def __init__(self, in_channels, out_channels, F, kernel_size1,  stride=1, bias=True, CS_ratio=99, act_type='relu', pad_type='zero'):
        super(PHI_inv, self).__init__()

        self.conv2 = ConvBlock2(in_channels, out_channels, 1, stride, bias=False, pad_type=pad_type)
        self.shuffle = nn.PixelShuffle(33)
    def forward(self, x):


        x1 = self.conv2(x)
        x1 = self.shuffle(x1)

        return x1


class PHI_GAP(nn.Module):
    def __init__(self, in_channels, out_channels, F, kernel_size1,  stride=1, bias=True,  CS_ratio=99, act_type='relu', pad_type='zero'):
        super(PHI_GAP, self).__init__()

        self.conv0 = ConvrelBlock(in_channels, F, kernel_size1, stride, bias=bias, pad_type=pad_type)

        self.conv1 = ConvrelBlock(F, F, kernel_size1, stride, bias=bias, pad_type=pad_type)
        self.down1 = downs_con(F,100, kernel_size1, scale=5)
        self.conv2 = ConvBlock2(100, out_channels, 1, stride, bias=bias, pad_type=pad_type)

    def forward(self, x):

        x1 = self.conv0(x)
        x1 = self.down0(x1)
        x1 = self.conv1(x1)
        x1 = self.down1(x1)
        x1 = self.conv2(x1)

        return x1

class make_sparse(nn.Module):
    def __init__(self, in_channels, out_channels, F, kernel_size1,  stride=1, bias=True, CS_ratio=109, act_type='relu', pad_type='zero'):
        super(make_sparse, self).__init__()

        ratio = 1.2
        C = int(CS_ratio*ratio)

        self.theta = torch.tensor([0.0001], device='cuda:0', dtype=torch.float, requires_grad=True)

        # D
        self.conv0 = ConvBlock2(1, F, 3, stride, bias=bias, pad_type=pad_type)
        # H
        self.conv1_0 = ConvrelBlock(F, F, 1, 1, bias=True, pad_type=pad_type, act_type=act_type)
        self.conv1 = nn.Conv2d(F, F, kernel_size=(3, 1), stride=1, padding=(1,0))
        self.conv2 = nn.Conv2d(F, F, kernel_size=(1, 3), stride=1, padding=(0,1))
        self.conv3 = ConvrelBlock(F, C , 1, 1, bias=False, pad_type=pad_type, act_type=act_type)

        self.conv4 = nn.Conv2d(C ,C ,kernel_size=11,stride=1,padding=5,groups=C , bias=True)
        self.conv4_2 = ConvBlock2(C , F, 1, 1, bias=False, pad_type=pad_type)

        # H^
        self.conv5_0 = ConvBlock2(F, C , 1, 1, bias=False, pad_type=pad_type)
        self.conv5 = nn.Conv2d(C , C , kernel_size=11, stride=1, padding=5, groups=C , bias=False)

        self.conv6 = ConvrelBlock(C , F, 1, 1, bias=True, pad_type=pad_type, act_type=act_type)
        self.conv7 = nn.Conv2d(F, F, kernel_size=(1, 3), stride=1, padding=(0,1))
        self.conv8 = nn.Conv2d(F, F, kernel_size=(3, 1), stride=1, padding=(1,0))
        self.conv8_1 = ConvrelBlock(F, F, 1, 1, bias=True, pad_type=pad_type, act_type=act_type)
        #
        # G
        self.conv9 = ConvBlock2(F, 1, kernel_size1, stride, bias=bias, pad_type=pad_type)

        if act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'relu6':
            self.act = nn.ReLU6(inplace=True)

    def forward(self, x):

        x1 = self.conv0(x)

        x2 = self.F(x1)


        x3 = self.soft(x2)


        x3 = self.F_inv(x3)
        #x2 = self.conv5(x2)

        x4 = self.conv9(x3)
        out = x4 + x


        rip_F = self.F_inv(x2) - x1

        return out, rip_F

    def F(self,x):
        x = self.conv1_0(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv4_2(x)

        return x

    def F_inv(self,x):
        x = self.conv5_0(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.act(x)
        x = self.conv8(x)
        x = self.act(x)
        x = self.conv8_1(x)

        return x

    def soft(self,x):
        x_ = torch.mul(torch.sign(x) , nn.functional.relu(torch.abs(x)-self.theta))

        return x_




class reconblcok(nn.Module):
    def __init__(self, in_channels, out_channels, F, kernel_size1,  stride=1, bias=True, CS_ratio=0.25, act_type='relu', pad_type='zero'):
        super(reconblcok, self).__init__()
        self.conv0 = ConvrelBlock(in_channels, F, kernel_size1, stride, bias=bias, pad_type=pad_type)
        self.conv1 = ConvrelBlock(F, F, 1, stride, bias=bias, pad_type=pad_type)

        self.down1 = downs(F, F, kernel_size1, scale=2)
        self.conv2 = ConvrelBlock(F, F, kernel_size1, stride, bias=bias, pad_type=pad_type)

        self.down2 = downs(F, F, kernel_size1, scale=2)
        self.conv3 = ConvrelBlock(F, F, kernel_size1, stride, bias=bias, pad_type=pad_type)

        self.convc = ConvrelBlock(F, F, kernel_size1, stride, bias=bias, pad_type=pad_type)

        self.conv4_1 = ConvrelBlock(2 * F, F, 1, stride, bias=bias, pad_type=pad_type)
        self.conv4 = ConvrelBlock(F, F, 3, stride, bias=bias, pad_type=pad_type)
        self.up1 = ups(F, F, scale=2)

        self.conv5_1 = ConvrelBlock(2 * F, F, 1, stride, bias=bias, pad_type=pad_type)
        self.conv5 = ConvrelBlock(F, F, 3, stride, bias=bias, pad_type=pad_type)
        self.up2 = ups(F, F, scale=2)

        self.conv6_1 = ConvrelBlock(2 * F, F, 1, stride, bias=bias, pad_type=pad_type)
        self.conv6 = ConvrelBlock(F, out_channels, 3, stride, bias=bias, pad_type=pad_type)

    def forward(self, x):
        x1 = self.conv0(x)
        x1 = self.conv1(x1) # 1

        x2 = self.down1(x1) #1/2
        x2 = self.conv2(x2) #1/2

        x3 = self.down2(x2) #1/4
        x3 = self.conv3(x3) #1/4

        x4 = self.convc(x3) #1/4

        x4 = torch.cat((x3,x4),1) #1/4
        x4 = self.conv4_1(x4) #1/4
        x4 = self.conv4(x4) #1/4
        x4 = self.up1(x4) #1/2

        x4 = torch.cat((x2, x4), 1) #1/2
        x4 = self.conv5_1(x4) #1/2
        x4 = self.conv5(x4) #1/2
        x4 = self.up2(x4) #1

        x4 = torch.cat((x1, x4), 1)  # 1/2
        x4 = self.conv6_1(x4)  # 1/2
        x4 = self.conv6(x4)  # 1/2

        x1 = x4 + x

        return x1


###########


################
# helper funcs
################

def get_valid_padding(kernel_size, dilation):
    """
    Padding value to remain feature size.
    """
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding
