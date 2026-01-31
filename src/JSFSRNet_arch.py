import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from . import Upsamplers as Upsamplers
import data.common as common
from scipy import signal

# 保留↓
class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


# 保留↓ 定义了激活层
def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'silu':
        layer = nn.SiLU(inplace)
    elif act_type == 'gelu':
        layer = nn.GELU()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


# 保留↓
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation  #  膨胀率，默认为 1，用于控制卷积核内部元素之间的间距。 计算了合适的填充量 padding，使得卷积操作后特征图大小不变。
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


class SKA(nn.Module):  # scalable kernel attention (SKA) block
    def __init__(self, n_feats, k=21, d=3, shrink=0.5, scale=2):
        super().__init__()
        f = int(n_feats*shrink)
        self.head = nn.Conv2d(n_feats, f, 1)
        self.proj_2 = nn.Conv2d(f, f, kernel_size=1)
        self.activation = nn.GELU()
        # self.LKA = nn.Sequential(
        #     nn.Conv2d(f, f, 1),
        #     conv_layer(f, f, k // d, dilation=1, groups=f),
        #     self.activation,
        #     nn.Conv2d(f, f, 1),
        #     conv_layer(f, f, 2*d-1, groups=f),
        #     self.activation,
        #     conv_layer(f, f, kernel_size=1),
        # )
        # 上面是原来的设计，下面是我的设计
        self.LKA1 = nn.Conv2d(f,f,1)
        self.LKA1_1 = conv_layer(f, f, 3, groups=f)
        self.LKA2 = nn.Conv2d(f,f,1)
        self.LKA2_1 = conv_layer(f, f, 5, groups=f)
        self.LKA3 = nn.Conv2d(f,f,1)
        self.LKA3_1 = conv_layer(f, f, 7, dilation=1, groups=f)
        self.LKA = conv_layer(f*3, f, kernel_size=1)

        self.tail = nn.Conv2d(f, n_feats, 1)
        self.scale = scale

    def forward(self, x):
        c1 = self.head(x)
        c2 = F.max_pool2d(c1, kernel_size=self.scale * 2 + 1, stride=self.scale)
        # c2 = self.LKA(c2)
        c2_1 = self.LKA1(c2)
        c2_2 = self.LKA1_1(c2_1)
        c2_3 = self.LKA2(c2_2)
        c2_4 = self.LKA2_1(c2_3)
        c2_5 = self.LKA3(c2_4)
        c2_6 = self.LKA3_1(c2_5)        
        c2 = self.LKA(torch.cat([c2_2, c2_4, c2_6], dim=1))

        c3 = F.interpolate(c2, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        a = self.tail(c3 + self.proj_2(c1))
        a = F.sigmoid(a)
        return x * a
    
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SDB(nn.Module):
    def __init__(self, in_channels, conv=nn.Conv2d, attn_shrink=0.25, act_type='silu', attentionScale=2):
        super(SDB, self).__init__()

        # kwargs = {'padding': 1}
        # self.dc = self.distilled_channels = in_channels // 2
        # self.rc = self.remaining_channels = in_channels

        # self.c1_d = conv_block(in_channels, self.dc, 1, act_type=act_type)
        # self.c1_r = conv(in_channels, self.rc, kernel_size=3,  **kwargs)
        # self.act = activation(act_type)

        # self.c2_d = conv_block(self.remaining_channels, self.dc, 1, act_type=act_type)
        # self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3,  **kwargs)

        # self.c3 = conv(self.remaining_channels, self.rc, kernel_size=3,  **{'padding': 1})

        # self.c3_d = conv_block(self.remaining_channels, self.dc, 1, act_type=act_type)
        # self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)

        # self.c4 = conv(self.remaining_channels, self.dc, kernel_size=5, **{'padding': 2})

        # self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)
    
        # self.esa = SKA(in_channels, k=21, d=3, shrink=attn_shrink, scale=attentionScale)
        self.distilled_channels = int(in_channels * attn_shrink)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.cca = CCALayer(self.distilled_channels * 4)

    def forward(self, input):
        # distilled_c1 = self.c1_d(input)
        # r_c1 = self.act(self.c1_r(input))

        # distilled_c2 = self.c2_d(r_c1)
        # r_c2 = self.act(self.c2_r(r_c1))
        # r_c3 = self.act(self.c3(r_c2))

        # distilled_c3 = self.c3_d(r_c3)
        # r_c4 = self.act(self.c3_r(r_c3))
        # r_c5 = self.act(self.c4(r_c4))

        # out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c5], dim=1)
        # out = self.c5(out)
        # out_fused = self.esa(out)
        # out_fused = out_fused + input

        # return out_fused
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused
    
def cal_distance(pa,pb): # 用于函数freq_filt
    dis = torch.sqrt(torch.tensor((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2))
    return dis

# def butterworth_highpass_filter(cutoff, order, shape,FrequencyBand):
#     """
#       创建巴特沃斯高通滤波器
#       Args:
#         cutoff: 截止频率
#         order: 巴特沃斯滤波器的阶数
#         shape: 滤波器的形状
#       Returns:
#         滤波器
#       """
#     # 创建一个单位矩阵
#     h = torch.ones(shape, dtype=torch.complex64).to("cuda:0")
#     # 计算滤波器的频率响应
#     if FrequencyBand == 'highpass':
#         for i in range(shape[0]):
#             for j in range(shape[1]):
#                 u = i - shape[0] // 2
#                 v = j - shape[1] // 2
#                 d = np.sqrt(u**2 + v**2)
#                 h[i, j] = 1. / (1. + (d / cutoff)**(2*order))  
#     elif FrequencyBand == 'midpass':
#         for i in range(shape[0]):
#             for j in range(shape[1]):
#                 u = i - shape[0] // 2
#                 v = j - shape[1] // 2
#                 d = np.sqrt(u**2 + v**2)
#                 h[i, j] = 1. / (1. + (d / cutoff)**(2*order))  
#     elif FrequencyBand == 'lowpass':
#         for i in range(shape[0]):
#             for j in range(shape[1]):
#                 u = i - shape[0] // 2
#                 v = j - shape[1] // 2
#                 d = np.sqrt(u**2 + v**2)
#                 h[i, j] = 1. / (1. + (d / cutoff)**(2*order))
#     return h

def butterworth_highpass_filter(cutoff, order, shape, FrequencyBand):
    """
    创建巴特沃斯滤波器，支持高通、低通和带通
    Args:
        cutoff: 截止频率。对于高通和低通是单个值，对于带通是元组(low_cut, high_cut)
        order: 阶数
        shape: 滤波器形状
        FrequencyBand: 滤波类型，'highpass', 'lowpass', 'bandpass'
    Returns:
        滤波器
    """
    

    h = torch.ones(shape, dtype=torch.complex64).to("cuda:0")
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    cutoff_h=40
    cutoff_l=90
    cutoff_m=(40,90)
    for i in range(rows):
        for j in range(cols):
            u = i - center_row
            v = j - center_col
            d = np.sqrt(u**2 + v**2)
            if d == 0:
                d = 1e-6  # 避免除以零

            if FrequencyBand == 'highpass':
                # 高通滤波器：允许高于截止频率的信号通过
                h_val = 1.0 / (1.0 + (cutoff_h / d)**(2 * order))
            elif FrequencyBand == 'lowpass':
                # 低通滤波器：允许低于截止频率的信号通过
                h_val = 1.0 / (1.0 + (d / cutoff_l)**(2 * order))
            elif FrequencyBand == 'midpass':
                # 带通滤波器：需要两个截止频率
                if isinstance(cutoff_m, (list, tuple)) and len(cutoff_m) == 2:
                    low_cut, high_cut = cutoff_m
                    # 高通部分，低于low_cut被衰减
                    highpass = 1.0 / (1.0 + (low_cut / d)**(2 * order))
                    # 低通部分，高于high_cut被衰减
                    lowpass = 1.0 / (1.0 + (d / high_cut)**(2 * order))
                    h_val = highpass * lowpass
                else:
                    raise ValueError("For bandpass, cutoff must be a tuple (low, high)")
            else:
                raise ValueError("FrequencyBand must be 'highpass', 'lowpass', or 'bandpass'")

            h[i, j] = h_val

    return h




def freq_filt(FrequencyBand,data): 
    # d,n =30, 1 # d 截止频率 n 滤波器阶数
    # # data的数据类型：torch.Size([1, 3, 600, 600]) ；torch.Size([32, 3, 48, 48])
    # # 下面的任务就是：对上面的数据格式进行频域滤波；
    # data = data.squeeze(0)  # 将data的数据类型从 nchw 变为chw
    # data = data.permute(1,2,0)  # 从chw变为hwc
    # s1 = torch.log(torch.abs(data))
    # center_point = tuple(map(lambda x: (x-1)/2, s1.shape))
    # data_h,data_w,data_c = data.shape
    # transform_matrix = torch.zeros(data.shape).to("cuda:0")
    # if FrequencyBand == 'highpass':
    #     for i in range(data_h):
    #         for j in range(data_w):
    #             dis = cal_distance(center_point,(i,j))
    #             transform_matrix[i,j] = 1/(1+(d/dis)**(2*n))
    # elif FrequencyBand == 'lowpass':
    #     for i in range(data_h):
    #         for j in range(data_w):
    #             dis = cal_distance(center_point,(i,j))
    #             transform_matrix[i,j] = 1/(1+(dis/d)**(2*n))
    # d_matrix = 0.8*transform_matrix+0.5
    # data_filterd = data*d_matrix
    # data_filterd = data_filterd.permute(2,0,1)
    # data_filterd = data_filterd.unsqueeze(0)
    # return  data_filterd
    cutoff, order = 30, 1
    h = butterworth_highpass_filter(cutoff, order, data.shape[-2:],FrequencyBand)
    d_matrix = 0.8*h+0.5
    filtered_image = data*d_matrix
    return filtered_image


class FDB(nn.Module):
    def __init__(self, in_channels, conv=nn.Conv2d, freq_band = 'lowpass',act_type='silu',norm_type='batch'):
        super(FDB, self).__init__()
        kwargs = {'padding': 1}
        self.freq_band = freq_band
        self.b1_conv = conv(in_channels, in_channels, kernel_size=3,  **kwargs)
        # self.b1_conv = conv_block(in_channels, in_channels, 3, act_type=act_type)
        self.b1_bn = norm(norm_type, in_channels)# 待确定
        self.b1_activate = activation(act_type)
    
    def forward(self, x):
            freq_data = freq_filt(self.freq_band,x)  # 低通滤波器
            ### 上面使用scipy的设定值滤波器
            ### 下面是自己写的巴特沃斯滤波器
            # print("The type of freq_data:",type(freq_data))
            # freq_data = freq_data.to("cuda:0", dtype=torch.cuda.FloatTensor, non_blocking=True)
            # print("走到这里了！")

            FDB1 = self.b1_conv(freq_data.real)
            FDB2 = self.b1_bn(FDB1)
            FDB3 = self.b1_activate(FDB2)
            output = torch.cat([FDB3,freq_data], dim=1) # 这里到底是用concat还是别的，仍然有待确定；
            
            ### 上面使用scipy的设定值滤波器
            ### 下面是自己写的巴特沃斯滤波器
            return output


class JSFSRNet(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=56, num_block=10, num_out_ch=3, upscale=3,
                 conv='BSConvU', upsampler='pixelshuffledirect', attn_shrink=0.25, act_type='gelu'):
        super(JSFSRNet, self).__init__()
        kwargs = {'padding': 1}

        if conv == 'BSConvU':
            self.conv = BSConvU
        else:
            self.conv = nn.Conv2d
        
        self.fea_conv1 = self.conv(num_in_ch * 4, num_feat, kernel_size=3, **kwargs) # 源代码

        self.B1 = SDB(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=2)
        self.B2 = SDB(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=2)
        self.B3 = SDB(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=2)
        self.B4 = SDB(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=2)
        self.B5 = SDB(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=3)
        self.B6 = SDB(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=3)
        self.B7 = SDB(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=3)
        self.B8 = SDB(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=4)
        self.B9 = SDB(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=4)
        self.B10 = SDB(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type,attentionScale=4)

        self.c1 = nn.Conv2d(num_feat * num_block + num_feat*3 , num_feat, 1)  # num_feat * num_block 为空域空间通道数  3*2 为频域空间通道数
        self.GELU = nn.GELU()
        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)

        if upsampler == 'pixelshuffledirect':
            self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pixelshuffleblock':
            self.upsampler = Upsamplers.PixelShuffleBlcok(in_feat=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'nearestconv':
            self.upsampler = Upsamplers.NearestConv(in_ch=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pa':
            self.upsampler = Upsamplers.PA_UP(nf=num_feat, unf=24, out_nc=num_out_ch)
        else:
            raise NotImplementedError(("Check the Upsampeler. None or not support yet"))

        ##### 上面是空域路径所用模块
        ##### 下面是频域路径所用模块

        self.F_low = FDB(in_channels=num_in_ch, conv=self.conv, freq_band = 'lowpass',act_type='silu',norm_type='batch')
        self.F_mid = FDB(in_channels=num_in_ch, conv=self.conv, freq_band = 'midpass',act_type='silu',norm_type='batch')
        self.F_high = FDB(in_channels=num_in_ch, conv=self.conv, freq_band = 'highpass',act_type='silu',norm_type='batch')
        self.fea_conv2 = self.conv(num_in_ch*2, num_feat, kernel_size=3, **kwargs) # 源代码

#########################替换1通道#####################################
    def forward(self, input):
        # add_img = torch.cat([add_img,add_img,add_img],dim=1)
        # 空间域通道路径
        # print("The type of input:",type(input))
        # print("The shape of input:",input.shape)

        input_spatial = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv1(input_spatial)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)
        out_B9 = self.B9(out_B8)
        out_B10 = self.B10(out_B9)
        # 加入频域通道路径 （注意input不是单通道的，处理的时候要进行单通道处理）
        input_freq_pre = input  # torch.Size([1, 3, 150, 150]) ；torch.Size([32, 3, 48, 48])
        input_freq = torch.fft.fft2(input_freq_pre) # torch.Size([1, 3, 600, 600]) ；torch.Size([32, 3, 48, 48])  得到输入input的傅里叶频谱图
        input_freq_shift = torch.fft.fftshift(input_freq) # torch.Size([1, 3, 600, 600]) ；torch.Size([32, 3, 48, 48]) 对傅里叶频谱进行中心化

        ## 进行频域处理
        # 1、自适应权重频段选择器
        out_F_L = self.F_low(input_freq_shift) # 进行学习与训练； 输入： torch.Size([1, 3, 600, 600]) ；torch.Size([32, 3, 48, 48])
        
        out_F_M = self.F_mid(input_freq_shift) #
        
        out_F_H = self.F_high(input_freq_shift) # 进行学习与训练；
        # 2、频域自学习处理单元
        out_FL_process = self.fea_conv2(out_F_L.real) # 暂时只考虑一个RGB通道  
        out_FM_process = self.fea_conv2(out_F_M.real)
        out_FH_process = self.fea_conv2(out_F_H.real) # 暂时只考虑一个RGB通道    

        # 3、将处理之后的结果返回为空域（先进行去中心化）；
        out_F_L = torch.fft.ifft2(torch.fft.ifftshift(out_FL_process)) # torch.Size([1, 56, 150, 150])
        out_F_M = torch.fft.ifft2(torch.fft.ifftshift(out_FM_process))
        out_F_H = torch.fft.ifft2(torch.fft.ifftshift(out_FH_process)) # torch.Size([1, 56, 150, 150])
        out_F_L=out_F_L.real
        out_F_M=out_F_M.real
        out_F_H=out_F_H.real

        # print("The shape of out_F_L:",out_F_L.shape)
        # print("The type of out_F_L:",type(out_F_L))

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8, out_B9, out_B10, out_F_L, out_F_M,out_F_H], dim=1) # , out_F1, out_F2
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)
        out_lr = self.c2(out_B) + out_fea
        output = self.upsampler(out_lr)
        
        return output



