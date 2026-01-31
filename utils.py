import os, torch, cv2, shutil
import numpy as np
import skimage.color as sc
from datetime import datetime
from PIL import Image
import sewar. full_ref as ref
import scipy.signal
import scipy.ndimage
import pyiqa.archs.brisque_arch as pyiqa


import copy
from typing import Optional, List



import torch.nn.functional as F
from torch import nn, Tensor
###################### 下面是添加的内容 ###########################################
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from math import exp
# from torchvision import models
# from option import args
######################## 上面是添加的内容 ########################################
class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=1,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True,
                 ):                                # num_decoder_layers=6, return_intermediate_dec=False
        # super().__init__()
        super(Transformer,self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = torch.nn.LayerNorm(d_model) if normalize_before else None
        self.encoder1 = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder2 = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder3 = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder4 = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder5 = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self): # 对参数进行初始化
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        src = self.encoder1(src, src_key_padding_mask=mask, pos=pos_embed)
        src = self.encoder2(src, src_key_padding_mask=mask, pos=pos_embed)
        src = self.encoder3(src, src_key_padding_mask=mask, pos=pos_embed)
        src = self.encoder4(src, src_key_padding_mask=mask, pos=pos_embed)
        memory = self.encoder5(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory 

class LossNetwork_transformer(torch.nn.Module):
    def __init__(self, Transformer_model):
        super(LossNetwork_transformer,self).__init__()
        self.transformer_layers = Transformer_model
        self.layer_name_mapping = {                         # 这里需要打印网络模型后重新编写 在调用之前打印 Transformer 网络结构；
            "encoder1":"encoder1",
            "encoder2":"encoder2",
            "encoder3":"encoder3",
            "encoder4":"encoder4",
            "encoder5":"encoder5",
        }
        self.weight = [1,1,1,1,1]
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(512 * 12 * 12, 512)  # 假设特征图尺寸为 12x12

    def output_features(self,x):
        output={}
        x = self.pool(torch.relu(self.conv1(x)))  # 输出形状: [batch_size, 64, 96, 96]
        x = self.pool(torch.relu(self.conv2(x)))  # 输出形状: [batch_size, 128, 48, 48]
        x = self.pool(torch.relu(self.conv3(x)))  # 输出形状: [batch_size, 256, 24, 24]
        x = self.pool(torch.relu(self.conv4(x)))  # 输出形状: [batch_size, 512, 12, 12]
        x = x.view(x.size(0), -1)  # 展平: [batch_size, 512 * 12 * 12]
        x = self.fc(x)  # 全连接层: [batch_size, 512]
        
        for name,module in self.transformer_layers._modules.items():
            # print("The name is:",name)
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]]=x
        return list(output.values())
    
    def forward(self,SR,HR):
        loss = []
        # print("The type of SR:",type(SR))        
        # print("The shape of SR:",SR.size())
        SR_features = self.output_features(SR)
        # print("The type of SR_features:",type(SR_features))        
        # print("The SR_features:",SR_features)      
        HR_features = self.output_features(HR)        
        for iter,(sr_feature, hr_feature,loss_weight) in enumerate(zip(SR_features, HR_features,self.weight)):
            loss.append(F.huber_loss(sr_feature, hr_feature, delta=1.5)*loss_weight)            
            # loss.append(F.mse_loss(sr_feature, hr_feature)*loss_weight)
        return sum(loss)  


class Multiscale(nn.Module): # 待完成；
    def __init__(self,conv ='BSConvU',num_in_ch=3,num_out_ch=1):
        super(Multiscale, self).__init__()
        if conv == 'BSConvU':
            self.conv = BSConvU_Loss
        else:
            self.conv = nn.Conv2d   
        
        self.scale_1 = nn.Sequential(
            self.conv(in_channels=num_in_ch,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
        )
        self.scale_2 = nn.Sequential(
            self.conv(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
        )
        self.scale_3 = nn.Sequential(
            self.conv(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
        )
        self.scale_4 = nn.Sequential(
            self.conv(in_channels=32,out_channels=16,kernel_size=5,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
        )
        self.scale_5 = nn.Sequential(
            self.conv(in_channels=16,out_channels=num_out_ch,kernel_size=7,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
        )
        self._reset_parameters()

    def _reset_parameters(self): # 对参数进行初始化
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
    def forward(self,x):
        out1=self.scale_1(x)
        print("The type of out1:",type(out1))
        print("The shape of out1:",out1.shape)
        out2=self.scale_2(out1)
        print("The type of out2:",type(out2))
        print("The shape of out2:",out2.shape)
        out3=self.scale_3(out2)
        print("The type of out3:",type(out3))
        print("The shape of out3:",out3.shape)
        out4=self.scale_4(out3)
        print("The type of out4:",type(out4))
        print("The shape of out4:",out4.shape)
        out=self.scale_5(out4)
        print("The type of final out:",type(out))
        print("The shape of final out:",out.shape)
        return out

class LossNetwork_multiscale(torch.nn.Module): # 待完成；
    def __init__(self, Multiscale_model):
        super(LossNetwork_multiscale, self).__init__()
        self.transformer_layers = Multiscale_model
        self.layer_name_mapping = {
            "scale_1":"scale_1",
            "scale_2":"scale_2",
            "scale_3":"scale_3",
            "scale_4":"scale_4",
            "scale_5":"scale_5",
        }
        self.weight =[1,1,1,1,1]
    def output_features(self,x):
        output={}
        for name,module in self.transformer_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]]=x
        return list(output.values())
    
    def forward(self,SR,HR):
        loss = []
        SR_features = self.output_features(SR)
        HR_features = self.output_features(HR)
        for iter,(sr_feature,hr_feature,loss_weight) in enumerate(zip(SR_features,HR_features,self.weight)):
            loss.append(F.huber_loss(sr_feature, hr_feature, delta=1)*loss_weight)  
            # loss.append(F.mse_loss(sr_feature,hr_feature)*loss_weight)
        return sum(loss)

class LossNetwork_vgg(torch.nn.Module):                                                  # 添加_感知损失
    def __init__(self, vgg_model):
        super(LossNetwork_vgg, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '13': "relu3_2",
            '22': "relu4_2",
            '31': "relu5_2"
        }
        #self.weight = [1/2.6,1/4.8,1/3.7,1/5.6,10/1.5]
        self.weight =[1.0,1.0,1.0,1.0,1.0]
    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        # print(output.keys())
        return list(output.values())
 
    def forward(self, SR, HR):
        loss = []
        SR_features = self.output_features(SR) # 这里输出的是SR的特征图
        HR_features = self.output_features(HR)         # 这里输出的是HR的特征图
        for iter,(sr_feature, hr_feature,loss_weight) in enumerate(zip(SR_features, HR_features,self.weight)):
            loss.append(F.mse_loss(sr_feature, hr_feature)*loss_weight)
        return sum(loss)   # ,SR_features    这里不计算感知损失

class BSConvU_Loss(torch.nn.Module):
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


def save_img(x, colors=3, value_range=255):
    if colors == 3:
        x = x.mul(value_range).clamp(0, value_range).round()
        x = x.numpy().astype(np.uint8)
        x = x.transpose((1, 2, 0))
        x = Image.fromarray(x)
    elif colors == 1:
        x = x[0, :, :].mul(value_range).clamp(0, value_range).round().numpy().astype(np.uint8)
        x = Image.fromarray(x).convert('L')
    return x


def crop_center(img, croph, cropw):
    h, w, c = img.shape

    if h < croph:
        img = cv2.copyMakeBorder(img, int(np.ceil((croph - h)/2)), int(np.ceil((croph - h)/2)), 0, 0, cv2.BORDER_DEFAULT)
    if w < cropw:
        img = cv2.copyMakeBorder(img, 0, 0, int(np.ceil((cropw - w)/2)), int(np.ceil((cropw - w)/2)), cv2.BORDER_DEFAULT)
    h, w, c = img.shape

    starth = h//2-(croph//2)
    startw = w//2-(cropw//2)
    return img[starth:starth+croph, startw:startw+cropw, :]


def quantize(img, rgb_range):
    return img.mul(rgb_range).clamp(0, rgb_range).round().div(rgb_range)


def rgb2ycbcrT(rgb):
    rgb = rgb.numpy().transpose(1, 2, 0) / 255
    yCbCr = sc.rgb2ycbcr(rgb)
    return torch.Tensor(yCbCr[:, :, 0])


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = 255*img1.astype(np.float64)
    img2 = 255*img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_SSIM_Y(input, target, rgb_range, shave):
    '''calculate SSIM
    the same outputs as MATLAB's
    '''

    c, h, w = input.size()
    input = quantize(input, rgb_range)
    target = quantize(target[:, 0:h, 0:w], rgb_range)
    if c > 1:
        input = rgb2ycbcrT(input).view(1, h, w)
        target = rgb2ycbcrT(target).view(1, h, w)
    input = input[0, shave:(h - shave), shave:(w - shave)]
    target = target[0, shave:(h - shave), shave:(w - shave)]
    return ssim(input.numpy(), target.numpy())


def calc_PSNR_Y(input, target, rgb_range, shave):
    c, h, w = input.size()
    input = quantize(input, rgb_range)
    target = quantize(target[:, 0:h, 0:w], rgb_range)
    if c > 1:
        input_Y = rgb2ycbcrT(input)
        target_Y = rgb2ycbcrT(target)
        diff = (input_Y - target_Y).view(1, h, w)
    else:
        diff = input - target
    diff = diff[:, shave:(h - shave), shave:(w - shave)]
    mse = diff.pow(2).mean()
    psnr = -10 * np.log10(mse)

    return psnr.data.numpy()


def calc_SSIM(input, target, rgb_range, shave):
    '''calculate SSIM
    the same outputs as MATLAB's
    '''

    c, h, w = input.shape
    input = quantize(input, rgb_range)
    target = quantize(target[:, 0:h, 0:w], rgb_range)
    input = input[:, shave:(h - shave), shave:(w - shave)]
    target = target[:, shave:(h - shave), shave:(w - shave)]
    ssim_value = 0
    for i in range(c):
        ssim_value += ssim(input[i, :, :].numpy(), target[i, :, :].numpy())
    return ssim_value / c


def calc_PSNR(input, target, rgb_range, shave):
    c, h, w = input.shape
    input = quantize(input, rgb_range)
    target = quantize(target[:, 0:h, 0:w], rgb_range)
    diff = input - target
    diff = diff[:, shave:(h - shave), shave:(w - shave)]
    mse = diff.pow(2).mean()
    psnr = -10 * np.log10(mse)

    return psnr.data.numpy()

def calc_SAM(SR_img,HR_img): # zzk add 
    return ref.sam(SR_img,HR_img)

def vifp_mscale(ref, dist): # zzk add 
    sigma_nsq=2
    eps = 1e-10

    num = 0.0
    den = 0.0
    for scale in range(1, 5):
       
        N = 2**(4-scale+1) + 1
        sd = N/5.0

        if (scale > 1):
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]
                
        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2
        
        sigma1_sq[sigma1_sq<0] = 0
        sigma2_sq[sigma2_sq<0] = 0
        
        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        
        g[sigma1_sq<eps] = 0
        sv_sq[sigma1_sq<eps] = sigma2_sq[sigma1_sq<eps]
        sigma1_sq[sigma1_sq<eps] = 0
        
        g[sigma2_sq<eps] = 0
        sv_sq[sigma2_sq<eps] = 0
        
        sv_sq[g<0] = sigma2_sq[g<0]
        g[g<0] = 0
        sv_sq[sv_sq<=eps] = eps
        
        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
        
    vifp = num/den

    return vifp

def calc_VIF(SR_img,HR_img): # zzk add 
    return vifp_mscale(SR_img,HR_img) 

def calc_Brisque(SR_img):
    SR_img = torch.unsqueeze(SR_img,0)
    print("The shape of SR_img before Brisque:", SR_img.shape)
    Bri_metrics = pyiqa.brisque(SR_img, pretrained_model_path = '/home/zzk/Mydoc/attachment/brisque_svm_weights.pth') #  brisque # 
    return np.float64(Bri_metrics.item())

def print_args(args):
    if args.train.lower() == 'train':
        args.save_img = False

        args.model_path = 'models/JSFSRNet_X{}'.format(args.scale) + datetime.now().strftime("_%Y%m%d_%H%M%S")

        args.resume = args.model_path + '/Checkpoints/checkpoint_epoch_0.pth'

        if not os.path.exists(args.model_path + '/Checkpoints/'):
            os.makedirs(args.model_path + '/Checkpoints')

        print(args)

    elif args.train.lower() == 'test':
        args.save_img = True

        args.model_path = 'models/JSFSRNet_X{}'.format(args.scale)
        args.resume = 'models/JSFSRNet_x{}.pth'.format(args.scale)
        # args.resume = 'models/JSFSRNet_X4_20230517_165608/Checkpoints/checkpoint_epoch_2.pth'


    return args





##############################################################################################################
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])    


class TransformerEncoderLayer(torch.nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")