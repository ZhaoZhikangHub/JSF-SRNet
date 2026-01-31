import os, time, torch
import skimage.color as sc
import imageio
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import utils
from tqdm import tqdm
import data.common as common
from option import args

# 感知损失环境
from torchvision.models import vgg19,vgg16
import torch.nn.functional as F
import cv2
from torchvision import models
from torchsummary import summary

import utils


def train(training_dataloader, optimizer, model, epoch, writer, args):
    # vgg_model = vgg19(pretrained=False).features[:].to(args.device)                     # 添加_感知损失
    # vgg_model.load_state_dict(torch.load('/home/zzk/Mydoc/attachment//vgg19-dcbb9e9d.pth'),strict=False)          # 添加_感知损失
    # vgg_model.eval()                                                                    # 添加_感知损失
    # for param in vgg_model.parameters():                                                # 添加_感知损失（可能需要删除）
    #     param.requires_grad = False                                                     # 添加_感知损失（可能需要删除）
    # criterion1 = utils.LossNetwork_vgg(vgg_model).to(args.device)                       # 添加_感知损失
    # criterion2 = nn.L1Loss(size_average=False).to(args.device)

    trans_model_transformer = utils.Transformer().to(args.device)                         # 添加_Transformer损失 (pretrained=False)
    trans_model_transformer.eval()                                                        # 添加_Transformer损失
    for param in trans_model_transformer.parameters():                                    # 添加_Transformer损失
        param.requires_grad = False                                           # 添加_Transformer损失

    trans_model_multiscale = utils.Multiscale().to(args.device)                        # 添加_Multiscale损失 (pretrained=False)
    trans_model_multiscale.eval()                                                      # 添加_Multiscale损失
    for param in trans_model_multiscale.parameters():                                  # 添加_Multiscale损失
        param.requires_grad = False                                         # 添加_Multiscale损失

    criterion_transformer = utils.LossNetwork_transformer(trans_model_transformer).to(args.device)   # 添加_Transformer损失
    criterion_multiscale = utils.LossNetwork_multiscale(trans_model_multiscale).to(args.device)  # 添加_Multiscale损失 
    criterion_l1 = nn.L1Loss(size_average=False).to(args.device)
    # criterion_l1 = nn.HuberLoss(delta=1).to(args.device)
    

    model.train()
    torch.cuda.empty_cache()

    with tqdm(total=len(training_dataloader), ncols=224) as pbar:
        for iteration, (LR_img, HR_img) in enumerate(training_dataloader):

            LR_img = Variable(LR_img).to(args.device)
            HR_img = Variable(HR_img).to(args.device)

            # SR_img = model(LR_img.float(),Y_img.float())  # 我修改的 #################################
            SR_img = model(LR_img.float())  # 我修改的 #################################

            # loss1 = criterion1(SR_img, HR_img)
            # loss2 = criterion2(SR_img, HR_img)
            # loss = loss1+loss2

            loss1 = criterion_transformer(SR_img, HR_img)*1e5
            loss2 = criterion_multiscale(SR_img, HR_img)*1e8
            loss3 = criterion_l1(SR_img, HR_img)
            loss = loss1+loss2+loss3 # 
            # print("注意：The loss1 is",loss1)
            # print("注意：The loss2 is",loss2)
            # print("注意：The loss3 is",loss3)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            time.sleep(0.1)
            pbar.update(1)
            pbar.set_postfix(Epoch=epoch,
                             LeaRate='{:.3e}'.format(optimizer.param_groups[0]['lr']),
                             Loss='{:.4f}'.format(loss))

            niter = (epoch - 1) * len(training_dataloader) + iteration
            
            if (niter + 1) % 200 == 0:
                writer.add_scalar('Train-Loss', loss, niter)

    torch.cuda.empty_cache()


def test(source_path, result_path, model, args, f_csv=None):
    model.eval()
    count = 0
    Avg_PSNR = 0
    Avg_SSIM = 0
    Avg_SAM = 0 
    Avg_VIF = 0
    Avg_BRI = 0
    Avg_Time = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    filename = os.listdir(source_path)
    filename.sort()
    val_length = len(filename)
    torch.cuda.empty_cache()

    with torch.no_grad():
        with tqdm(total=val_length, ncols=224) as pbar:
            for idx_img in range(val_length):
                img_name = filename[idx_img]
                HR_img = imageio.imread(os.path.join(source_path, img_name))
                # print("这是“imageio”的图像读入形状：",HR_img.shape)
                img_name, ext = os.path.splitext(img_name)
                source_path_LQ = source_path.split('HR')[0] + 'LR_bicubic/X{}'.format(args.scale)
                LR_img = imageio.imread(os.path.join(source_path_LQ, img_name + '.jpg'))  # 原来为 .png

                HR_img = common.set_channel(HR_img, args.n_colors)
                
                # Y_img = common.zzk_add(LR_img,'YYY')  # 我增加的 #################################
                # Y_img = common.np2Tensor(Y_img, args.value_range) # 我增加的 #################################

                HR_img = common.np2Tensor(HR_img, args.value_range)
                LR_img = common.set_channel(LR_img, args.n_colors)
                LR_img = common.np2Tensor(LR_img, args.value_range)
                
                # c, h, w = HR_img.shape
                LR_img = Variable(LR_img[None]).to(args.device)
                # Y_img = Variable(Y_img[None]).to(args.device) # 我增加的 #################################
                H, W = HR_img.shape[1:]
                HR_img = HR_img[:, :(H - H % args.scale), :(W - W % args.scale)]

                start.record()
                # SR_img = model(LR_img.float(),Y_img.float())# 我修改的 #################################
                SR_img = model(LR_img.float())
                end.record()
                torch.cuda.synchronize()
                Time = start.elapsed_time(end)
                Avg_Time += Time
                count += 1

                SR_img = SR_img.data[0].cpu().clamp(0, 1)

                PSNR = utils.calc_PSNR_Y(SR_img, HR_img, rgb_range=args.value_range, shave=args.scale)
                Avg_PSNR += PSNR
                SSIM = utils.calc_SSIM_Y(SR_img, HR_img, rgb_range=args.value_range, shave=args.scale)
                Avg_SSIM += SSIM
                SAM = utils.calc_SAM(SR_img,HR_img)
                Avg_SAM += SAM
                VIF = utils.calc_VIF(SR_img,HR_img)
                Avg_VIF += VIF
                BRI = utils.calc_Brisque(SR_img)
                Avg_BRI += BRI

                if f_csv:
                    SSIM = utils.calc_SSIM_Y(SR_img, HR_img, rgb_range=args.value_range, shave=args.scale)
                    Avg_SSIM += SSIM
                    f_csv.writerow([img_name, PSNR, SSIM, SAM, VIF, BRI, Time])

                if args.save_img:
                    SR_img = utils.save_img(SR_img, 3, 255)
                    SR_img.save(result_path + '/{}.jpg'.format(img_name))  # 原来为 .png

                time.sleep(0.1)
                pbar.update(1)
                pbar.set_postfix(PSNR='{:.2f}dB'.format(Avg_PSNR / count),
                                 SSIM='{:.4f}'.format(Avg_SSIM / count),
                                 SAM='{:.4f}'.format(Avg_SAM/count),
                                 VIF='{:.4f}'.format(Avg_VIF/count),
                                 BRI='{:.4f}'.format(Avg_BRI/count),
                                 TIME='{:.1f}ms'.format(Avg_Time / count),
                                 )
    torch.cuda.empty_cache()
    if f_csv:
        f_csv.writerow(['Avg', Avg_PSNR / count, Avg_SSIM / count,Avg_SAM/count, Avg_VIF/count, Avg_BRI/count, Avg_Time / count]) # 在 avg_time之前 
    return Avg_PSNR/count, Avg_SSIM/count, Avg_SAM/count, Avg_VIF/count,  Avg_BRI/count, Avg_Time/count   #

