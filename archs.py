import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *
__all__ = ['UNext']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
import pdb



class UNext(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP
    
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(8),
            nn.GELU(),
        )

        ##  左边第一

        self.encoder1 = InvertedResidual(in_channels = 8, out_channels = 16, expansion_factor = 2, stride = 1)
        self.ebn1 = nn.BatchNorm2d(16)

        ##  左边第二层


        self.encoder2 = InvertedResidual(in_channels = 16, out_channels = 32, expansion_factor = 2, stride = 1)
        self.ebn2 = nn.BatchNorm2d(32)

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        ##  左边第三层

        self.encoder3 = InvertedResidual(in_channels = 32, out_channels = 64, expansion_factor = 2, stride = 1)
        self.ebn3 = nn.BatchNorm2d(64)


        ##  左边第四层
        # self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
        #                                       embed_dim=embed_dims[1])

        self.encoder4 = InvertedResidual(in_channels = 64, out_channels = 128, expansion_factor = 2, stride = 1)
        self.ebn4 = nn.BatchNorm2d(128)

        # self.block4 = shift_Block(dim=embed_dims[1], mlp_ratio=2., attn_drop=0., drop_path=0., pixel=2, step=1,
        #                         step_pad_mode='c', pixel_pad_mode='c', shift_size=5)

        # self.norm4 = nn.BatchNorm2d(160)
        # self.down4 = nn.Conv2d(in_channels=160, out_channels=256, kernel_size=2, stride=2)

        ##  第五层
        # self.patch_embed5 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
        #                                       embed_dim=embed_dims[2])

        self.encoder5 = InvertedResidual(in_channels = 128, out_channels = 256, expansion_factor = 2, stride = 1)
        self.ebn5 = nn.BatchNorm2d(256)


        # self.block5 = shift_Block(dim=embed_dims[2], mlp_ratio=2., attn_drop=0., drop_path=0., pixel=2, step=1,
        #                         step_pad_mode='c', pixel_pad_mode='c', shift_size=5)
        # self.norm5 = nn.BatchNorm2d(256)


        self.decoder5 = InvertedResidual(in_channels = 256, out_channels = 128, expansion_factor = 2, stride = 1)
        self.dbn5 = nn.BatchNorm2d(128)


        ##  右边第四层

        # self.dblock4 = shift_Block(dim=embed_dims[1], mlp_ratio=2., attn_drop=0., drop_path=0., pixel=2, step=1,
        #                         step_pad_mode='c', pixel_pad_mode='c', shift_size=5)
        #
        # self.dnorm4 = nn.BatchNorm2d(160)

        self.decoder4 = InvertedResidual(in_channels = 128, out_channels = 64, expansion_factor = 2, stride = 1)
        self.dbn4 = nn.BatchNorm2d(64)


        ##  右边第三层

        self.decoder3 = InvertedResidual(in_channels = 64, out_channels = 32, expansion_factor = 2, stride = 1)
        self.dbn3 = nn.BatchNorm2d(32)


        ##  右边第二层

        self.decoder2 = InvertedResidual(in_channels = 32, out_channels = 16, expansion_factor = 2, stride = 1)
        self.dbn2 = nn.BatchNorm2d(16)


        ##  右边第一层

        self.decoder1 = InvertedResidual(in_channels = 16, out_channels = 8, expansion_factor = 2, stride = 1)
        self.dbn1 = nn.BatchNorm2d(8)


        self.final = nn.Conv2d(8, num_classes, kernel_size=1)


        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        
        B = x.shape[0]

        x = self.in_conv(x)

        ##  左边第一层
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out

        ##  左边第二层
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out

        ##  左边第三层

        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        ##  左边第四层

        out = F.relu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        # out,H,W = self.patch_embed4(out)
        # out = self.block4(out)
        # out = self.norm4(out)
        t4 = out

        ### 第五层

        out = F.relu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        # out ,H,W= self.patch_embed5(out)
        # out = self.block5(out)
        # out = self.norm5(out)

        out = F.relu(F.interpolate(self.dbn5(self.decoder5(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t4)

        ##  右边第四层

        _,_,H,W = out.shape
        # out = self.dblock4(out)
        # out = self.dnorm4(out)

        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)

        ##  右边第三层

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)

        ##  右边第二层

        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)

        ##  右边第一层

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))

        return self.final(out)

#EOF


# DW卷积
def Conv3x3BNReLU(in_channels,out_channels,stride,groups):
    return nn.Sequential(
            # stride=2 wh减半，stride=1 wh不变
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=stride, padding=3, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

# PW卷积
def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

# # PW卷积(Linear) 没有使用激活函数
def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

class InvertedResidual(nn.Module):
    # t = expansion_factor,也就是扩展因子，文章中取6
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = (in_channels * expansion_factor)
        # print("expansion_factor:", expansion_factor)
        # print("mid_channels:",mid_channels)

        # 先1x1卷积升维，再1x1卷积降维
        self.bottleneck = nn.Sequential(
            # 升维操作: 扩充维度是 in_channels * expansion_factor (6倍)
            Conv1x1BNReLU(in_channels, mid_channels),
            # DW卷积,降低参数量
            Conv3x3BNReLU(mid_channels, mid_channels, stride, groups=mid_channels),
            # 降维操作: 降维度 in_channels * expansion_factor(6倍) 降维到指定 out_channels 维度
            Conv1x1BN(mid_channels, out_channels)
        )

        # 第一种: stride=1 才有shortcut 此方法让原本不相同的channels相同
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        # 第一种:
        out = (out+self.shortcut(x)) if self.stride==1 else out
        return out


