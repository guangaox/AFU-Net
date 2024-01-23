# -*- coding: utf-8 -*-


from __future__ import print_function, division
import torch.nn as nn
import torch.utils.data
import torch
import torchvision.models
from torchvision import models
from plot import *
import cv2 as cv
from ASPP_model import ASPP
from myattention import *
from ResidualBlock import *

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv_block_e5(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(Conv_block_e5, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=4, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=4,bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class UP_CONV(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(UP_CONV, self).__init__()
        self.up = nn.Sequential(
            #nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x




class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)



    def forward(self, g, x):

            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            out = x * psi
            return out






class SAM_block(nn.Module):
    """
    SAM Attention Block
    """
    def __init__(self, d_model, H,W, kernel_size=3):
        super(SAM_block, self).__init__()

        self.Position_model = PositionAttentionModule(d_model=d_model,kernel_size=3,H=H,W=W)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g):
        g1 = self.Position_model(g)
        out = self.relu(g1)
        return out



class CAM_block(nn.Module):
    """
    CAM Attention Block
    """
    def __init__(self, d_model, H,W, kernel_size=3):
        super(CAM_block, self).__init__()

        self.Channel_model = ChannelAttentionModule(d_model=d_model,kernel_size=3,H=H,W=W)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, g):
        x1 = self.Channel_model(g)
        out = self.relu(x1)
        return out





def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class AFU_Net(nn.Module):

    def __init__(self, img_ch=3, output_ch=1):
        super(AFU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.base_model = models.resnet18(weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])

        self.layer1 = nn.Sequential(*self.base_layers[3:5])

        self.layer2 = self.base_layers[5]

        self.layer3 = self.base_layers[6]

        self.layer4 = self.base_layers[7]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])


        self.Up5 = UpSampleWithAttention(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])


        self.Up4 = UpSampleWithAttention(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])


        self.Up3 = UpSampleWithAttention(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])


        self.Up2 = UpSampleWithAttention(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])


        self.Up1 = UpSampleWithAttention(filters[0], filters[0])
        self.Att1 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Up0 = up_conv(filters[0], filters[0])
        self.Att0 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv0 = conv_block(filters[0], filters[0])

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(128, 64, 3, 1)

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)
        self.active = torch.nn.Sigmoid()



        self.SAM1   =   SAM_block(d_model=512, kernel_size=3, H=8, W=8)
        self.SAM2   =   SAM_block(d_model=256, kernel_size=3, H=16, W=16)

        self.CAM1   =   CAM_block(d_model=128, kernel_size=3, H=32, W=32)
        self.CAM2   =   CAM_block(d_model=64, kernel_size=3, H=64, W=64)
        self.CAM3   =   CAM_block(d_model=64, kernel_size=3, H=128, W=128)

        self.ASPP_BLOCK = ASPP(1024,1024)

    def forward(self, x):

    #   encoder
        x_original = self.conv_original_size0(x)
        x_original = self.conv_original_size1(x_original)
        e0 = self.layer0(x)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        e5 = self.ASPP_BLOCK(e5)


    # decoder

        d5 = self.Up5(e5)
        x4  = self.SAM1(d5)
        d5 = torch.cat((x4, d5), dim=1)


        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)

        x3  = self.SAM2(d4)
        d4 = torch.cat((x3, d4), dim=1)

        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)

        x2 = self.CAM1(d3)
        d3 = torch.cat((x2, d3), dim=1)


        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)

        x1 = self.CAM2(d2)
        d2 = torch.cat((x1, d2), dim=1)


        d2 = self.Up_conv2(d2)
        d1 = self.Up1(d2)


        x0 = self.CAM3(d1)
        d1 = torch.cat((x0, d1), dim=1)

        d1 = self.Up_conv1(d1)
        d0 = self.Up0(d1)

        x = torch.cat((x_original, d0), dim=1)
        d0 = self.conv_original_size2(x)
        out = self.Conv(d0)
        out = self.active(out)

        return out




if __name__=="__main__":

    x = torch.randn(1,3,256,256)
    model = AFU_Net()
    y = model(x)
    print(y.shape)
