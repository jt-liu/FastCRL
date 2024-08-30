import yaml
import torch
from torch import nn
import torch.nn.functional as F
from .fasternet.model_api import LitModel
from .fasternet.fasternet import Partial_conv3
from argparse import Namespace


def convbn(inc, ouc, kernel, stride, pad, dilate=1, groups=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=kernel, stride=stride, padding=pad,
                  dilation=dilate, groups=groups, bias=bias),
        nn.BatchNorm2d(ouc),
        nn.ReLU(True)
    )


class AdaFFT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.group1 = nn.Sequential(
            convbn(dim, dim // 2, 3, 1, 1),
            nn.Conv2d(dim // 2, dim // 2, 3, 1, 1)
        )
        self.group2 = nn.Sequential(
            convbn(dim, dim // 2, 3, 1, 1),
            nn.Conv2d(dim // 2, dim // 2, 3, 1, 1)
        )
        if self.dim == 96:
            self.complex_weight1 = nn.Parameter(torch.randn(120, 81, 2, dtype=torch.float32) * 0.02)
            self.complex_weight2 = nn.Parameter(torch.randn(120, 81, 2, dtype=torch.float32) * 0.02)
        elif self.dim == 192:
            self.complex_weight1 = nn.Parameter(torch.randn(60, 41, 2, dtype=torch.float32) * 0.02)
            self.complex_weight2 = nn.Parameter(torch.randn(60, 41, 2, dtype=torch.float32) * 0.02)
        elif self.dim == 384:
            self.complex_weight1 = nn.Parameter(torch.randn(30, 21, 2, dtype=torch.float32) * 0.02)
            self.complex_weight2 = nn.Parameter(torch.randn(30, 21, 2, dtype=torch.float32) * 0.02)
        elif self.dim == 768:
            self.complex_weight1 = nn.Parameter(torch.randn(15, 11, 2, dtype=torch.float32) * 0.02)
            self.complex_weight2 = nn.Parameter(torch.randn(15, 11, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        shortcut = x
        B, C, H, W = x.shape
        x1 = self.group1(x)
        x2 = self.group2(x)

        x1_fft = torch.fft.rfft2(x1, dim=(2, 3), norm='ortho')
        weight1 = torch.view_as_complex(self.complex_weight1)
        x1_fft = x1_fft * weight1
        y1 = torch.fft.irfft2(x1_fft, s=(H, W), dim=(2, 3), norm='ortho')

        x2_fft = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')
        weight2 = torch.view_as_complex(self.complex_weight2)
        x2_fft = x2_fft * weight2
        y2 = torch.fft.irfft2(x2_fft, s=(H, W), dim=(2, 3), norm='ortho')
        y = torch.cat([y1, y2], dim=1)
        return y + shortcut


class AdaFFTAblation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim == 96:
            h, w = 120, 160
        elif dim == 192:
            h, w = 60, 80
        elif dim == 384:
            h, w = 30, 40
        elif dim == 768:
            h, w = 15, 20
        self.dim = dim
        self.group1 = nn.Sequential(
            convbn(dim, dim // 2, 3, 1, 1),
            nn.Conv2d(dim // 2, dim // 2, 3, 1, 1)
        )
        self.group2 = nn.Sequential(
            convbn(dim, dim // 2, 3, 1, 1),
            nn.Conv2d(dim // 2, dim // 2, 3, 1, 1)
        )
        self.complex_weight1 = nn.Parameter(torch.randn(h, w // 2 + 1, dim // 2, 2, dtype=torch.float32) * 0.02)
        self.complex_weight2 = nn.Parameter(torch.randn(h, w // 2 + 1, dim // 2, 2, dtype=torch.float32) * 0.02)
        # self.group = nn.Sequential(
        #     convbn(dim, dim, 3, 1, 1),
        #     nn.Conv2d(dim, dim, 3, 1, 1)
        # )
        # self.complex_weight = nn.Parameter(torch.randn(h, w // 2 + 1, 2, dtype=torch.float32) * 0.02)
        # self.proj = nn.Sequential(convbn(dim, dim * 2, kernel=1, stride=1, pad=0),
        #                           nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0))
        # self.complex_weight1 = nn.Parameter(torch.randn(h, w // 2 + 1, 2, dtype=torch.float32) * 0.02)
        # self.complex_weight2 = nn.Parameter(torch.randn(h, w // 2 + 1, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        shortcut = x
        B, C, H, W = x.shape
        x1 = self.group1(x)
        x2 = self.group2(x)

        x1 = x1.permute(0, 2, 3, 1).contiguous()  # New shape B, H, W, C
        x1_fft = torch.fft.rfft2(x1, dim=(1, 2), norm='ortho')
        weight1 = torch.view_as_complex(self.complex_weight1)
        x1_fft = x1_fft * weight1
        y1 = torch.fft.irfft2(x1_fft, s=(H, W), dim=(1, 2), norm='ortho')

        x2 = x2.permute(0, 2, 3, 1).contiguous()  # New shape B, H, W, C
        x2_fft = torch.fft.rfft2(x2, dim=(1, 2), norm='ortho')
        weight2 = torch.view_as_complex(self.complex_weight2)
        x2_fft = x2_fft * weight2
        y2 = torch.fft.irfft2(x2_fft, s=(H, W), dim=(1, 2), norm='ortho')
        y = torch.cat([y1, y2], dim=3).permute(0, 3, 1, 2).contiguous()

        # x = self.group(x)
        # x_fft = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        # weight = torch.view_as_complex(self.complex_weight)
        # x_fft = x_fft * weight
        # y = torch.fft.irfft2(x_fft, s=(H, W), dim=(2, 3), norm='ortho')

        # x = self.group(x).permute(0, 2, 3, 1).contiguous()  # New shape B, H, W, C
        # x_fft = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        # weight = torch.view_as_complex(self.complex_weight)
        # x_fft = x_fft * weight
        # y = torch.fft.irfft2(x_fft, s=(H, W), dim=(1, 2), norm='ortho').permute(0, 3, 1, 2).contiguous()

        # x1 = x[:, : C//2, :, :]
        # x2 = x[:, C//2:, :, :]
        #
        # x1 = x1.permute(0, 2, 3, 1).contiguous()  # New shape B, H, W, C
        # x2 = x2.permute(0, 2, 3, 1).contiguous()
        # x1_fft = torch.fft.rfft2(x1, dim=(1, 2), norm='ortho')
        # weight = torch.view_as_complex(self.complex_weight)
        # x1_fft = x1_fft * weight
        # y1 = torch.fft.irfft2(x1_fft, s=(H, W), dim=(1, 2), norm='ortho')
        # y = torch.cat([y1, x2], dim=3).permute(0, 3, 1, 2).contiguous()
        # y = self.proj(y)

        # x1 = self.group1(x)
        # x2 = self.group2(x)
        #
        # x1_fft = torch.fft.rfft2(x1, dim=(2, 3), norm='ortho')
        # weight1 = torch.view_as_complex(self.complex_weight1)
        # x1_fft = x1_fft * weight1
        # y1 = torch.fft.irfft2(x1_fft, s=(H, W), dim=(2, 3), norm='ortho')
        #
        # x2_fft = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')
        # weight2 = torch.view_as_complex(self.complex_weight2)
        # x2_fft = x2_fft * weight2
        # y2 = torch.fft.irfft2(x2_fft, s=(H, W), dim=(2, 3), norm='ortho')
        # y = torch.cat([y1, y2], dim=1)

        # x_fft = torch.fft.rfft2(x.permute(0, 2, 3, 1).contiguous(), dim=(1, 2), norm='ortho')
        # weight = torch.view_as_complex(self.complex_weight)
        # x1_fft = x_fft * weight
        # y = torch.fft.irfft2(x1_fft, s=(H, W), dim=(1, 2), norm='ortho').permute(0, 3, 1, 2).contiguous()
        return y + shortcut
        # return shortcut


class FeaExtra(nn.Module):
    def __init__(self):
        super(FeaExtra, self).__init__()
        cfg_file = open('/data2/wzx/liujt/CRL/models/fasternet/cfg/fasternet_t2.yaml', errors='ignore')
        hyp = yaml.safe_load(cfg_file)
        param = Namespace(**hyp)
        basemodel = LitModel(1000, param).model
        # state_dict = torch.load('/data2/wzx/liujt/CRL/models/fasternet/pretrained/fasternet_t2.pth')
        # basemodel.load_state_dict(state_dict)
        basemodel.avgpool_pre_head = nn.Identity()
        basemodel.head = nn.Identity()
        self.basemodel = basemodel

    def forward(self, x):
        features = [x]
        # y = 1
        for k, v in self.basemodel._modules.items():
            if 'stages' == k:
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
                    # print(y, vi)
                    # y += 1
            else:
                features.append(v(features[-1]))
                # print(y, k)
                # y += 1
        # for i in range(len(features)):
        #     print(i, features[i].shape)
        return [features[8], features[6], features[4], features[2]]


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PartialConv, self).__init__()
        self.pconv = Partial_conv3(dim=in_channels, n_div=4, forward='split_cat')
        self.proj1 = nn.Sequential(convbn(in_channels, in_channels * 2, kernel=1, stride=1, pad=0),
                                   nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)
                                   )
        self.proj2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        short_cut = x
        x1 = self.pconv(x)
        x2 = self.proj1(x1) + short_cut
        x3 = self.proj2(x2)
        return x3


class Decoder(nn.Module):
    def __init__(self, inchannels=None, num_class=2):
        super(Decoder, self).__init__()
        if inchannels is None:
            assert "please give the channels of encoded features"
        self.InfoSimplify0 = AdaFFT(inchannels[0])
        self.conv1 = nn.Sequential(
            PartialConv(inchannels[0], inchannels[1]),
            nn.Upsample(scale_factor=2.0, mode='bilinear')
        )
        self.InfoSimplify11 = AdaFFT(inchannels[1])
        self.InfoSimplify12 = AdaFFT(inchannels[1])
        self.conv2 = nn.Sequential(
            PartialConv(inchannels[1], inchannels[2]),
            nn.Upsample(scale_factor=2.0, mode='bilinear')
        )
        self.InfoSimplify21 = AdaFFT(inchannels[2])
        self.InfoSimplify22 = AdaFFT(inchannels[2])
        self.conv3 = nn.Sequential(
            PartialConv(inchannels[2], inchannels[3]),
            nn.Upsample(scale_factor=2.0, mode='bilinear')
        )
        self.InfoSimplify31 = AdaFFT(inchannels[3])
        self.InfoSimplify32 = AdaFFT(inchannels[3])
        self.conv4 = nn.Sequential(
            PartialConv(inchannels[3], inchannels[3]),
            convbn(inchannels[3], inchannels[3] * num_class, 1, 1, 0, groups=3),
            nn.Conv2d(inchannels[3] * num_class, inchannels[3], 1, 1, 0, groups=3)
        )

    def forward(self, x):
        x0 = self.InfoSimplify0(x[0])
        x1 = self.conv1(x0)

        x1_info = self.InfoSimplify11(x1)
        x2_info = self.InfoSimplify12(x[1])
        x2 = x1_info + x2_info
        # x2 = x1 + x[1]
        x2 = self.conv2(x2)

        x2_info = self.InfoSimplify21(x2)
        x3_info = self.InfoSimplify22(x[2])
        x3 = x2_info + x3_info
        # x3 = x2 + x[2]
        x3 = self.conv3(x3)

        x3_info = self.InfoSimplify31(x3)
        x4_info = self.InfoSimplify32(x[3])
        x4 = x3_info + x4_info
        # x4 = x3 + x[3]
        x4 = self.conv4(x4)

        return x4
