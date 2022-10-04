"""
Depth Filler Network.

Author: Hongjie Fang.


版本改动：
    1. Conv before SAO
    2. last channel still 1*c

    s



"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from .dense import DenseBlock
from .duc import DenseUpsamplingConvolution
# from ..me.module import _OSA_module as _OSA_module
# import me.module._OSA_module as _OSA_module
from me.module import _OSA_module       # ubuntu Ver same name ....

class BlendBlock(nn.Module):

    def __init__(
            self, c_in, c_out, ksize_mid=3):
        super().__init__()
        print('down sample using maxpoo BN relu')
        c_mid = max(c_in // 2, 32)
        self.blend = nn.Sequential(
            nn.Conv2d(c_in, c_mid, 1, 1),
            nn.BatchNorm2d(c_mid),
            nn.LeakyReLU(),
            nn.Conv2d(c_mid, c_out, 1, 1),  # ? D2X  which I dont need just 11 now and oold is 3,1
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(),
            nn.Conv2d(c_out, c_out, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.blend(x)


class FusionBlock(nn.Module):
    '''
        让输入的特征图尺寸为原来的一半 即self.hidden_channels//2
        深度图提升为相同数量通道
        看看再说。。

    '''

    def __init__(self, c_feat_in):
        super(FusionBlock, self).__init__()
        self.map2img = nn.Sequential(
            nn.Conv2d(c_feat_in, c_feat_in + 1, 1, 1),
            nn.Sigmoid()
        )
        self.blend = BlendBlock(c_feat_in + 2, c_feat_in + 1)

        pass

    def forward(self, feature, depth):
        raw = self.map2img(feature)  # C self.h
        alpha = self.blend(torch.cat((depth, raw), dim=1))  # C self.h+1
        result = alpha * raw + (1 - alpha) * depth
        return result


class Vov4_7_15(nn.Module):
    """
        change: last channel to 1*C
    """

    def __init__(self, in_channels=4, hidden_channels=64, L=5, k=12, use_DUC=True, use_FusionHead=False, **kwargs):
        super(Vov4_7_15, self).__init__()
        # self.use_FusionHead = use_FusionHead
        self.use_FusionHead = False
        if self.use_FusionHead:
            print('use fusion head')
        else:
            print('do not use fusion head')
        self.main_conv_down = False
        if self.main_conv_down:
            print('using conv downr in main ')
        else:
            pass
        self.avg_down = False
        if self.avg_down:
            print('using avg_down in main ')
        else:
            pass

        if self.use_FusionHead:
            self.FusionHead = FusionBlock(3)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.L = L
        self.k = k
        self.use_DUC = use_DUC
        # First
        self.first = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense1: skip
        # self.dense1s_conv1 = nn.Sequential(
        #     nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
        #     nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
        #     nn.ReLU(True)
        # )
        # self.dense1s = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        # self.dense1s_conv2 = nn.Sequential(
        #     nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
        #     nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
        #     nn.ReLU(True)
        # )
        # Dense1: normal
        self.dense1_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels+1, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # self.dense1 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.dense1 = _OSA_module(self.hidden_channels, self.L * 4, self.hidden_channels, 5, 'Block1')
        # 在OSA那边调整了通道数量
        if self.main_conv_down:
            self.dense1_conv2 = nn.Sequential(
                nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
                nn.ReLU(True)
            )
        elif self.avg_down:
            self.dense1_conv2 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True),
                nn.BatchNorm2d(self.hidden_channels),
                nn.ReLU()

            )

        else:
            self.dense1_conv2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                nn.BatchNorm2d(self.hidden_channels),
                nn.ReLU()

            )


        # Dense2: skip
        # self.dense2s_conv1 = nn.Sequential(
        #     nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
        #     nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
        #     nn.ReLU(True)
        # )
        # self.dense2s = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        # self.dense2s_conv2 = nn.Sequential(
        #     nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
        #     nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
        #     nn.ReLU(True)
        # )
        # Dense2: normal
        self.dense2_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels+2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # self.dense2 = DenseBlock(self.hidden_channels+1, self.L, self.k, with_bn = True)
        self.dense2 = _OSA_module(self.hidden_channels, self.L * 4, self.hidden_channels, 5, 'Block2')
        if self.main_conv_down:
            self.dense2_conv2 = nn.Sequential(
                nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
                nn.ReLU(True)
            )
        elif self.avg_down:
            self.dense2_conv2 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True),
                nn.BatchNorm2d(self.hidden_channels),
                nn.ReLU()

            )
        else:
            self.dense2_conv2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                nn.BatchNorm2d(self.hidden_channels),
                nn.ReLU()

            )

        # Dense3: skip
        # self.dense3s_conv1 = nn.Sequential(
        #     nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
        #     nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
        #     nn.ReLU(True)
        # )
        # self.dense3s = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        # self.dense3s_conv2 = nn.Sequential(
        #     nn.Conv2d(self.k, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
        #     nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
        #     nn.ReLU(True)
        # )
        # Dense3: normal
        self.dense3_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels+2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # self.dense3 = DenseBlock(self.hidden_channels+1, self.L, self.k, with_bn = True)
        self.dense3 = _OSA_module(self.hidden_channels, self.L * 4, self.hidden_channels, 5, 'Block3')
        if self.main_conv_down:
            self.dense3_conv2 = nn.Sequential(
                nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
                nn.ReLU(True)
            )
        elif self.avg_down:
            self.dense3_conv2 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True),
                nn.BatchNorm2d(self.hidden_channels),
                nn.ReLU()

            )
        else:
            self.dense3_conv2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                nn.BatchNorm2d(self.hidden_channels),
                nn.ReLU()

            )
        # Dense4
        self.dense4_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels+2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # self.dense4 = DenseBlock(self.hidden_channels+1, self.L, self.k, with_bn = True)
        self.dense4 = _OSA_module(self.hidden_channels, self.L * 4, self.hidden_channels, 5, 'Block2')
        # self.dense4_conv2 = nn.AvgPool2d(kernel_size=3,stride=1,padding=1,ceil_mode=True)
        # DUC upsample 1
        self.updense1_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels+1, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # self.updense1 = DenseBlock(self.hidden_channels, self.L, self.k, with_bn = True)
        self.updense1 = _OSA_module(self.hidden_channels, self.L * 4, self.hidden_channels, 5, 'Block_up1')

        self.updense1_duc = self._make_upconv(self.hidden_channels, self.hidden_channels, upscale_factor=2)
        # DUC upsample 2
        self.updense2_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2+2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # self.updense2 = DenseBlock(self.hidden_channels+1, self.L, self.k, with_bn = True)
        self.updense2 = _OSA_module(self.hidden_channels, self.L * 4, self.hidden_channels, 5, 'Block_up2')
        self.updense2_duc = self._make_upconv(self.hidden_channels, self.hidden_channels, upscale_factor=2)
        # DUC upsample 3
        self.updense3_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2+2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # self.updense3 = DenseBlock(self.hidden_channels+1, self.L, self.k, with_bn = True)
        self.updense3 = _OSA_module(self.hidden_channels, self.L * 4, self.hidden_channels, 5, 'Block_up3')
        self.updense3_duc = self._make_upconv(self.hidden_channels, self.hidden_channels, upscale_factor=2)
        # DUC upsample 4
        self.updense4_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2+2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # self.updense4 = DenseBlock(self.hidden_channels+1, self.L, self.k, with_bn = True)
        self.updense4 = _OSA_module(self.hidden_channels, self.L * 4, self.k, 5, 'Block_up4')
        self.updense4_duc = self._make_upconv(self.k, self.hidden_channels, upscale_factor=2)
        # Final
        self.final = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_channels, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.skip_down1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1,
                      groups=self.hidden_channels),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU6(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU6(),
        )
        self.skip_down2 = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels * 2, kernel_size=3, stride=2, padding=1,
                      groups=self.hidden_channels * 2),
            nn.BatchNorm2d(self.hidden_channels * 2),
            nn.ReLU6(),
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU6(),
        )

        self.skip_down3 = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels * 2, kernel_size=3, stride=2, padding=1,
                      groups=self.hidden_channels * 2),
            nn.BatchNorm2d(self.hidden_channels * 2),
            nn.ReLU6(),
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU6(),

        )

        self.cdown = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU()
        )

        self.down_res1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1,
                      groups=self.hidden_channels),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU6(),
            nn.Conv2d(self.hidden_channels, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU6()
        )
        self.down_res2 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1,
                      groups=self.hidden_channels),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU6(),
            nn.Conv2d(self.hidden_channels, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU6()
        )
        self.down_res3 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1,
                      groups=self.hidden_channels),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU6(),
            nn.Conv2d(self.hidden_channels, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU6()
        )

        self.skip_up1 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels, stride=2, kernel_size=1, output_padding=1,
                               groups=self.hidden_channels),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU6(),
            nn.Conv2d(self.hidden_channels, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU6()
        )
        self.skip_up2 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels, stride=2, kernel_size=1, output_padding=1,
                               groups=self.hidden_channels),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU6(),
            nn.Conv2d(self.hidden_channels, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU6()
        )
        self.skip_up3 = nn.Sequential(
            # nn.Conv2d(self.hidden_channels*2,self.hidden_channels,kernel_size=1,stride=1),    # compress 1st
            nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels, stride=2, kernel_size=1, output_padding=1,
                               groups=self.hidden_channels),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU6(),
            nn.Conv2d(self.hidden_channels, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU6()
        )
        self.save_FM = False
        print('save FM:',self.save_FM)


        # manual optional
        # self.seg = nn.Sequential(
        #     nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
        #     nn.ReLU(True),
        #     nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
        #     nn.ReLU(True),
        #     nn.Conv2d(self.hidden_channels, 1, kernel_size=3, stride=1, padding=1),
        #     nn.Sigmoid()
        # )

    def _make_upconv(self, in_channels, out_channels, upscale_factor=2):
        if self.use_DUC:
            return DenseUpsamplingConvolution(in_channels, out_channels, upscale_factor=upscale_factor)
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=upscale_factor, stride=upscale_factor,
                                   padding=0, output_padding=0),
                nn.BatchNorm2d(out_channels, out_channels),
                nn.ReLU(True)
            )

    def forward(self, rgb, depth):

        if self.use_FusionHead: #(V1)
            div = 32.
            rgb = (rgb // div * div + div // 2.) / 255.
            nn, hh, ww = depth.shape
            depth = depth.view(nn, 1, hh, ww)
            h = self.first(self.FusionHead(rgb, depth))


        else:
            # 720 x 1280 (rgb, depth) -> 360 x 640 (h)
            nn, hh, ww = depth.shape
            depth = depth.view(nn, 1, hh, ww)
            h = self.first(torch.cat((rgb, depth), dim=1))



        # dense1: 360 x 640 (h, depth1) -> 180 x 320 (h, depth2)
        depth1 = F.interpolate(depth, scale_factor=0.5, mode="bilinear", align_corners=True)
        # dense1: skip
        h_d1s = h   # 160 120，给中间的


        # dense1: normal
        h = self.dense1_conv1(torch.cat((h, depth1), dim=1))    # 保留
        h = self.dense1(h)  # 深度不参与前卷积
        h = self.dense1_conv2(h)

        # dense2: 180 x 320 (h, depth2) -> 90 x 160 (h, depth3)
        depth2 = F.interpolate(depth1, scale_factor=0.5, mode="bilinear", align_corners=True)
        # dense2: skip
        h_d2s = h   # 80 60 给中间的
        h_d2d = self.skip_down2(torch.cat((h_d2s, self.skip_down1(h_d1s)), dim=1))

        # dense2: normal
        h = self.dense2_conv1(torch.cat((h, depth2, self.down_res1(h_d1s)), dim=1))
        h = self.dense2(h)
        h = self.dense2_conv2(h)

        # dense3: 90 x 160 (h, depth3) -> 45 x 80 (h, depth4)
        depth3 = F.interpolate(depth2, scale_factor=0.5, mode="bilinear", align_corners=True)
        # dense3: skip
        h_d3s = h
        h_d3d = self.skip_down3(torch.cat((h_d3s, h_d2d), dim=1))

        # h_d3s = self.dense3s_conv1(h)
        # h_d3s = self.dense3s(torch.cat((h_d3s, depth3), dim = 1))
        # h_d3s = self.dense3s_conv2(h_d3s)
        # dense3: normal
        h = self.dense3_conv1(
            torch.cat(
                (h, depth3, self.down_res2(h_d2s)),dim=1
                 )
        )
        h = self.dense3(h)
        h = self.dense3_conv2(h)

        # dense4: 45 x 80
        depth4 = F.interpolate(depth3, scale_factor=0.5, mode="bilinear", align_corners=True)
        h = self.dense4_conv1(torch.cat((h, depth4, self.down_res3(h_d3s)), dim=1))
        h = self.dense4(h)
        '''
            here 128 15 20 after avgpool becom 128 7 10 wow.
        '''
        # remove last avgpool here
        # h = self.dense4_conv2(h)
        # CAT HERE啊啊我靠 输入法？能用了？AAAAAA
        h = torch.cat((h, h_d3d), dim=1)    # 2*C 通道
        h = self.cdown(h)      # 1*C 通道
        if self.save_FM:
            FM = h
        h_skip3 = h # 2*C
        '''
            这里仔细写一下，这一层的OSA扩大了两倍隐藏层，且输出为2*hidden_channels
            同时这一层的上采样也是以 1*C为基础做的（之前是1*k）,
            这里要研究一下，duc比较费参数。
            跑跑看看不行改
        '''

        # updense1: 45 x 80 -> 90 x 160
        h = self.updense1_conv(torch.cat((h, depth4), dim=1))
        h = self.updense1(h)
        h = self.updense1_duc(h)
        h_skip1 = h

        # updense2: 90 x 160 -> 180 x 320
        h = torch.cat((h, h_d3s,depth3, self.skip_up3(h_skip3)), dim=1) # +1 +1
        h = self.updense2_conv(h)   #1 64 30 40
        # h = self.updense2(torch.cat((h,depth3), dim=1))
        h = self.updense2(h)
        h = self.updense2_duc(h)
        h_skip2 = h

        # updense3: 180 x 320 -> 360 x 640
        h = torch.cat((h, h_d2s, depth2, self.skip_up1(h_skip1)), dim=1)
        h = self.updense3_conv(h)
        h = self.updense3(h)
        h = self.updense3_duc(h)

        # updense4: 360 x 640 -> 720 x 1280
        h = torch.cat((h, h_d1s, depth1, self.skip_up2(h_skip2)), dim=1)
        h = self.updense4_conv(h)
        h = self.updense4(h)
        h = self.updense4_duc(h)
        # if self.save_FM:
        #     FM = h

        # seg
        # seg = self.seg(h)

        # final
        h = self.final(h)
        if self.save_FM:

            return rearrange(h, 'n 1 h w -> n h w'),FM
        else:
            return rearrange(h, 'n 1 h w -> n h w')
        # return rearrange(h, 'n 1 h w -> n h w'),seg