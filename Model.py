
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from module import DenseUpsamplingConvolution
from module import _OSA_module



class FDCT(nn.Module):


    def __init__(self, in_channels=4, hidden_channels=64, L=5, k=12, use_DUC=True, **kwargs):
        super(FDCT, self).__init__()



        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.L = L
        self.k = k
        self.use_DUC = use_DUC

        self.first = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )

        self.dense1_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels+1, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense1 = _OSA_module(self.hidden_channels, self.L * 4, self.hidden_channels, 5, 'Block1')
        self.dense1_conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU()
        )


        self.dense2_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels+2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense2 = _OSA_module(self.hidden_channels, self.L * 4, self.hidden_channels, 5, 'Block2')

        self.dense2_conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU()
        )


        self.dense3_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels+2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )

        self.dense3 = _OSA_module(self.hidden_channels, self.L * 4, self.hidden_channels, 5, 'Block3')

        self.dense3_conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU()

        )
        self.dense4_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels+2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense4 = _OSA_module(self.hidden_channels, self.L * 4, self.hidden_channels, 5, 'Block2')

        self.updense1_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels+1, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )

        self.updense1 = _OSA_module(self.hidden_channels, self.L * 4, self.hidden_channels, 5, 'Block_up1')

        self.updense1_duc = self._make_upconv(self.hidden_channels, self.hidden_channels, upscale_factor=2)

        self.updense2_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2+2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )

        self.updense2 = _OSA_module(self.hidden_channels, self.L * 4, self.hidden_channels, 5, 'Block_up2')
        self.updense2_duc = self._make_upconv(self.hidden_channels, self.hidden_channels, upscale_factor=2)
        self.updense3_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2+2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense3 = _OSA_module(self.hidden_channels, self.L * 4, self.hidden_channels, 5, 'Block_up3')
        self.updense3_duc = self._make_upconv(self.hidden_channels, self.hidden_channels, upscale_factor=2)
        self.updense4_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2+2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense4 = _OSA_module(self.hidden_channels, self.L * 4, self.k, 5, 'Block_up4')
        self.updense4_duc = self._make_upconv(self.k, self.hidden_channels, upscale_factor=2)

        self.final = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_channels, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )

        # Shortcut Fusion Module
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
        # Channel_down
        self.cdown = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU()
        )

        # Encoder side's shortcuts
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

        # Decoder side's shortcuts
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
            nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels, stride=2, kernel_size=1, output_padding=1,
                               groups=self.hidden_channels),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU6(),
            nn.Conv2d(self.hidden_channels, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU6()
        )


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



        # First Module
        nn, hh, ww = depth.shape
        depth = depth.view(nn, 1, hh, ww)
        h = self.first(torch.cat((rgb, depth), dim=1))



        depth1 = F.interpolate(depth, scale_factor=0.5, mode="bilinear", align_corners=True)
        h_d1s = h

        # Encoder

        h = self.dense1_conv1(torch.cat((h, depth1), dim=1))
        h = self.dense1(h)
        h = self.dense1_conv2(h)

        depth2 = F.interpolate(depth1, scale_factor=0.5, mode="bilinear", align_corners=True)
        h_d2s = h
        h_d2d = self.skip_down2(torch.cat((h_d2s, self.skip_down1(h_d1s)), dim=1))

        h = self.dense2_conv1(torch.cat((h, depth2, self.down_res1(h_d1s)), dim=1))
        h = self.dense2(h)
        h = self.dense2_conv2(h)

        depth3 = F.interpolate(depth2, scale_factor=0.5, mode="bilinear", align_corners=True)
        h_d3s = h
        h_d3d = self.skip_down3(torch.cat((h_d3s, h_d2d), dim=1))

        h = self.dense3_conv1(
            torch.cat(
                (h, depth3, self.down_res2(h_d2s)),dim=1
                 )
        )
        h = self.dense3(h)
        h = self.dense3_conv2(h)

        depth4 = F.interpolate(depth3, scale_factor=0.5, mode="bilinear", align_corners=True)
        h = self.dense4_conv1(torch.cat((h, depth4, self.down_res3(h_d3s)), dim=1))
        h = self.dense4(h)

        h = torch.cat((h, h_d3d), dim=1)
        h = self.cdown(h)
        h_skip3 = h

        # Decoder
        h = self.updense1_conv(torch.cat((h, depth4), dim=1))
        h = self.updense1(h)
        h = self.updense1_duc(h)
        h_skip1 = h

        h = torch.cat((h, h_d3s,depth3, self.skip_up3(h_skip3)), dim=1)
        h = self.updense2_conv(h)
        h = self.updense2(h)
        h = self.updense2_duc(h)
        h_skip2 = h

        h = torch.cat((h, h_d2s, depth2, self.skip_up1(h_skip1)), dim=1)
        h = self.updense3_conv(h)
        h = self.updense3(h)
        h = self.updense3_duc(h)

        h = torch.cat((h, h_d1s, depth1, self.skip_up2(h_skip2)), dim=1)
        h = self.updense4_conv(h)
        h = self.updense4(h)
        h = self.updense4_duc(h)

        h = self.final(h)

        return rearrange(h, 'n 1 h w -> n h w')
