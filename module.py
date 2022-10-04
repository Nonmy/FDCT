import torch
import torch.nn as nn
from collections import OrderedDict


def conv3x3(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
         nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
         nn.ReLU(inplace=True)),
    ]


def conv1x1(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
         nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
         nn.ReLU(inplace=True)),
    ]


class _OSA_module(nn.Module):
    def __init__(self,
                 in_ch,
                 stage_ch, 
                 concat_ch,  
                 layer_per_block,  
                 module_name,  
                 identity=False):  
        super(_OSA_module, self).__init__()

        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(
                OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, 'concat')))

    def forward(self, x):

        identity_feat = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        if self.identity:
            xt = xt + identity_feat

        return xt

      
      
class DenseUpsamplingConvolution(nn.Module):
    """
    Dense upsampling convolution module.
    """
    def __init__(self, inplanes, planes, upscale_factor = 2):
        super(DenseUpsamplingConvolution, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inplanes, planes * upscale_factor * upscale_factor, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(planes * upscale_factor * upscale_factor),
            nn.ReLU(True)
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):

        x = self.layer(x)
        x = self.pixel_shuffle(x)
        return x
