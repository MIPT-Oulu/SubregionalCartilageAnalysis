import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19_bn


logging.basicConfig()
logger = logging.getLogger('models')
logger.setLevel(logging.DEBUG)


def ConvBlock3(inp, out, activation):
    """3x3 ConvNet building block with different activations support.
    """
    if activation == 'relu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )
    elif activation == 'elu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, padding=1),
            nn.ELU(1, inplace=True)
        )


class Decoder(nn.Module):
    """Decoder class. for encoder-decoder architecture.
    """
    def __init__(self, input_channels, output_channels, depth=2, mode='bilinear',
                 activation='relu'):
        super().__init__()
        self.layers = nn.Sequential()
        self.ups_mode = mode
        for i in range(depth):
            tmp = []
            if i == 0:
                tmp.append(ConvBlock3(input_channels, output_channels, activation))
            else:
                tmp.append(ConvBlock3(output_channels, output_channels, activation))

            self.layers.add_module('conv_3x3_{}'.format(i), nn.Sequential(*tmp))

    def forward(self, x_big, x):
        x_ups = F.interpolate(x, size=x_big.size()[-2:], mode=self.ups_mode,
                              align_corners=True)
        y_cat = torch.cat([x_ups, x_big], 1)
        y = self.layers(y_cat)
        return y


class VGG19BNUnet(nn.Module):
    def __init__(self,
                 center_depth: int = 2,
                 input_channels: int = 3,
                 output_channels: int = 1,
                 activation='relu',
                 pretrained: bool = False,
                 restore_weights: bool = False,
                 path_weights: bool = None, **kwargs):
        """

        Args:
            center_depth: Number of block in the model neck.
            input_channels: Number of input channels.
            output_channels: Number of output channels (/classes).
            activation {'ReLU', 'ELU'}:  Activation function.
            pretrained: If True, uses the upstream-pretrained weights.
            restore_weights: If True, loads the state from `path_weights`.
            path_weights: See above.
        """
        super().__init__()
        logger.warning('Redundant model init arguments:\n{}'
                       .format(repr(kwargs)))

        modules = OrderedDict()

        # -------------------------- Encoder ---------------------------------
        backbone = vgg19_bn(pretrained=True if pretrained else None)
        # print(backbone.features); quit()
        modules['encoder1'] = backbone.features[0:6]  # Pooling is in the next
        modules['encoder2'] = backbone.features[6:13]
        modules['encoder3'] = backbone.features[13:26]
        modules['encoder4'] = backbone.features[26:39]
        modules['encoder5'] = backbone.features[39:]
        # --------------------------------------------------------------------

        # -------------------------- Center ----------------------------------
        modules['center'] = nn.Sequential(*[
            ConvBlock3(512, 512, activation=activation)
            for _ in range(center_depth)
        ])
        # --------------------------------------------------------------------

        # -------------------------- Decoder ---------------------------------
        modules['decoder5'] = Decoder(512+512, 512, activation=activation)
        modules['decoder4'] = Decoder(512+512, 256, activation=activation)
        modules['decoder3'] = Decoder(256+256, 128, activation=activation)
        modules['decoder2'] = Decoder(128+128, 64, activation=activation)
        modules['decoder1'] = Decoder(64+64, 64, activation=activation)

        modules['mixer'] = nn.Conv2d(64, output_channels,
                                     kernel_size=1, padding=0, stride=1, bias=True)
        # --------------------------------------------------------------------

        self.__dict__['_modules'] = modules
        if restore_weights:
            self.load_state_dict(torch.load(path_weights))

    def forward(self, x):
        tmp = x
        in0 = torch.cat([tmp, tmp, tmp], 1)

        in1 = self.__dict__['_modules']['encoder1'](in0)
        in2 = self.__dict__['_modules']['encoder2'](in1)
        in3 = self.__dict__['_modules']['encoder3'](in2)
        in4 = self.__dict__['_modules']['encoder4'](in3)
        in5 = self.__dict__['_modules']['encoder5'](in4)

        cen = self.__dict__['_modules']['center'](in5)

        out5 = self.__dict__['_modules']['decoder5'](in5, cen)
        out4 = self.__dict__['_modules']['decoder4'](in4, out5)
        out3 = self.__dict__['_modules']['decoder3'](in3, out4)
        out2 = self.__dict__['_modules']['decoder2'](in2, out3)
        out1 = self.__dict__['_modules']['decoder1'](in1, out2)

        ret = self.__dict__['_modules']['mixer'](out1)
        ret_up = F.interpolate(ret, size=x.size()[-2:],
                               mode='bilinear', align_corners=True)

        return ret_up
