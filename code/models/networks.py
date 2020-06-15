import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


class EncoderCNNPredictor(nn.Module):
    """
    CNN network to encode a 3d pic into a feature vector and then make a prediction with a dense layer
    """
    def __init__(self, in_layers=1):
        super(EncoderCNNPredictor, self).__init__()

        self.relu = nn.ReLU()
        layers = []
        out_layers = 8

        for i in range(4):
            layers.append(nn.Conv3d(in_layers, out_layers, 3, stride=i // 2 + 1, bias=False, padding=1))
            layers.append(nn.BatchNorm3d(out_layers))
            layers.append(self.relu)
            in_layers = out_layers
            out_layers *= 2

            if i % 2 == 0:
                layers.append(nn.AvgPool3d((3, 3, 3)))
            

        self.encoder = nn.Sequential(*layers)
        
        self.fl = nn.Flatten()

        self.predictor = nn.Sequential(
            nn.Linear(64,32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fl(x)
        x = self.predictor(x)
        return x

class G_Unet_add_all3D(nn.Module):
    """
    3D_UNET
    """
    def __init__(self, input_nc=1, output_nc=1, nz=1, num_downs=8, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False, gpu_ids=[], upsample='basic'):

        super(G_Unet_add_all3D, self).__init__()
        self.gpu_ids = gpu_ids
        self.nz = nz

        # construct unet blocks
        unet_block = UnetBlock_with_z3D(ngf * 8, ngf * 8, ngf * 8, nz, None, innermost=True,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
                                      stride=1)
        unet_block = UnetBlock_with_z3D(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout,
                                      upsample=upsample, stride=1)
        for i in range(num_downs - 6):  # 2 iterations
            if i == 0:
                stride = 2
            elif i == 1:
                stride = 1
            else:
                raise NotImplementedError("Too big, cannot handle!")

            unet_block = UnetBlock_with_z3D(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                            norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout,
                                            upsample=upsample, stride=stride)
        unet_block = UnetBlock_with_z3D(ngf * 4, ngf * 4, ngf * 8, nz, unet_block,
                                        norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
                                        stride=2)
        unet_block = UnetBlock_with_z3D(ngf * 2, ngf * 2, ngf * 4, nz, unet_block,
                                        norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
                                        stride=1)
        unet_block = UnetBlock_with_z3D(
            ngf, ngf, ngf * 2, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
            stride=2)
        unet_block = UnetBlock_with_z3D(input_nc, output_nc, ngf, nz, unet_block,
                                        outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample,
                                        stride=1)
        self.model = unet_block

        self.flatten = nn.Flatten()

    def forward(self, x, z):
        return self.flatten(self.model(x, z))


class UnetBlock_with_z3D(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz=0,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero',
                 stride=2):
        super(UnetBlock_with_z3D, self).__init__()
        
        downconv = []
        if padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = input_nc + nz
        downconv += [nn.Conv3d(input_nc, inner_nc,
                               kernel_size=3, stride=stride, padding=p)]

        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU() # change nl_layer() to nl_layer

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type, stride=stride)
            down = downconv
            up = [uprelu] + upconv + [nn.Sigmoid()]
        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type, stride=stride)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type, stride=stride)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), z.size(2), z.size(3), 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3), x.size(4))
            x_and_z = torch.cat([x, z_img], 1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)


def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero', stride=2):
    if upsample == 'basic':
        upconv = [nn.ConvTranspose3d(
            inplanes, outplanes, kernel_size=3, stride=stride, padding=1,
            output_padding=1 if stride == 2 else 0)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv


def fetch_simple_block3d(in_lay, out_lay, nl, norm_layer, stride=1, kw=3, padw=1):
    return [nn.Conv3d(in_lay, out_lay, kernel_size=kw,
                      stride=stride, padding=padw),
            nl(),
            norm_layer(out_lay)]



