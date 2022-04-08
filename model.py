import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchsummary import summary
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function
import math
from math import sqrt

import random

# Scaled weight - He initialization
# "explicitly scale the weights at runtime"
class ScaleW:
    '''
    Constructor: name - name of attribute to be scaled
    '''

    def __init__(self, name):
        self.name = name

    def scale(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        '''
        Apply runtime scaling to specific module
        '''
        hook = ScaleW(name)
        weight = getattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        del module._parameters[name]
        module.register_forward_pre_hook(hook)

    def __call__(self, module, whatever):
        weight = self.scale(module)
        setattr(module, self.name, weight)


# Quick apply for scaled weight
def quick_scale(module, name='weight'):
    ScaleW.apply(module, name)
    return module

class SLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        linear = nn.Linear(dim_in, dim_out)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = quick_scale(linear)

    def forward(self, x):
        return self.linear(x)

# Normalization on every element of input vector
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class FC_A(nn.Module):
    '''
    Learned affine transform A, this module is used to transform
    midiate vector w into a style vector
    '''

    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = SLinear(dim_latent, n_channel * 2)
        # "the biases associated with ys that we initialize to one"
        self.transform.linear.bias.data[:n_channel] = 1
        self.transform.linear.bias.data[n_channel:] = 0

    def forward(self, w):
        # Gain scale factor and bias with:
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style


# AdaIn (AdaptiveInstanceNorm)
class AdaIn(nn.Module):
    '''
    adaptive instance normalization
    '''

    def __init__(self, n_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)   ## default
        # print('here')
        # self.norm = nn.InstanceNorm2d(n_channel,track_running_stats=True)

    def forward(self, image, style):
        factor, bias = style.chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias
        return result

class convrelu(nn.Module):
    '''
    This is the general class of style-based convolutional blocks
    '''

    def __init__(self, in_channel, out_channel,kernel,padding,dim_latent):
        super().__init__()
        # Style generators
        self.style1 = FC_A(dim_latent, out_channel)
        # AdaIn
        self.adain = AdaIn(out_channel)
        self.lrelu = nn.LeakyReLU()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel, padding=padding)

    def forward(self, previous_result, latent_w):
        result = self.conv1(previous_result)
        result = self.adain(result, self.style1(latent_w))
        result = self.lrelu(result)

        return result


class AdaIn_multi_running_stats(nn.Module):
    '''
    adaptive instance normalization
    '''

    def __init__(self, n_channel, tasks):
        super().__init__()
        # self.norm = nn.InstanceNorm2d(n_channel)   ## default
        print('AdaIn_multi_running_stats')
        # self.norm = nn.InstanceNorm2d(n_channel,track_running_stats=True)
        self.norm = nn.ModuleDict({task: nn.InstanceNorm2d(n_channel, affine=False, track_running_stats=True)
                                for task in tasks})

    def forward(self, image, style, task):
        factor, bias = style.chunk(2, 1)
        result = self.norm[task](image)
        result = result * factor + bias
        return result

class convrelu_multi_running_stats(nn.Module):
    '''
    This is the general class of style-based convolutional blocks
    '''

    def __init__(self, in_channel, tasks, out_channel,kernel,padding,dim_latent):
        super().__init__()
        # Style generators
        self.style1 = FC_A(dim_latent, out_channel)
        # AdaIn
        self.adain = AdaIn_multi_running_stats(out_channel, tasks)
        self.lrelu = nn.LeakyReLU()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel, padding=padding)

    def forward(self, previous_result, latent_w, task):
        result = self.conv1(previous_result)
        result = self.adain(result, self.style1(latent_w), task)
        result = self.lrelu(result)

        return result


class AdaBn(nn.Module):
    '''
    adaptive instance normalization
    '''

    def __init__(self, n_channel):
        super().__init__()
        # self.norm = nn.InstanceNorm2d(n_channel)
        # self.norm = nn.BatchNorm2d(n_channel, affine=False, track_running_stats=False)  # default
        self.norm = nn.BatchNorm2d(n_channel, affine=False, track_running_stats=True)  

    def forward(self, image, style):
        factor, bias = style.chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias
        return result

class convrelu_bn(nn.Module):
    '''
    This is the general class of style-based convolutional blocks
    '''

    def __init__(self, in_channel, out_channel,kernel,padding,dim_latent):
        super().__init__()
        # Style generators
        self.style1 = FC_A(dim_latent, out_channel)
        # AdaIn
        self.adain = AdaBn(out_channel)
        self.lrelu = nn.LeakyReLU()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel, padding=padding)

    def forward(self, previous_result, latent_w):
        result = self.conv1(previous_result)
        result = self.adain(result, self.style1(latent_w))
        result = self.lrelu(result)

        return result


class AdaBn_multi_running_stats(nn.Module):
    '''
    adaptive instance normalization
    '''

    def __init__(self, n_channel, tasks):
        super().__init__()
        # self.norm = nn.InstanceNorm2d(n_channel)
        # self.norm = nn.BatchNorm2d(n_channel, affine=False, track_running_stats=False)  # default
        # self.norm = nn.BatchNorm2d(n_channel, affine=False, track_running_stats=True)  
        self.norm = nn.ModuleDict({task: nn.BatchNorm2d(n_channel, affine=False, track_running_stats=True)
                                for task in tasks})

    def forward(self, image, style, task):
        factor, bias = style.chunk(2, 1)
        result = self.norm[task](image)
        result = result * factor + bias
        return result

class convrelu_bn_multi_running_stats(nn.Module):
    '''
    This is the general class of style-based convolutional blocks
    '''

    def __init__(self, in_channel,tasks, out_channel,kernel,padding,dim_latent):
        super().__init__()
        # Style generators
        self.style1 = FC_A(dim_latent, out_channel)
        # AdaIn
        self.adain = AdaBn_multi_running_stats(out_channel, tasks)
        self.lrelu = nn.LeakyReLU()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel, padding=padding)

    def forward(self, previous_result, latent_w, task):
        result = self.conv1(previous_result)
        result = self.adain(result, self.style1(latent_w), task)
        result = self.lrelu(result)

        return result


class Intermediate_Generator(nn.Module):
    '''
    A mapping consists of multiple fully connected layers.
    Used to map the input to an intermediate latent space W.
    '''

    def __init__(self, n_fc, dim_latent):
        super().__init__()
        layers = [PixelNorm()]
        for i in range(n_fc):
            layers.append(SLinear(dim_latent, dim_latent))
            layers.append(nn.LeakyReLU(0.2))

        self.mapping = nn.Sequential(*layers)

    def forward(self, latent_z):
        latent_w = self.mapping(latent_z)
        return latent_w

base_model = torchvision.models.resnet.resnet18(pretrained=True)

class ResNetUNet2_2(nn.Module):
    ## no sigmoid in last layer
    ## don't use x_original 

    def __init__(self, n_class,n_fc=8,dim_latent=512):
        super().__init__()

        base_model = torchvision.models.resnet.resnet18(pretrained=True)
        self.fcs = Intermediate_Generator(n_fc, dim_latent)

        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0,dim_latent)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0,dim_latent)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0,dim_latent)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0,dim_latent)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0,dim_latent)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1,dim_latent)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1,dim_latent)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1,dim_latent)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1,dim_latent)

        # self.conv_original_size0 = convrelu(3, 64, 3, 1,dim_latent)
        # self.conv_original_size1 = convrelu(64, 64, 3, 1,dim_latent)
        self.conv_original_size2 = convrelu(128, 64, 3, 1,dim_latent)

        self.conv_last = nn.Sequential(nn.Conv2d(64, n_class, 1))

    def forward(self, input, latent_z):

        # input is the input image and latent_z is the 512-d input code for the corresponding task
        if type(latent_z) != type([]):
            #print('You should use list to package your latent_z')
            latent_z = [latent_z]

        # latent_w as well as current_latent is the intermediate vector
        latent_w = [self.fcs(latent) for latent in latent_z]
        current_latent1 = latent_w
        current_latent = current_latent1[0]

        # x_original = self.conv_original_size0(input,current_latent)
        # x_original = self.conv_original_size1(x_original,current_latent)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4,current_latent)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3,current_latent)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x,current_latent)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2,current_latent)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x,current_latent)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1,current_latent)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x,current_latent)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0,current_latent)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x,current_latent)

        x = self.upsample(x)
        # x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x,current_latent)

        out = self.conv_last(x)

        return out

class convrelu_nonadain(nn.Module):
    '''
    This is the general class of style-based convolutional blocks
    '''

    def __init__(self, in_channel, out_channel,kernel,padding,dim_latent):
        super().__init__()
        # Style generators
        # self.style1 = FC_A(dim_latent, out_channel)
        # AdaIn
        # self.adain = AdaIn(out_channel)
        self.lrelu = nn.ReLU(inplace=True)
        # self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn1 = nn.InstanceNorm2d(out_channel, affine=True)
        # Convolutional layers
        print('here1')
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel, padding=padding)

    def forward(self, previous_result, latent_w=None):
        result = self.conv1(previous_result)
        # result = self.adain(result, self.style1(latent_w))
        result = self.bn1(result)
        result = self.lrelu(result)

        return result

class ResNetUNet2_2_no_adain(nn.Module):
    ## no sigmoid in last layer
    ## don't use x_original 
    ## change all adain to bn

    def __init__(self, n_class,n_fc=8,dim_latent=512):
        super().__init__()

        base_model = torchvision.models.resnet.resnet18(pretrained=True)
        # self.fcs = Intermediate_Generator(n_fc, dim_latent)

        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu_nonadain(64, 64, 1, 0,dim_latent)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu_nonadain(64, 64, 1, 0,dim_latent)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu_nonadain(128, 128, 1, 0,dim_latent)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu_nonadain(256, 256, 1, 0,dim_latent)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu_nonadain(512, 512, 1, 0,dim_latent)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu_nonadain(256 + 512, 512, 3, 1,dim_latent)
        self.conv_up2 = convrelu_nonadain(128 + 512, 256, 3, 1,dim_latent)
        self.conv_up1 = convrelu_nonadain(64 + 256, 256, 3, 1,dim_latent)
        self.conv_up0 = convrelu_nonadain(64 + 256, 128, 3, 1,dim_latent)

        # self.conv_original_size0 = convrelu(3, 64, 3, 1,dim_latent)
        # self.conv_original_size1 = convrelu(64, 64, 3, 1,dim_latent)
        self.conv_original_size2 = convrelu_nonadain(128, 64, 3, 1,dim_latent)

        self.conv_last = nn.Sequential(nn.Conv2d(64, n_class, 1))

    def forward(self, input, latent_z):

        # input is the input image and latent_z is the 512-d input code for the corresponding task
        # if type(latent_z) != type([]):
        #     #print('You should use list to package your latent_z')
        #     latent_z = [latent_z]

        # latent_w as well as current_latent is the intermediate vector
        # latent_w = [self.fcs(latent) for latent in latent_z]
        # current_latent1 = latent_w
        # current_latent = current_latent1[0]

        # x_original = self.conv_original_size0(input,current_latent)
        # x_original = self.conv_original_size1(x_original,current_latent)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        # x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out
