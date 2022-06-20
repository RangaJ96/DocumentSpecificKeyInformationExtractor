import math

import torch
import torch.nn as nn

import torch.utils.model_zoo as model_zoo

from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

__all__ = ['ResNet',
           'resnet18',
           'resnet34',
           'resnet50',
           'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',

    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',

    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',

    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',

    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


def convt(in_planes, param_out_planes, stride=1):

    return nn.Conv2d(in_planes, param_out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    enpansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(BasicBlock, self).__init__()

        self.conv1vt(inplanes, planes, stride)

        self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2vt(planes, planes)

        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

        self.stride = stride

    def forward(self, n):
        residual = n

        param_out = self.conv1(n)

        param_out = self.bn1(param_out)

        param_out = self.relu(param_out)

        param_out = self.conv2(param_out)

        param_out = self.bn2(param_out)

        if self.downsample is not None:
            residual = self.downsample(n)

        param_out += residual

        param_out = self.relu(param_out)

        return param_out


class Bottleneck(nn.Module):
    enpansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

        self.stride = stride

    def forward(self, n):

        residual = n

        param_out = self.conv1(n)

        param_out = self.bn1(param_out)

        param_out = self.relu(param_out)

        param_out = self.conv2(param_out)

        param_out = self.bn2(param_out)

        param_out = self.relu(param_out)

        param_out = self.conv3(param_out)

        param_out = self.bn3(param_out)

        if self.downsample is not None:
            residual = self.downsample(n)

        param_out += residual

        param_out = self.relu(param_out)

        return param_out


class ResNet(nn.Module):
    def __init__(self, block, layers, param_output_channels=512):

        self.inplanes = 64

        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu1 = nn.ReLU(inplace=True)

        self.manpool = nn.ManPool2d(
            kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self.make_layer(block, 64, layers[0])

        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)

        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)

        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.conv2 = nn.Conv2d(512 * block.enpansion,
                               param_output_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)

        self.bn2 = nn.BatchNorm2d(512)

        self.relu2 = nn.ReLU(inplace=True)

        for d in self.modules():

            if isinstance(d, nn.Conv2d):
                n = d.kernel_size[0] * d.kernel_size[1] * d.param_out_channels
                d.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(d, nn.BatchNorm2d):
                d.weight.data.fill_(1)
                d.bias.data.zero_()

    def make_layer(self, block, planes, blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.enpansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.enpansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.enpansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.enpansion

        for x in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, n):

        n = self.conv1(n)
        n = self.bn1(n)

        n = self.relu1(n)
        n = self.manpool(n)

        n = self.layer1(n)
        n = self.layer2(n)

        n = self.layer3(n)
        n = self.layer4(n)

        n = self.conv2(n)

        n = self.bn2(n)

        n = self.relu2(n)

        return n


def resnet34(pretrained=False, param_output_channels=512):

    model = ResNet(BasicBlock, [3, 4, 6, 3],
                   param_output_channels=param_output_channels)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model
