from functools import partial
from inspect import signature

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np

from models.twod_models.common import TemporalPooling
from models.twod_models.temporal_modeling import temporal_modeling_module


__all__ = ['GoogLeNet', 'inception_v1']

model_urls = {
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}


class GoogLeNet(nn.Module):

    def __init__(self, num_frames, dropout, num_classes=1000,
                 temporal_module=None, without_t_stride=False, pooling_method='max'):
        super().__init__()
        self.pooling_method = pooling_method.lower()

        self.orig_num_frames = num_frames
        self.num_frames = num_frames
        self.without_t_stride = without_t_stride
        self.temporal_module = temporal_module

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        if temporal_module is not None:
            self.tam1 = temporal_module(duration=self.num_frames, channels=64)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        if temporal_module is not None:
            self.tam2 = temporal_module(duration=self.num_frames, channels=64)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        if temporal_module is not None:
            self.tam3 = temporal_module(duration=self.num_frames, channels=192)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        if not self.without_t_stride:
            self.t_pool1 = TemporalPooling(self.num_frames, 3, 2, self.pooling_method)
            self.num_frames = self.num_frames // 2
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        if temporal_module is not None:
            self.tam3a = temporal_module(duration=self.num_frames, channels=256)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        if temporal_module is not None:
            self.tam3b = temporal_module(duration=self.num_frames, channels=480)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        if not self.without_t_stride:
            self.t_pool2 = TemporalPooling(self.num_frames, 3, 2, self.pooling_method)
            self.num_frames = self.num_frames // 2
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        if temporal_module is not None:
            self.tam4a = temporal_module(duration=self.num_frames, channels=512)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        if temporal_module is not None:
            self.tam4b = temporal_module(duration=self.num_frames, channels=512)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        if temporal_module is not None:
            self.tam4c = temporal_module(duration=self.num_frames, channels=512)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        if temporal_module is not None:
            self.tam4d = temporal_module(duration=self.num_frames, channels=528)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        if temporal_module is not None:
            self.tam4e = temporal_module(duration=self.num_frames, channels=832)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        if not self.without_t_stride:
            self.t_pool3 = TemporalPooling(self.num_frames, 3, 2, self.pooling_method)
            self.num_frames = self.num_frames // 2

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        if temporal_module is not None:
            self.tam5a = temporal_module(duration=self.num_frames, channels=832)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        if temporal_module is not None:
            self.tam5b = temporal_module(duration=self.num_frames, channels=1024)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, num_classes)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def mean(self, modality='rgb'):
        return [0.5, 0.5, 0.5] if modality == 'rgb' else [0.5]

    def std(self, modality='rgb'):
        return [0.5, 0.5, 0.5] if modality == 'rgb' else [0.5]

    @property
    def network_name(self):
        name = ''
        if self.temporal_module is not None:
            param = signature(self.temporal_module).parameters
            temporal_module = str(param['name']).split("=")[-1][1:-1]
            blending_frames = str(param['blending_frames']).split("=")[-1]
            blending_method = str(param['blending_method']).split("=")[-1][1:-1]
            dw_conv = True if str(param['dw_conv']).split("=")[-1] == 'True' else False
            name += "{}-b{}-{}{}-".format(temporal_module, blending_frames,
                                         blending_method,
                                         "" if dw_conv else "-allc")
        name += 'inception-v1'
        if not self.without_t_stride:
            name += "-ts-{}".format(self.pooling_method)

        return name

    def forward(self, x):
        batch_size, c_t, h, w = x.shape
        x = x.view(batch_size * self.orig_num_frames, c_t // self.orig_num_frames, h, w)

        # N x 3 x 224 x 224
        x = self.conv1(x)
        if self.temporal_module is not None:
            x = self.tam1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        if self.temporal_module is not None:
            x = self.tam2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        if self.temporal_module is not None:
            x = self.tam3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)
        if not self.without_t_stride:
            x = self.t_pool1(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        if self.temporal_module is not None:
            x = self.tam3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        if self.temporal_module is not None:
            x = self.tam3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        if not self.without_t_stride:
            x = self.t_pool2(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        if self.temporal_module is not None:
            x = self.tam4a(x)
        # N x 512 x 14 x 14

        x = self.inception4b(x)
        if self.temporal_module is not None:
            x = self.tam4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        if self.temporal_module is not None:
            x = self.tam4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        if self.temporal_module is not None:
            x = self.tam4d(x)
        # N x 528 x 14 x 14

        x = self.inception4e(x)
        if self.temporal_module is not None:
            x = self.tam4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        if not self.without_t_stride:
            x = self.t_pool3(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        if self.temporal_module is not None:
            x = self.tam5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        if self.temporal_module is not None:
            x = self.tam5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)

        n_t, c = x.shape
        out = x.view(batch_size, -1, c)
        # average the prediction from all frames
        out = torch.mean(out, dim=1)
        return out


class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


def inception_v1(num_classes, without_t_stride, groups, dropout, temporal_module_name,
                 dw_conv, blending_frames, blending_method, pooling_method,
                 imagenet_pretrained=True, **kwargs):

    temporal_module = partial(temporal_modeling_module, name=temporal_module_name,
                              dw_conv=dw_conv,
                              blending_frames=blending_frames,
                              blending_method=blending_method) if temporal_module_name is not None \
        else None

    model = GoogLeNet(num_classes=num_classes, num_frames=groups, temporal_module=temporal_module,
                      dropout=dropout, without_t_stride=without_t_stride,
                      pooling_method=pooling_method)

    if imagenet_pretrained:
        state_dict = model_zoo.load_url(model_urls['googlenet'], map_location='cpu', progress=True)
        state_dict.pop('fc.weight', None)
        state_dict.pop('fc.bias', None)
        model.load_state_dict(state_dict, strict=False)
    return model
