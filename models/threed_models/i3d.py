import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from models.inflate_from_2d_model import inflate_from_2d_model

__all__ = ['i3d']

model_urls = {
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}


class I3D(nn.Module):
    def __init__(self, num_classes=1000, dropout_ratio=0.2, without_t_stride=False,
                 pooling_method='max'):
        super(I3D, self).__init__()
        self.pooling_method = pooling_method.lower()
        if self.pooling_method == 'avg':
            self.pooling_functor = F.avg_pool3d
        else:
            self.pooling_functor = F.max_pool3d
        self.without_t_stride = without_t_stride
        self.t_s = 1 if without_t_stride else 2
        self.conv1 = BasicConv3d(3, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3))
        self.conv2 = BasicConv3d(64, 64, kernel_size=1)
        self.conv3 = BasicConv3d(64, 192, kernel_size=3, stride=1, padding=1)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528,    256, 160, 320, 32, 128, 128)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.dropout = nn.Dropout(dropout_ratio)
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)

    def mean(self, modality='rgb'):
        return [0.5, 0.5, 0.5] if modality == 'rgb' else [0.5]

    def std(self, modality='rgb'):
        return [0.5, 0.5, 0.5] if modality == 'rgb' else [0.5]

    @property
    def network_name(self):
        name = 'i3d_v2'
        if not self.without_t_stride:
            name += "-ts-{}".format(self.pooling_method)
        return name

    def forward(self, x):
        # N x 3 x F x 224 x 224
        x = self.conv1(x)
        # N x 64 x F x 112 x 112
        x = self.pooling_functor(x, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        # N x 64 x F x 56 x 56
        x = self.conv2(x)
        # N x 64 x F x 56 x 56
        x = self.conv3(x)
        # N x 192 x F x 56 x 56
        # difference from the original I3D , which does temporal downsampling in the first conv
        x = self.pooling_functor(x, kernel_size=(3, 3, 3), stride=(self.t_s, 2, 2),
                                 padding=(1, 1, 1))
        # N x 192 x (F/2) x 28 x 28
        x = self.inception3a(x)
        # N x 256 x (F/2) x 28 x 28
        x = self.inception3b(x)
        # N x 480 x (F/2) x 28 x 28
        x = self.pooling_functor(x, kernel_size=(3, 3, 3), stride=(self.t_s, 2, 2),
                                 padding=(1, 1, 1))
        # N x 480 x (F/4) x 14 x 14
        x = self.inception4a(x)
        # N x 512 x (F/4) x 14 x 14
        x = self.inception4b(x)
        # N x 512 x (F/4) x 14 x 14
        x = self.inception4c(x)
        # N x 512 x (F/4) x 14 x 14
        x = self.inception4d(x)
        # N x 528 x (F/4) x 14 x 14
        x = self.inception4e(x)
        # N x 832 x (F/4) x 14 x 14
        x = self.pooling_functor(x, kernel_size=(3, 3, 3), stride=(self.t_s, 2, 2),
                                 padding=(1, 1, 1))
        # N x 832 x (F/8) x 7 x 7
        x = self.inception5a(x)
        # N x 832 x (F/8) x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x (F/8) x 7 x 7
        num_frames = x.shape[2]
        x = F.adaptive_avg_pool3d(x, output_size=(num_frames, 1, 1))
        # N x 1024 x ((F/8)-1) x 1 x 1
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        x = x.transpose(1, 2)
        n, c, nf = x.size()
        x = x.contiguous().view(n * c, -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.view(n, c, -1)
        # N x num_classes x ((F/8)-1)
        logits = torch.mean(x, 1)
        # N x num_classes
        return logits

class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv3d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv3d(in_channels, ch3x3red, kernel_size=1),
            BasicConv3d(ch3x3red, ch3x3, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv3d(in_channels, ch5x5red, kernel_size=1),
            BasicConv3d(ch5x5red, ch5x5, kernel_size=3, stride=1, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1, ceil_mode=True),
            BasicConv3d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0)):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)

        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def i3d(num_classes, dropout, without_t_stride, pooling_method, **kwargs):
    model = I3D(num_classes=num_classes, dropout_ratio=dropout,
                without_t_stride=without_t_stride, pooling_method=pooling_method)
    new_model_state_dict = model.state_dict()
    state_dict = model_zoo.load_url(model_urls['googlenet'], map_location='cpu', progress=True)
    state_d = inflate_from_2d_model(state_dict, new_model_state_dict,
                                    skipped_keys=['fc', 'aux1', 'aux2'])
    model.load_state_dict(state_d, strict=False)
    return model
