import torch.nn as nn


class TemporalPooling(nn.Module):

    def __init__(self, frames, kernel_size=3, stride=2, mode='avg'):
        """

        Parameters
        ----------
        frames (int): number of input frames
        kernel_size
        stride
        mode
        """
        super().__init__()
        self.frames = frames
        pad_size = (kernel_size - 1) // stride
        if mode == 'avg':
            self.pool = nn.AvgPool3d(kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1),
                                     padding=(pad_size, 0, 0))
        elif mode == 'max':
            self.pool = nn.MaxPool3d(kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1),
                                     padding=(pad_size, 0, 0))
        else:
            raise ValueError("only support avg or max")

    def forward(self, x):
        nt, c, h, w = x.shape
        x = x.view((-1, self.frames) + x.size()[1:]).transpose(1, 2)
        x = self.pool(x)
        x = x.transpose(1, 2).contiguous().view(-1, c, h, w)
        return x
