
import torch
import torch.nn.functional as F
import torch.nn as nn


class SEModule(nn.Module):

    def __init__(self, channels, dw_conv):
        super().__init__()
        ks = 1
        pad = (ks - 1) // 2
        self.fc1 = nn.Conv2d(channels, channels, kernel_size=ks,
                             padding=pad, groups=channels if dw_conv else 1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x


class TAM(nn.Module):

    def __init__(self, duration, channels, dw_conv=True, blending_frames=3, blending_method='sum'):
        super().__init__()
        self.blending_frames = blending_frames
        self.blending_method = blending_method

        if blending_frames == 3:
            self.prev_se = SEModule(channels, dw_conv)
            self.next_se = SEModule(channels, dw_conv)
            self.curr_se = SEModule(channels, dw_conv)
        else:
            self.blending_layers = nn.ModuleList([SEModule(channels, dw_conv) for _ in range(blending_frames)])
        self.relu = nn.ReLU(inplace=True)
        self.duration = duration

    def name(self):
        return "TAM-b{}-{}".format(self.blending_frames, self.blending_method)

    def forward(self, x):

        if self.blending_frames == 3:
            prev_x = self.prev_se(x)
            curr_x = self.curr_se(x)
            next_x = self.next_se(x)
            prev_x = prev_x.view((-1, self.duration) + prev_x.size()[1:])
            curr_x = curr_x.view((-1, self.duration) + curr_x.size()[1:])
            next_x = next_x.view((-1, self.duration) + next_x.size()[1:])

            prev_x = F.pad(prev_x, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1, ...]
            next_x = F.pad(next_x, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:, ...]

            out = torch.stack([prev_x, curr_x, next_x], dim=0)
        else:
            # multiple blending
            xs = [se(x) for se in self.blending_layers]
            xs = [x.view((-1, self.duration) + x.size()[1:]) for x in xs]

            shifted_xs = []
            for i in range(self.blending_frames):
                shift = i - (self.blending_frames // 2)
                x_temp = xs[i]
                n, t, c, h, w = x_temp.shape
                start_index = 0 if shift < 0 else shift
                end_index = t if shift < 0 else t + shift
                padding = None
                if shift < 0:
                    padding = (0, 0, 0, 0, 0, 0, abs(shift), 0)
                elif shift > 0:
                    padding = (0, 0, 0, 0, 0, 0, 0, shift)
                shifted_xs.append(F.pad(x_temp, padding)[:, start_index:end_index, ...]
                                  if padding is not None else x_temp)

            out = torch.stack(shifted_xs, dim=0)

        if self.blending_method == 'sum':
            out = torch.sum(out, dim=0)
        elif self.blending_method == 'max':
            out, _ = torch.max(out, dim=0)
        else:
            raise ValueError('Blending method %s not supported' % (self.blending_method))

        out = self.relu(out)
        # [N, T, C, N, H]
        n, t, c, h, w = out.shape
        out = out.view((-1, ) + out.size()[2:])
        # out = out.contiguous()

        return out


def temporal_modeling_module(name, duration, channels, dw_conv=True,
                             blending_frames=3, blending_method='sum'):
    if name is None or name == 'TSN':
        return None

    if name == 'TAM':
        return TAM(duration, channels, dw_conv, blending_frames, blending_method)
    else:
        raise ValueError('incorrect tsm module name %s' % name)

