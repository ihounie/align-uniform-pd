r"""
Adapted from https://github.com/HobbitLong/CMC/blob/f25c37e49196a1fe7dc5f7b559ed43c6fce55f70/models/alexnet.py
"""

import torch.nn as nn
from math import floor


class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)

def get_output_size(in_size, padding, kernel, stride):
    return floor((in_size+2*padding-kernel)/stride+1)
    
class SmallAlexNet(nn.Module):
    def __init__(self, in_channel=3, feat_dim=128, in_size=64):
        super(SmallAlexNet, self).__init__()

        conv_dim = in_size

        blocks = []

        # conv_block_1
        blocks.append(nn.Sequential(
            nn.Conv2d(in_channel, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))
        conv_dim = get_output_size(conv_dim, 1, 3, 1)
        conv_dim = get_output_size(conv_dim, 0, 3, 2)
        # conv_block_2
        blocks.append(nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))
        conv_dim = get_output_size(conv_dim, 1, 3, 1)
        conv_dim = get_output_size(conv_dim, 0, 3, 2)
        # conv_block_3
        blocks.append(nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))
        conv_dim = get_output_size(conv_dim, 1, 3, 1)
        # conv_block_4
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))
        conv_dim = get_output_size(conv_dim, 1, 3, 1)

        # conv_block_5
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))
        conv_dim = get_output_size(conv_dim, 1, 3, 1)
        conv_dim = get_output_size(conv_dim, 0, 3, 2)
        
        # fc6
        blocks.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * conv_dim * conv_dim, 4096, bias=False),  # 256 * 6 * 6 if 224 * 224
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        ))

        # fc7
        blocks.append(nn.Sequential(
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        ))

        # fc8
        blocks.append(nn.Sequential(
            nn.Linear(4096, feat_dim),
            L2Norm(),
        ))

        self.blocks = nn.ModuleList(blocks)
        self.init_weights_()

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init)

    def forward(self, x, *, layer_index=-1, intermediate_layer=None):
        if layer_index < 0:
            layer_index += len(self.blocks)
        for idx, layer in enumerate(self.blocks[:(layer_index + 1)]):
            x = layer(x)
            if idx==intermediate_layer:
                intermediate_act = x
        if intermediate_layer is not None:
            return x, intermediate_act
        else:
            return x

class AlexNet(SmallAlexNet):
    def __init__(self, num_layers = 5, in_channel=3, feat_dim=128, in_size=64):
        super(AlexNet, self).__init__()

        conv_dim = in_size

        blocks = []

        # conv_block_1
        blocks.append(nn.Sequential(
            nn.Conv2d(in_channel, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))
        conv_dim = get_output_size(conv_dim, 1, 3, 1)
        conv_dim = get_output_size(conv_dim, 0, 3, 2)
        out_channels=96
        # conv_block_2
        blocks.append(nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))
        conv_dim = get_output_size(conv_dim, 1, 3, 1)
        conv_dim = get_output_size(conv_dim, 0, 3, 2)
        out_channels=192
        # conv_block_3
        if num_layers>2:
            blocks.append(nn.Sequential(
                nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
            ))
            conv_dim = get_output_size(conv_dim, 1, 3, 1)
            out_channels=384
            if num_layers>3:
                # conv_block_4
                blocks.append(nn.Sequential(
                    nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(384),
                    nn.ReLU(inplace=True),
                ))    
                conv_dim = get_output_size(conv_dim, 1, 3, 1)
                out_channels=384
                if num_layers>4:
                    # conv_block_5
                    blocks.append(nn.Sequential(
                        nn.Conv2d(384, 192, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(192),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, 2),
                    ))
                    conv_dim = get_output_size(conv_dim, 1, 3, 1)
                    conv_dim = get_output_size(conv_dim, 0, 3, 2)
                    out_channels=192
                    if num_layers>5:
                        # added conv block
                        blocks.append(nn.Sequential(
                            nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(192),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(3, 2),
                        ))
                        conv_dim = get_output_size(conv_dim, 1, 3, 1)
                        conv_dim = get_output_size(conv_dim, 0, 3, 2)
                        out_channels=192
                        if num_layers>6:
                            # added conv block
                            blocks.append(nn.Sequential(
                                nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=False),
                                nn.BatchNorm2d(192),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(3, 2),
                            ))
                            conv_dim = get_output_size(conv_dim, 1, 3, 1)
                            conv_dim = get_output_size(conv_dim, 0, 3, 2)
                            out_channels=192
                            if num_layers>7:
                                # added conv block
                                blocks.append(nn.Sequential(
                                    nn.Conv2d(192, 96, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(96),
                                    nn.ReLU(inplace=True),
                                ))
                                conv_dim = get_output_size(conv_dim, 1, 3, 1)
                                out_channels=96
        blocks.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels * conv_dim * conv_dim, 4096, bias=False),  # 256 * 6 * 6 if 224 * 224
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        ))
        # fc7
        blocks.append(nn.Sequential(
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        ))

        blocks.append(nn.Sequential(
            nn.Linear(4096, feat_dim),
            L2Norm(),
        ))
        self.blocks = nn.ModuleList(blocks)
        self.init_weights_()

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init)

    def forward(self, x, *, layer_index=-1):
        if layer_index < 0:
            layer_index += len(self.blocks)
        for layer in self.blocks[:(layer_index + 1)]:
            x = layer(x)
        return x
