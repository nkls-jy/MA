import torch.nn as nn
from .DnCNN import make_net

# defining nonlocal means backnet with N³ blocks
class N3BackNet(nn.Module):

    # convolution blocks
    def createconvs(self, nplanes_in, nplanes_out, depth = 6, lastact='linear', bn_momentum=0.1, padding=False):
        features = [64, ]  * (depth-1) + [nplanes_out, ]
        kernels = [3, ] * depth
        dilats = [1, ] * depth
        acts = ['relu', ] * (depth-1) + [lastact, ]
        bns = [False, ] + [True, ] * (depth - 2) + [False, ]
        padding = None if padding else 0
        return make_net(nplanes_in, kernels, features, bns, acts, dilats=dilats, bn_momentum=bn_momentum, padding=padding)

    #shortcut connection
    def shortcut(self, x, pre):
        nshortcut = min(x.shape[1], pre.shape[1])
        p0 = (pre.shape[2] - x.shape[2])//2
        p1 = (pre.shape[3] - x.shape[3])//2
        y = x[:, :nshortcut, :, :] + pre[:, :nshortcut, p0:(pre.shape[2] - p0), p1:(pre.shape[3] - p1)]
        if nshortcut<x.shape[1]:
            from torch import cat
            y = cat((y, x[:, nshortcut:, :, :]), 1)

        return y

    # called when an object of class N3BackNet is created
    def __init__(self, nplanes_in, sizearea, n3block_opt, bn_momentum=0.1, padding=False):
        r"""
        :param nplanes_in: number of input features
        :param nplanes_out: number of output features
        :param nplanes_interm: number of intermediate features, i.e. number of output features for the DnCNN sub-networks
        :param nblocks: number of DnCNN sub-networks
        :param block_opt: options passed to DnCNNs
        :param nl_opt: options passed to N3Blocks
        :param residual: whether to have a global skip connection
        """
        # returens a proxy object (temp object of superclass)
        super(N3BackNet, self).__init__()
        from n3net.n3block import N3Block
        self.nplanes_in  = nplanes_in
        self.nplanes_out = nplanes_in

        # create layers 
        # 1-5: conv
        # 6: conv + skip
        self.convs1 = self.createconvs(nplanes_in, 8, depth=6, lastact='relu', bn_momentum=bn_momentum, padding=padding)
        # 7: N³ block
        self.n3block1 = N3Block(8, 8, **n3block_opt)
        # 8-12: conv
        # 13: conv + skip
        self.convs2 = self.createconvs(self.n3block1.nplanes_out, 8, depth=6, lastact='relu', bn_momentum=bn_momentum, padding=padding)
        # 14: N³ block
        self.n3block2 = N3Block(8, 8, **n3block_opt)
        # 15-19: conv
        # 20: conv (with softmax)
        self.convs3 = self.createconvs(self.n3block2.nplanes_out, sizearea*sizearea, depth=6, lastact='softmax', bn_momentum=bn_momentum, padding=padding)

    # forward pass through network
    def forward(self, x):
        x = self.shortcut(self.convs1(x), x)
        x = self.n3block1(x, x)
        x = self.shortcut(self.convs2(x), x)
        x = self.n3block2(x, x)
        x = self.convs3(x)
        return x

def make_backnet(nplanes_in, sizearea, bn_momentum=0.1, n3block_opt={}, padding=False):
    network_weights = N3BackNet(nplanes_in, sizearea, n3block_opt, bn_momentum, padding=padding)
    return network_weights
