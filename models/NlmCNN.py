import torch.nn as nn
from .DnCNN import make_net

def mul_weights_patches(x, w, kernel, stride=1, padding=False):
    if padding:
        import torch.nn.functional as F
        pad_row = -(x.shape[2] - kernel) % stride
        pad_col = -(x.shape[3] - kernel) % stride
        x = F.pad(x, (pad_col // 2, pad_col - pad_col // 2, pad_row // 2, pad_row - pad_row // 2))

    # Extract patches
    w = w.permute(0, 2, 3, 1)
    w = w.view(w.shape[0], 1, w.shape[1], w.shape[2], kernel, kernel)
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)

    pad_row = (w.shape[2] - patches.shape[2])//2
    pad_col = (w.shape[3] - patches.shape[3])//2
    if (pad_row<0) or (pad_col<0):
        patches = patches[:,:,max(-pad_row,0):(patches.shape[2] - max(-pad_row,0)), max(-pad_col,0):(patches.shape[3] - max(-pad_col,0)),:,:]
    if (pad_row>0) or (pad_col>0):
        w = w[:,:,max(pad_row,0):(w.shape[2] - max(pad_row,0)), max(pad_col,0):(w.shape[3] - max(pad_col,0)),:,:]

    y = (patches * w).sum((4,5))

    return y

class NlmCNN(nn.Module):
    r"""
    Implements a DnCNN network
    """
    def __init__(self, network_weights, sizearea, sar_data = False, padding=False):
        r"""
        :param residual: whether to add a residual connection from input to output
        :param padding: inteteger for padding
        """
        super(NlmCNN, self).__init__()

        self.network_weights = network_weights
        self.sizearea = sizearea
        self.padding  = padding
        self.sar_data = sar_data

    def forward_weights(self, x, reshape=False):
        if self.sar_data:
            x_in = x.abs().log() / 2.0
        else:
            x_in = x
        w = self.network_weights(x_in)
        if reshape:
            w = w.permute(0, 2, 3, 1)
            w = w.view(w.shape[0], w.shape[1], w.shape[2], self.sizearea, self.sizearea)
            return w
        else:
            return w

    def forward(self, x):
        w = self.forward_weights(x)
        y = mul_weights_patches(x, w, self.sizearea, stride=1, padding=self.padding)
        return y

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
        #print(f"as input: {x.size()}")
        x = self.shortcut(self.convs1(x), x)
        #print(f"after 1st conv block: {x.size()}")
        x = self.n3block1(x, x)
        #print(f"after 1st n3block: {x.size()}")
        x = self.shortcut(self.convs2(x), x)
        #print(f"after shortcut + 2nd conv block: {x.size()}")
        x = self.n3block2(x, x)
        #print(f"after 2nd n3block: {x.size()}")
        #print(f"current x shape: {x.shape}")
        #print(f"self.convs3: {self.convs3}")
        x = self.convs3(x)
        #print(x)
        #print(f"after 3rd conv block + softmax: {x.size()}")
        return x

def make_backnet(nplanes_in, sizearea, bn_momentum=0.1, n3block_opt={}, padding=False):
    network_weights = N3BackNet(nplanes_in, sizearea, n3block_opt, bn_momentum, padding=padding)
    return network_weights
