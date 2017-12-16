"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2017-2018
"""

import torch
import torch.nn as nn


class MaskedConv(nn.Conv2d):
    def __init__(self,mask_type,in_channels,out_channels,kernel_size,stride=1):
        super(MaskedConv,self).__init__(in_channels,out_channels,kernel_size,
                                        stride,padding=kernel_size//2)
        assert mask_type in ('A','B')
        mask = torch.ones(1,1,kernel_size,kernel_size)
        mask[:,:,kernel_size//2,kernel_size//2+(mask_type=='B'):] = 0
        mask[:,:,kernel_size//2+1:] = 0
        self.register_buffer('mask',mask)

    def forward(self,x):
        self.weight.data *= self.mask
        return super(MaskedConv,self).forward(x)


class GatedRes(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,
                 aux_channels=0):
        super(GatedRes,self).__init__()
        self.conv = MaskedConv('A',in_channels,2*out_channels,kernel_size,
                               stride)
        self.out_channels = out_channels
        if aux_channels!=2*out_channels and aux_channels!=0:
            self.aux_shortcut = nn.Sequential(
                nn.Conv2d(aux_channels,2*out_channels,1),
                nn.BatchNorm2d(2*out_channels,momentum=0.1))
        if in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,1),
                nn.BatchNorm2d(out_channels,momentum=0.1))
        self.batchnorm = nn.BatchNorm2d(out_channels,momentum=0.1)

    def forward(self,x):
        # check for aux input from first half of net stacked into x
        if x.dim()==5:
            x,aux = torch.split(x,1,dim=0)
            x = torch.squeeze(x,0)
            aux = torch.squeeze(x,0)
        else:
            aux = None
        x1 = self.conv(x)
        if aux is not None:
            if hasattr(self,'aux_shortcut'):
                aux = self.aux_shortcut(aux)
            x1 = (x1+aux)/2
        # split for gate (note: pytorch dims are [n,c,h,w])
        xf,xg = torch.split(x1,self.out_channels,dim=1)
        xf = torch.tanh(xf)
        xg = torch.sigmoid(xg)
        if hasattr(self,'shortcut'):
            x = self.shortcut(x)
        return x+self.batchnorm(xg*xf)


class PixelCNN(nn.Module):
    def __init__(self,in_channels,n_features,n_layers,n_scales,n_bins,
                 dropout=0.5):
        super(PixelCNN,self).__init__()

        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.n_scales = n_scales

        # Up pass
        self.input_batchnorm = nn.BatchNorm2d(in_channels,momentum=0.1)
        for s in range(n_scales):
            for l in range(n_layers):
                if s==0 and l==0:  # start with normal conv
                    block = nn.Sequential(
                        MaskedConv('A',in_channels+1,n_features,kernel_size=7),
                        nn.BatchNorm2d(n_features,momentum=0.1),
                        nn.ReLU())
                else:
                    if l==0:  # start new scale with strided conv
                        block = nn.Sequential(MaskedConv('A', n_features,
                                                         n_features, 3, 2))
                    else:
                        block = nn.Sequential(GatedRes(n_features, n_features))
                self.layers.append(block)

        # Down pass
        for s in range(n_scales):
            for l in range(n_layers):
                # dropout rate increases linearly in second half of net
                p_drop = dropout*(s*n_layers+l)/n_layers/n_scales
                block = [nn.Dropout2d(p_drop),
                         GatedRes(n_features, n_features,
                                  aux_channels=n_features)]
                # finish scale with upsampling
                if l == n_layers-1 and s < n_scales-1:
                    block.append(nn.Upsample(scale_factor=2, mode='nearest'))
                self.layers.append(nn.Sequential(*block))


        # Last layer: project to n_bins (output is [-1, n_bins, h, w])
        self.layers.append(
            nn.Sequential(nn.Dropout2d(dropout),
                          nn.Conv2d(n_features,n_bins,1),
                          nn.LogSoftmax(dim=1)))

    def forward(self,x):
        # Add channel of ones so network can tell where padding is
        x = nn.functional.pad(x,(0,0,0,0,0,1,0,0),mode='constant',value=1)

        # Up pass
        features = []
        i = -1
        for s in range(self.n_scales):
            for l in range(self.n_layers):
                i += 1
                x = self.layers[i](x)
                features.append(x)

        # Down pass
        for s in range(self.n_scales):
            for l in range(self.n_layers):
                i += 1
                x = self.layers[i](torch.stack((x,features.pop())))

        # Last layer
        i += 1
        x = self.layers[i](x)
        assert i==len(self.layers)-1
        assert len(features)==0
        return x







