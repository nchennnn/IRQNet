# There is a mmsegmentation-style decode_head implementation
# refer to: https://mmsegmentation.readthedocs.io/en/0.x/
# envs: torch1.13.0+cu117+mmsegmentation0.30.0
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from torch.nn import functional as F
from torchvision.ops import deform_conv2d
import numpy as np
import math

from ..builder import HEADS
from .decode_head import BaseDecodeHead

# standard SE block (for feature focus x <- f(x) * x)
# channel (int): base channel
# reduction (int): channel reduction ratio
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Wavelet Position Encoding
# in_dim (int): input dimension
# out_dim (int): output dimension
# cat_input (bool): whether to concatenate the input and output
# require_grad (bool): whether the parameters in position encoding require gradients
# wfr (float): the maximum of wavelet frequency
# h (float): the full-width at half-maximum
class Wavelet_Morlet_PositionEncoding(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 cat_input=True,
                 require_grad=False,
                 wfr=1.0,
                 h=1.0):

        super().__init__()
        assert in_dim == 2, "input dimension must be 2"
        assert out_dim % (in_dim * 2) == 0, "dimension must be dividable"

        n = out_dim // 2 // in_dim

        # sinusoidal part matrix
        freq = wfr * np.log(np.linspace(1, n, n)) / np.log(n)
        freq = np.stack([freq] + [np.zeros_like(freq)] * (in_dim - 1), axis=-1)
        freq = np.concatenate([np.roll(freq, i, axis=-1) for i in range(in_dim)], axis=0)

        # Gaussian part matrix
        basis = np.ones(n)
        basis = np.stack([basis] + [np.zeros_like(basis)] * (in_dim - 1), axis=-1)
        basis = np.concatenate([np.roll(basis, i, axis=-1) for i in range(in_dim)], axis=0)

        m = out_dim // 2
        eps = 1e-6
        # a: scale  (caution) must be a>=0 due to a ** 0.5 (implemented by after_train_iter hook)
        a = np.ones(m)
        # b: shift
        b = np.zeros(m) + eps

        self.freq = torch.FloatTensor(freq)
        self.basis = torch.FloatTensor(basis)
        self.a = torch.FloatTensor(a)
        self.b = torch.FloatTensor(b)
        if require_grad:
            self.freq = nn.Parameter(self.freq, requires_grad=True)
            self.basis = nn.Parameter(self.basis, requires_grad=True)
            self.a = nn.Parameter(self.a, requires_grad=True)
            self.b = nn.Parameter(self.b, requires_grad=True)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cat_input = cat_input
        self.require_grad = require_grad

        self.h = h

    def forward(self, x):
        if not self.require_grad:
            self.freq = self.freq.to(x.device)
            self.basis = self.basis.to(x.device)
            self.a = self.a.to(x.device)
            self.b = self.b.to(x.device)
        x1 = (torch.matmul(x, self.freq.T)-self.b)/(self.a)  # Sinusoidal part
        x2 = (torch.matmul(x, self.basis.T)-self.b)/(self.a)  # Gaussian part
        if self.cat_input:
            return torch.cat([x, (1. / self.a ** 0.5) * torch.sin(x1) * (torch.e ** ((-4 * math.log(2) * (x2 ** 2)) / (self.h ** 2))),
                              (1. / self.a ** 0.5) * torch.cos(x1) * (torch.e ** ((-4 * math.log(2) * (x2 ** 2)) / (self.h ** 2)))], dim=-1)
        else:
            return torch.cat([(1. / self.a ** 0.5) * torch.sin(x1) * (torch.e ** ((-4 * math.log(2) * (x2 ** 2)) / (self.h ** 2))),
                              (1. / self.a ** 0.5) * torch.cos(x1) * (torch.e ** ((-4 * math.log(2) * (x2 ** 2)) / (self.h ** 2)))], dim=-1)

# ablation study without deformable fusion (point-wise sampling)
# make coordinates at grid centers
# shape (list): height and width of coordinate map
# flatten (bool): whether to flatten the output
def make_coord(shape, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = -1, 1
        r = (v1 - v0) / (2 * n)  # pixel spacing
        seq = v0 + r + (2 * r) * torch.arange(n).float()  # coordinate of each point
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)  # map of 2D coordinates
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

# Deformable Fusion operation
# channel-wise and point-wise
# args: convolution-like args
class DeformableFusion(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        *,
        offset_groups=1,
        with_mask=False
    ):
        super().__init__()
        assert in_dim % groups == 0, "in_dim must be dividable"
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.ones([out_dim, in_dim // groups, kernel_size, kernel_size]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.bias = None

        self.with_mask = with_mask
        if with_mask:
            # batch_size, (2+1) * offset_groups * kernel_height * kernel_width, out_height, out_width
            self.param_generator = nn.Conv2d(in_dim, 3 * offset_groups * kernel_size * kernel_size, 3, 1, 1)
        else:
            # batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width
            self.param_generator = nn.Conv2d(in_dim, 2 * offset_groups * kernel_size * kernel_size, 3, 1, 1)

    def forward(self, x):
        if self.with_mask:
            oh, ow, mask = self.param_generator(x).chunk(3, dim=1)
            offset = torch.cat([oh, ow], dim=1)
            mask = mask.sigmoid()
        else:
            offset = self.param_generator(x)
            mask = None
        x = deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
        # return the feature and offset
        return x, offset

# Information Enhancement Module
# wpe (bool): whether to use wavelet position encoding
# df (bool): whether to use deformable fusion
# pos_dim (int): C_PE of position encoding
# wfr (float): the maximum of wavelet frequency
# h (float): the full-width at half-maximum
# channel (int): base_channel of this layer
# require_grad (bool): whether the parameters in position encoding require gradients
class ir(nn.Module):
    def __init__(self, wpe=False, df=False, pos_dim=40, wfr=1.0, h=1.0,
                 channel=32, require_grad=False):

        super(ir, self).__init__()
        self.wpe = wpe
        self.df = df

        if self.df:  # deformable fusion
            self.deformable_fusion=DeformableFusion(in_dim=channel,out_dim=channel,kernel_size=3,groups=channel, with_mask=False)
        if self.wpe:  # position encoding
            self.pos = Wavelet_Morlet_PositionEncoding(2, pos_dim, wfr=wfr, h=h, require_grad=require_grad)


    def forward(self, x, size):
        h, w = size  # target size (h_1, w_1)

        # get the position signals and features (point-wise, coordinate-based)
        rel_coord, q_feat = self.ir_feat(x, [h, w], df=self.df)

        # enhance position signals
        if self.wpe:
            rel_coord = self.pos(rel_coord)

        x = torch.cat([rel_coord, q_feat], dim=-1)
        return x

    def ir_feat(self, x, size, df=False):
        # base size (h_k, w_k)
        bs, hh, ww = x.shape[0], x.shape[-2], x.shape[-1]
        # target size (h_1, w_1)
        h, w = size
        if df:  # deformable fusion
            if hh<h or ww<w:
                # simple implementation (deformable fusion in target size)
                x = F.upsample(input=x, size=size, mode='bilinear', align_corners=True)

            x, offset = self.deformable_fusion(x)

            # [b, c, h, w] -> [b, h, w, c]
            x = x.permute(0, 2, 3, 1)
            offset = offset.permute(0, 2, 3, 1)

            # split->sum(h/w)->stack
            offset_h = offset[:,:,:,0::2]
            offset_w = offset[:,:,:,1::2]
            offset = torch.stack([torch.sum(offset_h, dim=-1),torch.sum(offset_w, dim=-1)],dim=-1)

            offset = offset.view(offset.shape[0],-1, offset.shape[-1])
            x = x.view(x.shape[0],-1, x.shape[-1])
            # return [b, h_1*w_1, c]
            return offset, x
        else:   # ablation study without deformable fusion
            coords = (make_coord((h, w)).cuda().flip(-1) + 1) / 2  # range [-1,1] -> [0,1]
            coords = coords.unsqueeze(0).expand(bs, *coords.shape)  # [b, h*w, 2]
            coords = (coords * 2 - 1).flip(-1)  # range[0,1] -> [-1,1]

            # [b, 2, hh, ww]
            feat_coords = make_coord((hh, ww), flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(bs, 2, *(hh, ww))

            coords_ = coords.clone()
            coords_.clamp_(-1 + 1e-6, 1 - 1e-6)
            # feature value
            q_feat = F.grid_sample(x, coords_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
            # coordinates
            q_coord = F.grid_sample(feat_coords, coords_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
            rel_coord = coords - q_coord
            rel_coord[:, :, 0] *= hh  # x.shape[-2]
            rel_coord[:, :, 1] *= ww  # x.shape[-1]
            return rel_coord, q_feat


# Implicit Representation Querying (Decode Head)
# pos_dim (int): C_PE of position encoding
# wfr (list): the maximum of wavelet frequency
# h (list): the full-width at half-maximum
# wpe (bool): whether to use wavelet position encoding
# df (bool): whether to use deformable fusion
# featfocus (bool): whether to use feature focus
@HEADS.register_module()
class IRHead(BaseDecodeHead):
    def __init__(self,
                 pos_dim=24,
                 wfr=None,
                 h=None,
                 wpe=False,
                 df=False,
                 featfocus=False,
                 **kwargs):
        if wfr is None:
            wfr = [1.0, 1.0, 1.0, 1.0]
        if h is None:
            h = [1, 1, 1, 1]
        self.pos_dim = pos_dim
        self.wpe = wpe
        self.df = df
        self.featfocus = featfocus

        super(IRHead, self).__init__(input_transform='multiple_select', **kwargs)

        self.conv_seg = None
        self.init_cfg = None
        self.base_channel = self.in_channels[0]  # HRNet base_channel (topmost layer)

        # 4-layer implicit representation querying
        self.ir1 = ir(pos_dim=self.pos_dim, require_grad=True, wpe=self.wpe, wfr=wfr[0], h=h[0],
                      df=self.df, channel=self.base_channel)
        self.ir2 = ir(pos_dim=self.pos_dim, require_grad=True, wpe=self.wpe, wfr=wfr[1], h=h[1],
                      df=self.df, channel=self.base_channel*2)
        self.ir3 = ir(pos_dim=self.pos_dim, require_grad=True, wpe=self.wpe, wfr=wfr[2], h=h[2],
                      df=self.df, channel=self.base_channel*4)
        self.ir4 = ir(pos_dim=self.pos_dim, require_grad=True, wpe=self.wpe, wfr=wfr[3], h=h[3],
                      df=self.df, channel=self.base_channel*8)

        # input dimension of implicit querying function
        in_dim = self.base_channel * 15
        if wpe:
            in_dim += (self.pos_dim + 2) * 4
        else:
            in_dim += 2*4

        # feature focus
        if featfocus:
            self.FF = nn.ModuleList([
                SELayer(channel=self.base_channel),
                SELayer(channel=self.base_channel * 2),
                SELayer(channel=self.base_channel * 4),
                SELayer(channel=self.base_channel * 8)
            ])
        else:
            self.FF = nn.ModuleList([
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
                nn.Identity()
            ])

        # implicit querying function (4-layer mlp)
        self.iqf = nn.Sequential(
            nn.Conv1d(in_dim, 1024, 1), build_norm_layer(self.norm_cfg, num_features=1024, postfix=11)[1], nn.ReLU(),
            nn.Conv1d(1024, 512, 1), build_norm_layer(self.norm_cfg, num_features=512, postfix=12)[1], nn.ReLU(),
            nn.Conv1d(512, 256, 1), build_norm_layer(self.norm_cfg, num_features=256, postfix=13)[1], nn.ReLU(),
            nn.Conv1d(256, self.num_classes, 1))

    def _forward_feature(self, inputs):
        # input_transform='multiple_select': 4-layer feature maps of HRNet
        x = self._transform_inputs(inputs)

        # feature focus
        x = [
            self.FF[0](x[0]),
            self.FF[1](x[1]),
            self.FF[2](x[2]),
            self.FF[3](x[3])
        ]

        # implicit representation querying
        b, _, h, w = x[0].shape
        x = [
            self.ir1(x[0], size=[h, w]),
            self.ir2(x[1], size=[h, w]),
            self.ir3(x[2], size=[h, w]),
            self.ir4(x[3], size=[h, w])
        ]
        x = torch.cat(x, dim=-1).permute(0, 2, 1)

        # implicit querying function (and reshape)
        output = self.iqf(x).view(b, -1, h, w)

        return output

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        # we do not use cls_seg because we have done the classifying in implicit querying function
        # output = self.cls_seg(output)
        return output
