import torch.nn as nn
import torch.nn.functional as F
import torch, math
import numpy as np
from .vision_transformer import *

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=1):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channles, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channles, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        #self.theta = theta

    def forward(self, x):
        #print('x.shape:', x.shape)
        out_normal = self.conv(x)
        #print('out_normal.shape:', out_normal.shape)
        # if math.fabs(self.theta - 0.) < 1e-8:
        #     return out_normal
        # else:
        # [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
        kernel_diff = self.conv.weight.sum(dim=[2, 3], keepdim=True)
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                            padding=0, groups=self.conv.groups)
        #print('out_diff.shape', out_diff.shape)
        return out_normal - out_diff


class HDCB(nn.Module):
    def __init__(self, dilation_rates, embedding_dim=32):
        super(HDCB, self).__init__()
        self.dilation_1 = nn.Sequential(
            Conv2d_cd(embedding_dim, embedding_dim, padding=dilation_rates[0], dilation=dilation_rates[0],
                      groups=embedding_dim),
            nn.PReLU()
        )
        self.d1_1x1 = nn.Sequential(
            nn.Conv2d(embedding_dim * 2, embedding_dim, kernel_size=1),
            nn.LeakyReLU()
        )

        self.dilation_2 = nn.Sequential(
            Conv2d_cd(embedding_dim, embedding_dim, padding=dilation_rates[1], dilation=dilation_rates[1],
                      groups=embedding_dim),
            nn.PReLU()
        )
        self.d2_1x1 = nn.Sequential(
            nn.Conv2d(embedding_dim * 3, embedding_dim, kernel_size=1),
            nn.LeakyReLU()
        )

        self.dilation_3 = nn.Sequential(
            Conv2d_cd(embedding_dim, embedding_dim, padding=dilation_rates[2], dilation=dilation_rates[2],
                      groups=embedding_dim),
            nn.PReLU()
        )
        self.d3_1x1 = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        d1 = self.dilation_1(x)
        d2 = self.dilation_2(self.d1_1x1(torch.cat([x, d1], dim=1)))
        d3 = self.dilation_3(self.d2_1x1(torch.cat([x, d1, d2], dim=1)))
        out = self.d3_1x1(torch.cat([x, d1, d2, d3], dim=1))

        return out


class Fuse_HF(nn.Module):
    def __init__(self, hf_dim=16, dilation_rates=[3, 2, 1], num_hdcb=3, res_prefix='or'):
        super(Fuse_HF, self).__init__()
        self.FEB = nn.Sequential(
            nn.Conv2d(2, hf_dim//2, 3, padding=1),
            nn.InstanceNorm2d(hf_dim//2),
            nn.LeakyReLU(),
            nn.Conv2d(hf_dim//2, hf_dim, 3, padding=1),
            nn.LeakyReLU()
        )
        self.num_hdcb = num_hdcb
        self.res_prefix = res_prefix
        for i in range(self.num_hdcb):
            hd_block = HDCB(dilation_rates=dilation_rates, embedding_dim=hf_dim)
            setattr(self, 'hf_hd_block_{}_{}'.format(self.res_prefix, str(i)), hd_block)
        self.scale = nn.Conv2d(hf_dim, 2, kernel_size=1)
        self.shift = nn.Conv2d(hf_dim, 2, kernel_size=1)
        self.final_fuse = nn.Conv2d(2, 1, 1)

    def forward(self, x1, x2):
        initial_feature = self.FEB(torch.cat([x1, x2], dim=1))
        embedding_feature = []
        for i in range(self.num_hdcb):
            self.hf_hd_block = getattr(self, 'hf_hd_block_{}_{}'.format(self.res_prefix, str(i)))
            initial_feature = self.hf_hd_block(initial_feature)
            embedding_feature.append(initial_feature)
        scale = self.scale(initial_feature)
        shift = self.shift(initial_feature)
        out = torch.cat([x1,x2],dim=1) * (scale + 1.) + shift
        out = self.final_fuse(F.leaky_relu(out))

        return out, embedding_feature

class Fuse_LF(nn.Module):
    def __init__(self, lf_dim=32, num_heads=2, mlp_ratio=2, qkv_bias=True, norm_layer=None, act_layer=nn.ReLU):
        super(Fuse_LF, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or 'GELU'
        self.FEB = nn.Sequential(
            nn.Conv2d(2, lf_dim//2, 3, padding=1),
            nn.InstanceNorm2d(lf_dim//2),
            nn.LeakyReLU(),
            nn.Conv2d(lf_dim//2, lf_dim, 3, padding=1),
            nn.LeakyReLU()
        )
        self.MRM_1 = MRM(dim=lf_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      norm_layer=norm_layer, act_layer=act_layer)
        self.MRM_2 = MRM(dim=lf_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      norm_layer=norm_layer, act_layer=act_layer)
        self.MRM_3 = MRM(dim=lf_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                      norm_layer=norm_layer, act_layer=act_layer)

        self.out = nn.Sequential(
            nn.Conv2d(lf_dim, lf_dim//2, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(lf_dim//2, 1, 3, padding=1)
        )

        self.apply(self.weight_init)

    def weight_init(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            stdv = 1. / math.sqrt(module.weight.size(1))
            module.weight.data.uniform_(-stdv, stdv)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.uniform_(-stdv, stdv)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x1, x2, e):
        input = self.FEB(torch.cat([x1, x2], dim=1))
        trans_out = self.MRM_1(input, e[0])
        trans_out = self.MRM_2(trans_out, e[1])
        trans_out = self.MRM_3(trans_out, e[2])
        out = trans_out + input
        out = self.out(out)
        out = torch.tanh(out)
        return out

class LapH(nn.Module):
    def __init__(self,  hf_dim=16, lf_dim=32, num_high=3):
        super(LapH, self).__init__()
        # Pre-processing
        self.lap_pyramid = Lap_Pyramid_Conv(num_high)
        # High frequency branch
        self.hf_branch_or = Fuse_HF(hf_dim=hf_dim, num_hdcb=3, res_prefix='or')
        self.hf_branch_12 = Fuse_HF(hf_dim=hf_dim, num_hdcb=3, res_prefix='12')
        self.hf_branch_14 = Fuse_HF(hf_dim=hf_dim, num_hdcb=3, res_prefix='14')
        # Low frequency branch
        self.lf_branch = Fuse_LF(lf_dim=lf_dim, num_heads=2)

    def forward(self, vi_images, ir_images):
        # Pyramids' resolution list: [1, 1/2, 1/4, ...]
        pyr_vi = self.lap_pyramid.pyramid_decom(img=vi_images)
        pyr_ir = self.lap_pyramid.pyramid_decom(img=ir_images)

        # High frequency branches
        high_fre_14, hf_features_14 = self.hf_branch_14(pyr_vi[-2], pyr_ir[-2])
        high_fre_12, hf_features_12 = self.hf_branch_12(pyr_vi[-3], pyr_ir[-3])
        high_fre_or, hf_features_or = self.hf_branch_or(pyr_vi[-4], pyr_ir[-4])

        # Low frequency branch
        low_fre = self.lf_branch(pyr_vi[-1], pyr_ir[-1], [i for i in zip(hf_features_14, hf_features_12, hf_features_or)])

        pyr_result = [high_fre_or, high_fre_12, high_fre_14, low_fre]
        fused_results = self.lap_pyramid.pyramid_recons(pyr_result)

        return fused_results, list(reversed(pyr_vi[:-1])), list(reversed(pyr_ir[:-1]))