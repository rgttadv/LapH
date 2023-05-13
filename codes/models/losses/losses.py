import math
import torch
import numpy as np
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from codes.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        window = create_window(window_size)
        self.register_buffer('window', window)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        return ssim(img1, img2, window=self.window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)

class GradientLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(GradientLoss, self).__init__()
        # sobel operator
        g_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        g_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        # scharr operator for weak edges
        # g_x = torch.Tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
        # g_y = torch.Tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
        self.g_x = g_x.reshape([1, 1, 3, 3])
        self.g_y = g_y.reshape([1, 1, 3, 3])
        self.register_buffer('g_x_kernel', self.g_x)
        self.register_buffer('g_y_kernel', self.g_y)
        self.eps = eps

    def compute_loss(self, prediction, gt):
        diff = prediction - gt
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

    def conv_gradient(self, img):
        n_channels, _, kw, kh = self.g_x_kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.g_x_kernel, groups=n_channels), F.conv2d(img, self.g_y_kernel, groups=n_channels)

    def construct_gt(self, source):
        s_gx, s_gy = self.conv_gradient(source)
        return s_gx * s_gx + s_gy * s_gy

    def forward(self, prediction, gt_source1, gt_source2):
        gt = self.construct_gt(gt_source1+gt_source2)
        pr_gx, pr_gy = self.conv_gradient(prediction)
        pr = pr_gx * pr_gx + pr_gy * pr_gy
        return self.compute_loss(pr, gt)

class ContrastLoss(nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()
        kernel_left = torch.Tensor([[0,0,0],[-1,1,0],[0,0,0]])
        self.kl = kernel_left.reshape([1, 1, 3, 3])
        self.register_buffer('weight_left', self.kl)

        kernel_right = torch.Tensor([[0,0,0],[0,1,-1],[0,0,0]])
        self.kr = kernel_right.reshape([1, 1, 3, 3])
        self.register_buffer('weight_right', self.kr)

        kernel_up = torch.Tensor([[0,-1,0],[0,1, 0 ],[0,0,0]])
        self.ku = kernel_up.reshape([1, 1, 3, 3])
        self.register_buffer('weight_up', self.ku)

        kernel_down = torch.Tensor([[0,0,0],[0,1, 0],[0,-1,0]])
        self.kd = kernel_down.reshape([1, 1, 3, 3])
        self.register_buffer('weight_down', self.kd)

    def compute_loss(self, vi_img, ir_img, vi_hf, ir_hf, fuse, kernel):
        vi_img = F.interpolate(vi_img, size=(vi_hf.shape[-2], vi_hf.shape[-1]), mode='bilinear', align_corners=True)
        ir_img = F.interpolate(ir_img, size=(vi_hf.shape[-2], vi_hf.shape[-1]), mode='bilinear', align_corners=True)
        fuse = F.interpolate(fuse, size=(vi_hf.shape[-2], vi_hf.shape[-1]), mode='bilinear', align_corners=True)
        # hf weight
        temp_conv_weights = torch.Tensor([1]).float().repeat([1, 1, kernel, kernel]).cuda()
        # vi
        vi_avg = F.avg_pool2d(vi_hf, kernel)
        vi_avg = F.interpolate(vi_avg, scale_factor=kernel, mode='nearest')
        vi_count_arr = ((vi_hf - vi_avg) > 0).float()
        vi_count = F.conv2d(vi_count_arr, weight=temp_conv_weights, stride=kernel)
        vi_count[vi_count == 0] = kernel * kernel
        vi_sum = F.conv2d(vi_hf * vi_count_arr, weight=temp_conv_weights, stride=kernel)
        vi_weight = vi_sum / vi_count
        # ir
        ir_avg = F.avg_pool2d(ir_hf, kernel)
        ir_avg = F.interpolate(ir_avg, scale_factor=kernel, mode='nearest')
        ir_count_arr = ((ir_hf - ir_avg) > 0).float()
        ir_count = F.conv2d(ir_count_arr, weight=temp_conv_weights, stride=kernel)
        ir_count[ir_count == 0] = kernel * kernel
        ir_sum = F.conv2d(ir_hf * ir_count_arr, weight=temp_conv_weights, stride=kernel)
        ir_weight = ir_sum / ir_count
        weight = F.softmax(torch.cat([vi_weight, ir_weight], dim=1), dim=1)
        final_vi_weight = weight[:, 0:1, :, :]
        final_ir_weight = weight[:, 1:2, :, :]

        # img related

        vi_img = F.avg_pool2d(vi_img, kernel)
        ir_img = F.avg_pool2d(ir_img, kernel)
        fuse = F.avg_pool2d(fuse, kernel)

        # compute loss
        # vi
        D_vi_left = F.conv2d(vi_img, self.weight_left, padding=1)
        D_vi_right = F.conv2d(vi_img, self.weight_right, padding=1)
        D_vi_up = F.conv2d(vi_img, self.weight_up, padding=1)
        D_vi_down = F.conv2d(vi_img, self.weight_down, padding=1)
        # ir
        D_ir_left = F.conv2d(ir_img, self.weight_left, padding=1)
        D_ir_right = F.conv2d(ir_img, self.weight_right, padding=1)
        D_ir_up = F.conv2d(ir_img, self.weight_up, padding=1)
        D_ir_down = F.conv2d(ir_img, self.weight_down, padding=1)

        D_fuse_left = F.conv2d(fuse, self.weight_left, padding=1)
        D_fuse_right = F.conv2d(fuse, self.weight_right, padding=1)
        D_fuse_up = F.conv2d(fuse, self.weight_up, padding=1)
        D_fuse_down = F.conv2d(fuse, self.weight_down, padding=1)

        D_left = torch.pow(D_fuse_left - (final_vi_weight * D_vi_left + final_ir_weight * D_ir_left),2)
        D_right = torch.pow(D_fuse_right - (final_vi_weight * D_vi_right + final_ir_weight * D_ir_right),2)
        D_up = torch.pow(D_fuse_up - (final_vi_weight * D_vi_up + final_ir_weight * D_ir_up),2)
        D_down = torch.pow(D_fuse_down - (final_vi_weight * D_vi_down + final_ir_weight * D_ir_down),2)

        loss = (D_left + D_right + D_up +D_down)
        return torch.mean(loss)

    def forward(self, vi_img, ir_img, vi_pyr, ir_pyr, fused):
        loss = 0
        #assert len(vi_pyr) == len(ir_pyr), 'the number of hf in both pyr should be the same'
        for i in range(len(vi_pyr)):
            loss += self.compute_loss(vi_img, ir_img, vi_pyr[i], ir_pyr[i], fused, 2**(i+1))

        return loss

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred, target, weight, eps=self.eps, reduction=self.reduction)


class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

        Args:
            loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight)

    def forward(self, pred, weight=None):
        y_diff = super(WeightedTVLoss, self).forward(
            pred[:, :, :-1, :], pred[:, :, 1:, :], weight=weight[:, :, :-1, :])
        x_diff = super(WeightedTVLoss, self).forward(
            pred[:, :, :, :-1], pred[:, :, :, 1:], weight=weight[:, :, :, :-1])

        loss = x_diff + y_diff

        return loss


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(
                        x_features[k] - gt_features[k],
                        p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(
                        x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) -
                        self._gram_mat(gt_features[k]),
                        p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(
                        self._gram_mat(x_features[k]),
                        self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'standard':
            self.loss = None
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(
            input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'standard':
            if is_disc:
                if target_is_real:
                    loss = -torch.mean(input)
                else:
                    loss = torch.mean(input)
            else:
                loss = -torch.mean(input)
        elif self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.

        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (
        path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def compute_gradient_penalty(D, real_samples, fake_samples):

    # Random weight term for interpolation between real and fake samples
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.cuda.FloatTensor(real_samples.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

