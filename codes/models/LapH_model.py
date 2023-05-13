import importlib
import torch
import torch.nn as nn
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import time
import warnings
import numpy as np
import torch.nn.functional as F

from codes.models.archs import define_network
from codes.models.base_model import BaseModel
from codes.models.losses import compute_gradient_penalty
from codes.utils import get_root_logger, imwrite, tensor2img
import matplotlib.pyplot as plt
from torchvision import transforms
import os, cv2

loss_module = importlib.import_module('codes.models.losses')
metric_module = importlib.import_module('codes.metrics')

class LapH_Model(BaseModel):

    def __init__(self, opt):
        super(LapH_Model, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', False))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses

        if train_opt.get('gradient_opt'):
            gradient_type = train_opt['gradient_opt'].pop('type')
            self.gradient_weight = train_opt['gradient_opt'].pop('loss_weight')
            cri_gradient_cls = getattr(loss_module, gradient_type)
            self.cri_gradient = cri_gradient_cls(**train_opt['gradient_opt']).to(
                self.device)
        else:
            self.cri_gradient = None

        if train_opt.get('ssim_opt'):
            ssim_type = train_opt['ssim_opt'].pop('type')
            self.ssim_weight = train_opt['ssim_opt'].pop('loss_weight')
            cri_ssim_cls = getattr(loss_module, ssim_type)
            self.cri_ssim = cri_ssim_cls(**train_opt['ssim_opt']).to(
                self.device)
        else:
            self.cri_ssim = None

        if train_opt.get('contrast_opt'):
            contrast_type = train_opt['contrast_opt'].pop('type')
            self.contrast_weight = train_opt['contrast_opt'].pop('loss_weight')
            cri_contrast_cls = getattr(loss_module, contrast_type)
            self.cri_contrast = cri_contrast_cls(**train_opt['contrast_opt']).to(
                self.device)
        else:
            self.cri_contrast = None

        if train_opt.get('gan_opt'):
            gan_type = train_opt['gan_opt'].pop('type')
            cri_gan_cls = getattr(loss_module, gan_type)
            self.cri_gan = cri_gan_cls(**train_opt['gan_opt']).to(self.device)
        if train_opt.get('gp_opt'):
            self.gp_weight = train_opt['gp_opt'].pop('loss_weight')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.vi = data['vi'].to(self.device)
        self.ir = data['ir'].to(self.device)

    def optimize_parameters(self, current_iter):
        # optimize net_g

        self.optimizer_g.zero_grad()
        self.output, pyr_vi, pyr_ir = self.net_g(self.vi, self.ir)

        l_g_total = 0
        loss_dict = OrderedDict()

        # gradient loss
        if self.cri_gradient:
            l_g_gradient = self.gradient_weight * self.cri_gradient(self.output, self.vi, self.ir)
            l_g_total += l_g_gradient
            loss_dict['l_g_gradient'] = l_g_gradient
        # ssim loss
        if self.cri_ssim:
            l_g_ssim = (1. - (self.cri_ssim(self.vi, self.output)*0.5 + self.cri_ssim(self.ir, self.output)*0.5)) * self.ssim_weight
            l_g_total += l_g_ssim
            loss_dict['l_g_ssim'] = l_g_ssim
        if self.cri_contrast:
            l_g_contrast = self.contrast_weight * (self.cri_contrast(self.vi, self.ir, pyr_vi, pyr_ir, self.output))
            l_g_total += l_g_contrast
            loss_dict['l_g_contrast'] = l_g_contrast

        l_g_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output, pyr_vi, pyr_ir = self.net_g(self.vi, self.ir)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['vi_path'][0]))[0]
            self.feed_data(val_data)
            #self.slide_inference()
            self.test()

            visuals = self.get_current_visuals()
            vi_img = tensor2img([visuals['vi']])
            ir_img = tensor2img([visuals['ir']])
            result_img = tensor2img([visuals['result']])

            # tentative for out of GPU memory
            del self.vi
            del self.ir
            del self.output
            #del self.count_mat
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             str(current_iter),
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')

                # plot_img = np.hstack((vi_img, ir_img, result_img))
                plot_img = result_img
                imwrite(plot_img, save_img_path)
            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    metric_function = getattr(metric_module, metric_type)
                    self.metric_results[name] += metric_function(vi_img, ir_img, result_img, **opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def slide_inference(self):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = int(self.opt['val']['stride']), int(self.opt['val']['stride'])
        h_crop, w_crop = int(self.opt['val']['crop_size']), int(self.opt['val']['crop_size'])
        batch_size, _, h_img, w_img = self.vi.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        self.output = self.vi.new_zeros((batch_size, 1, h_img, w_img))
        self.count_mat = self.vi.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_vi_img = self.vi[:, :, y1:y2, x1:x2]
                crop_ir_img = self.ir[:, :, y1:y2, x1:x2]
                self.net_g.eval()
                with torch.no_grad():
                    crop_output, pyr_vi, pyr_ir = self.net_g(crop_vi_img, crop_ir_img)
                self.net_g.train()
                self.output += F.pad(crop_output,
                               (int(x1), int(self.output.shape[3] - x2), int(y1),
                                int(self.output.shape[2] - y2)))

                self.count_mat[:, :, y1:y2, x1:x2] += 1
        assert (self.count_mat == 0).sum() == 0
        self.output = self.output.div_(self.count_mat)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['vi'] = self.vi.detach().cpu()
        out_dict['ir'] = self.ir.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
