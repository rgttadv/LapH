import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import time
import numpy as np

from codes.models.archs import define_network
from codes.models.base_model import BaseModel
from codes.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('codes.models.losses')
metric_module = importlib.import_module('codes.metrics')

class LapHTestModel(BaseModel):

    def __init__(self, opt):
        super(LapHTestModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True))

    def feed_data(self, data):
        self.vi = data['vi'].to(self.device)
        self.ir = data['ir'].to(self.device)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output, pyr_vi, pyr_ir = self.net_g(self.vi, self.ir)

    def test_speed(self, times_per_img=50, size=None):
        if size is not None:
            vi_img = self.vi.resize_(1, 3, size[0], size[1])
        else:
            vi_img = self.vi
            ir_img = self.ir
        self.net_g.eval()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(times_per_img):
                _, _, _ = self.net_g(vi_img, ir_img)
            torch.cuda.synchronize()
            self.duration = (time.time() - start) / times_per_img

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            vi_img = tensor2img([visuals['vi']])
            ir_img = tensor2img([visuals['ir']])
            result_img = tensor2img([visuals['result']])

            # tentative for out of GPU memory
            del self.vi
            del self.ir
            del self.output
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

                plot_img = result_img
                imwrite(plot_img, save_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(result_img, gt_img, **opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def nondist_validation_speed(self, dataloader, times_per_img, num_imgs, size=None):

        avg_duration = 0
        for idx, val_data in enumerate(dataloader):
            if idx > num_imgs:
                break
            img_name = osp.splitext(osp.basename(val_data['vi_path'][0]))[0]
            self.feed_data(val_data)
            self.test_speed(times_per_img, size=size)
            avg_duration += self.duration / num_imgs
            print(f'{idx} Testing {img_name} (shape: {self.vi.shape[2]} * {self.vi.shape[3]}) duration: {self.duration}')

        print(f'average duration is {avg_duration} seconds')


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
