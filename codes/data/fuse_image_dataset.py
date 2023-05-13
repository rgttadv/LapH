import random
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from codes.data.data_util import (paths_from_folder, paths_from_lmdb)
from codes.data.transforms import augment, paired_random_crop
from codes.utils import FileClient, imfrombytes, img2tensor

class FusionImageDataset(data.Dataset):

    def __init__(self, opt):
        super(FusionImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.vi_folder, self.ir_folder = opt['dataroot_vi'], opt['dataroot_ir']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.vi_folder, self.ir_folder]
            self.io_backend_opt['client_keys'] = ['vi', 'ir']
            self.paths_vi = paths_from_lmdb(self.vi_folder)
            self.paths_ir = paths_from_lmdb(self.ir_folder)

        elif self.io_backend_opt['type'] == 'disk':
            self.paths_vi = paths_from_folder(self.vi_folder)
            self.paths_ir = paths_from_folder(self.ir_folder)
        else:
            raise ValueError(
                f'io_backend not supported')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.

        vi_path = self.paths_vi[index % len(self.paths_vi)]
        img_bytes = self.file_client.get(vi_path, 'vi')
        img_vi = imfrombytes(img_bytes, flag='grayscale', float32=True)
        h, w = img_vi.shape
        img_vi = img_vi.reshape(h, w, 1)

        ir_path = self.paths_ir[index % len(self.paths_ir)]
        img_bytes = self.file_client.get(ir_path, 'ir')
        img_ir = imfrombytes(img_bytes, flag='grayscale', float32=True)
        img_ir = img_ir.reshape(h, w, 1)

        # augmentation for training
        if self.opt['phase'] == 'train':
            if_fix = self.opt['if_fix_size']
            gt_size = self.opt['gt_size']
            if not if_fix and self.opt['batch_size_per_gpu'] != 1:
                raise ValueError(
                    f'Param mismatch. Only support fix data shape if batchsize > 1 or num_gpu > 1.')

        # BGR to RGB, HWC to CHW, numpy to tensor

        img_vi, img_ir = img2tensor([img_vi, img_ir], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_vi, self.mean, self.std, inplace=True)
            normalize(img_ir, self.mean, self.std, inplace=True)

        return {
            'vi': img_vi,
            'ir': img_ir,
            'vi_path': vi_path,
            'ir_path': ir_path,
        }

    def __len__(self):
        return len(self.paths_vi)
