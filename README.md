# LapH
> Code of **Multi-modal Image Fusion via Deep Laplacian Pyramid Hybrid Network** <br>
> Xing Luo, Guizhong Fu, Jiangxin Yang, Yanlong Cao, and Yanpeng Cao

#### Getting started

- Install dependencies. (Python3)
```bash
pip install -r requirements.txt
```

- Prepare the datasets following the ```datasets``` folder and create lmdb files.
```bash
PYTHONPATH='./' CUDA_VISIBLE_DEVICES=0 python scripts/data_preparation/create_lmdb.py
```

- Quick training/testing/testing_speed demo. Change and adapt the yml file ```options/*/*.yml``` at your wish.
```bash
PYTHONPATH='./' CUDA_VISIBLE_DEVICES=0 python codes/test.py -opt options/test/test_LapH_IV.yml
```

- To fuse grayscale image with rgb image, see the ```matlab``` folder for further guidance.

### Citation
If you use this code for your research, please consider citing our paper.
```
@inproceedings{luo2023LapH,
  title={Multi-modal Image Fusion via Deep Laplacian Pyramid Hybrid Network},
  author={Xing Luo, Guizhong Fu, Jiangxin Yang, Yanlong Cao, and Yanpeng Cao},
  booktitle={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023}
}
```

### Acknowledgement
This work is largely built on framework from the excellent [BasicSR](https://github.com/xinntao/BasicSR) and [LPTN](https://github.com/csjliang/LPTN) project.
And the authors would also appreciate the open-source codes by existing wonderful image fusion works.