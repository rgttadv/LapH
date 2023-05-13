from codes.utils import scandir
from codes.utils.lmdb_util import make_lmdb_from_imgs


def create_PET_HMedical_lmdb():

    folder_path = 'datasets/Medical/PET_MRI/MRI'
    lmdb_path = 'datasets/Medical/PET_MRI/TEST_MRI.lmdb'
    img_path_list, keys = prepare_keys(folder_path,suffix=['png'])
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = 'datasets/Medical/PET_MRI/PET'
    lmdb_path = 'datasets/Medical/PET_MRI/TEST_PET.lmdb'
    img_path_list, keys = prepare_keys(folder_path,suffix=['png'])
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def create_SPECT_HMedical_lmdb():

    folder_path = 'datasets/Medical/SPECT_MRI/MRI'
    lmdb_path = 'datasets/Medical/SPECT_MRI/TEST_MRI.lmdb'
    img_path_list, keys = prepare_keys(folder_path,suffix=['png'])
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = 'datasets/Medical/SPECT_MRI/SPECT'
    lmdb_path = 'datasets/Medical/SPECT_MRI/TEST_SPECT.lmdb'
    img_path_list, keys = prepare_keys(folder_path,suffix=['png'])
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_IV_lmdb():

    folder_path = 'datasets/IR_VIS/VIS'
    lmdb_path = 'datasets/IR_VIS/TEST_VIS.lmdb'
    img_path_list, keys = prepare_keys(folder_path, suffix=['bmp','jpg'])
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = 'datasets/IR_VIS/IR'
    lmdb_path = 'datasets/IR_VIS/TSET_IR.lmdb'
    img_path_list, keys = prepare_keys(folder_path, suffix=['bmp','jpg'])
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def prepare_keys(folder_path, suffix):

    print('Reading image path list ...')
    img_path_list = []
    keys = []
    for i in suffix:
        img_path_list += sorted(list(scandir(folder_path, suffix=i, recursive=False)))
    for i in suffix:
        for j in sorted(img_path_list):
            if i in j:
                keys += [j.split('.'+i)[0]]

    return img_path_list, keys


if __name__ == '__main__':

    create_IV_lmdb()

