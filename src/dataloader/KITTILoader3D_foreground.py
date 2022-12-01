from os import listdir
from os.path import isfile, join


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader_sls_fusion(filepath, train_file, val_file, gth='dense'):
    """

    :param filepath:
    :param train_file:
    :param val_file:
    :return:
    """
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    left_lidar_fold = 'depth_map_4beams_left/'
    right_lidar_fold = 'depth_map_4beams_right/'
    calib = 'calib/'
    if gth=='dense':
        # train_idx: 3622, val_idx: 3696 take from ForeSEe gth, missing some file
        depth_dense_gth = 'dense_depth_gth/' # gth 11 accum LiDAR 64 beams
        mypath_train = filepath + '/' + depth_dense_gth + 'train'
        mypath_val = filepath + '/' + depth_dense_gth + 'val'
        train_idx = [f[:-4] for f in listdir(mypath_train) if isfile(join(mypath_train, f))]
        val_idx = [f[:-4] for f in listdir(mypath_val) if isfile(join(mypath_val, f))]
    else:
        # train_idx_0: 3712, val_idx_0:3769 from original file
        depth_64beams_gth = 'depth_map_64beams_left/' # gth sparse gth LiDAR 64 beams
        with open(train_file, 'r') as f:
            train_idx = [x.strip() for x in f.readlines() if len(x.strip())>0]
        with open(val_file, 'r') as f:
            val_idx = [x.strip() for x in f.readlines() if len(x.strip())>0]

    # train ############################################################################################################
    left_train = [filepath + '/' + left_fold + img + '.png' for img in train_idx if
                  isfile(filepath + '/' + left_fold + img + '.png')]
    right_train = [filepath + '/' + right_fold + img + '.png' for img in train_idx if
                   isfile(filepath + '/' + right_fold + img + '.png')]
    left_lidar_train = [filepath + '/' + left_lidar_fold + img + '.png' for img in train_idx if
                         isfile(filepath + '/' + left_lidar_fold + img + '.png')]
    right_lidar_train = [filepath + '/' + right_lidar_fold + img + '.png' for img in train_idx if
                          isfile(filepath + '/' + right_lidar_fold + img + '.png')]
    calib_train = [filepath + '/' + calib + img + '.txt' for img in train_idx if
                   isfile(filepath + '/' + calib + img + '.txt')]
    # gth
    if gth == 'dense': # 11 frames
        depth_gth_train = [filepath + '/' + depth_dense_gth + 'train/' + img + '.png' for img in train_idx if
                             isfile(filepath + '/' + depth_dense_gth + 'train/' + img + '.png')]
    elif gth == 'sparse': # 64 beams
        # depth_train = [filepath + '/' + depth_L + img + '.npy' for img in train_idx if isfile(filepath + '/' + depth_L + img + '.npy')] # depth 64
        depth_gth_train = [filepath + '/' + depth_64beams_gth + img + '.npy' for img in train_idx if
                              isfile(filepath + '/' + depth_64beams_gth + img + '.npy')]

    assert len(depth_gth_train) == len(train_idx), "Check dense depth gth train dataset prepare!"
    assert len(left_train) == len(train_idx), "Check left train dataset prepare!"
    assert len(right_train) == len(train_idx), "Check right train dataset prepare!"
    assert len(left_lidar_train) == len(train_idx), "Check left 4 beams train dataset prepare!"
    assert len(right_lidar_train) == len(train_idx), "Check right 4 beams train dataset prepare!"
    assert len(calib_train) == len(train_idx), "Check calib train dataset prepare!"
    # val ##############################################################################################################
    left_val = [filepath + '/' + left_fold + img + '.png' for img in val_idx if
                  isfile(filepath + '/' + left_fold + img + '.png')]
    right_val = [filepath + '/' + right_fold + img + '.png' for img in val_idx if
                   isfile(filepath + '/' + right_fold + img + '.png')]
    left_lidar_val = [filepath + '/' + left_lidar_fold + img + '.png' for img in val_idx if
                         isfile(filepath + '/' + left_lidar_fold + img + '.png')]
    right_lidar_val = [filepath + '/' + right_lidar_fold + img + '.png' for img in val_idx if
                          isfile(filepath + '/' + right_lidar_fold + img + '.png')]
    calib_val = [filepath + '/' + calib + img + '.txt' for img in val_idx if
                 isfile(filepath + '/' + calib + img + '.txt')]
    # gth
    if gth=='dense':
        depth_gth_val = [filepath + '/' + depth_dense_gth + 'val/' + img + '.png' for img in val_idx if
                           isfile(filepath + '/' + depth_dense_gth + 'val/' + img + '.png')]
    elif gth=='sparse':
        depth_gth_val = [filepath + '/' + depth_64beams_gth + img + '.npy' for img in val_idx if
                              isfile(filepath + '/' + depth_64beams_gth + img + '.npy')]
    assert len(depth_gth_val) == len(val_idx), "Check sparse depth gth val dataset prepare!"
    assert len(left_val)==len(val_idx), "Check left val dataset prepare!"
    assert len(right_val)==len(val_idx), "Check right val dataset prepare!"
    assert len(left_lidar_val)==len(val_idx), "Check left 4 beams val dataset prepare!"
    assert len(right_lidar_val)==len(val_idx), "Check right 4 beams val dataset prepare!"
    assert len(calib_val)==len(val_idx), "Check calib train dataset prepare!"

    return [left_train, right_train, left_lidar_train, right_lidar_train, depth_gth_train, calib_train], \
           [left_val, right_val, left_lidar_val, right_lidar_val, depth_gth_val, calib_val]


def dataloader_fdn_foreground(filepath, train_file, val_file, gth='dense'):
    """

    :param filepath:
    :param train_file:
    :param val_file:
    :return:
    """
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    left_lidar_fold = 'depth_map_lidar_left/'
    right_lidar_fold = 'depth_map_lidar_right/'
    calib = 'calib/'
    # gth
    depth_64beams_gth = 'depth_map_64beams_left/'  # sparse gth
    dense_depth_gth = 'dense_depth_gth/'

    mypath_train = filepath + '/' + depth_dense_gth + 'train'
    mypath_val = filepath + '/' + dense_depth_gth + 'val'
    # depth_L = 'depth_map_64beams_left/'

    from utils.kitti_object import kitti_object
    import numpy as np

    dataset = kitti_object('/home/zelt/PHD_1st_project/src/dataset/kitti_dataset')

    if gth == 'dense':  # 11 frames
        # train_idx: 3622, val_idx: 3696 take from ForeSEe gth, missing some file
        train_idx = [f[:-4] for f in listdir(mypath_train) if isfile(join(mypath_train, f))]
        val_idx = [f[:-4] for f in listdir(mypath_val) if isfile(join(mypath_val, f))]
    elif gth == 'sparse':  # 64 beams
        # train_idx_0: 3712, val_idx_0:3769 from original file
        with open(train_file, 'r') as f:
            train_idx = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        with open(val_file, 'r') as f:
            val_idx = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

    # train ############################################################################################################
    left_train = [filepath + '/' + left_fold + img + '.png' for img in train_idx if
                  isfile(filepath + '/' + left_fold + img + '.png')]
    right_train = [filepath + '/' + right_fold + img + '.png' for img in train_idx if
                   isfile(filepath + '/' + right_fold + img + '.png')]
    left_lidar_train = [filepath + '/' + left_lidar_fold + img + '.png' for img in train_idx if
                         isfile(filepath + '/' + left_lidar_fold + img + '.png')]
    right_lidar_train = [filepath + '/' + right_lidar_fold + img + '.png' for img in train_idx if
                          isfile(filepath + '/' + right_lidar_fold + img + '.png')]
    calib_train = [filepath + '/' + calib + img + '.txt' for img in train_idx if
                   isfile(filepath + '/' + calib + img + '.txt')]


    foreground_mask_train = get_foreground_mask(train_idx, left_train, dataset)


    # gth
    if gth == 'dense':  # 11 frames
        depth_gth_train = [filepath + '/' + dense_depth_gth + 'train/' + img + '.png' for img in train_idx if
                           isfile(filepath + '/' + dense_depth_gth + 'train/' + img + '.png')]
    elif gth == 'sparse':  # 64 beams
        # depth_train = [filepath + '/' + depth_L + img + '.npy' for img in train_idx if isfile(filepath + '/' + depth_L + img + '.npy')] # depth 64
        depth_gth_train = [filepath + '/' + depth_64beams_gth + img + '.npy' for img in train_idx if
                           isfile(filepath + '/' + depth_64beams_gth + img + '.npy')]
    assert len(depth_gth_train) == len(train_idx), "Check dense depth gth train dataset prepare!"
    assert len(left_train) == len(train_idx), "Check left train dataset prepare!"
    assert len(right_train) == len(train_idx), "Check right train dataset prepare!"
    assert len(left_lidar_train) == len(train_idx), "Check left 4 beams train dataset prepare!"
    assert len(right_lidar_train) == len(train_idx), "Check right 4 beams train dataset prepare!"
    assert len(calib_train) == len(train_idx), "Check calib train dataset prepare!"
    # val ##############################################################################################################
    left_val = [filepath + '/' + left_fold + img + '.png' for img in val_idx if
                isfile(filepath + '/' + left_fold + img + '.png')]
    right_val = [filepath + '/' + right_fold + img + '.png' for img in val_idx if
                 isfile(filepath + '/' + right_fold + img + '.png')]
    left_lidar_val = [filepath + '/' + left_lidar_fold + img + '.png' for img in val_idx if
                       isfile(filepath + '/' + left_lidar_fold + img + '.png')]
    right_lidar_val = [filepath + '/' + right_lidar_fold + img + '.png' for img in val_idx if
                        isfile(filepath + '/' + right_lidar_fold + img + '.png')]
    calib_val = [filepath + '/' + calib + img + '.txt' for img in val_idx if
                 isfile(filepath + '/' + calib + img + '.txt')]

    # foreground_mask_val = get_foreground_mask(val_idx, left_val, dataset)
    foreground_mask_val = [filepath + '/' + calib + img + '.txt' for img in val_idx if
                 isfile(filepath + '/' + calib + img + '.txt')]
    # gth
    if gth == 'dense':
        depth_gth_val = [filepath + '/' + dense_depth_gth + 'val/' + img + '.png' for img in val_idx if
                         isfile(filepath + '/' + dense_depth_gth + 'val/' + img + '.png')]
    elif gth == 'sparse':
        depth_gth_val = [filepath + '/' + depth_64beams_gth + img + '.npy' for img in val_idx if
                         isfile(filepath + '/' + depth_64beams_gth + img + '.npy')]
    assert len(depth_gth_val) == len(val_idx), "Check sparse depth gth val dataset prepare!"
    assert len(left_val) == len(val_idx), "Check left val dataset prepare!"
    assert len(right_val) == len(val_idx), "Check right val dataset prepare!"
    assert len(left_lidar_val) == len(val_idx), "Check left 4 beams val dataset prepare!"
    assert len(right_lidar_val) == len(val_idx), "Check right 4 beams val dataset prepare!"
    assert len(calib_val) == len(val_idx), "Check calib train dataset prepare!"

    return [left_train, right_train, left_lidar_train, right_lidar_train, foreground_mask_train, depth_gth_train,
            calib_train], \
           [left_val, right_val, left_lidar_val, right_lidar_val, foreground_mask_val, depth_gth_val, calib_val]

def default_loader(path):
    from PIL import Image
    return Image.open(path).convert('RGB')

def get_foreground_mask(list_idx, left_image_path, dataset):
    import numpy as np
    from tqdm import tqdm
    foreground_mask_list = []
    for k, i in tqdm(enumerate(list_idx)):
        # print(k)
        objects = dataset.get_label_objects(int(i))
        w, h = default_loader(left_image_path[k]).size
        foreground_mask = np.zeros([h, w])
        for obj in objects:
            foreground_mask[int(obj.ymin):int(obj.ymax), int(obj.xmin):int(obj.xmax)]= 1

        foreground_mask_list.append(foreground_mask)

    return foreground_mask_list

