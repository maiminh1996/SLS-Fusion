from os import listdir
from os.path import isfile, join

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    from PIL import Image
    return Image.open(path).convert('RGB')

def dataloader_SLSFusion(filepath, train_file, val_file, gth='dense'):
    """
    get data_path list
    :param filepath: path to kitti dataset
    :param train_file: train idx
    :param val_file: val idx
    :return:
    """
    left_folds = ['image_2/']
    right_folds = ['image_3/']
    left_lidar_folds = ['depth_map_4beams_left/']
    right_lidar_folds = ['depth_map_4beams_right/']

    calib = 'calib/'
    if gth=='dense': # TODO make dense depth folder: train and val in the 1 folder
        # train_idx: 3622, val_idx: 3696 take from ForeSEe gth, missing some file
        depth_dense_gth = 'dense_depth_gth/' # gth 11 accum LiDAR 64 beams
        mypath_train = filepath + '/' + depth_dense_gth + 'train'
        mypath_val = filepath + '/' + depth_dense_gth + 'val'
        train_idx = [f[:-4] for f in listdir(mypath_train) if isfile(join(mypath_train, f))]
        val_idx = [f[:-4] for f in listdir(mypath_val) if isfile(join(mypath_val, f))]
    elif gth=='sparse':
        # train_idx_0: 3712, val_idx_0:3769 from original file
        # depth_64beams_gth = 'depth_map_64beams_left/' # gth sparse gth LiDAR 64 beams
        depth_64beams_gth = 'production/velodyne_projected_64beams_left/'
        with open(train_file, 'r') as f:
            train_idx = [x.strip() for x in f.readlines() if len(x.strip())>0]
        with open(val_file, 'r') as f:
            val_idx = [x.strip() for x in f.readlines() if len(x.strip())>0]
    
    left_train = [filepath + '/' + left_fold + img + '.png' for img in train_idx for left_fold in left_folds if
                    isfile(filepath + '/' + left_fold + img + '.png')]
    right_train = [filepath + '/' + right_fold + img + '.png' for img in train_idx for right_fold in right_folds if
                    isfile(filepath + '/' + right_fold + img + '.png')]
    left_lidar_train = [filepath + '/' + left_lidar_fold + img + '.png' for img in train_idx for left_lidar_fold in left_lidar_folds if
                        isfile(filepath + '/' + left_lidar_fold + img + '.png')]
    right_lidar_train = [filepath + '/' + right_lidar_fold + img + '.png' for img in train_idx for right_lidar_fold in right_lidar_folds if
                        isfile(filepath + '/' + right_lidar_fold + img + '.png')]
    

    calib_train = [filepath + '/' + calib + img + '.txt' for img in train_idx if
                    isfile(filepath + '/' + calib + img + '.txt')]
    # calib_train = [filepath + '/' + calib + img + '.txt' for img in train_idx for i in range(2) if
    #                isfile(filepath + '/' + calib + img + '.txt')]
    if gth == 'dense': # 11 frames
        depth_gth_train = [filepath + '/' + depth_dense_gth + 'train/' + img + '.png' for img in train_idx if
                             isfile(filepath + '/' + depth_dense_gth + 'train/' + img + '.png')]
    elif gth == 'sparse': # 64 beams
        # depth_gth_train = [filepath + '/' + depth_64beams_gth + img + '.npy' for img in train_idx if
        #                       isfile(filepath + '/' + depth_64beams_gth + img + '.npy')]
        depth_gth_train = [filepath + '/' + depth_64beams_gth + img + '.png' for img in train_idx if
                              isfile(filepath + '/' + depth_64beams_gth + img + '.png')]
        # depth_gth_train = [filepath + '/' + depth_64beams_gth + img + '.png' for img in train_idx for i in range(2) if
        #                       isfile(filepath + '/' + depth_64beams_gth + img + '.png')]

    left_val = [filepath + '/' + left_fold + img + '.png' for img in val_idx for left_fold in left_folds if
                isfile(filepath + '/' + left_fold + img + '.png')]
    right_val = [filepath + '/' + right_fold + img + '.png' for img in val_idx for right_fold in right_folds if
                isfile(filepath + '/' + right_fold + img + '.png')]
    left_lidar_val = [filepath + '/' + left_lidar_fold + img + '.png' for img in val_idx for left_lidar_fold in left_lidar_folds if
                        isfile(filepath + '/' + left_lidar_fold + img + '.png')]
    right_lidar_val = [filepath + '/' + right_lidar_fold + img + '.png' for img in val_idx for right_lidar_fold in right_lidar_folds if
                        isfile(filepath + '/' + right_lidar_fold + img + '.png')]
    
    
    calib_val = [filepath + '/' + calib + img + '.txt' for img in val_idx if
                 isfile(filepath + '/' + calib + img + '.txt')]

    if gth=='dense':
        depth_gth_val = [filepath + '/' + depth_dense_gth + 'val/' + img + '.png' for img in val_idx if
                           isfile(filepath + '/' + depth_dense_gth + 'val/' + img + '.png')]
    elif gth=='sparse':
        # depth_gth_val = [filepath + '/' + depth_64beams_gth + img + '.npy' for img in val_idx if
        #                       isfile(filepath + '/' + depth_64beams_gth + img + '.npy')]
        depth_gth_val = [filepath + '/' + depth_64beams_gth + img + '.png' for img in val_idx if
                              isfile(filepath + '/' + depth_64beams_gth + img + '.png')]
        # depth_gth_val = [filepath + '/' + depth_64beams_gth + img + '.png' for img in val_idx for i in range(2) if
        #                       isfile(filepath + '/' + depth_64beams_gth + img + '.png')]

    return [left_train, right_train, left_lidar_train, right_lidar_train, depth_gth_train, calib_train], \
           [left_val, right_val, left_lidar_val, right_lidar_val, depth_gth_val, calib_val]


