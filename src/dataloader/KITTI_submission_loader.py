import os
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np
import torch
from os import listdir
from os.path import isfile, join
import skimage
import skimage.io

class SubmiteDataset(object):
    def __init__(self, filepath, split, dynamic_bs=False, gth='dense', separate=False):
        self.dynamic_bs = dynamic_bs
        self.gth = gth
        self.separate = separate
        
        # hardcode: change here
        left_folds = ['image_2/']
        right_folds = ['image_3/']
        left_lidar_folds = ['velodyne/']
        right_lidar_folds = ['velodyne/']
        # left_lidar_folds = ['depth_map_4beams_left/']
        # right_lidar_folds = ['depth_map_4beams_right/']
        
        calib_fold = 'calib/'

        dense_depth_gth = 'dense_depth_gth/'
        depth_64beams_gth = 'production/velodyne_projected_64beams_left/'
        mypath_train = filepath + '/' + dense_depth_gth + 'train'
        mypath_val = filepath + '/' + dense_depth_gth + 'val'
    
        if self.gth == 'dense':  # 11 frames
            # train_idx: 3622, val_idx: 3696 take from ForeSEe gth, missing some file
            val_idx = [f[:-4] for f in listdir(mypath_val) if isfile(join(mypath_val, f))]
        elif self.gth == 'sparse':  # 64 beams
            # train_idx_0: 3712, val_idx_0:3769 from original file
            with open(split, 'r') as f:
                val_idx = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        
        with open(split, 'r') as f:
            val_idx = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        image = sorted(val_idx)
        val_idx= sorted(val_idx)

        self.left_test = [filepath + '/' + left_fold + img + '.png' for img in val_idx for left_fold in left_folds if
                    isfile(filepath + '/' + left_fold + img + '.png')]
        self.right_test = [filepath + '/' + right_fold + img + '.png' for img in val_idx for right_fold in right_folds if 
                     isfile(filepath + '/' + right_fold + img + '.png')]
        self.left_4beams_test = [filepath + '/' + left_lidar_fold + img + '.png' for img in val_idx for left_lidar_fold in left_lidar_folds if
                           isfile(filepath + '/' + left_lidar_fold + img + '.png')]
        self.right_4beams_test = [filepath + '/' + right_lidar_fold + img + '.png' for img in val_idx for right_lidar_fold in right_lidar_folds if
                            isfile(filepath + '/' + right_lidar_fold + img + '.png')]

        self.calib_test = [filepath + '/' + calib_fold + img + '.txt' for img in val_idx if
                     isfile(filepath + '/' + calib_fold + img + '.txt')]
        

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.transform_lidar = transforms.Compose([
            transforms.ToTensor(),
        ])
    def setzeros(self, x):
        return torch.zeros_like(x)

    def __getitem__(self, item):
        left_img = self.left_test[item]
        right_img = self.right_test[item]
        left_4beams_img = self.left_4beams_test[item]
        right_4beams_img = self.right_4beams_test[item]
        calib_info = read_calib_file(self.calib_test[item])
        if self.dynamic_bs:
            calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * dynamic_baseline(calib_info)
        else:
            calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * 0.54

        imgL = Image.open(left_img).convert('RGB')
        imgR = Image.open(right_img).convert('RGB')
        left_4beams, left_mask = sparse_loader(left_4beams_img)
        right_4beams, right_mask = sparse_loader(right_4beams_img)

        imgL = self.trans(imgL)[None, :, :, :]
        imgR = self.trans(imgR)[None, :, :, :]
        left_4beams = self.transform_lidar(left_4beams).div(100)[None, :, :, :]  # normal and totensor
        right_4beams = self.transform_lidar(right_4beams).div(100)[None, :, :, :]
        left_mask = torch.tensor(left_mask).permute(2, 0, 1)[None, :, :, :]
        right_mask = torch.tensor(right_mask).permute(2, 0, 1)[None, :, :, :]

        # pad to (384, 1248)
        B, C, H, W = imgL.shape
        top_pad = 384 - H
        right_pad = 1248 - W
        imgL = F.pad(imgL, (0, right_pad, top_pad, 0), "constant", 0)
        imgR = F.pad(imgR, (0, right_pad, top_pad, 0), "constant", 0)
        left_4beams = F.pad(left_4beams, (0, right_pad, top_pad, 0), "constant", 0)
        right_4beams = F.pad(right_4beams, (0, right_pad, top_pad, 0), "constant", 0)
        left_mask = F.pad(left_mask, (0, right_pad, top_pad, 0), "constant", 0)
        right_mask = F.pad(right_mask, (0, right_pad, top_pad, 0), "constant", 0)

        filename = self.left_test[item].split('/')[-2] + '_' + self.left_test[item].split('/')[-1][:-4]

        if self.separate == 'lidar':
            imgL = self.setzeros(imgL) # (375, 1242, 3)
            imgR = self.setzeros(imgR)
        elif self.separate == 'cam':
            left_4beams = self.setzeros(left_4beams)
            right_4beams = self.setzeros(right_4beams)
            left_mask = self.setzeros(left_mask)
            right_mask = self.setzeros(right_mask)
        
        return imgL[0].float(), imgR[0].float(), left_4beams[0].float(), right_4beams[0].float(), left_mask[0].float(), right_mask[0].float(), calib.item(), H, W, filename


    def __len__(self):
        return len(self.left_test)

def sparse_loader(path):
    """
    sparse lidar
    :param path:
    :return:
    """
    img = skimage.io.imread(path)
    img = img * 1.0 / 256.0

    mask = np.where(img > 0.0, 1.0, 0.0)
    mask = np.reshape(mask, [img.shape[0], img.shape[1], 1])
    mask = mask.astype(np.float32)
    img = np.reshape(img, [img.shape[0], img.shape[1], 1])
    return img, mask

def read_calib_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def dataloader(filepath, split):
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    calib_fold = 'calib/'
    with open(split, 'r') as f:
        image = [x.strip() for x in f.readlines() if len(x.strip())>0]

    left_test = [filepath + left_fold + img + '.png' for img in image]
    right_test = [filepath + right_fold + img + '.png' for img in image]
    calib_test = [filepath + calib_fold + img + '.txt' for img in image]

    return left_test, right_test, calib_test

def dynamic_baseline(calib_info):
    P3 =np.reshape(calib_info['P3'], [3,4])
    P =np.reshape(calib_info['P2'], [3,4])
    baseline = P3[0,3]/(-P3[0,0]) - P[0,3]/(-P[0,0])
    return baseline