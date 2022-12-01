import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import skimage
import skimage.io

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')

def sparse_loader(path):
    """
    sparse lidar
    :param path:
    :return:
    """
    img = skimage.io.imread(path)
    img = img.astype(np.float32) / 256.0
    mask = np.where(img > 0.0, 1.0, 0.0)
    mask = np.reshape(mask, [img.shape[0], img.shape[1], 1])
    mask = mask.astype(np.float32)
    img = np.reshape(img, [img.shape[0], img.shape[1], 1])
    return img, mask


def disparity_loader(path):
    return np.load(path).astype(np.float32)


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

def kitti2015_disparity_loader(filepath, calib):
    disp = np.array(Image.open(filepath))/256.
    depth = np.zeros_like(disp)
    mask = disp > 0
    depth[mask] = calib / disp[mask]
    return depth


def dynamic_baseline(calib_info):
    P3 =np.reshape(calib_info['P3'], [3,4])
    P =np.reshape(calib_info['P2'], [3,4])
    baseline = P3[0,3]/(-P3[0,0]) - P[0,3]/(-P[0,0])
    return baseline


def dense_loader(path):
    import skimage
    import skimage.io
    img = skimage.io.imread(path)
    depth = img.astype(np.float32) / 256.0
    depth = np.reshape(depth, [img.shape[0], img.shape[1]])
    return depth

class myImageFloder_SLSFusion(data.Dataset):
    def __init__(self, data, training, dynamic_bs=False, loader=default_loader, sloader=sparse_loader, dloader=dense_loader, dploader=disparity_loader, gth='dense', separate=False):
        """
        :param data: left RGB, right RGB, left 4 beams, right 4 beams, left dense depth map 11 frames, calib path file
        :param training: flag training or eval mode
        :param dynamic_bs:
        :param loader:
        :param sloader:
        :param dploader:
        :param dloader:
        :param gth: dense: 11 frames, sparse: lidar 64 beams
        """
        left, right, left_4beams, right_4beams, left_dense_depth, calib = data
        self.left = left
        self.right = right
        self.left_4beams = left_4beams
        self.right_4beams = right_4beams
        self.dense_depth = left_dense_depth
        self.calib = calib

        self.training = training
        self.dynamic_bs = dynamic_bs
        self.gth = gth
        self.separate = separate

        self.loader = loader # for loader RGB
        self.sloader = sloader # for loader sparse lidar
        self.dense_loader = dloader # for dense lidar gth 11 frames
        self.dploader = dploader


        # stat for kitti OD
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # TODO
                                         std=[0.229, 0.224, 0.225])

        normalize1 = transforms.Normalize(mean=[0.01],
                                         std=[0.229])
        # transform for image left, right
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch
            # .FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            normalize
        ])
        # transform for lidar 4 beams
        self.transform_lidar = transforms.Compose([
            transforms.ToTensor(),
            # normalize1
        ])
    
    def setzeros(self, x):
        return torch.zeros_like(x)

    def __getitem__(self, index):
        """
        Load each batch data. this is called in training loop
        :param index:
        :return:
        """
        left = self.left[index]
        right = self.right[index]
        left_4beams = self.left_4beams[index]
        right_4beams = self.right_4beams[index]
        dense_depth = self.dense_depth[index]
        calib = self.calib[index]

        calib_info = read_calib_file(calib)

        if self.dynamic_bs: # TODO to understand
            dynamic_basel = dynamic_baseline(calib_info)
            calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * dynamic_basel

        else:
            calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * 0.54

        left_img = self.loader(left) # (375, 1242, 3)
        right_img = self.loader(right)
        

        left_4beams_img, left_mask = self.sloader(left_4beams)
        right_4beams_img, right_mask = self.sloader(right_4beams)

        dense_dataL = self.dense_loader(dense_depth)

        shape_img = np.shape(left_img)[:2]
        shape_4beams = np.shape(left_4beams_img)[:2]
        shape_gth = np.shape(dense_dataL)

        assert shape_img==shape_4beams, 'need to check left img {}: {}, left 4 beams {}: {}'.format(shape_img, left, shape_4beams, left_4beams)
        assert shape_img==shape_gth, 'need to check left img {}: {}, depth gth {}: {}'.format(shape_img, left, shape_gth, dense_depth)

        if self.training:
            
            w, h = left_img.size
            th, tw = 256, 512
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img_crop = left_img.crop((x1, y1, x1 + tw, y1 + th)) # TODO to check
            right_img_crop = right_img.crop((x1, y1, x1 + tw, y1 + th))
            left_4beams_crop = left_4beams_img[y1:y1 + th, x1:x1 + tw, :]  # [256, 512, 1]
            right_4beams_crop = right_4beams_img[y1:y1 + th, x1:x1 + tw, :]  # [256, 512, 1]
            left_mask_crop = left_mask[y1:y1 + th, x1:x1 + tw, :]
            right_mask_crop = right_mask[y1:y1 + th, x1:x1 + tw, :]
            dense_dataL = dense_dataL[y1:y1 + th, x1:x1 + tw]
            
            left_img_crop = self.transform(left_img_crop) # normal and totensor
            right_img_crop = self.transform(right_img_crop)
            left_4beams_crop = self.transform_lidar(left_4beams_crop).float().div(100) # 
            right_4beams_crop = self.transform_lidar(right_4beams_crop).float().div(100)

            left_mask_crop = torch.tensor(left_mask_crop).permute(2, 0, 1)
            right_mask_crop = torch.tensor(right_mask_crop).permute(2, 0, 1)
            
        else:
            w, h = left_img.size

            left_img_crop = left_img.crop((w - 1200, h - 352, w, h))
            right_img_crop = right_img.crop((w - 1200, h - 352, w, h))
            left_4beams_crop = left_4beams_img[h - 352:h, w - 1200:w, :]
            right_4beams_crop = right_4beams_img[h - 352:h, w - 1200:w, :]
            left_mask_crop = left_mask[h - 352:h, w - 1200:w, :]
            right_mask_crop = right_mask[h - 352:h, w - 1200:w, :]
            dense_dataL = dense_dataL[h - 352:h, w - 1200:w]

            left_img_crop = self.transform(left_img_crop)
            right_img_crop = self.transform(right_img_crop)
            
            left_4beams_crop = self.transform_lidar(left_4beams_crop).float().div(100)
            right_4beams_crop = self.transform_lidar(right_4beams_crop).float().div(100)
            left_mask_crop = torch.tensor(left_mask_crop).permute(2, 0, 1)
            right_mask_crop = torch.tensor(right_mask_crop).permute(2, 0, 1)

        left_img_crop = left_img_crop.float()
        right_img_crop = right_img_crop.float()
        left_4beams_crop = left_4beams_crop.float()
        right_4beams_crop = right_4beams_crop.float()
        left_mask_crop = left_mask_crop.float()
        right_mask_crop = right_mask_crop.float()
        dense_dataL = torch.from_numpy(dense_dataL).float() # [0: 80]

        if self.separate == 'lidar':
            left_img_crop = self.setzeros(left_img_crop) # (375, 1242, 3)
            right_img_crop = self.setzeros(right_img_crop)
        elif self.separate == 'cam':
            left_4beams_crop = self.setzeros(left_4beams_crop)
            right_4beams_crop = self.setzeros(right_4beams_crop)
            left_mask_crop = self.setzeros(left_mask_crop)
            right_mask_crop = self.setzeros(right_mask_crop)

        return left_img_crop, right_img_crop, left_4beams_crop, right_4beams_crop, left_mask_crop, right_mask_crop, dense_dataL, calib.item()

    def __len__(self):
        return len(self.left)


class myImageFloder_fdn_foreground(data.Dataset):
    def __init__(self, data, training, dynamic_bs=False, loader=default_loader, sloader=sparse_loader, dloader=dense_loader, dploader=disparity_loader, gth='dense', separate=False):
        """

        :param data: left RGB, right RGB, left 4 beams, right 4 beams, left dense depth map 11 frames, calib path file
        :param training: flag training or eval mode
        :param dynamic_bs: TODO to understand
        :param loader:
        :param sloader:
        :param dploader:
        :param dloader:
        """
        left, right, left_4beams, right_4beams, foreground_mask, left_dense_depth, calib = data

        self.left = left
        self.right = right
        self.left_4beams = left_4beams
        self.right_4beams = right_4beams
        self.foreground_mask = foreground_mask
        self.dense_depth = left_dense_depth
        self.calib = calib

        self.training = training
        self.dynamic_bs = dynamic_bs
        self.gth = gth

        self.loader = loader # for loader RGB
        self.sloader = sloader # for loader sparse lidar
        self.dense_loader = dloader # for dense lidar gth 11 frames
        self.dploader = dploader


        # stat for kitti OD
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        normalize1 = transforms.Normalize(mean=[0.01],
                                         std=[0.229])
        # transform for image left, right
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch
            # .FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            normalize
        ])
        # transform for lidar 4 beams
        self.transform_lidar = transforms.Compose([
            transforms.ToTensor(),
            # normalize1
        ])

    def __getitem__(self, index):
        """
        Load each batch data. this is called in training loop
        :param index:
        :return:
        """
        left = self.left[index]
        right = self.right[index]
        left_4beams = self.left_4beams[index]
        right_4beams = self.right_4beams[index]
        foreground_mask = self.foreground_mask[index]
        dense_depth = self.dense_depth[index]
        calib = self.calib[index]

        calib_info = read_calib_file(calib)

        if self.dynamic_bs: # TODO to understand
            calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * dynamic_baseline(calib_info)

        else:
            calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * 0.54

        ### Loader ###
        left_img = self.loader(left)

        right_img = self.loader(right)
        left_4beams_img, left_mask = self.sloader(left_4beams)
        right_4beams_img, right_mask = self.sloader(right_4beams)

        if self.gth=='dense':
            dense_dataL = self.dense_loader(dense_depth)
        elif self.gth=='sparse':
            dense_dataL = self.dploader(dense_depth)

        shape_img = np.shape(left_img)[:2]
        shape_4beams = np.shape(left_4beams_img)[:2]
        shape_gth = np.shape(dense_dataL)

        assert shape_img==shape_4beams, 'need to check left img {}: {}, left 4 beams {}: {}'.format(shape_img, left, shape_4beams, left_4beams)
        assert shape_img==shape_gth, 'need to check left img {}: {}, depth gth {}: {}'.format(shape_img, left,
                                                                                                      shape_gth,
                                                                                                      dense_depth)

        if self.training:
            w, h = left_img.size

            th, tw = 256, 512
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img_crop = left_img.crop((x1, y1, x1 + tw, y1 + th)) # TODO to check
            right_img_crop = right_img.crop((x1, y1, x1 + tw, y1 + th))
            left_4beams_crop = left_4beams_img[y1:y1 + th, x1:x1 + tw, :]  # [256, 512, 1]
            right_4beams_crop = right_4beams_img[y1:y1 + th, x1:x1 + tw, :]  # [256, 512, 1]
            left_mask_crop = left_mask[y1:y1 + th, x1:x1 + tw, :]
            right_mask_crop = right_mask[y1:y1 + th, x1:x1 + tw, :]
            foreground_mask_crop = foreground_mask[y1:y1 + th, x1:x1 + tw]
            dense_dataL = dense_dataL[y1:y1 + th, x1:x1 + tw]
            
            left_img_crop = self.transform(left_img_crop) # normal and totensor
            right_img_crop = self.transform(right_img_crop)
            left_4beams_crop = self.transform_lidar(left_4beams_crop).float().div(100) # TODO how to normalize
            right_4beams_crop = self.transform_lidar(right_4beams_crop).float().div(100)

            left_mask_crop = torch.tensor(left_mask_crop).permute(2, 0, 1)
            right_mask_crop = torch.tensor(right_mask_crop).permute(2, 0, 1)

        else:
            w, h = left_img.size

            left_img_crop = left_img.crop((w - 1200, h - 352, w, h))
            right_img_crop = right_img.crop((w - 1200, h - 352, w, h))
            left_4beams_crop = left_4beams_img[h - 352:h, w - 1200:w, :]
            right_4beams_crop = right_4beams_img[h - 352:h, w - 1200:w, :]
            left_mask_crop = left_mask[h - 352:h, w - 1200:w, :]
            right_mask_crop = right_mask[h - 352:h, w - 1200:w, :]
            # foreground_mask_crop = foreground_mask[h - 352:h, w - 1200:w]
            foreground_mask_crop = 0
            dense_dataL = dense_dataL[h - 352:h, w - 1200:w]

            left_img_crop = self.transform(left_img_crop)
            right_img_crop = self.transform(right_img_crop)
            left_4beams_crop = self.transform_lidar(left_4beams_crop).float().div(100)
            right_4beams_crop = self.transform_lidar(right_4beams_crop).float().div(100)
            left_mask_crop = torch.tensor(left_mask_crop).permute(2, 0, 1)
            right_mask_crop = torch.tensor(right_mask_crop).permute(2, 0, 1)
        
        left_img_crop = left_img_crop.float()
        right_img_crop = right_img_crop.float()
        left_4beams_crop = left_4beams_crop.float()
        right_4beams_crop = right_4beams_crop.float()
        left_mask_crop = left_mask_crop.float()
        right_mask_crop = right_mask_crop.float()
        # foreground_mask_crop = foreground_mask_crop.float()
        dense_dataL = torch.from_numpy(dense_dataL).float() # [0: 80]
        
        return left_img_crop, right_img_crop, left_4beams_crop, right_4beams_crop, left_mask_crop, right_mask_crop, foreground_mask_crop, dense_dataL, calib.item()
    
    def __len__(self):
        return len(self.left)


