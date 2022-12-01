import numpy as np
import torch
import random
import configargparse
import logging
import os

def cuda_random_seed(args):
    # SET random seed, cuda ###
    # check_cuda = not args.no_cuda and torch.cuda.is_available()
    check_cuda = torch.cuda.is_available()
    # torch.backends.cudnn.enabled = False # run out of mem
    torch.backends.cudnn.deterministic = True  # there is non-determinism in some cudnn functions
    torch.backends.cudnn.benchmark = False  # leads to faster runtime in the case input size dont vary
    random.seed(args.seed)  # it's used by some of the random transforms
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if check_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

def parse():
    parser = configargparse.ArgParser(description='SLS-Fusion')
    parser.add('-c', '--config',
                # default='../configs/generate_depth_map.config',
                default='../configs/sls_fusion_kitti_od.config',  # config in this file will overwrite all argument
                # default='../configs/sls_fusion_sceneflow.config',
                is_config_file=True, help='config file to overwrite all args')
    
    parser.add_argument('--seed', type=int,
                        default=12,
                        help='random seed')
    parser.add_argument('--save_path', type=str,
                        default='./results/sls_fusion_kitti_train_set',
                        help='path to save the log, tensorbaord and checkpoint, training loss')
    parser.add_argument('--dataset', 
                        default='kitti', 
                        choices=['sceneflow', 'kitti'],
                        help='train with sceneflow or kitti')
    parser.add_argument('--datapath',
                        default='../../dataset/Kitti/training',
                        help='root folder of the dataset')
    parser.add_argument('--split_train', default='../../dataset/Kitti/ImageSets/train.txt',
                        help='data splitting file for training')
    parser.add_argument('--split_val', 
                        default='../../dataset/Kitti/ImageSets/val.txt',
                        help='data splitting file for validation')
    parser.add_argument('--btrain', type=int, 
                        default=12,
                        help='training batch size')
    parser.add_argument('--bval', type=int, 
                        default=4,
                        help='validation batch size')
    parser.add_argument('--workers', type=int, 
                        default=8,
                        help='number of dataset workers')
    parser.add_argument('--depth_gth',
                        default='sparse',
                        # const='SDNet',
                        choices=['sparse', 'dense'],
                        help='depth 64 beams or depth 11 frames')
    parser.add_argument('--separate',
                        default=False,
                        help='False for cam+lidar, cam for cam only and lidar for lidar only')
    ##### network ###############################################
    parser.add_argument('--data_type', 
                        default='depth', 
                        choices=['disparity', 'depth'],
                        help='the network can predict either disparity or depth')
    parser.add_argument('--arch',
                        default='SLSFusion'
                        # const='SDNet',
                        , choices=['SDNet', 'PSMNet', 'SLSFusion'],
                        help='Model Name, default: SDNet.')
    parser.add_argument('--maxdisp', type=int, 
                        default=192,  # [0,191]
                        help='maxium disparity, the range of the disparity cost volume: [0, maxdisp-1]')
    parser.add_argument('--maxdepth', type=int, 
                        default=80,
                        help='the range of the depth cost volume: [1, maxdepth]')
    parser.add_argument('--down', type=float, 
                        default=2,
                        help='reduce x times resolution when build the depth cost volume')
    ##### learning rate #########################################
    parser.add_argument('--lr', type=float, 
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_stepsize', nargs='+', type=int, 
                        default=[200],
                        help='drop lr in each step')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.1,
                        help='gamma of the learning rate scheduler')
    ##### resume ################################################
    parser.add_argument('--resume', type=int, 
                        default=None,
                        help='path to a checkpoint')
    parser.add_argument('--pretrain', default=None,
                        help='path to pretrained model')
    parser.add_argument('--start_epoch', type=int, 
                        default=0,
                        help='start epoch')
    parser.add_argument('--epochs', type=int, 
                        default=300,
                        help='number of training epochs')
    parser.add_argument('--calib_value', type=float, 
                        default=1017,
                        help='Sceneflow, manually define focal length. (sceneflow does not have configuration)')
    parser.add_argument('--generate_depth_map', 
                        default=False, 
                        action='store_true',
                        help='if true, generate depth maps and save the in save_path/depth_maps/{data_tag}/')
    parser.add_argument('--checkpoint_interval', type=int, 
                        default=1,
                        help='save checkpoint every n epoch.')
    
    parser.add_argument('--dynamic_bs', 
                        default=True, 
                        action='store_true',
                        help='KITTI, If true, dynamically calculate baseline from calibration file. If false, use 0.54')
    
    parser.add_argument('--data_list', default='../../dataset/kitti/ImageSets/trainval.txt',
                        help='generate depth maps for all the data in this list')
    parser.add_argument('--data_tag', default=None,
                        help='the suffix of the depth maps folder')
    
    args = parser.parse_args()
    return args

def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('example')
    handler = logging.StreamHandler()
    handler.setFormatter(file_formatter)
    logger.addHandler(handler)

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) is not '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger