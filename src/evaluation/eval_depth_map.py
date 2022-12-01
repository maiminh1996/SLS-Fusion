import numpy as np
from os import listdir
from os.path import isfile, join
import configargparse
import os
from tqdm import tqdm
from eval_utils import *
from utils import setup_logger

def parse():
    parser = configargparse.ArgParser(description='evaluation depth map prediction')
    # parser.add('-c', '--config',
    #             default='',  # config in this file will overwrite all argument
    #             is_config_file=True, help='Config file will overwrite all args')
    parser.add_argument('--path_gth', type=str,
                        default='/media/zelt/Données/KITTI/3D_OD/training/production/velodyne_projected_64beams_left', # sparse depth
                        # default='/home/zelt/PHD_1st_project/dataset/Kitti/training/dense_depth_gth/val', # dense depth
                        help='')
    parser.add_argument('--path_pred', type=str,
                        default='/media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/04-02-2021_11-37-37/depth_maps/trainval',
                        # default=
                        help='')
    parser.add_argument('--split_val', type=str,
                        default='/media/zelt/Données/KITTI/3D_OD/training/ImageSets/val.txt',
                        help='')
    # /media/zelt/Données/KITTI/3D_OD/training/ImageSets
    args = parser.parse_args()
    return args

class MetricDepth:
    def __init__(self, path_gth, path_pred):
        self.path_gth = path_gth
        self.path_pred = path_pred
        # self.depth_gth, self.depth_pred, self.mask, self.num_valid_pixel = self.setup(self.path_gth, self.path_pred)
    
    def setup(self, min_depth=1, max_depth=80):
        # depth_gth = self.load_depth_npy(self.path_gth)
        depth_gth = self.load_depth(self.path_gth)
        # depth_pred = self.load_depth(self.path_pred)
        depth_pred = self.load_depth_npy(self.path_pred)
        mask = (depth_gth >= min_depth) * (depth_gth <= max_depth) * (depth_pred >= min_depth) * (depth_pred <= max_depth)
        num_valid_pixel = depth_gth[mask].shape[0]
        return depth_gth, depth_pred, mask, num_valid_pixel
        
    def load_depth(self, path):
        """
        
        """
        import skimage
        import skimage.io
        depth_img = skimage.io.imread(path)
        depth = depth_img.astype(np.float32) / 256.0
        return depth
    
    def load_depth_npy(self, path):
        depth = np.load(path).astype(np.float32)
        return depth
    
    def eval_depth_estimation(self, depth_valid_gth, depth_valid_pred):
        silog = scale_invariant_error_log(depth_valid_gth, depth_valid_pred)
        sq_error_rel = square_relative_error(depth_valid_gth, depth_valid_pred)
        abs_error_rel = absolute_relative_error(depth_valid_gth, depth_valid_pred)
        irmse = inverse_root_mean_spuare_error(depth_valid_gth, depth_valid_pred)
        return silog, sq_error_rel, abs_error_rel, irmse
        
    def eval_depth_completion(self, depth_valid_gth, depth_valid_pred):
        irmse = inverse_root_mean_spuare_error(depth_valid_gth, depth_valid_pred)
        imae = inverse_mean_absolute_error(depth_valid_gth, depth_valid_pred)
        rmse = root_mean_spuare_error(depth_valid_gth, depth_valid_pred)
        mae = mean_absolute_error(depth_valid_gth, depth_valid_pred)
        return irmse, imae, rmse, mae

    
        

if __name__ == '__main__':
    args = parse()

    # dense depth ground truth
    # args.path_gth = '/home/zelt/PHD_1st_project/dataset/Kitti/training/dense_depth_gth/val'
    args.path_pred = '/media/zelt/Données/KITTI/test_algo/save_weight_FDNet/results/fdn_kitti_train_set_after_sceneflow/depth_maps/val/'
    # args.path_pred = '/media/zelt/Données/KITTI/test_algo/save_weight_SDNet/results/sdn_kitti_train_set_paper/depth_maps/val'

    file_log = 'eval_depth.log'
    log = setup_logger(os.path.join(file_log))
    log.info("************************ New evaluation ************************")
    for key, value in sorted(vars(args).items()):
        log.info('args: ' + str(key) + ': ' + str(value))  # log all args infos
    with open(args.split_val, 'r') as f:
        val_idx = [x.strip() for x in f.readlines() if len(x.strip())>0]
    if args.path_gth == '/home/zelt/PHD_1st_project/dataset/Kitti/training/dense_depth_gth/val':
        val_idx = [f[:-4] for f in listdir(args.path_gth) if isfile(join(args.path_gth, f))]

    iRMSE = []
    iMAE = []
    RMSE = []
    MAE = [] 
    for k, i in enumerate(tqdm(val_idx)):
        path_gth_i = args.path_gth + '/' + i + '.png'
        path_pred_i = args.path_pred + '/' + i + '.npy'
        MetricDC = MetricDepth(path_gth_i, path_pred_i)
        depth_gth, depth_pred, mask, num_valid_pixel = MetricDC.setup()
        depth_valid_gth = depth_gth[mask]
        depth_valid_pred = depth_pred[mask]
        irmse, imae, rmse, mae = MetricDC.eval_depth_completion(depth_valid_gth, depth_valid_pred)
        iRMSE.append(irmse)
        iMAE.append(imae)
        RMSE.append(rmse)
        MAE.append(mae)
        # if k%10==0:
        #     log.debug("idx: {}, iRMSE: {}, iMAE: {}, RMSE: {}, MAE {}".format(i, irmse, imae, rmse, mae))
    iRMSE_moy = sum(iRMSE)/len(iRMSE)
    iMAE_moy = sum(iMAE)/len(iMAE)
    RMSE_moy = sum(RMSE)/len(RMSE)
    MAE_moy = sum(MAE)/len(MAE)
    
    log.info("Num sample: {}, iRMSE moy: {}, iMAE moy: {}, RMSE moy: {}, MAE moy: {}".format(len(iRMSE), iRMSE_moy, iMAE_moy, RMSE_moy, MAE_moy))



