import os
import argparse
import os.path as osp
import numpy as np
from data_utils.kitti_util import Calibration
from tqdm.auto import tqdm
from multiprocessing import Process, Queue, Pool

def filter_height(ptc_velo, threshold=1):
    return ptc_velo[ptc_velo[:, 2] < threshold]

def depth2ptc(depth, calib):
    vu = np.indices(depth.shape).reshape(2, -1).T
    vu[:, 0], vu[:, 1] = vu[:, 1], vu[:, 0].copy()
    uv = vu
    uv_depth = np.column_stack((uv.astype(float), depth.reshape(-1)))
    return calib.project_image_to_rect(uv_depth)

parser = argparse.ArgumentParser(description='gen pointcloud from depthmap')
parser.add_argument('--output_path', type=str)
parser.add_argument('--input_path', type=str)
parser.add_argument('--calib_path', type=str,
                    help='path to calibration files')
parser.add_argument('--split_file', type=str,
                    help='indices of scene to be corrected')
parser.add_argument('--i', type=int, default=None)
parser.add_argument('--threads', type=int, default=4)
args = parser.parse_args()

# def convert_and_save(args, i):
#     depth_map = np.load(osp.join(
#                 args.input_path, "{:06d}.npy".format(i)))
#     calib = Calibration(
#         osp.join(args.calib_path, "{:06d}.txt".format(i)))
#     ptc = filter_height(
#         calib.project_rect_to_velo(depth2ptc(depth_map, calib)))
#     ptc = np.hstack((ptc, np.ones((ptc.shape[0], 1)))).astype(np.float32)
#     with open(osp.join(args.output_path, '{:06d}.bin'.format(i)), 'wb') as f:
#         ptc.tofile(f)

def convert_and_save(args, i, pref_path):
    depth_map = np.load(osp.join(
                args.input_path, pref_path + "{:06d}.npy".format(i)))
    calib = Calibration(
        osp.join(args.calib_path, "{:06d}.txt".format(i)))
    ptc = filter_height(
        calib.project_rect_to_velo(depth2ptc(depth_map, calib)))
    ptc = np.hstack((ptc, np.ones((ptc.shape[0], 1)))).astype(np.float32)
    # with open(osp.join(args.output_path, pref_path + '{:06d}.bin'.format(i)), 'wb') as f:
    #     ptc.tofile(f)
    with open(osp.join(args.output_path, '{:06d}.bin'.format(i)), 'wb') as f:
        ptc.tofile(f)

if __name__ == "__main__":
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/04-02-2021_11-37-37/depth_maps_correct_4beams_hazing_bin --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/04-02-2021_11-37-37/depth_maps_correct_4beams_hazing --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/04-02-2021_11-37-37/depth_maps_correct_4beams_hazing_bin \
    # --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/04-02-2021_11-37-37/depth_maps_correct_4beams_hazing \
    # --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib \
    # --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4

    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/09-02-2021_13-45-30/depth_maps_correct_4beams_bin --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/09-02-2021_13-45-30/depth_maps_correct_4beams --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/09-02-2021_13-45-30/depth_maps_correct_4beams_bin \
    # --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/09-02-2021_13-45-30/depth_maps_correct_4beams \
    # --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib \
    # --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4

    # (8)
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/11-02-2021_11-16-25/depth_maps_correct_4beams_bin --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/11-02-2021_11-16-25/depth_maps_correct_4beams --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/11-02-2021_11-16-25/depth_maps_correct_4beams_bin \
    # --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/11-02-2021_11-16-25/depth_maps_correct_4beams \
    # --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib \
    # --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4

    # (11)
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/28-02-2021_00-54-16/depth_maps_correct_4beams_bin --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/28-02-2021_00-54-16/depth_maps_correct_4beams --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/28-02-2021_00-54-16/depth_maps_correct_4beams_bin \
    # --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/28-02-2021_00-54-16/depth_maps_correct_4beams \
    # --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib \
    # --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4


    # (14)
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/14-03-2021_23-29-24/depth_maps_correct_4beams_bin --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/14-03-2021_23-29-24/depth_maps_correct_4beams --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/14-03-2021_23-29-24/depth_maps_correct_4beams_bin \
    # --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/14-03-2021_23-29-24/depth_maps_correct_4beams \
    # --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib \
    # --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4

    # (14a)
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/18-03-2021_23-26-18/depth_maps_correct_4beams_bin --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/18-03-2021_23-26-18/depth_maps_correct_4beams --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/18-03-2021_23-26-18/depth_maps_correct_4beams_bin \
    # --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/18-03-2021_23-26-18/depth_maps_correct_4beams \
    # --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib \
    # --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4

    # (10)
    
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/18-03-2021_17-15-07/depth_maps_correct_4beams_bin --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/18-03-2021_17-15-07/depth_maps_correct_4beams --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/18-03-2021_17-15-07/depth_maps_correct_4beams_bin \
    # --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/18-03-2021_17-15-07/depth_maps_correct_4beams \
    # --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib \
    # --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4

    # (12)
    
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/21-03-2021_20-25-57/depth_maps_correct_4beams_bin --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/21-03-2021_20-25-57/depth_maps_correct_4beams --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/21-03-2021_20-25-57/depth_maps_correct_4beams_bin \
    # --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/21-03-2021_20-25-57/depth_maps_correct_4beams \
    # --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib \
    # --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4

    # python depthmap2ptc2.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/06-05-2021_07-20-40/depth_maps_correct_4beams_bin --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/06-05-2021_07-20-40/depth_maps_correct_4beams --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4
    # python depthmap2ptc2.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/06-05-2021_07-20-40/depth_maps_correct_4beams_bin \
    # --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/06-05-2021_07-20-40/depth_maps_correct_4beams \
    # --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib \
    # --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4
    # 06-05-2021_07-20-40

    # python depthmap2ptc2.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/kitti_no_image_generate/depth_maps_correct_4beams_bin --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/06-05-2021_07-20-40/depth_maps_correct_4beams --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4
    # python depthmap2ptc2.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/kitti_no_image_generate/depth_maps_correct_4beams_bin \
    # --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/kitti_no_image_generate/depth_maps_correct_4beams \
    # --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib \
    # --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4

    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_STEREO_LiDAR_64beams_new_depthmap/26-09-2022_09-40-17/pseudo_pointcloud --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_STEREO_LiDAR_64beams_new_depthmap/26-09-2022_09-40-17/depth_maps/trainval --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4

    # lidar 64 beams only
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_LiDAR_64beams_depthmap/27-09-2022_11-19-01/pseudo_pointcloud --input_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_LiDAR_64beams_depthmap/27-09-2022_11-19-01/depth_maps_corrected --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4
    
    # stereo + lidar 64 beams
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_STEREO_LiDAR_64beams_new_depthmap/26-09-2022_09-40-17/pseudo_pointcloud --input /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_STEREO_LiDAR_64beams_new_depthmap/26-09-2022_09-40-17/depth_maps_corrected  --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4

    # lidar 8 beams    
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_LiDAR_8beams_depthmap/30-09-2022_14-16-05/pseudo_pointcloud --input /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_LiDAR_8beams_depthmap/30-09-2022_14-16-05/depth_maps_corrected --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4

    # lidar 16 beams    
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_LiDAR_16beams_depthmap/04-10-2022_12-57-04/pseudo_pointcloud --input /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_LiDAR_16beams_depthmap/04-10-2022_12-57-04/depth_maps_corrected --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4

    # stereo + lidar 16 beams    
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_STEREO_LiDAR_16_beams_depthmap/07-10-2022_12-23-43/pseudo_pointcloud --input /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_STEREO_LiDAR_16_beams_depthmap/07-10-2022_12-23-43/depth_maps_corrected --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4

    # stereo + lidar 8 beams    
    # python depthmap2ptc.py --output_path /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_STEREO_LiDAR_8_beams_depthmap/10-10-2022_14-00-02/pseudo_pointcloud --input /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_STEREO_LiDAR_8_beams_depthmap/10-10-2022_14-00-02/depth_maps_corrected --calib_path /media/zelt/Données/KITTI/3D_OD/training/calib --split_file /media/zelt/Données/SLS-Fusion/dataset/kitti/ImageSets/trainval.txt --threads 4


    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.i is not None:
        convert_and_save(args, args.i)
        # i = args.i
        # depth_map = np.load(osp.join(args.input_path, "{:06d}.npy".format(i)))
        # calib = Calibration(osp.join(args.calib_path, "{:06d}.txt".format(i)))
        # ptc = filter_height(calib.project_rect_to_velo(
        #     depth2ptc(depth_map, calib)))
        # # do remember to use float32!!!，
        # ptc = np.hstack((ptc, np.ones((ptc.shape[0], 1)))).astype(np.float32)
        # with open(osp.join(args.output_path, '{:06d}.bin'.format(i)), 'wb') as f:
        #         ptc.tofile(f)
    else:
        with open(args.split_file) as f:
            idx_list = [int(x.strip())
                for x in f.readlines() if len(x.strip()) > 0]
        pbar = tqdm(total=len(idx_list))

        def update(*a):
            pbar.update()
        pool = Pool(args.threads)
        res = []
        # for i in idx_list:
        #     res.append((i, pool.apply_async(convert_and_save, args=(args, i),
        #                                     callback=update)))

        # for pref_path in ["image_21_", "image_left_beta0.039941_"]:
        # pref_path = "image_21_"
        # pref_path = "image_left_beta0.039941_"
        # pref_path = "image_2fog_"
        # pref_path = "image_2_"
        # print(pref_path)
        
        
        for i in idx_list:
            # res.append((i, pool.apply_async(convert_and_save, args=(args, i, pref_path),
            #                                 callback=update)))

            res.append((i, pool.apply_async(convert_and_save, args=(args, i, ""),
                                            callback=update)))
        pool.close()
        pool.join()
        pbar.clear(nolock=False)
        pbar.close()