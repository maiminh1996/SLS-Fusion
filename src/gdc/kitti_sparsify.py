import argparse
import os

import numpy as np
import tqdm


def pto_rec_map(velo_points, H=64, W=512, D=800):
    # depth, width, height
    valid_inds = (velo_points[:, 0] < 80) & \
                 (velo_points[:, 0] >= 0) & \
                 (velo_points[:, 1] < 50) & \
                 (velo_points[:, 1] >= -50) & \
                 (velo_points[:, 2] < 1) & \
                 (velo_points[:, 2] >= -2.5)
    velo_points = velo_points[valid_inds]

    x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]
    x_grid = (x * D / 80.).astype(int)
    x_grid[x_grid < 0] = 0
    x_grid[x_grid >= D] = D - 1

    y_grid = ((y + 50) * W / 100.).astype(int)
    y_grid[y_grid < 0] = 0
    y_grid[y_grid >= W] = W - 1

    z_grid = ((z + 2.5) * H / 3.5).astype(int)
    z_grid[z_grid < 0] = 0
    z_grid[z_grid >= H] = H - 1

    depth_map = - np.ones((D, W, H, 4))
    depth_map[x_grid, y_grid, z_grid, 0] = x
    depth_map[x_grid, y_grid, z_grid, 1] = y
    depth_map[x_grid, y_grid, z_grid, 2] = z
    depth_map[x_grid, y_grid, z_grid, 3] = i
    depth_map = depth_map.reshape((-1, 4))
    depth_map = depth_map[depth_map[:, 0] != -1.0]
    return depth_map


def pto_ang_map(velo_points, H=64, W=512, slice=1):
    """
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """

    dtheta = np.radians(0.4 * 64.0 / H)
    dphi = np.radians(90.0 / W)

    x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = np.radians(45.) - np.arcsin(y / r)
    phi_ = (phi / dphi).astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W - 1

    theta = np.radians(2.) - np.arcsin(z / d)
    theta_ = (theta / dtheta).astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H - 1

    depth_map = - np.ones((H, W, 4))
    depth_map[theta_, phi_, 0] = x
    depth_map[theta_, phi_, 1] = y
    depth_map[theta_, phi_, 2] = z
    depth_map[theta_, phi_, 3] = i
    depth_map = depth_map[0::slice, :, :]
    depth_map = depth_map.reshape((-1, 4))
    depth_map = depth_map[depth_map[:, 0] != -1.0]
    return depth_map


def gen_sparse_points(pl_data_path, args):
    pc_velo = np.fromfile(pl_data_path, dtype=np.float32).reshape((-1, 4))

    # depth, width, height
    valid_inds = (pc_velo[:, 0] < 120) & \
                 (pc_velo[:, 0] >= 0) & \
                 (pc_velo[:, 1] < 50) & \
                 (pc_velo[:, 1] >= -50) & \
                 (pc_velo[:, 2] < 1.5) & \
                 (pc_velo[:, 2] >= -2.5)
    pc_velo = pc_velo[valid_inds]

    return pto_ang_map(pc_velo, H=args.H, W=args.W, slice=args.slice)


def gen_sparse_points_all(args):
    outputfolder = args.sparse_pl_path
    os.makedirs(outputfolder, exist_ok=True)
    data_idx_list = sorted([x.strip() for x in os.listdir(args.pl_path) if x[-3:] == 'bin'])
    # with open(args.split_file) as f:
    #     data_idx_list = [int(x.strip()) for x in f.readlines() if len(x.strip()) > 0]
    

    # for data_idx in tqdm.tqdm(data_idx_list):
    #     sparse_points = gen_sparse_points(os.path.join(args.pl_path, data_idx), args)
    #     sparse_points = sparse_points.astype(np.float32)
    #     sparse_points.tofile(f'{outputfolder}/{data_idx}')
    # for pref_path in ["image_21_", "image_left_beta0.039941_"]:
    for data_idx in tqdm.tqdm(data_idx_list):
        sparse_points = gen_sparse_points(os.path.join(args.pl_path, data_idx), args)
        sparse_points = sparse_points.astype(np.float32)
        sparse_points.tofile(f'{outputfolder}/{data_idx}')


if __name__ == '__main__':
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/04-02-2021_11-37-37/depth_maps_correct_4beams_hazing_bin/  --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/04-02-2021_11-37-37/depth_maps_correct_4beams_hazing_bin_sparse
    # python ./src/preprocess/kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/04-02-2021_11-37-37/depth_maps_correct_4beams_hazing_bin/  \
    # --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/04-02-2021_11-37-37/depth_maps_correct_4beams_hazing_bin_sparse

    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/09-02-2021_13-45-30/depth_maps_correct_4beams_bin/ --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/09-02-2021_13-45-30/depth_maps_correct_4beams_bin_sparse
    # python ./src/preprocess/kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/09-02-2021_13-45-30/depth_maps_correct_4beams_bin/  \
    # --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/09-02-2021_13-45-30/depth_maps_correct_4beams_bin_sparse

    # (8)
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/11-02-2021_11-16-25/depth_maps_correct_4beams_bin/ --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/11-02-2021_11-16-25/depth_maps_correct_4beams_bin_sparse
    # python ./src/preprocess/kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/11-02-2021_11-16-25/depth_maps_correct_4beams_bin/  \
    # --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/11-02-2021_11-16-25/depth_maps_correct_4beams_bin_sparse
    
    # 11
    # python ./src/preprocess/kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/28-02-2021_00-54-16/depth_maps_correct_4beams_bin/ --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/28-02-2021_00-54-16/depth_maps_correct_4beams_bin_sparse
    # python ./src/preprocess/kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/28-02-2021_00-54-16/depth_maps_correct_4beams_bin/  \
    # --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/28-02-2021_00-54-16/depth_maps_correct_4beams_bin_sparse

    # 14
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/14-03-2021_23-29-24/depth_maps_correct_4beams_bin/ --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/14-03-2021_23-29-24/depth_maps_correct_4beams_bin_sparse
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/14-03-2021_23-29-24/depth_maps_correct_4beams_bin/  \
    # --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/14-03-2021_23-29-24/depth_maps_correct_4beams_bin_sparse

    # 14a
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/18-03-2021_23-26-18/depth_maps_correct_4beams_bin/ --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/18-03-2021_23-26-18/depth_maps_correct_4beams_bin_sparse
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/18-03-2021_23-26-18/depth_maps_correct_4beams_bin/  \
    # --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/18-03-2021_23-26-18/depth_maps_correct_4beams_bin_sparse

    # 10
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/18-03-2021_17-15-07/depth_maps_correct_4beams_bin/ --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/18-03-2021_17-15-07/depth_maps_correct_4beams_bin_sparse
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/18-03-2021_17-15-07/depth_maps_correct_4beams_bin/  \
    # --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/18-03-2021_17-15-07/depth_maps_correct_4beams_bin_sparse

    # 12
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/21-03-2021_20-25-57/depth_maps_correct_4beams_bin/ --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/21-03-2021_20-25-57/depth_maps_correct_4beams_bin_sparse
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/21-03-2021_20-25-57/depth_maps_correct_4beams_bin/  \
    # --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/21-03-2021_20-25-57/depth_maps_correct_4beams_bin_sparse
    # 21-03-2021_20-25-57


    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/06-05-2021_01-21-11/depth_maps_correct_4beams_bin/ --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/06-05-2021_01-21-11/depth_maps_correct_4beams_bin_sparse
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/06-05-2021_01-21-11/depth_maps_correct_4beams_bin/  \
    # --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/06-05-2021_01-21-11/depth_maps_correct_4beams_bin_sparse
    # 06-05-2021_01-21-11

    
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/06-05-2021_07-20-40/depth_maps_correct_4beams_bin/ --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/06-05-2021_07-20-40/depth_maps_correct_4beams_bin_sparse
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/06-05-2021_07-20-40/depth_maps_correct_4beams_bin/  \
    # --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/06-05-2021_07-20-40/depth_maps_correct_4beams_bin_sparse
    # 06-05-2021_07-20-40

    # python kitti_sparsify.py --pl_path  /media/zelt/Données/Pseudo_Lidar_V2/results/sdn_kitti_train_set/depth_maps_correct_4beams_bin/  --sparse_pl_path  /media/zelt/Données/Pseudo_Lidar_V2/results/sdn_kitti_train_set/depth_maps_correct_4beams_bin_sparse
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/Pseudo_Lidar_V2/results/sdn_kitti_train_set/depth_maps_correct_4beams_bin/ 
    # --sparse_pl_path  /media/zelt/Données/Pseudo_Lidar_V2
    # /results/sdn_kitti_train_set/depth_maps_correct_4beams_bin_sparse

    # lidar 64
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_LiDAR_64beams_depthmap/27-09-2022_11-19-01/pseudo_pointcloud/ --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_LiDAR_64beams_depthmap/27-09-2022_11-19-01/pseudo_pointcloud_64/

    # stereo + lidar64
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_STEREO_LiDAR_64beams_new_depthmap/26-09-2022_09-40-17/pseudo_pointcloud/ --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_STEREO_LiDAR_64beams_new_depthmap/26-09-2022_09-40-17/pseudo_pointcloud_64/

    # lidar 8
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_LiDAR_8beams_depthmap/30-09-2022_14-16-05/pseudo_pointcloud --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_LiDAR_8beams_depthmap/30-09-2022_14-16-05/pseudo_pointcloud_64/

    # lidar 16
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_LiDAR_16beams_depthmap/04-10-2022_12-57-04/pseudo_pointcloud --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_LiDAR_16beams_depthmap/04-10-2022_12-57-04/pseudo_pointcloud_64/

    # stereo  lidar 16
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_STEREO_LiDAR_16_beams_depthmap/07-10-2022_12-23-43/pseudo_pointcloud --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_STEREO_LiDAR_16_beams_depthmap/07-10-2022_12-23-43/pseudo_pointcloud_64/

    # stereo  lidar 8
    # python kitti_sparsify.py --pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_STEREO_LiDAR_8_beams_depthmap/10-10-2022_14-00-02/pseudo_pointcloud --sparse_pl_path  /media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set_STEREO_LiDAR_8_beams_depthmap/10-10-2022_14-00-02/pseudo_pointcloud_64/



    parser = argparse.ArgumentParser("Generate sparse pseudo-LiDAR points")
    parser.add_argument('--pl_path', default='/media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/04-02-2021_11-37-37/pseudo_lidar/trainval', help='pseudo-lidar path')
    parser.add_argument('--sparse_pl_path', default='/media/zelt/Données/SLS-Fusion/src/training/results/sls_fusion_kitti_train_set/04-02-2021_11-37-37/pseudo_lidar_sparse/trainval', help='sparsed pseudo lidar path')
    parser.add_argument('--slice', default=1, type=int)
    parser.add_argument('--H', default=64, type=int)
    parser.add_argument('--W', default=512, type=int)
    parser.add_argument('--D', default=700, type=int)
    args = parser.parse_args()

    gen_sparse_points_all(args)