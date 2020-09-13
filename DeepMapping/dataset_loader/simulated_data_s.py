import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from open3d import read_point_cloud
import scipy.io as sio
import utils


def find_valid_points(local_point_cloud):
    """
    find valid points in local point cloud
        invalid points have all zeros local coordinates
    local_point_cloud: <BxNxk> 
    valid_points: <BxN> indices  of valid point (0/1)
    """
    eps = 1e-6
    non_zero_coord = torch.abs(local_point_cloud) > eps
    valid_points = torch.sum(non_zero_coord, dim=-1)
    valid_points = valid_points > 0
    return valid_points


class SimulatedPointCloud(Dataset):
    def __init__(self, root, instance_num, trans_by_pose=None):
        # trans_by_pose: <Bx3> pose
        self._trans_by_pose = trans_by_pose
        #file_list = glob.glob(os.path.join(self.root, '*pcd'))
        self.file = root
        mats = sio.loadmat(self.file)
        point_clouds = mats['input_points']
        # pc: B*VIEW*2 256*240*2
        gt_pose = mats['gt_camera']
        # gp: B*3      256*3
        gt_map = mats['gt_points']
        # pc: B*VIEW*2 256*240*2
        
        self.point_clouds = torch.from_numpy(point_clouds) # <NxLx2>
        self.valid_points = find_valid_points(self.point_clouds) # <NxL>
        # number of points in each point cloud
        self.n_obs = self.point_clouds.shape[1]
        self.instance_indeces = range(instance_num)
        print(self.n_obs,self.point_clouds.shape)
        
    def __getitem__(self, index):
        pcd = self.point_clouds[index,:,:]  # <Lx2> 240*2
        valid_points = self.valid_points[index,:]
        if self._trans_by_pose is not None:
            pcd = pcd.unsqueeze(0)  # <1XLx2> 
            pose = self._trans_by_pose[index, :].unsqueeze(0)  # <1x3>
            pcd = utils.transform_to_global_2D(pose, pcd).squeeze(0)
        latent_index = self.instance_indeces[index]
        
        return pcd,valid_points,latent_index

    def __len__(self):
        return len(self.point_clouds)
