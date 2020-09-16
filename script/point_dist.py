import set_path
import os
import argparse
import functools
import copy
print = functools.partial(print,flush=True)

import numpy as np
import open3d as o3d
import torch
import utils 
from dataset_loader import AVD,AVDtrain
from utils import transform_to_global_2D, transform_to_global_AVD
colors = np.random.rand(3,16)

def add_y_coord_for_evaluation(pred_pos_DM):
    """
    pred_pos_DM (predicted position) estimated from DeepMapping only has x and z coordinate
    convert this to <x,y=0,z> for evaluation
    """
    n = pred_pos_DM.shape[0]
    x = pred_pos_DM[:,0]
    y = np.zeros_like(x)
    z = pred_pos_DM[:,1]
    return np.stack((x,y,z),axis=-1)

def np_to_pcd(xyz):
    """
    convert numpy array to point cloud object in open3d
    """
    xyz = xyz.reshape(-1,3)
    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(xyz)
    return pcd

def ang2mat(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return c,s

def pt_diff(a,b):
    dist = np.linalg.norm(a-b)
    return dist

def get_scores(checkpoint_dir):
    saved_json_file = os.path.join(checkpoint_dir,'opt.json')
    train_opt = utils.load_opt_from_json(saved_json_file)
    name = train_opt['name']
    data_dir = '../../../data/ActiveVisionDataset/'+ train_opt['data_dir'].split('/')[-1]
    subsample_rate = train_opt['subsample_rate']
    traj = train_opt['traj']
    print(data_dir)
    # load ground truth poses
    dataset = AVD(data_dir,traj,subsample_rate)
    gt_pose = dataset.gt
    gt_location = gt_pose[:,:3]
    pts = dataset.point_clouds#.numpy()
    #np.save("local.npy",pts)
    #np.save("pose.npy",gt_pose)
    # load predicted poses
    pred_file = os.path.join(opt.checkpoint_dir,'pose_est.npy')
    pred_pose = np.load(pred_file)
    pred_location = pred_pose[:,:2] * dataset.depth_scale # denormalization
    pred_location = add_y_coord_for_evaluation(pred_location)

    #print(gt_pose)
    print(pred_pose)
    ate,aligned_location = utils.compute_ate(pred_location,gt_location)
    print('{}, ate: {}'.format(name,ate))
    gt_pose[:,:3] = gt_pose[:,:3]/dataset.depth_scale 
    gt_yaw = np.arccos(gt_pose[:,5]/np.sqrt(gt_pose[:,3]*gt_pose[:,3]+gt_pose[:,5]*gt_pose[:,5]))
    gt_pose_xzth = np.vstack((gt_pose[:,0],gt_pose[:,2],-gt_yaw)).transpose()
    colors = np.array([[0,1,1],[0,0,0],[0,0,1],[1,0,1],[0.5,0.5,0.5],[0,0.5,0],[0,1,0],[0.5,0,0],[0,0,0.5],[0.5,0.5,0],[0.5,0,0.5],[1,0,0],[0.75,0.75,0.75],[0,0.5,0.5],[1,1,1],[1,1,0]])

    #path_or = '../../../bk_origin/DeepMapping/' 
    global_point_cloud_file = os.path.join(opt.checkpoint_dir,'obs_global_est.npy')
    #global_point_cloud_file_or = os.path.join(path_or,global_point_cloud_file[3:])
    pcds_ours = utils.load_obs_global_est(global_point_cloud_file,colors)
    #pcds_or = utils.load_obs_global_est(global_point_cloud_file_or,colors)
    pts_gt = transform_to_global_AVD(gt_pose_xzth,pts).numpy()
    #pts_or = np.load(global_point_cloud_file_or)
    pts_ours = np.load(global_point_cloud_file)

    pts_gt = pts_gt.reshape((16,-1,3))
    
    print(pt_diff(pts_gt,pts_ours))
    print(pt_diff(pts_gt,pts_or))
