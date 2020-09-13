from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from .networks import LocNetReg2D, LocNetRegAVD, MLP
from utils import transform_to_global_2D, transform_to_global_AVD


def get_M_net_inputs_labels(occupied_points, unoccupited_points):
    """
    get global coord (occupied and unoccupied) and corresponding labels
    """
    n_pos = occupied_points.shape[1]
    inputs = torch.cat((occupied_points, unoccupited_points), 1)
    bs, N, _ = inputs.shape

    gt = torch.zeros([bs, N, 1], device=occupied_points.device)
    gt.requires_grad_(False)
    gt[:, :n_pos, :] = 1
    return inputs, gt


def sample_unoccupied_point(local_point_cloud, n_samples):
    """
    sample unoccupied points along rays in local point cloud
    sensor located at origin
    local_point_cloud: <BxNxk>
    n_samples: number of samples on each ray
    """
    bs, L, k = local_point_cloud.shape
    unoccupied = torch.zeros(bs, L * n_samples, k,
                             device=local_point_cloud.device)
    for idx in range(1, n_samples + 1):
        fac = torch.rand(1).item()
        unoccupied[:, (idx - 1) * L:idx * L, :] = local_point_cloud * fac
    return unoccupied

class DeepMapping2D(nn.Module):
    def __init__(self, n_lat, loss_fn, n_obs=256, n_samples=19, dim=[2, 64, 512, 512, 256, 128, 1]):
        super(DeepMapping2D, self).__init__()
        self.n_obs = n_obs
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.loc_net = LocNetReg2D(n_lat, n_points=n_obs, out_dims=3)
        self.occup_net = MLP(dim)

    def forward(self, lats, obs_local,valid_points):
        # obs_local: <BxLx2>
        self.obs_local = deepcopy(obs_local)
        self.valid_points = valid_points
        
        obs = torch.cat([self.obs_local,lats],-1)
        self.pose_est = self.loc_net(obs)
        
        
        self.obs_global_est = transform_to_global_2D(
            self.pose_est, self.obs_local)
        
        '''
        theta = -65
        theta_rad = theta/180*3.1415925
        matrix = np.array([[np.cos(theta_rad),-np.sin(theta_rad)],[np.sin(theta_rad),np.cos(theta_rad)]])
        matrix_t = torch.from_numpy(matrix).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).cuda()
        matrix_t_r = matrix_t.repeat(self.obs_global_est.size()[0],self.obs_global_est.size()[1],1,1)
              
        matrixl = np.array([[np.cos(theta_rad),-np.sin(theta_rad),0],[np.sin(theta_rad),np.cos(theta_rad),0],[0,0,1]])
        matrixl_t = torch.from_numpy(matrixl).unsqueeze(0).type(torch.FloatTensor).cuda()
        matrixl_t_r = matrixl_t.repeat(self.pose_est.size()[0],1,1)
              
            
        #print(matrix_t.size())
        #print(self.obs_global_est.size())
        size_local = self.pose_est.size()
        ss = torch.zeros(0).cuda()
        for i in range(size_local[0]):
            tmp =  matrixl_t_r[i].mm(self.pose_est[i].unsqueeze(-1)).cuda() 
            self.pose_est[i] = tmp.squeeze()
                
                
                
        #self.obs_global_est = matrix_t * self.obs_global_est
        print(self.pose_est.size(),'dsd')
        size_global = self.obs_global_est.size()
        ss = torch.zeros(0).cuda()
        for i in range(size_global[0]):
            s = torch.zeros(0).cuda()
            for j in range(size_global[1]):
                tmp =  matrix_t_r[i][j].mm(self.obs_global_est[i][j].unsqueeze(-1)).cuda() 
                
                self.obs_global_est[i][j] = tmp.squeeze()
                #rint(tmp)
                #torch.cat([s,tmp.cuda()],0)
            #torch.cat([ss,s],2)
        
        '''
        
        if self.training:
            self.unoccupied_local = sample_unoccupied_point(
                self.obs_local, self.n_samples)
            self.unoccupied_global = transform_to_global_2D(
                self.pose_est, self.unoccupied_local)

            inputs, self.gt = get_M_net_inputs_labels(
                self.obs_global_est, self.unoccupied_global)
            self.occp_prob = self.occup_net(inputs)
            loss = self.compute_loss()
            return loss

    def compute_loss(self):
        valid_unoccupied_points = self.valid_points.repeat(1, self.n_samples)
        bce_weight = torch.cat(
            (self.valid_points, valid_unoccupied_points), 1).float()
        # <Bx(n+1)Lx1> same as occp_prob and gt
        bce_weight = bce_weight.unsqueeze(-1)

        if self.loss_fn.__name__ == 'bce_ch':
            loss = self.loss_fn(self.occp_prob, self.gt, self.obs_global_est,
                                self.valid_points, bce_weight, seq=4, gamma=0.1)  # BCE_CH
        elif self.loss_fn.__name__ == 'bce':
            loss = self.loss_fn(self.occp_prob, self.gt, bce_weight)  # BCE
        return loss



class DeepMapping_AVD(nn.Module):
    #def __init__(self, loss_fn, n_samples=35, dim=[3, 256, 256, 256, 256, 256, 256, 1]):
    def __init__(self, loss_fn, n_samples=35, dim=[3, 64, 512, 512, 256, 128, 1]):
        super(DeepMapping_AVD, self).__init__()
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.loc_net = LocNetRegAVD(out_dims=3) # <x,z,theta> y=0
        self.occup_net = MLP(dim)

    def forward(self, obs_local,valid_points):
        # obs_local: <BxHxWx3> 
        # valid_points: <BxHxW>
        
        self.obs_local = deepcopy(obs_local)
        self.valid_points = valid_points
        self.pose_est = self.loc_net(self.obs_local)

        bs = obs_local.shape[0]
        self.obs_local = self.obs_local.view(bs,-1,3)
        self.valid_points = self.valid_points.view(bs,-1)
        
        self.obs_global_est = transform_to_global_AVD(
            self.pose_est, self.obs_local)

        if self.training:
            self.unoccupied_local = sample_unoccupied_point(
                self.obs_local, self.n_samples)
            self.unoccupied_global = transform_to_global_AVD(
                self.pose_est, self.unoccupied_local)

            inputs, self.gt = get_M_net_inputs_labels(
                self.obs_global_est, self.unoccupied_global)
            self.occp_prob = self.occup_net(inputs)
            loss = self.compute_loss()
            return loss

    def compute_loss(self):
        valid_unoccupied_points = self.valid_points.repeat(1, self.n_samples)
        bce_weight = torch.cat(
            (self.valid_points, valid_unoccupied_points), 1).float()
        # <Bx(n+1)Lx1> same as occp_prob and gt
        bce_weight = bce_weight.unsqueeze(-1)

        if self.loss_fn.__name__ == 'bce_ch':
            loss = self.loss_fn(self.occp_prob, self.gt, self.obs_global_est,
                                self.valid_points, bce_weight, seq=2, gamma=0.9)  # BCE_CH
        elif self.loss_fn.__name__ == 'bce':
            loss = self.loss_fn(self.occp_prob, self.gt, bce_weight)  # BCE
        return loss


