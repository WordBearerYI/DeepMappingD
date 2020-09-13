import set_path
import os
import argparse
import functools
print = functools.partial(print,flush=True)

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
import loss
from models import DeepMapping2D
from dataset_loader import SimulatedPointCloud,SimulatedPointCloud_train

torch.backends.cudnn.deterministic = True
torch.manual_seed(2333)

parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str,default='test',help='experiment name')
parser.add_argument('-e','--n_epochs',type=int,default=888,help='number of epochs')
parser.add_argument('-b','--batch_size',type=int,default=128,help='batch_size')
parser.add_argument('-l','--loss',type=str,default='bce_ch',help='loss function')
parser.add_argument('-n','--n_samples',type=int,default=19,help='number of sampled unoccupied points along rays')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('-d','--data_dir',type=str,default='../data/2D/',help='dataset path')
parser.add_argument('-m','--model', type=str, default=None,help='pretrained model name')
parser.add_argument('-i','--init', type=str, default=None,help='init pose')
parser.add_argument('--log_interval',type=int,default=10,help='logging interval of saving results')
parser.add_argument('--lat_size',type=int, help='latent size of latent vector')

opt = parser.parse_args()
print('LSTM',opt)
checkpoint_dir = os.path.join('../results/2D','lstm_' + opt.name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
utils.save_opt(checkpoint_dir,opt)

os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

init_pose = None
latent_size = opt.lat_size
instances_per_scene = 256

latent_vecs = []
for i in range(instances_per_scene):
    vec = nn.Parameter(torch.ones(latent_size).normal_(0, 0.8).to(device))
    vec.requires_grad = True
    latent_vecs.append(vec)

w = torch.ones(latent_size).normal_(0, 0.8).to(device)
#w_r = torch.ones(latent_size).normal_(0, 0.8).to(device)
w.requires_grad = True
#w_r.requires_grad = True

dataset_train = SimulatedPointCloud_train(opt.data_dir,instances_per_scene,init_pose)
loader_train = DataLoader(dataset_train,batch_size=1,shuffle=False)

dataset = SimulatedPointCloud(opt.data_dir,instances_per_scene,init_pose)
loader = DataLoader(dataset,batch_size=opt.batch_size,shuffle=False)
#print(len(dataset))
loss_fn = eval('loss.'+opt.loss)

print('creating model')
model = DeepMapping2D(n_lat =latent_size, loss_fn=loss_fn, n_obs=dataset.n_obs, n_samples=opt.n_samples).to(device)

optimizer = optim.Adam(
            [
                {
                     "params":model.parameters(), "lr":opt.lr
                },
                {
                     "params": latent_vecs, "lr":opt.lr*2
                },
                {
                     "params": w, "lr":opt.lr
                }
            ]
            )


if opt.model is not None:
    utils.load_checkpoint(opt.model,model,optimizer)

print('start training')
for epoch in range(opt.n_epochs):
    
    training_loss= 0.0
    model.train()
    if epoch<50:
        loader_cur = loader_train
    else:
        loader_cur = loader
    for index,(obs_batch,valid_pt,index_latents) in enumerate(loader_cur):
        obs_batch = obs_batch.to(device).float().squeeze()
        valid_pt = valid_pt.to(device).squeeze()
        #print(index_latents)
        lats_inds = index_latents
        latent_inputs = torch.ones((len(lats_inds),latent_size), dtype=torch.float32, requires_grad=True).to(device) 
        i=0
        for i_lat_ts in lats_inds:
            #print(i_lat,index,epoch)
            i_lat = i_lat_ts.cpu().detach()
            if i_lat==0:
                latent = latent_vecs[i_lat]
            else:
                latent = (latent_vecs[i_lat] + w * latent_vecs[i_lat-1])/(1+torch.norm(w, p=2))
                latent_vecs[i_lat] = latent
            latent_inputs[i] *= latent_vecs[i_lat]
            i += 1
        
        latent_inputs = latent_inputs.unsqueeze(1).expand(-1,obs_batch.shape[1],-1)
        #print(obs_batch.size(),latent_inputs.size())
        loss = model(latent_inputs,obs_batch,valid_pt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        for i_lat in range(len(latent_vecs)):
            latent_vecs[i_lat] = latent_vecs[i_lat].detach()
          
    training_loss_epoch = training_loss/len(loader)
    
    if (epoch+1) % opt.log_interval==0:
        print('[{}/{}], training loss: {:.4f}'.format(
            epoch+1,opt.n_epochs,training_loss_epoch))
        
        obs_global_est_np = []
        pose_est_np = []
        with torch.no_grad():
            model.eval()
            for index,(obs_batch,valid_pt,index_latents) in enumerate(loader):
                latent_inputs = torch.ones((obs_batch.shape[0],latent_size), dtype=torch.float32).to(device) 
                i = 0
                for i_lat in lats_inds:
                    #latent_inputs = torch.cat([latent_inputs,latent_vecs[i_lat].unsqueeze(0)], 0)
                    latent_inputs[i] *= latent_vecs[i_lat]
                    i += 1
                latent_inputs = latent_inputs.unsqueeze(1).expand(-1,obs_batch.shape[1],-1)
                #print("sdsd",latent_inputs)
                obs_batch = obs_batch.to(device).float()
                valid_pt = valid_pt.to(device)
                model(latent_inputs,obs_batch,valid_pt)
                obs_global_est_np.append(model.obs_global_est.cpu().detach().numpy())
                pose_est_np.append(model.pose_est.cpu().detach().numpy())
                #sprint(model.obs_global_est.cpu().detach().numpy())
                
            pose_est_np = np.concatenate(pose_est_np)
            if init_pose is not None:
                pose_est_np = utils.cat_pose_2D(init_pose_np,pose_est_np)
            
            save_name = os.path.join(checkpoint_dir,'model_best.pth')
            utils.save_checkpoint(save_name,model,optimizer)
            
            obs_global_est_np = np.concatenate(obs_global_est_np)
            kwargs = {'e':epoch+1}
            valid_pt_np = dataset.valid_points.cpu().detach().numpy()
            utils.plot_global_point_cloud(obs_global_est_np, pose_est_np, valid_pt_np, checkpoint_dir,**kwargs)
            
            save_name = os.path.join(checkpoint_dir,'obs_global_est.npy')
            np.save(save_name,obs_global_est_np)

            save_name = os.path.join(checkpoint_dir,'pose_est.npy')
            np.save(save_name,pose_est_np)
            
    if (epoch+1) % (opt.log_interval*5) == 0:
        os.system('./run_eval_2D.sh '+ 'lstm_' + opt.name)
