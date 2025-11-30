import sys
sys.path.append('EfficientFlow')
from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
import copy
import time
import numpy as np
from EfficientFlow.sde_lib import ConsistencyFM

from EfficientFlow.model.common.normalizer import LinearNormalizer

from EfficientFlow.policy.base_policy import BasePolicy

from EfficientFlow.model.flow.mask_generator import LowdimMaskGenerator 

from EfficientFlow.common.pytorch_util import dict_apply

from EfficientFlow.common.model_util import print_params 
import warnings
warnings.filterwarnings("ignore")

from EfficientFlow.model.equi.equi_obs_encoder import EquivariantObsEnc
from EfficientFlow.model.equi.equi_conditional_unet1d import EquiMeanflowUNet
from EfficientFlow.model.vision.rot_randomizer import RotRandomizer
from functools import partial


def stopgrad(x):
    return x.detach()

def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    delta_sq = torch.mean(error ** 2, dim=(1, 2), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  
    return (stopgrad(w) * loss).mean()

class EquiMeanFlowPolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict, 
            horizon, 
            n_action_steps, 
            n_obs_steps,
            obs_as_global_cond=True,
            # image
            crop_shape=(76, 76), 
            # arch
            N=8, 
            enc_n_hidden=64, 
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True, 
            rot_aug=False, 
            condition_type="film",
            use_pc_color=False,
            pointnet_type="mlp",
            cfg_scale=2.0,
            cfg_uncond='u',
            flow_ratio=0.25,
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
   
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:  
            # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])
        
        self.enc = EquivariantObsEnc(
            obs_shape=obs_shape_meta['agentview_image']['shape'], 
            crop_shape=crop_shape, 
            n_hidden=enc_n_hidden, 
            N=N)

        obs_feature_dim = enc_n_hidden 
        global_cond_dim = obs_feature_dim * n_obs_steps 

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        
        self.diff = EquiMeanflowUNet(
            act_emb_dim=64,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            N=N,
        )

        
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.rot_randomizer = RotRandomizer() 

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.crop_shape = crop_shape 
        self.obs_feature_dim = obs_feature_dim 
        self.rot_aug = rot_aug 
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.predict_action_times_tlag=0
        self.compared_action_steps = self.n_action_steps

        self.time_dist =['lognorm', -0.4, 1.0]
        self.w = cfg_scale
        self.cfg_uncond = cfg_uncond 
        self.flow_ratio = flow_ratio
        self.cfg_ratio=0.1
        self.jvp_fn = torch.autograd.functional.jvp
        self.create_graph = True

        
    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm': 
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu# 
            samples = 1 / (1 + np.exp(-normal_samples))  

        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r
    

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor],preaction=None) -> Dict[str, torch.Tensor]: 
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        # normalize input
        
        nobs = self.normalizer.normalize(obs_dict)

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        model=self.diff


        # build input
        device = self.device
        dtype = self.dtype
            
         
        nobs_features = self.enc(nobs)
        local_cond = None
        global_cond = nobs_features.reshape(B, -1)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
         
        batch_size = cond_data.shape[0]
        noise = torch.randn(
            size=( batch_size, *cond_data.shape[1:]), 
            dtype=cond_data.dtype,
            device=cond_data.device
        )
        z = noise.detach().clone() 

        t = torch.ones((B,), device=device)
        r = torch.zeros((B,), device=device)

        z = z - model(z=z, t=t, r=r, global_cond=global_cond)
        naction_pred = z[..., :Da] 

        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        action_to_compare=action_pred[ : ,end-1:end-1+self.compared_action_steps]#7-11
        result = {
            'action': action,
            'action_pred': action_pred,
            'action_to_compare':action_to_compare
        }
        return result
    
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def compute_loss(self, batch):

        model=self.diff
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])       
        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)
        target = nactions
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory        
        nobs_features = self.enc(nobs)        
        # reshape back to B, Do
        global_cond = nobs_features.reshape(batch_size, -1)
        
        # gt & noise
        target = target
        x = target 
        e = torch.randn(trajectory.shape, device=trajectory.device)
        
        t,r = self.sample_t_r(target.shape[0], device=target.device)
              
        t_expand = t.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        r_expand = r.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])

        z = (1 - t_expand) * x + t_expand * e
        v = e - x
        c=global_cond
        
        assert self.cfg_ratio is not None     
        if self.w is not None:
            with torch.no_grad():          
                u_t = model(z=z, t=t, r=t, global_cond=c)
                
            guided_v_hat = self.w * v + (1 - self.w) * u_t
            v_hat=guided_v_hat
            
        model_partial = partial(model, global_cond=c)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)
        u_tgt = v_hat - (t_expand - r_expand) * dudt

        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
        mse_val = (stopgrad(error) ** 2).mean()

        return loss