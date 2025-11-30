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
from EfficientFlow.model.equi.equi_conditional_unet1d import EquiDiffusionUNet
from EfficientFlow.model.vision.rot_randomizer import RotRandomizer

class EfficientFlowPolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict, 
            horizon, 
            n_action_steps, 
            n_obs_steps,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
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
            eta=0.01,
            use_preaction_as_noise=False,
            num_of_trajectories=5,
            num_inference_step=1,
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        #assert len(action_shape) == 1 
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
 
        self.diff = EquiDiffusionUNet(
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

        self.eta = eta
        self.eps = 0.01
        self.delta = 0.01
        self.num_inference_step = num_inference_step
     
        self.num_of_trajectories=num_of_trajectories

        print('usepreactionasnoise:',use_preaction_as_noise)
        print('num_of_trajectorys:',num_of_trajectories)    
        print('num_inference_step',self.num_inference_step)
        print_params(self)
       

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor],preaction=None,) -> Dict[str, torch.Tensor]: 
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        self.print=1
        nobs = self.normalizer.normalize(obs_dict)

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype
                    
         
        nobs_features = self.enc(nobs)
        # reshape back to B, Do
        local_cond = None
        global_cond = nobs_features.reshape(B, -1)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
 
        
        n = self.num_of_trajectories
        
        batch_size = cond_data.shape[0]
        noise = torch.randn(
            size=(n, batch_size, *cond_data.shape[1:]),
            dtype=cond_data.dtype,
            device=cond_data.device
        )
        z = noise.detach().clone()  

        sde = ConsistencyFM('gaussian', noise_scale=1.0, sample_N=self.num_inference_step)
        dt = 1.0 / self.num_inference_step
        eps = self.eps

        for i in range(sde.sample_N):
            num_t = i / sde.sample_N * (1 - eps) + eps
            t = torch.ones(batch_size, device=z.device) * num_t 
            
            z_flat = z.view(-1, *z.shape[2:])
            t_flat = t.repeat_interleave(n) 
            
            if global_cond is not None:
                global_cond_expanded = global_cond.unsqueeze(0).expand(n, -1, -1) 
                global_cond_flat = global_cond_expanded.reshape(-1, *global_cond.shape[1:])
            else:
                global_cond_flat = None

            if local_cond is not None:
                local_cond_expanded = local_cond.unsqueeze(0).expand(n, -1, -1) 
                local_cond_flat = local_cond_expanded.reshape(-1, *local_cond.shape[1:]) 
            else:
                local_cond_flat = None
            
            pred_flat = self.diff(z_flat, t_flat * 99, local_cond=local_cond_flat, global_cond=global_cond_flat)
            pred = pred_flat.view(n, batch_size, *z.shape[2:]) 
            
            sigma_t = sde.sigma_t(num_t)
            pred_sigma = pred + (sigma_t**2)/(2*(1.-num_t)**2) * (0.5*num_t*(1.-num_t)*pred - 0.5*(2.-num_t)*z)
            z = z + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma)
      
        z = z.view(n, batch_size,T, Da)
        naction_pred = z[..., :Da]
        action_preds = self.normalizer['action'].unnormalize(naction_pred) 

        
        candidates = action_preds  
        
        if self.predict_action_times_tlag %10 != 0 and preaction is not None and n != 1:            
            candidate_init = candidates[:, :, :self.compared_action_steps, :] 
            history_end = preaction[:,:,:].unsqueeze(0)        
            distances = torch.norm(candidate_init - history_end, dim=(2,3))

            indices = torch.argmin(distances, dim=0) 
            selected_actions = candidates[indices, torch.arange(B)]
        else :
            selected_actions = candidates[ torch.randint(low=0, high=n, size=(1,))].squeeze(0)
            if self.predict_action_times_tlag!=0:
                self.predict_action_times_tlag = 0
        self.predict_action_times_tlag +=1
        # get action
        action_pred=selected_actions
        start = To - 1 
        end = start + self.n_action_steps
        action = selected_actions[:,start:end]

        action_to_compare=selected_actions[ : ,end-1:end-1+self.compared_action_steps]
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
        
        eps = self.eps
        delta  = self.delta
        
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
    
        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)
  
        target = nactions

        batch_size = nactions.shape[0]

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
        nobs_features = self.enc(nobs)
        
        global_cond = nobs_features.reshape(batch_size, -1)
                
        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)
        # gt & noise
        target = target
        a0 = torch.randn(trajectory.shape, device=trajectory.device)
        t = torch.rand(target.shape[0], device=target.device) * (1 - eps) + eps         
        r = torch.clamp(t + delta, max=1.0)        
        t_expand = t.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        r_expand = r.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        xt = t_expand * target + (1.-t_expand) * a0
        xr = r_expand * target + (1.-r_expand) * a0
        
        #apply mask
        xt[condition_mask] = cond_data[condition_mask]
        xr[condition_mask] = cond_data[condition_mask]
        
        vt = self.diff(xt, t*99, cond=local_cond, global_cond=global_cond)
        vr = self.diff(xr, r*99, local_cond=local_cond, global_cond=global_cond)

        # mask
        vt[condition_mask] = cond_data[condition_mask]
        vr[condition_mask] = cond_data[condition_mask]   
        vt = torch.nan_to_num(vt)
        vr = torch.nan_to_num(vr)
     
        loss_flow = torch.nn.functional.mse_loss(vr,(target-a0)) * delta**2 
        loss_FABO = ((vt - vr).pow(2) * ((1.0 - t_expand) ** 2)).mean()

        loss = loss_flow * 2 + loss_FABO  
        return loss