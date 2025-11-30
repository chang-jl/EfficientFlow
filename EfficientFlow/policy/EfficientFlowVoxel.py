from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as vision_models
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from EfficientFlow.model.common.normalizer import LinearNormalizer
from EfficientFlow.policy.base_image_policy import BaseImagePolicy
from EfficientFlow.common.robomimic_config_util import get_robomimic_config
from EfficientFlow.model.diffusion.mask_generator import LowdimMaskGenerator
from EfficientFlow.model.common.rotation_transformer import RotationTransformer
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
from robomimic.models.base_nets import SpatialSoftmax
try:
    import robomimic.models.base_nets as rmbn
    if not hasattr(rmbn, 'CropRandomizer'):
        raise ImportError("CropRandomizer is not in robomimic.models.base_nets")
except ImportError:
    import robomimic.models.obs_core as rmbn
import EfficientFlow.model.vision.crop_randomizer as dmvc
from EfficientFlow.common.pytorch_util import dict_apply, replace_submodules

import numpy as np
import itertools
from einops import rearrange, repeat
from EfficientFlow.sde_lib import ConsistencyFM
from EfficientFlow.model.equi.equi_obs_encoder import EquivariantObsEncVoxel
from EfficientFlow.model.equi.equi_conditional_unet1d import EquiDiffusionUNet
from EfficientFlow.model.vision.voxel_rot_randomizer import VoxelRotRandomizer


class EfficientFlowVoxelPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            # image
            crop_shape=(58, 58, 58),
            # arch
            N=8,
            enc_n_hidden=64,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            rot_aug=False,
            initialize=True,
            color=True,
            depth=True,        
            eta=0.01,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]


        if color and depth:
            obs_channel = 4
        elif color:
            obs_channel = 3
        elif depth:
            obs_channel = 1  

        self.enc = EquivariantObsEncVoxel(
            obs_shape=(obs_channel, 64, 64, 64), 
            crop_shape=crop_shape, 
            n_hidden=enc_n_hidden, 
            N=N,
            initialize=initialize,
            )
        
        obs_feature_dim = enc_n_hidden
        global_cond_dim = obs_feature_dim * n_obs_steps
        
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
        
        print("Enc params: %e" % sum(p.numel() for p in self.enc.parameters()))
        print("Diff params: %e" % sum(p.numel() for p in self.diff.parameters()))

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.rot_randomizer = VoxelRotRandomizer()

        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.crop_shape = crop_shape
        self.obs_feature_dim = obs_feature_dim
        self.rot_aug = rot_aug
        self.kwargs = kwargs
        self.noise_scheduler = noise_scheduler
        self.predict_action_times_tlag=0
        self.compared_action_steps = self.n_action_steps

        self.eta = eta
        self.num_of_trajectories=5
        self.eps = 1e-2
        self.delta =1e-2
        self.num_inference_step =1
        

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            weight_decay: float, 
            learning_rate: float, 
            betas: Tuple[float, float],
            eps: float
        ) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(), weight_decay=weight_decay, lr=learning_rate, betas=betas, eps=eps
        )
        return optimizer
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor],preaction=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        if 'agentview_image' in obs_dict:
            del obs_dict['agentview_image']
        obs_dict['voxels'][:, :, 1:] /= 255.0
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

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        nobs_features = self.enc(nobs)
        # reshape back to B, Do
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
        candidates = action_preds#torch.stack(all_action_preds, dim=0)
        
    
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
        # normalize input
        assert 'valid_mask' not in batch
        eps = self.eps

        delta  = self.delta

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)
                
        target = nactions

        batch_size = nactions.shape[0]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        nobs_features = self.enc(nobs)
        # reshape back to B, Do
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
        loss= loss_flow * 2   +  loss_FABO
        return loss 