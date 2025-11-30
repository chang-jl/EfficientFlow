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

class EquiCFMPolicy(BasePolicy):
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
            Conditional_ConsistencyFM=None,           
            eta=0.01,
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
        cprint(f"[FlowUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[FlowUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")

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
        

        if Conditional_ConsistencyFM is None:
                    Conditional_ConsistencyFM = {
                        'eps': 1e-2,
                        'num_segments': 2,
                        'boundary': 1,
                        'delta': 1e-2,
                        'alpha': 1e-5,
                        'num_inference_step': 5
                    }
        self.eta = eta
        self.eps = Conditional_ConsistencyFM['eps']
        self.num_segments = Conditional_ConsistencyFM['num_segments']
        self.boundary = Conditional_ConsistencyFM['boundary']
        self.delta = Conditional_ConsistencyFM['delta']
        self.alpha = Conditional_ConsistencyFM['alpha']
        self.num_inference_step = Conditional_ConsistencyFM['num_inference_step']

        print_params(self)
        
    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor],preaction=None,) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
      
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
        
        
        # run sampling
        noise = torch.randn(
            size=cond_data.shape, 
            dtype=cond_data.dtype,
            device=cond_data.device,
            generator=None)
        z = noise.detach().clone() 

        sde = ConsistencyFM('gaussian', 
                            noise_scale=1.0, 
                            use_ode_sampler='rk45', 
                            sigma_var=0.0, 
                            ode_tol=1e-5,   
                            sample_N= self.num_inference_step)

        # Uniform
        dt = 1./self.num_inference_step
        eps = self.eps

        for i in range(sde.sample_N): 
            num_t = i /sde.sample_N * (1 - eps) + eps
            t = torch.ones(z.shape[0], device=noise.device) * num_t
            pred = self.diff(z, t*99, local_cond=local_cond, global_cond=global_cond)
            sigma_t = sde.sigma_t(num_t) 
            pred_sigma = pred + (sigma_t**2)/(2*(sde.noise_scale**2)*((1.-num_t)**2)) * (0.5 * num_t * (1.-num_t) * pred - 0.5 * (2.-num_t)*z.detach().clone()) 
            z = z.detach().clone() + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(device)
        z[cond_mask] = cond_data[cond_mask] # a1
        # unnormalize prediction
        naction_pred = z[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        action_to_compare=action_pred[ : ,end-1:end-1+self.n_action_steps]
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
        num_segments = self.num_segments
        boundary = self.boundary
        delta  = self.delta
        alpha =  self.alpha
        reduce_op = torch.mean
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
         
        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)
        
        target = nactions

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
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

        segments = torch.linspace(0, 1, num_segments + 1, device=target.device)
        seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1) 
        segment_ends = segments[seg_indices] 
        segment_ends_expand = segment_ends.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        x_at_segment_ends = segment_ends_expand * target + (1.-segment_ends_expand) * a0 
    
        def f_euler(t_expand, segment_ends_expand, xt, vt):
            return xt + (segment_ends_expand - t_expand) * vt
        def threshold_based_f_euler(t_expand, segment_ends_expand, xt, vt, threshold, x_at_segment_ends):
            if (threshold, int) and threshold == 0:
                return x_at_segment_ends
      
            less_than_threshold = t_expand < threshold
            res = (
                less_than_threshold * f_euler(t_expand, segment_ends_expand, xt, vt)
                + (~less_than_threshold) * x_at_segment_ends
                )
            return res
       
        vt = self.diff(xt, t*99, cond=local_cond, global_cond=global_cond)
        vr = self.diff(xr, r*99, local_cond=local_cond, global_cond=global_cond)
        # mask
        vt[condition_mask] = cond_data[condition_mask]
        vr[condition_mask] = cond_data[condition_mask]

        vr = torch.nan_to_num(vr)
      
        ft = f_euler(t_expand, segment_ends_expand, xt, vt)
        fr = threshold_based_f_euler(r_expand, segment_ends_expand, xr, vr, boundary, x_at_segment_ends)
        
        losses_f = torch.square(ft - fr)
        losses_f = reduce_op(losses_f.reshape(losses_f.shape[0], -1), dim=-1)
    
        def masked_losses_v(vt, vr, threshold, segment_ends, t):
            if (threshold, int) and threshold == 0:
                return 0
    
            less_than_threshold = t_expand < threshold
      
            far_from_segment_ends = (segment_ends - t) > 1.01 * delta
            far_from_segment_ends = far_from_segment_ends.view(-1, 1, 1).repeat(1, trajectory.shape[1], trajectory.shape[2])
      
            losses_v = torch.square(vt - vr)
            losses_v = less_than_threshold * far_from_segment_ends * losses_v
            losses_v = reduce_op(losses_v.reshape(losses_v.shape[0], -1), dim=-1)
      
            return losses_v
    
        losses_v = masked_losses_v(vt, vr, boundary, segment_ends, t)

        loss = torch.mean(losses_f + alpha * losses_v)
        loss_dict = { 'bc_loss': 
                     loss.item(),}
        
        return loss, loss_dict
    