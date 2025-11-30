from typing import Dict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from EfficientFlow.model.common.normalizer import LinearNormalizer
from EfficientFlow.policy.base_image_policy import BaseImagePolicy
from EfficientFlow.model.diffusion.conditional_unet1d import ConditionalUnet1D
from EfficientFlow.model.diffusion.mask_generator import LowdimMaskGenerator
from EfficientFlow.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
try:
    import robomimic.models.base_nets as rmbn
    if not hasattr(rmbn, 'CropRandomizer'):
        raise ImportError("CropRandomizer is not in robomimic.models.base_nets")
except ImportError:
    import robomimic.models.obs_core as rmbn
import EfficientFlow.model.vision.crop_randomizer as dmvc
from EfficientFlow.common.pytorch_util import dict_apply, replace_submodules
from EfficientFlow.model.vision.rot_randomizer import RotRandomizer
from EfficientFlow.sde_lib import ConsistencyFM


class EfficientFlowNonEquiPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            rot_aug=False,
            eta=0.01,
            num_of_trajectories=5,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
        
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
       
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
        self.obs_as_global_cond = obs_as_global_cond
        self.rot_aug = rot_aug
        self.kwargs = kwargs

        
        self.num_inference_steps = num_inference_steps
        self.num_of_trajectories = num_of_trajectories
        self.predict_action_times_tlag=0

        self.eta = eta
        self.eps = 1e-2
        self.delta = 1e-2
        self.num_inference_step = 1
        self.num_of_trajectories=5

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    



    def predict_action(self, obs_dict: Dict[str, torch.Tensor],preaction=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
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
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

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
            
            pred_flat = self.model(z_flat, t_flat * 99, local_cond=local_cond_flat, global_cond=global_cond_flat)
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

        # get action
        self.compared_action_steps=self.n_action_steps
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        action_to_compare=action_pred[ : ,end-1:end-1+self.compared_action_steps]#7-11
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'action_to_compare': action_to_compare
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        # gt & noise
        target = nactions
        eps = self.eps

        delta  = self.delta

        
        

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

      
        vt = self.model(xt, t*99, cond=local_cond, global_cond=global_cond)
        vr = self.model(xr, r*99, local_cond=local_cond, global_cond=global_cond)
        # mask
        vt[condition_mask] = cond_data[condition_mask]
        vr[condition_mask] = cond_data[condition_mask]

        vr = torch.nan_to_num(vr)
      
        loss_flow = torch.nn.functional.mse_loss(vr,(target-a0)) * delta**2 
        loss_FABO = ((vt - vr).pow(2) * ((1.0 - t_expand) ** 2)).mean()

        loss= loss_flow * 2  +  loss_FABO 
        return loss