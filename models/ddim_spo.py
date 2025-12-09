import torch
from models.base_model import basemodel
import torch.cuda.amp as amp
from torch.functional import F
from torch.distributions import Normal
import time
import copy
from megatron_utils import mpu
import numpy as np
import utils.misc as utils
from tqdm import tqdm
from tqdm.auto import tqdm
import torch.distributed as dist
import pandas as pd
import wandb
import os
from petrel_client.client import Client
conf_path = '/mnt/shared-storage-user/xukaiyi/petrel-oss-python-sdk/petreloss.conf'
from einops import rearrange
import os
import torch
import numpy as np
import torchvision.utils as vutils
from utils.s3_client import s3_client
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import re
import io
import numpy as np


class spo(basemodel):
    def __init__(self, logger, **params) -> None:
        super().__init__(logger, **params)
        self.logger_print_time = False
        self.data_begin_time = time.time()
        self.diffusion_kwargs = params.get('diffusion_kwargs', {})


        ## init noise scheduler ##
        self.noise_scheduler_kwargs = self.diffusion_kwargs.get('noise_scheduler', {})
        self.noise_scheduler_type = list(self.noise_scheduler_kwargs.keys())[0]
        if self.noise_scheduler_type == 'DDPMScheduler':
            from src.diffusers import DDPMScheduler
            self.noise_scheduler = DDPMScheduler(**self.noise_scheduler_kwargs[self.noise_scheduler_type])
            num_train_timesteps = self.noise_scheduler_kwargs[self.noise_scheduler_type]['num_train_timesteps']
            self.noise_scheduler.set_timesteps(num_train_timesteps)
        elif self.noise_scheduler_type == 'DPMSolverMultistepScheduler':
            from src.diffusers import DPMSolverMultistepScheduler
            self.noise_scheduler = DPMSolverMultistepScheduler(**self.noise_scheduler_kwargs[self.noise_scheduler_type])
            num_train_timesteps = self.noise_scheduler_kwargs[self.noise_scheduler_type]['num_train_timesteps']
            self.noise_scheduler.set_timesteps(num_train_timesteps)
        else:
            raise NotImplementedError
        
        ## init noise scheduler for sampling ##
        self.sample_noise_scheduler_type = 'DDIMScheduler'
        if self.sample_noise_scheduler_type == 'DDIMScheduler':
            print("############# USING SAMPLER: DDIMScheduler #############")
            from src.diffusers import DDIMScheduler
            self.sample_noise_scheduler = DDIMScheduler(**self.noise_scheduler_kwargs[self.noise_scheduler_type])
            ## set num of inference
            self.sample_noise_scheduler.set_timesteps(20)
        elif self.sample_noise_scheduler_type == 'DDPMScheduler':
            print("############# USING SAMPLER: DDPMScheduler #############")
            from src.diffusers import DDPMScheduler
            self.sample_noise_scheduler = DDPMScheduler(**self.noise_scheduler_kwargs[self.noise_scheduler_type])
            self.sample_noise_scheduler.set_timesteps(1000)
        else:
            raise NotImplementedError

        ## important: scale the noise to get a reasonable noise process ##
        self.noise_scale = self.noise_scheduler_kwargs.get('noise_scale', 1.0)
        self.logger.info(f'####### noise scale: {self.noise_scale} ##########')
        self.s3_client = s3_client(bucket_name='zwl2', endpoint='http://10.135.0.241:80', user='zhangwenlong', jiqun = 'p') 
        self.client = Client(conf_path)

        ## scale factor ##
        self.scale_factor = 1.0 ## 1/std TODO: according to latent space
        self.logger.info(f'####### USE SCALE_FACTOR: {self.scale_factor} ##########')

        ## classifier free guidance ##
        self.classifier_free_guidance_kwargs = self.diffusion_kwargs.get('classifier_free_guidance', {})
        self.p_uncond = self.classifier_free_guidance_kwargs.get('p_uncond', 0.0)
        self.guidance_weight = self.classifier_free_guidance_kwargs.get('guidance_weight', 0.0)
        
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        self.threadPool = ThreadPoolExecutor(max_workers=16)
        print(params)
        self.beta_dpo = params['dpo']['beta_dpo']
        self.alpha = params['dpo']['alpha']
        self.logger.info(f'beta_dpo: {self.beta_dpo}')
        self.logger.info(f'alpha_dpo: {self.alpha}')
        self.update_interval = 378
        self.update_metric_dim = 378
        
        ## false表示优化far, true表示优化csi ##
        self.flag = False
        self.metric_flag = False
        self.ref_model = self.init_ref_model()
    
    @torch.no_grad()  
    def init_ref_model(self):
        main_key = list(self.model.keys())[0]                        # 主模型键名（UNet）
        ref_model = copy.deepcopy(self.model[main_key])         # 深拷贝UNet作为参考模型
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.to('cuda')
        return ref_model
            
    def data_preprocess(self, data):
        data_dict = {}
        inp_data = data['inputs'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        tar_data = data['data_samples'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        win_data = data['win_samples'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        lose_data = data['lose_samples'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        file_name = data['file_name']
        
        data_dict.update({'inputs': inp_data, 'data_samples': tar_data, 'win_data': win_data, 
                          'lose_data': lose_data, 'file_name': file_name})
        return data_dict
    
    @torch.no_grad()
    def denoise(self, template_data, cond_data, bs=1, vis=False, cfg=1, ensemble_member=1, seed = 0):
        """
        denoise from gaussian.
        """
        _, t, c, h, w = template_data.shape
        cond_data = cond_data[:bs, ...]
        generator = torch.Generator(device=template_data.device) #torch.manual_seed(0)
        generator.manual_seed(seed)
        latents = torch.randn(
            (bs*ensemble_member, t, c, h, w),
            generator=generator,
            device=template_data.device,
        ) 
        
        latents = latents * self.sample_noise_scheduler.init_noise_sigma
        model_kwargs = model_kwargs = dict(inp=cond_data, backbone_out=None)
        
        if cfg == 1:
            assert ensemble_member == 1
            ## iteratively denoise ##
            for t in tqdm(self.sample_noise_scheduler.timesteps) if (self.debug or vis) else self.sample_noise_scheduler.timesteps:
                ## predict the noise residual ##
                timestep = torch.ones((bs,), device=template_data.device) * t
                noise_pred = self.model[list(self.model.keys())[0]](latents, timestep, **model_kwargs)
                ## compute the previous noisy sample x_t -> x_{t-1} ##
                latents = self.sample_noise_scheduler.step(noise_pred, t, latents).prev_sample
            return latents
        else:
            print(f"guidance strength: {cfg}")
            ## for classifier free sampling ##
            cond_data = torch.cat([cond_data, torch.zeros_like(cond_data)])
            avg_latents = []
            for member in range(ensemble_member):
                member_latents = latents[member*bs:(member+1)*bs, ...]
                for t in tqdm(self.sample_noise_scheduler.timesteps) if (self.debug or vis) else self.sample_noise_scheduler.timesteps:
                    ## predict the noise residual ##
                    timestep = torch.ones((bs*2,), device=template_data.device) * t
                    latent_model_input = torch.cat([member_latents]*2)
                    latent_model_input = self.sample_noise_scheduler.scale_model_input(latent_model_input, t)
                    noise_pred = self.model[list(self.model.keys())[0]](x=latent_model_input, timesteps=timestep, cond=cond_data)
                    ########################
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg*(noise_pred_cond - noise_pred_uncond)
                    ## compute the previous noisy sample x_t -> x_{t-1} ##
                    member_latents = self.sample_noise_scheduler.step(noise_pred, t, member_latents).prev_sample
                avg_latents.append(member_latents)
            print('end sampling')
            avg_latents = torch.stack(avg_latents, dim=1)
            return avg_latents

    @torch.no_grad()
    def denoise_ref(self, template_data, cond_data, bs=1, vis=False, cfg=1, ensemble_member=1, seed = 0):
        """
        denoise from gaussian.
        """
        _, t, c, h, w = template_data.shape
        cond_data = cond_data[:bs, ...]
        generator = torch.Generator(device=template_data.device) #torch.manual_seed(0)
        generator.manual_seed(seed)
        latents = torch.randn(
            (bs*ensemble_member, t, c, h, w),
            generator=generator,
            device=template_data.device,
        ) 
        
        latents = latents * self.sample_noise_scheduler.init_noise_sigma
        model_kwargs = model_kwargs = dict(inp=cond_data, backbone_out=None)
        
        if cfg == 1:
            assert ensemble_member == 1
            ## iteratively denoise ##
            for t in tqdm(self.sample_noise_scheduler.timesteps) if (self.debug or vis) else self.sample_noise_scheduler.timesteps:
                ## predict the noise residual ##
                timestep = torch.ones((bs,), device=template_data.device) * t
                noise_pred = self.model[list(self.model.keys())[1]](latents, timestep, **model_kwargs)
                ## compute the previous noisy sample x_t -> x_{t-1} ##
                latents = self.sample_noise_scheduler.step(noise_pred, t, latents).prev_sample
            return latents
        else:
            print(f"guidance strength: {cfg}")
            ## for classifier free sampling ##
            cond_data = torch.cat([cond_data, torch.zeros_like(cond_data)])
            avg_latents = []
            for member in range(ensemble_member):
                member_latents = latents[member*bs:(member+1)*bs, ...]
                for t in tqdm(self.sample_noise_scheduler.timesteps) if (self.debug or vis) else self.sample_noise_scheduler.timesteps:
                    ## predict the noise residual ##
                    timestep = torch.ones((bs*2,), device=template_data.device) * t
                    latent_model_input = torch.cat([member_latents]*2)
                    latent_model_input = self.sample_noise_scheduler.scale_model_input(latent_model_input, t)
                    noise_pred = self.model[list(self.model.keys())[1]](x=latent_model_input, timesteps=timestep, cond=cond_data)
                    ########################
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg*(noise_pred_cond - noise_pred_uncond)
                    ## compute the previous noisy sample x_t -> x_{t-1} ##
                    member_latents = self.sample_noise_scheduler.step(noise_pred, t, member_latents).prev_sample
                avg_latents.append(member_latents)
            print('end sampling')
            avg_latents = torch.stack(avg_latents, dim=1)
            return avg_latents

    @torch.no_grad()
    def denoise_ref2(self, template_data, cond_data, bs=1, vis=False, cfg=1, ensemble_member=1, seed = 0):
        """
        denoise from gaussian.
        """
        _, t, c, h, w = template_data.shape
        cond_data = cond_data[:bs, ...]
        generator = torch.Generator(device=template_data.device) #torch.manual_seed(0)
        generator.manual_seed(seed)
        latents = torch.randn(
            (bs*ensemble_member, t, c, h, w),
            generator=generator,
            device=template_data.device,
        ) 
        
        latents = latents * self.sample_noise_scheduler.init_noise_sigma
        model_kwargs = model_kwargs = dict(inp=cond_data, backbone_out=None)
        
        if cfg == 1:
            assert ensemble_member == 1
            ## iteratively denoise ##
            for t in tqdm(self.sample_noise_scheduler.timesteps) if (self.debug or vis) else self.sample_noise_scheduler.timesteps:
                ## predict the noise residual ##
                timestep = torch.ones((bs,), device=template_data.device) * t
                noise_pred = self.model[list(self.model.keys())[2]](latents, timestep, **model_kwargs)
                ## compute the previous noisy sample x_t -> x_{t-1} ##
                latents = self.sample_noise_scheduler.step(noise_pred, t, latents).prev_sample
            return latents
        else:
            print(f"guidance strength: {cfg}")
            ## for classifier free sampling ##
            cond_data = torch.cat([cond_data, torch.zeros_like(cond_data)])
            avg_latents = []
            for member in range(ensemble_member):
                member_latents = latents[member*bs:(member+1)*bs, ...]
                for t in tqdm(self.sample_noise_scheduler.timesteps) if (self.debug or vis) else self.sample_noise_scheduler.timesteps:
                    ## predict the noise residual ##
                    timestep = torch.ones((bs*2,), device=template_data.device) * t
                    latent_model_input = torch.cat([member_latents]*2)
                    latent_model_input = self.sample_noise_scheduler.scale_model_input(latent_model_input, t)
                    noise_pred = self.model[list(self.model.keys())[2]](x=latent_model_input, timesteps=timestep, cond=cond_data)
                    ########################
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg*(noise_pred_cond - noise_pred_uncond)
                    ## compute the previous noisy sample x_t -> x_{t-1} ##
                    member_latents = self.sample_noise_scheduler.step(noise_pred, t, member_latents).prev_sample
                avg_latents.append(member_latents)
            print('end sampling')
            avg_latents = torch.stack(avg_latents, dim=1)
            return avg_latents


    @torch.no_grad()
    def denoise_ref3(self, template_data, cond_data, bs=1, vis=False, cfg=1, ensemble_member=1, seed = 0):
        """
        denoise from gaussian.
        """
        _, t, c, h, w = template_data.shape
        cond_data = cond_data[:bs, ...]
        generator = torch.Generator(device=template_data.device) #torch.manual_seed(0)
        generator.manual_seed(seed)
        latents = torch.randn(
            (bs*ensemble_member, t, c, h, w),
            generator=generator,
            device=template_data.device,
        ) 
        
        latents = latents * self.sample_noise_scheduler.init_noise_sigma
        model_kwargs = model_kwargs = dict(inp=cond_data, backbone_out=None)
        
        if cfg == 1:
            assert ensemble_member == 1
            ## iteratively denoise ##
            for t in tqdm(self.sample_noise_scheduler.timesteps) if (self.debug or vis) else self.sample_noise_scheduler.timesteps:
                ## predict the noise residual ##
                timestep = torch.ones((bs,), device=template_data.device) * t
                noise_pred = self.ref_model(latents, timestep, **model_kwargs)
                ## compute the previous noisy sample x_t -> x_{t-1} ##
                latents = self.sample_noise_scheduler.step(noise_pred, t, latents).prev_sample
            return latents
        else:
            print(f"guidance strength: {cfg}")
            ## for classifier free sampling ##
            cond_data = torch.cat([cond_data, torch.zeros_like(cond_data)])
            avg_latents = []
            for member in range(ensemble_member):
                member_latents = latents[member*bs:(member+1)*bs, ...]
                for t in tqdm(self.sample_noise_scheduler.timesteps) if (self.debug or vis) else self.sample_noise_scheduler.timesteps:
                    ## predict the noise residual ##
                    timestep = torch.ones((bs*2,), device=template_data.device) * t
                    latent_model_input = torch.cat([member_latents]*2)
                    latent_model_input = self.sample_noise_scheduler.scale_model_input(latent_model_input, t)
                    noise_pred = self.ref_model(x=latent_model_input, timesteps=timestep, cond=cond_data)
                    ########################
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg*(noise_pred_cond - noise_pred_uncond)
                    ## compute the previous noisy sample x_t -> x_{t-1} ##
                    member_latents = self.sample_noise_scheduler.step(noise_pred, t, member_latents).prev_sample
                avg_latents.append(member_latents)
            print('end sampling')
            avg_latents = torch.stack(avg_latents, dim=1)
            return avg_latents
        
    @torch.no_grad()
    def encode_stage(self, x):
        if utils.get_world_size() == 1 :
            z = self.model[list(self.model.keys())[1]].net.encode(x)
        else:
            z = self.model[list(self.model.keys())[1]].module.net.encode(x)
        return z.sample() * self.scale_factor

    @torch.no_grad()
    def decode_stage(self, z):
        z = z/self.scale_factor
        if utils.get_world_size() == 1 :
            z = self.model[list(self.model.keys())[1]].net.decode(z)
        else:
            z = self.model[list(self.model.keys())[1]].module.net.decode(z)
        return z

    @torch.no_grad()
    def init_scale_factor(self, z_tar):
        del self.scale_factor
        self.logger.info("### USING STD-RESCALING ###")
        _std = z_tar.std()
        if utils.get_world_size() == 1 :
            pass
        else:
            dist.barrier()
            dist.all_reduce(_std)
            _std = _std / dist.get_world_size()
        scale_factor = 1/_std
        self.logger.info(f'####### scale factor: {scale_factor.item()} ##########')
        self.register_buffer('scale_factor', scale_factor)

    def loss_dpo(self, pred, ref_pred, tar, time):
        dt = 1
        eps = 1e-6
        alpha_t = self.noise_scheduler.cal_sqrt_alpha_prod(pred, time) ** 2 / (self.noise_scheduler.cal_sqrt_alpha_prod(pred, time-dt) ** 2 + eps)
        wt = 0.5*(1/alpha_t)*(1-alpha_t)/ (1-self.noise_scheduler.cal_sqrt_alpha_prod(pred, time-dt)**2)
        
        terms = {}
        terms['train_pred'] = pred
        terms['ref_pred'] = ref_pred
        
        terms['train_loss'] = (pred - tar).pow(2).mean(dim=[1,2,3,4])
        terms['ref_loss'] = (ref_pred - tar).pow(2).mean(dim=[1,2,3,4])
        terms['w_loss'],terms['l_loss'] = terms['train_loss'].chunk(2)
        terms['ref_w_loss'],terms['ref_l_loss'] = terms['ref_loss'].chunk(2)
        terms['w_t'] = wt

        return terms
    
    def train_one_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp, tar, win, lose, file_name = data_dict['inputs'], data_dict['data_samples'],  data_dict['win_data'], data_dict['lose_data'], data_dict['file_name']
        # self.visualizer.save_pixel_image_rl(win_image=win, lose_image=lose, target_img = tar, step = step, file_name = file_name)
        # file_name = data_dict['file_name']
        b, t, c, h, w = tar.shape
        ## dpo_input ##
        inp_dpo = inp.repeat(2, 1, 1, 1, 1)
        # dpo_output ##
        tar_dpo = torch.cat((win, lose), dim=0)
        # dpo noise ##
        noise = torch.randn_like(tar)
        noise_dpo = noise.repeat(2, 1, 1, 1, 1)
        # dpo time ##
        bs = inp.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=inp.device)
        timesteps_dpo = timesteps.repeat(2)
        # dpo_add_noise ##
        noisy_tar_dpo = self.noise_scheduler.add_noise(tar_dpo, noise_dpo, timesteps_dpo)
        
        ## predict the noise ##
        model_kwargs = dict(inp=inp_dpo, backbone_out=None)
        
        noise_pred = self.model[list(self.model.keys())[0]](noisy_tar_dpo, timesteps_dpo, **model_kwargs)
        with torch.no_grad():
            far_noise_pred = self.model[list(self.model.keys())[1]](noisy_tar_dpo, timesteps_dpo, **model_kwargs)

        with torch.no_grad():
            ref_noise_pred = self.model[list(self.model.keys())[2]](noisy_tar_dpo, timesteps_dpo, **model_kwargs)
            
        # compute loss ##
        prob_loss_st2_dict = self.loss_dpo(noise_pred, far_noise_pred, noise_dpo, timesteps) ## important: rescale the loss
        w_loss_st2, l_loss_st2, wt_st2, ref_w_loss_st2, ref_l_loss_st2 = prob_loss_st2_dict['w_loss'], prob_loss_st2_dict['l_loss'], prob_loss_st2_dict['w_t'], prob_loss_st2_dict['ref_w_loss'], prob_loss_st2_dict['ref_l_loss']
        
        prob_loss_st1_dict = self.loss_dpo(far_noise_pred, ref_noise_pred, noise_dpo, timesteps)
        w_loss_st1, l_loss_st1, wt_st1, ref_w_loss_st1, ref_l_loss_st1 = prob_loss_st1_dict['w_loss'], prob_loss_st1_dict['l_loss'], prob_loss_st1_dict['w_t'], prob_loss_st1_dict['ref_w_loss'], prob_loss_st1_dict['ref_l_loss']
        
        model_diff_st2 = w_loss_st2 - l_loss_st2
        model_w_loss_st2 = w_loss_st2.mean()
        model_l_loss_st2 = l_loss_st2.mean()
        ref_model_w_loss_st2 = ref_w_loss_st2.mean()
        ref_model_l_loss_st2 = ref_l_loss_st2.mean()
        ref_model_diff_st2 = ref_w_loss_st2 - ref_l_loss_st2
        
        loss_st2 = model_diff_st2-ref_model_diff_st2
        

        model_diff_st1 = w_loss_st1 - l_loss_st1
        model_w_loss_st1 = w_loss_st1.mean()
        model_l_loss_st1 = l_loss_st1.mean()
        ref_model_w_loss_st1 = ref_w_loss_st1.mean()
        ref_model_l_loss_st1 = ref_l_loss_st1.mean()
        ref_model_diff_st1 = ref_w_loss_st1 - ref_l_loss_st1
        
        loss_st1 = model_diff_st1 - ref_model_diff_st1

        difloss = loss_st2
        loss = -torch.mean(F.logsigmoid(-self.beta_dpo*self.alpha*wt_st2*loss_st2 + self.beta_dpo*wt_st1*loss_st1))
        inter = torch.mean(difloss)
        
        
        ## update params of diffusion model ##
        loss.backward()
        self.optimizer[list(self.model.keys())[0]].step()
        self.optimizer[list(self.model.keys())[0]].zero_grad()

        if (step) % 1 ==0:
            z_sample_prediction = self.denoise(template_data=tar, cond_data=inp, bs=bs)
            z_sample_prediction_ref = self.denoise_ref(template_data=tar, cond_data=inp, bs=bs)
            # self.visualizer.save_pixel_image_rl_2(pred=z_sample_prediction, win_image=win, lose_image=lose, target_img = tar, target_ref = z_sample_prediction_ref, step = step, file_name=file_name)

            result_win = self.eval_metrics.update_single_sample(
                target=tar[0].unsqueeze(0),             
                pred=win[0].unsqueeze(0),   
                sample_names='win'
            )
            metrics_win = self.eval_metrics.compute_frame(result_win)
            
            result_lose = self.eval_metrics.update_single_sample(
                target=tar[0].unsqueeze(0),             
                pred=lose[0].unsqueeze(0),
                sample_names='lose'
            )
            metrics_lose = self.eval_metrics.compute_frame(result_lose)
            
            # z_sample_prediction = self.denoise(template_data=tar, cond_data=inp, bs=bs)

            result_pred = self.eval_metrics.update_single_sample(
                target=tar[0].unsqueeze(0),             
                pred=z_sample_prediction[0].unsqueeze(0),
                sample_names='pred'
            )
            
            metrics_pred = self.eval_metrics.compute_frame(result_pred)

            result_pred_ref = self.eval_metrics.update_single_sample(
                target=tar[0].unsqueeze(0),             
                pred=z_sample_prediction_ref[0].unsqueeze(0),
                sample_names='pred'
            )
            
            metrics_pred_ref = self.eval_metrics.compute_frame(result_pred_ref)

            self.logger.info(f"[Best    ] FAR: {metrics_win['avg']['far']:.4f} | CSI: {metrics_win['avg']['csi']:.4f}")
            self.logger.info(f"[Worst   ] FAR: {metrics_lose['avg']['far']:.4f} | CSI: {metrics_lose['avg']['csi']:.4f}")
            self.logger.info(f"[Pred    ] FAR: {metrics_pred['avg']['far']:.4f} | CSI: {metrics_pred['avg']['csi']:.4f}")
            self.logger.info(f"[PredRef ] FAR: {metrics_pred_ref['avg']['far']:.4f} | CSI: {metrics_pred_ref['avg']['csi']:.4f}")
       
        else:
            pass
        return {'model_w_loss': model_w_loss_st2.item(), 'model_l_loss': model_l_loss_st2.item(), 'ref_model_w_loss': ref_model_w_loss_st2.item(), 'ref_model_l_loss': ref_model_l_loss_st2.item(), 'total_loss': loss.item(), 'inter': inter.item()}


    @torch.no_grad()
    def test_one_step(self, batch_data, epoch, idx):
        data_dict = self.data_preprocess(batch_data)
        inp, tar, win, lose, file_name = data_dict['inputs'], data_dict['data_samples'],  data_dict['win_data'], data_dict['lose_data'], data_dict['file_name']

        b, t, c, h, w = tar.shape
        
        ## sample noise to add ##
        noise = torch.randn_like(tar)
        ## sample random timestep for each ##
        bs = inp.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=inp.device)
        noisy_tar = self.noise_scheduler.add_noise(tar, noise, timesteps)

        ## predict the noise residual ##
        model_kwargs = model_kwargs = dict(inp=inp, backbone_out=None)
        noise_pred = self.model[list(self.model.keys())[0]](noisy_tar, timesteps, **model_kwargs)

        ## denoise ##
        z_sample_prediction = self.denoise(template_data=tar, cond_data=inp, bs=bs)
        
        loss_records = {}
        ## evaluate other metrics ##
        data_dict = {}
        data_dict['gt'] = tar
        # print('-------------gt shape--------------', data_dict['gt'].shape)
        data_dict['pred'] = z_sample_prediction
        # print('-------------pred shape--------------', data_dict['pred'].shape)
        
        MSE_loss = torch.mean((z_sample_prediction - tar) ** 2).item()
        loss = self.loss(noise_pred, noise) ## important: rescale the loss
        
        ## evaluation ##

        data_dict['gt'] = data_dict['gt']
        data_dict['pred'] = data_dict['pred']

        ### compute meteorology metrics ###
        self.eval_metrics.update(target=data_dict['gt'], pred=data_dict['pred'])

        ### compute loss function ###
        loss_records.update({'MSE': torch.mean((data_dict['pred'] - data_dict['gt']) ** 2).item()})
        loss_records.update({'MAE': torch.mean(torch.abs(data_dict['pred'] - data_dict['gt'])).item()})
        crps_dict = self.eval_metrics.get_crps(target=data_dict['gt'], pred=data_dict['pred'])
        loss_records.update(crps_dict)
        ### compute image quality ###
        iq_dict = self.eval_metrics.get_single_frame_metrics(target=data_dict['gt'], pred=data_dict['pred']) ## get SSIM and PSNR
        loss_records.update(iq_dict)
        
        ### get csi preference and far and \pai_0 model results
        z_sample_prediction = self.denoise(template_data=tar, cond_data=inp, bs=bs)
        z_sample_prediction_far = self.denoise_ref(template_data=tar, cond_data=inp, bs=bs)
        z_sample_prediction_ref = self.denoise_ref2(template_data=tar, cond_data=inp, bs=bs)
        # self.visualizer.save_pixel_image_6methods(vis_args, step = epoch, batchidx=idx, file_name=file_name)
        # self.visualizer.save_pred_gt_images(z_sample_prediction, tar,  file_name=file_name)
        # self.visualizer.save_input_images(inp,step = epoch, batchidx=idx, file_name=file_name)
        ### get metric
        result_pred = self.eval_metrics.update_single_sample(
            target=tar[0].unsqueeze(0),             
            pred=z_sample_prediction[0].unsqueeze(0),
            sample_names='pred'
        )
        metrics_pred = self.eval_metrics.compute_frame(result_pred)
        csi1 = metrics_pred['avg']['csi']
        far1 = metrics_pred['avg']['far']

        result_far = self.eval_metrics.update_single_sample(
            target=tar[0].unsqueeze(0),             
            pred=z_sample_prediction_far[0].unsqueeze(0),
            sample_names='pred'
        )
        metrics_far = self.eval_metrics.compute_frame(result_far)
        csi2 = metrics_far['avg']['csi']
        far2 = metrics_far['avg']['far']

        result_pred_ref = self.eval_metrics.update_single_sample(
            target=tar[0].unsqueeze(0),             
            pred=z_sample_prediction_ref[0].unsqueeze(0),
            sample_names='pred_ref'
        )
        metrics_pred_ref = self.eval_metrics.compute_frame(result_pred_ref)
        csi3 = metrics_pred_ref['avg']['csi']
        far3 = metrics_pred_ref['avg']['far']

        result_win = self.eval_metrics.update_single_sample(
            target=tar[0].unsqueeze(0),             
            pred=win[0].unsqueeze(0),
            sample_names='pred_ref'
        )
        metrics_win = self.eval_metrics.compute_frame(result_win)
        csi4 = metrics_win['avg']['csi']
        far4 = metrics_win['avg']['far']
        
        result_lose = self.eval_metrics.update_single_sample(
            target=tar[0].unsqueeze(0),             
            pred=lose[0].unsqueeze(0),
            sample_names='pred_ref'
        )
        metrics_lose = self.eval_metrics.compute_frame(result_lose)
        csi5 = metrics_lose['avg']['csi']
        far5 = metrics_lose['avg']['far']
        
        
        vis_args = {"tar": tar,
                    "pred": z_sample_prediction,
                    "far": z_sample_prediction_far,
                    "ref": z_sample_prediction_ref,
                    "win": win,
                    "lose": lose,
                    "csi_pred": csi1,
                    "far_pred": far1,
                    "csi_far": csi2,
                    "far_far": far2,
                    "csi_ref": csi3,
                    "far_ref": far3,
                    "csi_win": csi4,
                    "far_win": csi4,
                    "csi_lose": csi5,
                    "far_lose": far5
                    }
        # if csi1 > csi3 + 0.01:
        self.visualizer.save_pixel_image_6methods(vis_args, step = epoch, batchidx=idx, file_name=file_name)
        
        return loss_records
    
    @torch.no_grad()
    def test(self, test_data_loader, epoch, step, state='Train'):
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)

        # set model to eval
        for key in self.model:
            self.model[key].eval()
            
        data_loader = test_data_loader
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Validating (Epoch {epoch})")

        for batch_idx, batch in progress_bar:
            if self.debug and step >= 2:
                break

            loss = self.test_one_step(batch, step, batch_idx)
            metric_logger.update(**loss)

            # 显示进度条状态
            bar_status = ', '.join([f"{k}: {v:.4f}" for k, v in loss.items()])
            progress_bar.set_postfix_str(bar_status)

        # compute meteorologic metrics
        losses = {}
        metrics = self.eval_metrics.compute()
        for thr, thr_dict in metrics.items():
            for k, v in thr_dict.items():
                losses.update({f'{thr}-{k}': v})
        self.eval_metrics.reset()

        metric_logger.update(**losses)

        self.logger.info('  '.join(
            [f'Epoch [{epoch + 1}](val stats)', "{meters}"]).format(
            meters=str(metric_logger)
        ))

        return metric_logger

    def save_data_preprocess(self, data):
        data_dict = {}
        inp_data = data['inputs'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        tar_data = data['data_samples'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        file_names = data['file_name']
        dataset_name = data['dataset_name'][0]
        data_dict.update({'inputs': inp_data, 'data_samples': tar_data, 'file_name': file_names, 
                          'dataset_name': dataset_name})
        return data_dict
    
    @torch.no_grad()
    def save_sample(self, dataloader, split, start_step=0):
        # 初始化线程池和metrics缓存
        self.threadPool = ThreadPoolExecutor(max_workers=16)
        self.all_metrics = []  # 用于缓存所有metrics结果
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        
        try:
            for step, batch in enumerate(dataloader):
                if step < start_step:
                    continue
                # 数据预处理
                data_dict = {}
                data_dict = self.save_data_preprocess(batch)
                inp, tar = data_dict['inputs'], data_dict['data_samples']
                file_names = data_dict['file_name']
                all_predictions = []
                bs = inp.shape[0]
                for sample_num in range(5):

                    seed = random.randint(0, 10000) 
                    z_sample_prediction = self.denoise(template_data=tar, cond_data=inp, bs=bs, seed=seed)

                    loss_records = {}
                    data_dict['gt'] = tar
                    data_dict['pred'] = z_sample_prediction
                
                    prediction = z_sample_prediction
                    all_predictions.append(prediction)
                    
                    sample_file_name = [f'sample{sample_num}_' + item for item in file_names]
                    self.save_pred_data(batch_data=prediction, file_names=sample_file_name, 
                                    dataset_name=data_dict['dataset_name'])
                    
                    result = self.eval_metrics.update_sample(
                        target=data_dict['gt'], 
                        pred=data_dict['pred'], 
                        sample_names=sample_file_name
                    )
                    metrics = self.eval_metrics.compute_sample(result)
                    self.all_metrics.append(metrics)
                    if step % 500 == 0:
                        self.visualizer.save_pixel_image(pred_image=z_sample_prediction, target_img=tar, step=step)
                
                sample_5 = torch.stack(all_predictions, dim=0).mean(dim=0)
                sample_5_file_name = ["sample5_" + item for item in file_names]
                gt_file_name = ["gt_" + item for item in file_names]
                
                data_dict['gt'] = tar
                data_dict['pred'] = sample_5
                result_sample_5 = self.eval_metrics.update_sample(
                    target=data_dict['gt'], 
                    pred=data_dict['pred'], 
                    sample_names=sample_5_file_name
                )
                metrics_sample_5 = self.eval_metrics.compute_sample(result_sample_5)
                self.all_metrics.append(metrics_sample_5)
                
                # 保存sample_5的结果
                self.save_pred_data(batch_data=sample_5, file_names=sample_5_file_name, 
                                    dataset_name=data_dict['dataset_name'])
                # 保存gt的结果
                self.save_pred_data(batch_data=tar, file_names=gt_file_name, 
                                    dataset_name=data_dict['dataset_name'])
                    
            self._flush_metrics_to_excel()  
            
        except Exception as e:
            self.logger.error(f"Error in save_sample: {str(e)}")
            raise
        finally:
            # 确保最后所有metrics都被写入
            self._flush_metrics_to_excel()
            # 关闭线程池
            self.threadPool.shutdown(wait=True)
        
        return None


    def _flush_metrics_to_excel(self, excel_path='/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/ddpmresults.xlsx'):
        if not hasattr(self, 'all_metrics') or not self.all_metrics:
            return
        
        try:
            # 先按文件名和sample编号排序
            all_metrics_sorted = []
            
            # 收集所有文件名（不包含sample前缀）
            base_filenames = set()
            for metrics in self.all_metrics:
                for sample_name in metrics.keys():
                    # 提取基础文件名（去掉sampleX_前缀）
                    base_name = '_'.join(sample_name.split('_')[1:])
                    base_filenames.add(base_name)
            
            # 按基础文件名和sample编号排序
            for base_name in sorted(base_filenames):
                for sample_num in range(6):  # sample0到sample5
                    sample_prefix = f"sample{sample_num}_"
                    full_name = sample_prefix + base_name
                    
                    # 在所有metrics中查找匹配的结果
                    for metrics in self.all_metrics:
                        if full_name in metrics:
                            row = {'Sample': full_name}
                            sample_data = metrics[full_name]
                            
                            row.update({
                                'MAE': sample_data.get('mae', None),
                                'MSE': sample_data.get('mse', None)
                            })
                            
                            # 添加阈值相关指标
                            for threshold, values in sample_data.items():
                                # print(threshold)
                                if threshold == 'avg':
                                    for metric_name, metric_value in values.items():
                                        row[f'{threshold}_{metric_name}'] = metric_value
                                if threshold not in ['mae', 'mse']:
                                    for metric_name, metric_value in values.items():
                                        row[f'{threshold}_{metric_name}'] = metric_value
                            
                            all_metrics_sorted.append(row)
            
            # 写入Excel
            df = pd.DataFrame(all_metrics_sorted)
            if os.path.exists(excel_path):
                existing_df = pd.read_excel(excel_path)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_excel(excel_path, index=False, engine='openpyxl')
            self.logger.info(f"成功写入 {len(all_metrics_sorted)} 条排序后的metrics到 {excel_path}")
            
            # 清空缓存
            self.all_metrics = []
            
        except Exception as e:
            self.logger.error(f"写入Excel失败: {str(e)}")
            # 保留缓存以便下次尝试写入
            
    @torch.no_grad()
    def save_pred_data(self, batch_data, file_names, dataset_name):
        root_dir = f'rankcast/val_data/{dataset_name}/ddpmcast'
        model_name = list(self.model.keys())[0]
        dataset_name = dataset_name
        if model_name == 'SimVP_Recur':
            ## figure out TAU or IncepU
            if utils.get_world_size() > 1:
                model_name = self.model[model_name].module.net.model_type
            else:
                model_name = self.model[model_name].net.model_type
        
        prefix_root = os.path.join(root_dir, model_name)
        bs = batch_data.shape[0]
        for i in range(bs):
            save_data = batch_data[i].cpu().numpy()
            save_file_name = f'{prefix_root}_fin/{file_names[i]}' 
            local_path = os.path.join('/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/temp', file_names[i])
            ### save ###
            # import pdb; pdb.set_trace()
            self.threadPool.submit(self.write_data, save_file_name, local_path, save_data)
            # self.uploader(save_data, save_file_name)

        return None

    @torch.no_grad()
    def save_win_lose_data(self, batch_data, file_names, dataset_name):
        root_dir = f'rankcast/val_data/{dataset_name}/ddpmcast'
        model_name = list(self.model.keys())[0]
        dataset_name = dataset_name
        if model_name == 'SimVP_Recur':
            ## figure out TAU or IncepU
            if utils.get_world_size() > 1:
                model_name = self.model[model_name].module.net.model_type
            else:
                model_name = self.model[model_name].net.model_type
        
        prefix_root = os.path.join(root_dir, model_name)

        save_data = batch_data.cpu().numpy()
        save_file_name = f'{prefix_root}_fin/{file_names}' 
        local_path = os.path.join('/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/temp', file_names)
        self.threadPool.submit(self.write_data, save_file_name, local_path, save_data)
        return None
    
    @torch.no_grad() 
    def write_data(self, save_file_name, local_path1, data1):
        np.save(local_path1, data1)
        print(local_path1)
        # print(save_file_name) 
        # p_s3_client = s3_client(bucket_name='zwl2', endpoint='http://10.135.0.241:80', user='zhangwenlong', jiqun = 'p') 
        # res = p_s3_client.upload_file(f'{local_path1}', 'zwl2', save_file_name)
        # os.remove(local_path1)
        
    @torch.no_grad()
    def eval_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples']
        original_tar = data_dict['original']
        b, t, c, h, w = tar.shape
        ## inp is coarse prediction in latent space, tar is gt in latent space
        z_tar = tar
        z_coarse_prediction = inp
        ## scale ##
        z_tar = z_tar * self.scale_factor
        z_coarse_prediction = z_coarse_prediction * self.scale_factor
        ## sample image ##
        losses = {}
        z_sample_prediction = self.denoise(template_data=z_tar, cond_data=z_coarse_prediction, bs=tar.shape[0], vis=True, cfg=self.cfg_weight, ensemble_member=self.ens_member)
        len_shape_prediction = len(z_sample_prediction.shape)
        assert len_shape_prediction == 6
        n = z_sample_prediction.shape[1]
        sample_predictions = []
        for i in range(n):
            member_z_sample_prediction = z_sample_prediction[:, i, ...]
            member_z_sample_prediction = rearrange(member_z_sample_prediction, 'b t c h w -> (b t) c h w').contiguous()
            member_sample_prediction = self.decode_stage(member_z_sample_prediction)
            member_sample_prediction = rearrange(member_sample_prediction, '(b t) c h w -> b t c h w', t=t)
            sample_predictions.append(member_sample_prediction) 
        sample_predictions = torch.stack(sample_predictions, dim=1)
        ## evaluate other metrics ##
        data_dict = {}
        if self.metrics_type == 'SEVIRSkillScore':
            data_dict['gt'] =  original_tar
            data_dict['pred'] = sample_predictions.mean(dim=1)
            self.eval_metrics.update(target=data_dict['gt'], pred=data_dict['pred'])
            ############
            sf_dict = self.eval_metrics.get_single_frame_metrics(target=data_dict['gt'], pred=data_dict['pred'])
            crps_dict = self.eval_metrics.get_crps(target=data_dict['gt'], pred=sample_predictions)
            losses.update(sf_dict)
            losses.update(crps_dict)
        else:
            pass
            ############
        ## save image ##
        if self.visualizer_type == 'sevir_visualizer' and (step) % 2 == 0:
            self.visualizer.save_pixel_image(pred_image=data_dict['pred'], target_img=data_dict['gt'], step=step)
        else:
            pass
        
        return losses
    
    @torch.no_grad()
    def test_final(self, test_data_loader, predict_length):
        self.test_data_loader = test_data_loader
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        if self.metrics_type == 'SEVIRSkillScore':
            self.scale_factor = 0.6786020398139954
        else:
            raise NotImplementedError
        # set model to eval
        for key in self.model:
            self.model[key].eval()

        if test_data_loader is not None:
            data_loader = test_data_loader
        else:
            raise ValueError("test_data_loader is None")

        from megatron_utils.tensor_parallel.data import get_data_loader_length
        total_step = get_data_loader_length(test_data_loader)

        # from utils.metrics import cal_FVD
        # self.fvd_computer = cal_FVD(use_gpu=True)

        for step, batch in enumerate(data_loader):
            if isinstance(batch, int):
                batch = None
            losses = self.eval_step(batch_data=batch, step=step)
            metric_logger.update(**losses)

            self.logger.info("#"*80)
            self.logger.info(step)
            if step % 10 == 0 or step == total_step-1:
                self.logger.info('  '.join(
                [f'Step [{step + 1}](val stats)',
                 "{meters}"]).format(
                    meters=str(metric_logger)
                 ))
        metrics = self.eval_metrics.compute()
        for thr, thr_dict in metrics.items():
            for k, v in thr_dict.items():
                losses.update({f'{thr}-{k}': v})
        ###################################################
        try:
            metric_logger.update(**losses)
            self.logger.info('final results: {meters}'.format(meters=str(metric_logger)))
        except:
            ## save as excel ##
            import pandas as pd
            df = pd.DataFrame.from_dict(losses)
            df.to_excel(f'{self.visualizer.exp_dir}/{self.visualizer.sub_dir}_losses.xlsx')
        return None

    def get_win_loss_frame_samples(self):
        self.all_metrics =[]
        path = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/test/ddpm_val.txt'
        with open(path, 'r') as f:
            files = [line.split(' ')[-1].strip() for line in f.readlines()]
        
        idx = 0
        # file可以只选择sample0
        
        for file in files:
            idx = idx + 1
            sample_data = []
            for i in range(6):
                datafile = os.path.join('cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast/diffcast_fin', file)
                datafile = datafile.replace('sample0',f'sample{i}')
                data = self.client.get(datafile)
                data_file = io.BytesIO(data)
                data = np.load(data_file, allow_pickle=True)
                sample_data.append(data)
                
            name = file.split('/')[-1].replace('.npy','')
            gt_file = re.sub(r'sample\d+_', 'gt_', datafile)
            gt = self.client.get(gt_file)
            gt = io.BytesIO(gt)
            gt = np.load(gt, allow_pickle=True)

            # 初始化保存 best / worst 的信息
            best_samples = []
            worst_samples = []

            # 遍历每一帧，计算 avg CSI，挑选最优最劣的 sample
            for frame in range(18):
                best_csi = -float('inf')
                worst_csi = float('inf')
                best_idx = -1
                worst_idx = -1

                for idx in range(5):  # 遍历 5 个 sample
                    result_patch = self.eval_metrics.update_frame(
                        target=gt[frame, :, :, :],               # shape: (1, 128, 128)
                        pred=sample_data[idx][frame, :, :, :],   # shape: (1, 128, 128)
                        sample_names=name
                    )
                    metrics = self.eval_metrics.compute_frame(result_patch)
                    avg_csi = metrics['avg']['csi']

                    if avg_csi > best_csi:
                        best_csi = avg_csi
                        best_idx = idx

                    if avg_csi < worst_csi:
                        worst_csi = avg_csi
                        worst_idx = idx

                # 保存每帧中最佳和最差 sample 的信息
                best_samples.append((frame, best_idx, best_csi))
                worst_samples.append((frame, worst_idx, worst_csi))

            # 合成完整的 best 和 worst sample 序列
            best_sequence = []
            worst_sequence = []

            for frame, best_idx, _ in best_samples:
                best_sequence.append(sample_data[best_idx][frame])  # shape: (1, 128, 128)

            for frame, worst_idx, _ in worst_samples:
                worst_sequence.append(sample_data[worst_idx][frame])  # shape: (1, 128, 128)

            best_sequence = np.stack(best_sequence, axis=0)
            worst_sequence = np.stack(worst_sequence, axis=0)

            best_sequence = torch.from_numpy(best_sequence)
            worst_sequence = torch.from_numpy(worst_sequence)
            
            win_name = 'frame_win_' + name + '.npy'
            self.save_win_lose_data(batch_data=best_sequence, file_names=win_name, 
                        dataset_name='sevir_128_10m_3h')
            lose_name = 'frame_lose_' + name + '.npy'
            self.save_win_lose_data(batch_data=worst_sequence, file_names=lose_name, 
                        dataset_name='sevir_128_10m_3h')
            