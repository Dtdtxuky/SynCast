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
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import pywt
from collections import defaultdict

        
class diffusion_model(basemodel):
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
        
        # 初始化线程池和metrics缓存
        self.threadPool = ThreadPoolExecutor(max_workers=16)
        
        ## guidance ##
        self.guidance_kwargs = self.diffusion_kwargs.get('guidance', {})
        self.type = self.guidance_kwargs.get('type', None)
        self.guidance_scale = self.guidance_kwargs.get('strength', 0.005)
        self.logger.info(f'type: {self.type}')
        self.logger.info(f'strength: {self.guidance_scale}')
        
    def data_preprocess(self, data):
        data_dict = {}
        inp_data = data['inputs'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        tar_data = data['data_samples'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        file_name = data['file_name']
        # original_tar = data['data_samples']['original'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        data_dict.update({'inputs': inp_data, 'data_samples': tar_data,'file_name': file_name})
        return data_dict
    
    @torch.no_grad()
    def denoise(self, template_data, cond_data, bs=1, vis=False, cfg=1, ensemble_member=1, seed = 0, tar = None):
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
                latents = self.sample_noise_scheduler.step(noise_pred, t, latents).prev_sample # 这里的latent就是最后的result，可以直接可视化
    
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


    def dps_guidance_denoise(self, template_data, cond_data, bs, guide_cond=None, mask=None, return_intermediate=False, save_interval=100, seed=0):
        """
        与原 denoise 类似，但增加对 gt 的测量引导 (梯度修正)。
        """
        
        # 1) 准备随机噪声作为初始 latent
        _, t, c, h, w = template_data.shape
        cond_data = cond_data[:bs, ...]
        generator = torch.Generator(device=template_data.device) #torch.manual_seed(0)
        generator.manual_seed(seed)
        latents = torch.randn(
            (bs, t, c, h, w),
            generator=generator,
            device=template_data.device,
        ) 
        
        latents = latents * self.sample_noise_scheduler.init_noise_sigma
        model_kwargs = dict(inp=cond_data, backbone_out=None)
        
        
        intermediate_preds = None
        diff_list = []

        if return_intermediate:
            num_save = 10  # 保存10张
            total_t = len(self.sample_noise_scheduler.timesteps)
            save_indices = np.linspace(0, total_t - 1, num_save, dtype=int).tolist()
            intermediate_preds = []
               
                
        for idx, t in enumerate(self.sample_noise_scheduler.timesteps, start=1):
            timestep = torch.ones((bs,), device=template_data.device) * t
            # print('timestep', timestep)
            # a) 预测噪声
            noise_pred = self.model[list(self.model.keys())[0]](latents, timestep, **model_kwargs)

            latents = latents.requires_grad_()

            out = self.sample_noise_scheduler.step(
                noise_pred, t, latents
            )
            
            x_tm1_prime = out['prev_sample']  
            x0_hat      = out['pred_original_sample']  

            # (3) 用 x_0_hat 做 decode_stage => A(x_0_hat)，再与 gt_cond 做对比
            if mask is not None:
                diff = (x0_hat - guide_cond) * mask
            else:
                diff = (x0_hat - guide_cond)
                
            norm = torch.linalg.norm(diff)

            # (4) 计算对 x_{t-1}' 的梯度
            grad = torch.autograd.grad(norm, latents, retain_graph=False, allow_unused=True)[0]
            
            # f) 用梯度修正 => x_{t-1}
            guidance_scale = getattr(self, "guidance_scale", 0.05)
            x_tm1 = x_tm1_prime - guidance_scale * grad
            latents = x_tm1.detach()
            
            # ====== 这里加一个判断：每隔 save_interval 记录一次 diff ====== 
            if (idx % save_interval == 0):
                diff_list.append((idx, norm.item(), diff.mean().item()))  
            
            if return_intermediate:
                if idx in save_indices:
                    intermediate_preds.append(x0_hat)
            
        return latents, intermediate_preds, diff_list
    
    @torch.no_grad()
    def _denoise(self, template_data, cond_data, bs=1, vis=False, cfg=1, ensemble_member=1, seed = 0, tar = None,file_names=None):
        """
        denoise from gaussian.
        """
        _, t, c, h, w = template_data.shape
        cond_data = cond_data[:bs, ...]

        generator = torch.Generator(device=template_data.device)
        generator.manual_seed(seed)
        latents = torch.randn((bs * ensemble_member, t, c, h, w), generator=generator, device=template_data.device)
        latents = latents * self.sample_noise_scheduler.init_noise_sigma

        model_kwargs = dict(inp=cond_data, backbone_out=None)
        print("start sampling")

        visual_timesteps = set(int(v) for v in get_visual_timesteps(total_steps = 1000))
        latents_list = {}
        metrics_by_step = {}
        latent_img_dict = {}
        print(self.sample_noise_scheduler.timesteps)
        if cfg == 1:
            assert ensemble_member == 1

            for timestep_val in tqdm(self.sample_noise_scheduler.timesteps) if (self.debug or vis) else self.sample_noise_scheduler.timesteps:
                t_int = int(timestep_val)
                timestep = torch.ones((bs,), device=template_data.device) * t_int

                noise_pred = self.model[list(self.model.keys())[0]](latents, timestep, **model_kwargs)
                latents = self.sample_noise_scheduler.step(noise_pred, t_int, latents).prev_sample

                if t_int in visual_timesteps:
                    latents_vis = latents[0].detach().cpu().numpy()
                    latents_list[t_int] = latents_vis
                    latent_img_dict[t_int] = (latents_vis * 255).clip(0, 255).astype(np.uint8)

                    result = self.eval_metrics.update_single_sample(
                        target=template_data[0].unsqueeze(0),
                        pred=latents,
                        sample_names=f"step_{t_int}"
                    )
                    metrics_by_step[t_int] = self.eval_metrics.compute_frame(result)

            print("end sampling")

            decoded_list = [(t, latents_list[t]) for t in visual_timesteps if t in latents_list]
            thresholds = [16, 74, 133, 160, 181, 219]

            visualize_timesteps_overlay(
                decoded_list,
                tgt_img=template_data[0].detach().cpu().numpy(),
                metrics_by_step=metrics_by_step,
                thresholds=thresholds,
                file_names=file_names
            )

            visualize_latent_images(
                latent_img_dict,
                save_path="/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/fig/vis_allStepDDIM",
                file_names=file_names
            )

        return latents


    def _denoise_wavelet(self, template_data, cond_data, bs=1, vis=True, cfg=1, ensemble_member=1, seed = 0, tar = None,file_names=None):
        """
        denoise from gaussian.
        """
        _, t, c, h, w = template_data.shape
        cond_data = cond_data[:bs, ...]

        generator = torch.Generator(device=template_data.device)
        generator.manual_seed(seed)
        latents = torch.randn((bs * ensemble_member, t, c, h, w), generator=generator, device=template_data.device)
        latents = latents * self.sample_noise_scheduler.init_noise_sigma

        model_kwargs = dict(inp=cond_data, backbone_out=None)
        print("start sampling")

        visual_timesteps = set(int(v) for v in get_visual_timesteps(total_steps = 1000))
        latents_list = {}
        metrics_by_step = {}
        latent_img_dict = {}
        metrics_by_step_wavelet = {}  # 新增：用于存储每步子图指标
        
        print(self.sample_noise_scheduler.timesteps)
        if cfg == 1:
            for timestep_val in tqdm(self.sample_noise_scheduler.timesteps) if (self.debug or vis) else self.sample_noise_scheduler.timesteps:
                t_int = int(timestep_val)
                timestep = torch.ones((bs,), device=template_data.device) * t_int

                noise_pred = self.model[list(self.model.keys())[0]](latents, timestep, **model_kwargs)
                latents = self.sample_noise_scheduler.step(noise_pred, t_int, latents).prev_sample

                if t_int in visual_timesteps:
                    latents_vis = latents[0].detach().cpu().numpy()  # (T, C, H, W)
                    latents_list[t_int] = latents_vis
                    latent_img_dict[t_int] = (latents_vis * 255).clip(0, 255).astype(np.uint8)
                     
                    # 原始指标评估
                    result = self.eval_metrics.update_single_sample(
                        target=template_data[0].unsqueeze(0),
                        pred=latents,
                        sample_names=f"step_{t_int}"
                    )
                    metrics_by_step[t_int] = self.eval_metrics.compute_frame(result)

                    # 小波变换 + 每个子带的指标
                    band_metrics = {}
                    # 小波变换 + 每个子带的整体 sample 指标（单通道）
                    band_frames = defaultdict(list)

                    for i in range(latents_vis.shape[0]):  # 遍历时间帧
                        pred_img = latents_vis[i, 0]  # shape: [H, W]
                        tgt_img = template_data[0, i, 0].detach().cpu().numpy()

                        coeffs_pred = pywt.dwt2(pred_img, 'haar')
                        coeffs_tgt = pywt.dwt2(tgt_img, 'haar')
                        pred_subbands = {
                            "LL": coeffs_pred[0],
                            "LH": coeffs_pred[1][0],
                            "HL": coeffs_pred[1][1],
                            "HH": coeffs_pred[1][2]
                        }
                        tgt_subbands = {
                            "LL": coeffs_tgt[0],
                            "LH": coeffs_tgt[1][0],
                            "HL": coeffs_tgt[1][1],
                            "HH": coeffs_tgt[1][2]
                        }

                        for key in pred_subbands:
                            band_frames[f"pred_{key}"].append(pred_subbands[key])
                            band_frames[f"tgt_{key}"].append(tgt_subbands[key])

                    band_metrics = {}
                    for band in ["LL", "LH", "HL", "HH"]:
                        pred_stack = np.stack(band_frames[f"pred_{band}"], axis=0)  # [T, H, W]
                        tgt_stack = np.stack(band_frames[f"tgt_{band}"], axis=0)    # [T, H, W]

                        pred_tensor = torch.tensor(pred_stack)[None, None]  # [1, 1, T, H, W]
                        tgt_tensor = torch.tensor(tgt_stack)[None, None]

                        eval_result = self.eval_metrics.update_single_sample(
                            target=tgt_tensor,
                            pred=pred_tensor,
                            sample_names=f"step_{t_int}_wavelet_{band}"
                        )
                        band_metrics[band] = self.eval_metrics.compute_frame(eval_result)

                    metrics_by_step_wavelet[t_int] = band_metrics

            # 可视化
            if vis:
                visualize_timesteps_overlay_wavelet(
                    metrics_by_step_wavelet=metrics_by_step_wavelet,
                    tgt_img=template_data[0].detach().cpu().numpy(),
                    thresholds=[16, 74, 133, 160, 181, 219],
                    file_names=file_names,
                    save_path="/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/fig/vis_wavelet"
                )
                
                visualize_timesteps_overlay(
                    [(t, latents_list[t]) for t in visual_timesteps if t in latents_list],
                    tgt_img=template_data[0].detach().cpu().numpy(),
                    metrics_by_step=metrics_by_step,
                    thresholds=[16, 74],
                    file_names=file_names
                )
    
                visualize_latent_images(
                    latent_img_dict,
                    save_path="/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/fig/vis_allStepDDIM",
                    file_names=file_names
                )

        return latents
    
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


    def train_one_step(self, batch_data, step):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples'] 
        file_name = data_dict['file_name']

        b, t, c, h, w = tar.shape
        
        # if self.scale_factor == 1:
        #     self.init_scale_factor(tar)
            
        ## sample noise to add ##
        noise = torch.randn_like(tar)
        ## sample random timestep for each ##
        bs = inp.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=inp.device)
        noisy_tar = self.noise_scheduler.add_noise(tar, noise, timesteps)

        ## predict the noise residual ##
        model_kwargs = dict(inp=inp, backbone_out=None)
        noise_pred = self.model[list(self.model.keys())[0]](noisy_tar, timesteps, **model_kwargs)

        loss = self.loss(noise_pred, noise) ## important: rescale the loss
        loss.backward()

        ## update params of diffusion model ##
        self.optimizer[list(self.model.keys())[0]].step()
        self.optimizer[list(self.model.keys())[0]].zero_grad()

        if self.visualizer_type == 'sevir_visualizer' and (step) % self.visualizer_step==0:
            z_sample_prediction = self.denoise(template_data=tar, cond_data=inp, bs=1)
            self.visualizer.save_pixel_image(pred_image=z_sample_prediction, target_img=tar, step=step)
        else:
            pass
        return {self.loss_type: loss.item()}
        
    @torch.no_grad()
    def test_one_step(self, batch_data):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples'] 
        file_name = data_dict['file_name']

        b, t, c, h, w = tar.shape
        
        ## sample noise to add ##
        noise = torch.randn_like(tar)
        
        ## sample random timestep for each ##
        bs = inp.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=inp.device)
        noisy_tar = self.noise_scheduler.add_noise(tar, noise, timesteps)

        ## predict the noise residual ##
        model_kwargs = model_kwargs = dict(inp=inp, backbone_out=None)
        # noise_pred = self.model[list(self.model.keys())[0]](noisy_tar, timesteps, **model_kwargs)

        ## denoise ##
        # if self.guidance == 'DPS':
        #     z_sample_prediction_dps, intermediate_preds, diff_list = self.dps_guidance_denoise(
        #         template_data=tar,
        #         cond_data=inp,
        #         bs=bs,
        #         guide_cond=tar,  # 需要用来做measurement guidance
        #         mask=None
        #     )
        #     print('diff list', diff_list)
        # z_sample_prediction = self._denoise_wavelet(template_data=tar, cond_data=inp, bs=1, file_names=file_name[0])
        z_sample_prediction = self.denoise(template_data=tar, cond_data=inp, bs=bs)
        
        loss_records = {}
        
        ## evaluate other metrics ##
        data_dict = {}
        data_dict['gt'] = tar
        data_dict['pred'] = z_sample_prediction
        MSE_loss = torch.mean((z_sample_prediction - tar) ** 2).item()
        
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
        
        ### compute perceptual quality ###
        # self.FID_computer.update(images_real=data_dict['gt'], images_fake=data_dict['pred'])
        # self.FVD_computer.update(videos_real=data_dict['gt'], videos_fake=data_dict['pred'])
        # lpips_score = self.LPIPS_computer(pred=data_dict['pred'], target=data_dict['gt'])
        # loss_records.update({'LPIPS': lpips_score})
        return loss_records


    def test_one_step_2(self, batch_data):
        data_dict = self.data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples'] 
        file_name = data_dict['file_name']

        b, t, c, h, w = tar.shape
        
        all_predictions = []
        
        ## sample noise to add ##
        noise = torch.randn_like(tar)
        ## sample random timestep for each ##
        bs = inp.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=inp.device)
        noisy_tar = self.noise_scheduler.add_noise(tar, noise, timesteps)

        ## predict the noise residual ##
        model_kwargs = model_kwargs = dict(inp=inp, backbone_out=None)

        # if self.guidance == 'DPS':
        # for i in range(10):
        z_sample_prediction_dps, intermediate_preds, diff_list = self.dps_guidance_denoise(
            template_data=tar,
            cond_data=inp,
            bs=bs,
            guide_cond=tar, 
            mask=None
        )


        z_sample_prediction = self.denoise(template_data=tar, cond_data=inp, bs=bs)
        loss_records = {}
        
        ## evaluate other metrics ##
        data_dict = {}
        data_dict['gt'] = tar
        data_dict['pred'] = z_sample_prediction_dps
        MSE_loss = torch.mean((z_sample_prediction_dps - tar) ** 2).item()
        
        ## 计算每一个dps优化后sample与未优化的指标
        for i in range(b):
            result_dps = self.eval_metrics.update_single_sample(
                    target=tar[i].unsqueeze(0),               # shape: (1, 128, 128)
                    pred=z_sample_prediction_dps[i].unsqueeze(0),   # shape: (1, 128, 128)
                    sample_names='name'
                )
            
            result_ori = self.eval_metrics.update_single_sample(
                    target=tar[i].unsqueeze(0),               # shape: (1, 128, 128)
                    pred=z_sample_prediction[i].unsqueeze(0),   # shape: (1, 128, 128)
                    sample_names='name'
                )
            
            metrics_dps = self.eval_metrics.compute_frame(result_dps)
            metrics_ori = self.eval_metrics.compute_frame(result_ori)
            
            # self.logger.info(f"result_dps_far:{metrics_dps['avg']['far']}")
            # self.logger.info(f"result_dps_csi:{metrics_dps['avg']['csi']}")
            
            # self.logger.info(f"result_ori_far:{metrics_ori['avg']['far']}")
            # self.logger.info(f"result_ori_csi:{metrics_ori['avg']['csi']}")
            
            
                    
        ## evaluation ##
        if self.metrics_type == 'SEVIRSkillScore':
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
            
            ### compute perceptual quality ###
            # self.FID_computer.update(images_real=data_dict['gt'], images_fake=data_dict['pred'])
            # self.FVD_computer.update(videos_real=data_dict['gt'], videos_fake=data_dict['pred'])
            # lpips_score = self.LPIPS_computer(pred=data_dict['pred'], target=data_dict['gt'])
            # loss_records.update({'LPIPS': lpips_score})
        else:
            raise NotImplementedError
        
        return loss_records


    def test_one_step_3(self, batch_data):
        data_dict = self.save_data_preprocess(batch_data)
        inp, tar = data_dict['inputs'], data_dict['data_samples'] 
        file_name = data_dict['file_name']

        b, t, c, h, w = tar.shape
        
        all_predictions = []
        
        ## sample noise to add ##
        noise = torch.randn_like(tar)
        ## sample random timestep for each ##
        bs = inp.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=inp.device)
        noisy_tar = self.noise_scheduler.add_noise(tar, noise, timesteps)

        ## predict the noise residual ##
        model_kwargs = model_kwargs = dict(inp=inp, backbone_out=None)

        # if self.guidance == 'DPS':
        for i in range(10):
            z_sample_prediction_dps, intermediate_preds, diff_list = self.dps_guidance_denoise(
                template_data=tar,
                cond_data=inp,
                bs=bs,
                guide_cond=tar, 
                mask=None,
                seed=i, 
            )
            
            prediction = z_sample_prediction_dps
            all_predictions.append(prediction)
                    
            sample_file_name = [f'sample{i}_' + item for item in file_name]
            self.save_pred_data(batch_data=prediction, file_names=sample_file_name, 
                                    dataset_name=data_dict['file_name'])

        loss_records = {}
        
        ## evaluate other metrics ##
        data_dict = {}
        data_dict['gt'] = tar
        data_dict['pred'] = z_sample_prediction_dps
        MSE_loss = torch.mean((z_sample_prediction_dps - tar) ** 2).item()
                 
        ## evaluation ##
        if self.metrics_type == 'SEVIRSkillScore':
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
            
            ### compute perceptual quality ###
            # self.FID_computer.update(images_real=data_dict['gt'], images_fake=data_dict['pred'])
            # self.FVD_computer.update(videos_real=data_dict['gt'], videos_fake=data_dict['pred'])
            # lpips_score = self.LPIPS_computer(pred=data_dict['pred'], target=data_dict['gt'])
            # loss_records.update({'LPIPS': lpips_score})
        else:
            raise NotImplementedError
        
        return loss_records
    
    # @torch.no_grad()
    def test(self, test_data_loader, epoch, step, state='Train'):
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        # set model to eval
        for key in self.model:
            self.model[key].eval()
        data_loader = test_data_loader

        for step, batch in enumerate(data_loader):
            if self.debug and step>= 2:
                break

            loss = self.test_one_step(batch)
            metric_logger.update(**loss)

        ## compute meteorologic metrics ##
        losses = {}
        metrics = self.eval_metrics.compute()
        for thr, thr_dict in metrics.items():
            for k, v in thr_dict.items():
                losses.update({f'{thr}-{k}': v})
        self.eval_metrics.reset()

        # ## compute perceptual quality ##
        # FID = self.FID_computer.compute()
        # FVD = self.FVD_computer.compute()
        # metric_logger.update(**{'FID': FID, 'FVD': FVD})
        metric_logger.update(**losses)
        self.logger.info('  '.join(
                [f'Epoch [{epoch + 1}](val stats)',
                 "{meters}"]).format(
                    meters=str(metric_logger)
                 ))
        
        return metric_logger


    def save_data_preprocess(self, data):
        data_dict = {}
        inp_data = data['inputs'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        tar_data = data['data_samples'].float().to(self.device, non_blocking=True, dtype=self.data_type)
        file_names = [item.split('/')[-1] for item in data['file_name']]
        dataset_name = data['dataset_name'][0]
        data_dict.update({'inputs': inp_data, 'data_samples': tar_data, 'file_name': file_names, 
                          'dataset_name': dataset_name})
        return data_dict
    

    # save 5次随机采样的sample
    # address：rankcast/dataset/BaseModel/sample_data/s0_name...
    @torch.no_grad()
    def save_sample(self, dataloader, split, start_step=0):
        self.threadPool = ThreadPoolExecutor(max_workers=16)
        self.all_metrics = []
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        try:
            for step, batch in enumerate(dataloader):
                print('step:', step)
                if step < start_step:
                    continue

                data_dict = {}
                data_dict = self.save_data_preprocess(batch)
                inp, tar = data_dict['inputs'], data_dict['data_samples']
                file_names = data_dict['file_name']
                all_predictions = []
                bs = inp.shape[0]
                print('batch_size:', bs)
                for sample_num in range(5):
                    seed = random.randint(0, 10000) 
                    start_time = time.time()
                    z_sample_prediction = self.denoise(template_data=tar, cond_data=inp, bs=bs, seed=seed)
                    end_time = time.time()
                    print(f'sample time: {end_time - start_time:.2f} s')
                    loss_records = {}
                    data_dict['gt'] = tar
                    data_dict['pred'] = z_sample_prediction
                
                    prediction = z_sample_prediction
                    all_predictions.append(prediction)
                    
                    sample_file_name = [f's{sample_num}_' + item for item in file_names]
                    self.save_pred_data(batch_data=prediction, file_names=sample_file_name, 
                                    dataset_name=data_dict['dataset_name'], cluster_name='sample_data')
                    
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
                sample_5_file_name = ["s5_" + item for item in file_names]
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
                
                print('check gt == sample_5', tar == sample_5)
                self.save_pred_data(batch_data=sample_5, file_names=sample_5_file_name, 
                                    dataset_name=data_dict['dataset_name'], cluster_name='sample_data')
                self.save_pred_data(batch_data=tar, file_names=gt_file_name, 
                                    dataset_name=data_dict['dataset_name'], cluster_name='sample_data')
                    
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

    # save sample to cluster 
    @torch.no_grad()
    def save_pred_data(self, batch_data, file_names, dataset_name, cluster_name):
        root_dir = f'ai4earth-write:ai4earth-pool5-2/rankcast/{dataset_name}/BaseModel/{cluster_name}'
        dataset_name = dataset_name
        
        bs = batch_data.shape[0]
        for i in range(bs):
            save_data = batch_data[i].cpu().numpy()
            save_file_name = f'{root_dir}/{file_names[i]}' 
            local_path = os.path.join('/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/temp', file_names[i])
            ### save ###
            np.save(local_path, save_data)
            self.threadPool.submit(self.update_remote, local_path, save_file_name, save_data)

        return None


    @torch.no_grad()
    def save_win_lose_data(self, batch_data, file_names, dataset_name, cluster_name):
        root_dir = f'ai4earth-write:ai4earth-pool5-2/rankcast/{dataset_name}/BaseModel/{cluster_name}'
        dataset_name = dataset_name
        try:
            save_data = batch_data.cpu().numpy()
        except:
            save_data = batch_data
        save_file_name = f'{root_dir}/{file_names}' 
        local_path = os.path.join('/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/temp', file_names)
        ### save ###
        np.save(local_path, save_data)
        print(save_file_name)
        self.update_remote(local_path, save_file_name, save_data)
        # self.threadPool.submit(self.update_remote, local_path, save_file_name)
        return None
    
    @torch.no_grad() 
    def write_data(self, save_file_name, local_path1, data1):
        np.save(local_path1, data1)
        print(save_file_name) 
        res = self.s3_client.upload_file(f'{local_path1}', 'zwl2', save_file_name)
        os.remove(local_path1)

    def update_remote(
            self,
            local_file: str,
            src_file: str,
            data
    ):
        import subprocess
        cmd = [
            "/mnt/petrelfs/xukaiyi/rclone-v1.68.2-linux-amd64/rclone",
            "copyto",
            local_file,
            src_file,
            "--s3-no-check-bucket"
        ]

        try:
            for attempt in range(3):
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    print(f"Upload succeeded on attempt {attempt + 1}")
                    break
                except subprocess.CalledProcessError as e:
                    print(f"Upload attempt {attempt + 1} failed: {e.stderr.decode().strip()}")
                    # often because no data update
                    np.save(local_file, data)
                    if attempt < 2:
                        time.sleep(2) 
                    else:
                        raise RuntimeError("Upload failed after 3 attempts.")
        finally:
            import os
            if os.path.exists(local_file):
                os.remove(local_file)
                
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

    ### save frame best on single metric
    def get_win_loss_frame_samples(self, dataset):
        self.all_metrics = []
        path = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/datasets/hko7/hko7_12min_list/hko7_rainy_valid_top.txt'
        dataset_name = dataset
        
        with open(path, 'r') as f:
            # files = [line.split(' ')[-1].strip() for line in f.readlines()]
            files = [line for line in f.readlines()]
        for file_idx, file in enumerate(files, start=1):
            print('idx:', file_idx)

            # 读取 6 个样本轨迹（s0 ~ s5）
            sample_data = []
            np_path = f"s0_{file}"
            last_datafile = None
            for i in range(6):
                datafile = os.path.join(
                    f"cluster3:s3://ai4earth-pool5-2/rankcast/{dataset_name}/BaseModel/sample_data",
                    np_path
                )
                datafile = datafile.replace('s0', f's{i}')
                last_datafile = datafile  # 用于下面构造 gt 路径
                print(datafile)

                data_bytes = self.client.get(datafile)
                data_file = io.BytesIO(data_bytes)
                arr = np.load(data_file, allow_pickle=True)  # 形状: (T, 1, 128, 128)
                sample_data.append(arr)

            name = file.split('/')[-1].replace('\n', '').replace('.npy', '')

            # 读取 GT（和 s?_* 同目录，命名 gt_*）
            gt_file = re.sub(r's\d+_', 'gt_', last_datafile)
            gt_bytes = self.client.get(gt_file)
            gt_f = io.BytesIO(gt_bytes)
            gt = np.load(gt_f, allow_pickle=True)  # 形状: (T, 1, 128, 128)
            T = gt.shape[0]
            S = len(sample_data)  # 样本数（通常 6）

            # 四种挑选：FAR 最好/最差；CSI 最好/最差
            best_far_val   = [np.inf]  * T
            worst_far_val  = [-np.inf] * T
            best_csi_val   = [-np.inf] * T
            worst_csi_val  = [np.inf]  * T

            best_far_idx   = [-1] * T
            worst_far_idx  = [-1] * T
            best_csi_idx   = [-1] * T
            worst_csi_idx  = [-1] * T

            # 逐帧逐样本评估
            for t in range(T):
                for s in range(S):
                    # 单帧评估
                    result_patch = self.eval_metrics.update_frame(
                        target=gt[t, :, :, :],               # (1, 128, 128)
                        pred=sample_data[s][t, :, :, :],     # (1, 128, 128)
                        sample_names=name
                    )
                    metrics = self.eval_metrics.compute_frame(result_patch)
                    far = float(metrics['avg']['far'])
                    csi = float(metrics['avg']['csi'])

                    # 处理 NaN：FAR 的 NaN 当作 +inf（最差），CSI 的 NaN 当作 -inf（最差）
                    far_for_best  = far if not np.isnan(far) else np.inf
                    far_for_worst = far if not np.isnan(far) else -np.inf
                    csi_for_best  = csi if not np.isnan(csi) else -np.inf
                    csi_for_worst = csi if not np.isnan(csi) else np.inf

                    # FAR：越小越好
                    if far_for_best < best_far_val[t]:
                        best_far_val[t] = far_for_best
                        best_far_idx[t] = s
                    if far_for_worst > worst_far_val[t]:
                        worst_far_val[t] = far_for_worst
                        worst_far_idx[t] = s

                    # CSI：越大越好
                    if csi_for_best > best_csi_val[t]:
                        best_csi_val[t] = csi_for_best
                        best_csi_idx[t] = s
                    if csi_for_worst < worst_csi_val[t]:
                        worst_csi_val[t] = csi_for_worst
                        worst_csi_idx[t] = s

            # 根据索引拼接四条序列 (T, 1, 128, 128)
            def build_sequence(idx_list):
                # 若出现 -1（极端情况所有指标都是 NaN），回退到第 0 个样本
                idx_list = [i if i >= 0 else 0 for i in idx_list]
                return np.stack([sample_data[idx][t] for t, idx in enumerate(idx_list)], axis=0)

            seq_best_far  = build_sequence(best_far_idx)
            seq_worst_far = build_sequence(worst_far_idx)
            seq_best_csi  = build_sequence(best_csi_idx)
            seq_worst_csi = build_sequence(worst_csi_idx)

            # 转 torch
            seq_best_far_t  = torch.from_numpy(seq_best_far)
            seq_worst_far_t = torch.from_numpy(seq_worst_far)
            seq_best_csi_t  = torch.from_numpy(seq_best_csi)
            seq_worst_csi_t = torch.from_numpy(seq_worst_csi)
            gt_t = torch.from_numpy(gt)

            self.save_win_lose_data(batch_data=seq_best_far_t,  file_names= 'best_far_' + name + '.npy', dataset_name=f"{dataset_name}", cluster_name = 'rank_frame/best_far')
            self.save_win_lose_data(batch_data=seq_worst_far_t, file_names= 'worst_far_' + name + '.npy', dataset_name=f"{dataset_name}", cluster_name = 'rank_frame/worst_far')
            self.save_win_lose_data(batch_data=seq_best_csi_t,  file_names= 'best_csi_' + name + '.npy', dataset_name=f"{dataset_name}", cluster_name = 'rank_frame/best_csi')
            self.save_win_lose_data(batch_data=seq_worst_csi_t, file_names= 'worst_csi_' + name + '.npy', dataset_name=f"{dataset_name}", cluster_name = 'rank_frame/worst_csi')

            # 评估四条序列（整段）
            def eval_and_log(tag, seq_t):
                result = self.eval_metrics.update_single_sample(
                    target=gt_t.unsqueeze(0),   # (1, T, 1, 128, 128) 若你的函数期望这类形状
                    pred=seq_t.unsqueeze(0),
                    sample_names=name
                )
                m = self.eval_metrics.compute_frame(result)
                self.logger.info(f"[{tag}] far: {m['avg']['far']}, csi: {m['avg']['csi']}")
                return m

            eval_and_log('WIN_FAR',  seq_best_far_t)
            eval_and_log('LOSE_FAR', seq_worst_far_t)
            eval_and_log('WIN_CSI',  seq_best_csi_t)
            eval_and_log('LOSE_CSI', seq_worst_csi_t)

        print('!!!!!!!!!!!!!!!!!!!!!!!!!!Finish!!!!!!!!!!!!!!!!!!!!!!!!!!')

    ### save sample best on two mtric
    def rank_by_sample_both(self, dataset, top_k=2):
        from scipy.stats import pearsonr
        path_list = []
        dataset_name = dataset
        path = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/datasets/hko7/hko7_12min_list/hko7_rainy_valid_top.txt'
        dataset_name = dataset
        
        with open(path, 'r') as f:
            # files = [line.split(' ')[-1].strip() for line in f.readlines()]
            files = [line for line in f.readlines()]
            
        my_idx = 0
        for file in files:
            print(my_idx)
            my_idx = my_idx + 1
            sample_data = []
            np_path = f"s0_{file}"
            for i in range(6):
                datafile = os.path.join(
                    f"cluster3:s3://ai4earth-pool5-2/rankcast/{dataset_name}/BaseModel/sample_data",
                    np_path
                )
                
                datafile = datafile.replace('s0',f's{i}')
                data = self.client.get(datafile)
                data_file = io.BytesIO(data)
                data = np.load(data_file, allow_pickle=True)
                sample_data.append(data)
        
            name = file.split('/')[-1].replace('\n', '').replace('.npy', '')
            gt_file = re.sub(r's\d+_', 'gt_', datafile)
            gt = self.client.get(gt_file)
            gt = io.BytesIO(gt)
            gt = np.load(gt, allow_pickle=True)
        
            csi_list = []
            far_list = []

            for idx in range(6): 
                result_patch = self.eval_metrics.update_single_sample(
                    target=torch.from_numpy(gt).unsqueeze(0),               
                    pred=torch.from_numpy(sample_data[idx]).unsqueeze(0),   
                    sample_names=name
                )
                metrics = self.eval_metrics.compute_frame(result_patch)
                avg_csi = metrics['avg']['csi']
                avg_far = metrics['avg']['far']
                
                csi_list.append(avg_csi)
                far_list.append(avg_far)

            csi_array = np.array(csi_list)
            far_array = np.array(far_list)

            # 排名：CSI越大越好，FAR越小越好
            csi_ranks = np.argsort(-csi_array)
            far_ranks = np.argsort(far_array)

            rank_summary = []
            for i in range(len(csi_array)):
                csi_rank = np.where(csi_ranks == i)[0][0] + 1
                far_rank = np.where(far_ranks == i)[0][0] + 1
                rank_sum = csi_rank + far_rank
                rank_summary.append((i, csi_rank, far_rank, rank_sum))

            # Win sample: CSI和FAR都在前top_k名
            win_candidates = [x for x in rank_summary if x[1] <= top_k and x[2] <= top_k]
            win_idx = sorted(win_candidates, key=lambda x: x[3])[0][0] if win_candidates else None

            # Lose sample: CSI和FAR都在后top_k名
            bottom_thresh = len(csi_array) - top_k + 1
            lose_candidates = [x for x in rank_summary if x[1] >= bottom_thresh and x[2] >= bottom_thresh]
            lose_idx = sorted(lose_candidates, key=lambda x: x[3])[0][0] if lose_candidates else None

            # 返回值
            win_sample = sample_data[win_idx] if win_idx is not None else None
            win_csi = csi_array[win_idx] if win_idx is not None else None
            win_far = far_array[win_idx] if win_idx is not None else None

            lose_sample = sample_data[lose_idx] if lose_idx is not None else None
            lose_csi = csi_array[lose_idx] if lose_idx is not None else None
            lose_far = far_array[lose_idx] if lose_idx is not None else None
    
            if win_idx is not None and lose_idx is not None:
                print(f"[{file}] ✅ Win Sample - idx: {win_idx}, CSI: {win_csi:.4f}, FAR: {win_far:.4f}")
                print(f"[{file}] ❌ Lose Sample - idx: {lose_idx}, CSI: {lose_csi:.4f}, FAR: {lose_far:.4f}")
                
                win_name = f'rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin/ddim_sample/sample_rank/win_{file}' 
                lose_name = f'rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin/ddim_sample/sample_rank/lose_{file}'

                base_dir = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/temp'
                paths_and_data = [
                    (os.path.join(base_dir, f'win_{file}'), win_name, win_sample),
                    (os.path.join(base_dir, f'lose_{file}'), lose_name, lose_sample)
                ]
                
                self.save_win_lose_data(batch_data=win_sample,  file_names= 'best_' + name + '.npy', dataset_name=f"{dataset_name}", cluster_name = 'rank_sample_both/best')
                self.save_win_lose_data(batch_data=lose_sample, file_names= 'worst_' + name + '.npy', dataset_name=f"{dataset_name}", cluster_name = 'rank_sample_both/worst')

            else:
                print(f"[{file}] skipped — reason:")
                if win_idx is None:
                    print("  ❌ No WIN sample: no sample with both CSI & FAR in top-{top_k}")
                if lose_idx is None:
                    print("  ❌ No LOSE sample: no sample with both CSI & FAR in bottom-{top_k}")

    ### save sample best on single metric
    def get_win_loss_samples_normal(self, dataset):
        self.all_metrics =[]
        path = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/datasets/hko7/hko7_12min_list/hko7_rainy_valid_top.txt'
        dataset_name = dataset
        
        with open(path, 'r') as f:
            # files = [line.split(' ')[-1].strip() for line in f.readlines()]
            files = [line for line in f.readlines()]
        
        index = 0
        # file可以只选择sample0
        for file in files:
            print('idx:', index)
            index = index + 1
            sample_data = []
            for i in range(6):
                np_path = f"s0_{file}"
                datafile = os.path.join(f"cluster3:s3://ai4earth-pool5-2/rankcast/{dataset_name}/BaseModel/sample_data", np_path)
                datafile = datafile.replace('s0',f's{i}')
                data = self.client.get(datafile)
                data_file = io.BytesIO(data)
                data = np.load(data_file, allow_pickle=True)
                sample_data.append(data)
                
            name = file.split('/')[-1].replace('\n', '').replace('.npy', '')
            gt_file = re.sub(r's\d+_', 'gt_', datafile)
            gt = self.client.get(gt_file)
            gt = io.BytesIO(gt)
            gt = np.load(gt, allow_pickle=True)

            best_samples = []
            worst_samples = []
            
            best_csi = -float('inf')
            worst_csi = float('inf')
            best_far = float('inf')
            worst_far = -float('inf')
                
            best_idx_csi = -1
            worst_idx_csi = -1

            best_idx_far = -1
            worst_idx_far = -1

            for idx in range(5):
                result_patch = self.eval_metrics.update_single_sample(
                    target=torch.from_numpy(gt).unsqueeze(0),               # shape: (1, 128, 128)
                    pred=torch.from_numpy(sample_data[idx]).unsqueeze(0),   # shape: (1, 128, 128)
                    sample_names=name
                )
                metrics = self.eval_metrics.compute_frame(result_patch)
                avg_csi = metrics['avg']['csi']
                avg_far = metrics['avg']['far']
                
                if avg_csi > best_csi:
                    best_csi = avg_csi
                    best_idx_csi = idx

                if avg_csi < worst_csi:
                    worst_csi = avg_csi
                    worst_idx_csi = idx
                
                if avg_far < best_far:
                    best_far = avg_far
                    best_idx_far = idx

                if avg_far > worst_far:
                    worst_far = avg_far
                    worst_idx_far = idx

            
            best_sequence1 = torch.from_numpy(sample_data[best_idx_far])
            worst_sequence1 = torch.from_numpy(sample_data[worst_idx_far])

            best_sequence2 = torch.from_numpy(sample_data[best_idx_csi])
            worst_sequence2 = torch.from_numpy(sample_data[worst_idx_csi])
            
            gt = torch.from_numpy(gt)
            
            
            # self.visualizer.save_pixel_image(pred_image=best_sequence1.unsqueeze(0), target_img=gt.unsqueeze(0), step=index)
            
            self.save_win_lose_data(batch_data=best_sequence1, file_names='best_far_' + name + '.npy', 
                         dataset_name=f"{dataset_name}", cluster_name = 'rank_sample/best_far')
            
            self.save_win_lose_data(batch_data=worst_sequence1, file_names='worst_far_' + name + '.npy', 
                         dataset_name=f"{dataset_name}",  cluster_name = 'rank_sample/worst_far')
            
            self.save_win_lose_data(batch_data=best_sequence2, file_names='best_csi_' + name + '.npy', 
                         dataset_name=f"{dataset_name}",  cluster_name = 'rank_sample/best_csi')

            self.save_win_lose_data(batch_data=worst_sequence2, file_names='worst_csi_' + name + '.npy', 
                         dataset_name=f"{dataset_name}",  cluster_name = 'rank_sample/worst_csi')

            result_win = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=best_sequence1.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_win = self.eval_metrics.compute_frame(result_win)

            self.logger.info(f"best far:{metrics_win['avg']['far']}")
            self.logger.info(f"best far-csi:{metrics_win['avg']['csi']}")
            
            result_lose = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=worst_sequence1.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_lose = self.eval_metrics.compute_frame(result_lose)
            
            self.logger.info(f"worst far:{metrics_lose['avg']['far']}")
            self.logger.info(f"worst far-csi:{metrics_lose['avg']['csi']}")

            result_win = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=best_sequence2.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_win = self.eval_metrics.compute_frame(result_win)

            self.logger.info(f"best csi:{metrics_win['avg']['csi']}")
            self.logger.info(f"best csi-far:{metrics_win['avg']['far']}")
            
            result_lose = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=worst_sequence2.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_lose = self.eval_metrics.compute_frame(result_lose)
            
            self.logger.info(f"worst csi:{metrics_lose['avg']['csi']}")
            self.logger.info(f"worst csi-far:{metrics_lose['avg']['far']}")
            
                    
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!Finish!!!!!!!!!!!!!!!!!!!!!!!!!!')
        

    ### save metric 
    def save_metric1(self, dataset):
        """逐帧选择最佳/最差样本，并记录指标（只保留 avg CSI 和 avg FAR)"""
        self.csi_metrics = []
        self.far_metrics = []

        path = "/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/datasets/sevir/path/valid_3h.txt"
        dataset_name = dataset

        with open(path, 'r') as f:
            files = [line.strip() for line in f.readlines()]

        for file_idx, file in enumerate(files, start=1):
            print('idx:', file_idx)
            # s3://weather_radar_datasets/sevir/val/vil-2019-SEVIR_VIL_RANDOMEVENTS_2019_0101_0430.h5-812.npy
            # 读取 6 个样本轨迹
            sample_data = []
            np_path = f"s0_{os.path.basename(file)}"
            last_datafile = None
            for i in range(6):
                datafile = os.path.join(
                    f"cluster3:s3://ai4earth-pool5-2/rankcast/{dataset_name}/BaseModel/sample_data",
                    np_path
                ).replace('s0', f's{i}')
                last_datafile = datafile

                data_bytes = self.client.get(datafile)
                data_file = io.BytesIO(data_bytes)
                arr = np.load(data_file, allow_pickle=True)
                sample_data.append(arr)

            name = os.path.basename(file).replace('.npy', '')

            # 读取 GT
            gt_file = re.sub(r's\d+_', 'gt_', last_datafile)
            gt_bytes = self.client.get(gt_file)
            gt_f = io.BytesIO(gt_bytes)
            gt = np.load(gt_f, allow_pickle=True)
            T, S = gt.shape[0], len(sample_data)

            # 初始化独立索引与指标
            best_far_val   = [np.inf]  * T
            worst_far_val  = [-np.inf] * T
            best_csi_val   = [-np.inf] * T
            worst_csi_val  = [np.inf]  * T

            best_far_idx   = [-1] * T
            worst_far_idx  = [-1] * T
            best_csi_idx   = [-1] * T
            worst_csi_idx  = [-1] * T

            # 逐帧计算
            for t in range(T):
                for s in range(S):
                    result_patch = self.eval_metrics.update_frame(
                        target=gt[t], pred=sample_data[s][t], sample_names=name
                    )
                    metrics = self.eval_metrics.compute_frame(result_patch)
                    far = float(metrics['avg']['far'])
                    csi = float(metrics['avg']['csi'])

                    far_for_best  = far if not np.isnan(far) else np.inf
                    far_for_worst = far if not np.isnan(far) else -np.inf
                    csi_for_best  = csi if not np.isnan(csi) else -np.inf
                    csi_for_worst = csi if not np.isnan(csi) else np.inf

                    if far_for_best < best_far_val[t]:
                        best_far_val[t], best_far_idx[t] = far_for_best, s
                    if far_for_worst > worst_far_val[t]:
                        worst_far_val[t], worst_far_idx[t] = far_for_worst, s
                    if csi_for_best > best_csi_val[t]:
                        best_csi_val[t], best_csi_idx[t] = csi_for_best, s
                    if csi_for_worst < worst_csi_val[t]:
                        worst_csi_val[t], worst_csi_idx[t] = csi_for_worst, s

            # 构造序列
            def build_sequence(idx_list):
                idx_list = [i if i >= 0 else 0 for i in idx_list]
                return np.stack([sample_data[idx][t] for t, idx in enumerate(idx_list)], axis=0)

            seq_best_far  = torch.from_numpy(build_sequence(best_far_idx))
            seq_worst_far = torch.from_numpy(build_sequence(worst_far_idx))
            seq_best_csi  = torch.from_numpy(build_sequence(best_csi_idx))
            seq_worst_csi = torch.from_numpy(build_sequence(worst_csi_idx))
            gt_t = torch.from_numpy(gt)

            # 评估整段
            def eval_seq(seq_t):
                result = self.eval_metrics.update_single_sample(
                    target=gt_t.unsqueeze(0), pred=seq_t.unsqueeze(0), sample_names=name
                )
                return self.eval_metrics.compute_frame(result)

            m_win_far = eval_seq(seq_best_far)
            m_lose_far = eval_seq(seq_worst_far)
            m_win_csi = eval_seq(seq_best_csi)
            m_lose_csi = eval_seq(seq_worst_csi)

            print(m_win_csi["avg"]["csi"])
            print(m_lose_csi["avg"]["csi"])
            
            # 保存指标（只保留 avg CSI 和 avg FAR）
            self.csi_metrics.append({
                "sample": name,
                "csi_win": m_win_csi["avg"]["csi"],
                "csi_lose": m_lose_csi["avg"]["csi"]
            })
            self.far_metrics.append({
                "sample": name,
                "far_win": m_win_far["avg"]["far"],
                "far_lose": m_lose_far["avg"]["far"]
            })

        # 写入 Excel
        self.flush_win_loss_to_excel(
            excel_path=f"/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/metric/BaseModel_rank_frame_{dataset_name}.xlsx"
        )

        self.flush_win_loss_to_excel(excel_path=f"/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/metric/BaseModel_rank_frame_{dataset_name}.xlsx")

    def save_metric(self, dataset):
        print('sevir')
        """逐帧选择最佳/最差样本，并记录指标（只保留 avg CSI 和 avg FAR)"""
        self.csi_metrics = []
        self.far_metrics = []

        path = "/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/datasets/sevir/path/valid_3h.txt"
        dataset_name = dataset

        with open(path, 'r') as f:
            files = [line.strip() for line in f.readlines()]

        for file_idx, file in enumerate(files, start=1):
            print('idx:', file_idx)
            # 读取 6 个样本轨迹
            sample_data = []
            np_path = f"sample0_{file.split('/')[-1]}"
            last_datafile = None
            for i in range(6):
                datafile = os.path.join(
                    f"cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin",
                    np_path
                ).replace('sample0', f'sample{i}')
                last_datafile = datafile

                data_bytes = self.client.get(datafile)
                data_file = io.BytesIO(data_bytes)
                arr = np.load(data_file, allow_pickle=True)
                sample_data.append(arr)

            name = os.path.basename(file).replace('.npy', '')

            # 读取 GT
            gt_file = re.sub(r'sample\d+_', 'gt_', last_datafile)
            gt_bytes = self.client.get(gt_file)
            gt_f = io.BytesIO(gt_bytes)
            gt = np.load(gt_f, allow_pickle=True)
            T, S = gt.shape[0], len(sample_data)

            # 初始化独立索引与指标
            best_far_val   = [np.inf]  * T
            worst_far_val  = [-np.inf] * T
            best_csi_val   = [-np.inf] * T
            worst_csi_val  = [np.inf]  * T

            best_far_idx   = [-1] * T
            worst_far_idx  = [-1] * T
            best_csi_idx   = [-1] * T
            worst_csi_idx  = [-1] * T

            # 逐帧计算
            for t in range(T):
                for s in range(S):
                    result_patch = self.eval_metrics.update_frame(
                        target=gt[t], pred=sample_data[s][t], sample_names=name
                    )
                    metrics = self.eval_metrics.compute_frame(result_patch)
                    far = float(metrics['avg']['far'])
                    csi = float(metrics['avg']['csi'])

                    far_for_best  = far if not np.isnan(far) else np.inf
                    far_for_worst = far if not np.isnan(far) else -np.inf
                    csi_for_best  = csi if not np.isnan(csi) else -np.inf
                    csi_for_worst = csi if not np.isnan(csi) else np.inf

                    if far_for_best < best_far_val[t]:
                        best_far_val[t], best_far_idx[t] = far_for_best, s
                    if far_for_worst > worst_far_val[t]:
                        worst_far_val[t], worst_far_idx[t] = far_for_worst, s
                    if csi_for_best > best_csi_val[t]:
                        best_csi_val[t], best_csi_idx[t] = csi_for_best, s
                    if csi_for_worst < worst_csi_val[t]:
                        worst_csi_val[t], worst_csi_idx[t] = csi_for_worst, s

            # 构造序列
            def build_sequence(idx_list):
                idx_list = [i if i >= 0 else 0 for i in idx_list]
                return np.stack([sample_data[idx][t] for t, idx in enumerate(idx_list)], axis=0)

            seq_best_far  = torch.from_numpy(build_sequence(best_far_idx))
            seq_worst_far = torch.from_numpy(build_sequence(worst_far_idx))
            seq_best_csi  = torch.from_numpy(build_sequence(best_csi_idx))
            seq_worst_csi = torch.from_numpy(build_sequence(worst_csi_idx))
            gt_t = torch.from_numpy(gt)

            # 评估整段
            def eval_seq(seq_t):
                result = self.eval_metrics.update_single_sample(
                    target=gt_t.unsqueeze(0), pred=seq_t.unsqueeze(0), sample_names=name
                )
                return self.eval_metrics.compute_frame(result)

            m_win_far = eval_seq(seq_best_far)
            m_lose_far = eval_seq(seq_worst_far)
            m_win_csi = eval_seq(seq_best_csi)
            m_lose_csi = eval_seq(seq_worst_csi)

            print(m_win_csi["avg"]["csi"])
            print(m_lose_csi["avg"]["csi"])
            
            # 保存指标（只保留 avg CSI 和 avg FAR）
            self.csi_metrics.append({
                "sample": name,
                "csi_win": m_win_csi["avg"]["csi"],
                "csi_lose": m_lose_csi["avg"]["csi"]
            })
            self.far_metrics.append({
                "sample": name,
                "far_win": m_win_far["avg"]["far"],
                "far_lose": m_lose_far["avg"]["far"]
            })

        # 写入 Excel
        self.flush_win_loss_to_excel(
            excel_path=f"/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/metric/BaseModel_rank_frame_{dataset_name}.xlsx"
        )
        
    def flush_win_loss_to_excel(self, excel_path='/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/win_lose_metrics.xlsx'):
        """保存四个sheet：CSI、FAR、CSI_Diff、FAR_Diff"""
        if not self.csi_metrics or not self.far_metrics:
            return

        df_csi = pd.DataFrame(self.csi_metrics)
        df_far = pd.DataFrame(self.far_metrics)

        # print(self.csi_metrics)
        # print(self.far_metrics)
        
        # 只算 CSI 差值
        df_csi_diff = df_csi.copy()
        df_csi_diff["csi_diff"] = df_csi_diff["csi_win"] - df_csi_diff["csi_lose"]
        df_csi_diff = df_csi_diff[["sample", "csi_diff"]]

        # 只算 FAR 差值
        df_far_diff = df_far.copy()
        df_far_diff["far_diff"] = df_far_diff["far_win"] - df_far_diff["far_lose"]
        df_far_diff = df_far_diff[["sample", "far_diff"]]

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_csi.to_excel(writer, sheet_name='CSI', index=False)
            df_far.to_excel(writer, sheet_name='FAR', index=False)
            df_csi_diff.to_excel(writer, sheet_name='CSI_Diff', index=False)
            df_far_diff.to_excel(writer, sheet_name='FAR_Diff', index=False)

        self.logger.info(f"指标保存到 {excel_path}")
        self.csi_metrics.clear()
        self.far_metrics.clear()
             
    def get_win_loss_samples(self):
        self.all_metrics =[]
        path = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/test/ddpm_val_fintune.txt'
        with open(path, 'r') as f:
            files = [line.split(' ')[-1].strip() for line in f.readlines()]
        
        index = 0
        # file可以只选择sample0
        
        for file in files:
            print('idx:', index)
            index = index + 1
            sample_data = []
            for i in range(5):
                datafile = os.path.join('cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin', file)
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

            # v1: 遍历每一帧，计算 avg CSI，挑选最优最劣的 sample
            # v2: 挑选最低的far作为win sample，最高的far作为lose sample
            # for frame in range(18):
            best_csi = -float('inf')
            worst_csi = float('inf')
            best_far = float('inf')
            worst_far = -float('inf')
                
            best_idx = -1
            worst_idx = -1

            for idx in range(6):  # 遍历 6 个 sample
                result_patch = self.eval_metrics.update_single_sample(
                    target=torch.from_numpy(gt).unsqueeze(0),               # shape: (1, 128, 128)
                    pred=torch.from_numpy(sample_data[idx]).unsqueeze(0),   # shape: (1, 128, 128)
                    sample_names=name
                )
                metrics = self.eval_metrics.compute_frame(result_patch)
                avg_csi = metrics['avg']['csi']
                avg_far = metrics['avg']['far']
                if avg_csi > best_csi:
                    best_csi = avg_csi
                    best_idx = idx

                if avg_csi < worst_csi:
                    worst_csi = avg_csi
                    worst_idx = idx
                
                # if avg_far < best_far:
                #     best_far = avg_far
                #     best_idx = idx

                # if avg_far > worst_far:
                #     worst_far = avg_far
                #     worst_idx = idx

            # # 合成完整的 best 和 worst sample 序列
            # best_sequence = []
            # worst_sequence = []


            # best_sequence = np.stack(best_sequence, axis=0)
            # worst_sequence = np.stack(worst_sequence, axis=0)

            best_sequence = torch.from_numpy(sample_data[best_idx])
            worst_sequence = torch.from_numpy(sample_data[worst_idx])
            gt = torch.from_numpy(gt)
            
            
            # self.visualizer.save_pixel_image(pred_image=best_sequence.unsqueeze(0), target_img=gt.unsqueeze(0), step=index)
            
            win_name = 'bysample_win_csi_' + name + '.npy'
            self.save_win_lose_data(batch_data=best_sequence, file_names=win_name, 
                        dataset_name='sevir_128_10m_3h')
            lose_name = 'bysample_lose_csi_' + name + '.npy'
            self.save_win_lose_data(batch_data=worst_sequence, file_names=lose_name, 
                        dataset_name='sevir_128_10m_3h')
            
            
            result_win = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=best_sequence.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_win = self.eval_metrics.compute_frame(result_win)

            self.logger.info(f"best far:{metrics_win['avg']['far']}")
            self.logger.info(f"best far-csi:{metrics_win['avg']['csi']}")
            
            result_lose = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=worst_sequence.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_lose = self.eval_metrics.compute_frame(result_lose)
            
            self.logger.info(f"worst far:{metrics_lose['avg']['far']}")
            self.logger.info(f"worst far-csi:{metrics_lose['avg']['csi']}")
                    
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!Finish!!!!!!!!!!!!!!!!!!!!!!!!!!')


    def rank_by_sample_guidance(self, txt_path, top_k=2):
        from scipy.stats import pearsonr
        path_list = []
        with open(txt_path, 'r') as f:
            files = [line.split(' ')[-1].strip() for line in f.readlines() if 'sample0' in line]
        my_idx = 0
        
        for file in files:
            print(my_idx)
            my_idx = my_idx + 1
            sample_data = []
            for i in range(10):
                datafile = os.path.join('cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_fintune/guidance-scale0.1/diffcast_fin', file)
                datafile = datafile.replace('sample0',f'sample{i}')
                data = self.client.get(datafile)
                data_file = io.BytesIO(data)
                data = np.load(data_file, allow_pickle=True)
                sample_data.append(data)
        
            name = file.split('/')[-1].replace('.npy','')
            gt_basepath = 'cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin/' + file
            gt_file = re.sub(r'sample\d+_', 'gt_', gt_basepath)
            gt = self.client.get(gt_file)
            gt = io.BytesIO(gt)
            gt = np.load(gt, allow_pickle=True)
        
            csi_list = []
            far_list = []

            for idx in range(6): 
                result_patch = self.eval_metrics.update_single_sample(
                    target=torch.from_numpy(gt).unsqueeze(0),               
                    pred=torch.from_numpy(sample_data[idx]).unsqueeze(0),   
                    sample_names=name
                )
                metrics = self.eval_metrics.compute_frame(result_patch)
                avg_csi = metrics['avg']['csi']
                avg_far = metrics['avg']['far']
                
                csi_list.append(avg_csi)
                far_list.append(avg_far)

            csi_array = np.array(csi_list)
            far_array = np.array(far_list)

            # 排名：CSI越大越好，FAR越小越好
            csi_ranks = np.argsort(-csi_array)
            far_ranks = np.argsort(far_array)

            rank_summary = []
            for i in range(len(csi_array)):
                csi_rank = np.where(csi_ranks == i)[0][0] + 1
                far_rank = np.where(far_ranks == i)[0][0] + 1
                rank_sum = csi_rank + far_rank
                rank_summary.append((i, csi_rank, far_rank, rank_sum))

            # Win sample: CSI和FAR都在前top_k名
            win_candidates = [x for x in rank_summary if x[1] <= top_k and x[2] <= top_k]
            win_idx = sorted(win_candidates, key=lambda x: x[3])[0][0] if win_candidates else None

            # Lose sample: CSI和FAR都在后top_k名
            bottom_thresh = len(csi_array) - top_k + 1
            lose_candidates = [x for x in rank_summary if x[1] >= bottom_thresh and x[2] >= bottom_thresh]
            lose_idx = sorted(lose_candidates, key=lambda x: x[3])[0][0] if lose_candidates else None

            # 返回值
            win_sample = sample_data[win_idx] if win_idx is not None else None
            win_csi = csi_array[win_idx] if win_idx is not None else None
            win_far = far_array[win_idx] if win_idx is not None else None

            lose_sample = sample_data[lose_idx] if lose_idx is not None else None
            lose_csi = csi_array[lose_idx] if lose_idx is not None else None
            lose_far = far_array[lose_idx] if lose_idx is not None else None
    
            if win_idx is not None and lose_idx is not None:
                print(f"[{file}] ✅ Win Sample - idx: {win_idx}, CSI: {win_csi:.4f}, FAR: {win_far:.4f}")
                print(f"[{file}] ❌ Lose Sample - idx: {lose_idx}, CSI: {lose_csi:.4f}, FAR: {lose_far:.4f}")
                
                win_name = f'rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin/ddim_sample/guidance/0.1/sample_rank/win_{file}' 
                lose_name = f'rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin/ddim_sample/guidance/0.1/sample_rank/lose_{file}'

                base_dir = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/temp'
                paths_and_data = [
                    (os.path.join(base_dir, f'win_{file}'), win_name, win_sample),
                    (os.path.join(base_dir, f'lose_{file}'), lose_name, lose_sample)
                ]
                
                for save_file_name, cluster, save_data in paths_and_data:
                    self.threadPool.submit(self.write_data, cluster, save_file_name, save_data)
            
            else:
                print(f"[{file}] skipped — reason:")
                if win_idx is None:
                    print("  ❌ No WIN sample: no sample with both CSI & FAR in top-{top_k}")
                if lose_idx is None:
                    print("  ❌ No LOSE sample: no sample with both CSI & FAR in bottom-{top_k}")


    def rank_by_frame(self, txt_path, top_k=2):
        from scipy.stats import pearsonr
        path_list = []
        with open(txt_path, 'r') as f:
            files = [line.split(' ')[-1].strip() for line in f.readlines() if 'sample0' in line]
        my_idx = 0
        
        for file in files:
            print(my_idx)
            my_idx = my_idx + 1
            sample_data = []
            for i in range(6):
                datafile = os.path.join('cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin', file)
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
        
            num_samples = len(sample_data)
            num_frames = gt.shape[0]
            win_idxs = []
            lose_idxs = []

            for frame in range(num_frames):
                csi_list, far_list = [], []

                for idx in range(num_samples):
                    result_patch = self.eval_metrics.update_frame(
                        target=gt[frame], pred=sample_data[idx][frame], sample_names=f"frame{frame}_sample{idx}"
                    )
                    metrics = self.eval_metrics.compute_frame(result_patch)
                    csi_list.append((idx, metrics['avg']['csi']))
                    far_list.append((idx, metrics['avg']['far']))

                # sort descending for CSI (higher better), ascending for FAR (lower better)
                csi_topk = [idx for idx, _ in sorted(csi_list, key=lambda x: -x[1])[:top_k]]
                far_topk = [idx for idx, _ in sorted(far_list, key=lambda x: x[1])[:top_k]]
                csi_bottomk = [idx for idx, _ in sorted(csi_list, key=lambda x: x[1])[:top_k]]
                far_bottomk = [idx for idx, _ in sorted(far_list, key=lambda x: -x[1])[:top_k]]

                # win sample: intersection of top-k in both CSI and FAR
                win_set = set(csi_topk).intersection(far_topk)
                lose_set = set(csi_bottomk).intersection(far_bottomk)

                # 构建 CSI 和 FAR 的 index → rank 字典
                csi_rank = {idx: rank for rank, (idx, _) in enumerate(sorted(csi_list, key=lambda x: -x[1]))}
                far_rank = {idx: rank for rank, (idx, _) in enumerate(sorted(far_list, key=lambda x: x[1]))}

                if win_set:
                    best_win_idx = min(win_set, key=lambda idx: csi_rank[idx] + far_rank[idx])
                    win_idxs.append((frame, best_win_idx))
                else:
                    continue

                if lose_set:
                    best_lose_idx = min(lose_set, key=lambda idx: csi_rank[idx] + far_rank[idx])
                    lose_idxs.append((frame, best_lose_idx))
                else:
                    continue

            # Assemble sequences
            if len(win_idxs)==num_frames and len(lose_idxs)==num_frames:
                win_sequence = np.stack([sample_data[idx][frame] for frame, idx in win_idxs], axis=0)
                lose_sequence = np.stack([sample_data[idx][frame] for frame, idx in lose_idxs], axis=0)

                best_sequence = torch.from_numpy(win_sequence)
                worst_sequence = torch.from_numpy(lose_sequence)
                gt = torch.from_numpy(gt)

                result_win = self.eval_metrics.update_single_sample(
                    target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                    pred=best_sequence.unsqueeze(0),   # shape: (1, 128, 128)
                    sample_names=name
                )
                metrics_win = self.eval_metrics.compute_frame(result_win)

                self.logger.info(f"best far:{metrics_win['avg']['far']}")
                self.logger.info(f"best csi:{metrics_win['avg']['csi']}")

                result_lose = self.eval_metrics.update_single_sample(
                    target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                    pred=worst_sequence.unsqueeze(0),   # shape: (1, 128, 128)
                    sample_names=name
                )
                metrics_lose = self.eval_metrics.compute_frame(result_lose)
                
                self.logger.info(f"worst far:{metrics_lose['avg']['far']}")
                self.logger.info(f"worst csi:{metrics_lose['avg']['csi']}")
            
                print('Success')
                
                win_name = f'rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin/ddim_sample/frame_rank/win_{file}' 
                lose_name = f'rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin/ddim_sample/frame_rank/lose_{file}'

                base_dir = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/temp'
                paths_and_data = [
                    (os.path.join(base_dir, f'win_{file}'), win_name, win_sequence),
                    (os.path.join(base_dir, f'lose_{file}'), lose_name, lose_sequence)
                ]
                
                for save_file_name, cluster, save_data in paths_and_data:
                    self.threadPool.submit(self.write_data, cluster, save_file_name, save_data)
            else:
                continue
       
                          
    def tag(self, txt_path):
    # 作用是读取path，一个个进行定量指标，检测是否是正/负相关，打tag，
    # 如果是正相关，rank，win sample是best or worst csi or far的sample
        from scipy.stats import pearsonr
        path_list = []
        with open(txt_path, 'r') as f:
            files = [line.split(' ')[-1].strip() for line in f.readlines() if 'sample0' in line]
        my_idx = 0
        
        for file in files:
            print(my_idx)
            my_idx = my_idx + 1
            sample_data = []
            for i in range(5):
                datafile = os.path.join('cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin', file)
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
        
            csi_list = []
            far_list = []

            for idx in range(5): 
                result_patch = self.eval_metrics.update_single_sample(
                    target=torch.from_numpy(gt).unsqueeze(0),               
                    pred=torch.from_numpy(sample_data[idx]).unsqueeze(0),   
                    sample_names=name
                )
                metrics = self.eval_metrics.compute_frame(result_patch)
                avg_csi = metrics['avg']['csi']
                avg_far = metrics['avg']['far']
                
                csi_list.append(avg_csi)
                far_list.append(avg_far)

            # 计算相关系数
            corr, _ = pearsonr(csi_list, far_list)

            # 转为 numpy array 以方便排序
            csi_array = np.array(csi_list)
            far_array = np.array(far_list)

            # 取最大CSI的索引 → 看这个样本的FAR在far_array中的排序
            best_csi_idx = np.argmax(csi_array)
            far_rank_when_csi_best = np.argsort(far_array)[::-1].tolist().index(best_csi_idx) + 1  # +1 为1-based排名

            # 取最小FAR的索引 → 看这个样本的CSI在csi_array中的排序
            best_far_idx = np.argmin(far_array)
            csi_rank_when_far_best = np.argsort(csi_array)[::-1].tolist().index(best_far_idx) + 1
            
            csi_max_idx = np.argmax(csi_array)
            csi_min_idx = np.argmin(csi_array)
            far_max_idx = np.argmax(far_array)
            far_min_idx = np.argmin(far_array)
                
      
            sample_csi_best = sample_data[csi_max_idx]
            sample_csi_worst = sample_data[csi_min_idx]
            sample_far_best = sample_data[far_min_idx]
            sample_far_worst = sample_data[far_max_idx]

            result_patch = self.eval_metrics.update_single_sample(
                target=torch.from_numpy(gt).unsqueeze(0),               
                pred=torch.from_numpy(sample_far_best).unsqueeze(0),   
                sample_names=name
            )
                
            metrics = self.eval_metrics.compute_frame(result_patch)
            print(f"csi: {metrics['avg']['csi']}")
            print(f"far: {metrics['avg']['far']}")
            

            result_patch = self.eval_metrics.update_single_sample(
                target=torch.from_numpy(gt).unsqueeze(0),               
                pred=torch.from_numpy(sample_far_worst).unsqueeze(0),   
                sample_names=name
            )
            
            metrics = self.eval_metrics.compute_frame(result_patch)
            print(f"csi: {metrics['avg']['csi']}")
            print(f"far: {metrics['avg']['far']}")
            
            # 判断正负相关
            if corr > 0.3:
                relation = 'PositiveCorrelation'
            elif corr < -0.3:
                relation = 'NegativeCorrelation'
            else:
                relation = 'NoSignificantCorrelation'

            csi_win_name = f'rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin/ddim_sample/{relation}/csi_win_{file}' 
            csi_lose_name = f'rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin/ddim_sample/{relation}/csi_lose_{file}' 
            far_win_name = f'rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin/ddim_sample/{relation}/far_win_{file}' 
            far_lose_name = f'rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin/ddim_sample/{relation}/far_lose_{file}'

            base_dir = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/temp'
            paths_and_data = [
                (os.path.join(base_dir, f'csi_win_{file}'), csi_win_name, sample_csi_best),
                (os.path.join(base_dir, f'csi_lose_{file}'), csi_lose_name, sample_csi_worst),
                (os.path.join(base_dir, f'far_win_{file}'), far_win_name, sample_far_best),
                (os.path.join(base_dir, f'far_lose_{file}'), far_lose_name, sample_far_worst),
            ]

            # self.logger.info(f'{file} - CSI 与 FAR 的相关系数为 {corr:.3f}，呈现 {relation}')
            # self.logger.info(f'    - CSI最好的样本, 其FAR在5个样本中的排名为:{far_rank_when_csi_best}')
            # self.logger.info(f'    - FAR最好的样本, 其CSI在5个样本中的排名为:{csi_rank_when_far_best}')
            
            # 异步保存数据
            for save_file_name, cluster, save_data in paths_and_data:
                self.threadPool.submit(self.write_data, cluster, save_file_name, save_data)


    def get_win_loss_samples_mm(self):
        self.all_metrics =[]
        path = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/test/all/tau.txt'
        with open(path, 'r') as f:
            files = [line.split(' ')[-1].strip() for line in f.readlines()]
        
        index = 0
        # file可以只选择sample0
        
        for file in files:
            print('idx:', index)
            index = index + 1
            sample_data = []
            
            path_list =  [
                'cluster3:s3://zwl2/rankcast/pred_data/sevir_128_10m_3h/flowcast/tau_fin',
                'cluster3:s3://zwl2/rankcast/pred_data/sevir_128_10m_3h/flowcast/EarthFormer_fin',
                'cluster3:s3://zwl2/rankcast/pred_data/sevir_128_10m_3h/flowcast/incepu_fin'
            ]
            for j in range(3):
                datafile = os.path.join(path_list[j], file)
                for i in range(11):
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

            best_samples = []
            worst_samples = []
            
            best_csi = -float('inf')
            worst_csi = float('inf')
            best_far = float('inf')
            worst_far = -float('inf')
                
            best_idx_csi = -1
            worst_idx_csi = -1

            best_idx_far = -1
            worst_idx_far = -1

            for idx in range(33):
                result_patch = self.eval_metrics.update_single_sample(
                    target=torch.from_numpy(gt).unsqueeze(0),               # shape: (1, 128, 128)
                    pred=torch.from_numpy(sample_data[idx]).unsqueeze(0),   # shape: (1, 128, 128)
                    sample_names=name
                )
                metrics = self.eval_metrics.compute_frame(result_patch)
                avg_csi = metrics['avg']['csi']
                avg_far = metrics['avg']['far']
                
                if avg_csi > best_csi:
                    best_csi = avg_csi
                    best_idx_csi = idx

                if avg_csi < worst_csi:
                    worst_csi = avg_csi
                    worst_idx_csi = idx
                
                if avg_far < best_far:
                    best_far = avg_far
                    best_idx_far = idx

                if avg_far > worst_far:
                    worst_far = avg_far
                    worst_idx_far = idx

            
            best_sequence1 = torch.from_numpy(sample_data[best_idx_far])
            worst_sequence1 = torch.from_numpy(sample_data[worst_idx_far])

            best_sequence2 = np.stack(sample_data[best_idx_csi], axis=0)
            worst_sequence2 = np.stack(sample_data[worst_idx_csi], axis=0)
            
            gt = torch.from_numpy(gt)
            
            
            self.visualizer.save_pixel_image(pred_image=best_sequence1.unsqueeze(0), target_img=gt.unsqueeze(0), step=index)
            
            win_name1 = 'mm_frame_win_far_' + name +'.npy'
            self.save_win_lose_data(batch_data=best_sequence1, file_names=win_name1, 
                        dataset_name='sevir_128_10m_3h')
            lose_name1 = 'mm_frame_lose_far_' + name + '.npy'
            self.save_win_lose_data(batch_data=worst_sequence1, file_names=lose_name1, 
                        dataset_name='sevir_128_10m_3h')
            

            win_name2 = 'mm_frame_win_csi_' + name + '.npy'
            self.save_win_lose_data(batch_data=best_sequence2, file_names=win_name2, 
                        dataset_name='sevir_128_10m_3h')
            lose_name2 = 'mm_frame_lose_csi_' + name + '.npy'
            self.save_win_lose_data(batch_data=worst_sequence2, file_names=lose_name2, 
                        dataset_name='sevir_128_10m_3h')

            result_win = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=best_sequence1.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_win = self.eval_metrics.compute_frame(result_win)

            self.logger.info(f"best far:{metrics_win['avg']['far']}")
            self.logger.info(f"best far-csi:{metrics_win['avg']['csi']}")
            
            result_lose = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=worst_sequence1.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_lose = self.eval_metrics.compute_frame(result_lose)
            
            self.logger.info(f"worst far:{metrics_lose['avg']['far']}")
            self.logger.info(f"worst far-csi:{metrics_lose['avg']['csi']}")

            result_win = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=best_sequence2.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_win = self.eval_metrics.compute_frame(result_win)

            self.logger.info(f"best far:{metrics_win['avg']['far']}")
            self.logger.info(f"best far-csi:{metrics_win['avg']['csi']}")
            
            result_lose = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=worst_sequence2.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_lose = self.eval_metrics.compute_frame(result_lose)
            
            self.logger.info(f"worst far:{metrics_lose['avg']['far']}")
            self.logger.info(f"worst far-csi:{metrics_lose['avg']['csi']}")
            
                    
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!Finish!!!!!!!!!!!!!!!!!!!!!!!!!!')


    def get_win_loss_frame_samples_guidance(self):
        self.all_metrics =[]
        path = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/test/guidance/guidance-scale0.005.txt'
        with open(path, 'r') as f:
            files = [line.split(' ')[-1].strip() for line in f.readlines()]
        
        index = 0
        
        for file in files:
            print('idx:', index)
            index = index + 1
            sample_data = []
            
            basepath = f'cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_fintune/guidance-scale{self.guidance_scale}/diffcast_fin/'
            
            for i in range(10):
                try:
                    datafile = basepath + file
                    datafile = datafile.replace('sample0',f'sample{i}')
                    data = self.client.get(datafile)
                    data_file = io.BytesIO(data)
                    data = np.load(data_file, allow_pickle=True)
                    sample_data.append(data)
                except:
                    continue
                        
            name = file.split('/')[-1].replace('.npy','')
            gt_basepath = 'cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin/' + file
            gt_file = re.sub(r'sample\d+_', 'gt_', gt_basepath)
            gt = self.client.get(gt_file)
            gt = io.BytesIO(gt)
            gt = np.load(gt, allow_pickle=True)

            # 初始化保存 best / worst 的信息
            best_samples_far = []
            worst_samples_far = []

            best_samples_csi = []
            worst_samples_csi = []

            for frame in range(18):
                best_csi = -float('inf')
                worst_csi = float('inf')
                best_far = float('inf')
                worst_far = -float('inf')
                
                best_idx_csi = -1
                worst_idx_csi = -1
                best_idx_far = -1
                worst_idx_far = -1

                for idx in range(len(sample_data)):  # 遍历 5 个 sample
                    result_patch = self.eval_metrics.update_frame(
                        target=gt[frame, :, :, :],               # shape: (1, 128, 128)
                        pred=sample_data[idx][frame, :, :, :],   # shape: (1, 128, 128)
                        sample_names=name
                    )
                    metrics = self.eval_metrics.compute_frame(result_patch)
                    avg_csi = metrics['avg']['csi']
                    avg_far = metrics['avg']['far']
                    
                    if avg_csi > best_csi:
                        best_csi = avg_csi
                        best_idx_csi = idx

                    if avg_csi < worst_csi:
                        worst_csi = avg_csi
                        worst_idx_csi = idx
                    
                    if avg_far < best_far:
                        best_far = avg_far
                        best_idx_far = idx

                    if avg_far > worst_far:
                        worst_far = avg_far
                        worst_idx_far = idx

                best_samples_far.append((frame, best_idx_far, best_far))
                worst_samples_far.append((frame, worst_idx_far, worst_far))

                best_samples_csi.append((frame, best_idx_csi, best_csi))
                worst_samples_csi.append((frame, worst_idx_csi, worst_csi))
                
            best_sequence_far = []
            worst_sequence_far = []

            best_sequence_csi = []
            worst_sequence_csi = []
            
            for frame, best_idx, _ in best_samples_far:
                best_sequence_far.append(sample_data[best_idx][frame])  # shape: (1, 128, 128)

            for frame, worst_idx, _ in worst_samples_far:
                worst_sequence_far.append(sample_data[worst_idx][frame])  # shape: (1, 128, 128)

            for frame, best_idx, _ in best_samples_csi:
                best_sequence_csi.append(sample_data[best_idx][frame])  # shape: (1, 128, 128)

            for frame, worst_idx, _ in worst_samples_csi:
                worst_sequence_csi.append(sample_data[worst_idx][frame])  # shape: (1, 128, 128)
                

            best_sequence_far = np.stack(best_sequence_far, axis=0)
            worst_sequence_far = np.stack(worst_sequence_far, axis=0)

            best_sequence_far = torch.from_numpy(best_sequence_far)
            worst_sequence_far = torch.from_numpy(worst_sequence_far)

            best_sequence_csi = np.stack(best_sequence_csi, axis=0)
            worst_sequence_csi = np.stack(worst_sequence_csi, axis=0)

            best_sequence_csi = torch.from_numpy(best_sequence_csi)
            worst_sequence_csi = torch.from_numpy(worst_sequence_csi)

            gt = torch.from_numpy(gt)
            
            
            # self.visualizer.save_pixel_image(pred_image=best_sequence.unsqueeze(0), target_img=gt.unsqueeze(0), step=index)
            
            win_name = f'guidance_{self.guidance_scale}_frame_win_far' + name + '.npy'
            self.save_win_lose_data(batch_data=best_sequence_far, file_names=win_name, 
                        dataset_name='sevir_128_10m_3h')
            lose_name = f'guidance_{self.guidance_scale}frame_lose_far' + name + '.npy'
            self.save_win_lose_data(batch_data=worst_sequence_far, file_names=lose_name, 
                        dataset_name='sevir_128_10m_3h')
            
            
            result_win = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=best_sequence_far.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_win = self.eval_metrics.compute_frame(result_win)

            self.logger.info(f"best far:{metrics_win['avg']['far']}")
            self.logger.info(f"best far-csi:{metrics_win['avg']['csi']}")
            
            result_lose = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=worst_sequence_far.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_lose = self.eval_metrics.compute_frame(result_lose)
            
            self.logger.info(f"worst far:{metrics_lose['avg']['far']}")
            self.logger.info(f"worst far-csi:{metrics_lose['avg']['csi']}")

            win_name = f'guidance_{self.guidance_scale}_frame_win_csi' + name + '.npy'
            self.save_win_lose_data(batch_data=best_sequence_csi, file_names=win_name, 
                        dataset_name='sevir_128_10m_3h')
            lose_name = f'guidance_{self.guidance_scale}_frame_lose_csi' + name + '.npy'
            self.save_win_lose_data(batch_data=worst_sequence_csi, file_names=lose_name, 
                        dataset_name='sevir_128_10m_3h')
            
            result_win = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=best_sequence_csi.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_win = self.eval_metrics.compute_frame(result_win)

            self.logger.info(f"best csi:{metrics_win['avg']['csi']}")
            self.logger.info(f"best csi-far:{metrics_win['avg']['far']}")
            
            result_lose = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=worst_sequence_csi.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_lose = self.eval_metrics.compute_frame(result_lose)
            
            self.logger.info(f"worst csi:{metrics_lose['avg']['csi']}")
            self.logger.info(f"worst csi-far:{metrics_lose['avg']['far']}")
            
                    
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!Finish!!!!!!!!!!!!!!!!!!!!!!!!!!')       


    def get_win_loss_samples_mm(self):
        self.all_metrics =[]
        path = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/test/all/tau.txt'
        with open(path, 'r') as f:
            files = [line.split(' ')[-1].strip() for line in f.readlines()]
        
        index = 0
        # file可以只选择sample0
        
        for file in files:
            print('idx:', index)
            index = index + 1
            sample_data = []
            
            path_list =  [
                'cluster3:s3://zwl2/rankcast/pred_data/sevir_128_10m_3h/flowcast/tau_fin',
                'cluster3:s3://zwl2/rankcast/pred_data/sevir_128_10m_3h/flowcast/EarthFormer_fin',
                'cluster3:s3://zwl2/rankcast/pred_data/sevir_128_10m_3h/flowcast/incepu_fin'
            ]
            for j in range(3):
                datafile = os.path.join(path_list[j], file)
                for i in range(11):
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

            best_samples = []
            worst_samples = []
            
            best_csi = -float('inf')
            worst_csi = float('inf')
            best_far = float('inf')
            worst_far = -float('inf')
                
            best_idx_csi = -1
            worst_idx_csi = -1

            best_idx_far = -1
            worst_idx_far = -1

            for idx in range(33):
                result_patch = self.eval_metrics.update_single_sample(
                    target=torch.from_numpy(gt).unsqueeze(0),               # shape: (1, 128, 128)
                    pred=torch.from_numpy(sample_data[idx]).unsqueeze(0),   # shape: (1, 128, 128)
                    sample_names=name
                )
                metrics = self.eval_metrics.compute_frame(result_patch)
                avg_csi = metrics['avg']['csi']
                avg_far = metrics['avg']['far']
                
                if avg_csi > best_csi:
                    best_csi = avg_csi
                    best_idx_csi = idx

                if avg_csi < worst_csi:
                    worst_csi = avg_csi
                    worst_idx_csi = idx
                
                if avg_far < best_far:
                    best_far = avg_far
                    best_idx_far = idx

                if avg_far > worst_far:
                    worst_far = avg_far
                    worst_idx_far = idx

            
            best_sequence1 = torch.from_numpy(sample_data[best_idx_far])
            worst_sequence1 = torch.from_numpy(sample_data[worst_idx_far])

            best_sequence2 = np.stack(sample_data[best_idx_csi], axis=0)
            worst_sequence2 = np.stack(sample_data[worst_idx_csi], axis=0)
            
            gt = torch.from_numpy(gt)
            
            
            self.visualizer.save_pixel_image(pred_image=best_sequence1.unsqueeze(0), target_img=gt.unsqueeze(0), step=index)
            
            win_name1 = 'mm_frame_win_far_' + name +'.npy'
            self.save_win_lose_data(batch_data=best_sequence1, file_names=win_name1, 
                        dataset_name='sevir_128_10m_3h')
            lose_name1 = 'mm_frame_lose_far_' + name + '.npy'
            self.save_win_lose_data(batch_data=worst_sequence1, file_names=lose_name1, 
                        dataset_name='sevir_128_10m_3h')
            

            win_name2 = 'mm_frame_win_csi_' + name + '.npy'
            self.save_win_lose_data(batch_data=best_sequence2, file_names=win_name2, 
                        dataset_name='sevir_128_10m_3h')
            lose_name2 = 'mm_frame_lose_csi_' + name + '.npy'
            self.save_win_lose_data(batch_data=worst_sequence2, file_names=lose_name2, 
                        dataset_name='sevir_128_10m_3h')

            result_win = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=best_sequence1.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_win = self.eval_metrics.compute_frame(result_win)

            self.logger.info(f"best far:{metrics_win['avg']['far']}")
            self.logger.info(f"best far-csi:{metrics_win['avg']['csi']}")
            
            result_lose = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=worst_sequence1.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_lose = self.eval_metrics.compute_frame(result_lose)
            
            self.logger.info(f"worst far:{metrics_lose['avg']['far']}")
            self.logger.info(f"worst far-csi:{metrics_lose['avg']['csi']}")

            result_win = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=best_sequence2.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_win = self.eval_metrics.compute_frame(result_win)

            self.logger.info(f"best far:{metrics_win['avg']['far']}")
            self.logger.info(f"best far-csi:{metrics_win['avg']['csi']}")
            
            result_lose = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=worst_sequence2.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_lose = self.eval_metrics.compute_frame(result_lose)
            
            self.logger.info(f"worst far:{metrics_lose['avg']['far']}")
            self.logger.info(f"worst far-csi:{metrics_lose['avg']['csi']}")
            
                    
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!Finish!!!!!!!!!!!!!!!!!!!!!!!!!!')


    def get_win_loss_frame_samples_guidance(self):
        self.all_metrics =[]
        path = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/test/guidance/guidance-scale0.005.txt'
        with open(path, 'r') as f:
            files = [line.split(' ')[-1].strip() for line in f.readlines()]
        
        index = 0
        
        for file in files:
            print('idx:', index)
            index = index + 1
            sample_data = []
            
            basepath = f'cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_fintune/guidance-scale{self.guidance_scale}/diffcast_fin/'
            
            for i in range(10):
                try:
                    datafile = basepath + file
                    datafile = datafile.replace('sample0',f'sample{i}')
                    data = self.client.get(datafile)
                    data_file = io.BytesIO(data)
                    data = np.load(data_file, allow_pickle=True)
                    sample_data.append(data)
                except:
                    continue
                        
            name = file.split('/')[-1].replace('.npy','')
            gt_basepath = 'cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin/' + file
            gt_file = re.sub(r'sample\d+_', 'gt_', gt_basepath)
            gt = self.client.get(gt_file)
            gt = io.BytesIO(gt)
            gt = np.load(gt, allow_pickle=True)

            # 初始化保存 best / worst 的信息
            best_samples_far = []
            worst_samples_far = []

            best_samples_csi = []
            worst_samples_csi = []

            for frame in range(18):
                best_csi = -float('inf')
                worst_csi = float('inf')
                best_far = float('inf')
                worst_far = -float('inf')
                
                best_idx_csi = -1
                worst_idx_csi = -1
                best_idx_far = -1
                worst_idx_far = -1

                for idx in range(len(sample_data)):  # 遍历 5 个 sample
                    result_patch = self.eval_metrics.update_frame(
                        target=gt[frame, :, :, :],               # shape: (1, 128, 128)
                        pred=sample_data[idx][frame, :, :, :],   # shape: (1, 128, 128)
                        sample_names=name
                    )
                    metrics = self.eval_metrics.compute_frame(result_patch)
                    avg_csi = metrics['avg']['csi']
                    avg_far = metrics['avg']['far']
                    
                    if avg_csi > best_csi:
                        best_csi = avg_csi
                        best_idx_csi = idx

                    if avg_csi < worst_csi:
                        worst_csi = avg_csi
                        worst_idx_csi = idx
                    
                    if avg_far < best_far:
                        best_far = avg_far
                        best_idx_far = idx

                    if avg_far > worst_far:
                        worst_far = avg_far
                        worst_idx_far = idx

                best_samples_far.append((frame, best_idx_far, best_far))
                worst_samples_far.append((frame, worst_idx_far, worst_far))

                best_samples_csi.append((frame, best_idx_csi, best_csi))
                worst_samples_csi.append((frame, worst_idx_csi, worst_csi))
                
            best_sequence_far = []
            worst_sequence_far = []

            best_sequence_csi = []
            worst_sequence_csi = []
            
            for frame, best_idx, _ in best_samples_far:
                best_sequence_far.append(sample_data[best_idx][frame])  # shape: (1, 128, 128)

            for frame, worst_idx, _ in worst_samples_far:
                worst_sequence_far.append(sample_data[worst_idx][frame])  # shape: (1, 128, 128)

            for frame, best_idx, _ in best_samples_csi:
                best_sequence_csi.append(sample_data[best_idx][frame])  # shape: (1, 128, 128)

            for frame, worst_idx, _ in worst_samples_csi:
                worst_sequence_csi.append(sample_data[worst_idx][frame])  # shape: (1, 128, 128)
                

            best_sequence_far = np.stack(best_sequence_far, axis=0)
            worst_sequence_far = np.stack(worst_sequence_far, axis=0)

            best_sequence_far = torch.from_numpy(best_sequence_far)
            worst_sequence_far = torch.from_numpy(worst_sequence_far)

            best_sequence_csi = np.stack(best_sequence_csi, axis=0)
            worst_sequence_csi = np.stack(worst_sequence_csi, axis=0)

            best_sequence_csi = torch.from_numpy(best_sequence_csi)
            worst_sequence_csi = torch.from_numpy(worst_sequence_csi)

            gt = torch.from_numpy(gt)
            
            
            # self.visualizer.save_pixel_image(pred_image=best_sequence.unsqueeze(0), target_img=gt.unsqueeze(0), step=index)
            
            win_name = f'guidance_{self.guidance_scale}_frame_win_far' + name + '.npy'
            self.save_win_lose_data(batch_data=best_sequence_far, file_names=win_name, 
                        dataset_name='sevir_128_10m_3h')
            lose_name = f'guidance_{self.guidance_scale}frame_lose_far' + name + '.npy'
            self.save_win_lose_data(batch_data=worst_sequence_far, file_names=lose_name, 
                        dataset_name='sevir_128_10m_3h')
            
            
            result_win = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=best_sequence_far.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_win = self.eval_metrics.compute_frame(result_win)

            self.logger.info(f"best far:{metrics_win['avg']['far']}")
            self.logger.info(f"best far-csi:{metrics_win['avg']['csi']}")
            
            result_lose = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=worst_sequence_far.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_lose = self.eval_metrics.compute_frame(result_lose)
            
            self.logger.info(f"worst far:{metrics_lose['avg']['far']}")
            self.logger.info(f"worst far-csi:{metrics_lose['avg']['csi']}")

            win_name = f'guidance_{self.guidance_scale}_frame_win_csi' + name + '.npy'
            self.save_win_lose_data(batch_data=best_sequence_csi, file_names=win_name, 
                        dataset_name='sevir_128_10m_3h')
            lose_name = f'guidance_{self.guidance_scale}_frame_lose_csi' + name + '.npy'
            self.save_win_lose_data(batch_data=worst_sequence_csi, file_names=lose_name, 
                        dataset_name='sevir_128_10m_3h')
            
            result_win = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=best_sequence_csi.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_win = self.eval_metrics.compute_frame(result_win)

            self.logger.info(f"best csi:{metrics_win['avg']['csi']}")
            self.logger.info(f"best csi-far:{metrics_win['avg']['far']}")
            
            result_lose = self.eval_metrics.update_single_sample(
                target=gt.unsqueeze(0),               # shape: (1, 128, 128)
                pred=worst_sequence_csi.unsqueeze(0),   # shape: (1, 128, 128)
                sample_names=name
            )
            metrics_lose = self.eval_metrics.compute_frame(result_lose)
            
            self.logger.info(f"worst csi:{metrics_lose['avg']['csi']}")
            self.logger.info(f"worst csi-far:{metrics_lose['avg']['far']}")
            
                    
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!Finish!!!!!!!!!!!!!!!!!!!!!!!!!!')       
        


    def vis_denoise(self, dataloader, split, start_step=0):
        # 初始化线程池和metrics缓存
        self.threadPool = ThreadPoolExecutor(max_workers=16)
        self.all_metrics = []  # 用于缓存所有metrics结果
        metric_logger = utils.MetricLogger(delimiter="  ", sync=True)
        

        for step, batch in enumerate(dataloader):
            print('step:', step)
            # 数据预处理
            data_dict = {}
            data_dict = self.save_data_preprocess(batch)
            inp, tar = data_dict['inputs'], data_dict['data_samples']
            file_names = data_dict['file_name']
            all_predictions = []
            bs = inp.shape[0]
            print('batch_size:', bs)
            z_sample_prediction = self._denoise(template_data=tar, cond_data=inp, bs=1, file_names=file_names[0])
    
    
    def _flush_metrics_to_excel(self, excel_path='/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/ddpm_ddmpsample2_finetune_results.xlsx'):
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

