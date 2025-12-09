import torch as th
import numpy as np
import logging

import enum

from . import path
from .utils import EasyDict, log_state, mean_flat
from .integrators import ode, sde
import torch

class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)

class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()

class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan, # 选择这种加噪
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def prior_logp(self, z):
        '''
            Standard multivariate normal prior
            Assume z is batched
        '''
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.
        return th.vmap(_fn)(z)
    

    def check_interval(
        self, 
        train_eps, 
        sample_eps, 
        *, 
        diffusion_form="SBDM",
        sde=False, 
        reverse=False, 
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
            and (self.model_type != ModelType.VELOCITY or sde): # avoid numerical issue by taking a first semi-implicit step

            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        
        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1


    def sample_dpo(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
        # win data与lose data的噪声要相同，shape（2b,t,c,w,h）
        x0 = th.randn_like(x1).chunk(2)[0].repeat(2 ,1 ,1 ,1 ,1)
        
        # 获得采样时间的上下界
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        
        # 得到时间步t
        t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        t = t.chunk(2)[0].repeat(2)
        t = t.to(x1)
        
        return t, x0, x1
    
    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
        # x0是噪声的起点
        x0 = th.randn_like(x1)
        # 获得采样时间的上下界
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        # 得到时间步t
        t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        t = t.to(x1)
        return t, x0, x1


    def training_spo_losses(
            self, 
            model_train,
            model_far,
            model_ref,
            x1, 
            model_kwargs=None
        ):
            """Loss for training the score model
            Args:
            - model: backbone model; could be score, noise, or velocity
            - x1: datapoint
            - model_kwargs: additional arguments for the model
            """
            if model_kwargs == None:
                model_kwargs = {}
                
            t, x0, x1 = self.sample(x1) # x1是被加噪的output，x0是对应的噪声，t是随机的时刻
            t = t.chunk(2)[0].repeat(2)
            x0 = x0.chunk(2)[0].repeat(2, 1, 1, 1, 1) 
            t, xt, ut = self.path_sampler.plan(t, x0, x1) # xt是在t时刻下加噪的状态，ut是该时刻下的速度场 
            tw = t.chunk(2)[0]
            
            dt = 1/1000
            
            # 计算alpha t与w_t
            alpha_t = self.path_sampler.compute_alpha_t(tw)[0]**2/self.path_sampler.compute_alpha_t(tw-dt)[0]**2
            w_t = 0.5*(1/alpha_t)*(1-alpha_t)/(1-self.path_sampler.compute_alpha_t(tw-dt)[0]**2)
            
            import torch 
            train_oup = model_train(xt, t, **model_kwargs)

            with torch.no_grad():
                far_oup = model_far(xt, t, **model_kwargs)
                
            with torch.no_grad():
                ref_oup = model_ref(xt, t, **model_kwargs)

            terms = {}
            terms['train_pred'] = train_oup
            terms['far_pred'] = far_oup
            terms['ref_pred'] = ref_oup
            
            if self.model_type == ModelType.VELOCITY:
                terms['train_loss'] = (train_oup - ut).pow(2).mean(dim=[1,2,3,4])
                terms['far_loss'] = (far_oup - ut).pow(2).mean(dim=[1,2,3,4])
                terms['ref_loss'] = (ref_oup - ut).pow(2).mean(dim=[1,2,3,4])
                terms['w_loss'],terms['l_loss'] = terms['train_loss'].chunk(2)
                terms['far_w_loss'],terms['far_l_loss'] = terms['far_loss'].chunk(2)
                terms['ref_w_loss'],terms['ref_l_loss'] = terms['ref_loss'].chunk(2)
                terms['w_t'] = w_t
                
            else: 
                _, drift_var = self.path_sampler.compute_drift(xt, t)
                sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
                if self.loss_type in [WeightType.VELOCITY]:
                    weight = (drift_var / sigma_t) ** 2
                elif self.loss_type in [WeightType.LIKELIHOOD]:
                    weight = drift_var / (sigma_t ** 2)
                elif self.loss_type in [WeightType.NONE]:
                    weight = 1
                else:
                    raise NotImplementedError()
                
                if self.model_type == ModelType.NOISE:
                    terms['loss'] = mean_flat(weight * ((train_oup - x0) ** 2))
                else:
                    terms['loss'] = mean_flat(weight * ((train_oup * sigma_t + x0) ** 2))
                    
            return terms


    def training_dpo_losses(
            self, 
            model_train,
            model_ref,
            x1, 
            model_kwargs=None
        ):
            """Loss for training the score model
            Args:
            - model: backbone model; could be score, noise, or velocity
            - x1: datapoint
            - model_kwargs: additional arguments for the model
            """
            if model_kwargs == None:
                model_kwargs = {}
                
            t, x0, x1 = self.sample(x1) # x1是被加噪的output，x0是对应的噪声，t是随机的时刻
            t = t.chunk(2)[0].repeat(2)
            x0 = x0.chunk(2)[0].repeat(2, 1, 1, 1, 1) 
            t, xt, ut = self.path_sampler.plan(t, x0, x1) # xt是在t时刻下加噪的状态，ut是该时刻下的速度场 
            tw = t.chunk(2)[0]
            
            dt = 1/1000
            
            # 计算alpha t与w_t
            alpha_t = self.path_sampler.compute_alpha_t(tw)[0]**2/self.path_sampler.compute_alpha_t(tw-dt)[0]**2
            w_t = 0.5*(1/alpha_t)*(1-alpha_t)/(1-self.path_sampler.compute_alpha_t(tw-dt)[0]**2)
            
            # TODO:后面要修改
            import torch 
            train_oup = model_train(xt, t, **model_kwargs)
            with torch.no_grad():
                ref_oup = model_ref(xt, t, **model_kwargs)


            terms = {}
            terms['train_pred'] = train_oup
            terms['ref_pred'] = ref_oup
            
            if self.model_type == ModelType.VELOCITY:
                terms['train_loss'] = (train_oup - ut).pow(2).mean(dim=[1,2,3,4])
                terms['ref_loss'] = (ref_oup - ut).pow(2).mean(dim=[1,2,3,4])
                terms['w_loss'],terms['l_loss'] = terms['train_loss'].chunk(2)
                terms['ref_w_loss'],terms['ref_l_loss'] = terms['ref_loss'].chunk(2)
                terms['w_t'] = w_t
                
            else: 
                _, drift_var = self.path_sampler.compute_drift(xt, t)
                sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
                if self.loss_type in [WeightType.VELOCITY]:
                    weight = (drift_var / sigma_t) ** 2
                elif self.loss_type in [WeightType.LIKELIHOOD]:
                    weight = drift_var / (sigma_t ** 2)
                elif self.loss_type in [WeightType.NONE]:
                    weight = 1
                else:
                    raise NotImplementedError()
                
                if self.model_type == ModelType.NOISE:
                    terms['loss'] = mean_flat(weight * ((train_oup - x0) ** 2))
                else:
                    terms['loss'] = mean_flat(weight * ((train_oup * sigma_t + x0) ** 2))
                    
            return terms

    def training_dspo_losses(
        self, 
        model_train, 
        model_ref, 
        x1, 
        model_kwargs=None
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        if model_kwargs == None:
            model_kwargs = {}
        t, x0, x1 = self.sample(x1) # x1是output，x0是对应的噪声，t是随机的时刻
        
        t = t.chunk(2)[0].repeat(2)        
        t, xt, ut = self.path_sampler.plan(t, x0, x1) # xt是在t时刻下加噪的状态，ut是该时刻下的速度场
        # print('--------t---------', t)
        tw = t.chunk(2)[0]
        
        dt = 1/1000
        epsilon = 1e-6 
        alpha_t = torch.nan_to_num(self.path_sampler.compute_alpha_t(tw)[0]/(self.path_sampler.compute_alpha_t(tw-dt)[0] + epsilon), nan=1.0, posinf=1.0, neginf=-1.0)
        alpha_t = torch.clamp(alpha_t, max=1-epsilon)

        # print('-----------------alpha_t max-------------', alpha_t.max().item())
        # print('-----------------alpha_t min-------------', alpha_t.min().item())
        # print('-----------------alpha_t contains NaN?-------------', torch.isnan(alpha_t).any().item())
            
        train_oup = model_train(xt, t, **model_kwargs)
        with torch.no_grad():
            ref_oup = model_ref(xt, t, **model_kwargs)


        terms = {}
        terms['train_pred'] = train_oup
        terms['ref_pred'] = ref_oup
        
        terms['train_pred_w'],  terms['train_pred_l'] = terms['train_pred'].chunk(2)
        terms['ref_pred_w'],  terms['ref_pred_l'] = terms['ref_pred'].chunk(2)
        terms['ut_w'], terms['ut_l'] = ut.chunk(2)
        
        if self.model_type == ModelType.VELOCITY:
            terms['train_loss'] = (train_oup - ut).pow(2).mean(dim=[1,2,3,4])
            terms['ref_loss'] = (ref_oup - ut).pow(2).mean(dim=[1,2,3,4])
            terms['w_loss'],terms['l_loss'] = terms['train_loss'].chunk(2)
            terms['ref_w_loss'],terms['ref_l_loss'] = terms['ref_loss'].chunk(2)
            r_w = -0.5*(1-alpha_t)/((1-self.path_sampler.compute_alpha_t(tw)[0])*alpha_t + epsilon)*(torch.mean(terms['w_loss'])-torch.mean(terms['ref_w_loss']))
            r_l = -0.5*(1-alpha_t)/((1-self.path_sampler.compute_alpha_t(tw)[0])*alpha_t + epsilon)*(torch.mean(terms['l_loss'])-torch.mean(terms['ref_l_loss']))


            tw = tw
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            term_1 = 1/(1-self.path_sampler.compute_alpha_t(tw)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))**0.5*(-terms['ut_w']+ terms['train_pred_w'] + epsilon)

            var1 = (1 - self.path_sampler.compute_alpha_t(tw)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))

            # print('-----------------var1 max-------------', var1.max().item())
            # print('-----------------var1 min-------------', var1.min().item())
            # print('-----------------var1 contains NaN?-------------', torch.isnan(var1).any().item())

            # # 约束 var1，避免除零
            var1 = torch.clamp(var1, min=1e-8)  # epsilon 增大，防止除零

            var2 = (alpha_t**0.5)

            # print('-----------------var2 max-------------', var2.max().item())
            # print('-----------------var2 min-------------', var2.min().item())
            # print('-----------------var2 contains NaN?-------------', torch.isnan(var2).any().item())

            var3 = terms['train_pred_w'] - terms['ref_pred_w']

            # print('-----------------var3 max-------------', var3.max().item())
            # print('-----------------var3 min-------------', var3.min().item())
            # print('-----------------var3 contains NaN?-------------', torch.isnan(var3).any().item())

            # 计算 var4，避免除零
            var4 = 1 / (var1**0.5 + epsilon)

            # print('-----------------var4 max-------------', var4.max().item())
            # print('-----------------var4 min-------------', var4.min().item())
            # print('-----------------var4 contains NaN?-------------', torch.isnan(var4).any().item())

            # 计算 var5，避免除零
            var5 = 1 / (var2 + epsilon)

            # print('-----------------var5 max-------------', var5.max().item())
            # print('-----------------var5 min-------------', var5.min().item())
            # print('-----------------var5 contains NaN?-------------', torch.isnan(var5).any().item())
            
            # var6 = terms['train_pred_w']
            # print('-----------------var6 max-------------', var6.max().item())
            # print('-----------------var6 min-------------', var6.min().item())
            # print('-----------------var6 contains NaN?-------------', torch.isnan(var6).any().item())
            
            term_2 = (1/var1**0.5 + epsilon) * var3


            Lambda = 5
            r_w = r_w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            r_l = r_l.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # print('term_1', torch.mean(term_1))
            # print('term_2', torch.mean(term_2))
            # print('torch.sigmoid(r_w-r_l)', torch.mean(torch.sigmoid(r_w-r_l)))
            # print('(term_1 - Lambda * term_2*(1-torch.sigmoid(r_w-r_l)))', torch.mean(term_1 - Lambda * term_2*(1-torch.sigmoid(r_w-r_l))))
            # print('loss', torch.mean((term_1 - Lambda * term_2*(1-torch.sigmoid(r_w-r_l)))**2))
            
            return terms, torch.mean((term_1 - Lambda * term_2*(1-torch.sigmoid(r_w-r_l)))**2)
        
        else: 
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()
            
            if self.model_type == ModelType.NOISE:
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
                
        return terms
    
    # def training_dspo_losses(
    #     self, 
    #     model_train, 
    #     model_ref, 
    #     x1, 
    #     model_kwargs=None
    # ):
    #     """Loss for training the score model
    #     Args:
    #     - model: backbone model; could be score, noise, or velocity
    #     - x1: datapoint
    #     - model_kwargs: additional arguments for the model
    #     """
    #     if model_kwargs == None:
    #         model_kwargs = {}
    #     t, x0, x1 = self.sample(x1) # x1是output，x0是对应的噪声，t是随机的时刻
        
    #     t = t.chunk(2)[0].repeat(2)        
    #     t, xt, ut = self.path_sampler.plan(t, x0, x1) # xt是在t时刻下加噪的状态，ut是该时刻下的速度场
        
    #     tw = t.chunk(2)[0]
        
    #     dt = 1/1000
    #     epsilon = 1e-6 

    #     # 计算 alpha_t
    #     alpha_t_numerator = self.path_sampler.compute_alpha_t(tw)[0]
    #     alpha_t_denominator = self.path_sampler.compute_alpha_t(tw - dt)[0] + epsilon
    #     alpha_t = alpha_t_numerator / alpha_t_denominator
    #     alpha_t = torch.clamp(alpha_t, max=1 - epsilon)

    #     print("alpha_t_numerator min:", alpha_t_numerator.min().item(), "max:", alpha_t_numerator.max().item())
    #     print("alpha_t_denominator min:", alpha_t_denominator.min().item(), "max:", alpha_t_denominator.max().item())
    #     print("alpha_t min:", alpha_t.min().item(), "max:", alpha_t.max().item())

    #     # 计算模型输出
    #     train_oup = model_train(xt, t, **model_kwargs)
    #     with torch.no_grad():
    #         ref_oup = model_ref(xt, t, **model_kwargs)

    #     # 组织 terms 变量
    #     terms = {
    #         'train_pred': train_oup,
    #         'ref_pred': ref_oup
    #     }

    #     # 分割变量
    #     terms['train_pred_w'], terms['train_pred_l'] = terms['train_pred'].chunk(2)
    #     terms['ref_pred_w'], terms['ref_pred_l'] = terms['ref_pred'].chunk(2)
    #     terms['ut_w'], terms['ut_l'] = ut.chunk(2)

    #     print("train_pred_w min:", terms['train_pred_w'].min().item(), "max:", terms['train_pred_w'].max().item())
    #     print("ref_pred_w min:", terms['ref_pred_w'].min().item(), "max:", terms['ref_pred_w'].max().item())
    #     print("ut_w min:", terms['ut_w'].min().item(), "max:", terms['ut_w'].max().item())

    #     if self.model_type == ModelType.VELOCITY:
    #         # 计算 loss
    #         terms['train_loss'] = (train_oup - ut).pow(2).mean(dim=[1,2,3,4])
    #         terms['ref_loss'] = (ref_oup - ut).pow(2).mean(dim=[1,2,3,4])
            
    #         terms['w_loss'], terms['l_loss'] = terms['train_loss'].chunk(2)
    #         terms['ref_w_loss'], terms['ref_l_loss'] = terms['ref_loss'].chunk(2)

    #         # 计算 r_w 和 r_l
    #         denom = (1 - self.path_sampler.compute_alpha_t(tw)[0]) * alpha_t
    #         denom = torch.clamp(denom, min=epsilon)  # 避免除零
    #         r_w = -0.5 * (1 - alpha_t) / denom * (torch.mean(terms['w_loss']) - torch.mean(terms['ref_w_loss']) + epsilon)
    #         r_l = -0.5 * (1 - alpha_t) / denom * (torch.mean(terms['l_loss']) - torch.mean(terms['ref_l_loss']) + epsilon)

    #         print("r_w min:", r_w.min().item(), "max:", r_w.max().item())
    #         print("r_l min:", r_l.min().item(), "max:", r_l.max().item())

    #         # 计算 term_1
    #         one_minus_alpha_t = 1 - self.path_sampler.compute_alpha_t(tw)[0]
    #         one_minus_alpha_t = torch.clamp(one_minus_alpha_t, min=epsilon)  # 避免除零

    #         sqrt_one_minus_alpha_t = one_minus_alpha_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) ** 0.5
    #         inv_sqrt_one_minus_alpha_t = 1 / (sqrt_one_minus_alpha_t + epsilon)

    #         term_1 = inv_sqrt_one_minus_alpha_t * (-terms['ut_w'] + terms['train_pred_w'] + epsilon)

    #         print("term_1 min:", term_1.min().item(), "max:", term_1.max().item())

    #         # 计算 term_2
    #         sqrt_alpha_t = alpha_t ** 0.5
    #         inv_sqrt_alpha_t = 1 / (sqrt_alpha_t + epsilon)

    #         diff = terms['train_pred_w'] - terms['ref_pred_w']

    #         print("diff min:", diff.min().item(), "max:", diff.max().item())

    #         term_2 = inv_sqrt_one_minus_alpha_t * inv_sqrt_alpha_t * diff

    #         # 检查 term_2 是否包含 NaN
    #         if torch.isnan(term_2).any():
    #             print("term_2 contains NaN!")

    #         # 打印 term_2 的最大最小值
    #         print("term_2 min:", term_2.min().item(), "max:", term_2.max().item())

    #         Lambda = 0.5
    #         r_w = r_w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #         r_l = r_l.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    #         # print('term_1', torch.mean(term_1))
    #         print('term_2', torch.mean(term_2))
    #         # print('torch.sigmoid(r_w-r_l)', torch.mean(torch.sigmoid(r_w-r_l)))
    #         # print('(term_1 - Lambda * term_2*(1-torch.sigmoid(r_w-r_l)))', torch.mean(term_1 - Lambda * term_2*(1-torch.sigmoid(r_w-r_l))))
    #         print('loss', torch.mean((term_1 - Lambda * term_2*(1-torch.sigmoid(r_w-r_l)))**2))
            
    #         return terms, torch.mean((term_1 - Lambda * term_2*(1-torch.sigmoid(r_w-r_l)))**2)
        
    #     else: 
    #         _, drift_var = self.path_sampler.compute_drift(xt, t)
    #         sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
    #         if self.loss_type in [WeightType.VELOCITY]:
    #             weight = (drift_var / sigma_t) ** 2
    #         elif self.loss_type in [WeightType.LIKELIHOOD]:
    #             weight = drift_var / (sigma_t ** 2)
    #         elif self.loss_type in [WeightType.NONE]:
    #             weight = 1
    #         else:
    #             raise NotImplementedError()
            
    #         if self.model_type == ModelType.NOISE:
    #             terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
    #         else:
    #             terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
                
    #     return terms
    
    # def training_dpo_losses(
    #     self, 
    #     model,  
    #     x1, 
    #     model_kwargs=None
    # ):
    #     """Loss for training the score model
    #     Args:
    #     - model: backbone model; could be score, noise, or velocity
    #     - x1: datapoint
    #     - model_kwargs: additional arguments for the model
    #     """
    #     if model_kwargs == None:
    #         model_kwargs = {}
    #     # dpo change:win sample与lose sample的time与noise保持一致
    #     t, x0, x1 = self.sample(x1) # x1是被加噪的output，x0是对应的噪声，t是随机的时刻
    #     t = t.chunk(2)[0].repeat(2)
    #     x0 = x0.chunk(2)[0].repeat(2, 1, 1, 1, 1) 
    #     t, xt, ut = self.path_sampler.plan(t, x0, x1) # xt是在t时刻下加噪的状态，ut是该时刻下的速度场
    #     tw = t.chunk(2)[0]
        
    #     dt = 1/1000
        
    #     # 计算alpha t与w_t
    #     alpha_t = self.path_sampler.compute_alpha_t(tw)[0]**2/self.path_sampler.compute_alpha_t(tw-dt)[0]**2
    #     w_t = 0.5*(1/alpha_t)*(1-alpha_t)/(1-self.path_sampler.compute_alpha_t(tw-dt)[0]**2)
        
    #     # TODO:后面要修改
    #     import torch 
    #     # model_output =  torch.randn(16, 18, 1, 128, 128).to('cuda')
    #     model_output = model(xt, t, **model_kwargs)


    #     B, *_, C = xt.shape
    #     assert model_output.size() == (B, *xt.size()[1:-1], C)

    #     terms = {}
    #     terms['pred'] = model_output
    #     if self.model_type == ModelType.VELOCITY:
    #         terms['loss'] = (model_output - ut).pow(2).mean(dim=[1,2,3,4])
    #         terms['w_loss'],terms['l_loss'] = terms['loss'].chunk(2)
    #         terms['w_t'] = w_t
            
    #     else: 
    #         _, drift_var = self.path_sampler.compute_drift(xt, t)
    #         sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
    #         if self.loss_type in [WeightType.VELOCITY]:
    #             weight = (drift_var / sigma_t) ** 2
    #         elif self.loss_type in [WeightType.LIKELIHOOD]:
    #             weight = drift_var / (sigma_t ** 2)
    #         elif self.loss_type in [WeightType.NONE]:
    #             weight = 1
    #         else:
    #             raise NotImplementedError()
            
    #         if self.model_type == ModelType.NOISE:
    #             terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
    #         else:
    #             terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
                
    #     return terms
    
    
    def training_losses(
        self, 
        model,  
        x1, 
        model_kwargs=None
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        if model_kwargs == None:
            model_kwargs = {}
        t, x0, x1 = self.sample(x1) # t(,8)，x1是待预测的变量
        t, xt, ut = self.path_sampler.plan(t, x0, x1) # xt是在t时刻下加噪的状态，ut是该时刻下的速度场
        model_output = model(xt, t, **model_kwargs)
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        terms = {}
        terms['pred'] = model_output
        if self.model_type == ModelType.VELOCITY:
            terms['loss'] = mean_flat(((model_output - ut) ** 2))
            
        else: 
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()
            
            if self.model_type == ModelType.NOISE:
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
                
        return terms
    

    def get_drift(
        self
    ):
        """member function for obtaining the drift of the probability flow ODE"""
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output) # by change of variable
        
        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)
        
        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode
        
        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn
    

    def get_score(
        self,
    ):
        """member function for obtaining score of 
            x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        else:
            raise NotImplementedError()
        
        return score_fn


class Sampler:
    """Sampler class for the transport model"""
    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """
        
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()
    
    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):

        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion
        
        sde_drift = \
            lambda x, t, model, **kwargs: \
                self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
    
        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion
    
    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""
    
        if last_step is None:
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x
        elif last_step == "Mean":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)
            

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample
    
    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )
        
        return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):
        
        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """
        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)
        
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            # x是纯噪声,model是flow match的预测model，kwargs里面有inp（condition）pred1
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn