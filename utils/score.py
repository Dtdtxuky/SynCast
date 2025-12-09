r"""Score modules"""

import math
import torch
import torch.nn as nn

from torch import Size, Tensor
from tqdm import tqdm
from typing import *
#from zuko.utils import broadcast
from .nn import *
from .SiT import SiT
import h5py
class TimeEmbedding(nn.Sequential):
    r"""Creates a time embedding.

    Arguments:
        features: The number of embedding features.
    """

    def __init__(self, features: int):
        super().__init__(
            nn.Linear(32, 256),
            nn.SiLU(),
            nn.Linear(256, features),
        )

        self.register_buffer('freqs', torch.pi * torch.arange(1, 16 + 1))

    def forward(self, t: Tensor) -> Tensor:
        t = self.freqs * t.unsqueeze(dim=-1)
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        return super().forward(t)


class ScoreNet(nn.Module):
    r"""Creates a score network.

    Arguments:
        features: The number of features.
        context: The number of context features.
        embedding: The number of time embedding features.
    """

    def __init__(self, features: int, context: int = 0, embedding: int = 16, **kwargs):
        super().__init__()

        self.embedding = TimeEmbedding(embedding)
        self.network = ResMLP(features + context + embedding, features, **kwargs)

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        t = self.embedding(t)

        if c is None:
            x, t = broadcast(x, t, ignore=1)
            x = torch.cat((x, t), dim=-1)
        else:
            x, t, c = broadcast(x, t, c, ignore=1)
            x = torch.cat((x, t, c), dim=-1)

        return self.network(x)


class ScoreUNet(nn.Module):
    r"""Creates a U-Net score network.

    Arguments:
        channels: The number of channels.
        context: The number of context channels.
        embedding: The number of time embedding features.
    """

    def __init__(self, channels: int, context: int = 0, embedding: int = 64, **kwargs):
        super().__init__()

        self.embedding = TimeEmbedding(embedding)
        self.network = UNet(channels + context, channels, embedding, **kwargs)

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        dims = self.network.spatial + 1

        if c is None:
            y = x
        else:
            y = torch.cat(broadcast(x, c, ignore=dims), dim=-dims)
            

        y = y.reshape(-1, *y.shape[-dims:])
        t = t.reshape(-1)
        t = self.embedding(t)

        return self.network(y, t).reshape(x.shape)

class ScoreSiT(nn.Module):
    r"""Creates a SiT score network.

    Arguments:
        channels: The number of channels.
        context: The number of context channels.
        embedding: The number of time embedding features.
    """

    def __init__(self, channels: int = 256, context: int = 0, input_size=72, patch_size=2, depth=12, hidden_size=1152, num_heads=12,):
        super().__init__()

        # self.embedding = TimeEmbedding(embedding)
        self.network = SiT(input_size=input_size,
                        patch_size=patch_size,
                        in_channels=channels,
                        depth=depth, 
                        hidden_size=hidden_size, 
                        num_heads=num_heads)

    def forward(self, x: Tensor, t: Tensor, context: Tensor = None, cross = True) -> Tensor:
        '''
        dims=3
        if c is None:
            y = x
        else:
            y = torch.cat(broadcast(x, c, ignore=dims), dim=-dims)

        y = y.reshape(-1, *y.shape[-dims:])
        '''
        t = t.reshape(-1)
        
        return self.network(x, context, t).reshape(x.shape)

class MCScoreWrapper(nn.Module):
    r"""Disguises a `ScoreUNet` as a score network for a Markov chain."""

    def __init__(self, score: nn.Module):
        super().__init__()

        self.score = score

    def forward(
        self,
        x: Tensor,  # (B, L, C, H, W)
        t: Tensor,  # ()
        c: Tensor = None,  # TODO
    ) -> Tensor:
        return self.score(x.transpose(1, 2), t, c).transpose(1, 2)


class MCScoreNet(nn.Module):
    r"""Creates a score network for a Markov chain.

    Arguments:
        features: The number of features.
        context: The number of context features.
        order: The order of the Markov chain.
    """

    def __init__(self, features: int, context: int = 0, order: int = 1, **kwargs):
        super().__init__()

        self.order = order

        if kwargs.get('spatial', 0) > 0:
            build = ScoreUNet
        else:
            build = ScoreNet

        self.kernel = build(features * (2 * order + 1), context, **kwargs)

    def forward(
        self,
        x: Tensor,  # (B, L, C, H, W)
        t: Tensor,  # ()
        c: Tensor = None,  # (C', H, W)
    ) -> Tensor:
        if not self.order:
            s = self.kernel(x, t, c)
        else:
            x = self.unfold(x, self.order)
            s = self.kernel(x, t, c)
            s = self.fold(s, self.order)

        return s

    @staticmethod
    @torch.jit.script_if_tracing
    def unfold(x: Tensor, order: int) -> Tensor:
        x = x.unfold(1, 2 * order + 1, 1)
        x = x.movedim(-1, 2)
        x = x.flatten(2, 3)

        return x

    @staticmethod
    @torch.jit.script_if_tracing
    def fold(x: Tensor, order: int) -> Tensor:
        x = x.unflatten(2, (2 * order  + 1, -1))

        return torch.cat((
            x[:, 0, :order],
            x[:, :, order],
            x[:, -1, -order:],
        ), dim=1)


class LocalScoreUNet(ScoreUNet):
    r"""Creates a score U-Net with a forcing channel."""

    def __init__(
        self,
        channels: int,
        size: int = 64,
        **kwargs,
    ):
        super().__init__(channels, 1, **kwargs)

        domain = 2 * torch.pi / size * (torch.arange(size) + 1 / 2)
        forcing = torch.sin(4 * domain).expand(1, size, size).clone()

        self.register_buffer('forcing', forcing)

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        return super().forward(x, t, self.forcing)

class VPSDE(nn.Module):
    r"""Creates a noise scheduler for the variance preserving (VP) SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = 1 - \alpha(t)^2 + \eta^2

    Arguments:
        eps: A noise estimator :math:`\epsilon_\phi(x, t)`.
        shape: The event shape.
        alpha: The choice of :math:`\alpha(t)`.
        eta: A numerical stability term.
    """

    def __init__(
        self,
        eps: nn.Module,
        shape: Size,
        alpha: str = 'cos',
        eta: float = 1e-3,
    ):
        super().__init__()

        self.eps = eps
        self.shape = shape
        self.dims = tuple(range(-len(shape), 0))
        self.eta = eta

        if alpha == 'lin':
            self.alpha = lambda t: 1 - (1 - eta) * t
        elif alpha == 'cos':
            self.alpha = lambda t: torch.cos(math.acos(math.sqrt(eta)) * t) ** 2
        elif alpha == 'exp':
            self.alpha = lambda t: torch.exp(math.log(eta) * t**2)
        else:
            raise ValueError()

        self.register_buffer('device', torch.empty(()))

    def mu(self, t: Tensor) -> Tensor:
        return self.alpha(t)

    def sigma(self, t: Tensor) -> Tensor:
        return (1 - self.alpha(t) ** 2 + self.eta ** 2).sqrt()

    def forward(self, x: Tensor, t: Tensor, train: bool = False) -> Tensor:
        r"""Samples from the perturbation kernel :math:`p(x(t) | x)`."""

        t = t.reshape(t.shape + (1,) * len(self.shape))

        eps = torch.randn_like(x)
        x = self.mu(t) * x + self.sigma(t) * eps

        if train:
            return x, eps
        else:
            return x

    def sample(
        self,
        shape: Size = (),
        c: Tensor = None, #bkg field as condtion
        guidance: Tensor = None, #it would be the obs guidance
        steps: int = 256,
        corrections: int = 4,
        tau: float = 0.5,
    ) -> Tensor:
        r"""Samples from :math:`p(x(0))`.

        Arguments:
            shape: The batch shape.
            c: The optional context.
            steps: The number of discrete time steps.
            corrections: The number of Langevin corrections per time steps.
            tau: The amplitude of Langevin steps.
        """
        if isinstance(shape, int):
            shape = (shape,)
        x = torch.randn(shape + self.shape).to(self.device)
        x = x.reshape(-1, *self.shape)

        time = torch.linspace(1, 0, steps + 1).to(self.device)
        dt = 1 / steps

        with torch.no_grad():
            for step, t in enumerate(time[:-1]):
                r = self.mu(t - dt) / self.mu(t)
                
                x = r * x + (self.sigma(t - dt) - r * self.sigma(t)) * self.eps(x, t, c)

                # Corrector
                for _ in range(corrections):
                    z = torch.randn_like(x)
                    eps = self.eps(x, t - dt, c)
                    delta = tau / eps.square().mean(dim=self.dims, keepdim=True)

                    x = x - (delta * eps + torch.sqrt(2 * delta) * z) * self.sigma(t - dt)
                
                '''
                if (step) % 5 ==0:
                    with h5py.File("./test/noguide_emab32256data_0_.h5", "a") as f:
                        print(x.shape)
                        f.create_dataset("sampled_{}".format(step), data=x.detach().cpu().numpy())
                        #f.create_dataset("truth", data=y_target.detach().cpu().numpy())
                '''
                

        return x.reshape(shape + self.shape)

    def sample_laop(
        self,
        A: Callable[[Tensor], Tensor],
        shape: Size = (),
        c: Tensor = None, #bkg field as condtion
        guidance: Tensor = None, #it would be the obs guidance
        steps: int = 128,
        corrections: int = 2,
        tau: float = 0.5,
    ) -> Tensor:
        r"""Samples from :math:`p(x(0))`.

        Arguments:
            shape: The batch shape.
            c: The optional context.
            steps: The number of discrete time steps.
            corrections: The number of Langevin corrections per time steps.
            tau: The amplitude of Langevin steps.
        """
        if isinstance(shape, int):
            shape = (shape,)
        x = torch.randn(shape + self.shape).to(self.device)
        x = x.reshape(-1, *self.shape)

        time = torch.linspace(1, 0, steps + 1).to(self.device)
        dt = 1 / steps

        with torch.no_grad():
            for step, t in enumerate(time[:-1]):
                r = self.mu(t - dt) / self.mu(t)
                
                x = r * x + (self.sigma(t - dt) - r * self.sigma(t)) * self.eps(x, t, c)

                # Corrector
                for _ in range(corrections):
                    z = torch.randn_like(x)
                    eps = self.eps(x, t - dt, c)
                    delta = tau / eps.square().mean(dim=self.dims, keepdim=True)

                    x = x - (delta * eps + torch.sqrt(2 * delta) * z) * self.sigma(t - dt)
                if (step+1)%2 ==0:
                    x_,loss = self.latent_optimization(guidance,x,A)
                    sigma = 100 * (1-self.mu(t-dt)**2)/(self.mu(t)**2) * (1 - self.mu(t)**2 / self.mu(t-dt)**2)
                    x = self.resample(x_,x,sigma,self.mu(t))
                    #print('loss:',loss)
                '''
                if (step) % 5 ==0:
                    with h5py.File("./test/20__laop_guide99_emab32256data_0_.h5", "a") as f:
                        print(x.shape)
                        f.create_dataset("sampled_{}".format(step), data=x.detach().cpu().numpy())
                        #f.create_dataset("truth", data=y_target.detach().cpu().numpy())
                '''
                
                

        return x.reshape(shape + self.shape)
    
    def latent_optimization(self, measurement, x_init, operator_fn, eps=1e-3, max_iters=20, lr=None):

        """
        Function to compute argmin_z ||y - A( D(x) )||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            z_init:                Starting point for optimization
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations
        
        Optimal parameters seem to be at around 500 steps, 200 steps for inpainting.

        """

        # Base case
        if not x_init.requires_grad:
            x_init = x_init.requires_grad_()

        if lr is None:
            lr_val = 1e-2
        else:
            lr_val = lr.item()

        loss = torch.nn.MSELoss() # MSE loss
        optimizer = torch.optim.AdamW([x_init], lr=lr_val) # Initializing optimizer ###change the learning rate
        measurement = measurement.detach() # Need to detach for weird PyTorch reasons
        # Training loop
        init_loss = 0
        losses = []
        
        with torch.enable_grad():
            for itr in range(max_iters):
                optimizer.zero_grad()
                output = loss(measurement, operator_fn( x_init  ))          

                if itr == 0:
                    init_loss = output.detach().clone()
                
                output.backward() # Take GD step
                optimizer.step()
                cur_loss = output.detach().cpu().numpy() 

                # Convergence criteria

                if itr < 10: # may need tuning for early stopping
                    losses.append(cur_loss)
                else:
                    losses.append(cur_loss)
                    if losses[0] < cur_loss:
                        break
                    else:
                        losses.pop(0)
                    
                if cur_loss < eps**2:  # needs tuning according to noise level for early stopping
                    break


        return x_init, init_loss       
    
    def resample(self,xt_opt,xt,sigma,mu):
        """
        Function to resample x_t based on ReSample paper.
        """
        noise = torch.randn_like(xt_opt, device=xt.device)
        
        return (sigma * mu * xt_opt + (1 - mu**2) * xt)/(sigma + 1 - mu**2) + noise * torch.sqrt(1/(1/sigma + 1/(1-mu**2)))

    def loss(self, x: Tensor, c: Tensor = None, w: Tensor = None) -> Tensor:
        r"""Returns the denoising loss."""
        #c is the bkg conditions

        t = torch.rand(x.shape[0], dtype=x.dtype, device=x.device)
        x, eps = self.forward(x, t, train=True)

        return self.eps(x, t, c), eps
        '''
        err = (self.eps(x, t, c) - eps).square()
        if w is None:
            return err.mean()
        else:
            return (err * w).mean() / w.mean()
        '''
    def dpoloss(self, ref_model, xw,xl,c):
        tw = torch.rand(xw.shape[0], dtype=xw.dtype, device=xw.device) # 随机生成时刻(0,1)之间

        dt = 1/1000
        alpha_t = self.mu(tw)**2/self.mu(tw-dt)**2

        wt = 0.5*(1/alpha_t)*(1-alpha_t)/(1-self.mu(tw-dt)**2)
        #print(wt)

        tw = tw.reshape(tw.shape + (1,) * len(self.shape))
        tl = tw

        eps_w = torch.randn_like(xw)
        
        noised_xw = self.mu(tw) * xw + self.sigma(tw) * eps_w
        noised_xl = self.mu(tl) * xl + self.sigma(tl) * eps_w
        
        ref_noise_prediction_xw = ref_model(noised_xw,tw,c)
        ref_noise_prediction_xl = ref_model(noised_xl,tl,c)

        noise_prediction_xw = self.eps(noised_xw,tw,c)
        noise_prediction_xl = self.eps(noised_xl,tl,c)

        
        return noise_prediction_xw, noise_prediction_xl, ref_noise_prediction_xw, ref_noise_prediction_xl, eps_w, eps_w, wt
    
    def dspoloss(self, ref_model, xw,xl,c):
        tw = torch.rand(xw.shape[0], dtype=xw.dtype, device=xw.device) # w_sample的timestep
        noised_xw, eps_w = self.forward(xw, tw, train=True) # 加噪之后的w_sample, noise
        tl = tw 
        noised_xl, eps_l = self.forward(xl, tl, train=True)

        ref_noise_prediction_xw = ref_model(noised_xw,tw,c)
        ref_noise_prediction_xl = ref_model(noised_xl,tl,c)

        noise_prediction_xw = self.eps(noised_xw,tw,c)
        noise_prediction_xl = self.eps(noised_xl,tl,c)

        dt = 1/1000
        alpha_t = self.mu(tw)/self.mu(tw-dt)
        r_w = -0.5*(1-alpha_t)/((1-self.mu(tw))*alpha_t)*(torch.mean((eps_w-noise_prediction_xw)**2)-torch.mean((eps_w-ref_noise_prediction_xw)**2))
        r_l = -0.5*(1-alpha_t)/((1-self.mu(tw))*alpha_t)*(torch.mean((eps_l-noise_prediction_xl)**2)-torch.mean((eps_l-ref_noise_prediction_xl)**2))

        tw = tw
        alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        term_1 = 1/(1-self.mu(tw).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))**0.5*(-eps_w+noise_prediction_xw)
        term_2 = (1/(1-self.mu(tw).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))**0.5)*(1/alpha_t**0.5)*(noise_prediction_xw-ref_noise_prediction_xw)

        Lambda = 10
        r_w = r_w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        r_l = r_l.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return torch.mean((term_1 - Lambda*term_2*(1-torch.sigmoid(r_w-r_l)))**2)

    #def dspoloss(self, ref_model, xw,xl,c):



class SubVPSDE(VPSDE):
    r"""Creates a noise scheduler for the sub-variance preserving (sub-VP) SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = (1 - \alpha(t)^2 + \eta)^2
    """

    def sigma(self, t: Tensor) -> Tensor:
        return 1 - self.alpha(t) ** 2 + self.eta


class SubSubVPSDE(VPSDE):
    r"""Creates a noise scheduler for the sub-sub-VP SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = (1 - \alpha(t) + \eta)^2
    """

    def sigma(self, t: Tensor) -> Tensor:
        return 1 - self.alpha(t) + self.eta


class DPSGaussianScore(nn.Module):
    r"""Creates a score module for Gaussian inverse problems.

    .. math:: p(y | x) = N(y | A(x), Σ)

    References:
        | Diffusion Posterior Sampling for General Noisy Inverse Problems (Chung et al., 2022)
        | https://arxiv.org/abs/2209.14687

    Note:
        This module returns :math:`-\sigma(t) s(x(t), t | y)`.
    """

    def __init__(
        self,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        sde: VPSDE,
        zeta: float = 1.0,
    ):
        super().__init__()

        self.register_buffer('y', y)

        self.A = A
        self.sde = sde
        self.zeta = zeta

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            eps = self.sde.eps(x, t)
            x_ = (x - sigma * eps) / mu
            err = (self.y - self.A(x_)).square().sum()

        s, = torch.autograd.grad(err, x)
        s = -s * self.zeta / err.sqrt()

        return eps - sigma * s


class GaussianScore(nn.Module):
    r"""Creates a score module for Gaussian inverse problems.

    .. math:: p(y | x) = N(y | A(x), Σ)

    Note:
        This module returns :math:`-\sigma(t) s(x(t), t | y)`.
    """

    def __init__(
        self,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        std: Union[float, Tensor],
        sde: VPSDE,
        gamma: Union[float, Tensor] = 1e-2,
        detach: bool = False,
        scale_factor =1
    ):
        super().__init__()

        self.register_buffer('y', y)
        self.register_buffer('std', torch.as_tensor(std))
        self.register_buffer('gamma', torch.as_tensor(gamma))

        self.A = A
        self.scale_factor = scale_factor
        self.sde = sde
        self.detach = detach

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        if self.detach:
            eps = self.sde.eps(x, t, c)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            if not self.detach:
                eps = self.sde.eps(x, t, c)

            x_ = (x - sigma * eps) / mu

            err = self.y - self.A(x_)
            var = self.std ** 2 + self.gamma * (sigma / mu) ** 2

            log_p = -(err ** 2 / var).sum() / 2 / self.y.numel() 

        # s, = torch.autograd.grad(log_p, x) * self.scale_factor
        s, = torch.autograd.grad(log_p, x) # 
        s *= self.scale_factor

        # print('err:', err.mean().item(), err.abs().mean().item())
        # print('mean:',eps.mean().item(),s.mean().item())
        # print('max:',eps.max().item(),s.max().item())
        # print('min:',eps.min().item(),s.min().item())
        # print(self.std ** 2, self.gamma * (sigma / mu) ** 2)
        # print(sigma)
        if s.max() > 1:
            s = s/s.max()

        return eps - sigma * s
    

class GuidanceSampling(nn.Module):
    r"""Creates a score module for Gaussian inverse problems.

    .. math:: p(y | x) = N(y | A(x), Σ)

    Note:
        This module returns :math:`-\sigma(t) s(x(t), t | y)`.
    """

    def __init__(
        self,
        A: Callable[[Tensor], Tensor],
        std: Union[float, Tensor],
        sde: VPSDE,
        gamma: Union[float, Tensor] = 1e-2,
        detach: bool = False,
        scale_factor =1e10,
        device = "cuda",
        shape = (69,32,64)
    ):
        super().__init__()

        
        self.register_buffer('std', torch.as_tensor(std))
        self.register_buffer('gamma', torch.as_tensor(gamma))

        self.A = A
        self.scale_factor = scale_factor
        self.sde = sde
        self.detach = detach
        self.device = device
        self.shape = shape
        self.dims = tuple(range(-len(shape), 0))

    def corrected_score(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        # get the correction term at timestep t
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)
        
        if self.detach:
            eps = self.sde.eps(x, t, c)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            if not self.detach:
                eps = self.sde.eps(x, t, c)

            x_ = (x - sigma * eps) / mu

            err = self.y - self.A(x_) #self.y is the guidance (observation)
            err = err*self.obs_mask
            var = self.std ** 2 + self.gamma * (sigma / mu) ** 2

            log_p = -(err ** 2 / var).sum() / 2 / self.y.numel() 

        s, = torch.autograd.grad(log_p, x) # 
        #print(s)
        s *= self.scale_factor
        
        
        if s.max() > 1:
            s = 2*s/s.max()
        #print(sigma*s/eps)
        #print((sigma*s/eps).max())

        return eps - sigma * s
    
    def sample(
        self,
        shape: Size = (),
        c: Tensor = None, #bkg field as condtion
        guidance: Tensor = None, #it would be the obs guidance
        obs_mask: Tensor = None,
        steps: int = 256,
        corrections: int = 4,
        tau: float = 0.5,
    ) -> Tensor:
        r"""Samples from :math:`p(x(0))`.

        Arguments:
            shape: The batch shape.
            c: The optional context.
            steps: The number of discrete time steps.
            corrections: The number of Langevin corrections per time steps.
            tau: The amplitude of Langevin steps.
        """
        if isinstance(shape, int):
            shape = (shape,)
        x = torch.randn(shape + self.shape).to(self.device)
        x = x.reshape(-1, *self.shape)

        time = torch.linspace(1, 0, steps + 1).to(self.device)
        dt = 1 / steps
        self.y = guidance
        self.obs_mask = obs_mask

        
        with torch.no_grad():
            for step, t in enumerate(time[:-1]):
                r = self.sde.mu(t - dt) / self.sde.mu(t)
                
                x = r * x + (self.sde.sigma(t - dt) - r * self.sde.sigma(t)) * self.corrected_score(x, t, c)

                # Corrector
                for _ in range(corrections):
                    z = torch.randn_like(x)
                    eps = self.corrected_score(x, t - dt, c)
                    delta = tau / eps.square().mean(dim=self.dims, keepdim=True)

                    x = x - (delta * eps + torch.sqrt(2 * delta) * z) * self.sde.sigma(t - dt)
                '''
                if (step) % 5 ==0:
                    with h5py.File("./test/guide50_emab32256data_0_.h5", "a") as f:
                        print(x.shape)
                        f.create_dataset("sampled_{}".format(step), data=x.detach().cpu().numpy())
                #
                '''
                #

        return x.reshape(shape + self.shape)