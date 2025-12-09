import torch
import torch.nn as nn
from functools import partial
from einops import rearrange


if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/lustre/gongjunchao/workdir_lustre/RankCast')

from networks.diffcast.utils.blocks import (Downsample, ResnetBlock, RandomOrLearnedSinusoidalPosEmb,
                        SinusoidalPosEmb, TemporalAttention, Residual, PreNorm, Attention, Upsample,
                        default, exists)



class Unet(nn.Module):
    def __init__(
        self,
        dim,
        T_in,
        dim_mults=(1, 2, 4, 8),
        self_condition = True,
        resnet_block_groups = 8,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = T_in * 2
        self.self_condition = self_condition
        input_channels = self.channels

        init_dim = dim
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        
        
        self.frag_idx_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        # print(str(in_out))

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in * 2, dim_in, time_emb_dim = time_dim * 2),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim * 2),
                Residual(PreNorm(dim_in, TemporalAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim * 2)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim * 2)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim * 2),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim * 2),
                Residual(PreNorm(dim_out, TemporalAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))


        self.out_dim = T_in

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim * 2)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, cond=None, ctx=None, idx=None):
        
        # x: (b, t, c, h, w)
        # cond: (b, t, c, h, w)
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        if exists(cond):
            cond = rearrange(cond, 'b t c h w -> b (t c) h w')
        
        cond = default(cond, lambda: torch.zeros_like(x))
        x = torch.cat((cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        f_idx = self.frag_idx_mlp(idx)
        t = torch.cat((t, f_idx), dim = 1)

        h = []

        for idx, (block1, block2, attn, downsample) in enumerate(self.downs):
            x = block1(torch.cat((x, ctx[idx]),dim=1), t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        
        x = rearrange(x, 'b (t c) h w -> b t c h w', t=self.out_dim)
        return x


if __name__ == "__main__":
    b = 8
    inp_length, pred_length = 10, 12
    totoal_length = inp_length + pred_length
    c = 1
    h, w = 256, 256
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #
    input_data = torch.randn((b, inp_length, c, h, w)).to(device) #torch.randn((b, inp_length, c, h, w)).to(device)
    target = torch.randn((b, pred_length, c, h, w)).to(device) #
    t_steps = torch.full((b,), 15, device = device, dtype = torch.long)
    
    unet = Unet(dim = 64, T_in = inp_length, dim_mults=(1, 2, 4, 8)).to(device)
    
    pass