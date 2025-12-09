import torch.nn as nn
import torch


if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/lustre/gongjunchao/workdir_lustre/RankCast')

from networks.diffcast.utils.blocks import Downsample, ResnetBlock, ConvGRUCell



# model
class ContextNet(nn.Module):
    def __init__(
        self,
        dim,    # must be same as Unet
        dim_mults=(1, 2, 4, 8),     # must be same as Unet
        channels = 1,
    ):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.dim_mults = dim_mults
        
        self.init_conv = nn.Conv2d(channels, dim, 7, padding = 3)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # dims = [channels, *map(lambda m: dim * m, dim_mults)]
        # in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) -1 )
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_in),
                        ConvGRUCell(dim_in, dim_in, 3, n_layer=1),
                        Downsample(dim_in, dim_out) if not is_last else nn.Identity()
                    ]
                )
            )

    
    def init_state(self, shape, device):
        for i, ml in enumerate(self.downs):
            temp_shape = list(shape)
            temp_shape[-2] //= 2 ** i
            temp_shape[-1] //= 2 ** i
            ml[1].init_hidden(temp_shape, device)
            
    def _forward(self, x):
        x = self.init_conv(x)
        context = []
        for i, (resnet, conv, downsample) in enumerate(self.downs):
            x = resnet(x)
            x = conv(x)
            context.append(x)
            x = downsample(x)
        return context
    
    def forward(self, frames):
        b, t, c, h, w = frames.shape
        state_shape = (b, c, h, w)
        self.init_state(state_shape, frames.device)
        local_ctx = None
        globla_ctx = None
        for i in range(t):
            globla_ctx = self._forward(frames[:,i])
            if i == 5:
                local_ctx = [h.clone() for h in globla_ctx]
        return globla_ctx, local_ctx


if __name__ == "__main__":
    b = 8
    inp_length, pred_length = 10, 12
    totoal_length = inp_length + pred_length
    c = 1
    h, w = 256, 256

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #
    input_data = torch.randn((b, inp_length, c, h, w)).to(device) #torch.randn((b, inp_length, c, h, w)).to(device)
    target = torch.randn((b, pred_length, c, h, w)).to(device)
    
    model = ContextNet(dim=64, dim_mults=(1, 2, 4, 8)).to(device)
    globla_ctx, local_ctx = model(torch.cat([input_data, target], dim=1))
    print(len(globla_ctx), len(local_ctx))
    print(globla_ctx[0].shape, local_ctx[0].shape)
    print(globla_ctx[0].shape, local_ctx[0].shape)

    from fvcore.nn.parameter_count import parameter_count_table
    from fvcore.nn.flop_count import flop_count
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    flops = FlopCountAnalysis(model, torch.cat([input_data, target], dim=1))
    print(flop_count_table(flops))
# srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u contextNet.py #