import torch
import torch.nn as nn

if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/lustre/gongjunchao/workdir_lustre/RankCast')

from networks.diffcast.unet import Unet
from networks.diffcast.contextNet import ContextNet


class diffcast(nn.Module):
    def __init__(
            self, unet_kwargs, ctxNet_kwargs,
            **kwargs
    ):  
        super().__init__()
        self.unet = Unet(**unet_kwargs)
        self.ctxNet = ContextNet(**ctxNet_kwargs)
    
    def forward(self, noise_pred, timesteps, inp, backbone_out):
        """
        noise_pred: (B, T_out, C, H, W) ## epsilon
        time_steps: (B, )
        inp: (B, T_in, C, H, W)  radar
        backbone_out: (B, T_out, C, H, W) prediction of radar
        """
        B = noise_pred.shape[0]
        if backbone_out != None:
            global_ctx, local_ctx = self.ctxNet(torch.cat([inp, backbone_out], dim=1))
        else:
            global_ctx, local_ctx = self.ctxNet(inp)
        x = self.unet(x=noise_pred, time=timesteps, ctx=global_ctx,
                      cond=None, idx=torch.zeros(B, dtype=noise_pred.dtype, device=noise_pred.device))
        
        return x
    
if __name__ == "__main__":
    print('start')
    b = 16
    inp_length, pred_length = 6, 18
    total_length = inp_length + pred_length
    c = 1
    h, w = 128, 128
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #
    input_data = torch.randn((b, inp_length, c, h, w)).to(device) #torch.randn((b, inp_length, c, h, w)).to(device)
    target = torch.randn((b, pred_length, c, h, w)).to(device) #
    timesteps = torch.full((b,), 15, device = device, dtype = torch.float)

    print('load yaml from config')
    import yaml
    from omegaconf import OmegaConf
    cfg_path = '/mnt/lustre/gongjunchao/workdir_lustre/RankCast/configs/DiffCast/sevir.yaml'
    with open(cfg_path, 'r') as cfg_file:
      cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
    
    print('end')
    backbone_kwargs = cfg_params['model']['params']['sub_model']['diffcast']
    model = diffcast(**backbone_kwargs)
    model.to(device)

    import torch.nn.functional as F
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(5):
        start.record()
        pred = model(noise_pred=target, timesteps=timesteps, inp=input_data, backbone_out=target,)
        loss = F.mse_loss(pred, target)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is None:
                print(f'{n} has no grad')
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))
        memory = torch.cuda.memory_reserved() / (1024. * 1024)
        print("memory:", memory)
        
    from fvcore.nn.parameter_count import parameter_count_table
    from fvcore.nn.flop_count import flop_count
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    flops = FlopCountAnalysis(model, (target, timesteps,
                                       input_data, target))
    print(flop_count_table(flops))

# srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:1 python -u model.py #
    