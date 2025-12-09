import torch
from vit.model import ViT
from util.evaluation import plot_inp_result, val_model
from datasets.Inf_Radar_st1 import CMA_Dataset
from petrel_client.client import Client
import time
import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from util.s3_client import s3_client
import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.backends.cudnn as cudnn

client = Client(conf_path="~/petreloss.conf")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from datetime import datetime
import torch
import torch.distributed
import core.logger as Logger
import core.metrics as Metrics
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from tqdm import tqdm
import math
from collections import defaultdict

def renorm_input_data(data):
    # mean and std for the first 6 channels
    means = torch.array([0.08, 275.12, 265.25, 253.35, 239.82, 0.12])
    stds = torch.array([0.12, 20.56, 22.45, 17.42, 10.79, 0.19])
    
    # min and max for the 7th channel
    min_val = 0
    max_val = 48.00
    
    # Create a copy of the data to hold the denormalized result
    denormalized_data = torch.zeros_like(data)
    
    # Apply denormalization to the first 6 channels
    for i in range(6):
        denormalized_data[:, i, :, :] = data[:, i, :, :] * stds[i] + means[i]
    
    # Apply denormalization to the 7th channel
    denormalized_data[:, 6, :, :] = data[:, 6, :, :] * (max_val - min_val) + min_val
    
    # The 8th channel remains unchanged (assuming no normalization was applied)
    denormalized_data[:, 7, :, :] = data[:, 7, :, :]
    
    return denormalized_data

def renorm_label_data(data, mask=None):
    mean = 13.763
    std = 11.767
    
    # Ensure mean and std are on the same device as the input data
    device = data.device
    mean = torch.tensor(mean, device=device)
    std = torch.tensor(std, device=device)
    
    denormalized_label = data * std + mean
    if mask is not None:
        denormalized_label = torch.where(mask.bool(), denormalized_label, torch.tensor(-1280.0))
        
    denormalized_label = denormalized_label.numpy()
    
    return denormalized_label

def unnormalize_input(normalized_tensor):
    # 每个通道的max和min值
    inchannel_max = torch.tensor([300.0, 250.0, 300.0, 50.0])
    inchannel_min = torch.tensor([200.0, 200.0, 200.0, 0.1])

    unnormalized_tensor = torch.empty_like(normalized_tensor)
    for i in range(normalized_tensor.shape[1]):  # 第二个维度是通道数
        unnormalized_tensor[:, i] = normalized_tensor[:, i] * (inchannel_max[i] - inchannel_min[i]) + inchannel_min[i]
    return unnormalized_tensor

def unnormalize_label(normalized_label, mask=None):
    denormalized_label = normalized_label * 45.0
    if mask is not None:
        denormalized_label = torch.where(mask.bool(), denormalized_label, torch.tensor(-1280.0))
    denormalized_label = denormalized_label.numpy()
    return denormalized_label

def reverse_sqrtlog_minmax_norm(transformed_data, mask=None):
    min_val = 0.0
    max_val = 50.0
    if isinstance(transformed_data, torch.Tensor):
        transformed_data = transformed_data.detach().cpu().numpy()
        
    sqrt_data = np.expm1(transformed_data)
    
    norm_data = sqrt_data ** 2
    
    # 反转 min-max 归一化
    original_data = norm_data * (max_val - min_val) + min_val
    if mask is not None:
        original_data = torch.where(mask.bool(), original_data, torch.tensor(-1280.0))
        
    # original_data = original_data.numpy()
    
    return original_data
    
def visualize3(label,full_prediction, save_path):
    rgb_colors = [
        (245, 253, 255),
        (111, 239, 255),  # 5
        (95, 207, 239),   # 10
        (79, 175, 223),   # 15
        (47,  95, 191),   # 20
        (31,  63, 175),   # 25
        (15,  31, 159),   # 30
        (247, 239,  63),  # 35
        (239, 191,  55),  # 40
        (231, 143,  47),  # 45
        (207,  15,  23),  # 50
        (183,   7,  15),  # 55
        (159,   0,   8),  # 60
    ]

    hex_colors = ['#%02x%02x%02x' % color for color in rgb_colors]

    bounds = [-32, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

    norm = BoundaryNorm(bounds, ncolors=len(hex_colors))
    cmap = ListedColormap(hex_colors, N=len(hex_colors))

    fig, axs = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)

    im1 = axs[0].contourf(label, levels=bounds, cmap=cmap, norm=norm)
    axs[0].set_title('Label')

    # im2 = axs[1].contourf(prediction_mask, levels=bounds, cmap=cmap, norm=norm)
    # axs[1].set_title('Prediction with mask')

    im2 = axs[1].contourf(full_prediction, levels=bounds, cmap=cmap, norm=norm)
    axs[1].set_title('Full Prediction')

    # cbar = fig.colorbar(im3, ax=axs.ravel().tolist(), orientation='vertical', ticks=bounds, shrink=0.5, pad=0.05)
    # cbar.ax.set_ylabel('CR Intensity')
    cbar = fig.colorbar(im2, ax=axs, orientation='vertical', ticks=bounds, shrink=0.8, pad=0.05)
    # cbar = fig.colorbar(im1, ax=axs, orientation='horizontal', ticks=bounds, shrink=0.8, pad=0.15)
    cbar.ax.set_ylabel('Prcp Intensity')

    # plt.tight_layout()
    plt.savefig(save_path)
    # print('final_result save to:', save_path)
    plt.close()
    
def calculate_mse_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return mse, float('inf')
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    return mse, psnr

def restore_large_image(patches, coords, image_shape=(1501,1751), patch_size=256):
    """
    将小图根据坐标还原为完整大图。
    
    Args:
        patches (list of np.ndarray): 小图列表。
        coords (list of tuple): 每个小图的 (x, y) 坐标。
        image_shape (tuple): 大图的目标形状 (height, width)。
        patch_size (int): 小图的大小。

    Returns:
        np.ndarray: 还原后的完整大图。
    """
    large_image = np.zeros(image_shape, dtype=patches[0].dtype)

    for patch, (x, y) in zip(patches, coords):
        x = int(x)
        y = int(y)
        large_image[x:x + patch_size, y:y + patch_size] = patch
    
    return large_image

def write_data(name, local_path): 
    client = s3_client(bucket_name='zwl2', endpoint='http://10.135.0.241:80', user='zhangwenlong', jiqun = 'p')
    ceph_path = os.path.join('cma_radar_vit_st1/256x256_patch', name) 
    print(ceph_path) 
    res = client.upload_file(f'{local_path}', 'zwl2', ceph_path)
    os.remove(local_path)

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    
    parser.add_argument('--name', default='vit', type=str)
    parser.add_argument('--config', default='/mnt/petrelfs/xukaiyi/CodeSpace/cma_model_vit/config/vit_10M_radar_st1.yaml', type=str)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--phase', default='train',type=str)
    
    # wandblogger
    parser.add_argument('--enable_wandb', default=True, help='use wandb or not')
    
    parser.add_argument('--log_wandb_ckpt', default=True, help='use wandb or not')
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    
    # ？
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')
    # ？
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    
    # 判断需不需要finetune
    parser.add_argument('--ckptPath', default = '/mnt/petrelfs/xukaiyi/CodeSpace/cma_model_vit/experiments/vit_241201_124837/checkpoint-30.pth', help='resume from checkpoint')
    parser.add_argument('--ckpt', default = False, help='resume from checkpoint')
    
    # Optimizer parameters
    
    # 一般默认值是多少
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # 一般默认值是多少
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    # ？
    parser.add_argument('--blr', type=float, default=2e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    # ？
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    # ？
    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--break_after_epoch', type=int, metavar='N', help='break training after X epochs, to tune hyperparams and avoid messing with training schedule')


    # Dataset parameters
    parser.add_argument('--data_path', default='/shared/yossi_gandelsman/arxiv/arxiv_data/', type=str,
                        help='dataset path')
    parser.add_argument('--data_path_val', default='/shared/yossi_gandelsman/arxiv/arxiv_data/', type=str,
                        help='val dataset path')
    parser.add_argument('--imagenet_percent', default=1, type=float)
    parser.add_argument('--subsample', action='store_true')
    parser.set_defaults(subsample=False)
    parser.add_argument('--output_dir', default='/mnt/petrelfs/xukaiyi/CodeSpace/cma_model_vit/experiments',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')


    return parser

def inf(epoch):
    args = get_args_parser()
    args = args.parse_args()
    
    # import util.misc as misc
    # from util.misc import NativeScalerWithGradNormCount as NativeScaler

    # misc.init_distributed_mode(args)
    # device = torch.device(args.device)
    # seed = 3407
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    
    
    # cudnn.benchmark = True


    from data.Inf_Radar_test_st1 import CMA_Dataset
    dataset_cma_val = CMA_Dataset('/mnt/petrelfs/xukaiyi/CodeSpace/cma_model_vit/data/test_st2_path.txt', phase='train')
    data_loader_val = torch.utils.data.DataLoader(
            dataset_cma_val,
            batch_size=1,
            num_workers=8)

#     if True: 
#         num_tasks = misc.get_world_size()
#         global_rank = misc.get_rank()
#         sampler_val = torch.utils.data.DistributedSampler(
#             dataset_cma_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
#         )
#     else:
#         sampler_train = torch.utils.data.RandomSampler(dataset_train)

#     data_loader_val = torch.utils.data.DataLoader(
#     dataset_cma_val, sampler=sampler_val,
#     batch_size=1,
#     num_workers=8,
#     pin_memory=args.pin_mem,
#     drop_last=True,
# )
        
    model = ViT(image_size=[1520,1800], patch_size=[40, 40], in_chans=4, out_chans=1, dim=512, depth=8, heads=8, mlp_dim=512, dim_head=32).to(device)
    checkpoint = torch.load(f'/mnt/petrelfs/xukaiyi/CodeSpace/cma_model_vit/Experiment/250307_133941/checkpoint-99.pth', map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model'])
    
    
    rmse = 0.0
    idx = 0
    RMSElist = []
    thresholds = [5.,15.,25.,30.,40.]
    total_pod = {thr: 0.0 for thr in thresholds}
    total_far = {thr: 0.0 for thr in thresholds}
    total_csi = {thr: 0.0 for thr in thresholds}
    
    current_time_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result_path = '/mnt/petrelfs/xukaiyi/CodeSpace/cma_model_vit/experiments'+current_time_filename
    os.makedirs(result_path, exist_ok=True)
    
    # 初始化分组结构
    grouped_sr_patches = defaultdict(list)
    grouped_hr_patches = defaultdict(list)
    grouped_coords = defaultdict(list)

    save_dir = '/mnt/petrelfs/xukaiyi/CodeSpace/cma_model_vit/VitInfResult'
    
    from tqdm import tqdm

    for _, val_data in enumerate(tqdm(data_loader_val, desc="Processing validation data")):
        idx += 1

        inp = val_data['SR']
        oup = val_data['HR']
        
        print("idx:", idx)
        
        inp, oup = inp.to(device), oup.to(device)

        model.eval()
        target = model(inp)
        
    
        # oup, target = oup.cpu().numpy(), target.detach().cpu().numpy()
        
        
        sr_squeezed = torch.squeeze(target)
        hr_squeezed = torch.squeeze(oup)
        
        sr_squeezed = sr_squeezed.detach().cpu().numpy()
        hr_squeezed = hr_squeezed.detach().cpu().numpy()
        
        name = val_data['Name'][0]
        x = int(val_data['x'] )        
        y = int(val_data['y'] )

        new_x = int(x/4)
        new_y = int(y/4)
        
        renorm_sr_i = (sr_squeezed[0:4*256, 0:4*256] * 65.0)
        renorm_hr_i = (hr_squeezed[0:4*256, 0:4*256] * 65.0)
        
        renorm_sr_resized = cv2.resize(renorm_sr_i, (256, 256), interpolation=cv2.INTER_LINEAR)
        # renorm_sr = reverse_sqrtlog_minmax_norm(sr_squeezed)  
        # renorm_hr = reverse_sqrtlog_minmax_norm(hr_squeezed)
        
        pred_filename = f"pred_{name}_{new_x}_{new_y}.npy"  # Predicted 文件名
        np.save(save_dir +'/'+ pred_filename, renorm_sr_resized)
        write_data(pred_filename, save_dir +'/'+ pred_filename)
        
        # gt_filename = f"gt_{name}_{x}_{y}.npy"  # gt 文件名
        # np.save(save_dir +'/'+ gt_filename, renorm_hr)
        # write_data(gt_filename, save_dir +'/'+ gt_filename)
        
        
    #     grouped_sr_patches[name].append(renorm_sr)
    #     grouped_hr_patches[name].append(renorm_hr)
    #     grouped_coords[name].append((x, y))
        

        # save_path='{}/{}_{}_{}.png'.format(result_path, name, x, y)
        # visualize3(renorm_hr_i, renorm_sr_i, save_path)
        # print(f"Visualization saved for {name}: {save_path}")


    # # 存储最终还原的大图
    # restored_sr_images = {}
    # restored_hr_images = {}

    # # 遍历每个分组
    # for name in grouped_sr_patches:
    #     # 还原大图
    #     restored_sr_images[name] = restore_large_image(
    #         grouped_sr_patches[name], grouped_coords[name])
    #     restored_hr_images[name] = restore_large_image(
    #         grouped_hr_patches[name], grouped_coords[name])

    
    # RMSElist = []
    # total_pod = {thr: 0.0 for thr in thresholds}
    # total_far = {thr: 0.0 for thr in thresholds}
    # total_csi = {thr: 0.0 for thr in thresholds}
    
    # idx = 0
    # # 逐个还原大图并可视化
    # for name in restored_sr_images:
    #     idx = idx+1
    #     renorm_hr = restored_hr_images[name]
    #     renorm_sr = restored_sr_images[name]
        
    #     # 计算指标
    #     mse,_ = calculate_mse_psnr(renorm_sr, renorm_hr)
    #     RMSElist.append(math.sqrt(mse))

    #     #todo 可以加上某个结果的200个sample全过程用来做分析。
    #     for thr in thresholds:                       
    #         has_event_target = (renorm_hr >= thr) 
    #         has_event_predict = (renorm_sr >= thr)
            
    #         hit = np.sum(has_event_target & has_event_predict).astype(int)
    #         miss = np.sum(has_event_target & ~has_event_predict ).astype(int)
    #         false_alarm = np.sum(~has_event_target & has_event_predict ).astype(int)
    #         no_event = np.sum(~has_event_target).astype(int)
            
    #         pod = hit / (hit + miss) if (hit + miss) > 0 else float(2)
    #         far = false_alarm / no_event if no_event > 0 else float(2)
    #         csi = hit / (hit + miss + false_alarm) if (hit + miss + false_alarm) > 0 else float(2)
            
    #         total_pod[thr] += pod
    #         total_far[thr] += far
    #         total_csi[thr] += csi
            
    #         # print(f"Threshold: {thr}, POD: {pod:.4f}, FAR: {far:.4f}, CSI: {csi:.4f}")
    #         # print('# validation # RMSE: {:.4e}'.format(math.sqrt(mse)))
            
        
    #     AVG_RMSE = sum(RMSElist)/len(RMSElist)   
    #     for thr in thresholds:
    #         avg_pod = total_pod[thr] / idx
    #         avg_far = total_far[thr] / idx
    #         avg_csi = total_csi[thr] / idx
    #         print(f"Threshold: {thr}, AVG POD: {avg_pod:.4f}, AVG FAR: {avg_far:.4f}, AVG CSI: {avg_csi:.4f}")
    #         print('# validation # RMSE: {:.4e}'.format(AVG_RMSE))      
        
    #     save_path = f"{result_path}/{name}.png"

    #     visualize3(renorm_hr, renorm_sr, save_path=save_path)

    #     print(f"Visualization saved for {name}: {save_path}")

#     print('begin to val')
#     begin_val_time = time.time()
#     val_model(model, data_loader_val, epoch, '/mnt/petrelfs/zhouzhiwang/codeespace/cma_data/cma_model_vit/ckpt_result',device=device)

#     end_val_time = time.time()
#     print('end to val')
#     print(f'{epoch} val time {end_val_time-begin_val_time}')


inf(49)