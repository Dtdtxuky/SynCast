if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/lustre/gongjunchao/workdir_lustre/RankCast')
import torch
import numpy as np
from typing import  Any, Optional, Sequence
from torchmetrics import Metric
from utils.misc import is_dist_avail_and_initialized
import torch.distributed as dist
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import einops
import torch.nn.functional as F
from einops import rearrange
import scipy
from torchvision.transforms.functional import center_crop

from einops import rearrange




@torch.no_grad()
def cal_SSIM(gt, pred, is_img=True):
    '''
    iter_cal=True, gt.shape=pred.shape=[nb b t c h w]
    iter_cal=Fasle, gt.shape=pred.shape=[n t c h w]
    '''
    cal_ssim = StructuralSimilarityIndexMeasure(data_range=int(torch.max(gt)-torch.min(gt)) ).to(gt.device)
    if is_img:
        pred = torch.maximum(pred, torch.min(gt))
        pred = torch.minimum(pred, torch.max(gt))
    pred = einops.rearrange(pred, 'n t c h w -> (n t) c h w')
    gt = einops.rearrange(gt, 'n t c h w -> (n t) c h w')
    ssim = cal_ssim(pred, gt).cpu()
    
    # print(ssim)
    # ssim = cal_ssim_2(pred, gt).cpu()
    
    return ssim.item()

@torch.no_grad()
def cal_PSNR(gt, pred, is_img=True):
    '''
    gt.shape=pred.shape=[n t c h w]
    '''
    cal_psnr = PeakSignalNoiseRatio(data_range=255).to(gt.device)
    if is_img:
        pred = torch.maximum(pred, torch.min(gt))
        pred = torch.minimum(pred, torch.max(gt))
    pred = einops.rearrange(pred, 'n t c h w -> (n t) c h w')
    gt = einops.rearrange(gt, 'n t c h w -> (n t) c h w')
    psnr = 0
    for n in range(pred.shape[0]):
        psnr += cal_psnr(pred[n], gt[n]).cpu()
    return (psnr / pred.shape[0]).item()

@torch.no_grad()
def cal_MSE(gt, pred):
    """
    gt with shape [n t c h w]
    pred with shape [n t c h w]
    """
    mse = torch.mean((gt - pred)**2)
    return mse.item()

@torch.no_grad()
def cal_MAE(gt, pred):
    """
    gt with shape [n t c h w]
    pred with shape [n t c h w]
    """
    mae = torch.mean(torch.abs(gt - pred))
    return mae.item()

@torch.no_grad()
def cal_CRPS(gt, pred, type='avg', scale=4, mode='mean', eps=1e-10):
    """
    gt: (b, t, c, h, w)
    pred: (b, n, t, c, h, w)
    """
    assert mode in ['mean', 'raw'], 'CRPS mode should be mean or raw'
    _normal_dist = torch.distributions.Normal(0, 1)
    _frac_sqrt_pi = 1 / np.sqrt(np.pi)

    b, n, t, _, _, _ = pred.shape
    gt = rearrange(gt, 'b t c h w -> (b t) c h w')
    pred = rearrange(pred, 'b n t c h w -> (b n t) c h w')
    if type == 'avg':
        pred = F.avg_pool2d(pred, scale, stride=scale)
        gt = F.avg_pool2d(gt, scale, stride=scale)
    elif type == 'max':
        pred = F.max_pool2d(pred, scale, stride=scale)
        gt = F.max_pool2d(gt, scale, stride=scale)
    else:
        gt = gt
        pred = pred
    gt = rearrange(gt, '(b t) c h w -> b t c h w', b=b)
    pred = rearrange(pred, '(b n t) c h w -> b n t c h w', b=b, n=n)

    pred_mean = torch.mean(pred, dim=1)
    pred_std = torch.std(pred, dim=1) if n > 1 else torch.zeros_like(pred_mean)
    normed_diff = (pred_mean - gt + eps) / (pred_std + eps)
    cdf = _normal_dist.cdf(normed_diff)
    pdf = _normal_dist.log_prob(normed_diff).exp()

    crps = (pred_std + eps) * (normed_diff * (2 * cdf - 1) + 2 * pdf - _frac_sqrt_pi)
    if mode == "mean":
        return torch.mean(crps).item()
    return crps.item()
    

def _threshold(target, pred ,T):
    t = (target >= T).float()
    p = (pred >= T).float()
    is_nan = torch.logical_or(torch.isnan(target),
                              torch.isnan(pred))
    t[is_nan] = 0
    p[is_nan] = 0
    return t, p

    
class MeteoNetSkillScore(object):
    def __init__(self, seq_len,
                  no_ssim=True, threholds=None, mode='0',
                  layout = 'NTCHW',
                  preprocess_type = 'MeteoNet', dist_eval=False,
                  metrics_list=['csi', 'csi-4-avg', 'csi-16-avg',
                               'csi-4-max', 'csi-16-max', 'bias',
                                'sucr', 'pod', 'hss', 'far'],
                  eps=1e-4):
        self.metrics_list = metrics_list
        self.eps=eps
        self.layout = layout
        self.preprocess_type = preprocess_type
        self.dist_eval = dist_eval
        
        # self.dbz_thresholds = np.array([0.1, 5, 10, 15, 20]) if threholds is None else threholds
        self.dbz_thresholds = np.array([19, 28, 35, 40, 47]) if threholds is None else threholds
        self._g_thresholds = [self.dbz_to_pixel(threshold) for threshold in self.dbz_thresholds]
        self.threshold_list = self._g_thresholds

        self._seq_len = seq_len
        self._no_ssim = no_ssim
        # self._total_batch_num = 0
        # self.begin()

        self.mode = mode
        
        if mode in ("0", ):
            self.keep_seq_len_dim = False
            state_shape = (len(self.threshold_list), )
        elif mode in ("1", "2"):
            self.keep_seq_len_dim = True
            assert isinstance(self.seq_len, int), "seq_len must be provided when we need to keep seq_len dim."
            state_shape = (len(self.threshold_list), self.seq_len)
        else:
            raise NotImplementedError(f"mode {mode} not supported!")
        
        self.hits = torch.zeros(state_shape)
        self.misses = torch.zeros(state_shape)
        self.fas = torch.zeros(state_shape)
        self.cor = torch.zeros(state_shape)

        ## pooling csi ##
        self.hits_avg_pool_4 = torch.zeros(state_shape)
        self.misses_avg_pool_4 = torch.zeros(state_shape)
        self.fas_avg_pool_4 = torch.zeros(state_shape)

        self.hits_max_pool_4 = torch.zeros(state_shape)
        self.misses_max_pool_4 = torch.zeros(state_shape)
        self.fas_max_pool_4 = torch.zeros(state_shape)

        self.hits_avg_pool_16 = torch.zeros(state_shape)
        self.misses_avg_pool_16 = torch.zeros(state_shape)
        self.fas_avg_pool_16 = torch.zeros(state_shape)

        self.hits_max_pool_16 = torch.zeros(state_shape)
        self.misses_max_pool_16 = torch.zeros(state_shape)
        self.fas_max_pool_16 = torch.zeros(state_shape)


    def pod(self, hits, misses, fas, eps):
        return hits / (hits + misses + eps)
    
    def far(self, hits, misses, fas, eps):
        return fas / (hits + fas + eps)

    def csi(self, hits, misses, fas, eps):
        return hits / (hits + misses + fas + eps)
    
    def sucr(self, hits, misses, fas, eps):
        return hits / (hits + fas + eps)

    def bias(self, hits, misses, fas, eps):
        bias = (hits + fas) / (hits + misses + eps)
        logbias = torch.pow(bias / torch.log(torch.tensor(2.0)), 2.0)
        return logbias
    
    def hss(self, hits, misses, fas, cor, eps):
        hss = 2 * (hits * cor - misses * fas) / ((hits + misses) * (misses + cor) + (hits + fas) * (fas + cor) + eps)
        return hss
    
    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        dist.barrier()
        dist.all_reduce(self.hits)
        dist.all_reduce(self.misses)
        dist.all_reduce(self.fas)
        ### avg 4 ###
        dist.all_reduce(self.hits_avg_pool_4)
        dist.all_reduce(self.misses_avg_pool_4)
        dist.all_reduce(self.fas_avg_pool_4)
        ### max 4 ###
        dist.all_reduce(self.hits_max_pool_4)
        dist.all_reduce(self.misses_max_pool_4)
        dist.all_reduce(self.fas_max_pool_4)
        ### avg 16 ###
        dist.all_reduce(self.hits_avg_pool_16)
        dist.all_reduce(self.misses_avg_pool_16)
        dist.all_reduce(self.fas_avg_pool_16)
        ### max 16 ###
        dist.all_reduce(self.hits_max_pool_16)
        dist.all_reduce(self.misses_max_pool_16)
        dist.all_reduce(self.fas_max_pool_16)
    
    @property
    def hits_misses_fas_reduce_dims(self):
        if not hasattr(self, "_hits_misses_fas_reduce_dims"):
            seq_dim = self.layout.find('T')
            self._hits_misses_fas_reduce_dims = list(range(len(self.layout)))
            if self.keep_seq_len_dim:
                self._hits_misses_fas_reduce_dims.pop(seq_dim)
        return self._hits_misses_fas_reduce_dims
    
    def preprocess(self, pred, target):
        if self.preprocess_type == "MeteoNet":
            pred = pred.detach() 
            target = target.detach()
        else:
            raise NotImplementedError
        return pred, target
    
    def preprocess_pool(self, pred, target, pool_size=4, type='avg'):
        if self.preprocess_type == "MeteoNet":
            pred = pred.detach()
            target = target.detach() 
        else:
            raise NotImplementedError
        b, t, _, _, _ = pred.shape
        pred = rearrange(pred, 'b t c h w -> (b t) c h w')
        target = rearrange(target, 'b t c h w -> (b t) c h w')
        if type == 'avg':
            pred = F.avg_pool2d(pred, kernel_size=pool_size, stride=pool_size)
            target = F.avg_pool2d(target, kernel_size=pool_size, stride=pool_size)
        elif type == 'max':
            pred = F.max_pool2d(pred, kernel_size=pool_size, stride=pool_size)
            target = F.max_pool2d(target, kernel_size=pool_size, stride=pool_size)
        pred = rearrange(pred, '(b t) c h w -> b t c h w', b=b)
        target = rearrange(target, '(b t) c h w -> b t c h w', b=b)
        return pred, target
    
    def calc_seq_hits_misses_fas(self, pred, target, threshold):
        with torch.no_grad():
            t, p = _threshold(target, pred, threshold)
            hits = torch.sum(t * p, dim=self.hits_misses_fas_reduce_dims).int()
            misses = torch.sum(t * (1 - p), dim=self.hits_misses_fas_reduce_dims).int()
            fas = torch.sum((1 - t) * p, dim=self.hits_misses_fas_reduce_dims).int()
            cor = torch.sum((1 - t) * (1 - p), dim=self.hits_misses_fas_reduce_dims).int()
        return hits, misses, fas, cor
    
    def dbz_to_pixel(self, dbz):
        pixel_vals = dbz / 70.0
        return pixel_vals

    @torch.no_grad()
    def update(self, pred, target):
        ## pool 1 ##
        self.hits = self.hits.to(pred.device)
        self.misses = self.misses.to(pred.device)
        self.fas = self.fas.to(pred.device)
        self.cor = self.cor.to(pred.device)

        _pred, _target = self.preprocess(pred, target)
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits[i] += hits
            self.misses[i] += misses
            self.fas[i] += fas
            self.cor[i] += cor
        ## max pool 4 ##
        self.hits_max_pool_4 = self.hits_max_pool_4.to(pred.device)
        self.misses_max_pool_4 = self.misses_max_pool_4.to(pred.device)
        self.fas_max_pool_4 = self.fas_max_pool_4.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='max')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_max_pool_4[i] += hits
            self.misses_max_pool_4[i] += misses
            self.fas_max_pool_4[i] += fas 
        ## max pool 16 ##
        self.hits_max_pool_16 = self.hits_max_pool_16.to(pred.device)
        self.misses_max_pool_16 = self.misses_max_pool_16.to(pred.device)
        self.fas_max_pool_16 = self.fas_max_pool_16.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='max')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_max_pool_16[i] += hits
            self.misses_max_pool_16[i] += misses
            self.fas_max_pool_16[i] += fas 
        ## avg pool 4 ##
        self.hits_avg_pool_4 = self.hits_avg_pool_4.to(pred.device)
        self.misses_avg_pool_4 = self.misses_avg_pool_4.to(pred.device)
        self.fas_avg_pool_4 = self.fas_avg_pool_4.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='avg')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_avg_pool_4[i] += hits
            self.misses_avg_pool_4[i] += misses
            self.fas_avg_pool_4[i] += fas 
        ## avg pool 16 ##
        self.hits_avg_pool_16 = self.hits_avg_pool_16.to(pred.device)
        self.misses_avg_pool_16 = self.misses_avg_pool_16.to(pred.device)
        self.fas_avg_pool_16 = self.fas_avg_pool_16.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='avg')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_avg_pool_16[i] += hits
            self.misses_avg_pool_16[i] += misses
            self.fas_avg_pool_16[i] += fas 

    @torch.no_grad()
    def update_sample(self, pred, target, sample_names):
        """
        更新当前统计数据，每个样本独立存储计算结果。
        
        :param pred: 预测值 (B, ...)
        :param target: 真实值 (B, ...)
        :param sample_names: 样本名称列表 (长度 B)
        """
        # 确保 pred 和 target 在相同的设备上
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32, device=self.hits.device)

        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32, device=self.hits.device)
        
        _pred, _target = self.preprocess(pred, target)
        _pred = _pred.unsqueeze(1)
        _target = _target.unsqueeze(1)
        
        results = {}  
        
        for sample_idx, sample_name in enumerate(sample_names):
            results[sample_name] = {
                "hits": [],
                "misses": [],
                "fas": [],
                "cor": [],
                "fss_pool1": [],
                "fss_pool4": [],
                "fss_pool8": [],
                "fss_pool16": []
            }

            results[sample_name]["mae"] = torch.abs(_pred[sample_idx] - _target[sample_idx]).mean().item()
            results[sample_name]["mse"] = ((_pred[sample_idx] - _target[sample_idx]) ** 2).mean().item()
        
            for i, threshold in enumerate(self.threshold_list):
                hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred[sample_idx], _target[sample_idx], threshold)
                results[sample_name]["hits"].append(hits)
                results[sample_name]["misses"].append(misses)
                results[sample_name]["fas"].append(fas)
                results[sample_name]["cor"].append(cor)

                results[sample_name]["fss_pool1"].append(self.compute_fss(_pred[sample_idx], _target[sample_idx], threshold, 1))
                results[sample_name]["fss_pool4"].append(self.compute_fss(_pred[sample_idx], _target[sample_idx], threshold, 4))
                results[sample_name]["fss_pool8"].append(self.compute_fss(_pred[sample_idx], _target[sample_idx], threshold, 8))
                results[sample_name]["fss_pool16"].append(self.compute_fss(_pred[sample_idx], _target[sample_idx], threshold, 16))

        ## 处理 max pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='max')
        _pred = _pred.unsqueeze(1)
        _target = _target.unsqueeze(1)
        for sample_idx, sample_name in enumerate(sample_names):
            results[sample_name]["hits_max_pool_4"] = []
            results[sample_name]["misses_max_pool_4"] = []
            results[sample_name]["fas_max_pool_4"] = []
            
            for i, threshold in enumerate(self.threshold_list):
                hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred[sample_idx], _target[sample_idx], threshold)
                results[sample_name]["hits_max_pool_4"].append(hits)
                results[sample_name]["misses_max_pool_4"].append(misses)
                results[sample_name]["fas_max_pool_4"].append(fas)

        ## 处理 max pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='max')
        _pred = _pred.unsqueeze(1)
        _target = _target.unsqueeze(1)
        for sample_idx, sample_name in enumerate(sample_names):
            results[sample_name]["hits_max_pool_16"] = []
            results[sample_name]["misses_max_pool_16"] = []
            results[sample_name]["fas_max_pool_16"] = []
            
            for i, threshold in enumerate(self.threshold_list):
                hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred[sample_idx], _target[sample_idx], threshold)
                results[sample_name]["hits_max_pool_16"].append(hits)
                results[sample_name]["misses_max_pool_16"].append(misses)
                results[sample_name]["fas_max_pool_16"].append(fas)

        ## 处理 avg pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='avg')
        _pred = _pred.unsqueeze(1)
        _target = _target.unsqueeze(1)
        for sample_idx, sample_name in enumerate(sample_names):
            results[sample_name]["hits_avg_pool_4"] = []
            results[sample_name]["misses_avg_pool_4"] = []
            results[sample_name]["fas_avg_pool_4"] = []
            
            for i, threshold in enumerate(self.threshold_list):
                hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred[sample_idx], _target[sample_idx], threshold)
                results[sample_name]["hits_avg_pool_4"].append(hits)
                results[sample_name]["misses_avg_pool_4"].append(misses)
                results[sample_name]["fas_avg_pool_4"].append(fas)

        ## 处理 avg pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='avg')
        _pred = _pred.unsqueeze(1)
        _target = _target.unsqueeze(1)
        for sample_idx, sample_name in enumerate(sample_names):
            results[sample_name]["hits_avg_pool_16"] = []
            results[sample_name]["misses_avg_pool_16"] = []
            results[sample_name]["fas_avg_pool_16"] = []
            
            for i, threshold in enumerate(self.threshold_list):
                hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred[sample_idx], _target[sample_idx], threshold)
                results[sample_name]["hits_avg_pool_16"].append(hits)
                results[sample_name]["misses_avg_pool_16"].append(misses)
                results[sample_name]["fas_avg_pool_16"].append(fas)
                
        return results
    
    def _get_hits_misses_fas(self, metric_name):
        if metric_name.endswith('-4-avg'):
            hits = self.hits_avg_pool_4
            misses = self.misses_avg_pool_4
            fas = self.fas_avg_pool_4
        elif metric_name.endswith('-16-avg'):
            hits = self.hits_avg_pool_16
            misses = self.misses_avg_pool_16
            fas = self.fas_avg_pool_16
        elif metric_name.endswith('-4-max'):
            hits = self.hits_max_pool_4
            misses = self.misses_max_pool_4
            fas = self.fas_max_pool_4
        elif metric_name.endswith('-16-max'):
            hits = self.hits_max_pool_16
            misses = self.misses_max_pool_16
            fas = self.fas_max_pool_16
        else:
            hits = self.hits
            misses = self.misses
            fas = self.fas
        return [hits, misses, fas]

    def _get_correct_negtives(self):
        return self.cor
    
    @torch.no_grad()
    def compute(self):
        if self.dist_eval:
            self.synchronize_between_processes()
        
        metrics_dict = {'pod': self.pod,
                        'csi': self.csi,
                        'csi-4-avg': self.csi, 
                        'csi-16-avg': self.csi,
                        'csi-4-max': self.csi, 
                        'csi-16-max': self.csi,
                        'sucr': self.sucr,
                        'bias': self.bias,
                        'hss': self.hss,
                        'far': self.far}
        ret = {}
        for threshold in self.dbz_thresholds:
            ret[threshold] = {}
        ret["avg"] = {}

        for metrics in self.metrics_list:
            if self.keep_seq_len_dim:
                score_avg = np.zeros((self.seq_len, ))
            else:
                score_avg = 0
            hits, misses, fas = self._get_hits_misses_fas(metrics)
            # scores = metrics_dict[metrics](self.hits, self.misses, self.fas, self.eps)
            if metrics != 'hss':
                scores = metrics_dict[metrics](hits, misses, fas, self.eps)
            else:
                cor = self._get_correct_negtives()
                scores = metrics_dict[metrics](hits, misses, fas, cor, self.eps)
            scores = scores.detach().cpu().numpy()
            for i, threshold in enumerate(self.dbz_thresholds):
                if self.keep_seq_len_dim:
                    score = scores[i]  # shape = (seq_len, )
                else:
                    score = scores[i].item()  # shape = (1, )
                if self.mode in ("0", "1"):
                    ret[threshold][metrics] = score
                elif self.mode in ("2", ):
                    ret[threshold][metrics] = np.mean(score).item()
                else:
                    raise NotImplementedError
                score_avg += score
            score_avg /= len(self.dbz_thresholds)
            if self.mode in ("0", "1"):
                ret["avg"][metrics] = score_avg
            elif self.mode in ("2",):
                ret["avg"][metrics] = np.mean(score_avg).item()
            else:
                raise NotImplementedError
        return ret

    @torch.no_grad()
    def get_single_frame_metrics(self, target, pred, metrics=['ssim', 'psnr', ]): #'cspr', 'cspr-4-avg', 'cspr-16-avg', 'cspr-4-max', 'cspr-16-max'
        metric_funcs = {
            'ssim': cal_SSIM,
            'psnr': cal_PSNR
        }
        metrics_dict = {}
        for metric in metrics:
            metric_fun = metric_funcs[metric]
            metrics_dict[metric] = metric_fun(gt=target*255., pred=pred*255., is_img=False)
        return metrics_dict
    
    @torch.no_grad()
    def get_crps(self, target, pred):
        """
        pred: (b, t, c, h, w)/(b, n, t, c, h, w)
        target: (b, t, c, h, w)
        """
        if len(pred.shape) == 5:
            pred = pred.unsqueeze(1)
        crps = cal_CRPS(gt=target, pred=pred, type='none')
        crps_avg_4 = cal_CRPS(gt=target, pred=pred, type='avg', scale=4)
        crps_avg_16 = cal_CRPS(gt=target, pred=pred, type='avg', scale=16)
        crps_max_4 = cal_CRPS(gt=target, pred=pred, type='max', scale=4)
        crps_max_16 = cal_CRPS(gt=target, pred=pred, type='max', scale=16)
        crps_dict = {
            'crps': crps,
            'crps_avg_4': crps_avg_4,
            'crps_avg_16': crps_avg_16,
            'crps_max_4': crps_max_4,
            'crps_max_16': crps_max_16
        }
        return crps_dict

    def reset(self):
        self.hits = self.hits*0
        self.misses = self.misses*0
        self.fas = self.fas*0

        self.hits_avg_pool_4 *= 0
        self.hits_avg_pool_16 *= 0
        self.hits_max_pool_4 *= 0
        self.hits_max_pool_16 *= 0

        self.misses_avg_pool_4 *= 0
        self.misses_avg_pool_16 *= 0
        self.misses_max_pool_4 *= 0
        self.misses_max_pool_16 *= 0
 
        self.fas_avg_pool_4 *= 0
        self.fas_avg_pool_16 *= 0
        self.fas_max_pool_4  *= 0
        self.fas_max_pool_16  *= 0

    def compute_sample(self, results):
        if self.dist_eval:
            self.synchronize_between_processes()

        metrics_dict = {
            'pod': self.pod,
            'csi': self.csi,
            'sucr': self.sucr,
            'bias': self.bias,
            'hss': self.hss,
            'far': self.far
        }
        
        ret = {}
        
        # Initialize structure to hold per-sample results
        for sample_name in results.keys():
            ret[sample_name] = {}
            for threshold in self.threshold_list:
                ret[sample_name][threshold] = {}
            ret[sample_name]["avg"] = {}
        # Process each sample separately
        for sample_name, sample_results in results.items():
            # Compute metrics for each threshold for this sample
            ret[sample_name]["mae"] = sample_results['mae']
            ret[sample_name]["mse"] = sample_results['mse']
            
            for index, threshold in enumerate(self.threshold_list):
                # Get basic contingency table values
                hits = sample_results["hits"][index]
                misses = sample_results["misses"][index]
                fas = sample_results["fas"][index]
                cor = sample_results["cor"][index]
                
                # Get pooled values
                hits_max_pool_4 = sample_results["hits_max_pool_4"][index]
                misses_max_pool_4 = sample_results["misses_max_pool_4"][index]
                fas_max_pool_4 = sample_results["fas_max_pool_4"][index]
                
                hits_max_pool_16 = sample_results["hits_max_pool_16"][index]
                misses_max_pool_16 = sample_results["misses_max_pool_16"][index]
                fas_max_pool_16 = sample_results["fas_max_pool_16"][index]
                
                hits_avg_pool_4 = sample_results["hits_avg_pool_4"][index]
                misses_avg_pool_4 = sample_results["misses_avg_pool_4"][index]
                fas_avg_pool_4 = sample_results["fas_avg_pool_4"][index]
                
                hits_avg_pool_16 = sample_results["hits_avg_pool_16"][index]
                misses_avg_pool_16 = sample_results["misses_avg_pool_16"][index]
                fas_avg_pool_16 = sample_results["fas_avg_pool_16"][index]

                # Store FSS scores directly
                ret[sample_name][threshold]['fss_pool1'] = sample_results["fss_pool1"][index].item()
                ret[sample_name][threshold]['fss_pool4'] = sample_results["fss_pool4"][index].item()
                ret[sample_name][threshold]['fss_pool8'] = sample_results["fss_pool8"][index].item()
                ret[sample_name][threshold]['fss_pool16'] = sample_results["fss_pool16"][index].item()

                # Compute basic metrics
                for metric in self.metrics_list:
                    if metric == 'csi-4-avg':
                        score = self.csi(hits_avg_pool_4, misses_avg_pool_4, fas_avg_pool_4, self.eps)
                    elif metric == 'csi-16-avg':
                        score = self.csi(hits_avg_pool_16, misses_avg_pool_16, fas_avg_pool_16, self.eps)
                    elif metric == 'csi-4-max':
                        score = self.csi(hits_max_pool_4, misses_max_pool_4, fas_max_pool_4, self.eps)
                    elif metric == 'csi-16-max':
                        score = self.csi(hits_max_pool_16, misses_max_pool_16, fas_max_pool_16, self.eps)
                    elif metric != 'hss':
                        score = metrics_dict[metric](hits, misses, fas, self.eps)
                    else:
                        score = metrics_dict[metric](hits, misses, fas, cor, self.eps)
                    
                    ret[sample_name][threshold][metric] = score.item()

            # Compute average metrics for this sample across thresholds
            for metric in self.metrics_list:
                threshold_scores = [ret[sample_name][t][metric] for t in self.threshold_list]
                ret[sample_name]["avg"][metric] = np.mean(threshold_scores)

        return ret

    def compute_fss(self, pred, target, threshold, patch_size):
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            ### patch pooling ###
            t, p = _threshold(target, pred, threshold)
            patch_t = rearrange(t, 'b t c (h p1) (w p2) -> b t c h p1 w p2', p1=patch_size, p2=patch_size)
            patch_t = patch_t.sum(dim=(-1, -3))
            patch_p = rearrange(p, 'b t c (h p1) (w p2) -> b t c h p1 w p2', p1=patch_size, p2=patch_size)
            patch_p = patch_p.sum(dim=(-1, -3))
            ### compute fss ###
            fss = 2 * torch.sum(patch_t * patch_p, dim=(-1, -2)) / (self.eps + torch.sum(patch_t*patch_t, dim=(-1, -2)) + torch.sum(patch_p * patch_p, dim=(-1, -2)))
            avg_fss = fss.mean()
        return avg_fss

    @torch.no_grad()
    def update_frame(self, pred, target, sample_names):
        """
        更新当前统计数据，每个样本独立存储计算结果。
        
        :param pred: 预测值 (B, ...)
        :param target: 真实值 (B, ...)
        :param sample_names: 样本名称列表 (长度 B)
        """
        # 确保 pred 和 target 在相同的设备上
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32, device=self.hits.device) # 1,128,128

        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32, device=self.hits.device) # 1,128,128

        pred = pred.unsqueeze(1).unsqueeze(1)
        target = target.unsqueeze(1).unsqueeze(1)
        
        _pred, _target = self.preprocess(pred, target)
        
        results = {}  
        

        results = {
            "hits": [],
            "misses": [],
            "fas": [],
            "cor": [],
            "fss_pool1": [],
            "fss_pool4": [],
            "fss_pool8": [],
            "fss_pool16": []
        }

        results["mae"] = torch.abs(_pred - _target).mean().item()
        results["mse"] = ((_pred - _target) ** 2).mean().item()
    
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits"].append(hits)
            results["misses"].append(misses)
            results["fas"].append(fas)
            results["cor"].append(cor)

            results["fss_pool1"].append(self.compute_fss(_pred, _target, threshold, 1))
            results["fss_pool4"].append(self.compute_fss(_pred, _target, threshold, 4))
            results["fss_pool8"].append(self.compute_fss(_pred, _target, threshold, 8))
            results["fss_pool16"].append(self.compute_fss(_pred, _target, threshold, 16))

        ## 处理 max pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='max')
        
        results["hits_max_pool_4"] = []
        results["misses_max_pool_4"] = []
        results["fas_max_pool_4"] = []
        
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_max_pool_4"].append(hits)
            results["misses_max_pool_4"].append(misses)
            results["fas_max_pool_4"].append(fas)

        ## 处理 max pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='max')
        
        results["hits_max_pool_16"] = []
        results["misses_max_pool_16"] = []
        results["fas_max_pool_16"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_max_pool_16"].append(hits)
            results["misses_max_pool_16"].append(misses)
            results["fas_max_pool_16"].append(fas)

        ## 处理 avg pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='avg')
    
        results["hits_avg_pool_4"] = []
        results["misses_avg_pool_4"] = []
        results["fas_avg_pool_4"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_avg_pool_4"].append(hits)
            results["misses_avg_pool_4"].append(misses)
            results["fas_avg_pool_4"].append(fas)

        ## 处理 avg pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='avg')

        results["hits_avg_pool_16"] = []
        results["misses_avg_pool_16"] = []
        results["fas_avg_pool_16"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_avg_pool_16"].append(hits)
            results["misses_avg_pool_16"].append(misses)
            results["fas_avg_pool_16"].append(fas)
                
        return results

    def compute_frame(self, results):
        if self.dist_eval:
            self.synchronize_between_processes()

        metrics_dict = {
            'pod': self.pod,
            'csi': self.csi,
            'sucr': self.sucr,
            'bias': self.bias,
            'hss': self.hss,
            'far': self.far
        }
        
        ret = {}
        
        for threshold in self.threshold_list:
            ret[threshold] = {}
        ret["avg"] = {}

        # Compute metrics for each threshold for this sample
        ret["mae"] = results['mae']
        ret["mse"] = results['mse']
            
        for index, threshold in enumerate(self.threshold_list):
            # Get basic contingency table values
            hits = results["hits"][index]
            misses = results["misses"][index]
            fas = results["fas"][index]
            cor = results["cor"][index]
            
            # Get pooled values
            hits_max_pool_4 = results["hits_max_pool_4"][index]
            misses_max_pool_4 = results["misses_max_pool_4"][index]
            fas_max_pool_4 = results["fas_max_pool_4"][index]
            
            hits_max_pool_16 = results["hits_max_pool_16"][index]
            misses_max_pool_16 = results["misses_max_pool_16"][index]
            fas_max_pool_16 = results["fas_max_pool_16"][index]
            
            hits_avg_pool_4 = results["hits_avg_pool_4"][index]
            misses_avg_pool_4 = results["misses_avg_pool_4"][index]
            fas_avg_pool_4 = results["fas_avg_pool_4"][index]
            
            hits_avg_pool_16 = results["hits_avg_pool_16"][index]
            misses_avg_pool_16 = results["misses_avg_pool_16"][index]
            fas_avg_pool_16 = results["fas_avg_pool_16"][index]

            # Store FSS scores directly
            ret[threshold]['fss_pool1'] = results["fss_pool1"][index].item()
            ret[threshold]['fss_pool4'] = results["fss_pool4"][index].item()
            ret[threshold]['fss_pool8'] = results["fss_pool8"][index].item()
            ret[threshold]['fss_pool16'] = results["fss_pool16"][index].item()

            # Compute basic metrics
            for metric in self.metrics_list:
                if metric == 'csi-4-avg':
                    score = self.csi(hits_avg_pool_4, misses_avg_pool_4, fas_avg_pool_4, self.eps)
                elif metric == 'csi-16-avg':
                    score = self.csi(hits_avg_pool_16, misses_avg_pool_16, fas_avg_pool_16, self.eps)
                elif metric == 'csi-4-max':
                    score = self.csi(hits_max_pool_4, misses_max_pool_4, fas_max_pool_4, self.eps)
                elif metric == 'csi-16-max':
                    score = self.csi(hits_max_pool_16, misses_max_pool_16, fas_max_pool_16, self.eps)
                elif metric != 'hss':
                    score = metrics_dict[metric](hits, misses, fas, self.eps)
                else:
                    score = metrics_dict[metric](hits, misses, fas, cor, self.eps)
                
                ret[threshold][metric] = score.item()

        # Compute average metrics for this sample across thresholds
        for metric in self.metrics_list:
            threshold_scores = [ret[t][metric] for t in self.threshold_list]
            ret["avg"][metric] = np.mean(threshold_scores)

        return ret

    def update_single_sample(self, pred, target, sample_names):
        """
        更新当前统计数据，每个样本独立存储计算结果。
        
        :param pred: 预测值 (B, ...)
        :param target: 真实值 (B, ...)
        :param sample_names: 样本名称列表 (长度 B)
        """
        # 确保 pred 和 target 在相同的设备上
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32, device=self.hits.device) # 1,128,128

        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32, device=self.hits.device) # 1,128,128
        
        _pred, _target = self.preprocess(pred, target)
        
        results = {}  
        

        results = {
            "hits": [],
            "misses": [],
            "fas": [],
            "cor": [],
            "fss_pool1": [],
            "fss_pool4": [],
            "fss_pool8": [],
            "fss_pool16": []
        }

        results["mae"] = torch.abs(_pred - _target).mean().item()
        results["mse"] = ((_pred - _target) ** 2).mean().item()
    
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits"].append(hits)
            results["misses"].append(misses)
            results["fas"].append(fas)
            results["cor"].append(cor)

            results["fss_pool1"].append(self.compute_fss(_pred, _target, threshold, 1))
            results["fss_pool4"].append(self.compute_fss(_pred, _target, threshold, 4))
            results["fss_pool8"].append(self.compute_fss(_pred, _target, threshold, 8))
            results["fss_pool16"].append(self.compute_fss(_pred, _target, threshold, 16))

        ## 处理 max pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='max')
        
        results["hits_max_pool_4"] = []
        results["misses_max_pool_4"] = []
        results["fas_max_pool_4"] = []
        
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_max_pool_4"].append(hits)
            results["misses_max_pool_4"].append(misses)
            results["fas_max_pool_4"].append(fas)

        ## 处理 max pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='max')
        
        results["hits_max_pool_16"] = []
        results["misses_max_pool_16"] = []
        results["fas_max_pool_16"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_max_pool_16"].append(hits)
            results["misses_max_pool_16"].append(misses)
            results["fas_max_pool_16"].append(fas)

        ## 处理 avg pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='avg')
    
        results["hits_avg_pool_4"] = []
        results["misses_avg_pool_4"] = []
        results["fas_avg_pool_4"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_avg_pool_4"].append(hits)
            results["misses_avg_pool_4"].append(misses)
            results["fas_avg_pool_4"].append(fas)

        ## 处理 avg pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='avg')

        results["hits_avg_pool_16"] = []
        results["misses_avg_pool_16"] = []
        results["fas_avg_pool_16"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_avg_pool_16"].append(hits)
            results["misses_avg_pool_16"].append(misses)
            results["fas_avg_pool_16"].append(fas)
                
        return results
    
class HKOSkillScore(object):
    def __init__(self, seq_len,
                  no_ssim=True, threholds=None, mode='0',
                  layout = 'NTCHW',
                  preprocess_type = 'hko7', dist_eval=False,
                  metrics_list=['csi', 'csi-4-avg', 'csi-16-avg',
                               'csi-4-max', 'csi-16-max', 'bias',
                                'sucr', 'pod', 'hss', 'far'],
                  eps=1e-4):
        self.metrics_list = metrics_list
        self.eps=eps
        self.layout = layout
        self.preprocess_type = preprocess_type
        self.dist_eval = dist_eval

        self._thresholds = np.array([0.5, 2, 5, 10, 30]) if threholds is None else threholds
        self.g_thresholds = [self.rainfall_to_pixel(threshold) for threshold in self._thresholds]
        self.threshold_list = self.g_thresholds

        self._seq_len = seq_len
        self._no_ssim = no_ssim
        self._exclude_mask = torch.tensor(1-self.get_exclude_mask())
        # self._total_batch_num = 0
        # self.begin()

        self.mode = mode
        
        if mode in ("0", ):
            self.keep_seq_len_dim = False
            state_shape = (len(self.threshold_list), )
        elif mode in ("1", "2"):
            self.keep_seq_len_dim = True
            assert isinstance(self.seq_len, int), "seq_len must be provided when we need to keep seq_len dim."
            state_shape = (len(self.threshold_list), self.seq_len)
        else:
            raise NotImplementedError(f"mode {mode} not supported!")
        
        self.hits = torch.zeros(state_shape)
        self.misses = torch.zeros(state_shape)
        self.fas = torch.zeros(state_shape)
        self.cor = torch.zeros(state_shape)

        ## pooling csi ##
        self.hits_avg_pool_4 = torch.zeros(state_shape)
        self.misses_avg_pool_4 = torch.zeros(state_shape)
        self.fas_avg_pool_4 = torch.zeros(state_shape)

        self.hits_max_pool_4 = torch.zeros(state_shape)
        self.misses_max_pool_4 = torch.zeros(state_shape)
        self.fas_max_pool_4 = torch.zeros(state_shape)

        self.hits_avg_pool_16 = torch.zeros(state_shape)
        self.misses_avg_pool_16 = torch.zeros(state_shape)
        self.fas_avg_pool_16 = torch.zeros(state_shape)

        self.hits_max_pool_16 = torch.zeros(state_shape)
        self.misses_max_pool_16 = torch.zeros(state_shape)
        self.fas_max_pool_16 = torch.zeros(state_shape)


    def pod(self, hits, misses, fas, eps):
        return hits / (hits + misses + eps)
    
    def far(self, hits, misses, fas, eps):
        return fas / (hits + fas + eps)

    def csi(self, hits, misses, fas, eps):
        return hits / (hits + misses + fas + eps)
    
    def sucr(self, hits, misses, fas, eps):
        return hits / (hits + fas + eps)

    def bias(self, hits, misses, fas, eps):
        bias = (hits + fas) / (hits + misses + eps)
        logbias = torch.pow(bias / torch.log(torch.tensor(2.0)), 2.0)
        return logbias
    
    def hss(self, hits, misses, fas, cor, eps):
        hss = 2 * (hits * cor - misses * fas) / ((hits + misses) * (misses + cor) + (hits + fas) * (fas + cor) + eps)
        return hss
    
    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        dist.barrier()
        dist.all_reduce(self.hits)
        dist.all_reduce(self.misses)
        dist.all_reduce(self.fas)
        ### avg 4 ###
        dist.all_reduce(self.hits_avg_pool_4)
        dist.all_reduce(self.misses_avg_pool_4)
        dist.all_reduce(self.fas_avg_pool_4)
        ### max 4 ###
        dist.all_reduce(self.hits_max_pool_4)
        dist.all_reduce(self.misses_max_pool_4)
        dist.all_reduce(self.fas_max_pool_4)
        ### avg 16 ###
        dist.all_reduce(self.hits_avg_pool_16)
        dist.all_reduce(self.misses_avg_pool_16)
        dist.all_reduce(self.fas_avg_pool_16)
        ### max 16 ###
        dist.all_reduce(self.hits_max_pool_16)
        dist.all_reduce(self.misses_max_pool_16)
        dist.all_reduce(self.fas_max_pool_16)
    
    @property
    def hits_misses_fas_reduce_dims(self):
        if not hasattr(self, "_hits_misses_fas_reduce_dims"):
            seq_dim = self.layout.find('T')
            self._hits_misses_fas_reduce_dims = list(range(len(self.layout)))
            if self.keep_seq_len_dim:
                self._hits_misses_fas_reduce_dims.pop(seq_dim)
        return self._hits_misses_fas_reduce_dims
    
    def preprocess(self, pred, target):
        if self.preprocess_type == "hko7":
            pred = pred.detach() / (1. / 255.) 
            target = target.detach() / (1. / 255.)
        else:
            raise NotImplementedError
        return pred, target
    
    def preprocess_pool(self, pred, target, pool_size=4, type='avg'):
        pred = pred.detach() / (1. / 255.)
        target = target.detach() / (1. / 255.)
        b, t, _, _, _ = pred.shape
        pred = rearrange(pred, 'b t c h w -> (b t) c h w')
        target = rearrange(target, 'b t c h w -> (b t) c h w')
        if type == 'avg':
            pred = F.avg_pool2d(pred, kernel_size=pool_size, stride=pool_size)
            target = F.avg_pool2d(target, kernel_size=pool_size, stride=pool_size)
        elif type == 'max':
            pred = F.max_pool2d(pred, kernel_size=pool_size, stride=pool_size)
            target = F.max_pool2d(target, kernel_size=pool_size, stride=pool_size)
        pred = rearrange(pred, '(b t) c h w -> b t c h w', b=b)
        target = rearrange(target, '(b t) c h w -> b t c h w', b=b)
        return pred, target
    
    def calc_seq_hits_misses_fas(self, pred, target, threshold):
        if self._exclude_mask.device != pred.device:
            self._exclude_mask = self._exclude_mask.to(pred.device)
        mask = self._exclude_mask ## (h, w)
        with torch.no_grad():
            t, p = _threshold(target, pred, threshold)
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(pred.shape[-2], pred.shape[-1]), mode='nearest').squeeze(0).squeeze(0)
            hits = torch.sum(mask*t * p, dim=self.hits_misses_fas_reduce_dims).int()
            misses = torch.sum(mask*t * (1 - p), dim=self.hits_misses_fas_reduce_dims).int()
            fas = torch.sum(mask*(1 - t) * p, dim=self.hits_misses_fas_reduce_dims).int()
            cor = torch.sum((1 - t) * (1 - p), dim=self.hits_misses_fas_reduce_dims).int()
        return hits, misses, fas, cor
    
        
    def get_exclude_mask(self):
        import numpy as np
        import cv2
        with np.load('/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/datasets/hko7/hko7_12min_list/mask_dat.npz') as dat:
            exclude_mask = dat['exclude_mask'][:]
        # 最近邻插值到 128x128
        exclude_mask = cv2.resize(
            exclude_mask.astype(np.uint8),  
            (128, 128),
            interpolation=cv2.INTER_NEAREST
        )
        return exclude_mask
        
    def rainfall_to_pixel(self, rainfall_intensity, a=58.53, b=1.56):
        dBR = np.log10(rainfall_intensity) * 10.0
        # dBZ = 10b log(R) +10log(a)
        dBZ = dBR * b + 10.0 * np.log10(a)
        pixel_vals = (dBZ + 10.0) / 70.0
        return pixel_vals * 255.0
        
    @torch.no_grad()
    def update(self, pred, target):
        ## pool 1 ##
        self.hits = self.hits.to(pred.device)
        self.misses = self.misses.to(pred.device)
        self.fas = self.fas.to(pred.device)
        self.cor = self.cor.to(pred.device)
        _pred, _target = self.preprocess(pred, target)
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits[i] += hits
            self.misses[i] += misses
            self.fas[i] += fas
            self.cor[i] += cor
        ## max pool 4 ##
        self.hits_max_pool_4 = self.hits_max_pool_4.to(pred.device)
        self.misses_max_pool_4 = self.misses_max_pool_4.to(pred.device)
        self.fas_max_pool_4 = self.fas_max_pool_4.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='max')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_max_pool_4[i] += hits
            self.misses_max_pool_4[i] += misses
            self.fas_max_pool_4[i] += fas 
        ## max pool 16 ##
        self.hits_max_pool_16 = self.hits_max_pool_16.to(pred.device)
        self.misses_max_pool_16 = self.misses_max_pool_16.to(pred.device)
        self.fas_max_pool_16 = self.fas_max_pool_16.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='max')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_max_pool_16[i] += hits
            self.misses_max_pool_16[i] += misses
            self.fas_max_pool_16[i] += fas 
        ## avg pool 4 ##
        self.hits_avg_pool_4 = self.hits_avg_pool_4.to(pred.device)
        self.misses_avg_pool_4 = self.misses_avg_pool_4.to(pred.device)
        self.fas_avg_pool_4 = self.fas_avg_pool_4.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='avg')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_avg_pool_4[i] += hits
            self.misses_avg_pool_4[i] += misses
            self.fas_avg_pool_4[i] += fas 
        ## avg pool 16 ##
        self.hits_avg_pool_16 = self.hits_avg_pool_16.to(pred.device)
        self.misses_avg_pool_16 = self.misses_avg_pool_16.to(pred.device)
        self.fas_avg_pool_16 = self.fas_avg_pool_16.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='avg')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_avg_pool_16[i] += hits
            self.misses_avg_pool_16[i] += misses
            self.fas_avg_pool_16[i] += fas 

    @torch.no_grad()
    def update_sample(self, pred, target, sample_names):
        """
        更新当前统计数据，每个样本独立存储计算结果。
        
        :param pred: 预测值 (B, ...)
        :param target: 真实值 (B, ...)
        :param sample_names: 样本名称列表 (长度 B)
        """
        # 确保 pred 和 target 在相同的设备上
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32, device=self.hits.device)

        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32, device=self.hits.device)
        
        _pred, _target = self.preprocess(pred, target)
        _pred = _pred.unsqueeze(1)
        _target = _target.unsqueeze(1)
        
        results = {}  
        
        for sample_idx, sample_name in enumerate(sample_names):
            results[sample_name] = {
                "hits": [],
                "misses": [],
                "fas": [],
                "cor": [],
                "fss_pool1": [],
                "fss_pool4": [],
                "fss_pool8": [],
                "fss_pool16": []
            }

            results[sample_name]["mae"] = torch.abs(_pred[sample_idx] - _target[sample_idx]).mean().item()
            results[sample_name]["mse"] = ((_pred[sample_idx] - _target[sample_idx]) ** 2).mean().item()
        
            for i, threshold in enumerate(self.threshold_list):
                hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred[sample_idx], _target[sample_idx], threshold)
                results[sample_name]["hits"].append(hits)
                results[sample_name]["misses"].append(misses)
                results[sample_name]["fas"].append(fas)
                results[sample_name]["cor"].append(cor)

                results[sample_name]["fss_pool1"].append(self.compute_fss(_pred[sample_idx], _target[sample_idx], threshold, 1))
                results[sample_name]["fss_pool4"].append(self.compute_fss(_pred[sample_idx], _target[sample_idx], threshold, 4))
                results[sample_name]["fss_pool8"].append(self.compute_fss(_pred[sample_idx], _target[sample_idx], threshold, 8))
                results[sample_name]["fss_pool16"].append(self.compute_fss(_pred[sample_idx], _target[sample_idx], threshold, 16))

        ## 处理 max pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='max')
        _pred = _pred.unsqueeze(1)
        _target = _target.unsqueeze(1)
        for sample_idx, sample_name in enumerate(sample_names):
            results[sample_name]["hits_max_pool_4"] = []
            results[sample_name]["misses_max_pool_4"] = []
            results[sample_name]["fas_max_pool_4"] = []
            
            for i, threshold in enumerate(self.threshold_list):
                hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred[sample_idx], _target[sample_idx], threshold)
                results[sample_name]["hits_max_pool_4"].append(hits)
                results[sample_name]["misses_max_pool_4"].append(misses)
                results[sample_name]["fas_max_pool_4"].append(fas)

        ## 处理 max pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='max')
        _pred = _pred.unsqueeze(1)
        _target = _target.unsqueeze(1)
        for sample_idx, sample_name in enumerate(sample_names):
            results[sample_name]["hits_max_pool_16"] = []
            results[sample_name]["misses_max_pool_16"] = []
            results[sample_name]["fas_max_pool_16"] = []
            
            for i, threshold in enumerate(self.threshold_list):
                hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred[sample_idx], _target[sample_idx], threshold)
                results[sample_name]["hits_max_pool_16"].append(hits)
                results[sample_name]["misses_max_pool_16"].append(misses)
                results[sample_name]["fas_max_pool_16"].append(fas)

        ## 处理 avg pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='avg')
        _pred = _pred.unsqueeze(1)
        _target = _target.unsqueeze(1)
        for sample_idx, sample_name in enumerate(sample_names):
            results[sample_name]["hits_avg_pool_4"] = []
            results[sample_name]["misses_avg_pool_4"] = []
            results[sample_name]["fas_avg_pool_4"] = []
            
            for i, threshold in enumerate(self.threshold_list):
                hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred[sample_idx], _target[sample_idx], threshold)
                results[sample_name]["hits_avg_pool_4"].append(hits)
                results[sample_name]["misses_avg_pool_4"].append(misses)
                results[sample_name]["fas_avg_pool_4"].append(fas)

        ## 处理 avg pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='avg')
        _pred = _pred.unsqueeze(1)
        _target = _target.unsqueeze(1)
        for sample_idx, sample_name in enumerate(sample_names):
            results[sample_name]["hits_avg_pool_16"] = []
            results[sample_name]["misses_avg_pool_16"] = []
            results[sample_name]["fas_avg_pool_16"] = []
            
            for i, threshold in enumerate(self.threshold_list):
                hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred[sample_idx], _target[sample_idx], threshold)
                results[sample_name]["hits_avg_pool_16"].append(hits)
                results[sample_name]["misses_avg_pool_16"].append(misses)
                results[sample_name]["fas_avg_pool_16"].append(fas)
                
        return results
    
    def _get_hits_misses_fas(self, metric_name):
        if metric_name.endswith('-4-avg'):
            hits = self.hits_avg_pool_4
            misses = self.misses_avg_pool_4
            fas = self.fas_avg_pool_4
        elif metric_name.endswith('-16-avg'):
            hits = self.hits_avg_pool_16
            misses = self.misses_avg_pool_16
            fas = self.fas_avg_pool_16
        elif metric_name.endswith('-4-max'):
            hits = self.hits_max_pool_4
            misses = self.misses_max_pool_4
            fas = self.fas_max_pool_4
        elif metric_name.endswith('-16-max'):
            hits = self.hits_max_pool_16
            misses = self.misses_max_pool_16
            fas = self.fas_max_pool_16
        else:
            hits = self.hits
            misses = self.misses
            fas = self.fas
        return [hits, misses, fas]

    def _get_correct_negtives(self):
        return self.cor
    
    @torch.no_grad()
    def compute(self):
        if self.dist_eval:
            self.synchronize_between_processes()
        
        metrics_dict = {'pod': self.pod,
                        'csi': self.csi,
                        'csi-4-avg': self.csi, 
                        'csi-16-avg': self.csi,
                        'csi-4-max': self.csi, 
                        'csi-16-max': self.csi,
                        'sucr': self.sucr,
                        'bias': self.bias,
                        'hss': self.hss,
                        'far': self.far}
        ret = {}
        for threshold in self._thresholds:
            ret[threshold] = {}
        ret["avg"] = {}

        for metrics in self.metrics_list:
            if self.keep_seq_len_dim:
                score_avg = np.zeros((self.seq_len, ))
            else:
                score_avg = 0
            hits, misses, fas = self._get_hits_misses_fas(metrics)
            # scores = metrics_dict[metrics](self.hits, self.misses, self.fas, self.eps)
            # scores = metrics_dict[metrics](hits, misses, fas, self.eps)
            if metrics != 'hss':
                scores = metrics_dict[metrics](hits, misses, fas, self.eps)
            else:
                cor = self._get_correct_negtives()
                scores = metrics_dict[metrics](hits, misses, fas, cor, self.eps)
            scores = scores.detach().cpu().numpy()
            for i, threshold in enumerate(self._thresholds):
                if self.keep_seq_len_dim:
                    score = scores[i]  # shape = (seq_len, )
                else:
                    score = scores[i].item()  # shape = (1, )
                if self.mode in ("0", "1"):
                    ret[threshold][metrics] = score
                elif self.mode in ("2", ):
                    ret[threshold][metrics] = np.mean(score).item()
                else:
                    raise NotImplementedError
                score_avg += score
            score_avg /= len(self._thresholds)
            if self.mode in ("0", "1"):
                ret["avg"][metrics] = score_avg
            elif self.mode in ("2",):
                ret["avg"][metrics] = np.mean(score_avg).item()
            else:
                raise NotImplementedError
        return ret

    @torch.no_grad()
    def get_single_frame_metrics(self, target, pred, metrics=['ssim', 'psnr', ]): #'cspr', 'cspr-4-avg', 'cspr-16-avg', 'cspr-4-max', 'cspr-16-max'
        metric_funcs = {
            'ssim': cal_SSIM,
            'psnr': cal_PSNR
        }
        metrics_dict = {}
        for metric in metrics:
            metric_fun = metric_funcs[metric]
            metrics_dict[metric] = metric_fun(gt=target*255., pred=pred*255., is_img=False)
        return metrics_dict
    
    @torch.no_grad()
    def get_crps(self, target, pred):
        """
        pred: (b, t, c, h, w)/(b, n, t, c, h, w)
        target: (b, t, c, h, w)
        """
        if len(pred.shape) == 5:
            pred = pred.unsqueeze(1)
        crps = cal_CRPS(gt=target, pred=pred, type='none')
        crps_avg_4 = cal_CRPS(gt=target, pred=pred, type='avg', scale=4)
        crps_avg_16 = cal_CRPS(gt=target, pred=pred, type='avg', scale=16)
        crps_max_4 = cal_CRPS(gt=target, pred=pred, type='max', scale=4)
        crps_max_16 = cal_CRPS(gt=target, pred=pred, type='max', scale=16)
        crps_dict = {
            'crps': crps,
            'crps_avg_4': crps_avg_4,
            'crps_avg_16': crps_avg_16,
            'crps_max_4': crps_max_4,
            'crps_max_16': crps_max_16
        }
        return crps_dict

    def reset(self):
        self.hits = self.hits*0
        self.misses = self.misses*0
        self.fas = self.fas*0

        self.hits_avg_pool_4 *= 0
        self.hits_avg_pool_16 *= 0
        self.hits_max_pool_4 *= 0
        self.hits_max_pool_16 *= 0

        self.misses_avg_pool_4 *= 0
        self.misses_avg_pool_16 *= 0
        self.misses_max_pool_4 *= 0
        self.misses_max_pool_16 *= 0
 
        self.fas_avg_pool_4 *= 0
        self.fas_avg_pool_16 *= 0
        self.fas_max_pool_4  *= 0
        self.fas_max_pool_16  *= 0

    def compute_sample(self, results):
        if self.dist_eval:
            self.synchronize_between_processes()

        metrics_dict = {
            'pod': self.pod,
            'csi': self.csi,
            'sucr': self.sucr,
            'bias': self.bias,
            'hss': self.hss,
            'far': self.far
        }
        
        ret = {}
        
        # Initialize structure to hold per-sample results
        for sample_name in results.keys():
            ret[sample_name] = {}
            for threshold in self.threshold_list:
                ret[sample_name][threshold] = {}
            ret[sample_name]["avg"] = {}
        # Process each sample separately
        for sample_name, sample_results in results.items():
            # Compute metrics for each threshold for this sample
            ret[sample_name]["mae"] = sample_results['mae']
            ret[sample_name]["mse"] = sample_results['mse']
            
            for index, threshold in enumerate(self.threshold_list):
                # Get basic contingency table values
                hits = sample_results["hits"][index]
                misses = sample_results["misses"][index]
                fas = sample_results["fas"][index]
                cor = sample_results["cor"][index]
                
                # Get pooled values
                hits_max_pool_4 = sample_results["hits_max_pool_4"][index]
                misses_max_pool_4 = sample_results["misses_max_pool_4"][index]
                fas_max_pool_4 = sample_results["fas_max_pool_4"][index]
                
                hits_max_pool_16 = sample_results["hits_max_pool_16"][index]
                misses_max_pool_16 = sample_results["misses_max_pool_16"][index]
                fas_max_pool_16 = sample_results["fas_max_pool_16"][index]
                
                hits_avg_pool_4 = sample_results["hits_avg_pool_4"][index]
                misses_avg_pool_4 = sample_results["misses_avg_pool_4"][index]
                fas_avg_pool_4 = sample_results["fas_avg_pool_4"][index]
                
                hits_avg_pool_16 = sample_results["hits_avg_pool_16"][index]
                misses_avg_pool_16 = sample_results["misses_avg_pool_16"][index]
                fas_avg_pool_16 = sample_results["fas_avg_pool_16"][index]

                # Store FSS scores directly
                ret[sample_name][threshold]['fss_pool1'] = sample_results["fss_pool1"][index].item()
                ret[sample_name][threshold]['fss_pool4'] = sample_results["fss_pool4"][index].item()
                ret[sample_name][threshold]['fss_pool8'] = sample_results["fss_pool8"][index].item()
                ret[sample_name][threshold]['fss_pool16'] = sample_results["fss_pool16"][index].item()

                # Compute basic metrics
                for metric in self.metrics_list:
                    if metric == 'csi-4-avg':
                        score = self.csi(hits_avg_pool_4, misses_avg_pool_4, fas_avg_pool_4, self.eps)
                    elif metric == 'csi-16-avg':
                        score = self.csi(hits_avg_pool_16, misses_avg_pool_16, fas_avg_pool_16, self.eps)
                    elif metric == 'csi-4-max':
                        score = self.csi(hits_max_pool_4, misses_max_pool_4, fas_max_pool_4, self.eps)
                    elif metric == 'csi-16-max':
                        score = self.csi(hits_max_pool_16, misses_max_pool_16, fas_max_pool_16, self.eps)
                    elif metric != 'hss':
                        score = metrics_dict[metric](hits, misses, fas, self.eps)
                    else:
                        score = metrics_dict[metric](hits, misses, fas, cor, self.eps)
                    
                    ret[sample_name][threshold][metric] = score.item()

            # Compute average metrics for this sample across thresholds
            for metric in self.metrics_list:
                threshold_scores = [ret[sample_name][t][metric] for t in self.threshold_list]
                ret[sample_name]["avg"][metric] = np.mean(threshold_scores)

        return ret

    def compute_fss(self, pred, target, threshold, patch_size):
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            ### patch pooling ###
            t, p = _threshold(target, pred, threshold)
            patch_t = rearrange(t, 'b t c (h p1) (w p2) -> b t c h p1 w p2', p1=patch_size, p2=patch_size)
            patch_t = patch_t.sum(dim=(-1, -3))
            patch_p = rearrange(p, 'b t c (h p1) (w p2) -> b t c h p1 w p2', p1=patch_size, p2=patch_size)
            patch_p = patch_p.sum(dim=(-1, -3))
            ### compute fss ###
            fss = 2 * torch.sum(patch_t * patch_p, dim=(-1, -2)) / (self.eps + torch.sum(patch_t*patch_t, dim=(-1, -2)) + torch.sum(patch_p * patch_p, dim=(-1, -2)))
            avg_fss = fss.mean()
        return avg_fss

    @torch.no_grad()
    def update_frame(self, pred, target, sample_names):
        """
        更新当前统计数据，每个样本独立存储计算结果。
        
        :param pred: 预测值 (B, ...)
        :param target: 真实值 (B, ...)
        :param sample_names: 样本名称列表 (长度 B)
        """
        # 确保 pred 和 target 在相同的设备上
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32, device=self.hits.device) # 1,128,128

        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32, device=self.hits.device) # 1,128,128

        pred = pred.unsqueeze(1).unsqueeze(1)
        target = target.unsqueeze(1).unsqueeze(1)
        
        _pred, _target = self.preprocess(pred, target)
        
        results = {}  
        

        results = {
            "hits": [],
            "misses": [],
            "fas": [],
            "cor": [],
            "fss_pool1": [],
            "fss_pool4": [],
            "fss_pool8": [],
            "fss_pool16": []
        }

        results["mae"] = torch.abs(_pred - _target).mean().item()
        results["mse"] = ((_pred - _target) ** 2).mean().item()
    
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits"].append(hits)
            results["misses"].append(misses)
            results["fas"].append(fas)
            results["cor"].append(cor)

            results["fss_pool1"].append(self.compute_fss(_pred, _target, threshold, 1))
            results["fss_pool4"].append(self.compute_fss(_pred, _target, threshold, 4))
            results["fss_pool8"].append(self.compute_fss(_pred, _target, threshold, 8))
            results["fss_pool16"].append(self.compute_fss(_pred, _target, threshold, 16))

        ## 处理 max pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='max')
        
        results["hits_max_pool_4"] = []
        results["misses_max_pool_4"] = []
        results["fas_max_pool_4"] = []
        
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_max_pool_4"].append(hits)
            results["misses_max_pool_4"].append(misses)
            results["fas_max_pool_4"].append(fas)

        ## 处理 max pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='max')
        
        results["hits_max_pool_16"] = []
        results["misses_max_pool_16"] = []
        results["fas_max_pool_16"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_max_pool_16"].append(hits)
            results["misses_max_pool_16"].append(misses)
            results["fas_max_pool_16"].append(fas)

        ## 处理 avg pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='avg')
    
        results["hits_avg_pool_4"] = []
        results["misses_avg_pool_4"] = []
        results["fas_avg_pool_4"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_avg_pool_4"].append(hits)
            results["misses_avg_pool_4"].append(misses)
            results["fas_avg_pool_4"].append(fas)

        ## 处理 avg pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='avg')

        results["hits_avg_pool_16"] = []
        results["misses_avg_pool_16"] = []
        results["fas_avg_pool_16"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_avg_pool_16"].append(hits)
            results["misses_avg_pool_16"].append(misses)
            results["fas_avg_pool_16"].append(fas)
                
        return results

    def compute_frame(self, results):
        if self.dist_eval:
            self.synchronize_between_processes()

        metrics_dict = {
            'pod': self.pod,
            'csi': self.csi,
            'sucr': self.sucr,
            'bias': self.bias,
            'hss': self.hss,
            'far': self.far
        }
        
        ret = {}
        
        for threshold in self.threshold_list:
            ret[threshold] = {}
        ret["avg"] = {}

        # Compute metrics for each threshold for this sample
        ret["mae"] = results['mae']
        ret["mse"] = results['mse']
            
        for index, threshold in enumerate(self.threshold_list):
            # Get basic contingency table values
            hits = results["hits"][index]
            misses = results["misses"][index]
            fas = results["fas"][index]
            cor = results["cor"][index]
            
            # Get pooled values
            hits_max_pool_4 = results["hits_max_pool_4"][index]
            misses_max_pool_4 = results["misses_max_pool_4"][index]
            fas_max_pool_4 = results["fas_max_pool_4"][index]
            
            hits_max_pool_16 = results["hits_max_pool_16"][index]
            misses_max_pool_16 = results["misses_max_pool_16"][index]
            fas_max_pool_16 = results["fas_max_pool_16"][index]
            
            hits_avg_pool_4 = results["hits_avg_pool_4"][index]
            misses_avg_pool_4 = results["misses_avg_pool_4"][index]
            fas_avg_pool_4 = results["fas_avg_pool_4"][index]
            
            hits_avg_pool_16 = results["hits_avg_pool_16"][index]
            misses_avg_pool_16 = results["misses_avg_pool_16"][index]
            fas_avg_pool_16 = results["fas_avg_pool_16"][index]

            # Store FSS scores directly
            ret[threshold]['fss_pool1'] = results["fss_pool1"][index].item()
            ret[threshold]['fss_pool4'] = results["fss_pool4"][index].item()
            ret[threshold]['fss_pool8'] = results["fss_pool8"][index].item()
            ret[threshold]['fss_pool16'] = results["fss_pool16"][index].item()

            # Compute basic metrics
            for metric in self.metrics_list:
                if metric == 'csi-4-avg':
                    score = self.csi(hits_avg_pool_4, misses_avg_pool_4, fas_avg_pool_4, self.eps)
                elif metric == 'csi-16-avg':
                    score = self.csi(hits_avg_pool_16, misses_avg_pool_16, fas_avg_pool_16, self.eps)
                elif metric == 'csi-4-max':
                    score = self.csi(hits_max_pool_4, misses_max_pool_4, fas_max_pool_4, self.eps)
                elif metric == 'csi-16-max':
                    score = self.csi(hits_max_pool_16, misses_max_pool_16, fas_max_pool_16, self.eps)
                elif metric != 'hss':
                    score = metrics_dict[metric](hits, misses, fas, self.eps)
                else:
                    score = metrics_dict[metric](hits, misses, fas, cor, self.eps)
                
                ret[threshold][metric] = score.item()

        # Compute average metrics for this sample across thresholds
        for metric in self.metrics_list:
            threshold_scores = [ret[t][metric] for t in self.threshold_list]
            ret["avg"][metric] = np.mean(threshold_scores)

        return ret

    def update_single_sample(self, pred, target, sample_names):
        """
        更新当前统计数据，每个样本独立存储计算结果。
        
        :param pred: 预测值 (B, ...)
        :param target: 真实值 (B, ...)
        :param sample_names: 样本名称列表 (长度 B)
        """
        # 确保 pred 和 target 在相同的设备上
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32, device=self.hits.device) # 1,128,128

        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32, device=self.hits.device) # 1,128,128
        
        _pred, _target = self.preprocess(pred, target)
        
        results = {}  
        

        results = {
            "hits": [],
            "misses": [],
            "fas": [],
            "cor": [],
            "fss_pool1": [],
            "fss_pool4": [],
            "fss_pool8": [],
            "fss_pool16": []
        }

        results["mae"] = torch.abs(_pred - _target).mean().item()
        results["mse"] = ((_pred - _target) ** 2).mean().item()
    
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits"].append(hits)
            results["misses"].append(misses)
            results["fas"].append(fas)
            results["cor"].append(cor)

            results["fss_pool1"].append(self.compute_fss(_pred, _target, threshold, 1))
            results["fss_pool4"].append(self.compute_fss(_pred, _target, threshold, 4))
            results["fss_pool8"].append(self.compute_fss(_pred, _target, threshold, 8))
            results["fss_pool16"].append(self.compute_fss(_pred, _target, threshold, 16))

        ## 处理 max pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='max')
        
        results["hits_max_pool_4"] = []
        results["misses_max_pool_4"] = []
        results["fas_max_pool_4"] = []
        
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_max_pool_4"].append(hits)
            results["misses_max_pool_4"].append(misses)
            results["fas_max_pool_4"].append(fas)

        ## 处理 max pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='max')
        
        results["hits_max_pool_16"] = []
        results["misses_max_pool_16"] = []
        results["fas_max_pool_16"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_max_pool_16"].append(hits)
            results["misses_max_pool_16"].append(misses)
            results["fas_max_pool_16"].append(fas)

        ## 处理 avg pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='avg')
    
        results["hits_avg_pool_4"] = []
        results["misses_avg_pool_4"] = []
        results["fas_avg_pool_4"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_avg_pool_4"].append(hits)
            results["misses_avg_pool_4"].append(misses)
            results["fas_avg_pool_4"].append(fas)

        ## 处理 avg pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='avg')

        results["hits_avg_pool_16"] = []
        results["misses_avg_pool_16"] = []
        results["fas_avg_pool_16"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_avg_pool_16"].append(hits)
            results["misses_avg_pool_16"].append(misses)
            results["fas_avg_pool_16"].append(fas)
                
        return results
    
class SEVIRSkillScore(object):
    def __init__(self,
                 layout='NHWT', # 数据的维度排列方式
                 mode='0',
                 seq_len=None,
                 preprocess_type='sevir',
                 threshold_list=[16, 74, 133, 160, 181, 219],
                 metrics_list=['csi', 'csi-4-avg', 'csi-16-avg',
                               'csi-4-max', 'csi-16-max', 'bias',
                                'sucr', 'pod', 'hss', 'far'], #['csi', 'bias', 'sucr', 'pod'],
                 dist_eval=False,
                #  device='cuda',
                 eps=1e-4,):
        self.layout = layout
        self.preprocess_type = preprocess_type
        self.threshold_list = threshold_list
        self.metrics_list = metrics_list
        self.eps = eps
        self.mode = mode
        self.seq_len = seq_len
        
        self.dist_eval = dist_eval
        # self.device = device
        
        if mode in ("0", ):
            self.keep_seq_len_dim = False
            state_shape = (len(self.threshold_list), )
        elif mode in ("1", "2"):
            self.keep_seq_len_dim = True
            assert isinstance(self.seq_len, int), "seq_len must be provided when we need to keep seq_len dim."
            state_shape = (len(self.threshold_list), self.seq_len)
        else:
            raise NotImplementedError(f"mode {mode} not supported!")
        
        self.hits = torch.zeros(state_shape)
        self.misses = torch.zeros(state_shape)
        self.fas = torch.zeros(state_shape)
        self.cor = torch.zeros(state_shape)

        ## pooling fss ##
        self.fss_pool1 = torch.zeros(state_shape)
        self.fss_pool4 = torch.zeros(state_shape)
        self.fss_pool8 = torch.zeros(state_shape)
        self.fss_pool16 = torch.zeros(state_shape)
        self.fss_cnt = 0


        ## pooling csi ##
        self.hits_avg_pool_4 = torch.zeros(state_shape)
        self.misses_avg_pool_4 = torch.zeros(state_shape)
        self.fas_avg_pool_4 = torch.zeros(state_shape)

        self.hits_max_pool_4 = torch.zeros(state_shape)
        self.misses_max_pool_4 = torch.zeros(state_shape)
        self.fas_max_pool_4 = torch.zeros(state_shape)

        self.hits_avg_pool_16 = torch.zeros(state_shape)
        self.misses_avg_pool_16 = torch.zeros(state_shape)
        self.fas_avg_pool_16 = torch.zeros(state_shape)

        self.hits_max_pool_16 = torch.zeros(state_shape)
        self.misses_max_pool_16 = torch.zeros(state_shape)
        self.fas_max_pool_16 = torch.zeros(state_shape)


    def pod(self, hits, misses, fas, eps):
        return hits / (hits + misses + eps)
    
    def far(self, hits, misses, fas, eps):
        return fas / (hits + fas + eps)

    def csi(self, hits, misses, fas, eps):
        return hits / (hits + misses + fas + eps)
    
    def sucr(self, hits, misses, fas, eps):
        return hits / (hits + fas + eps)

    # def bias(self, hits, misses, fas, eps):
    #     bias = (hits + fas) / (hits + misses + eps)
    #     logbias = torch.pow(bias / torch.log(torch.tensor(2.0)), 2.0)
    #     return logbias

    def bias(self, hits, misses, fas, eps):
        bias = (hits + fas) / (hits + misses + eps)
        return bias
    
    def hss(self, hits, misses, fas, cor, eps):
        hss = 2 * (hits * cor - misses * fas) / ((hits + misses) * (misses + cor) + (hits + fas) * (fas + cor) + eps)
        return hss


    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        dist.barrier()
        dist.all_reduce(self.hits)
        dist.all_reduce(self.misses)
        dist.all_reduce(self.fas)
        ### avg 4 ###
        dist.all_reduce(self.hits_avg_pool_4)
        dist.all_reduce(self.misses_avg_pool_4)
        dist.all_reduce(self.fas_avg_pool_4)
        ### max 4 ###
        dist.all_reduce(self.hits_max_pool_4)
        dist.all_reduce(self.misses_max_pool_4)
        dist.all_reduce(self.fas_max_pool_4)
        ### avg 16 ###
        dist.all_reduce(self.hits_avg_pool_16)
        dist.all_reduce(self.misses_avg_pool_16)
        dist.all_reduce(self.fas_avg_pool_16)
        ### max 16 ###
        dist.all_reduce(self.hits_max_pool_16)
        dist.all_reduce(self.misses_max_pool_16)
        dist.all_reduce(self.fas_max_pool_16)

    @property
    def hits_misses_fas_reduce_dims(self):
        if not hasattr(self, "_hits_misses_fas_reduce_dims"):
            seq_dim = self.layout.find('T')
            self._hits_misses_fas_reduce_dims = list(range(len(self.layout)))
            if self.keep_seq_len_dim:
                self._hits_misses_fas_reduce_dims.pop(seq_dim)
        return self._hits_misses_fas_reduce_dims

    def preprocess(self, pred, target):
        if self.preprocess_type == "sevir":
            pred = pred.detach() / (1. / 255.)
            target = target.detach() / (1. / 255.)
        elif self.preprocess_type == "meteonet":
            pred = pred.detach() / (1. / 70.)
            target = target.detach() / (1. / 70.)
        else:
            raise NotImplementedError
        return pred, target

    def preprocess_pool(self, pred, target, pool_size=4, type='avg'):
        if self.preprocess_type == "sevir":
            pred = pred.detach() / (1. / 255.)
            target = target.detach() / (1. / 255.)
        elif self.preprocess_type == "meteonet":
            pred = pred.detach() / (1. / 70.)
            target = target.detach() / (1. / 70.)
        b, t, _ , _, _ = pred.shape
        pred = rearrange(pred, 'b t c h w -> (b t) c h w')
        target = rearrange(target, 'b t c h w -> (b t) c h w')
        if type == 'avg':
            pred = F.avg_pool2d(pred, kernel_size=pool_size, stride=pool_size)
            target = F.avg_pool2d(target, kernel_size=pool_size, stride=pool_size)
        elif type == 'max':
            pred = F.max_pool2d(pred, kernel_size=pool_size, stride=pool_size)
            target = F.max_pool2d(target, kernel_size=pool_size, stride=pool_size)
        pred = rearrange(pred, '(b t) c h w -> b t c h w', b=b)
        target = rearrange(target, '(b t) c h w -> b t c h w', b=b)
        return pred, target


    def calc_seq_hits_misses_fas(self, pred, target, threshold):
        with torch.no_grad():
            t, p = _threshold(target, pred, threshold)
            hits = torch.sum(t * p, dim=self.hits_misses_fas_reduce_dims).int()
            misses = torch.sum(t * (1 - p), dim=self.hits_misses_fas_reduce_dims).int()
            fas = torch.sum((1 - t) * p, dim=self.hits_misses_fas_reduce_dims).int()
            cor = torch.sum((1 - t) * (1 - p), dim=self.hits_misses_fas_reduce_dims).int()
        return hits, misses, fas, cor
    
    def compute_fss(self, pred, target, threshold, patch_size):
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            ### patch pooling ###
            t, p = _threshold(target, pred, threshold)
            patch_t = rearrange(t, 'b t c (h p1) (w p2) -> b t c h p1 w p2', p1=patch_size, p2=patch_size)
            patch_t = patch_t.sum(dim=(-1, -3))
            patch_p = rearrange(p, 'b t c (h p1) (w p2) -> b t c h p1 w p2', p1=patch_size, p2=patch_size)
            patch_p = patch_p.sum(dim=(-1, -3))
            ### compute fss ###
            fss = 2 * torch.sum(patch_t * patch_p, dim=(-1, -2)) / (self.eps + torch.sum(patch_t*patch_t, dim=(-1, -2)) + torch.sum(patch_p * patch_p, dim=(-1, -2)))
            avg_fss = fss.mean()
        return avg_fss

    @torch.no_grad()
    def update_frame(self, pred, target, sample_names):
        """
        更新当前统计数据，每个样本独立存储计算结果。
        
        :param pred: 预测值 (B, ...)
        :param target: 真实值 (B, ...)
        :param sample_names: 样本名称列表 (长度 B)
        """
        # 确保 pred 和 target 在相同的设备上
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32, device=self.hits.device) # 1,128,128

        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32, device=self.hits.device) # 1,128,128

        pred = pred.unsqueeze(1).unsqueeze(1)
        target = target.unsqueeze(1).unsqueeze(1)
        
        _pred, _target = self.preprocess(pred, target)
        
        results = {}  
        

        results = {
            "hits": [],
            "misses": [],
            "fas": [],
            "cor": [],
            "fss_pool1": [],
            "fss_pool4": [],
            "fss_pool8": [],
            "fss_pool16": []
        }

        results["mae"] = torch.abs(_pred - _target).mean().item()
        results["mse"] = ((_pred - _target) ** 2).mean().item()
    
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits"].append(hits)
            results["misses"].append(misses)
            results["fas"].append(fas)
            results["cor"].append(cor)

            results["fss_pool1"].append(self.compute_fss(_pred, _target, threshold, 1))
            results["fss_pool4"].append(self.compute_fss(_pred, _target, threshold, 4))
            results["fss_pool8"].append(self.compute_fss(_pred, _target, threshold, 8))
            results["fss_pool16"].append(self.compute_fss(_pred, _target, threshold, 16))

        ## 处理 max pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='max')
        
        results["hits_max_pool_4"] = []
        results["misses_max_pool_4"] = []
        results["fas_max_pool_4"] = []
        
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_max_pool_4"].append(hits)
            results["misses_max_pool_4"].append(misses)
            results["fas_max_pool_4"].append(fas)

        ## 处理 max pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='max')
        
        results["hits_max_pool_16"] = []
        results["misses_max_pool_16"] = []
        results["fas_max_pool_16"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_max_pool_16"].append(hits)
            results["misses_max_pool_16"].append(misses)
            results["fas_max_pool_16"].append(fas)

        ## 处理 avg pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='avg')
    
        results["hits_avg_pool_4"] = []
        results["misses_avg_pool_4"] = []
        results["fas_avg_pool_4"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_avg_pool_4"].append(hits)
            results["misses_avg_pool_4"].append(misses)
            results["fas_avg_pool_4"].append(fas)

        ## 处理 avg pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='avg')

        results["hits_avg_pool_16"] = []
        results["misses_avg_pool_16"] = []
        results["fas_avg_pool_16"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_avg_pool_16"].append(hits)
            results["misses_avg_pool_16"].append(misses)
            results["fas_avg_pool_16"].append(fas)
                
        return results
    @torch.no_grad()
    def update(self, pred, target):
        ## pool 1 ##
        self.hits = self.hits.to(pred.device)
        self.misses = self.misses.to(pred.device)
        self.fas = self.fas.to(pred.device)
        self.cor = self.cor.to(pred.device)

        ### pooling fss ###
        self.fss_pool1 = self.fss_pool1.to(pred.device)
        self.fss_pool4 = self.fss_pool4.to(pred.device)
        self.fss_pool8 = self.fss_pool8.to(pred.device)
        self.fss_pool16 = self.fss_pool16.to(pred.device)


        _pred, _target = self.preprocess(pred, target)
        self.fss_cnt = self.fss_cnt + 1
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits[i] += hits
            self.misses[i] += misses
            self.fas[i] += fas
            self.cor[i] += cor
            self.fss_pool1[i] = (self.fss_pool1[i]*(self.fss_cnt-1) + self.compute_fss(_pred, _target, threshold, 1)) / self.fss_cnt
            self.fss_pool4[i] = (self.fss_pool4[i]*(self.fss_cnt-1) + self.compute_fss(_pred, _target, threshold, 4)) / self.fss_cnt
            self.fss_pool8[i] = (self.fss_pool8[i]*(self.fss_cnt-1) + self.compute_fss(_pred, _target, threshold, 8)) / self.fss_cnt
            self.fss_pool16[i] = (self.fss_pool16[i]*(self.fss_cnt-1) + self.compute_fss(_pred, _target, threshold, 16)) / self.fss_cnt
        
        ## max pool 4 ##
        self.hits_max_pool_4 = self.hits_max_pool_4.to(pred.device)
        self.misses_max_pool_4 = self.misses_max_pool_4.to(pred.device)
        self.fas_max_pool_4 = self.fas_max_pool_4.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='max')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_max_pool_4[i] += hits
            self.misses_max_pool_4[i] += misses
            self.fas_max_pool_4[i] += fas 
        ## max pool 16 ##
        self.hits_max_pool_16 = self.hits_max_pool_16.to(pred.device)
        self.misses_max_pool_16 = self.misses_max_pool_16.to(pred.device)
        self.fas_max_pool_16 = self.fas_max_pool_16.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='max')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_max_pool_16[i] += hits
            self.misses_max_pool_16[i] += misses
            self.fas_max_pool_16[i] += fas 
        ## avg pool 4 ##
        self.hits_avg_pool_4 = self.hits_avg_pool_4.to(pred.device)
        self.misses_avg_pool_4 = self.misses_avg_pool_4.to(pred.device)
        self.fas_avg_pool_4 = self.fas_avg_pool_4.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='avg')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_avg_pool_4[i] += hits
            self.misses_avg_pool_4[i] += misses
            self.fas_avg_pool_4[i] += fas 
        ## avg pool 16 ##
        self.hits_avg_pool_16 = self.hits_avg_pool_16.to(pred.device)
        self.misses_avg_pool_16 = self.misses_avg_pool_16.to(pred.device)
        self.fas_avg_pool_16 = self.fas_avg_pool_16.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='avg')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_avg_pool_16[i] += hits
            self.misses_avg_pool_16[i] += misses
            self.fas_avg_pool_16[i] += fas 

    @torch.no_grad()
    def update_sample(self, pred, target, sample_names):
        """
        更新当前统计数据，每个样本独立存储计算结果。
        
        :param pred: 预测值 (B, ...)
        :param target: 真实值 (B, ...)
        :param sample_names: 样本名称列表 (长度 B)
        """
        # 确保 pred 和 target 在相同的设备上
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32, device=self.hits.device)

        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32, device=self.hits.device)
        
        _pred, _target = self.preprocess(pred, target)
        _pred = _pred.unsqueeze(1)
        _target = _target.unsqueeze(1)
        
        results = {}  
        
        for sample_idx, sample_name in enumerate(sample_names):
            results[sample_name] = {
                "hits": [],
                "misses": [],
                "fas": [],
                "cor": [],
                "fss_pool1": [],
                "fss_pool4": [],
                "fss_pool8": [],
                "fss_pool16": []
            }

            results[sample_name]["mae"] = torch.abs(_pred[sample_idx] - _target[sample_idx]).mean().item()
            results[sample_name]["mse"] = ((_pred[sample_idx] - _target[sample_idx]) ** 2).mean().item()
        
            for i, threshold in enumerate(self.threshold_list):
                hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred[sample_idx], _target[sample_idx], threshold)
                results[sample_name]["hits"].append(hits)
                results[sample_name]["misses"].append(misses)
                results[sample_name]["fas"].append(fas)
                results[sample_name]["cor"].append(cor)

                results[sample_name]["fss_pool1"].append(self.compute_fss(_pred[sample_idx], _target[sample_idx], threshold, 1))
                results[sample_name]["fss_pool4"].append(self.compute_fss(_pred[sample_idx], _target[sample_idx], threshold, 4))
                results[sample_name]["fss_pool8"].append(self.compute_fss(_pred[sample_idx], _target[sample_idx], threshold, 8))
                results[sample_name]["fss_pool16"].append(self.compute_fss(_pred[sample_idx], _target[sample_idx], threshold, 16))

        ## 处理 max pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='max')
        _pred = _pred.unsqueeze(1)
        _target = _target.unsqueeze(1)
        for sample_idx, sample_name in enumerate(sample_names):
            results[sample_name]["hits_max_pool_4"] = []
            results[sample_name]["misses_max_pool_4"] = []
            results[sample_name]["fas_max_pool_4"] = []
            
            for i, threshold in enumerate(self.threshold_list):
                hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred[sample_idx], _target[sample_idx], threshold)
                results[sample_name]["hits_max_pool_4"].append(hits)
                results[sample_name]["misses_max_pool_4"].append(misses)
                results[sample_name]["fas_max_pool_4"].append(fas)

        ## 处理 max pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='max')
        _pred = _pred.unsqueeze(1)
        _target = _target.unsqueeze(1)
        for sample_idx, sample_name in enumerate(sample_names):
            results[sample_name]["hits_max_pool_16"] = []
            results[sample_name]["misses_max_pool_16"] = []
            results[sample_name]["fas_max_pool_16"] = []
            
            for i, threshold in enumerate(self.threshold_list):
                hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred[sample_idx], _target[sample_idx], threshold)
                results[sample_name]["hits_max_pool_16"].append(hits)
                results[sample_name]["misses_max_pool_16"].append(misses)
                results[sample_name]["fas_max_pool_16"].append(fas)

        ## 处理 avg pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='avg')
        _pred = _pred.unsqueeze(1)
        _target = _target.unsqueeze(1)
        for sample_idx, sample_name in enumerate(sample_names):
            results[sample_name]["hits_avg_pool_4"] = []
            results[sample_name]["misses_avg_pool_4"] = []
            results[sample_name]["fas_avg_pool_4"] = []
            
            for i, threshold in enumerate(self.threshold_list):
                hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred[sample_idx], _target[sample_idx], threshold)
                results[sample_name]["hits_avg_pool_4"].append(hits)
                results[sample_name]["misses_avg_pool_4"].append(misses)
                results[sample_name]["fas_avg_pool_4"].append(fas)

        ## 处理 avg pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='avg')
        _pred = _pred.unsqueeze(1)
        _target = _target.unsqueeze(1)
        for sample_idx, sample_name in enumerate(sample_names):
            results[sample_name]["hits_avg_pool_16"] = []
            results[sample_name]["misses_avg_pool_16"] = []
            results[sample_name]["fas_avg_pool_16"] = []
            
            for i, threshold in enumerate(self.threshold_list):
                hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred[sample_idx], _target[sample_idx], threshold)
                results[sample_name]["hits_avg_pool_16"].append(hits)
                results[sample_name]["misses_avg_pool_16"].append(misses)
                results[sample_name]["fas_avg_pool_16"].append(fas)
                
        return results

    @torch.no_grad()
    def update_frame(self, pred, target, sample_names):
        """
        更新当前统计数据，每个样本独立存储计算结果。
        
        :param pred: 预测值 (B, ...)
        :param target: 真实值 (B, ...)
        :param sample_names: 样本名称列表 (长度 B)
        """
        # 确保 pred 和 target 在相同的设备上
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32, device=self.hits.device) # 1,128,128

        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32, device=self.hits.device) # 1,128,128

        pred = pred.unsqueeze(1).unsqueeze(1)
        target = target.unsqueeze(1).unsqueeze(1)
        
        _pred, _target = self.preprocess(pred, target)
        
        results = {}  
        

        results = {
            "hits": [],
            "misses": [],
            "fas": [],
            "cor": [],
            "fss_pool1": [],
            "fss_pool4": [],
            "fss_pool8": [],
            "fss_pool16": []
        }

        results["mae"] = torch.abs(_pred - _target).mean().item()
        results["mse"] = ((_pred - _target) ** 2).mean().item()
    
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits"].append(hits)
            results["misses"].append(misses)
            results["fas"].append(fas)
            results["cor"].append(cor)

            results["fss_pool1"].append(self.compute_fss(_pred, _target, threshold, 1))
            results["fss_pool4"].append(self.compute_fss(_pred, _target, threshold, 4))
            results["fss_pool8"].append(self.compute_fss(_pred, _target, threshold, 8))
            results["fss_pool16"].append(self.compute_fss(_pred, _target, threshold, 16))

        ## 处理 max pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='max')
        
        results["hits_max_pool_4"] = []
        results["misses_max_pool_4"] = []
        results["fas_max_pool_4"] = []
        
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_max_pool_4"].append(hits)
            results["misses_max_pool_4"].append(misses)
            results["fas_max_pool_4"].append(fas)

        ## 处理 max pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='max')
        
        results["hits_max_pool_16"] = []
        results["misses_max_pool_16"] = []
        results["fas_max_pool_16"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_max_pool_16"].append(hits)
            results["misses_max_pool_16"].append(misses)
            results["fas_max_pool_16"].append(fas)

        ## 处理 avg pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='avg')
    
        results["hits_avg_pool_4"] = []
        results["misses_avg_pool_4"] = []
        results["fas_avg_pool_4"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_avg_pool_4"].append(hits)
            results["misses_avg_pool_4"].append(misses)
            results["fas_avg_pool_4"].append(fas)

        ## 处理 avg pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='avg')

        results["hits_avg_pool_16"] = []
        results["misses_avg_pool_16"] = []
        results["fas_avg_pool_16"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_avg_pool_16"].append(hits)
            results["misses_avg_pool_16"].append(misses)
            results["fas_avg_pool_16"].append(fas)
                
        return results

    def update_single_sample(self, pred, target, sample_names):
        """
        更新当前统计数据，每个样本独立存储计算结果。
        
        :param pred: 预测值 (B, ...)
        :param target: 真实值 (B, ...)
        :param sample_names: 样本名称列表 (长度 B)
        """
        # 确保 pred 和 target 在相同的设备上
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, dtype=torch.float32, device=self.hits.device) # 1,128,128

        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32, device=self.hits.device) # 1,128,128
        
        _pred, _target = self.preprocess(pred, target)
        
        results = {}  
        

        results = {
            "hits": [],
            "misses": [],
            "fas": [],
            "cor": [],
            "fss_pool1": [],
            "fss_pool4": [],
            "fss_pool8": [],
            "fss_pool16": []
        }

        results["mae"] = torch.abs(_pred - _target).mean().item()
        results["mse"] = ((_pred - _target) ** 2).mean().item()
    
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits"].append(hits)
            results["misses"].append(misses)
            results["fas"].append(fas)
            results["cor"].append(cor)

            results["fss_pool1"].append(self.compute_fss(_pred, _target, threshold, 1))
            results["fss_pool4"].append(self.compute_fss(_pred, _target, threshold, 4))
            results["fss_pool8"].append(self.compute_fss(_pred, _target, threshold, 8))
            results["fss_pool16"].append(self.compute_fss(_pred, _target, threshold, 16))

        ## 处理 max pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='max')
        
        results["hits_max_pool_4"] = []
        results["misses_max_pool_4"] = []
        results["fas_max_pool_4"] = []
        
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_max_pool_4"].append(hits)
            results["misses_max_pool_4"].append(misses)
            results["fas_max_pool_4"].append(fas)

        ## 处理 max pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='max')
        
        results["hits_max_pool_16"] = []
        results["misses_max_pool_16"] = []
        results["fas_max_pool_16"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_max_pool_16"].append(hits)
            results["misses_max_pool_16"].append(misses)
            results["fas_max_pool_16"].append(fas)

        ## 处理 avg pool 4 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='avg')
    
        results["hits_avg_pool_4"] = []
        results["misses_avg_pool_4"] = []
        results["fas_avg_pool_4"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_avg_pool_4"].append(hits)
            results["misses_avg_pool_4"].append(misses)
            results["fas_avg_pool_4"].append(fas)

        ## 处理 avg pool 16 ##
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='avg')

        results["hits_avg_pool_16"] = []
        results["misses_avg_pool_16"] = []
        results["fas_avg_pool_16"] = []
            
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            results["hits_avg_pool_16"].append(hits)
            results["misses_avg_pool_16"].append(misses)
            results["fas_avg_pool_16"].append(fas)
                
        return results
    

    def _get_hits_misses_fas(self, metric_name):
        if metric_name.endswith('-4-avg'):
            hits = self.hits_avg_pool_4
            misses = self.misses_avg_pool_4
            fas = self.fas_avg_pool_4
        elif metric_name.endswith('-16-avg'):
            hits = self.hits_avg_pool_16
            misses = self.misses_avg_pool_16
            fas = self.fas_avg_pool_16
        elif metric_name.endswith('-4-max'):
            hits = self.hits_max_pool_4
            misses = self.misses_max_pool_4
            fas = self.fas_max_pool_4
        elif metric_name.endswith('-16-max'):
            hits = self.hits_max_pool_16
            misses = self.misses_max_pool_16
            fas = self.fas_max_pool_16
        else:
            hits = self.hits
            misses = self.misses
            fas = self.fas
        return [hits, misses, fas]
    
    def _get_correct_negtives(self):
        return self.cor
    
    @torch.no_grad()
    def compute(self):
        if self.dist_eval:
            self.synchronize_between_processes()
        
        metrics_dict = {'pod': self.pod,
                        'csi': self.csi,
                        'csi-4-avg': self.csi, 
                        'csi-16-avg': self.csi,
                        'csi-4-max': self.csi, 
                        'csi-16-max': self.csi,
                        'sucr': self.sucr,
                        'bias': self.bias,
                        'hss': self.hss,
                        'far': self.far}
        ret = {}
        for threshold in self.threshold_list:
            ret[threshold] = {}
        ret["avg"] = {}
        
        ### adding fss score ###
        ret['avg']['fss_pool1'] = self.fss_pool1.mean().item()
        ret['avg']['fss_pool4'] = self.fss_pool4.mean().item()
        ret['avg']['fss_pool8'] = self.fss_pool8.mean().item()
        ret['avg']['fss_pool16'] = self.fss_pool16.mean().item()

        for i, threshold in enumerate(self.threshold_list):
            ret[threshold]['fss_pool1'] = self.fss_pool1[i].item()
            ret[threshold]['fss_pool4'] = self.fss_pool4[i].item()
            ret[threshold]['fss_pool8'] = self.fss_pool8[i].item()
            ret[threshold]['fss_pool16'] = self.fss_pool16[i].item()

        for metrics in self.metrics_list:
            if self.keep_seq_len_dim:
                score_avg = np.zeros((self.seq_len, ))
            else:
                score_avg = 0
            hits, misses, fas = self._get_hits_misses_fas(metrics)
            # scores = metrics_dict[metrics](self.hits, self.misses, self.fas, self.eps)
            if metrics != 'hss':
                scores = metrics_dict[metrics](hits, misses, fas, self.eps)
            else:
                cor = self._get_correct_negtives()
                scores = metrics_dict[metrics](hits, misses, fas, cor, self.eps)
            scores = scores.detach().cpu().numpy()
            for i, threshold in enumerate(self.threshold_list):
                if self.keep_seq_len_dim:
                    score = scores[i]  # shape = (seq_len, )
                else:
                    score = scores[i].item()  # shape = (1, )
                if self.mode in ("0", "1"):
                    ret[threshold][metrics] = score
                elif self.mode in ("2", ):
                    ret[threshold][metrics] = np.mean(score).item()
                else:
                    raise NotImplementedError
                score_avg += score
            score_avg /= len(self.threshold_list)
            if self.mode in ("0", "1"):
                ret["avg"][metrics] = score_avg
            elif self.mode in ("2",):
                ret["avg"][metrics] = np.mean(score_avg).item()
            else:
                raise NotImplementedError
        return ret

    # def compute_sample(self, results):
    #     if self.dist_eval:
    #         self.synchronize_between_processes()

    #     metrics_dict = {'pod': self.pod,
    #                     'csi': self.csi,
    #                     'csi-4-avg': self.csi, 
    #                     'csi-16-avg': self.csi,
    #                     'csi-4-max': self.csi, 
    #                     'csi-16-max': self.csi,
    #                     'sucr': self.sucr,
    #                     'bias': self.bias,
    #                     'hss': self.hss,
    #                     'far': self.far}
        
    #     ret = {}
    #     for threshold in self.threshold_list:
    #         ret[threshold] = {}
    #     ret["avg"] = {}

    #     # Initialize accumulators for metrics
    #     hits_accum = {threshold: [] for threshold in self.threshold_list}
    #     misses_accum = {threshold: [] for threshold in self.threshold_list}
    #     fas_accum = {threshold: [] for threshold in self.threshold_list}
    #     cor_accum = {threshold: [] for threshold in self.threshold_list}

    #     # Initialize FSS pool accumulators
    #     fss_pool1_accum = {threshold: [] for threshold in self.threshold_list}
    #     fss_pool4_accum = {threshold: [] for threshold in self.threshold_list}
    #     fss_pool8_accum = {threshold: [] for threshold in self.threshold_list}
    #     fss_pool16_accum = {threshold: [] for threshold in self.threshold_list}

    #     index = 0
    #     # Loop over the sample results and aggregate metrics
    #     for sample_name, sample_results in results.items():
    #         for threshold in self.threshold_list:
    #             # Aggregate hits, misses, fas, cor
    #             hits_accum[threshold].extend(sample_results["hits"][index])
    #             misses_accum[threshold].extend(sample_results["misses"][index])
    #             fas_accum[threshold].extend(sample_results["fas"][index])
    #             cor_accum[threshold].extend(sample_results["cor"][index])

    #             # Aggregate fss scores
    #             fss_pool1_accum[threshold].extend(sample_results["fss_pool1"][index])
    #             fss_pool4_accum[threshold].extend(sample_results["fss_pool4"][index])
    #             fss_pool8_accum[threshold].extend(sample_results["fss_pool8"][index])
    #             fss_pool16_accum[threshold].extend(sample_results["fss_pool16"][index])
                
    #             index = index + 1

    #     index = 0
    #     # Compute metrics for each threshold and for average
    #     for threshold in self.threshold_list:
    #         # Compute fss scores for each threshold
    #         ret[threshold]['fss_pool1'] = np.mean(fss_pool1_accum[threshold][index])
    #         ret[threshold]['fss_pool4'] = np.mean(fss_pool4_accum[threshold][index])
    #         ret[threshold]['fss_pool8'] = np.mean(fss_pool8_accum[threshold][index])
    #         ret[threshold]['fss_pool16'] = np.mean(fss_pool16_accum[threshold][index])
    #         index = index + 1

    #         # Compute other metrics
    #         for metrics in self.metrics_list:
    #             if metrics != 'hss':
    #                 scores = metrics_dict[metrics](hits_accum[threshold], misses_accum[threshold], fas_accum[threshold], self.eps)
    #             else:
    #                 cor = cor_accum[threshold]
    #                 scores = metrics_dict[metrics](hits_accum[threshold], misses_accum[threshold], fas_accum[threshold], cor, self.eps)
                
    #             scores = np.mean(scores.detach().cpu().numpy())
    #             ret[threshold][metrics] = scores

    #     # Compute average metrics
    #     for metrics in self.metrics_list:
    #         score_avg = np.mean([np.mean(metrics_dict[metrics](hits_accum[threshold], misses_accum[threshold], fas_accum[threshold], self.eps))
    #                             for threshold in self.threshold_list])
    #         ret["avg"][metrics] = score_avg

    #     return ret

    def compute_sample(self, results):
        if self.dist_eval:
            self.synchronize_between_processes()

        metrics_dict = {
            'pod': self.pod,
            'csi': self.csi,
            'sucr': self.sucr,
            'bias': self.bias,
            'hss': self.hss,
            'far': self.far
        }
        
        ret = {}
        
        # Initialize structure to hold per-sample results
        for sample_name in results.keys():
            ret[sample_name] = {}
            for threshold in self.threshold_list:
                ret[sample_name][threshold] = {}
            ret[sample_name]["avg"] = {}
        # Process each sample separately
        for sample_name, sample_results in results.items():
            # Compute metrics for each threshold for this sample
            ret[sample_name]["mae"] = sample_results['mae']
            ret[sample_name]["mse"] = sample_results['mse']
            
            for index, threshold in enumerate(self.threshold_list):
                # Get basic contingency table values
                hits = sample_results["hits"][index]
                misses = sample_results["misses"][index]
                fas = sample_results["fas"][index]
                cor = sample_results["cor"][index]
                
                # Get pooled values
                hits_max_pool_4 = sample_results["hits_max_pool_4"][index]
                misses_max_pool_4 = sample_results["misses_max_pool_4"][index]
                fas_max_pool_4 = sample_results["fas_max_pool_4"][index]
                
                hits_max_pool_16 = sample_results["hits_max_pool_16"][index]
                misses_max_pool_16 = sample_results["misses_max_pool_16"][index]
                fas_max_pool_16 = sample_results["fas_max_pool_16"][index]
                
                hits_avg_pool_4 = sample_results["hits_avg_pool_4"][index]
                misses_avg_pool_4 = sample_results["misses_avg_pool_4"][index]
                fas_avg_pool_4 = sample_results["fas_avg_pool_4"][index]
                
                hits_avg_pool_16 = sample_results["hits_avg_pool_16"][index]
                misses_avg_pool_16 = sample_results["misses_avg_pool_16"][index]
                fas_avg_pool_16 = sample_results["fas_avg_pool_16"][index]

                # Store FSS scores directly
                ret[sample_name][threshold]['fss_pool1'] = sample_results["fss_pool1"][index].item()
                ret[sample_name][threshold]['fss_pool4'] = sample_results["fss_pool4"][index].item()
                ret[sample_name][threshold]['fss_pool8'] = sample_results["fss_pool8"][index].item()
                ret[sample_name][threshold]['fss_pool16'] = sample_results["fss_pool16"][index].item()

                # Compute basic metrics
                for metric in self.metrics_list:
                    if metric == 'csi-4-avg':
                        score = self.csi(hits_avg_pool_4, misses_avg_pool_4, fas_avg_pool_4, self.eps)
                    elif metric == 'csi-16-avg':
                        score = self.csi(hits_avg_pool_16, misses_avg_pool_16, fas_avg_pool_16, self.eps)
                    elif metric == 'csi-4-max':
                        score = self.csi(hits_max_pool_4, misses_max_pool_4, fas_max_pool_4, self.eps)
                    elif metric == 'csi-16-max':
                        score = self.csi(hits_max_pool_16, misses_max_pool_16, fas_max_pool_16, self.eps)
                    elif metric != 'hss':
                        score = metrics_dict[metric](hits, misses, fas, self.eps)
                    else:
                        score = metrics_dict[metric](hits, misses, fas, cor, self.eps)
                    
                    ret[sample_name][threshold][metric] = score.item()

            # Compute average metrics for this sample across thresholds
            for metric in self.metrics_list:
                threshold_scores = [ret[sample_name][t][metric] for t in self.threshold_list]
                ret[sample_name]["avg"][metric] = np.mean(threshold_scores)

        return ret

    def compute_frame(self, results):
        if self.dist_eval:
            self.synchronize_between_processes()

        metrics_dict = {
            'pod': self.pod,
            'csi': self.csi,
            'sucr': self.sucr,
            'bias': self.bias,
            'hss': self.hss,
            'far': self.far
        }
        
        ret = {}
        
        for threshold in self.threshold_list:
            ret[threshold] = {}
        ret["avg"] = {}

        # Compute metrics for each threshold for this sample
        ret["mae"] = results['mae']
        ret["mse"] = results['mse']
            
        for index, threshold in enumerate(self.threshold_list):
            # Get basic contingency table values
            hits = results["hits"][index]
            misses = results["misses"][index]
            fas = results["fas"][index]
            cor = results["cor"][index]
            
            # Get pooled values
            hits_max_pool_4 = results["hits_max_pool_4"][index]
            misses_max_pool_4 = results["misses_max_pool_4"][index]
            fas_max_pool_4 = results["fas_max_pool_4"][index]
            
            hits_max_pool_16 = results["hits_max_pool_16"][index]
            misses_max_pool_16 = results["misses_max_pool_16"][index]
            fas_max_pool_16 = results["fas_max_pool_16"][index]
            
            hits_avg_pool_4 = results["hits_avg_pool_4"][index]
            misses_avg_pool_4 = results["misses_avg_pool_4"][index]
            fas_avg_pool_4 = results["fas_avg_pool_4"][index]
            
            hits_avg_pool_16 = results["hits_avg_pool_16"][index]
            misses_avg_pool_16 = results["misses_avg_pool_16"][index]
            fas_avg_pool_16 = results["fas_avg_pool_16"][index]

            # Store FSS scores directly
            ret[threshold]['fss_pool1'] = results["fss_pool1"][index].item()
            ret[threshold]['fss_pool4'] = results["fss_pool4"][index].item()
            ret[threshold]['fss_pool8'] = results["fss_pool8"][index].item()
            ret[threshold]['fss_pool16'] = results["fss_pool16"][index].item()

            # Compute basic metrics
            for metric in self.metrics_list:
                if metric == 'csi-4-avg':
                    score = self.csi(hits_avg_pool_4, misses_avg_pool_4, fas_avg_pool_4, self.eps)
                elif metric == 'csi-16-avg':
                    score = self.csi(hits_avg_pool_16, misses_avg_pool_16, fas_avg_pool_16, self.eps)
                elif metric == 'csi-4-max':
                    score = self.csi(hits_max_pool_4, misses_max_pool_4, fas_max_pool_4, self.eps)
                elif metric == 'csi-16-max':
                    score = self.csi(hits_max_pool_16, misses_max_pool_16, fas_max_pool_16, self.eps)
                elif metric != 'hss':
                    score = metrics_dict[metric](hits, misses, fas, self.eps)
                else:
                    score = metrics_dict[metric](hits, misses, fas, cor, self.eps)
                
                ret[threshold][metric] = score.item()

        # Compute average metrics for this sample across thresholds
        for metric in self.metrics_list:
            threshold_scores = [ret[t][metric] for t in self.threshold_list]
            ret["avg"][metric] = np.mean(threshold_scores)

        return ret
    
    @torch.no_grad()
    def get_single_frame_metrics(self, target, pred, metrics=['ssim', 'psnr', ]): #'cspr', 'cspr-4-avg', 'cspr-16-avg', 'cspr-4-max', 'cspr-16-max'
        metric_funcs = {
            'ssim': cal_SSIM,
            'psnr': cal_PSNR
        }
        metrics_dict = {}
        for metric in metrics:
            metric_fun = metric_funcs[metric]
            metrics_dict[metric] = metric_fun(gt=target*255., pred=pred*255., is_img=False)
        return metrics_dict
    
    @torch.no_grad()
    def get_crps(self, target, pred):
        """
        pred: (b, t, c, h, w)/(b, n, t, c, h, w)
        target: (b, t, c, h, w)
        """
        if len(pred.shape) == 5:
            pred = pred.unsqueeze(1)
        crps = cal_CRPS(gt=target, pred=pred, type='none')
        crps_avg_4 = cal_CRPS(gt=target, pred=pred, type='avg', scale=4)
        crps_avg_16 = cal_CRPS(gt=target, pred=pred, type='avg', scale=16)
        crps_max_4 = cal_CRPS(gt=target, pred=pred, type='max', scale=4)
        crps_max_16 = cal_CRPS(gt=target, pred=pred, type='max', scale=16)
        crps_dict = {
            'crps': crps,
            'crps_avg_4': crps_avg_4,
            'crps_avg_16': crps_avg_16,
            'crps_max_4': crps_max_4,
            'crps_max_16': crps_max_16
        }
        return crps_dict



    def reset(self):
        self.hits = self.hits*0
        self.misses = self.misses*0
        self.fas = self.fas*0

        self.hits_avg_pool_4 *= 0
        self.hits_avg_pool_16 *= 0
        self.hits_max_pool_4 *= 0
        self.hits_max_pool_16 *= 0

        self.misses_avg_pool_4 *= 0
        self.misses_avg_pool_16 *= 0
        self.misses_max_pool_4 *= 0
        self.misses_max_pool_16 *= 0
 
        self.fas_avg_pool_4 *= 0
        self.fas_avg_pool_16 *= 0
        self.fas_max_pool_4  *= 0
        self.fas_max_pool_16  *= 0


@torch.no_grad()
class cal_FVD:
    def __init__(self, use_gpu=True, resize_crop=False):
        '''
        iter_cal=True, gt.shape=pred.shape=[nb b t c h w]
        iter_cal=Fasle, gt.shape=pred.shape=[n t c h w]
        '''
        
        self.use_gpu = use_gpu
        self.resize_crop = resize_crop
        # detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
        self.detector = torch.jit.load("/mnt/cache/gongjunchao/workdir/radar_forecasting/utils/fvd/i3d_torchscript.pt").eval()
        if torch.cuda.is_available() and self.use_gpu:
            self.detector = self.detector.cuda()
        self.feats = []
    
    def preprocess(self, video):
        """
        video: (b, t, 1, h, w) in [0, 1]
        this function transform the domain to [-1, 1] 
        (b, t, 1, h, w) -> (b, t, 3, h, w)
        """
        video = video.repeat(1, 1, 3, 1, 1)
        video = video * 2 - 1
        return video

    @torch.no_grad()
    def __call__(self, videos_real, videos_fake):
        feats_fake = []
        feats_real = []
        detector_kwargs = dict(rescale=False, resize=False, return_features=True)
        
        videos_fake = self.preprocess(videos_fake)
        videos_real = self.preprocess(videos_real)

        videos_fake = einops.rearrange(
            self.bilinear_interpolation(videos_fake), 'n t c h w -> n c t h w'
        )
        videos_real = einops.rearrange(
            self.bilinear_interpolation(videos_real), 'n t c h w -> n c t h w'
        )
        if torch.cuda.is_available() and self.use_gpu:
            videos_fake, videos_real = videos_fake.cuda(), videos_real.cuda()
        # print(videos_fake.shape, videos_real.shape)
        # videos_fake = videos_fake.repeat(1, 1, 10, 1, 1)
        # videos_real = videos_real.repeat(1, 1, 10, 1, 1)
        feats_fake = self.detector(videos_fake, **detector_kwargs).cpu()
        feats_real = self.detector(videos_real, **detector_kwargs).cpu()
        self.feats.append(torch.stack([feats_fake, feats_real], dim=0))
        return
    
    def update(self, videos_real, videos_fake):
        self(videos_real=videos_real, videos_fake=videos_fake)
        return

    def _reset(self):
        self.feats = []

    def compute(self):
        feats = torch.cat(self.feats, dim=1)
        fake_feats = feats[0]
        real_feats = feats[1]
        fvd = self._cal_FVD(feats_fake=fake_feats, feats_real=real_feats)
        return fvd

    def bilinear_interpolation(self, image):
        N, T, C, H, W = image.shape
        def my_resize(img):
            img = img.view(-1, C, H, W)
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            img = img.view(N, T, C, 224, 224)  
            return img
        def my_resize_crop(img):
            img = img.view(-1, C, H, W)
            if H<W:
                img = F.interpolate(img, size=(224, int(W*224/H)), mode='bilinear', align_corners=False)
                img = img.view(N, T, C, 224, int(W*224/H))  
            else:   # W<=H
                img = F.interpolate(img, size=(int(H*224/W), 224), mode='bilinear', align_corners=False)
                img = img.view(N, T, C, int(H*224/W), 224)  
            return center_crop(img, (224, 224))
        if H == W and H < 224:
            return my_resize(img=image)
        elif self.resize_crop:
            return my_resize_crop(img=image)
        else: 
            return my_resize(img=image)

    def _cal_FVD(self, feats_fake, feats_real):
        def compute_fvd(feats_fake, feats_real):
            mu_gen, sigma_gen = compute_stats(feats_fake)
            mu_real, sigma_real = compute_stats(feats_real)
            m = np.square(mu_gen - mu_real).sum()
            s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
            fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
            return float(fid)

        def compute_stats(feats):
            feats = feats.reshape(-1, feats.shape[-1])
            mu = feats.mean(axis=0)
            sigma = np.cov(feats, rowvar=False)
            return mu, sigma
        return compute_fvd(feats_fake, feats_real)


@torch.no_grad()
class cal_LPIPS():
    def __init__(self, use_gpu=True):
        from utils.lpips_net.modules.lpips import LPIPS
        self.net = LPIPS(net_type='vgg', version='0.1').cuda()

    def _cal_LPIPS(self, pred, target):
        """
        shape of pred and target: (b, t, 1, h, w)
        val_range: [0, 1]
        """
        ### to -1, 1
        b, t, _, _, _ = pred.shape
        pred_img_view = rearrange(pred, 'b t c h w -> (b t) c h w')
        target_img_view = rearrange(target, 'b t c h w -> (b t) c h w')

        pred = self.preprocess(pred_img_view)
        target = self.preprocess(target_img_view)
        loss = self.net.forward(pred, target) ## (b,t, 1)
        loss = loss.mean()
        return loss.item()
    
    def preprocess(self, image):
        """
        radar_image: (b, 1, h, w) in [0, 1]
        this function transforms the shape from (b, 1, h, w) to (b, 3, h, w)
        this function also transform the domain [0, 1] to [-1, 1]  
        """
        image = image.repeat(1, 3, 1, 1)
        image = image * 2 - 1
        return image

    def __call__(self, pred, target):
        """
        pred&targe: (b, t, 1, h, w)
        """
        return self._cal_LPIPS(pred, target)
        

        



if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/lustre/gongjunchao/workdir_lustre/RankCast')

    eval_metrics = cal_LPIPS()

    inp = torch.clamp(torch.randn(2, 10, 1, 256, 256).cuda(), 0, 1)
    target = torch.clamp(torch.randn(2, 10, 1, 256, 256).cuda(), 0, 1)

    loss = eval_metrics(inp, target)

    #             #  layout: str = "NHWT",
    #             #  mode: str = "0",
    #             #  seq_len: Optional[int] = None,
    #             #  preprocess_type: str = "sevir",
    #             #  threshold_list: Sequence[int] = (16, 74, 133, 160, 181, 219),
    #             #  metrics_list: Sequence[str] = ("csi", "bias", "sucr", "pod"),
    #             #  eps: float = 1e-4,
    #             #  dist_sync_on_step: bool = False,
    #             #  ):
    # data_dict= {}
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # import numpy as np
    # ## b, t, c, h, w
    # torch.manual_seed(0)
    # data_dict['pred'] = torch.randn(3, 12, 1, 480, 480).to(device)
    # data_dict['gt'] = torch.randn(3, 12, 1, 480, 480).to(device)
    # ## sevir metrics compute ##
    # eval_metrics.update(pred=data_dict["pred"], target=data_dict['gt'])
    # losses = eval_metrics.compute()
    # single_frame_dict = eval_metrics.get_single_frame_metrics(target=data_dict['gt'], pred=data_dict['pred'])
    # crps_dict = eval_metrics.get_crps(target=data_dict['gt'], pred=data_dict['pred'])
    # import pdb; pdb.set_trace()

    # ## fvd compute ##
    # data_dict= {}
    # fvd_computer = cal_FID()
    # fvd_computer.update(data_dict['pred'].repeat(1, 1, 3, 1, 1), data_dict['gt'].repeat(1, 1, 3, 1, 1))
    # fvd_computer.update(data_dict['pred'].repeat(1, 1, 3, 1, 1), data_dict['gt'].repeat(1, 1, 3, 1, 1))
    # fvd = fvd_computer.compute()

    
    # _ = torch.manual_seed(123)
    # fvd_computer = cal_FID()
    # # generate two slightly overlapping image intensity distributions
    # data_dict = {}
    # data_dict['gt'] = torch.randint(0, 200, (100, 3, 256, 256), dtype=torch.uint8) / 255.0 
    # data_dict['pred'] = torch.randint(100, 255, (100, 3, 256, 256), dtype=torch.uint8) / 255.0
    # fvd_computer.update(data_dict['pred'], data_dict['gt'])
    # fvd = fvd_computer.compute()
    # print('fvd:', fvd)

## srun -p ai4earth --kill-on-bad-exit=1 --quotatype=auto --gres=gpu:0 python -u metrics.py ##