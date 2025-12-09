import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn as nn
import torch
try:
    from petrel_client.client import Client
except:
    pass
import io



class meteonet_24(Dataset):
    def __init__(self, split, input_length=6, pred_length=18, data_dir='cluster2:s3://meteonet_data/24Frames',base_freq='5min', height=128, width=128, sa_way='rank_sample_both', metric='far', model='Diffcast', **kwargs):
        super().__init__()
        assert input_length == 6, pred_length==18
        self.input_length = 6
        self.pred_length = 18
        self.total_length = self.input_length + self.pred_length

        self.file_list = self._init_file_list(split)
        self.split = split
        self.data_dir = data_dir
        self.client = Client("/mnt/shared-storage-user/xukaiyi/petrel-oss-python-sdk/petreloss.conf")
        self.width = width
        self.height = height
        self.sa_way = sa_way
        self.metric = metric
        self.model = model 
        self.perferdata_dir = f"cluster3:s3://ai4earth-pool5-2/rankcast/meteonet_128_10m_3h/{model}/{sa_way}"
        print(self.perferdata_dir)
        
    def _init_file_list(self, split):
        if split == 'train':
            txt_path = '/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/datasets/meteonet/path/train_3h.txt'
        elif split == 'valid':
            txt_path = '/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/datasets/meteonet/path/valid_3h.txt'
        elif split == 'test':
            txt_path = '/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/datasets/meteonet/path/test_3h.txt'
        files = []
        with open(f'{txt_path}', 'r') as file:
            for line in file.readlines():
                files.append(line.strip())
        return files
    
    def __len__(self):
        return len(self.file_list)

    def get_file_name(self, index):
        file = self.file_list[index]
        file_name = file.split('/')[-1]
        return file_name
    
    def _load_frames(self, file):
        file_path = os.path.join(self.data_dir, file)
        with io.BytesIO(self.client.get(file_path)) as f:
            frame_data = np.load(f)
        tensor = torch.from_numpy(frame_data) / 70 ##TODO: get max
        ## 1, h, w, t -> t, c, h, w
        tensor = tensor.unsqueeze(dim=1)
        return tensor
    

    def _load_win_samples(self, file):
        try:
            file_path = os.path.join(self.perferdata_dir,
                                    f"best_{self.metric}",
                                    f"best_{self.metric}_{file}")
            with io.BytesIO(self.client.get(file_path)) as f:
                frame_data = np.load(f)
            return torch.from_numpy(frame_data)
        except Exception as e:
            if self.split != 'test':
                print(f"[ERROR] Failed to load win sample: {file} -> {e}")
            return torch.zeros(18, 1, 128, 128, dtype=torch.float32)

    def _load_lose_samples(self, file):
        try:
            file_path = os.path.join(self.perferdata_dir,
                                    f"worst_{self.metric}",
                                    f"worst_{self.metric}_{file}")
            with io.BytesIO(self.client.get(file_path)) as f:
                frame_data = np.load(f)
            return torch.from_numpy(frame_data)
        except Exception as e:
            if self.split != 'test':
                print(f"[ERROR] Failed to load win sample: {file} -> {e}")
            return torch.zeros(18, 1, 128, 128, dtype=torch.float32)

    def _load_win_samples2(self, file):
        try:
            file_path = os.path.join(self.perferdata_dir,
                                    "best",
                                    f"best_{file}")
            with io.BytesIO(self.client.get(file_path)) as f:
                frame_data = np.load(f)
            return torch.from_numpy(frame_data)
        except Exception as e:
            if self.split == 'test':
                return torch.zeros(18, 1, 128, 128, dtype=torch.float32)
            if self.sa_way=='rank_sample_both':
                return None

    def _load_lose_samples2(self, file):
        try:
            file_path = os.path.join(self.perferdata_dir,
                                    "worst",
                                    f"worst_{file}")
            with io.BytesIO(self.client.get(file_path)) as f:
                frame_data = np.load(f)
            return torch.from_numpy(frame_data)
        except Exception as e:
            if self.split == 'test':
                return torch.zeros(18, 1, 128, 128, dtype=torch.float32)
            if self.sa_way=='rank_sample_both':
                return None

    
    def _resize(self, frame_data):
        _, _, H, W = frame_data.shape 
        if H != self.height or W != self.height:
            frame_data = nn.functional.interpolate(frame_data, size=(self.height, self.width), mode='bilinear')
        return frame_data
    
    def __getitem__(self, index):
        file = self.file_list[index]
        frame_data = self._load_frames(file)
        # print('frame_data shape', frame_data.shape)
        # resize from 400x400 to 128x128
        frame_data = self._resize(frame_data)
        if self.sa_way != 'rank_sample_both':
            win_samples = self._load_win_samples(file)
            lose_samples = self._load_lose_samples(file)
        else:
            win_samples = self._load_win_samples2(file)
            lose_samples = self._load_lose_samples2(file)

        if win_samples == None:
            return self.__getitem__(index+1)
            
        return {
            'inputs':frame_data[:self.input_length],
            'data_samples': frame_data[self.input_length:self.input_length+self.pred_length],
            'file_name': self.get_file_name(index),
            'win_samples': win_samples,
            'lose_samples': lose_samples,
            'dataset_name': f"meteonet_128_10m_3h_{self.sa_way}_{self.metric}_{self.model}" 
        }
    


import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors

def visualize_input_and_target_separately(inputs, targets, save_dir, sample_id=0, use_meteo_cmap=True):
    """
    将输入帧和输出帧分别横排画图，每一类保存为一张图

    Args:
        inputs (Tensor): [1, t_in, 1, H, W]
        targets (Tensor): [1, t_out, 1, H, W]
        save_dir (str): 输出目录
        sample_id (int): 样本编号，用于命名
        use_meteo_cmap (bool): 是否使用行业雷达色带
    """
    os.makedirs(save_dir, exist_ok=True)

    inputs_np = inputs[0, :, 0].cpu().numpy() * 70  # [t_in, H, W]
    targets_np = targets[0, :, 0].cpu().numpy() * 70  # [t_out, H, W]

    def plot_sequence(frames, title_prefix, save_name):
        n = frames.shape[0]
        H, W = frames.shape[1:]
        fig, axes = plt.subplots(1, n, figsize=(n * 2.5, 3))

        if use_meteo_cmap:
            cmap = colors.ListedColormap([
                'lavender','indigo','mediumblue','dodgerblue','skyblue','cyan',
                'olivedrab','lime','greenyellow','orange','red','magenta','pink'
            ])
            bound_max = max(np.max(frames), 56) + 10
            bounds = [0,4,8,12,16,20,24,32,40,48,56,bound_max]
            norm = colors.BoundaryNorm(bounds, cmap.N)
        else:
            cmap = 'gist_ncar'
            norm = None

        if n == 1:
            axes = [axes]  # 单帧时也统一为 list
        for t in range(n):
            ax = axes[t]
            ax.imshow(frames[t], cmap=cmap, norm=norm)
            ax.set_title(f'{title_prefix} t={t}')
            ax.axis('off')

        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes, orientation='vertical', shrink=0.7, pad=0.01
        )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, save_name), dpi=200)
        print(os.path.join(save_dir, save_name))
        plt.close()

    plot_sequence(inputs_np, title_prefix='Input', save_name=f'input_row_sample_{sample_id}.png')
    plot_sequence(targets_np, title_prefix='Target', save_name=f'target_row_sample_{sample_id}.png')


if __name__ == "__main__":
    dataset = meteonet_24(split='valid', input_length=6, pred_length=18, data_dir='cluster2_2:s3://meteonet_data/24Frames')
    print(len(dataset))

    import time
    st_time = time.time()
    _max = 0
    
    for i in range(len(dataset)):
        if i > 10:
            break
        
        sample = dataset[i]
        inputs = sample['inputs'].unsqueeze(0)       # [1, 6, 1, H, W]
        targets = sample['data_samples'].unsqueeze(0) # [1, 18, 1, H, W]
        win_samples = sample['win_samples']
        lose_samples = sample['lose_samples']
        print(win_samples.shape)
        print(lose_samples.shape)
        print(sample['data_samples'].max())
        print(sample['win_samples'].max())
        
        print(sample['file_name'])
        
        visualize_input_and_target_separately(
            inputs=inputs,
            targets=targets,
            save_dir='/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/datasets/meteonet/vis',
            sample_id=i,
            use_meteo_cmap=True
        )
        
        # data = dataset.__getitem__(i)
        # ed_time = time.time()
        # _max = max(data['inputs'].max(), _max, data['data_samples'].max())

        # print((ed_time - st_time)/(i+1))
        # print(data['inputs'].shape)
        # print(data['data_samples'].shape)
        # print(_max)

### srun -p ai4earth --kill-on-bad-exit=1 --quotatype=reserved --gres=gpu:0 python -u meteonet.py ###
        
