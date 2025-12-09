import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
import torch.nn as nn
try:
    from petrel_client.client import Client
except:
    pass
import io



# SEVIR Dataset constants
SEVIR_DATA_TYPES = ['vis', 'ir069', 'ir107', 'vil', 'lght']
SEVIR_RAW_DTYPES = {'vis': np.int16,
                    'ir069': np.int16,
                    'ir107': np.int16,
                    'vil': np.uint8,
                    'lght': np.int16}
LIGHTING_FRAME_TIMES = np.arange(- 120.0, 125.0, 5) * 60
SEVIR_DATA_SHAPE = {'lght': (48, 48), }
PREPROCESS_SCALE_SEVIR = {'vis': 1,  # Not utilized in original paper
                          'ir069': 1 / 1174.68,
                          'ir107': 1 / 2562.43,
                          'vil': 1 / 47.54,
                          'lght': 1 / 0.60517}
PREPROCESS_OFFSET_SEVIR = {'vis': 0,  # Not utilized in original paper
                           'ir069': 3683.58,
                           'ir107': 1552.80,
                           'vil': - 33.44,
                           'lght': - 0.02990}
PREPROCESS_SCALE_01 = {'vis': 1,
                       'ir069': 1,
                       'ir107': 1,
                       'vil': 1 / 255,  # currently the only one implemented
                       'lght': 1}
PREPROCESS_OFFSET_01 = {'vis': 0,
                        'ir069': 0,
                        'ir107': 0,
                        'vil': 0,  # currently the only one implemented
                        'lght': 0}

def get_sevir_dataset(split, input_length=13, pred_length=12, data_dir='/mnt/data/oss_beijing/video_prediction_dataset/sevir/sevir', base_freq='5min', height=384, width=384, **kwargs):
    if data_dir == 'radar:s3://weather_radar_datasets/sevir':
        return sevir_sproject(split, input_length=input_length, pred_length=pred_length, data_dir=data_dir, base_freq=base_freq, height=height, width=width, **kwargs)
    else:
        raise NotImplementedError



class sevir_sproject(Dataset):
    def __init__(self, split, input_length=6, pred_length=18, data_dir='radar:s3://weather_radar_datasets/sevir', base_freq='5min', height=384, width=384, **kwargs):
        super().__init__()
        assert input_length == 6, pred_length==18
        self.input_length = input_length
        self.pred_length = pred_length

        self.height = height
        self.width = width

        self.file_list = self._init_file_list(split)
        
        ## sproject client ##
        self.client = Client("/mnt/shared-storage-user/xukaiyi/petrel-oss-python-sdk/petreloss.conf")

        ## partial valid for efficiency in diffusion test ##
        self.partial_valid = kwargs.get('partial_valid', 1.0)
        print('-------------------------------------partial_valid--------------------------------', self.partial_valid)
        
        if self.partial_valid < 1.0 and split == 'valid':
            ## set random seed to pertube self.fiel_list ##
            indices = np.arange(len(self.file_list))
            np.random.seed(0)
            np.random.shuffle(indices)
            cut_indices = indices[:int(len(self.file_list)*self.partial_valid)]
            self.file_list = [self.file_list[i] for i in cut_indices]
            print(f'Partial valid mode: {self.partial_valid} with {len(self.file_list)} samples')


    def _init_file_list(self, split):
        if split == 'train':
            txt_path = '/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/sevir_info/train.txt'
        elif split == 'valid':
            txt_path = '/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/sevir_info/val.txt'
        elif split == 'test':
            txt_path = '/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/sevir_info/test.txt'
        files = []
        with open(f'{txt_path}', 'r') as file:
            for line in file.readlines():
                files.append(line.strip())
        print('---------------------------------', txt_path ,'----------------------------------------')
        return files
    
    def __len__(self):
        return len(self.file_list)
    
    def _resize(self, frame_data):
        _, _, H, W = frame_data.shape 
        if H != self.height or W != self.height:
            frame_data = nn.functional.interpolate(frame_data, size=(self.height, self.width), mode='bilinear')
        return frame_data

    def _load_frames(self, file):
        file_path = 'cluster2hdd_new:' + file
        with io.BytesIO(self.client.get(file_path)) as f:
            frame_data = np.load(f)
        tensor = torch.from_numpy(frame_data) / 255
        ## 1, h, w, t -> t, c, h, w
        tensor = tensor.permute(3, 0, 1, 2)
        return tensor
    
    def _load_loseframes(self, file):
        # 不用调整维度
        
        # 不用进行归一化/255
        
        # 直接读取的是预测结果
        
        pass

    def get_file_name(self, index):
        file = self.file_list[index]
        file_name = file.split('/')[-1]
        return file_name

    def __getitem__(self, index):
        file = self.file_list[index]
        frame_data = self._load_frames(file)
        ### transform frame data from 5min interval to 10min interval ###
        indices = list(range(1, 48, 2))
        frame_data = frame_data[indices]
        frame_data = self._resize(frame_data)
        packed_results = dict()
        packed_results['inputs'] = frame_data[:self.input_length]
        packed_results['data_samples'] = frame_data[self.input_length:self.input_length+self.pred_length]
        packed_results['file_name'] = self.get_file_name(index)
        packed_results['dataset_name'] = 'sevir_128_10m_3h'
        return packed_results


if __name__ == "__main__":
    dataset = get_sevir_dataset(split='valid', partial_valid=0.4, input_length=6, pred_length=18, data_dir='radar:s3://weather_radar_datasets/sevir', base_freq='10min', height=128, width=128)
    print(len(dataset))


    import time
    st_time = time.time()
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        ed_time = time.time()
        print((ed_time - st_time)/(i+1))
        print(data['inputs'].shape)
        print(data['data_samples'].shape)
        print(data['file_name'])

### srun -p ai4earth --kill-on-bad-exit=1 --quotatype=reserved --gres=gpu:0 python -u dataset_128_10m_3h.py ###