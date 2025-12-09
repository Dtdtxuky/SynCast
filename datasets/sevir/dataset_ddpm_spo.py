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

def get_sevir_dataset(split, input_length=13, pred_length=12, data_dir='/mnt/data/oss_beijing/video_prediction_dataset/sevir/sevir', base_freq='5min', height=384, width=384, guidance=0.1, **kwargs):
    if data_dir == 'radar:s3://weather_radar_datasets/sevir':
        return sevir_sproject(split, input_length=input_length, pred_length=pred_length, data_dir=data_dir, base_freq=base_freq, height=height, width=width, guidance=guidance, **kwargs)
    else:
        raise NotImplementedError



class sevir_sproject(Dataset):
    def __init__(self, split, metric = 'csi', model = 'ddim', input_length=6, pred_length=18, data_dir='radar:s3://weather_radar_datasets/sevir', base_freq='5min', height=384, width=384, guidance=0.1, **kwargs):
        super().__init__()
        assert input_length == 6, pred_length==18
        self.input_length = input_length
        self.pred_length = pred_length

        self.height = height
        self.width = width

        self.file_list = self._init_file_list(split, metric, model, guidance)
        
        ## sproject client ##
        self.client = Client("~/petreloss.conf")

        ## partial valid for efficiency in diffusion test ##
        self.partial_valid = kwargs.get('partial_valid', 1.0)
        if self.partial_valid < 1.0 and split == 'valid':
            ## set random seed to pertube self.fiel_list ##
            indices = np.arange(len(self.file_list))
            np.random.seed(0)
            np.random.shuffle(indices)
            cut_indices = indices[:int(len(self.file_list)*self.partial_valid)]
            self.file_list = [self.file_list[i] for i in cut_indices]
            print(f'Partial valid mode: {self.partial_valid} with {len(self.file_list)} samples')


    def _init_file_list(self, split, metric, model, guidance=0.1):
        if split == 'train':
            print('train')
            txt_path = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/sevir_info/train.txt'
        elif split == 'valid':
            print('valid')
            print('metric:', metric)
            print('model:', model)
            txt_path = f'/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/test/ddpm_ddim_lose_sample.txt'
        elif split == 'test':
            print('test')
            txt_path = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/sevir_info/test.txt'
        files = []
        with open(f'{txt_path}', 'r') as file:
            for line in file.readlines():
                if ' ' in line:
                    files.append(line.split(' ')[-1].strip())
                else:
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
        file_path = 'cluster2hdd_new:s3://weather_radar_datasets/sevir/val/' + file.split('sample0_')[-1]
        try:
            with io.BytesIO(self.client.get(file_path)) as f:
                frame_data = np.load(f)
            tensor = torch.from_numpy(frame_data) / 255
            return tensor.permute(3, 0, 1, 2)  # t, c, h, w
        except Exception as e:
            file_path = 'cluster2hdd_new:' + file
            with io.BytesIO(self.client.get(file_path)) as f:
                frame_data = np.load(f)
            tensor = torch.from_numpy(frame_data) / 255
            return tensor.permute(3, 0, 1, 2)  # t, c, h, w
        
    def _load_winframes(self, file):
        # file_path = 'cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin/ddim_sample/frame_rank/' + file
        # file_path = 'cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_fintune/diffcast_fin/' + file.replace('frame_', '_frame_')
        file_path = 'cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_fintune/diffcast_fin/' + file
        file_path = file_path.replace('lose', 'win')
        try:
            with io.BytesIO(self.client.get(file_path)) as f:
                frame_data = np.load(f)
            return torch.from_numpy(frame_data)
        except Exception as e:
            # print(f"Error loading {file_path}: {str(e)}")
            return torch.from_numpy(np.random.rand(18, 1, 128, 128).astype(np.float32))

    def _load_loseframes(self, file):
        ## 最差的sample用自己生成的
        # local_path = 'bysample_lose_far_sample0_' + file.split('sample0_')[-1]
        file_path = 'cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_fintune/diffcast_fin/' + file
        
        try:
            with io.BytesIO(self.client.get(file_path)) as f:
                frame_data = np.load(f)
            return torch.from_numpy(frame_data)
        except Exception as e:
            # print(f"Error loading {file_path}: {str(e)}")
            return torch.from_numpy(np.random.rand(18, 1, 128, 128).astype(np.float32))

    def _load_winframes_another(self, file):
        # file_path = 'cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin/ddim_sample/frame_rank/' + file
        file_path = 'cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_fintune/diffcast_fin/' + file.replace('frame_', '_frame_')
        file_path = file_path.replace('lose', 'win').replace('far', 'csi')
        
        try:
            with io.BytesIO(self.client.get(file_path)) as f:
                frame_data = np.load(f)
            return torch.from_numpy(frame_data)
        except Exception as e:
            # print(f"Error loading {file_path}: {str(e)}")
            return torch.from_numpy(np.random.rand(18, 1, 128, 128).astype(np.float32))

    def _load_loseframes_another(self, file):
        # farme-csi/far
        local_path = 'bysample_lose_csi_sample0_' + file.split('sample0_')[-1]
        file_path = 'cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_fintune/diffcast_fin/' + local_path
        # frame-csi and far
        # file_path = 'cluster3:s3://zwl2/rankcast/val_data/sevir_128_10m_3h/ddpmcast_finetune/diffcast_fin/ddim_sample/frame_rank/' + file
        try:
            with io.BytesIO(self.client.get(file_path)) as f:
                frame_data = np.load(f)
            return torch.from_numpy(frame_data)
        except Exception as e:
            # print(f"Error loading {file_path}: {str(e)}")
            return torch.from_numpy(np.random.rand(18, 1, 128, 128).astype(np.float32))
          

    def get_file_name(self, index):
        file = self.file_list[index]
        file_name = file.split('/')[-1]
        return file_name

    def _validate_files(self):
        """过滤无效文件路径"""
        valid_files = []
        for file in self.file_list:
            # 检查主数据文件
            main_path = 'cluster2hdd_new:' + file
            # 检查lose数据文件
            lose_file = file.split('/')[-1]
            lose_path = 'cluster3:s3://zwl2/rankcast/val_data/flow_single/sevir_128_10m_3h/flowcast/diffcast/' + lose_file
            try:
                # 双重校验文件是否存在
                if self.client.contains(main_path) and self.client.contains(lose_path):
                    valid_files.append(file)
                else:
                    print(f"Missing files for {file}")
            except:
                print(f"Error checking {file}")
        
        print(f"Original files: {len(self.file_list)} -> Valid files: {len(valid_files)}")
        self.file_list = valid_files

    def __getitem__(self, index):
        max_attempts = len(self.file_list)
        
        for attempt in range(max_attempts):
            current_idx = (index + attempt) % len(self.file_list)
            file = self.file_list[current_idx]
            
            frame_data = self._load_frames(file)
            win_data = self._load_winframes(file)
            lose_data = self._load_loseframes(file)

            ## 取另一个指标的win lose sample
            win_data_1 = self._load_winframes_another(file)
            lose_data_1 = self._load_loseframes_another(file)
            
            # 数据预处理逻辑
            indices = list(range(1, 48, 2))
            frame_data = frame_data[indices]
            frame_data = self._resize(frame_data)
            
            return {
                'inputs': frame_data[:self.input_length],
                'data_samples': frame_data[self.input_length:self.input_length+self.pred_length],
                'win_samples': win_data,
                'lose_samples': lose_data,
                'file_name': self.get_file_name(current_idx),
                'ano_win_samples': win_data_1,
                'ano_lose_samples': lose_data_1, 
                'dataset_name': 'sevir_128_10m_3h_ddpm_dpo'
            }


    def _get_dummy_data(self):
        """生成与正常数据结构相同的虚拟数据"""
        dummy_shape = (self.input_length, 1, self.height, self.width)
        return {
            'inputs': torch.zeros(dummy_shape),
            'data_samples': torch.zeros((self.pred_length, 1, self.height, self.width)),
            'lose_samples': torch.zeros_like(torch.Tensor(1)),  # 根据实际lose_samples形状调整
            'file_name': 'dummy',
            'dataset_name': 'sevir_128_10m_3h_dpo'
        }
        
    # def __getitem__(self, index):
    #     max_attempts = len(self.file_list)
    #     attempts = 0
        
    #     for attmpt in range(max_attempts):
    #         file = self.file_list[index]
    #         frame_data = self._load_frames(file)
    #         lose_data = self._load_loseframes(file)
    #         if lose_data is not None and frame_data is not None:
    #             indices = list(range(1, 48, 2))
    #             frame_data = frame_data[indices]
    #             frame_data = self._resize(frame_data)
    #             packed_results = dict()
    #             packed_results['inputs'] = frame_data[:self.input_length]
    #             packed_results['data_samples'] = frame_data[self.input_length:self.input_length+self.pred_length]
    #             packed_results['lose_samples'] = lose_data
    #             packed_results['file_name'] = self.get_file_name(index)
    #             packed_results['dataset_name'] = 'sevir_128_10m_3h_dpo'
    #             return packed_results
    #         else:
    #             index = (index + 1) % len(self.file_list)
    #             attempts += 1


if __name__ == "__main__":
    dataset = get_sevir_dataset(split='valid', partial_valid=1, input_length=6, pred_length=18, data_dir='radar:s3://weather_radar_datasets/sevir', base_freq='10min', height=128, width=128)
    print(len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=1)
    for inp in loader:
        print(inp['file_name'])
        pass

    # import time
    # st_time = time.time()
    # for i in range(len(dataset)):
    #     print('------------------------i--------------------------', i)
    #     data = dataset.__getitem__(i)
    #     ed_time = time.time()
    #     print((ed_time - st_time)/(i+1))
    #     print(data['inputs'].shape)
    #     print(data['data_samples'].shape)
    #     print(data['file_name'])
    #     print(data['lose_samples'])

### srun -p ai4earth --kill-on-bad-exit=1 --quotatype=reserved --gres=gpu:0 python -u dataset_128_10m_3h.py ###