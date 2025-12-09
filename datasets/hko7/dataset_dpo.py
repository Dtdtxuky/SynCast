import pandas as pd
import datetime
import cv2
import numpy as np
import threading
import os
import io
import logging
import struct
import time
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import argparse
try:
    from petrel_client import client
except:
    pass

import multiprocessing
from multiprocessing import Process, Queue, Pool
from multiprocessing import shared_memory


def convert_datetime_to_filepath(date_time):
    Image_Path = 'cluster2hdd_new:s3://weather_radar_datasets/HKO-7/radarPNG'
    ret = os.path.join("%04d" % date_time.year,
                       "%02d" % date_time.month,
                       "%02d" % date_time.day,
                       'RAD%02d%02d%02d%02d%02d00.png'
                       % (date_time.year - 2000, date_time.month, date_time.day,
                          date_time.hour, date_time.minute))
    ret = os.path.join(Image_Path, ret)
    return ret

def get_exclude_mask():
    with np.load('/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/datasets/hko7/hko7_12min_list/mask_dat.npz') as dat:
        exclude_mask = dat['exclude_mask'][:]
        return exclude_mask

def _load_frames(loader, datetime_clips, total_length, height, width, mask):
    _exclude_mask = mask
    paths = [convert_datetime_to_filepath(datetime_clips[i]) for i in range(len(datetime_clips))]
    read_storage = []
    for i in range(len(paths)):
        read_storage.append(loader(paths[i]))
    frame_dat = np.array(read_storage)
    frame_dat = frame_dat * _exclude_mask
    import cv2
    frame_data = np.array([cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR) for frame in frame_dat])
    data_batch = torch.from_numpy(frame_data) / 255.0
    return data_batch

class PetrelLoader:
    def __init__(self):
        try:
            from petrel_client import client
        except ImportError:
            raise ImportError('Please install petrel_client to enable '
                              'PetrelBackend.')
        self._client = client.Client("/mnt/shared-storage-user/xukaiyi/petrel-oss-python-sdk/petreloss.conf")

    def __call__(self, path):
        st = time.time()
        img_bytes = self._client.get(path)
        et = time.time()
        try:
            assert(img_bytes is not None)
        except:
            print(path)
            return None
        st = time.time()
        img_mem_view = memoryview(img_bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        frame = np.array(img)
        et = time.time()
        return frame

class data_hko(Dataset):
    def __init__(self, split, input_length, pred_length, base_freq, height=128, width=128, sa_way='rank_sample', metric='far', model='BaseModel',**kwargs):
        super().__init__()
        ## load data pkl ##
        if split == "train":
            pd_path = '/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/datasets/hko7/hko7_12min_list/hko7_rainy_train.pkl'
        elif split == "valid":
            pd_path = '/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/datasets/hko7/hko7_12min_list/hko7_rainy_valid.pkl'
        elif split == "test":
            pd_path = '/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/datasets/hko7/hko7_12min_list/hko7_rainy_test.pkl'
        
        self.input_length = input_length
        self.pred_length = pred_length
        self.total_length = self.input_length + self.pred_length

        self.df = pd.read_pickle(pd_path)
        self.init_times = self.df[slice(0, len(self.df), self.total_length)] 
        self.base_freq = base_freq
        self.height = height
        self.width = width
        self.split = split

        self.sa_way = sa_way
        self.metric = metric
        self.model = model 
        self.perferdata_dir = f"cluster3:s3://ai4earth-pool5-2/rankcast/hko7_128_12m_3h/{model}/{sa_way}"
        print(self.perferdata_dir)
        
        self.loader = PetrelLoader()
        print('{} number: {}'.format(str(self.split), len(self.init_times)))

        self.client = client.Client("/mnt/shared-storage-user/xukaiyi/petrel-oss-python-sdk/petreloss.conf")
        self._exclude_mask = 1-get_exclude_mask()[np.newaxis, :, :]

        print("Function arguments:")
        print("  split       :", split)
        print("  input_length:", input_length)
        print("  pred_length :", pred_length)
        print("  base_freq   :", base_freq)
        print("  height      :", height)
        print("  width       :", width)
        print("  kwargs      :", kwargs)
    def __len__(self):
        return len(self.init_times)
    

    def _load_frames(self, datetime_clips, total_length):
        paths = [convert_datetime_to_filepath(datetime_clips[i]) for i in range(len(datetime_clips))]
        read_storage = []
        for i in range(len(paths)):
            read_storage.append(self.loader(paths[i]))
        frame_dat = np.array(read_storage)
        frame_dat = frame_dat * self._exclude_mask
        data_batch = torch.from_numpy(frame_dat) / 255.0
        return data_batch

    def _load_win_samples(self, datetime_clips, mask):
        _exclude_mask = mask
        path = datetime_clips
        _exclude_mask = np.array([cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR) for frame in _exclude_mask])
        try:
            file_path = os.path.join(self.perferdata_dir,
                                    f"best_{self.metric}",
                                    f"best_{self.metric}_{path}")
            with io.BytesIO(self.client.get(file_path)) as f:
                frame_data = np.load(f)
            return torch.from_numpy(frame_data)
        except Exception as e:
            if self.split != 'test':
                print(f"[ERROR] Failed to load win sample: {path} -> {e}")
            return torch.zeros(15, 1, 128, 128, dtype=torch.float32)

    def _load_lose_samples(self, datetime_clips, mask):
        _exclude_mask = mask
        path = datetime_clips
        _exclude_mask = np.array([cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR) for frame in _exclude_mask])
        try:
            file_path = os.path.join(self.perferdata_dir,
                                    f"worst_{self.metric}",
                                    f"worst_{self.metric}_{path}")
            with io.BytesIO(self.client.get(file_path)) as f:
                frame_data = np.load(f)
            return torch.from_numpy(frame_data)
        except Exception as e:
            if self.split != 'test':
                print(f"[ERROR] Failed to load win sample: {path} -> {e}")
            return torch.zeros(15, 1, 128, 128, dtype=torch.float32)
        
    def _load_win_samples1(self, datetime_clips, mask):
        _exclude_mask = mask
        path = datetime_clips
        _exclude_mask = np.array([cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR) for frame in _exclude_mask])
        try:
            file_path = os.path.join(self.perferdata_dir,
                                    f"best",
                                    f"best_{path}")
            # print(file_path)
            with io.BytesIO(self.client.get(file_path)) as f:
                frame_data = np.load(f)
            return torch.from_numpy(frame_data)
        except Exception as e:
            if self.split == 'test':
                return torch.zeros(15, 1, 128, 128, dtype=torch.float32)
            if self.sa_way=='rank_sample_both':
                return None

    def _load_lose_samples1(self, datetime_clips, mask):
        _exclude_mask = mask
        path = datetime_clips
        _exclude_mask = np.array([cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR) for frame in _exclude_mask])
        try:
            file_path = os.path.join(self.perferdata_dir,
                                    f"worst",
                                    f"worst_{path}")
            with io.BytesIO(self.client.get(file_path)) as f:
                frame_data = np.load(f)
            return torch.from_numpy(frame_data)
        except Exception as e:
            if self.split == 'test':
                return torch.zeros(15, 1, 128, 128, dtype=torch.float32)
            if self.sa_way=='rank_sample_both':
                return None

    def rainfall_to_pixel(self, rainfall_intensity, a=58.53, b=1.56):
        # dBZ = 10b log(R) +10log(a)
        dBR = np.log10(rainfall_intensity) * 10.0
        dBZ = dBR * b + 10.0 * np.log10(a)
        pixel_vals = (dBZ + 10.0) / 70.0
        return pixel_vals
 
    def __getitem__(self, idx):
        start_time = datetime.datetime.strptime(self.init_times[idx], "%Y-%m-%d %H:%M:%S")
        datetime_clips = pd.date_range(start=start_time, periods=self.total_length, freq=self.base_freq)
        try:
            if self.sa_way != 'rank_sample_both':
                win_samples = self._load_win_samples(f"{self.init_times[idx]}.npy", mask=self._exclude_mask)
                lose_samples = self._load_lose_samples(f"{self.init_times[idx]}.npy", mask=self._exclude_mask)
            else:
                win_samples = self._load_win_samples1(f"{self.init_times[idx]}.npy", mask=self._exclude_mask)
                lose_samples = self._load_lose_samples1(f"{self.init_times[idx]}.npy", mask=self._exclude_mask)

            if win_samples == None:
                return self.__getitem__(idx+1)
            
            frame_data = _load_frames(self.loader, datetime_clips, total_length=self.total_length, height=self.height, width=self.width, mask=self._exclude_mask)
            packed_results = dict()
            packed_results['inputs'] = torch.unsqueeze(frame_data[:self.input_length], dim=1)
            packed_results['data_samples'] = torch.unsqueeze(frame_data[self.input_length:self.input_length+self.pred_length], dim=1)
            packed_results['win_samples'] = win_samples
            packed_results['lose_samples'] = lose_samples
            packed_results['file_name'] = f"{self.split}/{self.init_times[idx]}.npy"
            packed_results['dataset_name'] = "hko7_128_12m_3h"
            return packed_results
        except:
            print(f"error in reading {start_time}")
            return self.__getitem__(idx+1)
            

if __name__ == '__main__':
    height = 480
    width = 480
    base_freq = '12min'
    total_length =20 
    dataset = data_hko(split='valid', input_length=5, pred_length=15, base_freq=base_freq, sa_way='rank_sample_both')
    print(len(dataset))

    st_time = time.time()
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        print(data['inputs'].shape)
        print(data['data_samples'].shape)
        print(data['win_samples'].shape)
        print(data['lose_samples'].shape)
        ed_time = time.time()
        print("time cost: ", (ed_time - st_time)/(i + 1))
        if i == 5:
            break

### srun -p ai4earth --kill-on-bad-exit=1 --quotatype=reserved --gres=gpu:0 python -u hko7.py ###


# import pandas as pd
# import datetime
# import cv2
# import numpy as np
# import threading
# import os
# import io
# import logging
# import struct
# import time
# import torch
# import torch.distributed as dist
# from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn
# from torch.utils.data.distributed import DistributedSampler
# import argparse
# try:
#     from petrel_client import client
# except:
#     pass

# import multiprocessing
# from multiprocessing import Process, Queue, Pool
# from multiprocessing import shared_memory


# def convert_datetime_to_filepath(date_time):
#     Image_Path = 'cluster2hdd_new:s3://weather_radar_datasets/HKO-7/radarPNG'
#     ret = os.path.join("%04d" % date_time.year,
#                        "%02d" % date_time.month,
#                        "%02d" % date_time.day,
#                        'RAD%02d%02d%02d%02d%02d00.png'
#                        % (date_time.year - 2000, date_time.month, date_time.day,
#                           date_time.hour, date_time.minute))
#     ret = os.path.join(Image_Path, ret)
#     return ret

# def get_exclude_mask():
#     with np.load('/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/datasets/hko7/hko7_12min_list/mask_dat.npz') as dat:
#         exclude_mask = dat['exclude_mask'][:]
#         return exclude_mask

# def _load_frames(loader, datetime_clips, total_length, height, width, mask):
#     _exclude_mask = mask
#     paths = [convert_datetime_to_filepath(datetime_clips[i]) for i in range(len(datetime_clips))]
#     read_storage = []
#     for i in range(len(paths)):
#         read_storage.append(loader(paths[i]))
#     frame_dat = np.array(read_storage)
#     frame_dat = frame_dat * _exclude_mask
#     import cv2
#     frame_data = np.array([cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR) for frame in frame_dat])
#     data_batch = torch.from_numpy(frame_data) / 255.0
#     return data_batch



# class PetrelLoader:
#     def __init__(self):
#         try:
#             from petrel_client import client
#         except ImportError:
#             raise ImportError('Please install petrel_client to enable '
#                               'PetrelBackend.')
#         self._client = client.Client("/mnt/shared-storage-user/xukaiyi/petrel-oss-python-sdk/petreloss.conf")

#     def __call__(self, path):
#         st = time.time()
#         img_bytes = self._client.get(path)
#         et = time.time()
#         try:
#             assert(img_bytes is not None)
#         except:
#             print(path)
#             return None
#         st = time.time()
#         img_mem_view = memoryview(img_bytes)
#         img_array = np.frombuffer(img_mem_view, np.uint8)
#         img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
#         frame = np.array(img)
#         et = time.time()
#         return frame

# class data_hko(Dataset):
#     def __init__(self, split, input_length, pred_length, base_freq, height=128, width=128, sa_way='rank_sample', metric='far', model='BaseModel',**kwargs):
#         super().__init__()
#         ## load data pkl ##
#         if split == "train":
#             pd_path = '/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/datasets/hko7/hko7_12min_list/hko7_rainy_train.pkl'
#         elif split == "valid":
#             pd_path = '/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/datasets/hko7/hko7_12min_list/hko7_rainy_valid.pkl'
#         elif split == "test":
#             pd_path = '/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/datasets/hko7/hko7_12min_list/hko7_rainy_test.pkl'
        
#         self.input_length = input_length
#         self.pred_length = pred_length
#         self.total_length = self.input_length + self.pred_length

#         self.df = pd.read_pickle(pd_path)
#         self.init_times = self.df[slice(0, len(self.df), self.total_length)] 
#         self.base_freq = base_freq
#         self.height = height
#         self.width = width
#         self.split = split

#         self.sa_way = sa_way
#         self.metric = metric
#         self.model = model 
#         self.perferdata_dir = f"cluster3:s3://ai4earth-pool5-2/rankcast/hko7_128_12m_3h/{model}/{sa_way}"
#         print(self.perferdata_dir)
        
#         self.loader = PetrelLoader()
#         print('{} number: {}'.format(str(self.split), len(self.init_times)))

#         self.client = client.Client("/mnt/shared-storage-user/xukaiyi/petrel-oss-python-sdk/petreloss.conf")
#         self._exclude_mask = 1-get_exclude_mask()[np.newaxis, :, :]

#         print("Function arguments:")
#         print("  split       :", split)
#         print("  input_length:", input_length)
#         print("  pred_length :", pred_length)
#         print("  base_freq   :", base_freq)
#         print("  height      :", height)
#         print("  width       :", width)
#         print("  kwargs      :", kwargs)
#     def __len__(self):
#         return len(self.init_times)

#     def _load_win_samples(self, datetime_clips, mask):
#         _exclude_mask = mask
#         path = datetime_clips
#         print(path)
#         try:
#             file_path = os.path.join(self.perferdata_dir,
#                                     f"best_{self.metric}",
#                                     f"best_{self.metric}_{path}")
#             print(file_path)
#             with io.BytesIO(self.client.get(file_path)) as f:
#                 frame_data = np.load(f)
#             return torch.from_numpy(frame_data)
#         except Exception as e:
#             if self.split != 'test':
#                 print(f"[ERROR] Failed to load win sample: {path} -> {e}")
#             return torch.zeros(15, 1, 128, 128, dtype=torch.float32)

#     def _load_lose_samples(self, datetime_clips, mask):
#         _exclude_mask = mask
#         path = datetime_clips
#         print(path)
#         try:
#             file_path = os.path.join(self.perferdata_dir,
#                                     f"worst_{self.metric}",
#                                     f"worst_{self.metric}_{path}")
#             print(file_path)
#             with io.BytesIO(self.client.get(file_path)) as f:
#                 frame_data = np.load(f)
#             return torch.from_numpy(frame_data)
#         except Exception as e:
#             if self.split != 'test':
#                 print(f"[ERROR] Failed to load win sample: {path} -> {e}")
#             return torch.zeros(15, 1, 128, 128, dtype=torch.float32)
        
#     def _load_win_samples1(self, datetime_clips, mask):
#         _exclude_mask = mask
#         path = datetime_clips
#         print(path)
#         try:
#             file_path = os.path.join(self.perferdata_dir,
#                                     f"best",
#                                     f"best_{path}")
#             print(file_path)
#             with io.BytesIO(self.client.get(file_path)) as f:
#                 frame_data = np.load(f)
#             return torch.from_numpy(frame_data)
#         except Exception as e:
#             if self.split != 'test':
#                 print(f"[ERROR] Failed to load win sample: {path} -> {e}")
#             return torch.zeros(15, 1, 128, 128, dtype=torch.float32)

#     def _load_lose_samples2(self, datetime_clips, mask):
#         _exclude_mask = mask
#         path = datetime_clips
#         print(path)
#         try:
#             file_path = os.path.join(self.perferdata_dir,
#                                     f"worst",
#                                     f"worst_{path}")
#             print(file_path)
#             with io.BytesIO(self.client.get(file_path)) as f:
#                 frame_data = np.load(f)
#             return torch.from_numpy(frame_data)
#         except Exception as e:
#             if self.split != 'test':
#                 print(f"[ERROR] Failed to load win sample: {path} -> {e}")
#             return torch.zeros(15, 1, 128, 128, dtype=torch.float32)
        
        
#     def _load_frames(self, datetime_clips, total_length):
#         paths = [convert_datetime_to_filepath(datetime_clips[i]) for i in range(len(datetime_clips))]
#         read_storage = []
#         for i in range(len(paths)):
#             read_storage.append(self.loader(paths[i]))
#         frame_dat = np.array(read_storage)
#         frame_dat = frame_dat * self._exclude_mask
#         data_batch = torch.from_numpy(frame_dat) / 255.0
#         return data_batch

#     def rainfall_to_pixel(self, rainfall_intensity, a=58.53, b=1.56):
#         # dBZ = 10b log(R) +10log(a)
#         dBR = np.log10(rainfall_intensity) * 10.0
#         dBZ = dBR * b + 10.0 * np.log10(a)
#         pixel_vals = (dBZ + 10.0) / 70.0
#         return pixel_vals
 
#     def __getitem__(self, idx):
#         start_time = datetime.datetime.strptime(self.init_times[idx], "%Y-%m-%d %H:%M:%S")
#         datetime_clips = pd.date_range(start=start_time, periods=self.total_length, freq=self.base_freq)

#         frame_data = _load_frames(self.loader, datetime_clips, total_length=self.total_length, height=self.height, width=self.width, mask=self._exclude_mask)
#         print(f"frame_data shape: {frame_data.shape}")
#         if self.sa_way != 'rank_sample_both':
#             win_samples = self._load_win_samples(f"{self.init_times[idx]}.npy", mask=self._exclude_mask)
#             lose_samples = self._load_lose_samples(f"{self.init_times[idx]}.npy", mask=self._exclude_mask)
#         else:
#             win_samples = self._load_win_samples2(f"{self.init_times[idx]}.npy", mask=self._exclude_mask)
#             lose_samples = self._load_lose_samples2(f"{self.init_times[idx]}.npy", mask=self._exclude_mask)
#         packed_results = dict()
#         packed_results['inputs'] = torch.unsqueeze(frame_data[:self.input_length], dim=1)
#         packed_results['data_samples'] = torch.unsqueeze(frame_data[self.input_length:self.input_length+self.pred_length], dim=1)
#         packed_results['win_samples'] = win_samples
#         packed_results['lose_samples'] = lose_samples
#         packed_results['file_name'] = f"{self.split}/{self.init_times[idx]}.npy"
#         packed_results['dataset_name'] = "hko7_128_12m_3h"
            
#         # try:
#         #     frame_data = _load_frames(self.loader, datetime_clips, total_length=self.total_length, height=self.height, width=self.width, mask=self._exclude_mask)
#         #     if self.sa_way != 'rank_sample_both':
#         #         win_samples = self._load_win_samples(file)
#         #         lose_samples = self._load_lose_samples(file)
#         #     else:
#         #         win_samples = self._load_win_samples(file)
#         #         lose_samples = self._load_lose_samples(file)
#         #     packed_results = dict()
#         #     packed_results['inputs'] = torch.unsqueeze(frame_data[:self.input_length], dim=1)
#         #     packed_results['data_samples'] = torch.unsqueeze(frame_data[self.input_length:self.input_length+self.pred_length], dim=1)
#         #     packed_results['win_samples'] = win_samples
#         #     packed_results['lose_samples'] = lose_samples
#         #     packed_results['file_name'] = f"{self.split}/{self.init_times[idx]}.npy"
#         #     packed_results['dataset_name'] = "hko7_128_12m_3h"
#         #     return packed_results
#         # except:
#         #     print(f"error in reading {start_time}")
#         #     return self.__getitem__(idx+1)
            

# if __name__ == '__main__':
#     height = 480
#     width = 480
#     base_freq = '12min'
#     total_length =20 
#     dataset = data_hko(split='valid', input_length=5, pred_length=15, base_freq=base_freq)
#     print(len(dataset))

#     st_time = time.time()
#     for i in range(len(dataset)):
#         data = dataset.__getitem__(i)
#         print(data['inputs'].shape)
#         print(data['data_samples'].shape)
#         print(data['win_samples'].shape)
#         ed_time = time.time()
#         print("time cost: ", (ed_time - st_time)/(i + 1))
#         if i == 5:
#             break

# ### srun -p ai4earth --kill-on-bad-exit=1 --quotatype=reserved --gres=gpu:0 python -u hko7.py ###
