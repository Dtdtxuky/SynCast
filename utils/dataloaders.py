from functools import partial
import weakref
import torch
import torch.utils.data



class MultiDatasetDummySampler:
    def __init__(self):
        self.dataloader = None

    def set_epoch(self, epoch):
        if comm.get_world_size() > 1:
            for dataloader in self.dataloader.dataloaders:
                dataloader.sampler.set_epoch(epoch)
        return


class MultiDatasetDataloader:
    ### reference to https://github1s.com/Pointcept/Pointcept/blob/main/pointcept/datasets/dataloader.py ###
    """
    Multiple Datasets Dataloader, batch data from a same dataset and mix up ratio determined by loop of each sub dataset.
    The overall length is determined by the main dataset (first) and loop of concat dataset.
    """

    def __init__(
        self,
        concat_dataset,
        split,
        batch_size_per_gpu: int,
        dataloader_params: dict,
        sampler_params: dict,
        num_worker_per_gpu=0, ## dervie from dataloader_params
        mix_prob=0,
        seed=None,
    ):
        from utils.distributedsample import TrainingSampler
        try :
            from petrel_client.utils.data import DataLoader
        except:
            from torch.utils.data import DataLoader
        ### only use this dataloader in training ###
        assert split == 'train'

        self.datasets = concat_dataset.datasets
        self.ratios = [dataset.loop for dataset in self.datasets]
        # reset data loop, original loop serve as ratios
        for dataset in self.datasets:
            dataset.loop = 1
        # # determine union training epoch by main dataset
        # self.datasets[0].loop = concat_dataset.loop
        # build sub-dataloaders
        num_worker_per_gpu = dataloader_params['num_workers']
        num_workers = num_worker_per_gpu // len(self.datasets)
        dataloader_params['num_workers'] = num_workers

        self.dataloaders = []
        for dataset_id, dataset in enumerate(self.datasets):
            # if comm.get_world_size() > 1:
            #     sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            # else:
            #     sampler = None
            sampler = TrainingSampler(size=len(dataset), shuffle=True)
            dataloader = DataLoader(
                dataset, batch_size=batch_size_per_gpu,
                sampler=sampler, **dataloader_params
            )

            self.dataloaders.append(
                dataloader
            )
            

    def __iter__(self):
        iterator = [iter(dataloader) for dataloader in self.dataloaders]
        while True:
            for i in range(len(self.ratios)):
                for _ in range(self.ratios[i]):
                    try:
                        batch = next(iterator[i])
                    except StopIteration:
                        if i == 0:
                            return
                        else:
                            iterator[i] = iter(self.dataloaders[i])
                            batch = next(iterator[i])
                    yield batch

    def __len__(self):
        main_data_loader_length = len(self.dataloaders[0])
        return (
            main_data_loader_length // self.ratios[0] * sum(self.ratios)
            + main_data_loader_length % self.ratios[0]
        )
