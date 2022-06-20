import os
import argparse
import collections

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import model as arch_module
from data_utils import dataset as dataset_module

from data_utils.dataset import BatchCollateFn
from parse_config import ConfigParser
from trainer import Trainer

import torch.nn as nn
import torch.optim as optim


SEED = 456
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config: ConfigParser, local_master: bool, logger=None):

    train_dataset = config.init_obj('train_dataset', dataset_module)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    train_data_loader = config.init_obj('train_data_loader', torch.utils.data.dataloader,
                                        dataset=train_dataset,
                                        sampler=train_sampler,
                                        shuffle=False,
                                        collate_fn=BatchCollateFn())

    val_dataset = config.init_obj('validation_dataset', dataset_module)
    val_data_loader = config.init_obj('val_data_loader', torch.utils.data.dataloader,
                                      dataset=val_dataset,
                                      collate_fn=BatchCollateFn())

    model = config.init_obj('model_arch', arch_module)

    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
    lr_scheduler = config.init_obj(
        'lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, optimizer,
                      config=config,
                      data_loader=train_data_loader,
                      valid_data_loader=val_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


def entry_point(config: ConfigParser):

    local_world_size = config['local_world_size']

    if torch.cuda.is_available():
        if torch.cuda.device_count() < local_world_size:
            raise RuntimeError(f'the number of GPU ({torch.cuda.device_count()}) is less than '
                               f'the number of processes ({local_world_size}) running on each node')
        local_master = config['local_rank'] == 0
    else:
        raise RuntimeError(
            'CUDA is not available, Distributed training is not supported.')

    env_dict = {
        key: os.environ[key]
        for key in ('MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE')
    }

    dist.init_process_group(backend='nccl', init_method='env://')

    main(config, local_master, None)

    dist.destroy_process_group()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Distributed Training')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to be available (default: all)')

    CustomArgs = collections.namedtuple(
        'CustomArgs', 'flags default type target help')

    config = ConfigParser.from_args(args, None)

    entry_point(config)
