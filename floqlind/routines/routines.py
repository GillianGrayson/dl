import torch
from torch.utils.data import Subset


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(f'Device: {device}')
    return device


def train_val_dataset(dataset, val_split=0.25, seed=42):
    print(f'original dataset size : {len(dataset)}')
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    datasets = {}
    datasets['train'], datasets['val'] = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    print(f'train dataset size : {len(datasets["train"])}')
    print(f'val dataset size : {len(datasets["val"])}')
    return datasets