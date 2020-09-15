import pandas as pd
import numpy as np
import torch
import os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


# custom dataset
class TwoSpinsDataset(Dataset):
    def __init__(self, path, suffix, transforms=None):

        s = 16
        num_channels = 3

        self.norms = np.loadtxt(f'{path}/norm_dl_{suffix}.txt')
        num_points = self.norms.shape[0]

        fn_txt = f'{path}/props_dl_{suffix}.txt'
        fn_npz = f'{path}/props_dl_{suffix}.npz'

        if os.path.isfile(fn_npz):
            self.images = np.load(fn_npz)['images']
        else:
            data = np.loadtxt(fn_txt)

            data[:, 2] = np.angle(data[:, 0] + 1j * data[:, 1])

            for n_id in range(0, num_channels):
                data[:, n_id] = (255.0 * (data[:, n_id] - np.min(data[:, n_id])) / np.ptp(data[:, n_id]))

            data = data.astype(np.uint8)

            self.images = np.zeros((num_points, s, s, num_channels), dtype=np.uint8)
            for point_id in tqdm(range(0, num_points), mininterval=10.0, desc='raw data processing'):
                s_id = point_id * s
                for n_id in range(0, num_channels):
                    for row_id in range(0, s):
                        for col_id in range(0, s):
                            global_id = s_id + row_id * s + col_id
                            self.images[point_id][row_id][col_id][n_id] = data[global_id][n_id]
            np.savez_compressed(fn_npz, images=self.images)

        self.transforms = transforms

    def __len__(self):
        return (len(self.norms))

    def __getitem__(self, i):
        data = self.images[i]

        if self.transforms:
            data = self.transforms(data)

        return (data, self.norms[i])


if __name__ == '__main__':
    device = get_device()

    path = 'E:/YandexDisk/Work/dl/datasets/floquet_lindbladian/two_spins'
    suffix = 'ampl(0.1000_0.1000_100)_freq(0.1000_0.1000_100)_phase(0.0000_0.0000_0)'

    # define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data = TwoSpinsDataset(path, suffix, transform)

    train_loader = DataLoader(data, batch_size=8, shuffle=False)
    a = iter(train_loader)
    single_batch = next(iter(train_loader))

    img = transforms.ToPILImage()(single_batch[0][0])
    img.show()

    img.save('out1.bmp')

    ololo = 1