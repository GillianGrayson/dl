from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from floqlind.routines.routines import get_device
from floqlind.routines.model import initialize_model
from floqlind.routines.dataset import FloqLindDataset, FloqLindDataset2D
import torch
import torch.nn as nn
from floqlind.routines.infrastructure import get_path
import numpy as np
import os
import re


if __name__ == '__main__':

    device = get_device()

    system = 'two_spins'
    if system in ['ospm', 'two_spins']:
        size = 16
    elif system in ['os']:
        size = 4
    else:
        raise ValueError('unsupported test system')
    input_size = 224

    path = get_path() + f'/dl/datasets/floquet_lindbladian/{system}'

    num_points = 200
    suffix = f'ampl(0.5000_0.5000_{num_points})_freq(0.0500_0.0500_{num_points})_phase(0.0000_0.0000_0)'


    function = 'log'

    transforms_regular = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    dataset = FloqLindDataset(path, size, suffix, function, transforms_regular)

    fig_path = f'{path}/props_as_figures'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    target_indexes = [(10, 160), (4, 80), (42, 124), (190, 190)]
    for point in target_indexes:
        x = point[0] - 1
        y = point[1] - 1
        ind = x * num_points + y
        sample = dataset[ind]

        img = transforms.ToPILImage()(sample[0])

        fn = f'{fig_path}/x({point[0]})_y({point[1]})_norm({sample[1]:0.4f}).bmp'

        img.save(fn)
