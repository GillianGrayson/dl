from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from floqlind.routines.routines import get_device, train_val_dataset
from floqlind.routines.model import build_model_base, build_model_advance, params_to_learn
from floqlind.routines.dataset import load_df_data
import torch.optim as optim
import torch
import torch.nn as nn
from floqlind.routines.train import train_classification_model
from floqlind.routines.infrastructure import get_path
import numpy as np
import os
import re


if __name__ == '__main__':

    device = get_device()

    system_train = 'two_spins'
    size = 16

    path_train = get_path() + f'/dl/datasets/floquet_lindbladian/{system_train}'
    suffix_train = 'ampl(0.5000_2.0000_50)_freq(0.0500_0.2000_50)_phase(0.0000_0.0000_0)'

    label_type = 'class'

    prop_df, eval_df, reshuffle_eval_df = load_df_data(path_train, size, suffix_train, label_type)

    ololo = 1