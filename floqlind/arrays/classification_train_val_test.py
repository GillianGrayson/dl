from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from floqlind.routines.routines import get_device, train_val_dataset
from floqlind.routines.model import build_model_base, build_model_advance, params_to_learn
from floqlind.routines.dataset import load_df_data, scale_df_data
from floqlind.routines.classification import *
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

    system_name = 'two_spins'
    size = 16

    path = get_path() + f'/dl/datasets/floquet_lindbladian/{system_name}'
    suffix_train = 'ampl(0.5000_2.0000_50)_freq(0.0500_0.2000_50)_phase(0.0000_0.0000_0)'
    suffix_test = 'ampl(0.2000_0.2000_500)_freq(0.0200_0.0200_500)_phase(0.0000_0.0000_0)'

    label_type = 'class'

    train_prop_df, train_eval_df, train_reshuffle_eval_df, train_prop_scale, train_eval_scale, train_reshuffle_eval_scale = load_df_data(path, size, suffix_train, label_type, norm=True)
    test_prop_df, test_eval_df, test_reshuffle_eval_df = scale_df_data(path, size, suffix_test, label_type,
                                                                       train_prop_scale, train_eval_scale,
                                                                       train_reshuffle_eval_scale)

    # models = specify_models()
    # best_models_reshuffle = auto_train_binary_classifier(train_reshuffle_eval_df, 'label', models)
    # test_binary_classifier(test_reshuffle_eval_df, 'label', best_models_reshuffle)
    #
    # models = specify_models()
    # best_models_eval = auto_train_binary_classifier(train_eval_df, 'label', models)
    # test_binary_classifier(test_eval_df, 'label', best_models_eval)

    models = specify_models()
    best_models_prop = auto_train_binary_classifier(train_prop_df, 'label', models)
    test_binary_classifier(test_prop_df, 'label', best_models_prop)



