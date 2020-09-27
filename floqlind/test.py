from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from floqlind.routines.routines import get_device
from floqlind.routines.model import initialize_model
from floqlind.routines.dataset import FloqLindDataset
import torch
import torch.nn as nn
from floqlind.routines.infrastructure import get_path
import numpy as np
import os
import re


if __name__ == '__main__':

    device = get_device()

    system_train = 'two_spins'
    system_test = 'two_spins'
    if system_test in ['ospm', 'two_spins']:
        size = 16
    elif system_test in ['os']:
        size = 4
    else:
        raise ValueError('unsupported test system')

    path_train = get_path() + f'/dl/datasets/floquet_lindbladian/{system_train}'
    path_test = get_path() + f'/dl/datasets/floquet_lindbladian/{system_test}'

    suffix_train = 'ampl(0.5000_0.5000_200)_freq(0.0500_0.0500_200)_phase(0.0000_0.0000_0)'
    suffix_test = 'ampl(0.5000_0.5000_200)_freq(0.0500_0.0500_200)_phase(1.5708_0.0000_0)'

    # Models to choose from [resnet, resnet50_2D, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet"

    feature_type = 'eval'
    transforms_type = 'regular'
    label_type = 'log'

    model_dir = f'{path_train}/{model_name}_{feature_type}_{transforms_type}_{label_type}_{suffix_train}'

    num_classes = 1
    feature_extract = False

    batch_size = 1

    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    model = model.to(device)

    train_loss_history = []
    val_loss_history = []
    epoch_last = 0
    best_loss = np.inf
    epoches = []
    last_epoch = 0
    for file in os.listdir(model_dir):
        if re.match('checkpoint_[-+]?[0-9]+.pth', file):
            epoches = epoches + re.findall(r'\d+', file)
    if len(epoches) > 0:
        epoches = list(map(int, epoches))
        last_epoch = max(epoches)
        checkpoint = torch.load(f'{model_dir}/checkpoint_{last_epoch}.pth')
        epoch_last = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        train_loss_history = checkpoint['train_loss']
        val_loss_history = checkpoint['val_loss']
        best_loss = checkpoint['best_loss']

    print(model)

    if transforms_type == 'regular':
        transforms_regular = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif transforms_type == 'noNorm':
        transforms_regular = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError(f'Unsupported transforms_type: {transforms_type}')

    dataset = FloqLindDataset(path_test, size, suffix_test, feature_type, label_type, transforms_regular)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    criterion = nn.MSELoss()

    model.eval()

    current_loss = 0.0
    outputs_all = []
    losses_all = []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.view(-1, 1)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            outputs_all.append(outputs.item())
            losses_all.append(loss.item())

        # statistics
        current_loss += loss.item() * inputs.size(0)

    epoch_loss = current_loss / len(dataloader.dataset)
    print(f'Test loss: {epoch_loss}')

    save_dir = f'{model_dir}/test/{system_test}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savetxt(f'{save_dir}/norms_predicted_{last_epoch}_{suffix_test}.txt', np.asarray(outputs_all), fmt='%0.8e')
    np.savetxt(f'{save_dir}/loss_{last_epoch}_{suffix_test}.txt', np.asarray(losses_all), fmt='%0.8e')

