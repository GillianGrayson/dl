from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from floqlind.routines import get_device, train_val_dataset
from floqlind.model import initialize_model, params_to_learn
from floqlind.two_spins.dataset import TwoSpinsDataset, TwoSpinsDataset2D
import torch.optim as optim
import torch
import torch.nn as nn
from floqlind.train import train_model
from floqlind.infrastructure import get_path
import numpy as np
import os
import re


if __name__ == '__main__':

    device = get_device()

    path = get_path() + '/dl/datasets/floquet_lindbladian/two_spins'
    #suffix = 'ampl(0.2000_0.2000_500)_freq(0.0200_0.0200_500)_phase(0.0000_0.0000_0)'
    suffix = 'ampl(0.0500_0.1000_100)_freq(0.0500_0.1000_100)_phase(0.0000_0.0000_0)'
    suffix_model = 'ampl(0.1000_0.1000_100)_freq(0.1000_0.1000_100)_phase(0.0000_0.0000_0)'

    # Models to choose from [resnet, resnet50_2D, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet"
    function = 'log'
    is_2D = False

    model_dir = f'{path}/{model_name}_{function}_{suffix_model}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    num_classes = 1
    feature_extract = False

    batch_size = 1


    if is_2D:
        if not model_name.endswith('2D'):
            raise ValueError('Wrong model for 2D')

    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # Send the model to GPU
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

    # define transforms
    if is_2D:
        transforms_regular = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5], [0.25, 0.25])
        ])
    else:
        transforms_regular = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    if is_2D:
        dataset = TwoSpinsDataset2D(path, suffix, function, transforms_regular)
    else:
        dataset = TwoSpinsDataset(path, suffix, function, transforms_regular)

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

    np.savetxt(f'{model_dir}/norms_predicted_{last_epoch}_{suffix}.txt', np.asarray(outputs_all), fmt='%0.8e')
    np.savetxt(f'{model_dir}/loss_{last_epoch}_{suffix}.txt', np.asarray(losses_all), fmt='%0.8e')

