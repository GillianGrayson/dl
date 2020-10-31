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
from tqdm import tqdm


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

    suffix_train = 'ampl(0.5000_2.0000_50)_freq(0.0500_0.2000_50)_phase(0.0000_0.0000_0)'
    suffix_test = 'ampl(0.2000_0.2000_500)_freq(0.0200_0.0200_500)_phase(0.0000_0.0000_0)'

    # Models to choose from [resnet, resnet50_2D, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet"
    use_pretrained = False

    feature_type = 'prop'
    transforms_type = 'regular'
    label_type = 'class'

    if use_pretrained:
        model_dir = f'{path_train}/classification/{model_name}_{feature_type}_{transforms_type}_{label_type}_{suffix_train}'
    else:
        model_dir = f'{path_train}/classification/{model_name}_scratch_{feature_type}_{transforms_type}_{label_type}_{suffix_train}'

    num_classes = 2
    feature_extract = False

    batch_size = 1

    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    model = model.to(device)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    epoch_last = 0
    best_acc = 0
    epoches = []
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
        train_acc_history = checkpoint['train_acc']
        val_loss_history = checkpoint['val_loss']
        val_acc_history = checkpoint['val_acc']
        best_acc = checkpoint['best_acc']
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

    criterion = nn.CrossEntropyLoss()

    model.eval()

    current_loss = 0.0
    running_corrects = 0
    outputs_all = []
    losses_all = []
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.long()
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            outputs_all.append(preds.item())
            losses_all.append(loss.item())

        # statistics
        current_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = current_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    print(f'Test loss: {epoch_loss}')
    print(f'Test acc: {epoch_acc}')

    save_dir = f'{model_dir}/test/{system_test}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savetxt(f'{save_dir}/norms_predicted_{last_epoch}_{suffix_test}.txt', np.asarray(outputs_all), fmt='%0.8e')
    np.savetxt(f'{save_dir}/loss_{last_epoch}_{suffix_test}.txt', np.asarray(losses_all), fmt='%0.8e')

