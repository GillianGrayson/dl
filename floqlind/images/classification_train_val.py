from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from floqlind.routines.routines import get_device, train_val_dataset
from floqlind.routines.model import build_model_base, build_model_advance, params_to_learn
from floqlind.routines.dataset import FloqLindDataset
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

    feature_type = 'prop'
    transforms_type = 'regular'
    label_type = 'class'

    # Models to choose from [resnet, vgg, densenet, inception, resnet50_2D]
    model_name = "resnet50"
    build_type = 'base'
    use_pretrained = False

    if use_pretrained:
        model_dir = f'{path_train}/classification/{model_name}_{feature_type}_{transforms_type}_{label_type}_{suffix_train}'
    else:
        model_dir = f'{path_train}/classification/{model_name}_scratch_{feature_type}_{transforms_type}_{label_type}_{suffix_train}'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    num_classes = 2
    feature_extract = False

    batch_size = 16
    num_epochs = 100

    is_continue = True

    if build_type == 'base':
        model, input_size = build_model_base(model_name, num_classes, feature_extract=feature_extract, use_pretrained=use_pretrained)
    else:
        model, input_size = build_model_advance(model_name, num_classes, feature_extract=feature_extract, use_pretrained=use_pretrained)
    # Send the model to GPU
    model = model.to(device)
    # optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    epoch_last = 0
    best_acc = 0
    if is_continue:
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
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            train_loss_history = checkpoint['train_loss']
            train_acc_history = checkpoint['train_acc']
            val_loss_history = checkpoint['val_loss']
            val_acc_history = checkpoint['val_acc']
            best_acc = checkpoint['best_acc']
    print(model)

    # define transforms
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

    dataset = FloqLindDataset(path_train, size, suffix_train, feature_type, label_type, transforms_regular)

    datasets_dict = train_val_dataset(dataset, 0.20, seed=1337)

    dataloaders_dict = {
        x: torch.utils.data.DataLoader(datasets_dict[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']
    }

    params_to_learn(model, feature_extract)

    criterion = nn.CrossEntropyLoss()

    model, bets_acc, train_loss_history_curr, train_acc_history_curr, val_loss_history_curr, val_acc_history_curr = train_classification_model(
        device,
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        best_acc,
        num_epochs=num_epochs,
        is_inception=(model_name == "inception")
        )

    train_loss_history += train_loss_history_curr
    train_acc_history += train_acc_history_curr
    val_loss_history += val_loss_history_curr
    val_acc_history += val_acc_history_curr

    torch.save({
        'epoch': num_epochs + epoch_last,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss_history,
        'train_acc': train_acc_history,
        'val_loss': val_loss_history,
        'val_acc': val_acc_history,
        'best_acc': best_acc
    }, f'{model_dir}/checkpoint_{num_epochs + epoch_last}.pth')

    np.savetxt(f'{model_dir}/train_loss_{num_epochs + epoch_last}.txt', np.asarray(train_loss_history), fmt='%0.16e')
    np.savetxt(f'{model_dir}/train_acc_{num_epochs + epoch_last}.txt', np.asarray(train_acc_history), fmt='%0.16e')
    np.savetxt(f'{model_dir}/val_loss_{num_epochs + epoch_last}.txt', np.asarray(val_loss_history), fmt='%0.16e')
    np.savetxt(f'{model_dir}/val_acc_{num_epochs + epoch_last}.txt', np.asarray(val_acc_history), fmt='%0.16e')
