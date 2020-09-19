from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from floqlind.routines import get_device, train_val_dataset
from floqlind.model import initialize_model, params_to_learn
from floqlind.two_spins.dataset import TwoSpinsDataset
import torch.optim as optim
import torch
import torch.nn as nn
from floqlind.train import train_model
import numpy as np
import os


if __name__ == '__main__':

    device = get_device()

    path = 'E:/YandexDisk/Work/dl/datasets/floquet_lindbladian/two_spins'
    suffix = 'ampl(0.1000_0.1000_100)_freq(0.1000_0.1000_100)_phase(0.0000_0.0000_0)'

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "vgg"
    model_dir = f'{path}/{model_name}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Number of classes in the dataset
    num_classes = 1

    # Batch size for training (change depending on how much memory you have)
    batch_size = 32

    # Number of epochs to train for
    num_epochs = 100

    # Flag for feature extracting. When False, we finetune the whole model,
    # when True we only update the reshaped layer params
    feature_extract = False

    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    print(model)

    # define transforms
    transforms_regular = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = TwoSpinsDataset(path, suffix, transforms_regular)
    datasets_dict = train_val_dataset(dataset, 0.20, seed=1337)

    dataloaders_dict = {
        x: torch.utils.data.DataLoader(datasets_dict[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']
    }

    # Send the model to GPU
    model = model.to(device)

    params_to_learn(model, feature_extract)

    #optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    criterion = nn.MSELoss()

    model, train_loss_history, val_loss_history = train_model(device,
                                                              model,
                                                              dataloaders_dict,
                                                              criterion, optimizer,
                                                              num_epochs=num_epochs,
                                                              is_inception=(model_name == "inception")
                                                              )

    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss_history[-1],
        'val_loss': val_loss_history[-1],
    }, f'{model_dir}/checkpoint_{num_epochs}.pth')

    np.savetxt(f'{path}/{model_name}/train_loss_{num_epochs}.txt', np.asarray(train_loss_history), fmt='%0.16e')
    np.savetxt(f'{path}/{model_name}/val_loss_{num_epochs}.txt', np.asarray(val_loss_history), fmt='%0.16e')
