from torchvision.transforms import transforms
from floqlind.routines.routines import get_device
from floqlind.routines.dataset import FloqLindDataset
from floqlind.routines.infrastructure import get_path
import os


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

    feature_type = 'eval'
    transforms_type = 'noNorm'
    label_type = 'log'

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

    dataset = FloqLindDataset(path, size, suffix, feature_type, label_type, transforms_regular)

    fig_path = f'{path}/figures/images/{feature_type}_{transforms_type}'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    target_indexes = [(10, 160), (4, 80), (42, 124), (190, 190)]
    for point in target_indexes:
        x = point[0] - 1
        y = point[1] - 1
        ind = x * num_points + y
        sample = dataset[ind]

        img = transforms.ToPILImage()(sample[0])

        fn = f'{fig_path}/x({point[0]})_y({point[1]})_num_points({num_points})_norm({sample[1]:0.4f}).bmp'

        img.save(fn)
