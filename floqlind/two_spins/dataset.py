import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class TwoSpinsDataset(Dataset):

    def __init__(self, path, suffix, transforms=None):

        s = 16
        num_channels = 3

        self.norms = np.loadtxt(f'{path}/norm_dl_1_{suffix}.txt')

        min_norm = np.amin(self.norms, initial=10, where=self.norms>0)
        self.norms = np.log10(self.norms + min_norm)
        self.norms = self.norms.astype(np.float32)
        num_points = self.norms.shape[0]

        fn_txt = f'{path}/props_dl_{suffix}.txt'
        fn_npz = f'{path}/props_dl_{suffix}.npz'

        if os.path.isfile(fn_npz):
            data = np.load(fn_npz)['data']
        else:
            data = np.zeros((num_points * s * s, 3), dtype=np.float64)
            f = open(fn_txt)
            row_id = 0
            for line in tqdm(f, mininterval=60.0, desc='reading raw file'):
                line_list = line.split('\t')
                line_list[-1] = line_list[-1].rstrip()
                data[row_id, 0] = np.float64(line_list[0])
                data[row_id, 1] = np.float64(line_list[1])
                data[row_id, 2] = np.float64(line_list[2])
                row_id += 1
            f.close()

            np.savez_compressed(fn_npz, data=data)

        data[:, 2] = np.angle(data[:, 0] + 1j * data[:, 1])

        for n_id in range(0, num_channels):
            data[:, n_id] = (255.0 * (data[:, n_id] - np.min(data[:, n_id])) / np.ptp(data[:, n_id]))

        data = data.astype(np.uint8)

        self.images = np.zeros((num_points, s, s, num_channels), dtype=np.uint8)
        for point_id in tqdm(range(0, num_points), mininterval=10.0, desc='raw dataset processing'):
            s_id = point_id * s
            for n_id in range(0, num_channels):
                for row_id in range(0, s):
                    for col_id in range(0, s):
                        global_id = s_id + row_id * s + col_id
                        self.images[point_id][row_id][col_id][n_id] = data[global_id][n_id]

        self.transforms = transforms

    def __len__(self):
        return (len(self.norms))

    def __getitem__(self, i):
        data = self.images[i]

        if self.transforms:
            data = self.transforms(data)

        return (data, self.norms[i])
