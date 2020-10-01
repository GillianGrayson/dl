import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from numpy import linalg as LA
from floqlind.routines.pdf import PDF


def norm_processing(norms, label_type):
    if label_type == 'log':
        min_norm = np.amin(norms, initial=10, where=norms > 0)
        norms = np.log10(norms + min_norm)
    elif label_type == 'log_with_add':
        norms = np.log10(norms + 1e-16)
    elif label_type == 'class':
        tmp = np.zeros(len(norms), dtype=np.int)
        num_zeros = 0
        for x_id, x in enumerate(norms):
            if x > 0:
                tmp[x_id] = 1
            else:
                tmp[x_id] = 0
                num_zeros += 1
        norms = tmp
    return norms

def load_features(size, num_subj, path, suffix):

    fn_txt = f'{path}/props_dl_{suffix}.txt'
    fn_npz = f'{path}/props_dl_3d_{suffix}.npz'

    if os.path.isfile(fn_npz):
        data = np.load(fn_npz)['data']
    else:
        data = np.zeros((num_subj * size * size, 3), dtype=np.float64)
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

    return data


class FloqLindDataset(Dataset):

    def __init__(self, path, size, suffix, features_type, label_type, transforms):

        self.transforms = transforms

        s = size
        num_channels = 3

        self.norms = np.loadtxt(f'{path}/norm_dl_1_{suffix}.txt')
        self.norms = self.norms.astype(np.float32)
        self.norms = norm_processing(self.norms, label_type)

        num_subj = self.norms.shape[0]

        data = load_features(size, num_subj, path, suffix)


        fn_npz = f'{path}/images_{features_type}_{suffix}.npz'
        if os.path.isfile(fn_npz):
            self.images = np.load(fn_npz)['images']
        else:

            if features_type == 'prop':

                data[:, 2] = np.angle(data[:, 0] + 1j * data[:, 1])

                for n_id in range(0, num_channels):
                    data[:, n_id] = (255.0 * (data[:, n_id] - np.min(data[:, n_id])) / np.ptp(data[:, n_id]))

                data = data.astype(np.uint8)

                self.images = np.zeros((num_subj, s, s, num_channels), dtype=np.uint8)
                for point_id in tqdm(range(0, num_subj), mininterval=10.0, desc='raw dataset processing'):
                    start_id = point_id * s * s
                    for n_id in range(0, num_channels):
                        for row_id in range(0, s):
                            for col_id in range(0, s):
                                global_id = start_id + row_id * s + col_id
                                self.images[point_id][row_id][col_id][n_id] = data[global_id][n_id]

            elif features_type == 'eval':
                pdf_size = 224

                x_bin_s = -1.05
                x_bin_f = 1.05

                y_bin_s = -1.05
                y_bin_f = 1.05

                x_bin_shift = (x_bin_f - x_bin_s) / pdf_size
                y_bin_shift = (y_bin_f - y_bin_s) / pdf_size

                x_bin_centers = np.linspace(
                    x_bin_s + 0.5 * x_bin_shift,
                    x_bin_f - 0.5 * x_bin_shift,
                    pdf_size
                )
                y_bin_centers = np.linspace(
                    y_bin_s + 0.5 * y_bin_shift,
                    y_bin_f - 0.5 * y_bin_shift,
                    pdf_size
                )

                cross_x = np.vstack([x_bin_centers, np.zeros(len(x_bin_centers), dtype=np.float64)]).T
                cross_y = np.vstack([np.zeros(len(y_bin_centers), dtype=np.float64), y_bin_centers]).T
                cross_data = np.concatenate((cross_x, cross_y), axis=0)
                cross_pdf = PDF(x_bin_s, x_bin_f, pdf_size, y_bin_s, y_bin_f, pdf_size)
                cross_pdf.update(cross_data)
                cross_pdf.release()
                cross_pdf.to_int()

                phase = np.linspace(0.0, 2.0 * np.pi, 10000)
                circle_data = np.vstack([np.cos(phase), np.sin(phase)]).T
                circle_pdf = PDF(x_bin_s, x_bin_f, pdf_size, y_bin_s, y_bin_f, pdf_size)
                circle_pdf.update(circle_data)
                circle_pdf.release()
                circle_pdf.to_int()

                self.images = np.zeros((num_subj, pdf_size, pdf_size, num_channels), dtype=np.uint8)

                for point_id in tqdm(range(0, num_subj), mininterval=10.0, desc='raw dataset processing'):
                    start_id = point_id * s * s

                    prop_mtx = np.zeros((s, s), dtype=complex)
                    for row_id in range(0, s):
                        for col_id in range(0, s):
                            global_id = start_id + row_id * s + col_id
                            prop_mtx[row_id][col_id] = data[global_id][0] + 1j * data[global_id][1]

                    evals = LA.eigvals(prop_mtx)
                    evals_data = np.vstack((np.real(evals), np.imag(evals))).T
                    evals_pdf = PDF(x_bin_s, x_bin_f, pdf_size, y_bin_s, y_bin_f, pdf_size)
                    evals_pdf.update(evals_data)
                    evals_pdf.release()
                    evals_pdf.to_int()

                    for row_id in range(0, pdf_size):
                        for col_id in range(0, pdf_size):
                            self.images[point_id][row_id][col_id][0] = evals_pdf.pdf[row_id][col_id]
                            self.images[point_id][row_id][col_id][1] = cross_pdf.pdf[row_id][col_id]
                            self.images[point_id][row_id][col_id][2] = circle_pdf.pdf[row_id][col_id]
            else:
                raise ValueError(f'Unsupported feature_type: {features_type}')

            np.savez_compressed(fn_npz, images=self.images)

    def __len__(self):
        return (len(self.norms))

    def __getitem__(self, i):
        data = self.images[i]

        if self.transforms:
            data = self.transforms(data)

        return (data, self.norms[i])
