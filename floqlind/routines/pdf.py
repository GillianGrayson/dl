import numpy as np
import math


class PDF:

    def __init__(
            self,
            x_bin_s,
            x_bin_f,
            x_num_bins,
            y_bin_s,
            y_bin_f,
            y_num_bins
    ):
        self.x_bin_s = x_bin_s
        self.x_bin_f = x_bin_f
        self.x_num_bins = x_num_bins
        self.y_bin_s = y_bin_s
        self.y_bin_f = y_bin_f
        self.y_num_bins = y_num_bins

        self.x_bin_shift = (x_bin_f - x_bin_s) / x_num_bins
        self.y_bin_shift = (y_bin_f - y_bin_s) / y_num_bins

        self.x_bin_centers = np.linspace(
            self.x_bin_s + 0.5 * self.x_bin_shift,
            self.x_bin_f - 0.5 * self.x_bin_shift,
            self.x_num_bins
        )

        self.y_bin_centers = np.linspace(
            self.y_bin_s + 0.5 * self.y_bin_shift,
            self.y_bin_f - 0.5 * self.y_bin_shift,
            self.y_num_bins
        )

        self.pdf = np.zeros((self.x_num_bins, self.y_num_bins), dtype=np.float64)
        self.inc_count = 0
        self.non_inc_count = 0

    def update(self, data):
        for d_id in range(data.shape[0]):
            x = data[d_id][0]
            y = data[d_id][1]

            if (x >= self.x_bin_s) and (x <= self.x_bin_f) and (y >= self.y_bin_s) and (y <= self.y_bin_f):
                x_id = math.floor((x - self.x_bin_s) / (self.x_bin_shift + 1e-10))
                y_id = math.floor((y - self.y_bin_s) / (self.y_bin_shift + 1e-10))

                self.pdf[x_id][y_id] += 1
                self.inc_count += 1
            else:
                self.non_inc_count += 1

    def release(self):
        self.pdf /= (self.inc_count * self.x_bin_shift * self.y_bin_shift)
        self.norm = np.sum(self.pdf) * self.x_bin_shift * self.y_bin_shift

    def to_int(self):
        self.pdf = (255.0 * (self.pdf - np.min(self.pdf)) / np.ptp(self.pdf))

