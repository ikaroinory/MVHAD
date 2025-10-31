from typing import Literal

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset


class MVHADDataset(Dataset):
    def __init__(
        self,
        data: ndarray,
        slide_window: int,
        slide_stride: int,
        *,
        mode: Literal['train', 'test'],
        dtype=None
    ):
        x, y, label = self.__cut(data, slide_window, slide_stride, mode, dtype)

        self.x: Tensor = x
        self.y: Tensor = y
        self.label: Tensor = label

        self.num_nodes = self.x.shape[1]

    @staticmethod
    def __cut(
        data: ndarray,
        slide_window: int,
        slide_stride: int,
        mode: Literal['train', 'test'],
        dtype
    ) -> tuple[Tensor, Tensor, Tensor]:
        if mode == 'test':
            slide_stride = 1

        attack_labels = data[:, -1]
        data = data[:, :-1]

        x = []
        y = []
        labels = []

        for i in range(0, len(data) - slide_window, slide_stride):
            x.append(data[i:i + slide_window])
            y.append(data[i + slide_window])
            labels.append(attack_labels[i + slide_window])
        x = torch.tensor(np.array(x), dtype=dtype).permute(0, 2, 1)
        y = torch.tensor(np.array(y), dtype=dtype)
        labels = torch.tensor(np.array(labels), dtype=torch.long)

        return x, y, labels

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.label[idx]
