import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sconf import Config
from torch.utils.data import Dataset, DataLoader


PATH_DICT = {
    "mnist": {
        "train": {
            "image": "train-images-idx3-ubyte",
            "label": "train-labels-idx1-ubyte",
        },
        "test": {
            "image": "t10k-images-idx3-ubyte",
            "label": "t10k-labels-idx1-ubyte",
        },
    },
}


def name_to_path(
        dataset_name: str,
        type: str = "train",
        datatype: str = "image"
    ):
    return os.path.join("data", dataset_name, PATH_DICT[dataset_name][type][datatype])


class DIGITSDataset(Dataset):
    def __init__(
        self,
        config: Config,
        dataset_type: str,
    ):
        super().__init__()
        data_path = name_to_path(config.dataset_name, type=dataset_type, datatype="image")
        with open(data_path, 'rb') as image_file:
            image_file.read(16)
            buffer = image_file.read()
        images = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        images = images.reshape(-1, 1, 28, 28)

        data_path = name_to_path(config.dataset_name, type=dataset_type, datatype="label")
        with open(data_path, 'rb') as label_file:
            label_file.read(8)
            buffer = label_file.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)

        self.data = {}
        self.data["image"] = torch.tensor(images / 255, dtype=torch.float)
        self.data["label"] = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data["label"])

    def __getitem__(self, index):
        """
        Returns:
            `dict`:
                - `input_ids` (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
                - `attention_mask` (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
                - `label` (`torch.LongTensor` of shape `(batch_size,)`)
        """
        return {k: v[index] for k, v in self.data.items()}


Datasets = {
    "handwritten_digits": DIGITSDataset,
}
