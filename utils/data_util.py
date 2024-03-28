import os
import random
from typing import Any, List, Tuple, Optional, Union

import numpy as np
import torch
from pathlib import Path
from sconf import Config


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def save_config_file(config: Config, path: Union[str, os.PathLike], verbose: Optional[bool] = True):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    if verbose:
        print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")


def evaluate_accuracy(
    predictions: Union[np.ndarray, List, torch.Tensor],
    labels: Union[np.ndarray, List, torch.Tensor],
) -> Tuple[int, int]:
    if isinstance(predictions, List):
        predictions = np.array(predictions)
    elif isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(labels, List):
        labels = np.array(labels)
    elif isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    assert isinstance(predictions, np.ndarray), "The type of predictions should be List, numpy.ndarray, or torch.Tensor"
    assert isinstance(predictions, np.ndarray), "The type of labels should be List, numpy.ndarray, or torch.Tensor"
    assert predictions.shape == labels.shape, "shape of predictions and labels should be same"
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)
    correct = np.sum(predictions == labels)
    total = len(predictions)

    return correct, total


def evaluate_mae(
    predictions: Union[np.ndarray, List, torch.Tensor],
    labels: Union[np.ndarray, List, torch.Tensor],
) -> np.ndarray:  # shape = [label_dim]
    if isinstance(predictions, List):
        predictions = np.array(predictions)
    elif isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(labels, List):
        labels = np.array(labels)
    elif isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    assert isinstance(predictions, np.ndarray), "The type of predictions should be List, numpy.ndarray, or torch.Tensor"
    assert isinstance(labels, np.ndarray), "The type of labels should be List, numpy.ndarray, or torch.Tensor"
    assert predictions.shape == labels.shape, "shape of predictions and labels should be same"
    predictions = predictions.reshape(-1, predictions.shape[-1])
    labels = labels.reshape(-1, labels.shape[-1])
    mae = np.mean(np.abs(predictions - labels), axis=0)

    return mae
