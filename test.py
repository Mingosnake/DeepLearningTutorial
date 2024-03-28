#!/usr/bin/env python
import argparse
import logging

import numpy as np
import torch
from pathlib import Path
from sconf import Config
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import Datasets
from utils.data_util import evaluate_accuracy, evaluate_mae
from utils.modeling import Models


DEVICE = "cuda:1"

"""
Example:
    $ ./test.py --result result/mnist_classification/YYMMDD_HHmmss
"""
def test(config, device):
    test_set = Datasets[config.data_domain](config, dataset_type="test")
    test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=False)

    model = Models[config.model](config)
    model.load_state_dict(torch.load(Path(config.save_path) / "model.pt"))
    model.to(device)
    model.eval()

    predictions = []
    labels = []
    with torch.no_grad():
        with tqdm(test_loader, desc="Test") as tqdm_loader:
            for i, batch in enumerate(tqdm_loader):
                prediction = model.inference(**batch)

                if i == 0:
                    predictions = prediction.cpu().numpy()
                    labels = batch["label"].numpy()
                else:
                    predictions = np.append(predictions, prediction.cpu().numpy(), axis=0)
                    labels = np.append(labels, batch["label"].numpy(), axis=0)

    return predictions, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=str, required=True)
    args, left_argv = parser.parse_known_args()

    config = Config(Path(args.result) / "config.yaml")
    config.argv_update(left_argv)

    logging.basicConfig(filename=Path(config.save_path) / "test.log", level=logging.INFO, format="%(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())

    predictions, labels = test(config, device=DEVICE)
    correct, total = evaluate_accuracy(predictions, labels)
    logging.info(f"Accuracy : {correct / total * 100:.2f} %")
