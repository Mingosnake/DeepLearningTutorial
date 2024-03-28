#!/usr/bin/env python
import argparse
import logging
import os
from datetime import datetime
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from pathlib import Path
from sconf import Config
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.dataset import Datasets
from utils.data_util import save_config_file, set_random_seed
from utils.modeling import Models, print_trainable_parameters


"""
Example:
    $ ./train.py --config config/mnist_classification.yaml
"""
def train(config):
    set_random_seed(2024)
    logging.basicConfig(filename=Path(config.save_path) / "train.log", level=logging.INFO, format="%(message)s")
    summary_writer = SummaryWriter(log_dir=config.save_path)

    def log(writer, tag, scalar_value, global_step = None, walltime = None, new_style = False, double_precision = False):
        writer.add_scalar(tag, scalar_value, global_step, walltime, new_style, double_precision)
        body = f"{tag}: {scalar_value:.3f}"
        prefix = f"Epoch {global_step + 1:03}| " if global_step is not None else ""
        logging.info(prefix + body)

    # data
    dataset = Datasets[config.data_domain](config, dataset_type="train")
    train_set, valid_set = random_split(dataset, [len(dataset) - len(dataset) // 5, len(dataset) // 5])
    train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=config.batch_size, shuffle=False)

    # model
    model = Models[config.model](config)
    if "load_exp_version" in config:
        model.load_state_dict(torch.load(
            Path(config.result_path) / config.exp_name / config.load_exp_version / "model.pt"
        ))
        print(f"load from `{Path(config.result_path) / config.exp_name / config.load_exp_version}`")
    model.to(config.device)
    print_trainable_parameters(model)

    # train
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    best_model = model.state_dict()
    best_valid_loss = float("inf")

    def one_epoch(data_loader, model, optimizer, epoch, epochs, train = True):
        epoch_loss = 0
        prefix = "Train" if train else "Valid"
        with tqdm(data_loader, desc=f"{prefix} Epoch {epoch + 1}/{epochs}") as tqdm_loader:
            for i, batch in enumerate(tqdm_loader):
                loss = model(**batch)
                if train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()

                if (i + 1) == len(data_loader):
                    epoch_loss = epoch_loss / len(data_loader)
                    tqdm_loader.set_postfix(loss=epoch_loss)
                else:
                    tqdm_loader.set_postfix(loss=loss.item())
        return epoch_loss

    for epoch in range(config.epochs):
        model.train()
        train_loss = one_epoch(train_loader, model, optimizer, epoch, config.epochs, train=True)
        with torch.no_grad():
            model.eval()
            valid_loss = one_epoch(valid_loader, model, optimizer, epoch, config.epochs, train=False)

        print(f"Epoch {(epoch + 1):03} | train loss: {train_loss:.3f} | valid loss: {valid_loss:.3f}")
        log(summary_writer, "loss/train", train_loss, epoch)
        log(summary_writer, "loss/valid", valid_loss, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model.state_dict()
            torch.save(best_model, Path(config.save_path) / "model.pt")
            print(f"Model saved to `{config.save_path}`")
            config.last_epoch = config.past_epochs + epoch + 1
            save_config_file(config, config.save_path, verbose=False)

    summary_writer.flush()
    summary_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)

    config.exp_name = os.path.basename(args.config).split(".")[0]
    config.exp_version = datetime.now().strftime("%Y%m%d_%H%M%S") if not args.exp_version else args.exp_version
    if "load_exp_version" in config:
        past_config = Config(Path(config.result_path) / config.exp_name / config.load_exp_version / "config.yaml")
        config.past_epochs = past_config.last_epoch
    else:
        config.past_epochs = 0

    config.save_path = Path(config.result_path) / config.exp_name / config.exp_version
    save_config_file(config, config.save_path)

    train(config)
