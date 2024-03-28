from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def print_trainable_parameters(model: nn.Module):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


class ResidualNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        return x + F.relu(self.conv2(F.relu(self.conv1(x))))


class ResidualBlock(nn.Module):
    def __init__(self, pre_hid_dim, hid_dim, n_layer):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels=pre_hid_dim, out_channels=hid_dim, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hid_dim, out_channels=hid_dim, kernel_size=3, padding="same"),
            nn.ReLU(),
        )
        self.res = nn.Sequential(
            *[ResidualNet(
                in_channels=hid_dim,
                out_channels=hid_dim,
                kernel_size=3,
                padding="same",
            ) for _ in range(n_layer - 1)]
        )

    def forward(self, x):
        x = self.down(x)
        x = self.res(x)
        return x


class ImageClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        hid_dim = config.hidden_dimension
        n_layers = config.num_layers
        assert sum(n < 2 for n in n_layers) == 0, "all num_layers should be equal or larger than 2"

        self.input_conv = nn.Conv2d(in_channels=1, out_channels=hid_dim, kernel_size=3, padding="same")
        self.res_blocks = nn.Sequential(
            ResidualBlock(hid_dim, hid_dim, n_layers[0]),
            *[ResidualBlock(hid_dim << (i - 1), hid_dim << i, n_layers[i]) for i in range(1, len(n_layers))],
        )
        self.global_average_pooling = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(hid_dim << (len(n_layers) - 1), 10)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        x = image.to(self.input_conv.weight.device)
        x = self.input_conv(x)
        x = self.res_blocks(x)
        x = self.global_average_pooling(x).view(*x.shape[:2])
        x = self.projection(x)
        label = label.to(x.device)
        return self.cross_entropy_loss(x.view(-1, x.shape[-1]), label.view(-1))

    def inference(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        x = image.to(self.input_conv.weight.device)
        x = self.input_conv(x)
        x = self.res_blocks(x)
        x = self.global_average_pooling(x).view(*x.shape[:2])
        x = self.projection(x)
        return torch.argmax(x, dim=-1)


Models = {
    "image_classifier": ImageClassifier,
}
