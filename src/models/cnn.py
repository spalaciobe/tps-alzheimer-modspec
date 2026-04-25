"""ModSpecCNN — réplica fiel de la arquitectura de Lopes et al. 2023.

Tabla 3 del paper (hiperparámetros tras tuning):
    - Input tensor: 45 × 45 × 20 (en su dataset; aquí 19 canales por ds004504)
    - 2 capas convolucionales, kernel 3×3, ReLU
    - 3 capas fully-connected, LeakyReLU (incluye la final de clasificación)
    - Dropout 85% (en bloques conv y FC)
    - L2 regularization 1e-2 → weight_decay del optimizer
    - Optimizer Nadam, lr=1e-4
    - Batch 4, 50 épocas
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModSpecCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 19,
        conv_filters: tuple[int, int] = (32, 64),
        kernel_size: int = 3,
        conv_dropout: float = 0.85,
        fc_units: tuple[int, int] = (128, 64),
        fc_dropout: float = 0.85,
        fc_negative_slope: float = 0.1,
        num_classes: int = 2,
        input_hw: tuple[int, int] = (45, 45),
    ):
        super().__init__()
        pad = kernel_size // 2

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, conv_filters[0], kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.Dropout2d(conv_dropout),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.Dropout2d(conv_dropout),
            nn.MaxPool2d(2),
        )

        # Determinar shape post-conv con un dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_hw)
            flat_dim = self.block2(self.block1(dummy)).flatten(1).shape[1]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_units[0]),
            nn.LeakyReLU(fc_negative_slope, inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_units[0], fc_units[1]),
            nn.LeakyReLU(fc_negative_slope, inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_units[1], num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Capa target para Grad-CAM: última conv del Block 2."""
        return self.block2[0]
