from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNGRUTransformerActor(nn.Module):
    """Actor with 2D CNN over (time × features), GRU over time, Transformer over assets."""

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        window: int,
        action_dim: int,
        cnn_channels: int = 32,
        gru_hidden: int = 32,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        mlp_hidden: int = 64,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.n_features = n_features
        self.window = window
        self.action_dim = action_dim

        self.cnn2d = nn.Conv2d(
            in_channels=1,
            out_channels=cnn_channels,
            kernel_size=(3, 3),
            padding=1,
        )

        self.gru_input_size = cnn_channels * n_features
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=gru_hidden,
            batch_first=True,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gru_hidden,
            nhead=n_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )

        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden * n_assets + n_assets, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(mlp_hidden, action_dim)
        self.log_std_head = nn.Linear(mlp_hidden, action_dim)

        self.LOG_STD_MIN = -5.0
        self.LOG_STD_MAX = 1.0

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = obs.shape[0]
        feat_size = self.n_assets * self.window * self.n_features

        feat_flat = obs[:, :feat_size]
        prev_w = obs[:, feat_size:]  # [B, n_assets]

        feats = feat_flat.view(batch_size, self.n_assets, self.window, self.n_features)

        x = feats.view(batch_size * self.n_assets, self.window, self.n_features)
        x = x.unsqueeze(1)  # [B*A, 1, T, F]

        x = F.relu(self.cnn2d(x))  # [B*A, C, T, F]

        x = x.permute(0, 2, 1, 3).contiguous()  # [B*A, T, C, F]
        x = x.view(batch_size * self.n_assets, self.window, -1)  # [B*A, T, C*F]

        _, h_n = self.gru(x)
        h_n = h_n.squeeze(0)  # [B*A, H]

        asset_emb = h_n.view(batch_size, self.n_assets, -1)

        asset_emb = self.transformer(asset_emb)  # [B, A, H]

        flat_emb = asset_emb.reshape(batch_size, -1)  # [B, A*H]

        x = torch.cat([flat_emb, prev_w], dim=-1)  # [B, A*H + A]
        x = self.mlp(x)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
