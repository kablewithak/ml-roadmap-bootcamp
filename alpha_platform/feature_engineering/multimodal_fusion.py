"""
Multi-modal fusion network for combining alternative data sources.

Combines satellite imagery, NLP features, and structured data into
unified representations for alpha generation.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from alpha_platform.utils.config import get_config
from alpha_platform.utils.logger import get_logger

logger = get_logger(__name__)


class MultiModalFusionNetwork(nn.Module):
    """
    Multi-modal fusion network using cross-modal transformers.

    Combines:
    - Visual features from satellite imagery
    - Text embeddings from NLP
    - Structured numerical features
    - Temporal sequences
    """

    def __init__(
        self,
        image_feature_dim: int = 512,
        text_feature_dim: int = 768,
        numerical_feature_dim: int = 64,
        hidden_dim: int = 512,
        num_attention_heads: int = 8,
        num_transformer_layers: int = 4,
        dropout: float = 0.1,
        output_dim: int = 256,
    ):
        """
        Initialize multi-modal fusion network.

        Args:
            image_feature_dim: Dimension of image features
            text_feature_dim: Dimension of text embeddings
            numerical_feature_dim: Dimension of numerical features
            hidden_dim: Hidden dimension for fusion
            num_attention_heads: Number of attention heads
            num_transformer_layers: Number of transformer layers
            dropout: Dropout rate
            output_dim: Output feature dimension
        """
        super().__init__()

        self.config = get_config()
        self.hidden_dim = hidden_dim

        # Projection layers to common dimension
        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.numerical_projection = nn.Sequential(
            nn.Linear(numerical_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Modality embeddings (learnable)
        self.image_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.text_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.numerical_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Cross-modal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1), nn.Softmax(dim=1)
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        logger.info(
            f"Initialized MultiModalFusionNetwork with "
            f"{num_transformer_layers} layers, {num_attention_heads} heads"
        )

    def forward(
        self,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        numerical_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through fusion network.

        Args:
            image_features: [batch, image_dim] or None
            text_features: [batch, text_dim] or None
            numerical_features: [batch, numerical_dim] or None
            mask: Optional attention mask

        Returns:
            fused_features: [batch, output_dim]
            attention_weights: [batch, num_modalities]
        """
        batch_size = (
            image_features.shape[0]
            if image_features is not None
            else text_features.shape[0]
            if text_features is not None
            else numerical_features.shape[0]
        )

        modality_features = []

        # Project each modality
        if image_features is not None:
            img_proj = self.image_projection(image_features)
            img_proj = img_proj.unsqueeze(1) + self.image_embedding.expand(
                batch_size, -1, -1
            )
            modality_features.append(img_proj)

        if text_features is not None:
            text_proj = self.text_projection(text_features)
            text_proj = text_proj.unsqueeze(1) + self.text_embedding.expand(
                batch_size, -1, -1
            )
            modality_features.append(text_proj)

        if numerical_features is not None:
            num_proj = self.numerical_projection(numerical_features)
            num_proj = num_proj.unsqueeze(1) + self.numerical_embedding.expand(
                batch_size, -1, -1
            )
            modality_features.append(num_proj)

        # Concatenate modalities
        if not modality_features:
            raise ValueError("At least one modality must be provided")

        # [batch, num_modalities, hidden_dim]
        combined_features = torch.cat(modality_features, dim=1)

        # Cross-modal transformer
        # [batch, num_modalities, hidden_dim]
        fused_features = self.transformer(combined_features, src_key_padding_mask=mask)

        # Attention pooling across modalities
        attention_weights = self.attention_pool(fused_features)  # [batch, num_mod, 1]
        pooled_features = (fused_features * attention_weights).sum(
            dim=1
        )  # [batch, hidden_dim]

        # Output projection
        output = self.output_projection(pooled_features)  # [batch, output_dim]

        return output, attention_weights.squeeze(-1)

    def get_attention_weights(
        self,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        numerical_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Get attention weights for each modality.

        Returns:
            Dictionary with modality importance scores
        """
        with torch.no_grad():
            _, attention_weights = self.forward(
                image_features, text_features, numerical_features
            )

        # Convert to dictionary
        weights_dict = {}
        idx = 0

        if image_features is not None:
            weights_dict["image"] = float(attention_weights[0, idx].item())
            idx += 1

        if text_features is not None:
            weights_dict["text"] = float(attention_weights[0, idx].item())
            idx += 1

        if numerical_features is not None:
            weights_dict["numerical"] = float(attention_weights[0, idx].item())
            idx += 1

        return weights_dict


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism.

    Allows one modality to attend to another.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize cross-modal attention.

        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            query: Query features [batch, seq_len, dim]
            key: Key features [batch, seq_len, dim]
            value: Value features [batch, seq_len, dim]
            mask: Optional attention mask

        Returns:
            attended_features: [batch, seq_len, dim]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        # Multi-head attention
        attended, attention_weights = self.multihead_attn(
            query, key, value, attn_mask=mask, need_weights=True
        )

        # Residual connection and normalization
        output = self.norm(query + self.dropout(attended))

        return output, attention_weights


class TemporalConvolutionalNetwork(nn.Module):
    """
    Temporal Convolutional Network for time series features.

    Uses dilated causal convolutions for efficient temporal modeling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        """
        Initialize TCN.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of TCN layers
            kernel_size: Convolutional kernel size
            dropout: Dropout rate
        """
        super().__init__()

        self.input_projection = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout,
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.output_projection = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, features]

        Returns:
            Output tensor [batch, seq_len, output_dim]
        """
        # Transpose to [batch, features, seq_len] for Conv1d
        x = x.transpose(1, 2)

        # Input projection
        x = self.input_projection(x)

        # TCN layers
        x = self.tcn(x)

        # Output projection
        x = self.output_projection(x)

        # Transpose back to [batch, seq_len, output_dim]
        x = x.transpose(1, 2)

        return x


class TemporalBlock(nn.Module):
    """Single temporal block with dilated causal convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        """Initialize temporal block."""
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove padding from causal convolution."""

    def __init__(self, chomp_size: int):
        """Initialize chomper."""
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Remove padding."""
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x
