"""Transformer-based disturbance detector and adaptive controller blender."""

from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class TransformerDisturbanceDetector(nn.Module):
    """
    Lightweight transformer encoder that analyses an observation history
    and outputs:
        1. disturbance probability  p_t in [0, 1]
        2. blending weight          alpha_t in [0, 1]
           alpha_t -> 1  means use optimal controller
           alpha_t -> 0  means use robust controller
    """

    def __init__(
        self,
        obs_dim: int,
        sequence_length: int = 20,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.sequence_length = sequence_length
        self.d_model = d_model

        self.input_projection = nn.Linear(obs_dim, d_model)

        self.positional_encoding = nn.Parameter(
            self._sinusoidal_pe(sequence_length, d_model), requires_grad=False
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.disturbance_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Magnitude head: estimates L2 norm of disturbance (δ̂_t)
        # No output activation — raw positive prediction, trained with MSE
        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Gating head: α_t = f_φ(p_t, δ̂_t)  — paper Eq (15)
        # Input is the 2-vector [disturbance_prob, magnitude_estimate]
        self.blending_head = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, L, D)

    # ------------------------------------------------------------------
    def _encode(self, obs_sequence: torch.Tensor) -> torch.Tensor:
        """Shared encoder pass. Returns last-token embedding (B, d_model)."""
        seq_len = obs_sequence.shape[1]
        x = self.input_projection(obs_sequence)
        x = x + self.positional_encoding[:, :seq_len, :]
        encoded = self.transformer_encoder(x)
        return encoded[:, -1, :]

    def forward(
        self, obs_sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_sequence: (B, L, obs_dim)
        Returns:
            disturbance_prob: (B,)
            blending_weight:  (B,)
        """
        final = self._encode(obs_sequence)
        p = self.disturbance_head(final).squeeze(-1)          # (B,)
        delta_hat = self.magnitude_head(final).squeeze(-1)    # (B,) raw magnitude estimate
        gate_input = torch.stack([p, delta_hat], dim=-1)      # (B, 2)
        alpha = self.blending_head(gate_input).squeeze(-1)    # (B,)
        return p, alpha

    def forward_all(
        self, obs_sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns all three outputs in a single encoder pass.
        Args:
            obs_sequence: (B, L, obs_dim)
        Returns:
            disturbance_prob:    (B,)
            magnitude_estimate:  (B,)  raw (no activation)
            blending_weight:     (B,)
        """
        final = self._encode(obs_sequence)
        p = self.disturbance_head(final).squeeze(-1)
        delta_hat = self.magnitude_head(final).squeeze(-1)
        gate_input = torch.stack([p, delta_hat], dim=-1)
        alpha = self.blending_head(gate_input).squeeze(-1)
        return p, delta_hat, alpha


# ======================================================================
class HistoryBuffer:
    """Fixed-length deque that stores recent observations for the detector."""

    def __init__(self, obs_dim: int, max_length: int = 20, device: torch.device = torch.device("cpu")):
        self.obs_dim = obs_dim
        self.max_length = max_length
        self.device = device
        self.buffer: deque[np.ndarray] = deque(maxlen=max_length)
        self.reset()

    def reset(self) -> None:
        self.buffer.clear()
        for _ in range(self.max_length):
            self.buffer.append(np.zeros(self.obs_dim, dtype=np.float32))

    def add(self, obs: np.ndarray) -> None:
        self.buffer.append(obs.copy())

    def get_sequence(self) -> torch.Tensor:
        """Return (1, L, obs_dim) tensor."""
        seq = np.array(list(self.buffer), dtype=np.float32)
        return torch.as_tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)


# ======================================================================
class DisturbanceDetectorTrainer:
    """Supervised trainer for TransformerDisturbanceDetector."""

    def __init__(
        self,
        model: TransformerDisturbanceDetector,
        learning_rate: float = 1e-4,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def train_step(
        self,
        obs_seq: torch.Tensor,
        true_dist: torch.Tensor,
        true_alpha: torch.Tensor,
        true_magnitude: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        pred_dist, pred_mag, pred_alpha = self.model.forward_all(obs_seq)
        loss_dist = self.bce(pred_dist, true_dist)
        loss_alpha = self.mse(pred_alpha, true_alpha)
        loss = loss_dist + loss_alpha
        if true_magnitude is not None:
            loss_mag = self.mse(pred_mag, true_magnitude)
            loss = loss + loss_mag
        else:
            loss_mag = None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        result = {
            "loss_disturbance": loss_dist.item(),
            "loss_blending": loss_alpha.item(),
            "total_loss": loss.item(),
        }
        if loss_mag is not None:
            result["loss_magnitude"] = loss_mag.item()
        return result

    @torch.no_grad()
    def evaluate(
        self,
        obs_seq: torch.Tensor,
        true_dist: torch.Tensor,
        true_alpha: torch.Tensor,
        true_magnitude: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        self.model.eval()
        pred_dist, pred_mag, pred_alpha = self.model.forward_all(obs_seq)
        loss_dist = self.bce(pred_dist, true_dist)
        loss_alpha = self.mse(pred_alpha, true_alpha)
        loss = loss_dist + loss_alpha
        if true_magnitude is not None:
            loss_mag = self.mse(pred_mag, true_magnitude)
            loss = loss + loss_mag
        else:
            loss_mag = None
        result = {
            "loss_disturbance": loss_dist.item(),
            "loss_blending": loss_alpha.item(),
            "total_loss": loss.item(),
        }
        if loss_mag is not None:
            result["loss_magnitude"] = loss_mag.item()
        return result


# ======================================================================
class AdaptiveControllerBlender:
    """
    Blends optimal and robust actions using the detector's alpha output.

        a_t = alpha * a_opt + (1 - alpha) * a_rob
    """

    def __init__(
        self,
        detector: TransformerDisturbanceDetector,
        history_buffer: HistoryBuffer,
        smoothing_window: int = 5,
    ):
        self.detector = detector
        self.history_buffer = history_buffer
        self.alpha_history: deque[float] = deque(maxlen=smoothing_window)

    def get_blended_action(
        self,
        obs: np.ndarray,
        action_optimal: np.ndarray,
        action_robust: np.ndarray,
        use_smoothing: bool = True,
    ) -> Tuple[np.ndarray, float, float]:
        self.history_buffer.add(obs)
        self.detector.eval()
        with torch.no_grad():
            obs_seq = self.history_buffer.get_sequence()
            dist_prob, alpha = self.detector(obs_seq)
            alpha_val = alpha.item()
            dist_val = dist_prob.item()

        if use_smoothing:
            self.alpha_history.append(alpha_val)
            alpha_val = float(np.mean(self.alpha_history))

        blended = alpha_val * action_optimal + (1.0 - alpha_val) * action_robust
        return blended, alpha_val, dist_val

    def reset(self) -> None:
        self.history_buffer.reset()
        self.alpha_history.clear()
