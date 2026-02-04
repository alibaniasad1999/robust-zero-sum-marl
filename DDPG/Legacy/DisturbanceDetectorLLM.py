import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Tuple, Dict

class TransformerDisturbanceDetector(nn.Module):
    """
    Transformer-based model that analyzes observation history to:
    1. Detect disturbance presence
    2. Output blending weight α for controller mixing
    
    α = 0 → Use robust controller fully
    α = 1 → Use optimal controller fully
    0 < α < 1 → Blend both controllers
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
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.device = device
        
        # Input embedding: project observation to d_model
        self.input_projection = nn.Linear(obs_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            self._create_positional_encoding(sequence_length, d_model),
            requires_grad=False
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output heads
        # Head 1: Disturbance detection (binary classification)
        self.disturbance_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Probability of disturbance
        )
        
        # Head 2: Blending weight α (continuous [0,1])
        self.blending_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # α ∈ [0, 1]
        )
        
        self.to(device)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
    
    def forward(self, obs_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_sequence: (batch_size, sequence_length, obs_dim)
        
        Returns:
            disturbance_prob: (batch_size,) - probability of disturbance
            blending_weight: (batch_size,) - α for controller blending
        """
        batch_size, seq_len, _ = obs_sequence.shape
        
        # Project observations to d_model
        x = self.input_projection(obs_sequence)  # (B, L, d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Pass through transformer
        encoded = self.transformer_encoder(x)  # (B, L, d_model)
        
        # Use the last token's representation for prediction
        final_hidden = encoded[:, -1, :]  # (B, d_model)
        
        # Predict disturbance probability
        disturbance_prob = self.disturbance_head(final_hidden).squeeze(-1)  # (B,)
        
        # Predict blending weight α
        blending_weight = self.blending_head(final_hidden).squeeze(-1)  # (B,)
        
        return disturbance_prob, blending_weight


class HistoryBuffer:
    """Buffer to maintain observation history for transformer input."""
    
    def __init__(self, obs_dim: int, max_length: int = 20, device: torch.device = torch.device("cpu")):
        self.obs_dim = obs_dim
        self.max_length = max_length
        self.device = device
        self.buffer = deque(maxlen=max_length)
        self.reset()
    
    def reset(self):
        """Reset buffer with zeros."""
        self.buffer.clear()
        for _ in range(self.max_length):
            self.buffer.append(np.zeros(self.obs_dim, dtype=np.float32))
    
    def add(self, obs: np.ndarray):
        """Add new observation to buffer."""
        self.buffer.append(obs.copy())
    
    def get_sequence(self) -> torch.Tensor:
        """Get current sequence as tensor."""
        sequence = np.array(list(self.buffer), dtype=np.float32)
        return torch.as_tensor(sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def get_batch_sequences(self, obs_sequences: np.ndarray) -> torch.Tensor:
        """
        Convert batch of observation sequences to tensor.
        Args:
            obs_sequences: (batch_size, sequence_length, obs_dim)
        """
        return torch.as_tensor(obs_sequences, dtype=torch.float32, device=self.device)


class DisturbanceDetectorTrainer:
    """Trainer for the transformer disturbance detector."""
    
    def __init__(
        self,
        model: TransformerDisturbanceDetector,
        learning_rate: float = 1e-4,
        device: torch.device = torch.device("cpu")
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss functions
        self.disturbance_criterion = nn.BCELoss()
        self.blending_criterion = nn.MSELoss()
    
    def compute_loss(
        self,
        obs_sequence: torch.Tensor,
        true_disturbance: torch.Tensor,
        true_blending: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss.
        
        Args:
            obs_sequence: (batch_size, sequence_length, obs_dim)
            true_disturbance: (batch_size,) - ground truth disturbance labels
            true_blending: (batch_size,) - ground truth optimal blending weights
        """
        pred_disturbance, pred_blending = self.model(obs_sequence)
        
        # Disturbance detection loss (binary cross-entropy)
        loss_disturbance = self.disturbance_criterion(pred_disturbance, true_disturbance)
        
        # Blending weight loss (MSE)
        loss_blending = self.blending_criterion(pred_blending, true_blending)
        
        # Combined loss with weighting
        total_loss = loss_disturbance + 2.0 * loss_blending
        
        info = {
            'loss_disturbance': loss_disturbance.item(),
            'loss_blending': loss_blending.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, info
    
    def train_step(
        self,
        obs_sequence: torch.Tensor,
        true_disturbance: torch.Tensor,
        true_blending: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss, info = self.compute_loss(obs_sequence, true_disturbance, true_blending)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return info
    
    def evaluate(
        self,
        obs_sequence: torch.Tensor,
        true_disturbance: torch.Tensor,
        true_blending: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluation step without gradient updates."""
        self.model.eval()
        with torch.no_grad():
            _, info = self.compute_loss(obs_sequence, true_disturbance, true_blending)
        return info


class AdaptiveControllerBlender:
    """
    Blends optimal and robust controllers based on transformer predictions.
    
    action = α * action_optimal + (1 - α) * action_robust
    where α ∈ [0, 1] is predicted by the transformer
    """
    
    def __init__(
        self,
        detector: TransformerDisturbanceDetector,
        history_buffer: HistoryBuffer,
        smoothing_window: int = 5
    ):
        self.detector = detector
        self.history_buffer = history_buffer
        self.smoothing_window = smoothing_window
        self.alpha_history = deque(maxlen=smoothing_window)
        
    def get_blended_action(
        self,
        obs: np.ndarray,
        action_optimal: np.ndarray,
        action_robust: np.ndarray,
        use_smoothing: bool = True
    ) -> Tuple[np.ndarray, float, float]:
        """
        Get blended action based on transformer prediction.
        
        Returns:
            blended_action: The final action to execute
            alpha: Blending weight used
            disturbance_prob: Predicted disturbance probability
        """
        # Add observation to history
        self.history_buffer.add(obs)
        
        # Get prediction from transformer
        self.detector.eval()
        with torch.no_grad():
            obs_sequence = self.history_buffer.get_sequence()
            disturbance_prob, alpha = self.detector(obs_sequence)
            
            alpha = alpha.item()
            disturbance_prob = disturbance_prob.item()
        
        # Optional: Smooth alpha over time for stability
        if use_smoothing:
            self.alpha_history.append(alpha)
            alpha = np.mean(self.alpha_history)
        
        # Blend actions: α * optimal + (1-α) * robust
        blended_action = alpha * action_optimal + (1 - alpha) * action_robust
        
        return blended_action, alpha, disturbance_prob
    
    def reset(self):
        """Reset history buffers."""
        self.history_buffer.reset()
        self.alpha_history.clear()
