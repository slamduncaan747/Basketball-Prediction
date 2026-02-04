import torch
import torch.nn as nn
import math
from typing import Dict, Optional

class GameContextEncoding(nn.Module):
    """
    Encodes game state (score, time, momentum) into vectors.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Projections for continuous variables
        self.score_diff_proj = nn.Linear(1, d_model // 4)
        self.game_progress_proj = nn.Linear(1, d_model // 4)
        self.clock_proj = nn.Linear(1, d_model // 4)
        self.momentum_proj = nn.Linear(10, d_model // 4)
        
        # Embeddings for discrete variables
        self.period_embedding = nn.Embedding(6, d_model // 4)
        self.team_embedding = nn.Embedding(3, d_model // 4)
        
        # Combine everything
        self.combine = nn.Linear(d_model + d_model // 4, d_model)

    def forward(self, score_diff, game_progress, clock, period, team_indicator, momentum_features):
        # Unsqueeze adds a dimension to match Linear layer input requirements (Batch, Seq, 1)
        score_emb = self.score_diff_proj(score_diff.unsqueeze(-1))
        progress_emb = self.game_progress_proj(game_progress.unsqueeze(-1))
        clock_emb = self.clock_proj(clock.unsqueeze(-1))
        momentum_emb = self.momentum_proj(momentum_features)
        
        # Clamp values to prevent index out of bounds errors
        period_clamped = period.clamp(0, 5)
        period_emb = self.period_embedding(period_clamped)
        
        team_clamped = (team_indicator + 1).clamp(0, 2)
        team_emb = self.team_embedding(team_clamped)
        
        # Concatenate all features and project down to d_model size
        context = torch.cat([score_emb, progress_emb, clock_emb, momentum_emb, period_emb, team_emb], dim=-1)
        return self.combine(context)


class BasketballMomentumTransformer(nn.Module):
    """
    The main Transformer Neural Network.
    """
    def __init__(self, vocab_size=20, d_model=128, n_heads=4, n_layers=2, dropout=0.1, num_classes=3):
        super().__init__()
        self.d_model = d_model
        
        # 1. Event Encoder
        self.event_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Context Encoder
        self.context_encoder = GameContextEncoding(d_model)
        
        # 3. Positional Encoding (Learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model))
        
        # 4. Input Fusion
        self.input_combine = nn.Linear(d_model * 2, d_model)
        
        # 5. Transformer Core
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 6. Output Head
        self.score_classifier = nn.Linear(d_model, num_classes) 

    def forward(self, event_ids, team_indicators, score_differentials, game_progress, 
                clock_normalized, periods, momentum_features):
        
        # Embed Events
        event_emb = self.event_embedding(event_ids) * math.sqrt(self.d_model)
        
        # Embed Context
        context_emb = self.context_encoder(
            score_differentials, game_progress, clock_normalized, 
            periods, team_indicators, momentum_features
        )
        
        # Combine Event + Context
        combined = torch.cat([event_emb, context_emb], dim=-1)
        x = self.input_combine(combined)
        
        # Add Positional Encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Run Transformer
        x = self.transformer(x)
        
        # Predict
        return {'score_logits': self.score_classifier(x)}


class MomentumLoss(nn.Module):
    """
    Custom Loss Function needed by train.py
    """
    def __init__(self, score_weight=1.0, momentum_weight=0.5, label_smoothing=0.1):
        super().__init__()
        self.score_weight = score_weight
        self.score_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, outputs, targets):
        # Flatten predictions to (Batch * Seq_Len, Classes)
        logits = outputs['score_logits'].view(-1, 3)
        targets = targets.view(-1)
        
        loss = self.score_loss(logits, targets)
        return {'total_loss': loss, 'score_loss': loss}


def create_model(config=None):
    """Helper to initialize the model from a config dictionary."""
    if config is None: config = {}
    # Filter config keys that match the model arguments
    valid_keys = ['vocab_size', 'd_model', 'n_heads', 'n_layers', 'dropout', 'num_classes']
    model_kwargs = {k: v for k, v in config.items() if k in valid_keys}
    return BasketballMomentumTransformer(**model_kwargs)


def count_parameters(model):
    """Helper to count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)