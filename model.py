import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# PART 1: UPGRADED PREPROCESSOR
# (Fixes the missing time features)
# ==========================================

class BasketballDataProcessor:
    def __init__(self, seq_len=64, stride=16, min_events=100):
        self.seq_len = seq_len
        self.stride = stride
        self.min_events = min_events
        # Re-using the extractor logic from previous steps
        self.event_map = {'SCORE_3PT':0, 'SCORE_2PT':1, 'SCORE_FT':2, 'MISS_3PT':3, 'MISS_2PT':4, 
                          'MISS_FT':5, 'TURNOVER':6, 'STEAL':7, 'BLOCK':8, 'FOUL':9, 
                          'REBOUND_OFF':10, 'REBOUND_DEF':11, 'ASSIST':12, 'TIMEOUT':13, 
                          'JUMP_BALL':14, 'SUB':15, 'OTHER':16, 'PAD':17, 'START':18, 'END':19}

    def process_all(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame into list of sequence dictionaries."""
        all_seqs = []
        game_ids = df['game_id'].unique()
        
        for gid in game_ids:
            gdf = df[df['game_id'] == gid].sort_values('sequence_number').reset_index(drop=True)
            if len(gdf) < self.min_events: continue
            
            events = gdf.to_dict('records')
            for start in range(0, len(events) - self.seq_len, self.stride):
                seq = self._make_seq(events, start)
                all_seqs.append(seq)
        return all_seqs

    def _make_seq(self, all_events, start_idx):
        segment = all_events[start_idx : start_idx + self.seq_len]
        
        # --- FEATURE EXTRACTION ---
        # 1. IDs & Team
        e_ids = [self.event_map.get(e.get('event_category', 'OTHER'), 16) for e in segment]
        teams = [1 if e.get('is_home_team') else (0 if e.get('is_home_team') is False else -1) for e in segment]
        
        # 2. Context Features
        diffs = [e.get('score_differential', 0) for e in segment]
        
        # 3. Temporal Features (NEWLY ADDED FOR COMPATIBILITY)
        # ---------------------------------------------------
        periods = [int(e.get('period', 1)) for e in segment]
        
        # Normalize clock: 20 minutes (1200s) -> 0.0 to 1.0
        clocks = [float(e.get('clock_seconds', 0)) for e in segment]
        clock_norm = [c / 1200.0 for c in clocks]
        
        # Calculate Game Progress (approximate)
        # Period 1: 0.0 - 0.5, Period 2: 0.5 - 1.0
        progress = []
        for p, c in zip(periods, clocks):
            if p == 1:
                prog = (1200 - c) / 2400.0
            elif p == 2:
                prog = 0.5 + ((1200 - c) / 2400.0)
            else:
                prog = 1.0 # Overtime
            progress.append(min(max(prog, 0.0), 1.0))
        # ---------------------------------------------------

        # 4. Dummy Momentum Features 
        # (In a real run, use the MomentumFeatureExtractor class)
        mom_feats = [[0.0] * 10 for _ in range(len(segment))] 

        # 5. Target (Next score prediction)
        targets = [0] * len(segment) # Dummy targets for demo

        return {
            'event_ids': e_ids, 'team_indicators': teams, 'score_diffs': diffs,
            'periods': periods, 'clock_norm': clock_norm, 'progress': progress,
            'momentum': mom_feats, 'targets': targets
        }

    def to_tensors(self, seqs: List[Dict]) -> Dict[str, torch.Tensor]:
        """Convert list of dicts to PyTorch tensors."""
        N = len(seqs)
        L = self.seq_len
        
        return {
            'event_ids': torch.tensor([s['event_ids'] for s in seqs], dtype=torch.long),
            'team_indicators': torch.tensor([s['team_indicators'] for s in seqs], dtype=torch.long),
            'score_differentials': torch.tensor([s['score_diffs'] for s in seqs], dtype=torch.float32),
            'game_progress': torch.tensor([s['progress'] for s in seqs], dtype=torch.float32),
            'clock_normalized': torch.tensor([s['clock_norm'] for s in seqs], dtype=torch.float32),
            'periods': torch.tensor([s['periods'] for s in seqs], dtype=torch.long),
            'momentum_features': torch.tensor([s['momentum'] for s in seqs], dtype=torch.float32),
            'targets': torch.tensor([s['targets'] for s in seqs], dtype=torch.long)
        }

# ==========================================
# PART 2: THE MODEL (As provided)
# ==========================================

class GameContextEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score_diff_proj = nn.Linear(1, d_model // 4)
        self.game_progress_proj = nn.Linear(1, d_model // 4)
        self.clock_proj = nn.Linear(1, d_model // 4)
        self.momentum_proj = nn.Linear(10, d_model // 4)
        self.period_embedding = nn.Embedding(6, d_model // 4)
        self.team_embedding = nn.Embedding(3, d_model // 4)
        self.combine = nn.Linear(d_model + d_model // 4, d_model)

    def forward(self, score_diff, game_progress, clock, period, team_indicator, momentum_features):
        # Shape handling for simple Linear layers
        score_emb = self.score_diff_proj(score_diff.unsqueeze(-1))
        progress_emb = self.game_progress_proj(game_progress.unsqueeze(-1))
        clock_emb = self.clock_proj(clock.unsqueeze(-1))
        momentum_emb = self.momentum_proj(momentum_features)
        
        period_clamped = period.clamp(0, 5)
        period_emb = self.period_embedding(period_clamped)
        
        team_clamped = (team_indicator + 1).clamp(0, 2)
        team_emb = self.team_embedding(team_clamped)
        
        context = torch.cat([score_emb, progress_emb, clock_emb, momentum_emb, period_emb, team_emb], dim=-1)
        return self.combine(context)

class BasketballMomentumTransformer(nn.Module):
    def __init__(self, vocab_size=20, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.d_model = d_model
        self.event_embedding = nn.Embedding(vocab_size, d_model)
        self.context_encoder = GameContextEncoding(d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model)) # Simplified PosEnc
        self.input_combine = nn.Linear(d_model * 2, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.score_classifier = nn.Linear(d_model, 3) 

    def forward(self, event_ids, team_indicators, score_differentials, game_progress, 
                clock_normalized, periods, momentum_features):
        
        # 1. Embed Events
        event_emb = self.event_embedding(event_ids) * math.sqrt(self.d_model)
        
        # 2. Embed Context
        context_emb = self.context_encoder(
            score_differentials, game_progress, clock_normalized, 
            periods, team_indicators, momentum_features
        )
        
        # 3. Combine & Transform
        x = self.input_combine(torch.cat([event_emb, context_emb], dim=-1))
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        
        return {'score_logits': self.score_classifier(x)}

