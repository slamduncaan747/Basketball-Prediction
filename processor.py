import pandas as pd
import torch
import numpy as np
import json
import logging
from typing import List, Dict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasketballDataProcessor:
    def __init__(self, seq_len=64, stride=16, min_events=100):
        self.seq_len = seq_len
        self.stride = stride
        self.min_events = min_events
        self.event_map = {
            'SCORE_3PT':0, 'SCORE_2PT':1, 'SCORE_FT':2, 'MISS_3PT':3, 'MISS_2PT':4, 
            'MISS_FT':5, 'TURNOVER':6, 'STEAL':7, 'BLOCK':8, 'FOUL':9, 
            'REBOUND_OFF':10, 'REBOUND_DEF':11, 'ASSIST':12, 'TIMEOUT':13, 
            'JUMP_BALL':14, 'SUB':15, 'OTHER':16, 'PAD':17, 'START':18, 'END':19
        }

    def process_all(self, df: pd.DataFrame) -> List[Dict]:
        all_seqs = []
        game_ids = df['game_id'].unique()
        logger.info(f"Processing {len(game_ids)} games...")
        
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
        
        # --- 1. Basic Features ---
        e_ids = [self.event_map.get(e.get('event_category', 'OTHER'), 16) for e in segment]
        teams = [1 if e.get('is_home_team') else (0 if e.get('is_home_team') is False else -1) for e in segment]
        diffs = [e.get('score_differential', 0) for e in segment]
        periods = [int(e.get('period', 1)) for e in segment]
        
        # --- 2. Temporal Features ---
        clocks = [float(e.get('clock_seconds', 0)) for e in segment]
        clock_norm = [c / 1200.0 for c in clocks]
        
        progress = []
        for p, c in zip(periods, clocks):
            if p == 1: prog = (1200 - c) / 2400.0
            elif p == 2: prog = 0.5 + ((1200 - c) / 2400.0)
            else: prog = 1.0
            progress.append(min(max(prog, 0.0), 1.0))

        # --- 3. Momentum Features (Placeholder) ---
        mom_feats = [[0.0] * 10 for _ in range(len(segment))] 

        # --- 4. Targets ---
        targets = []
        target_window = 5
        for i in range(len(segment)):
            curr_tid = segment[i].get('team_id')
            global_idx = start_idx + i
            future = all_events[global_idx+1 : global_idx+1+target_window]
            
            outcome = 0 
            for fe in future:
                if fe.get('scoring_play'):
                    outcome = 2 if fe.get('team_id') == curr_tid else 1
                    break
            targets.append(outcome)

        # === KEY FIX: MATCH NAMES TO TRAIN.PY ===
        return {
            'event_ids': e_ids,
            'team_indicators': teams,
            'score_differentials': diffs,   # Renamed from score_diffs
            'game_progress': progress,      # Renamed from progress
            'clock_normalized': clock_norm, # Renamed from clock_norm
            'periods': periods,
            'momentum_features': mom_feats, # Renamed from momentum
            'targets': targets
        }

    def save_data(self, sequences: List[Dict], output_dir: str):
        if not sequences:
            logger.warning("No sequences generated! Check your input data.")
            return

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Convert list of dicts to dict of lists
        data_dict = {k: [s[k] for s in sequences] for k in sequences[0].keys()}
        
        logger.info(f"Saving {len(sequences)} sequences to {output_dir}...")
        
        for key, val in data_dict.items():
            file_path = out_path / f"{key}.npy"
            np.save(file_path, np.array(val))
            logger.info(f"Saved {file_path}")
            
        with open(out_path / 'vocab.json', 'w') as f:
            json.dump({'vocab_size': len(self.event_map), 'sequence_length': self.seq_len}, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    df = pd.read_parquet(args.input)
    proc = BasketballDataProcessor()
    seqs = proc.process_all(df)
    proc.save_data(seqs, args.output)