import requests
import pandas as pd
import time
import argparse
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ESPNCollegeBasketballScraper:
    SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
    GAME_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
    
    def get_games_for_date(self, date: str) -> List[Dict]:
        try:
            response = self.session.get(self.SCOREBOARD_URL, params={'dates': date, 'limit': 500}, timeout=30)
            response.raise_for_status()
            data = response.json()
            games = []
            for event in data.get('events', []):
                game_info = {
                    'game_id': event.get('id'),
                    'status': event.get('status', {}).get('type', {}).get('name'),
                    'home_team': None, 'away_team': None
                }
                competitions = event.get('competitions', [])
                if competitions:
                    comp = competitions[0]
                    for team in comp.get('competitors', []):
                        if team.get('homeAway') == 'home':
                            game_info['home_team'] = team.get('team', {}).get('displayName')
                        else:
                            game_info['away_team'] = team.get('team', {}).get('displayName')
                games.append(game_info)
            return games
        except Exception as e:
            logger.error(f"Error fetching games for {date}: {e}")
            return []
    
    def get_play_by_play(self, game_id: str) -> Optional[Dict]:
        try:
            response = self.session.get(self.GAME_URL, params={'event': game_id}, timeout=30)
            response.raise_for_status()
            data = response.json()
            game_data = {'game_id': game_id, 'plays': [], 'teams': {}}
            for team in data.get('boxscore', {}).get('teams', []):
                team_info = team.get('team', {})
                game_data['teams'][team_info.get('id')] = {
                    'name': team_info.get('displayName'),
                    'home_away': team.get('homeAway')
                }
            for play in data.get('plays', []):
                play_event = self._parse_play(play, game_data['teams'])
                if play_event:
                    game_data['plays'].append(play_event)
            time.sleep(self.delay)
            return game_data
        except Exception as e:
            logger.error(f"Error fetching play-by-play for game {game_id}: {e}")
            return None
    
    def _parse_play(self, play: Dict, teams: Dict) -> Optional[Dict]:
        try:
            team_id = play.get('team', {}).get('id')
            team_info = teams.get(team_id, {})
            play_event = {
                'play_id': play.get('id'),
                'period': play.get('period', {}).get('number'),
                'clock': play.get('clock', {}).get('displayValue'),
                'clock_seconds': self._clock_to_seconds(play.get('clock', {}).get('displayValue')),
                'team_name': team_info.get('name'),
                'is_home_team': team_info.get('home_away') == 'home', # Critical for processor
                'text': play.get('text'),
                'home_score': play.get('homeScore'),
                'away_score': play.get('awayScore'),
                'score_value': play.get('scoreValue', 0),
                'scoring_play': play.get('scoringPlay', False)
            }
            if play_event['home_score'] is not None and play_event['away_score'] is not None:
                play_event['score_differential'] = play_event['home_score'] - play_event['away_score']
            else:
                play_event['score_differential'] = 0
            
            play_event['event_category'] = self._classify_event(play_event)
            return play_event
        except Exception:
            return None

    def _clock_to_seconds(self, clock_str: str) -> Optional[int]:
        if not clock_str: return None
        try:
            parts = clock_str.split(':')
            return int(parts[0]) * 60 + int(parts[1]) if len(parts) == 2 else int(parts[0])
        except: return None
    
    def _classify_event(self, play: Dict) -> str:
        text = (play.get('text') or '').lower()
        score_val = play.get('score_value', 0)
        if play.get('scoring_play'):
            return f'SCORE_{score_val}PT' if score_val in [2,3] else 'SCORE_FT'
        if 'miss' in text:
            return 'MISS_3PT' if 'three' in text or '3pt' in text else ('MISS_FT' if 'free' in text else 'MISS_2PT')
        if 'turnover' in text: return 'TURNOVER'
        if 'steal' in text: return 'STEAL'
        if 'block' in text: return 'BLOCK'
        if 'foul' in text: return 'FOUL'
        if 'rebound' in text: return 'REBOUND_OFF' if 'offensive' in text else 'REBOUND_DEF'
        if 'timeout' in text: return 'TIMEOUT'
        if 'substitution' in text or 'enters' in text: return 'SUB'
        return 'OTHER'
    
    def scrape_date_range(self, start_date: str, end_date: str, output_path: str):
        all_plays = []
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        current = start
        while current <= end:
            date_str = current.strftime('%Y%m%d')
            logger.info(f"Scraping {date_str}...")
            games = self.get_games_for_date(date_str)
            for game in games:
                if game.get('status') == 'STATUS_FINAL':
                    data = self.get_play_by_play(game['game_id'])
                    if data:
                        for p in data['plays']:
                            p.update({'game_id': game['game_id'], 'date': date_str})
                            all_plays.append(p)
            current += timedelta(days=1)
        
        df = pd.DataFrame(all_plays)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if output_path.endswith('.parquet'):
            df.to_parquet(output_path)
        else:
            df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} plays to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, required=True)
    parser.add_argument('--end', type=str, required=True)
    parser.add_argument('--output', type=str, default='data/games.parquet')
    args = parser.parse_args()
    scraper = ESPNCollegeBasketballScraper()
    scraper.scrape_date_range(args.start, args.end, args.output)