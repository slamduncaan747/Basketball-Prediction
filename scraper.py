"""
Fast College Basketball Data Scraper
=====================================
Uses multiple data sources for maximum speed and reliability:

1. SportsDataverse Pre-built Data (FASTEST)
   - Downloads pre-processed parquet files from GitHub releases
   - Contains 15+ years of play-by-play data
   - Updated daily during season

2. ESPN API with Async/Concurrent Requests (For live/recent data)
   - Uses asyncio + aiohttp for parallel requests
   - 10-50x faster than sequential scraping

Usage:
    # Download full seasons of historical data (recommended)
    python scraper.py --source sportsdataverse --seasons 2024 2023 2022 --output data/

    # Scrape specific date range from ESPN (for recent games)
    python scraper.py --source espn --start 20240301 --end 20240315 --output data/

    # Combine both: historical + recent
    python scraper.py --source both --seasons 2024 --start 20240301 --end 20240315 --output data/
"""

import asyncio
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
import logging
import time
import io

# Optional async support
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SPORTSDATAVERSE DATA LOADER (FASTEST - Pre-built Parquet Files)
# =============================================================================

class SportsDataverseLoader:
    """
    Load pre-built play-by-play data from SportsDataverse GitHub releases.
    This is the FASTEST way to get historical college basketball data.
    
    Data available from 2006-present, updated daily during season.
    """
    
    # Direct URLs to parquet files hosted on GitHub releases
    PBP_URL_TEMPLATE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_pbp/play_by_play_{season}.parquet"
    SCHEDULE_URL_TEMPLATE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_schedules/mbb_schedule_{season}.parquet"
    TEAM_BOX_URL_TEMPLATE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_team_boxscores/team_box_{season}.parquet"
    PLAYER_BOX_URL_TEMPLATE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_player_boxscores/player_box_{season}.parquet"
    
    def __init__(self):
        self.session = requests.Session()
    
    def load_pbp(self, seasons: List[int], show_progress: bool = True) -> pd.DataFrame:
        """
        Load play-by-play data for multiple seasons.
        
        Args:
            seasons: List of seasons (e.g., [2024, 2023, 2022])
                     Note: Season 2024 = 2023-24 season
            show_progress: Whether to show download progress
            
        Returns:
            DataFrame with all play-by-play data
        """
        all_data = []
        
        for season in seasons:
            url = self.PBP_URL_TEMPLATE.format(season=season)
            logger.info(f"Downloading play-by-play data for {season} season...")
            
            try:
                response = self.session.get(url, timeout=120)
                response.raise_for_status()
                
                # Read parquet from bytes
                df = pd.read_parquet(io.BytesIO(response.content))
                df['season'] = season
                all_data.append(df)
                
                logger.info(f"  ✓ Loaded {len(df):,} plays from {season}")
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    logger.warning(f"  ✗ Season {season} data not available yet")
                else:
                    logger.error(f"  ✗ Error loading {season}: {e}")
            except Exception as e:
                logger.error(f"  ✗ Error loading {season}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total: {len(combined):,} plays from {len(all_data)} seasons")
        
        return combined
    
    def load_schedule(self, seasons: List[int]) -> pd.DataFrame:
        """Load schedule data for multiple seasons."""
        all_data = []
        
        for season in seasons:
            url = self.SCHEDULE_URL_TEMPLATE.format(season=season)
            try:
                response = self.session.get(url, timeout=60)
                response.raise_for_status()
                df = pd.read_parquet(io.BytesIO(response.content))
                df['season'] = season
                all_data.append(df)
                logger.info(f"  ✓ Loaded {len(df):,} games from {season} schedule")
            except Exception as e:
                logger.warning(f"  ✗ Could not load {season} schedule: {e}")
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    def load_team_box(self, seasons: List[int]) -> pd.DataFrame:
        """Load team box scores for multiple seasons."""
        all_data = []
        
        for season in seasons:
            url = self.TEAM_BOX_URL_TEMPLATE.format(season=season)
            try:
                response = self.session.get(url, timeout=60)
                response.raise_for_status()
                df = pd.read_parquet(io.BytesIO(response.content))
                df['season'] = season
                all_data.append(df)
                logger.info(f"  ✓ Loaded {len(df):,} team box scores from {season}")
            except Exception as e:
                logger.warning(f"  ✗ Could not load {season} team box: {e}")
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# =============================================================================
# ASYNC ESPN SCRAPER (FAST - For Recent/Live Data)
# =============================================================================

class AsyncESPNScraper:
    """
    Async ESPN scraper using aiohttp for concurrent requests.
    10-50x faster than sequential requests.
    """
    
    SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
    GAME_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"
    
    def __init__(self, max_concurrent: int = 10, delay: float = 0.1):
        """
        Args:
            max_concurrent: Maximum concurrent requests
            delay: Delay between batches (be respectful to servers)
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp required for async scraping. Install with: pip install aiohttp")
        
        self.max_concurrent = max_concurrent
        self.delay = delay
        self.semaphore = None
    
    async def _fetch_json(self, session, url: str, params: dict = None) -> Optional[dict]:
        """Fetch JSON with rate limiting."""
        async with self.semaphore:
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
            except Exception as e:
                logger.warning(f"Error fetching {url}: {e}")
                return None
    
    async def get_games_for_date(self, session, date: str) -> List[Dict]:
        """Get all games for a specific date."""
        data = await self._fetch_json(session, self.SCOREBOARD_URL, {'dates': date, 'limit': 500})
        
        if not data:
            return []
        
        games = []
        for event in data.get('events', []):
            game_info = {
                'game_id': event.get('id'),
                'status': event.get('status', {}).get('type', {}).get('name'),
                'home_team': None,
                'away_team': None,
                'home_score': None,
                'away_score': None,
            }
            
            competitions = event.get('competitions', [])
            if competitions:
                comp = competitions[0]
                for team in comp.get('competitors', []):
                    if team.get('homeAway') == 'home':
                        game_info['home_team'] = team.get('team', {}).get('displayName')
                        game_info['home_score'] = int(team.get('score', 0) or 0)
                    else:
                        game_info['away_team'] = team.get('team', {}).get('displayName')
                        game_info['away_score'] = int(team.get('score', 0) or 0)
            
            games.append(game_info)
        
        return games
    
    async def get_play_by_play(self, session, game_id: str) -> Optional[Dict]:
        """Get play-by-play for a single game."""
        data = await self._fetch_json(session, self.GAME_URL, {'event': game_id})
        
        if not data:
            return None
        
        game_data = {'game_id': game_id, 'plays': [], 'teams': {}}
        
        # Get teams
        for team in data.get('boxscore', {}).get('teams', []):
            team_info = team.get('team', {})
            game_data['teams'][team_info.get('id')] = {
                'name': team_info.get('displayName'),
                'home_away': team.get('homeAway')
            }
        
        # Get plays
        for i, play in enumerate(data.get('plays', [])):
            parsed = self._parse_play(play, game_data['teams'], i)
            if parsed:
                game_data['plays'].append(parsed)
        
        return game_data
    
    def _parse_play(self, play: Dict, teams: Dict, index: int) -> Optional[Dict]:
        """Parse a single play."""
        try:
            team_id = play.get('team', {}).get('id')
            team_info = teams.get(team_id, {})
            
            clock_str = play.get('clock', {}).get('displayValue', '')
            clock_seconds = None
            if clock_str:
                parts = clock_str.split(':')
                if len(parts) == 2:
                    clock_seconds = int(parts[0]) * 60 + int(parts[1])
            
            text = (play.get('text') or '').lower()
            score_value = play.get('scoreValue', 0)
            scoring_play = play.get('scoringPlay', False)
            
            # Classify event
            event_category = self._classify_event(text, scoring_play, score_value)
            
            home_score = play.get('homeScore')
            away_score = play.get('awayScore')
            
            return {
                'play_id': play.get('id'),
                'sequence_number': play.get('sequenceNumber', index),
                'period': play.get('period', {}).get('number'),
                'clock': clock_str,
                'clock_seconds': clock_seconds,
                'team_name': team_info.get('name'),
                'is_home_team': team_info.get('home_away') == 'home',
                'text': play.get('text'),
                'home_score': home_score,
                'away_score': away_score,
                'score_value': score_value,
                'scoring_play': scoring_play,
                'score_differential': (home_score - away_score) if home_score is not None and away_score is not None else 0,
                'event_category': event_category,
                'coordinate_x': play.get('coordinate', {}).get('x') if play.get('coordinate') else None,
                'coordinate_y': play.get('coordinate', {}).get('y') if play.get('coordinate') else None,
            }
        except Exception:
            return None
    
    def _classify_event(self, text: str, scoring_play: bool, score_value: int) -> str:
        """Classify event type."""
        if scoring_play:
            if score_value == 3:
                return 'SCORE_3PT'
            elif score_value == 2:
                return 'SCORE_2PT'
            elif score_value == 1:
                return 'SCORE_FT'
        
        if 'miss' in text:
            if 'three' in text or '3pt' in text:
                return 'MISS_3PT'
            elif 'free throw' in text:
                return 'MISS_FT'
            return 'MISS_2PT'
        
        if 'turnover' in text:
            return 'TURNOVER'
        if 'steal' in text:
            return 'STEAL'
        if 'block' in text:
            return 'BLOCK'
        if 'foul' in text:
            return 'FOUL'
        if 'offensive rebound' in text:
            return 'REBOUND_OFF'
        if 'rebound' in text:
            return 'REBOUND_DEF'
        if 'timeout' in text:
            return 'TIMEOUT'
        if 'substitution' in text or 'enters' in text:
            return 'SUB'
        
        return 'OTHER'
    
    async def scrape_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Scrape all games in a date range concurrently.
        
        Args:
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            
        Returns:
            DataFrame with all play-by-play data
        """
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y%m%d'))
            current += timedelta(days=1)
        
        all_plays = []
        
        async with aiohttp.ClientSession(
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        ) as session:
            
            # Step 1: Get all games for all dates concurrently
            logger.info(f"Fetching games for {len(dates)} dates...")
            
            tasks = [self.get_games_for_date(session, date) for date in dates]
            results = await asyncio.gather(*tasks)
            
            # Collect completed games
            games_to_fetch = []
            for date, games in zip(dates, results):
                for game in games:
                    if game.get('status') == 'STATUS_FINAL':
                        games_to_fetch.append((date, game))
            
            logger.info(f"Found {len(games_to_fetch)} completed games")
            
            # Step 2: Fetch play-by-play for all games concurrently
            if games_to_fetch:
                logger.info("Fetching play-by-play data...")
                
                # Process in batches to avoid overwhelming the server
                batch_size = self.max_concurrent * 2
                
                for i in range(0, len(games_to_fetch), batch_size):
                    batch = games_to_fetch[i:i + batch_size]
                    
                    tasks = [
                        self.get_play_by_play(session, game['game_id'])
                        for date, game in batch
                    ]
                    
                    pbp_results = await asyncio.gather(*tasks)
                    
                    for (date, game), pbp in zip(batch, pbp_results):
                        if pbp and pbp['plays']:
                            for play in pbp['plays']:
                                play['game_id'] = game['game_id']
                                play['date'] = date
                                play['home_team'] = game.get('home_team')
                                play['away_team'] = game.get('away_team')
                                play['final_home_score'] = game.get('home_score')
                                play['final_away_score'] = game.get('away_score')
                                all_plays.append(play)
                    
                    logger.info(f"  Processed {min(i + batch_size, len(games_to_fetch))}/{len(games_to_fetch)} games")
                    
                    await asyncio.sleep(self.delay)
        
        df = pd.DataFrame(all_plays)
        logger.info(f"Total: {len(df):,} plays from {len(games_to_fetch)} games")
        
        return df
    
    def scrape_date_range_sync(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Synchronous wrapper for scrape_date_range."""
        return asyncio.run(self.scrape_date_range(start_date, end_date))


# =============================================================================
# THREADED ESPN SCRAPER (ALTERNATIVE - No async required)
# =============================================================================

class ThreadedESPNScraper:
    """
    Thread-based ESPN scraper for environments without asyncio support.
    Uses ThreadPoolExecutor for concurrent requests.
    """
    
    SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
    GAME_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"
    
    def __init__(self, max_workers: int = 10, delay: float = 0.1):
        self.max_workers = max_workers
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _fetch_games(self, date: str) -> Tuple[str, List[Dict]]:
        """Fetch games for a single date."""
        try:
            response = self.session.get(
                self.SCOREBOARD_URL,
                params={'dates': date, 'limit': 500},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            games = []
            for event in data.get('events', []):
                game_info = {
                    'game_id': event.get('id'),
                    'status': event.get('status', {}).get('type', {}).get('name'),
                    'home_team': None,
                    'away_team': None,
                    'home_score': None,
                    'away_score': None,
                }
                
                competitions = event.get('competitions', [])
                if competitions:
                    comp = competitions[0]
                    for team in comp.get('competitors', []):
                        if team.get('homeAway') == 'home':
                            game_info['home_team'] = team.get('team', {}).get('displayName')
                            game_info['home_score'] = int(team.get('score', 0) or 0)
                        else:
                            game_info['away_team'] = team.get('team', {}).get('displayName')
                            game_info['away_score'] = int(team.get('score', 0) or 0)
                
                games.append(game_info)
            
            return date, games
        except Exception as e:
            logger.warning(f"Error fetching games for {date}: {e}")
            return date, []
    
    def _fetch_pbp(self, game_info: Tuple[str, Dict]) -> Tuple[str, Dict, List[Dict]]:
        """Fetch play-by-play for a single game."""
        date, game = game_info
        game_id = game['game_id']
        
        try:
            time.sleep(self.delay)  # Rate limiting
            
            response = self.session.get(
                self.GAME_URL,
                params={'event': game_id},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse teams
            teams = {}
            for team in data.get('boxscore', {}).get('teams', []):
                team_info = team.get('team', {})
                teams[team_info.get('id')] = {
                    'name': team_info.get('displayName'),
                    'home_away': team.get('homeAway')
                }
            
            # Parse plays
            plays = []
            for i, play in enumerate(data.get('plays', [])):
                parsed = self._parse_play(play, teams, i)
                if parsed:
                    parsed['game_id'] = game_id
                    parsed['date'] = date
                    parsed['home_team'] = game.get('home_team')
                    parsed['away_team'] = game.get('away_team')
                    parsed['final_home_score'] = game.get('home_score')
                    parsed['final_away_score'] = game.get('away_score')
                    plays.append(parsed)
            
            return date, game, plays
        except Exception as e:
            logger.warning(f"Error fetching PBP for {game_id}: {e}")
            return date, game, []
    
    def _parse_play(self, play: Dict, teams: Dict, index: int) -> Optional[Dict]:
        """Parse a single play event."""
        try:
            team_id = play.get('team', {}).get('id')
            team_info = teams.get(team_id, {})
            
            clock_str = play.get('clock', {}).get('displayValue', '')
            clock_seconds = None
            if clock_str:
                parts = clock_str.split(':')
                if len(parts) == 2:
                    clock_seconds = int(parts[0]) * 60 + int(parts[1])
            
            text = (play.get('text') or '').lower()
            score_value = play.get('scoreValue', 0)
            scoring_play = play.get('scoringPlay', False)
            
            home_score = play.get('homeScore')
            away_score = play.get('awayScore')
            
            return {
                'play_id': play.get('id'),
                'sequence_number': play.get('sequenceNumber', index),
                'period': play.get('period', {}).get('number'),
                'clock': clock_str,
                'clock_seconds': clock_seconds,
                'team_name': team_info.get('name'),
                'is_home_team': team_info.get('home_away') == 'home',
                'text': play.get('text'),
                'home_score': home_score,
                'away_score': away_score,
                'score_value': score_value,
                'scoring_play': scoring_play,
                'score_differential': (home_score - away_score) if home_score is not None and away_score is not None else 0,
                'event_category': self._classify_event(text, scoring_play, score_value),
                'coordinate_x': play.get('coordinate', {}).get('x') if play.get('coordinate') else None,
                'coordinate_y': play.get('coordinate', {}).get('y') if play.get('coordinate') else None,
            }
        except Exception:
            return None
    
    def _classify_event(self, text: str, scoring_play: bool, score_value: int) -> str:
        """Classify event type."""
        if scoring_play:
            if score_value == 3:
                return 'SCORE_3PT'
            elif score_value == 2:
                return 'SCORE_2PT'
            elif score_value == 1:
                return 'SCORE_FT'
        
        if 'miss' in text:
            if 'three' in text or '3pt' in text:
                return 'MISS_3PT'
            elif 'free throw' in text:
                return 'MISS_FT'
            return 'MISS_2PT'
        
        if 'turnover' in text:
            return 'TURNOVER'
        if 'steal' in text:
            return 'STEAL'
        if 'block' in text:
            return 'BLOCK'
        if 'foul' in text:
            return 'FOUL'
        if 'offensive rebound' in text:
            return 'REBOUND_OFF'
        if 'rebound' in text:
            return 'REBOUND_DEF'
        if 'timeout' in text:
            return 'TIMEOUT'
        if 'substitution' in text or 'enters' in text:
            return 'SUB'
        
        return 'OTHER'
    
    def scrape_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Scrape all games in a date range using thread pool.
        
        Args:
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            
        Returns:
            DataFrame with all play-by-play data
        """
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y%m%d'))
            current += timedelta(days=1)
        
        # Step 1: Get all games concurrently
        logger.info(f"Fetching games for {len(dates)} dates...")
        
        games_to_fetch = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._fetch_games, date): date for date in dates}
            
            for future in as_completed(futures):
                date, games = future.result()
                for game in games:
                    if game.get('status') == 'STATUS_FINAL':
                        games_to_fetch.append((date, game))
        
        logger.info(f"Found {len(games_to_fetch)} completed games")
        
        # Step 2: Fetch play-by-play concurrently
        all_plays = []
        
        if games_to_fetch:
            logger.info("Fetching play-by-play data...")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._fetch_pbp, g): g for g in games_to_fetch}
                
                completed = 0
                for future in as_completed(futures):
                    date, game, plays = future.result()
                    all_plays.extend(plays)
                    
                    completed += 1
                    if completed % 50 == 0:
                        logger.info(f"  Processed {completed}/{len(games_to_fetch)} games")
        
        df = pd.DataFrame(all_plays)
        logger.info(f"Total: {len(df):,} plays from {len(games_to_fetch)} games")
        
        return df


# =============================================================================
# DATA STANDARDIZATION
# =============================================================================

def standardize_pbp_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize play-by-play data from different sources into a common format.
    
    This ensures consistent column names and types regardless of whether
    data came from SportsDataverse or ESPN scraper.
    """
    # Map SportsDataverse columns to our standard format
    column_mapping = {
        'game_id': 'game_id',
        'id': 'play_id',
        'sequence_number': 'sequence_number',
        'period_number': 'period',
        'half': 'period',
        'clock_display_value': 'clock',
        'clock_minutes': 'clock_minutes',
        'clock_seconds': 'clock_seconds',
        'home_score': 'home_score',
        'away_score': 'away_score',
        'scoring_play': 'scoring_play',
        'score_value': 'score_value',
        'text': 'text',
        'type_text': 'event_type',
        'home_team_name': 'home_team',
        'away_team_name': 'away_team',
        'coordinate_x': 'coordinate_x',
        'coordinate_y': 'coordinate_y',
    }
    
    # Rename columns that exist
    rename_dict = {old: new for old, new in column_mapping.items() if old in df.columns and old != new}
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # Ensure required columns exist
    required_cols = ['game_id', 'sequence_number', 'period', 'home_score', 'away_score']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    
    # Add event_category if missing
    if 'event_category' not in df.columns and 'text' in df.columns:
        df['event_category'] = df['text'].apply(lambda x: classify_event_text(str(x).lower() if pd.notna(x) else ''))
    
    # Calculate score_differential if missing
    if 'score_differential' not in df.columns:
        df['score_differential'] = df['home_score'].fillna(0) - df['away_score'].fillna(0)
    
    return df


def classify_event_text(text: str) -> str:
    """Classify event based on text description."""
    text = text.lower()
    
    if 'made' in text:
        if 'three' in text or '3pt' in text or 'three-point' in text:
            return 'SCORE_3PT'
        elif 'free throw' in text:
            return 'SCORE_FT'
        else:
            return 'SCORE_2PT'
    
    if 'miss' in text:
        if 'three' in text or '3pt' in text:
            return 'MISS_3PT'
        elif 'free throw' in text:
            return 'MISS_FT'
        return 'MISS_2PT'
    
    if 'turnover' in text:
        return 'TURNOVER'
    if 'steal' in text:
        return 'STEAL'
    if 'block' in text:
        return 'BLOCK'
    if 'foul' in text:
        return 'FOUL'
    if 'offensive rebound' in text:
        return 'REBOUND_OFF'
    if 'rebound' in text:
        return 'REBOUND_DEF'
    if 'timeout' in text:
        return 'TIMEOUT'
    if 'substitution' in text or 'enters' in text:
        return 'SUB'
    
    return 'OTHER'


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fast College Basketball Data Scraper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download historical data from SportsDataverse (FASTEST)
    python scraper.py --source sportsdataverse --seasons 2024 2023 2022

    # Scrape recent games from ESPN
    python scraper.py --source espn --start 20240301 --end 20240315

    # Use both sources
    python scraper.py --source both --seasons 2024 --start 20240301 --end 20240315
        """
    )
    
    parser.add_argument(
        '--source', 
        type=str, 
        choices=['sportsdataverse', 'espn', 'espn-threaded', 'both'],
        default='sportsdataverse',
        help='Data source: sportsdataverse (fastest), espn (async), espn-threaded, or both'
    )
    parser.add_argument(
        '--seasons', 
        type=int, 
        nargs='+',
        default=[2024],
        help='Seasons to download (for sportsdataverse). e.g., 2024 2023 2022'
    )
    parser.add_argument('--start', type=str, help='Start date YYYYMMDD (for ESPN)')
    parser.add_argument('--end', type=str, help='End date YYYYMMDD (for ESPN)')
    parser.add_argument(
        '--output', 
        type=str, 
        default='data/plays.parquet',
        help='Output file path (.parquet or .csv)'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=10,
        help='Max concurrent requests for ESPN scraping'
    )
    
    args = parser.parse_args()
    
    all_data = []
    
    # Source: SportsDataverse
    if args.source in ['sportsdataverse', 'both']:
        logger.info("=" * 60)
        logger.info("Loading from SportsDataverse...")
        logger.info("=" * 60)
        
        loader = SportsDataverseLoader()
        df = loader.load_pbp(args.seasons)
        
        if not df.empty:
            df = standardize_pbp_data(df)
            all_data.append(df)
    
    # Source: ESPN
    if args.source in ['espn', 'espn-threaded', 'both']:
        if not args.start or not args.end:
            logger.error("--start and --end required for ESPN source")
            return
        
        logger.info("=" * 60)
        logger.info(f"Scraping from ESPN ({args.start} to {args.end})...")
        logger.info("=" * 60)
        
        if args.source == 'espn-threaded' or not AIOHTTP_AVAILABLE:
            if args.source == 'espn' and not AIOHTTP_AVAILABLE:
                logger.warning("aiohttp not available, using threaded scraper instead")
            scraper = ThreadedESPNScraper(max_workers=args.max_concurrent)
            df = scraper.scrape_date_range(args.start, args.end)
        else:
            scraper = AsyncESPNScraper(max_concurrent=args.max_concurrent)
            df = scraper.scrape_date_range_sync(args.start, args.end)
        
        if not df.empty:
            all_data.append(df)
    
    # Combine and save
    if not all_data:
        logger.error("No data collected!")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates if combining sources
    if len(all_data) > 1 and 'game_id' in combined.columns and 'sequence_number' in combined.columns:
        combined = combined.drop_duplicates(subset=['game_id', 'sequence_number'])
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.output.endswith('.parquet'):
        combined.to_parquet(args.output, index=False)
    else:
        combined.to_csv(args.output, index=False)
    
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total plays: {len(combined):,}")
    logger.info(f"Unique games: {combined['game_id'].nunique():,}")
    logger.info(f"Saved to: {args.output}")
    
    if 'event_category' in combined.columns:
        logger.info("\nEvent breakdown:")
        print(combined['event_category'].value_counts())


if __name__ == "__main__":
    main()