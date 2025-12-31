# streamlit program to display French Bridge (ffbridge) game results and statistics.
# Invoke from system prompt using: streamlit run ffbridge_streamlit.py

# todo (priority):

# todo:
# before production, ask claude to check for bugs and concurrency issues.
# move any Roy Rene code into mlBridgeBPLib
# get lancelot api working again modeling on ffbridge (legacy?) api.
# show df of freq of scores; freq, score, matchpoints
# implement ffbridge_auth_playwright.py code to get bearer token. use it if .env doesn't exist or hatch a scheme to refresh token.(?)
# tell ffbridge to unblock my ip address
# Refactor common postmortem methods into ml bridge class. Sync with other postmortem projects.
# Decide on whether to use faster RRN code or slower be-nice-to-server code? Does it matter?
# Some tournament result pages (Monday Simultané Octopus) omit Contract e.g. 34350. lancelot api doesn't know of the event.


import streamlit as st
import streamlit_chat
from streamlit_extras.bottom_container import bottom
from stqdm import stqdm
# from streamlit_autocomplete.autocomplete import st_textcomplete_autocomplete  # Not working as expected


import pathlib
import pandas as pd # only used for __version__ for now. might need for plotting later as pandas plotting support is better than polars.
import polars as pl
import requests
import duckdb
import json
import sys
import os
import platform
import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv

from urllib.parse import urlparse
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from typing_extensions import TypedDict

import endplay # for __version__

# Only declared to display version information
#import fastai
import numpy as np
import pandas as pd
#import safetensors
#import sklearn
#import torch
import mlBridgeLib.mlBridgeBPLib

# assumes symlinks are created in current directory.
sys.path.append(str(pathlib.Path.cwd().joinpath('streamlitlib')))  # global
sys.path.append(str(pathlib.Path.cwd().joinpath('mlBridgeLib')))  # global # Requires "./mlBridgeLib" be in extraPaths in .vscode/settings.json
sys.path.append(str(pathlib.Path.cwd().joinpath('ffbridgelib')))  # global

import mlBridgeFFLib
import streamlitlib
import time
#import mlBridgeLib
from mlBridgeLib.mlBridgeAugmentLib import (
    AllAugmentations,
)
from mlBridgeLib.mlBridgePostmortemLib import PostmortemBase
#import mlBridgeEndplayLib

# Type definitions for better type checking
class ApiUrlConfig(TypedDict):
    url: str
    should_cache: bool

class ApiUrlsDict(TypedDict):
    simultaneous_deals: ApiUrlConfig
    simultaneous_description_by_organization_id: ApiUrlConfig
    simultaneous_tournaments_by_organization_id: ApiUrlConfig
    my_infos: ApiUrlConfig
    members: ApiUrlConfig
    person: ApiUrlConfig
    organization_by_person_organization_id: ApiUrlConfig
    person_by_person_organization_id: ApiUrlConfig

class DataFramesDict(TypedDict):
    boards: Optional[pl.DataFrame]
    score_frequency: Optional[pl.DataFrame]


def _derive_person_organization_id_scalar(members_df: pl.DataFrame) -> Optional[int]:
    """Derive a single person_organization_id from members DataFrame.
    Preference order:
    1) st.session_state.org_id if present in seasons_organization_id
    2) last value of unique(non-null) seasons_organization_id
    3) None
    """
    try:
        if members_df is None or 'seasons_organization_id' not in members_df.columns:
            return None
        person_org_series = members_df['seasons_organization_id'].drop_nulls().unique()
        candidate_ids = person_org_series.to_list() if hasattr(person_org_series, 'to_list') else list(person_org_series)
        org_id = st.session_state.get('org_id')
        if org_id in candidate_ids:
            return int(org_id)
        if len(candidate_ids):
            return int(candidate_ids[-1])
    except Exception:
        pass
    return None

def make_api_request_licencie(full_url: str, headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """Make API request with full URL
    
    Args:
        full_url: The complete URL to make the API request to
        headers: Optional additional headers to include in the request
        
    Returns:
        JSON response data as dictionary, or None if request failed
    """
    from urllib.parse import urlparse
    
    # Parse domain from URL
    parsed_url = urlparse(full_url)
    domain = parsed_url.netloc
    
    # Get appropriate token for domain
    token = st.session_state.ffbridge_easi_token
    if not token:
        return None
    
    # Default headers
    default_headers = {
        "Authorization": f"Bearer {token}",
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.9,fr;q=0.8",
        "origin": "https://www.ffbridge.fr",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    # Merge with provided headers
    if headers:
        default_headers.update(headers)
    
    try:
        print(f"Making API request to: {full_url}")
        print(f"Using domain: {domain}")
        print(f"Using token: {token[:20]}...")
        
        response = requests.get(full_url, headers=default_headers, timeout=30)
        response.raise_for_status()
        
        return response.json()
        
    except Exception as e:
        st.error(f"API request failed: {e}")
        return None

# Legacy function - now handled by the base class
def ShowDataFrameTable(df: pl.DataFrame, key: str, query: str = 'SELECT * FROM self', show_sql_query: bool = True, height_rows: int = 25) -> Optional[pl.DataFrame]:
    """Legacy function - use app.ShowDataFrameTable instead"""
    if 'app' in st.session_state:
        return st.session_state.app.ShowDataFrameTable(df, key, query, show_sql_query, height_rows)
    else:
        # Fallback for backward compatibility
        if show_sql_query and st.session_state.show_sql_query:
            st.text(f"SQL Query: {query}")

        if 'from self' not in query.lower():
            query = 'FROM self ' + query
        
        try:
            con = get_session_duckdb_connection()
            result_df = con.execute(query).pl()
            if show_sql_query and st.session_state.show_sql_query:
                st.text(f"Result is a dataframe of {len(result_df)} rows.")
            streamlitlib.ShowDataFrameTable(result_df, key, height_rows=height_rows)
        except Exception as e:
            st.error(f"duckdb exception: error:{e} query:{query}")
            return None
        
        return result_df


def game_url_on_change() -> None:
    """Handle game URL input change event"""
    st.session_state.game_url = st.session_state.create_sidebar_game_url_on_change
    st.session_state.sql_query_mode = False


def chat_input_on_submit() -> None:
    """Handle chat input submission and process SQL queries"""
    if 'app' in st.session_state:
        st.session_state.app.chat_input_on_submit()
    else:
        # Fallback for backward compatibility
        prompt = st.session_state.main_prompt_chat_input
        sql_query = process_prompt_macros(prompt)
        if not st.session_state.sql_query_mode:
            st.session_state.sql_query_mode = True
            st.session_state.sql_queries.clear()
        st.session_state.sql_queries.append((prompt,sql_query))
        st.session_state.main_section_container = st.empty()
        st.session_state.main_section_container = st.container()
        with st.session_state.main_section_container:
            for i, (prompt,sql_query) in enumerate(st.session_state.sql_queries):
                ShowDataFrameTable(st.session_state.df, query=sql_query, key=f'user_query_main_doit_{i}')


def single_dummy_sample_count_on_change() -> None:
    """Handle single dummy sample count input change event"""
    st.session_state.single_dummy_sample_count = st.session_state.single_dummy_sample_count_number_input
    change_game_state(st.session_state.player_id, st.session_state.session_id)
    st.session_state.sql_query_mode = False


def sql_query_on_change() -> None:
    """Handle SQL query input change event"""
    st.session_state.show_sql_query = st.session_state.show_sql_query_checkbox
    #st.session_state.sql_query_mode = False # don't alter sql query mode.


def debug_mode_on_change() -> None:
    """Handle debug mode input change event"""
    st.session_state.debug_mode = st.session_state.debug_mode_checkbox
    #st.session_state.sql_query_mode = False # don't alter sql query mode.


def generic_input_on_change() -> None:
    """Generic handler for input change events that disable SQL query mode"""
    st.session_state.sql_query_mode = False


def debug_player_id_names_change() -> None:
    # assign changed selectbox value (debug_player_id_names_selectbox). e.g. ['2663279','Robert Salita']
    player_id_name = st.session_state.debug_player_id_names_selectbox
    change_game_state(player_id_name[0], None)


# Legacy callback aliases - all delegate to generic handler
group_id_on_change = generic_input_on_change
session_id_on_change = generic_input_on_change  
team_id_on_change = generic_input_on_change
simultane_id_on_change = generic_input_on_change
teams_id_on_change = generic_input_on_change
org_id_on_change = generic_input_on_change
player_license_number_on_change = generic_input_on_change


def clear_cache() -> None:
    """Clear all cache files in the cache directory"""
    cache_dir = pathlib.Path(st.session_state.cache_dir)
    if cache_dir.exists():
        cleared_count = 0
        for file in cache_dir.rglob('*'):
            if file.is_file():
                try:
                    file.unlink()
                    cleared_count += 1
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
        st.success(f"Cleared {cleared_count} cache files from {cache_dir}")
    else:
        st.info("Cache directory does not exist")


def filter_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Filter DataFrame to show boards played by specific player and partner
    
    Args:
        df: Input DataFrame containing board data
        
    Returns:
        Filtered DataFrame with additional boolean columns for board filtering
    """

    # Columns used for filtering to a specific player_id and partner_id. Needs multiple with_columns() to unnest overlapping columns.
    full_directions_d = {'N':'north', 'E':'east', 'S':'south', 'W':'west'}
    if f"lineup_{full_directions_d[st.session_state.player_direction]}Player_id" in df.columns:
        df = df.with_columns(
        pl.col(f'lineup_{full_directions_d[st.session_state.player_direction]}Player_id').eq(pl.lit(str(st.session_state.player_id))).alias('Boards_I_Played'), # player_id could be numeric
        pl.col('Declarer_ID').eq(pl.lit(str(st.session_state.player_license_number))).alias('Boards_I_Declared'), # player_id could be numeric
        pl.col('Declarer_ID').eq(pl.lit(str(st.session_state.partner_license_number))).alias('Boards_Partner_Declared'), # partner_id could be numeric
    )
    elif "Pair_Direction" in df.columns:
        # todo: better way to determine Boards_I_Played than above?
        df = df.with_columns(
            pl.col(f'Player_ID_{st.session_state.player_direction}').eq(pl.lit(str(st.session_state.player_id))).alias('Boards_I_Played'), # player_id could be numeric
        )
        df = df.with_columns(
            pl.col('Boards_I_Played').and_(pl.col('Declarer_Direction').eq(st.session_state.player_direction)).alias('Boards_I_Declared'), # player_id could be numeric
            pl.col('Boards_I_Played').and_(pl.col('Declarer_Direction').eq(st.session_state.partner_direction)).alias('Boards_Partner_Declared'), # partner_id could be numeric
        )
    else:
        st.error(f"Unable to match pair to boards.")
    df = df.with_columns(
        pl.col('Boards_I_Played').alias('Boards_We_Played'),
        pl.col('Boards_I_Played').alias('Our_Boards'),
        (pl.col('Boards_I_Declared') | pl.col('Boards_Partner_Declared')).alias('Boards_We_Declared'),
    )
    df = df.with_columns(
        (pl.col('Boards_I_Played') & ~pl.col('Boards_We_Declared') & pl.col('Contract').ne('PASS')).alias('Boards_Opponent_Declared'),
    )

    return df


def extract_group_id_session_id_team_id() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract group ID, session ID, and team ID from session state
    
    Returns:
        Tuple of (group_id, session_id, team_id) - all may be None
    """
    parsed_url = urlparse(st.session_state.game_url)
    #print(f"parsed_url:{parsed_url}")
    path_parts = parsed_url.path.split('/')
    #print(f"path_parts:{path_parts}")

    # Find indices by keywords instead of fixed positions
    if 'groups' in path_parts:
        group_index = path_parts.index('groups') + 1
    else:
        st.error(f"Invalid or missing group in URL: {st.session_state.game_url}")
        return True
    if 'sessions' in path_parts:
        session_index = path_parts.index('sessions') + 1
    else:
        st.error(f"Invalid or missing session in URL: {st.session_state.game_url}")
        return True
    if 'pairs' in path_parts:
        pair_index = path_parts.index('pairs') + 1
    else:
        st.error(f"Invalid or missing pair in URL: {st.session_state.game_url}")
        return True
    #print(f"group_index:{group_index} session_index:{session_index} pair_index:{pair_index}")
    
    extracted_group_id = int(path_parts[group_index])
    extracted_session_id = int(path_parts[session_index])
    extracted_team_id = int(path_parts[pair_index])
    st.session_state.group_id = extracted_group_id
    st.session_state.session_id = extracted_session_id
    st.session_state.team_id = extracted_team_id
    #print(f"extracted_group_id:{extracted_group_id} extracted_session_id:{extracted_session_id} extracted_team_id:{extracted_team_id}")
    return False

from typing import Dict, Any, List
from urllib.parse import urlparse

def create_directory_structure(path: pathlib.Path) -> None:
    """Create directory structure if it doesn't exist"""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def get_path_from_url(url: str) -> pathlib.Path:
    """Extract path from URL and convert to Path object"""
    parsed_url = urlparse(url)
    path = pathlib.Path(parsed_url.path.lstrip('/'))
    return path

def fetch_json(url: str) -> List[Dict[str, Any]]:
    """Fetch JSON data from the specified URL"""
    try:
        headers = {
            "Authorization": f"Bearer {st.session_state.ffbridge_bearer_token}",
            "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.9,fr;q=0.8",
            "origin": "https://www.ffbridge.fr",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        raise


def save_json(data: List[Dict[str, Any]], file_path: pathlib.Path) -> None:
    """Save JSON data to the specified file path"""
    try:
        create_directory_structure(file_path.parent)
        json_file = file_path.with_suffix('.json')
        json_file.write_text(json.dumps(data, indent=2), encoding='utf-8')
        #print(f"\nJSON has been saved to {json_file}")
    except IOError as e:
        print(f"Error saving file: {e}")
        raise


def create_dataframe(data: List[Dict[str, Any]]) -> pl.DataFrame:
    """Create a Polars DataFrame from the JSON data"""
    try:
        # Convert list of dictionaries directly to DataFrame
        df = pl.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        raise


# todo: cache requests
def get_ffbridge_data_using_url_licencie(api_urls_d: Dict[str, Tuple[str, bool]], show_progress: bool = True) -> Tuple[Dict[str, pl.DataFrame], Dict[str, Tuple[str, bool]]]:
    """Fetch FFBridge data using URL configuration dictionary
    
    Args:
        api_urls_d: Dictionary mapping API names to (URL, should_cache) tuples
        
    Returns:
        Tuple of (DataFrames dictionary, API URLs dictionary)
    """
    try:
        
        dfs = {}
        nb_deals = None

        if show_progress:
            # Create progress bar
            total_apis = len(api_urls_d)
            progress_bar = st.progress(0)
            progress_text = st.empty()
        
        for idx, (k, (url, should_cache)) in enumerate(api_urls_d.items()):

            if show_progress:
                # Update progress
                progress = (idx + 1) / total_apis
                progress_bar.progress(progress)
                progress_text.text(f"Processing API {idx + 1}/{total_apis}: {k}")
            
            df = get_df_from_api_url_licencie(k, url, should_cache)
    
            dfs[k] = df
            print(f"dfs[{k}] shape: {dfs[k].shape}")
            print(f"dfs[{k}] columns: {dfs[k].columns}")
            print(f"dfs[{k}]:{dfs[k]}")

        if show_progress:
            # Complete progress bar
            progress_bar.progress(1.0)
            progress_text.text("✅ All API requests completed successfully!")
            
            # Clean up progress indicators after a brief delay
            #time.sleep(1)
            progress_bar.empty()
            progress_text.empty()
    except Exception as e:
        print(f"Error getting ffbridge data using url licencie: {e}")
        raise
    return dfs, api_urls_d


@st.cache_data
def _cached_read_parquet(file_path: str) -> pl.DataFrame:
    """Cached parquet file reader for Streamlit
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        Polars DataFrame containing the parquet data
    """
    return pl.read_parquet(file_path)

def get_df_from_api_url_licencie(k: str, url: str, should_cache: bool) -> pl.DataFrame:
    """Get DataFrame from API URL with optional caching
    
    Args:
        k: API key/name for identification
        url: API URL to fetch data from
        should_cache: Whether to cache the result
        
    Returns:
        Polars DataFrame containing the API response data
    """
    print(f"requesting API: {k}:{url} (cache: {should_cache})")

    # Check for existing parquet cache file first
    from werkzeug.utils import secure_filename
    sanitized_url = secure_filename(url)
    parquet_cache_file = pathlib.Path(st.session_state.cache_dir) / f"{sanitized_url}.parquet"

    if should_cache and parquet_cache_file.exists():
        print(f"Loading {k} from parquet cache: {parquet_cache_file}")
        df = _cached_read_parquet(str(parquet_cache_file))
        
        # Special handling for simultaneous_dealsNumber to set nb_deals
        if k == 'simultaneous_dealsNumber':
            st.session_state.nb_deals = df['nb_deals'][0]
        
        return df  # Skip the match statement and proceed to next iteration

    df = get_df_from_api_name_licencie(k, url)

    # Save to parquet cache if caching is enabled
    if should_cache:
        df.write_parquet(parquet_cache_file)
        print(f"Saved {k} to parquet cache: {parquet_cache_file}")
    
    # Assert that all columns are fully flattened (no List or Struct types remain)
    # todo: use? df.select(pl.col(pl.List, pl.Struct)).is_empty()
    remaining_complex_cols = [(col, df[col].dtype) for col in df.columns if isinstance(df[col].dtype, (pl.List, pl.Struct))]
    print(f"remaining_complex_cols:{remaining_complex_cols}")
    #assert not remaining_complex_cols, f"Found unexploded/unnested columns after cleanup: {remaining_complex_cols}"
    
    return df


def get_df_from_api_name_licencie(k: str, url: str) -> pl.DataFrame:
    """Get DataFrame from API by name with specific processing logic
    
    Args:
        k: API key/name for identification
        url: API URL to fetch data from
        
    Returns:
        Polars DataFrame containing the processed API response data
    """

    # Ensure df is defined even if a branch fails unexpectedly
    df = pl.DataFrame([])

    match k:
        case 'search':
            json_data = make_api_request_licencie(url)
            if json_data is None:
                raise Exception(f"Failed to get data from {url}")
            df = pl.DataFrame(pd.json_normalize(json_data, sep='_'))
        case 'simultaneous_deals':
            json_datas = []
            for i in range(1, st.session_state.nb_deals+1):
                deal_url = url.format(i=i)
                json_data = make_api_request_licencie(deal_url)
                if json_data is None:
                    raise Exception(f"Failed to get data from {deal_url}")
                
                assert isinstance(json_data, dict), f"Expected a dict, got {type(json_data)}"
                json_datas.append(json_data)
            
            df = pl.DataFrame(pd.json_normalize(json_datas, sep='_'))
            for exploded_col_name in ['teams_players_name', 'teams_opponents_name']: #['frequencies', 'frequencies_organizations', 'teams_players_name', 'teams_opponents_name']:
                exploded_col = df.explode(exploded_col_name)
                struct_fields = exploded_col[exploded_col_name].struct.fields
                # Rename struct fields first, then unnest
                df = df.with_columns(
                    pl.col(exploded_col_name).list.eval(
                        pl.element().struct.rename_fields([f"{exploded_col_name}_{field}" for field in struct_fields])
                    )
                ).explode(exploded_col_name).unnest(exploded_col_name)
                #print(df)
        case 'simultaneous_description_by_organization_id':
            json_datas = []
            for i in range(1, st.session_state.nb_deals+1):
                desc_url = url.format(i=i)
                json_data = make_api_request_licencie(desc_url)
                if json_data is None:
                    raise Exception(f"Failed to get data from {desc_url}")
                
                assert isinstance(json_data, list), f"Expected a list, got {type(json_data)}"
        
                # Add Board column to each record
                for record in json_data:
                    record['Board'] = i
                
                json_datas.extend(json_data)
            
            df = pl.DataFrame(pd.json_normalize(json_datas, sep='_'))
        case 'simultaneous_dealsNumber':
            json_data = make_api_request_licencie(url)
            if json_data is None:
                raise Exception(f"Failed to get data from {url}")
            
            assert isinstance(json_data, dict), f"Expected a dict, got {type(json_data)}"
            df = pl.DataFrame(pd.json_normalize(json_data, sep='_'))
            assert len(df) == 1, f"Expected 1 row, got {len(df)}"
            st.session_state.nb_deals = df['nb_deals'][0]
        case 'simultaneous_roadsheets':
            # simultaneous_roadsheets columns:
            # ['roadsheets_deals_contract', 'roadsheets_deals_dealNumber', 'roadsheets_deals_declarant',
            # 'roadsheets_deals_first_card', 'roadsheets_deals_opponentsAvgNote', 'roadsheets_deals_opponentsNote',
            # 'roadsheets_deals_opponentsOrientation', 'roadsheets_deals_opponentsScore', 'roadsheets_deals_result',
            # 'roadsheets_deals_teamAvgNote', 'roadsheets_deals_teamNote', 'roadsheets_deals_teamOrientation',
            # 'roadsheets_deals_teamScore', 'roadsheets_teams_cpt', 'roadsheets_teams_opponents', 'roadsheets_teams_players']
            json_data = make_api_request_licencie(url)
            if json_data is None:
                raise Exception(f"Failed to get data from {url}")
    
            # Create DataFrame from the JSON response. json_data can be a dict or a list.
            df = pl.DataFrame(pd.json_normalize(json_data, sep='_'))
            
            # Get the struct fields and rename them before unnesting
            exploded_col = df.explode('roadsheets') # https://api.ffbridge.fr/api/v1/simultaneous-tournaments/32178/teams/4230171/roadsheets
            struct_fields = exploded_col['roadsheets'].struct.fields
            
            # Rename struct fields first, then unnest
            df = df.with_columns(
                pl.col('roadsheets').list.eval(
                    pl.element().struct.rename_fields([f"roadsheets_{field}" for field in struct_fields])
                )
            ).explode('roadsheets').unnest('roadsheets')
            
            # Continue with deals if present
            if 'roadsheets_deals' in df.columns:
                struct_fields = df.explode('roadsheets_deals')['roadsheets_deals'].struct.fields
                df = df.with_columns(
                    pl.col('roadsheets_deals').list.eval(
                        pl.element().struct.rename_fields([f"roadsheets_deals_{field}" for field in struct_fields])
                    )
                ).explode('roadsheets_deals').unnest('roadsheets_deals')
            
            # Continue with teams if present
            if 'roadsheets_teams' in df.columns:
                struct_fields = df['roadsheets_teams'].struct.fields
                df = df.with_columns(
                    pl.col('roadsheets_teams').struct.rename_fields([f"roadsheets_teams_{field}" for field in struct_fields])
                ).unnest('roadsheets_teams')

            # Create horizontal columns for player names by orientation
            assert 'roadsheets_teams_players' in df.columns, f"roadsheets_teams_players not found in df"
            assert 'roadsheets_teams_opponents' in df.columns, f"roadsheets_teams_opponents not found in df"
            
            df = df.with_columns([
                pl.col('roadsheets_deals_teamOrientation').str.replace('EO', 'EW') # translate French EO to English EW
            ])
            # df = df.with_columns([
            #     pl.when(pl.col('roadsheets_deals_teamOrientation') == pair_direction)
            #     .then(pl.col('roadsheets_teams_players').list.get(player_index))
            #     .otherwise(pl.col('roadsheets_teams_opponents').list.get(player_index))
            #     .alias(f'roadsheets_player_{pair_direction[player_index].lower()}')
            #     for player_index,pair_direction in [(0,'NS'),(1,'NS'),(0,'EW'),(1,'EW')]
            # ]).drop(['roadsheets_teams_players', 'roadsheets_teams_opponents'])
        case 'simultaneous_tournaments' | 'simultaneous_tournaments_by_organization_id':
            # simultaneous_tournaments columns:
            # ['id', 'label', 'startDate', 'endDate', 'teams', 'simultaneous_id', 'simultaneous_label', 'simultaneous_startDate',
            # 'simultaneous_endDate', 'simultaneous_teams', 'simultaneous_simultaneous_id', 'simultaneous_simultaneous_label',
            # 'simultaneous_simultaneous_startDate', 'simultaneous_simultaneous_endDate', 'simultaneous_simultaneous_teams']
            json_data = make_api_request_licencie(url)
            if json_data is None:
                raise Exception(f"Failed to get data from {url}")
    
            # Create DataFrame from the JSON response. json_data can be a dict or a list.
            df = pl.DataFrame(pd.json_normalize(json_data, sep='_'))
            # todo: at least one game doesn't have an 'id' to rename. https://api.ffbridge.fr/api/v1/simultaneous-tournaments/2991057
            df = df.rename({'id': 'simultane_id'})
            
            # Explode to get individual structs, then get struct fields  
            exploded_col = df.explode('teams')
            struct_fields = exploded_col['teams'].struct.fields
            
            # Rename struct fields first, then unnest
            df = df.with_columns(
                pl.col('teams').list.eval(
                    pl.element().struct.rename_fields([f"team_{field}" for field in struct_fields])
                )
            ).explode('teams').unnest('teams')
            
            df = df.with_columns([
                pl.col('team_orientation').str.replace('EO', 'EW') # translate French EO to English EW
            ])

            # Unnest team_organization if it exists
            if 'team_organization' in df.columns:
                # Rename struct fields first to avoid conflicts
                struct_fields = df['team_organization'].struct.fields
                df = df.with_columns(
                    pl.col('team_organization').struct.rename_fields([f"team_organization_{field}" for field in struct_fields])
                ).unnest('team_organization')
            
            # Explode and unnest team_players if it exists
            if 'team_players' in df.columns:
                # Rename struct fields first to avoid conflicts
                struct_fields = df.explode('team_players')['team_players'].struct.fields
                df = df.with_columns(
                    pl.col('team_players').list.eval(
                        pl.element().struct.rename_fields([f"team_players_{field}" for field in struct_fields])
                    )
                ).explode('team_players').unnest('team_players')
                # todo: split team_players into player and partner using similar logic to below.
                # df = df.with_columns([
                #     pl.when(pl.col('roadsheets_deals_teamOrientation') == pair_direction)
                #     .then(pl.col('roadsheets_teams_players').list.get(player_index))
                #     .otherwise(pl.col('roadsheets_teams_opponents').list.get(player_index))
                #     .alias(f'roadsheets_player_{pair_direction[player_index].lower()}')
                #     for player_index,pair_direction in [(0,'NS'),(1,'NS'),(0,'EW'),(1,'EW')]
                # ]).drop(['roadsheets_teams_players', 'roadsheets_teams_opponents'])
        case 'members':
            # Members data changes frequently, so use should_cache flag to control caching
            json_data = make_api_request_licencie(url)
            if json_data is None:
                raise Exception(f"Failed to get data from {url}")
    
            # Create DataFrame from the JSON response. json_data can be a dict or a list.
            df = pl.DataFrame(pd.json_normalize(json_data, sep='_'))
            # season is list[struct[7]]
            # regularity_tournament_points is list[struct[7]]
            for exploded_col_name in ['seasons', 'regularity_tournament_points']:
                # Check if column exists and contains struct data
                if exploded_col_name in df.columns:
                    # Check if the column contains any non-null struct data
                    non_null_data = df.filter(pl.col(exploded_col_name).is_not_null())
                    if len(non_null_data) > 0:
                        try:
                            exploded_col = df.explode(exploded_col_name)
                            # Filter out null values before getting struct fields
                            non_null_exploded = exploded_col.filter(pl.col(exploded_col_name).is_not_null())
                            if len(non_null_exploded) > 0:
                                struct_fields = non_null_exploded[exploded_col_name].struct.fields
                                # Rename struct fields first, then unnest
                                df = df.with_columns(
                                    pl.col(exploded_col_name).list.eval(
                                        pl.element().struct.rename_fields([f"{exploded_col_name}_{field}" for field in struct_fields])
                                    )
                                ).explode(exploded_col_name).unnest(exploded_col_name)
                            else:
                                print(f"⚠️ Column '{exploded_col_name}' contains only null values, skipping struct processing")
                        except Exception as e:
                            print(f"⚠️ Error processing column '{exploded_col_name}': {e}. Skipping struct processing.")
                    else:
                        print(f"⚠️ Column '{exploded_col_name}' is empty or all null, skipping struct processing")
                else:
                    print(f"⚠️ Column '{exploded_col_name}' not found in DataFrame, skipping")
        case _:
            json_data = make_api_request_licencie(url)
            if json_data is None:
                raise Exception(f"Failed to get data from {url} possibly due to data not yet available. Try again in 24 hours.")
    
            # Create DataFrame from the JSON response. json_data can be a dict or a list.
            try:
                df = pl.DataFrame(pd.json_normalize(json_data, sep='_'))
            except Exception as e:
                raise Exception(f"Failed to create DataFrame from {url} possibly due to data not yet available. Try again in 24 hours. {e}")
            if 'functions' in df.columns: # my_infos['functions'] is a list of null. ignore it.
                df = df.drop('functions')
    # Handle any remaining List or Struct columns that couldn't be processed
    unprocessed_cols = [(col, df[col].dtype) for col in df.columns if isinstance(df[col].dtype, (pl.List, pl.Struct))]
    if unprocessed_cols:
        print(f"unprocessed_cols:{unprocessed_cols}")
        # print(f"⚠️ Converting {len(unprocessed_cols)} unprocessed List/Struct columns to null columns: {[col for col, dtype in unprocessed_cols]}")
        # for col, dtype in unprocessed_cols:
        #     # Convert List(Null) or problematic Struct columns to simple null columns
        #     df = df.with_columns(pl.lit(None).alias(col))
    return df


# todo: clean up this function. use get_ffbridge_date_using_url_licencie() as a template (match statement, cache handling).
def get_ffbridge_lancelot_data_using_url():

    try:

        # if extract_group_id_session_id_team_id():
        #    return None

        # games_urls from https://api-lancelot.ffbridge.fr/results/search/me?currentPage=1 items[0]->session->id->199238 and items[0]->group->phase->stade->organization->ffbCode->5802079
        # https://api-lancelot.ffbridge.fr/results/sessions/199238/ranking?simultaneousId=5802079 returns a list of teams. search team->player[1-8] for matching "license_number": 9500754. parent will have team->id->9159276
        # https://api-lancelot.ffbridge.fr/results/teams/9159276 will return team info.
        # https://api-lancelot.ffbridge.fr/results/teams/9159276/session/199238/scores will return hand record (deal, dds, ...) and board results.

        api_url_file_d = {
            'my_ranking': (f"https://api-lancelot.ffbridge.fr/results/sessions/201337/ranking/{st.session_state.player_id}", f"my_ranking/{st.session_state.player_id}"),
            'games': (f"https://api-lancelot.ffbridge.fr/results/search/me?currentPage=1", f"games/{st.session_state.player_id}"),
            'group_url': (f"https://api-lancelot.ffbridge.fr/competitions/groups/{st.session_state.group_id}?context%5B%5D=result_status&context%5B%5D=result_data", f"groups/{st.session_state.group_id}"), # gets group data but no results
            #'team_url': f"https://api-lancelot.ffbridge.fr/results/teams/{st.session_state.team_id}", # gets team data but no results
            #'me': (https://api-lancelot.ffbridge.fr/persons/me)
            'session_url': (f"https://api-lancelot.ffbridge.fr/competitions/sessions/{st.session_state.session_id}", f"sessions/{st.session_state.session_id}"), # gets session data but no results
            'ranking_url': (f"https://api-lancelot.ffbridge.fr/results/sessions/{st.session_state.session_id}/ranking", f"rankings/{st.session_state.session_id}"), # gets ranking data but no results
        }
        #api_url = f"https://api-lancelot.ffbridge.fr/results/teams/{extracted_team_id}/session/{extracted_session_id}/scores"
        #api_url = f'https://api-lancelot.ffbridge.fr/competitions/results/groups/{extracted_group_id}/sessions/{extracted_session_id}/pairs/{extracted_team_id}'
        response_jsons = {}
        dfs = {}
        for k,(v,cache_path) in api_url_file_d.items():
            print(f"requesting {k}:{v}:{cache_path}")
            file_path = pathlib.Path(st.session_state.cache_dir).joinpath(cache_path)
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    response_jsons[k] = json.load(f)
            else:
                response_jsons[k] = fetch_json(v)
                # Check if the request was successful and extract the data
                if 'success' in response_jsons[k] and not response_jsons[k].get('success'):
                    raise ValueError(f"API request failed: {response_jsons[k]}")
                save_json(response_jsons[k], file_path)
            match k:
                case 'my_ranking':
                    dfs[k] = pl.DataFrame(pd.json_normalize(response_jsons[k],sep='_'))
                    print(f"my_ranking:{dfs[k]}")
                case 'games':
                    dfs[k] = pl.DataFrame(pd.json_normalize(response_jsons[k],sep='_'))
                    items = dfs[k].explode('items').unnest('items')
                    groups = items['group'].to_frame().unnest('group')
                    print(f"groups:{groups}")
                    phases = groups['phase'].to_frame().unnest('phase')
                    print(f"phases:{phases}")
                    stades = phases['stade'].to_frame().unnest('stade')
                    print(f"stades:{stades}")
                    organizations = stades['organization'].to_frame().unnest('organization')
                    print(f"organizations:{organizations}")
                    sessions = items['session'].to_frame().unnest('session')
                    print(f"sessions:{sessions}")
                    sessions = sessions.explode('organizations')
                    print(f"organizations:{sessions}")
                    simultaneous = sessions['simultaneous'].to_frame().unnest('simultaneous')
                    print(f"simultaneous:{simultaneous}")
                    # st.session_state.player_id = dfs[k]['player_id'].to_list()[0]
                    # st.session_state.group_id = dfs[k]['group_id'].to_list()[0]
                    # st.session_state.session_id = dfs[k]['session_id'].to_list()[0]
                    # st.session_state.team_id = dfs[k]['team_id'].to_list()[0]
                    return dfs, api_url_file_d
                case '_':
                    dfs[k] = pl.DataFrame(pd.json_normalize(response_jsons[k],sep='_'))
            print(f"dfs[{k}]:{dfs[k].columns}")
            print(f"dfs[{k}]:{dfs[k].shape}")
            print(f"dfs[{k}]:{dfs[k]}")

            if st.session_state.debug_mode:
                st.caption(f"{k}:{api_url_file_d[k][0]} Shape:{dfs[k].shape}")
                st.dataframe(dfs[k],selection_mode='single-row')

        group_df = dfs['group_url']
        session_df = dfs['session_url']
        teams_df = dfs['ranking_url']

        assert len(group_df) == 1, f"Expected 1 row, got {len(group_df)}"
        group_d = group_df.to_dicts()[0]
        assert len(session_df) == 1, f"Expected 1 row, got {len(session_df)}"
        session_d = session_df.to_dicts()[0]

        teams_df = teams_df.with_columns(
            pl.col(f'team_player1_id').cast(pl.Int64),
            pl.col(f'team_player1_license_number').cast(pl.Int64),
            pl.col(f'team_player2_id').cast(pl.Int64),
            pl.col(f'team_player2_license_number').cast(pl.Int64),
        )

        print(f"Unnested teams_df columns: {teams_df.columns}")
        print(f"Unnested teams_df shape: {teams_df.shape}")

        same_players_in_player1_and_player2 = set(teams_df['team_player1_id']).intersection(set(teams_df['team_player2_id']))
        if same_players_in_player1_and_player2:
            print(f"same_players_in_player1_and_player2:{same_players_in_player1_and_player2}")
            print(teams_df.filter(pl.col('team_player1_id').is_in(same_players_in_player1_and_player2))['team_player1_lastName'])
            # Remove rows where either player1_id or player2_id is in the overlapping set
            # not sure if this situation is handled optimally. https://ffbridge.fr/competitions/results/groups/8199/sessions/195371/pairs/9051691
            teams_df = teams_df.filter(
                ~(pl.col('team_player1_id').is_in(same_players_in_player1_and_player2) | 
                pl.col('team_player2_id').is_in(same_players_in_player1_and_player2))
            )

        # Get the column values
        player1_dict = teams_df.drop_nulls('team_player1_license_number').select(
            'team_player1_license_number', 'team_player1_id'
        ).unique().to_dict(as_series=False)

        # Convert to the proper format with keys and values swapped
        player1_id_to_license_number_dict = dict(zip(
            player1_dict['team_player1_id'],      # Now using IDs as keys
            player1_dict['team_player1_license_number']    # And FFB IDs as values
        ))

        # Same for player2
        player2_dict = teams_df.drop_nulls('team_player2_license_number').select(
            'team_player2_license_number', 'team_player2_id'
        ).unique().to_dict(as_series=False)

        player2_id_to_license_number_dict = dict(zip(
            player2_dict['team_player2_id'],      # Now using IDs as keys
            player2_dict['team_player2_license_number']    # And FFB IDs as values
        ))

        # Check if they're disjoint
        assert set(player1_id_to_license_number_dict.keys()).isdisjoint(set(player2_id_to_license_number_dict.keys()))

        # Merge the dictionaries
        id_to_license_number_dict = {**player1_id_to_license_number_dict, **player2_id_to_license_number_dict}
        print(f"id_to_license_number_dict:{id_to_license_number_dict}")

        # Process each team to get their scores
        teams_jsons = []
        assert not teams_df['team_id'].is_duplicated().all(), f"teams_df['team_id'] has duplicates: {teams_df['team_id']}"
        for team_id in stqdm(teams_df['team_id'], desc='Downloading team scores...'):
            #print(f"Processing team_id: {team_id}")
            api_url_file_d = {
                f"team_id:{team_id}": (f"https://api-lancelot.ffbridge.fr/results/teams/{team_id}/session/{st.session_state.session_id}/scores", f"scores/{team_id}_{st.session_state.session_id}.json"),
            }
            for k,(v,cache_path) in api_url_file_d.items():
                print(f"requesting {k}:{v}:{cache_path}")
                file_path = pathlib.Path(st.session_state.cache_dir).joinpath(cache_path)
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        request_json = json.load(f)
                else:
                    request_json = fetch_json(v)
                    # Check if the request was successful and extract the data
                    if 'success' in request_json and not request_json.get('success'):
                        raise ValueError(f"API request failed: {request_json}")
                    save_json(request_json, file_path)
                if all([b['contract'] == '' for b in request_json]):
                    if team_id == st.session_state.team_id:
                        raise ValueError(f"Game is missing contract data for selected team. Fatal error. {st.session_state.session_id} {session_d['label']} {v}")
                    else:
                        print(f"Game is missing contract data for team:{team_id} {session_d['label']} {v}")
                        continue
                teams_jsons.extend(request_json)

        df = pl.DataFrame(pd.json_normalize(teams_jsons,sep='_'))
        # teams_dfs = {}  
        # for k,v in teams_jsons.items():
        #     teams_dfs[k] = pl.DataFrame(pd.json_normalize(v,sep='_')) #pl.DataFrame(v) #.drop(['eastPlayer'])

        # dtypes_df = create_df_of_dtypes(teams_dfs)
        # print(f"dtypes_df:{dtypes_df}")

        # for col in dtypes_df.columns:
        #     print(f"{col}:{dtypes_df[col].value_counts()}")

        # # need to unnest any structs but before must rename to avoid conflicts
        # cols = None
        # for k,v in teams_dfs.items():
        #     if cols is None:
        #         cols = v.columns
        #     else:
        #         assert set(cols) == set(v.columns), set(cols)-set(v.columns)
        
        # all_pairs_dfs = {}
        # for k,v in teams_dfs.items():
        # for col in teams_dfs.items():
        #     print(f"col:{col}")
        #     if col == 'lineup': # lineup is a struct of varying types (string vs None)
        #         #continue
        #         print('bypassing lineup') # error: https://ffbridge.fr/competitions/results/groups/7878/sessions/183872/pairs/8413302
        #         ldf = []
        #         for k,v in teams_dfs.items():
        #             df = teams_dfs[k] # unnest_structs_recursive(pl.DataFrame(teams_dfs[k])['lineup'].to_frame())
        #             ldf.append(df)
        #         all_pairs_dfs[col] = concat_with_schema_unification(ldf)
        #         continue
        #     all_pairs_dfs[col] = pl.concat([v[col] for k,v in teams_dfs.items()]).to_frame()
        #     pass

        # # Print the structure to debug
        # df = pl.concat(all_pairs_dfs.values(),how='horizontal')
        # print("DataFrame columns:", df.columns)
        # df = unnest_structs_recursive(df)
        # #ShowDataFrameTable(df, key='team_and_session_df')


        #st.session_state.group_id = st.session_state.group_id # same
        df = df.with_columns(
            pl.lit(st.session_state.group_id).alias('group_id'),
            pl.lit(st.session_state.session_id).alias('session_id'),
            #pl.lit(st.session_state.team_id).alias('team_id'),
        )
        df = df.unique(keep='first') # remove duplicated rows. Caused by two players being in partnership (one row per player)?

        for direction in ['north', 'east', 'south', 'west']:
            df = df.with_columns([
                pl.col(f'lineup_{direction}Player_id')
                .map_elements(lambda x: id_to_license_number_dict.get(x, None),return_dtype=pl.Int64)
                .alias(f'lineup_{direction}Player_license_number')
            ])
            #assert df[f'lineup_{direction}Player_license_number'].is_not_null().all() # could be None for sitout

        # extract a df of only the requested team_id. must be a single row.
        # using [0] because all rows of team data have identical values.
        team_df = teams_df.filter(pl.col('team_id').eq(st.session_state.team_id))
        assert len(team_df) == 1, f"Expected 1 row, got {len(team_df)}"
        team_d = team_df.to_dicts()[0]
        # Update the session state
        st.session_state.player_id = team_d['team_player1_id']
        st.session_state.partner_id = team_d['team_player2_id']
        st.session_state.player_license_number = team_d['team_player1_license_number']
        st.session_state.partner_license_number = team_d['team_player2_license_number']
        st.session_state.player_name = team_d['team_player1_firstName'] + ' ' + team_d['team_player1_lastName']
        st.session_state.partner_name = team_d['team_player2_firstName'] + ' ' + team_d['team_player2_lastName']
        st.session_state.pair_direction = team_d['orientation']
        st.session_state.player_direction = st.session_state.pair_direction[0]
        st.session_state.partner_direction = st.session_state.pair_direction[1]
        st.session_state.opponent_pair_direction = 'EW' if st.session_state.pair_direction == 'NS' else 'NS' # opposite of pair_direction
        st.session_state.section_name = team_d['section']
        st.session_state.organization_name = group_d['phase_stade_organization_name']
        st.session_state.game_description = group_d['phase_stade_competitionDivision_competition_label']

        print(f"st.session_state.group_id:{st.session_state.group_id} st.session_state.session_id:{st.session_state.session_id} st.session_state.team_id:{st.session_state.team_id} st.session_state.player_id:{st.session_state.player_id} st.session_state.partner_id:{st.session_state.partner_id} st.session_state.player_direction:{st.session_state.player_direction} st.session_state.partner_direction:{st.session_state.partner_direction} st.session_state.opponent_pair_direction:{st.session_state.opponent_pair_direction}")

    except Exception as e:
        st.error(f"Error getting team or scores data: {e}")
        # todo: should be replying on reset_game_data() to set defaults.
        st.session_state.group_id = st.session_state.group_id_default
        st.session_state.session_id = st.session_state.session_id_default
        st.session_state.team_id = st.session_state.team_id_default
        st.session_state.player_id = st.session_state.player_id_default
        st.session_state.player_license_number = st.session_state.player_license_number_default
        st.session_state.partner_id = st.session_state.partner_id_default
        st.session_state.partner_license_number = st.session_state.partner_license_number_default
        st.session_state.player_name = st.session_state.player_name_default
        st.session_state.partner_name = st.session_state.partner_name_default
        st.session_state.player_direction = st.session_state.player_direction_default
        st.session_state.partner_direction = st.session_state.partner_direction_default
        st.session_state.opponent_pair_direction = st.session_state.opponent_pair_direction_default
        st.session_state.section_name = st.session_state.section_name_default
        st.session_state.organization_name = st.session_state.organization_name_default
        st.session_state.game_description = st.session_state.game_description_default
        return None, None

    #st.session_state.df = df
    return {'games': df}, api_url_file_d


def get_ffbridge_licencie_get_urls(api_urls_d: Dict[str, Tuple[str, bool]]) -> Tuple[Dict[str, pl.DataFrame], Dict[str, Tuple[str, bool]]]:
    """Get FFBridge data using URL configuration and display results
    
    Args:
        api_urls_d: Dictionary mapping API names to (URL, should_cache) tuples
        
    Returns:
        Tuple of (DataFrames dictionary, API URLs dictionary)
    """

    dfs, api_urls_d = get_ffbridge_data_using_url_licencie(api_urls_d)

    if st.session_state.debug_mode:
        for k,v in dfs.items():
            st.caption(f"{k}:{api_urls_d[k][0]} Shape:{dfs[k].shape}")  # Use the URL part of the tuple
            st.dataframe(v,selection_mode='single-row')

    return dfs, api_urls_d


def populate_game_urls_for_player(player_id: str) -> bool:
    """Populate st.session_state.game_urls_d for a given player without changing session_id.
    Returns True if games were populated (length > 0), False otherwise.
    """
    if player_id in st.session_state.game_urls_d and st.session_state.game_urls_d[player_id]:
        return True
    api_urls_d = {
        'members': (f"https://api.ffbridge.fr/api/v1/members/{player_id}", False),
        'person': (f"https://api.ffbridge.fr/api/v1/licensee-results/results/person/{player_id}?date=all&place=0&type=0", False),
    }
    try:
        dfs, _ = get_ffbridge_licencie_get_urls(api_urls_d)
        if 'tournament_id' in dfs['person'].columns:
            st.session_state.game_urls_d[player_id] = {k: v for k, v in zip(dfs['person']['tournament_id'], dfs['person'].to_dicts())}
        else:
            if 'id' in dfs['person'].columns:
                st.session_state.game_urls_d[player_id] = {k: v for k, v in zip(dfs['person']['id'], dfs['person'].to_dicts())}
            elif len(dfs['person']) > 0:
                st.session_state.game_urls_d[player_id] = {i: v for i, v in enumerate(dfs['person'].to_dicts())}
            else:
                # Preserve any existing cache; otherwise leave empty
                if not (player_id in st.session_state.game_urls_d and st.session_state.game_urls_d[player_id]):
                    st.session_state.game_urls_d[player_id] = {}
        st.session_state.person_organization_id = _derive_person_organization_id_scalar(dfs['members'])
        return len(st.session_state.game_urls_d.get(player_id, {})) > 0
    except Exception:
        # Retry once
        try:
            dfs, _ = get_ffbridge_licencie_get_urls(api_urls_d)
            if 'tournament_id' in dfs['person'].columns:
                st.session_state.game_urls_d[player_id] = {k: v for k, v in zip(dfs['person']['tournament_id'], dfs['person'].to_dicts())}
            else:
                if 'id' in dfs['person'].columns:
                    st.session_state.game_urls_d[player_id] = {k: v for k, v in zip(dfs['person']['id'], dfs['person'].to_dicts())}
                elif len(dfs['person']) > 0:
                    st.session_state.game_urls_d[player_id] = {i: v for i, v in enumerate(dfs['person'].to_dicts())}
                else:
                    if not (player_id in st.session_state.game_urls_d and st.session_state.game_urls_d[player_id]):
                        st.session_state.game_urls_d[player_id] = {}
            st.session_state.person_organization_id = dfs['members']['seasons_organization_id']
            return len(st.session_state.game_urls_d.get(player_id, {})) > 0
        except Exception as e2:
            # Keep cached games if present
            return len(st.session_state.game_urls_d.get(player_id, {})) > 0


def change_game_state(player_id: str, session_id: str) -> None: # todo: rename to session_id?

    print(f"=== change_game_state START: player_id={player_id}, session_id={session_id} ===")

    st.markdown('<div style="height: 50px;"><a name="top-of-report"></a></div>', unsafe_allow_html=True)

    con = get_session_duckdb_connection()

    with st.spinner(f"Retrieving a list of games for {player_id} ..."):
        t = time.time()
        if player_id not in st.session_state.game_urls_d:
            if False:
                dfs, api_urls_d = get_ffbridge_lancelot_data_using_url()
                assert False, "todo: implement next line"
                st.session_state.game_urls_d[player_id] = {k:v for k,v in zip(dfs['games']['items'], dfs['person'].to_dicts())}
            else:
                api_urls_d = {
                    'members': (f"https://api.ffbridge.fr/api/v1/members/{player_id}", False),
                    'person': (f"https://api.ffbridge.fr/api/v1/licensee-results/results/person/{player_id}?date=all&place=0&type=0", False),
                }
                try:
                    dfs, api_urls_d = get_ffbridge_licencie_get_urls(api_urls_d)
                    
                    # Debug: Check what columns are available in the person DataFrame
                    if st.session_state.get('debug_mode', False):
                        print(f"Person DataFrame columns: {dfs['person'].columns}")
                        print(f"Person DataFrame shape: {dfs['person'].shape}")
                    
                    # Handle missing tournament_id column gracefully
                    if 'tournament_id' in dfs['person'].columns:
                        st.session_state.game_urls_d[player_id] = {k:v for k,v in zip(dfs['person']['tournament_id'], dfs['person'].to_dicts())}
                    else:
                        # Try alternative column names or use a different approach
                        if 'id' in dfs['person'].columns:
                            st.session_state.game_urls_d[player_id] = {k:v for k,v in zip(dfs['person']['id'], dfs['person'].to_dicts())}
                        elif len(dfs['person']) > 0:
                            # Use row index as key if no suitable ID column found
                            st.session_state.game_urls_d[player_id] = {i:v for i,v in enumerate(dfs['person'].to_dicts())}
                        else:
                            # Preserve existing cached games if present
                            if player_id in st.session_state.game_urls_d and st.session_state.game_urls_d[player_id]:
                                st.warning("Using cached games due to empty results.")
                            else:
                                st.session_state.game_urls_d[player_id] = {}
                    
                    # Derive a single person_organization_id scalar (not a Series)
                    st.session_state.person_organization_id = _derive_person_organization_id_scalar(dfs['members'])
                    
                except Exception as e:
                    # Retry once before falling back to cache
                    try:
                        dfs, api_urls_d = get_ffbridge_licencie_get_urls(api_urls_d)
                        if 'tournament_id' in dfs['person'].columns:
                            st.session_state.game_urls_d[player_id] = {k:v for k,v in zip(dfs['person']['tournament_id'], dfs['person'].to_dicts())}
                        else:
                            if 'id' in dfs['person'].columns:
                                st.session_state.game_urls_d[player_id] = {k:v for k,v in zip(dfs['person']['id'], dfs['person'].to_dicts())}
                            elif len(dfs['person']) > 0:
                                st.session_state.game_urls_d[player_id] = {i:v for i,v in enumerate(dfs['person'].to_dicts())}
                            else:
                                if player_id in st.session_state.game_urls_d and st.session_state.game_urls_d[player_id]:
                                    st.warning("Using cached games due to empty results.")
                                else:
                                    st.session_state.game_urls_d[player_id] = {}
                        st.session_state.person_organization_id = _derive_person_organization_id_scalar(dfs['members'])
                    except Exception as e2:
                        print(f"Error loading player data for {player_id}: {e2}")
                        st.error(f"Error loading player data: {str(e2)}")
                        # Only clear if no cache exists; otherwise, keep cached games
                        if not (player_id in st.session_state.game_urls_d and st.session_state.game_urls_d[player_id]):
                            st.session_state.game_urls_d[player_id] = {}
                            return True  # Only signal error if we have nothing to show
        game_urls = st.session_state.game_urls_d[player_id]
        if game_urls is None:
            st.error(f"Player number {player_id} not found.")
            return True  # Return True to indicate error
        if len(game_urls) == 0:
            st.error(f"Could not find any games for {player_id}.")
            return True  # Return error if no games found
        elif session_id is None:
            iterator = iter(game_urls)
            #next(iterator)  # Skip first
            session_id = next(iterator)  # Get second
            #session_id = next(iter(game_urls))  # default to most recent club game
        st.session_state.player_id = player_id
        print(f"session_id:{session_id}")
        st.session_state.session_id = session_id
        print(f"st.session_state.session_id:{st.session_state.session_id}")
        st.session_state.simultane_id = session_id
        st.session_state.org_id = game_urls[session_id]['organization_id']
        api_urls_d = {
            'simultaneous_tournaments': (f"https://api.ffbridge.fr/api/v1/simultaneous-tournaments/{st.session_state.simultane_id}", False),
        }
        dfs, api_urls_d = get_ffbridge_licencie_get_urls(api_urls_d)
        simultaneous_tournaments_df = dfs['simultaneous_tournaments']
        st.session_state.player_row = simultaneous_tournaments_df.filter(
            pl.col('team_players_id').cast(pl.Int64) == int(st.session_state.player_id)
        )
        if st.session_state.debug_mode:
            st.caption(f"player_row")
            st.dataframe(st.session_state.player_row,selection_mode='single-row')
        st.session_state.player_id = st.session_state.player_row['team_players_id'].first()
        st.session_state.player_license_number = st.session_state.player_row['team_players_license_number'].str.strip_chars_start('0').first()
        st.session_state.pair_direction = st.session_state.player_row['team_orientation'].first()        
        st.session_state.opponent_pair_direction = 'EW' if st.session_state.pair_direction == 'NS' else 'NS' # opposite of pair_direction
        st.session_state.player_position = 0 if st.session_state.player_row['team_players_position'].first() == 1 else 1
        st.session_state.partner_position = 0 if st.session_state.player_position == 1 else 1
        st.session_state.player_direction = st.session_state.pair_direction[st.session_state.player_position]
        st.session_state.partner_direction = st.session_state.pair_direction[st.session_state.partner_position]
        st.session_state.team_id = st.session_state.player_row['team_id'].first()
        st.session_state.section_name = st.session_state.player_row['team_section_name'].first()
        st.session_state.simultaneeCode = st.session_state.player_row['simultaneeCode'].first()
        st.session_state.organization_code = st.session_state.player_row['team_organization_code'].first()
        st.session_state.organization_name = st.session_state.player_row['team_organization_name'].first()
        st.session_state.tournament_date = datetime.fromisoformat(st.session_state.player_row['date'].first()).strftime('%Y-%m-%d')
        st.session_state.game_description = st.session_state.player_row['name'].first()
        st.session_state.player_name = st.session_state.player_row['team_players_firstname'].first() + ' ' + st.session_state.player_row['team_players_lastname'].first()
        st.session_state.game_url = f"https://licencie.ffbridge.fr/#/resultats/simultane/{st.session_state.simultane_id}/details/{st.session_state.team_id}?orgId={st.session_state.org_id}"
        st.session_state.team_number = st.session_state.player_row['team_table_number'].first()
        # find same team_id but partner_position
        st.session_state.partner_row = simultaneous_tournaments_df.filter(
            pl.col('team_id').eq(st.session_state.team_id) &
            pl.col('team_players_position').eq(st.session_state.partner_position+1)
        )
        if st.session_state.debug_mode:
            st.caption(f"partner_row")
            st.dataframe(st.session_state.partner_row,selection_mode='single-row')
        # might need more partner info?
        st.session_state.partner_id = st.session_state.partner_row['team_players_id'].first()
        st.session_state.partner_license_number = st.session_state.partner_row['team_players_license_number'].first()
        st.session_state.partner_name = st.session_state.partner_row['team_players_firstname'].first() + ' ' + st.session_state.partner_row['team_players_lastname'].first()
        print('get_ffbridge_results_from_player_number time:', time.time()-t) # takes 4s

    with st.spinner(f'Preparing Bridge Game Postmortem Report...'):
        # Use the entered URL or fallback to default.
        #st.session_state.game_url = st.session_state.game_url_input.strip()
        #if st.session_state.game_url is None or st.session_state.game_url.strip() == "":
        #    return True

        # Fetch initial data using the URL.
        # if (st.session_state.game_url.startswith('https://ffbridge.fr') or 
        #     st.session_state.game_url.startswith('https://www.ffbridge.fr')):
        #     df = get_ffbridge_data_using_url()
        #     df = ffbridgelib.convert_ffdf_api_to_mldf(df) # warning: drops columns from df.
        # elif st.session_state.game_url.startswith('https://licencie.ffbridge.fr'):
        # Use the API endpoint instead of the web page
        # api_urls values are tuples of (url, should_cache) where should_cache=False means always request fresh data
        api_urls_d = {
            'roadsheets': (f"https://api.ffbridge.fr/api/v1/simultaneous-tournaments/{st.session_state.simultane_id}/teams/{st.session_state.team_id}/roadsheets", False),
            'simultaneous_roadsheets': (f"https://api.ffbridge.fr/api/v1/simultaneous-tournaments/{st.session_state.simultane_id}/teams/{st.session_state.team_id}/roadsheets", False),
            'simultaneous_dealsNumber': (f"https://api.ffbridge.fr/api/v1/simultaneous-tournaments/{st.session_state.simultane_id}/teams/{st.session_state.team_id}/dealsNumber", False),
            'simultaneous_deals': (f"https://api.ffbridge.fr/api/v1/simultaneous-tournaments/{st.session_state.simultane_id}/teams/{st.session_state.team_id}/deals/{{i}}", False),
            #'simultaneous_descriptions': (f"https://api.ffbridge.fr/api/v1/simultaneous-tournaments/{st.session_state.simultane_id}/teams/{st.session_state.team_id}/deals/{{i}}/descriptions", False),
            'simultaneous_description_by_organization_id': (f"https://api.ffbridge.fr/api/v1/simultaneous/{st.session_state.simultane_id}/deals/{{i}}/descriptions?organization_id={st.session_state.org_id}", False),
            'simultaneous_tournaments_by_organization_id': (f"https://api.ffbridge.fr/api/v1/simultaneous-tournaments/{st.session_state.simultane_id}?organization_id={st.session_state.org_id}", False),
            'my_infos': (f"https://api.ffbridge.fr/api/v1/users/my/infos", False),
            'members': (f"https://api.ffbridge.fr/api/v1/members/{st.session_state.player_id}", False),
            'person': (f"https://api.ffbridge.fr/api/v1/licensee-results/results/person/{st.session_state.player_id}?date=all&place=0&type=0", False),
            'organization_by_person_organization_id': (f"https://api.ffbridge.fr/api/v1/licensee-results/results/organization/{st.session_state.org_id}?date=all&person_organization_id={str(st.session_state.person_organization_id or '')}&place=0&type=0", False),
            'person_by_person_organization_id': (f"https://api.ffbridge.fr/api/v1/licensee-results/results/person/{st.session_state.player_id}?date=all&person_organization_id={str(st.session_state.person_organization_id or '')}&place=0&type=0", False),
        }
        dfs, api_urls = get_ffbridge_licencie_get_urls(api_urls_d)
        if st.session_state.simultaneeCode  == 'RRN':
            # RRN (Roy Rene simultaneious tournament) has no deal related columns in the simultaneous_deals dataframe.
            # so we need to get the boards from the tournament date and add the deal related columns to the simultaneous_deals dataframe.
            st.session_state.tournament_id = mlBridgeLib.mlBridgeBPLib.get_teams_by_tournament_date(st.session_state.tournament_date)
            max_deals = 36 # todo: is max_deals (number of deals) available in any API at this point?
            deal_numbers = dfs['simultaneous_roadsheets']['roadsheets_deals_dealNumber'].unique().to_list()
            # uses st.session_state.player_license_number to get boards because Roy Rene website works with player_license_number to get boards.
            with st.spinner(f"Roy Rene tournaments require an extra step. Takes 1 to 3 minutes..."):
                # Get the Bridge+ club page to find the player's route link
                async def get_player_route_url():
                    # Fetch the club teams page which contains links to each pair's route
                    teams_df = await mlBridgeLib.mlBridgeBPLib.get_teams_by_tournament_async(
                        st.session_state.tournament_id, 
                        st.session_state.organization_code
                    )
                    
                    # Normalize player_id by stripping leading zeros for robust string comparison
                    norm_player_id = str(st.session_state.player_license_number).lstrip('0')
                    
                    # Find the team where the player_id matches either Player1_ID or Player2_ID
                    player_team = teams_df.filter(
                        (pl.col('Player1_ID').cast(pl.Utf8).str.strip_chars_start('0') == norm_player_id) | 
                        (pl.col('Player2_ID').cast(pl.Utf8).str.strip_chars_start('0') == norm_player_id)
                    )
                    
                    # Check if player was found
                    if len(player_team) == 0:
                        raise ValueError(f"Player {st.session_state.player_license_number} not found in tournament {st.session_state.tournament_id}, club {st.session_state.organization_code}")
                    
                    # Extract section and team number from the teams data
                    section = player_team['Section'].first()
                    team_number = player_team['Team_Number'].first()
                    
                    # Build the route URL using the extracted parameters
                    route_url = f"https://www.bridgeplus.com/nos-simultanes/resultats/?p=route&res=sim&tr={st.session_state.tournament_id}&cl={st.session_state.organization_code}&sc={section}&eq={team_number}"
                    
                    return route_url, section, team_number
                
                # Get the route URL by finding the player in the club page
                st.session_state.route_url, st.session_state.section_name, bridgeplus_team_number = asyncio.run(get_player_route_url())
                print(f"Found player route URL: {st.session_state.route_url}")
                print(f"Getting route data from: {st.session_state.route_url}")
                if False:
                    # calls internal async version which takes 60s. almost 3x faster than asyncio version below
                    #  -- but this version doesn't show progress bar -- might overwhelm server as I'm getting blacklisted(?).
                    boards_dfs = mlBridgeLib.mlBridgeBPLib.get_all_boards_for_player(st.session_state.tournament_id, st.session_state.organization_code, st.session_state.player_license_number, max_deals=36)
                else:
                    # Get boards data with progress bar by processing boards one by one using the existing function
                    boards_dfs = {'boards': None, 'score_frequency': None}
                    
                    try:
                        
                        async def get_boards_with_progress():
                            # First, get the route data to see which boards this player actually played
                            # We need to find the player's team first to get the route data
                            # teams_df = await mlBridgeLib.mlBridgeBPLib.get_teams_by_tournament_async(st.session_state.tournament_id, st.session_state.organization_code)
                            
                            # # Normalize player_id by stripping leading zeros for robust string comparison
                            # norm_player_id = st.session_state.player_license_number.lstrip('0')
                            
                            # # Find the team where the player_id matches either Player1_ID or Player2_ID
                            # player_team = teams_df.filter(
                            #     (pl.col('Player1_ID').str.strip_chars_start('0') == norm_player_id) | 
                            #     (pl.col('Player2_ID').str.strip_chars_start('0') == norm_player_id)
                            # )
                            
                            # # Check if player was found
                            # if len(player_team) == 0:
                            #     raise ValueError(f"Player {st.session_state.player_license_number} not found in tournament {st.session_state.tournament_id}, club {st.session_state.organization_code}")
                            
                            # # Get the section and team number from the extracted data
                            # sc = player_team['Section'].first()
                            # team_number = player_team['Team_Number'].first()
                            
                            # print(f"Found player {st.session_state.player_license_number} in team {team_number}, section {sc}")
                            
                            # Get the route data to see which boards this team actually played
                            # e.g. "https://www.bridgeplus.com/nos-simultanes/resultats/?p=route&res=sim&tr=S202602&cl=5802079&sc=A&eq=212"
                            played_boards = []
                            async with mlBridgeLib.mlBridgeBPLib.get_browser_context_async() as context:
                                try:
                                    route_results = await mlBridgeLib.mlBridgeBPLib.request_board_results_dataframe_async(st.session_state.route_url, context)
                                    if len(route_results) == 0:
                                        st.warning(f"No route data found for team {st.session_state.team_number}")
                                        return {'boards': pl.DataFrame(), 'score_frequency': pl.DataFrame()}
                                    else:
                                        played_boards = route_results['Board'].to_list()
                                        print(f"Found {len(played_boards)} boards played by team {st.session_state.team_number}: {played_boards}")
                                except Exception as e:
                                    print(f"Error getting route data for team {st.session_state.team_number}: {e}")
                                    raise
                            
                            if not played_boards:
                                print(f"No boards found in route data for team {st.session_state.team_number}, returning empty results")
                                return {'boards': pl.DataFrame(), 'score_frequency': pl.DataFrame()}
                            
                            # Create progress bar for board processing
                            progress_bar = st.progress(0)
                            progress_text = st.empty()
                            
                            # Now get board data only for the boards that were actually played using the existing function
                            all_boards = []
                            all_frequency = []
                            
                            async with mlBridgeLib.mlBridgeBPLib.get_browser_context_async() as context:
                                for idx, deal_num in enumerate(played_boards):
                                    try:
                                        # Update progress
                                        progress = (idx + 1) / len(played_boards)
                                        progress_bar.progress(progress)
                                        progress_text.text(f"Processing board {idx + 1}/{len(played_boards)}: Board {deal_num}")
                                        
                                        # Use the existing function to get the specific board for this player
                                        result = await mlBridgeLib.mlBridgeBPLib.get_board_for_player_async(
                                            st.session_state.tournament_id, 
                                            st.session_state.organization_code, 
                                            st.session_state.player_license_number, 
                                            str(deal_num), 
                                            context
                                        )
                                        
                                        if len(result['boards']) > 0:
                                            all_boards.append(result['boards'])
                                        if len(result['score_frequency']) > 0:
                                            all_frequency.append(result['score_frequency'])
                                    except Exception as e:
                                        print(f"Failed to scrape board {deal_num} for player {st.session_state.player_license_number}: {e}")
                                        continue
                            
                            # Complete progress bar
                            progress_bar.progress(1.0)
                            progress_text.text("✅ All boards processed successfully!")
                            
                            # Clean up progress indicators after a brief delay
                            import time
                            time.sleep(1)
                            progress_bar.empty()
                            progress_text.empty()
                            
                            # Combine all boards and frequency data
                            if all_boards:
                                combined_boards = pl.concat(all_boards, how='vertical_relaxed')
                            else:
                                combined_boards = pl.DataFrame()
                            
                            if all_frequency:
                                combined_frequency = pl.concat(all_frequency, how='vertical_relaxed')
                            else:
                                combined_frequency = pl.DataFrame()
                            
                            return {
                                'boards': combined_boards,
                                'score_frequency': combined_frequency
                            }
                        
                        # Run the async function
                        boards_dfs = asyncio.run(get_boards_with_progress())
                        
                    except Exception as e:
                        st.error(f"Error getting boards for player {st.session_state.player_license_number}: {e}")
                        boards_dfs = {'boards': pl.DataFrame(), 'score_frequency': pl.DataFrame()}

            if st.session_state.debug_mode:
                for k,v in boards_dfs.items():
                    st.caption(f"{k}: Shape:{v.shape}")
                    st.dataframe(v,selection_mode='single-row')

            df = dfs['simultaneous_roadsheets']
            # 'roadsheets_deals_dealNumber', 'roadsheets_deals_opponentsAvgNote', 'roadsheets_deals_opponentsNote', 'roadsheets_deals_opponentsOrientation', 'roadsheets_deals_opponentsScore',
            # 'roadsheets_deals_teamAvgNote', 'roadsheets_deals_teamNote', 'roadsheets_deals_teamOrientation', 'roadsheets_deals_teamScore',
            # 'roadsheets_teams_cpt', 'roadsheets_player_[nesw]'
            if st.session_state.pair_direction == 'NS':
                # not liking that only one of the two columns (nsScore or ewScore) has a value. I prefer to have both with opposite signs.
                # although this may be an issue for director adjustments. Creating new columns (Score_NS and Score_EW) with opposite signs.
                df = df.with_columns([
                    pl.when(pl.col('roadsheets_deals_teamScore').str.contains(r'^\d+$'))
                        .then(pl.col('roadsheets_deals_teamScore'))
                        .otherwise('-'+pl.col('roadsheets_deals_opponentsScore'))
                        .cast(pl.Int16)
                        .alias('Score_NS'),
                ])
                df = df.with_columns([
                    pl.when(pl.col('roadsheets_deals_opponentsScore').str.contains(r'^\d+$'))
                        .then(pl.col('roadsheets_deals_opponentsScore'))
                        .otherwise('-'+pl.col('roadsheets_deals_teamScore'))
                        .cast(pl.Int16)
                        .alias('Score_EW'),
                ])
                df = df.with_columns([
                    pl.col('roadsheets_deals_teamNote').cast(pl.Float32).alias('MP_NS'),
                    pl.col('roadsheets_deals_opponentsNote').cast(pl.Float32).alias('MP_EW'),
                ])
                df = df.with_columns(
                    (pl.col('roadsheets_deals_teamAvgNote')/100).round(2).alias('Pct_NS'),
                    (pl.col('roadsheets_deals_opponentsAvgNote')/100).round(2).alias('Pct_EW'),
                )
                df = df.with_columns([
                    pl.col('roadsheets_teams_players').list.get(0).alias('Player_Name_N'),
                    pl.col('roadsheets_teams_players').list.get(1).alias('Player_Name_S'),
                    pl.col('roadsheets_teams_opponents').list.get(0).alias('Player_Name_E'),
                    pl.col('roadsheets_teams_opponents').list.get(1).alias('Player_Name_W'),
                ])
            else:
                df = df.with_columns([
                    pl.when(pl.col('roadsheets_deals_teamScore').str.contains(r'^\d+$'))
                        .then(pl.col('roadsheets_deals_teamScore'))
                        .otherwise('-'+pl.col('roadsheets_deals_opponentsScore'))
                        .cast(pl.Int16)
                        .alias('Score_EW'),
                ])
                df = df.with_columns([
                    pl.when(pl.col('roadsheets_deals_opponentsScore').str.contains(r'^\d+$'))
                        .then(pl.col('roadsheets_deals_opponentsScore'))
                        .otherwise('-'+pl.col('roadsheets_deals_teamScore'))
                        .cast(pl.Int16)
                        .alias('Score_NS'),
                ])
                df = df.with_columns([
                    pl.col('roadsheets_deals_teamNote').cast(pl.Float32).alias('MP_EW'),
                    pl.col('roadsheets_deals_opponentsNote').cast(pl.Float32).alias('MP_NS'),
                ])
                df = df.with_columns(
                    (pl.col('roadsheets_deals_teamAvgNote')/100).round(2).alias('Pct_EW'),
                    (pl.col('roadsheets_deals_opponentsAvgNote')/100).round(2).alias('Pct_NS'),
                )
                df = df.with_columns([
                    pl.col('roadsheets_teams_players').list.get(0).alias('Player_Name_E'),
                    pl.col('roadsheets_teams_players').list.get(1).alias('Player_Name_W'),
                    pl.col('roadsheets_teams_opponents').list.get(0).alias('Player_Name_N'),
                    pl.col('roadsheets_teams_opponents').list.get(1).alias('Player_Name_S'),
                ])
            df = df.with_columns([
                pl.col('roadsheets_deals_dealNumber').cast(pl.UInt32).alias('Board'),
                pl.lit(st.session_state.team_id).alias('team_id'),
                pl.lit(st.session_state.organization_code).alias('team_organization_code'),
            ])
            # df = df.with_columns([
            #     pl.col('roadsheets_player_n').alias('Player_Name_N'),
            #     pl.col('roadsheets_player_e').alias('Player_Name_E'),
            #     pl.col('roadsheets_player_s').alias('Player_Name_S'),
            #     pl.col('roadsheets_player_w').alias('Player_Name_W'),
            # ])
            df = df.select([pl.exclude('^roadsheets_.*$')])
            # # create columns to match missing deal related columns
            # simultaneous_tournaments_df = simultaneous_tournaments_df.with_columns([
            #     pl.lit(st.session_state.tournament_id).alias('tournament_id'),
            #     pl.lit(st.session_state.organization_code).alias('club_id'),
            # ])
            # simultaneous_tournaments_df = simultaneous_tournaments_df.with_columns([
            #     pl.when(pl.col('team_orientation') == 'NS').then(pl.col('team_percent').cast(pl.Float64)/100).otherwise(1-(pl.col('team_percent').cast(pl.Float64)/100)).alias('Pct_NS'),
            # ])
            # simultaneous_tournaments_df = simultaneous_tournaments_df.with_columns([
            #     pl.when(pl.col('team_orientation') == 'EW').then(pl.col('team_percent').cast(pl.Float64)/100).otherwise(1-(pl.col('team_percent').cast(pl.Float64)/100)).alias('Pct_EW'),
            # ])
            # player_n_df = simultaneous_tournaments_df.filter(pl.col('team_orientation') == 'NS').filter(pl.col('team_players_position') == 1).drop('team_players_position')
            # player_n_df = player_n_df['team_organization_code','team_id','team_table_number','team_players_id','team_players_firstname','team_players_lastname']
            # player_n_df = player_n_df.rename({'team_players_id':'Player_ID_N','team_players_firstname':'Player_Name_N','team_players_lastname':'Player_Lastname_N'})
            # player_e_df = simultaneous_tournaments_df.filter(pl.col('team_orientation') == 'EW').filter(pl.col('team_players_position') == 1).drop('team_players_position')
            # player_e_df = player_e_df['team_organization_code','team_id','team_table_number','team_players_id','team_players_firstname','team_players_lastname']
            # player_e_df = player_e_df.rename({'team_players_id':'Player_ID_E','team_players_firstname':'Player_Name_E','team_players_lastname':'Player_Lastname_E'})
            # player_s_df = simultaneous_tournaments_df.filter(pl.col('team_orientation') == 'NS').filter(pl.col('team_players_position') == 2).drop('team_players_position')
            # player_s_df = player_s_df['team_organization_code','team_id','team_table_number','team_players_id','team_players_firstname','team_players_lastname']
            # player_s_df = player_s_df.rename({'team_players_id':'Player_ID_S','team_players_firstname':'Player_Name_S','team_players_lastname':'Player_Lastname_S'})
            # player_w_df = simultaneous_tournaments_df.filter(pl.col('team_orientation') == 'EW').filter(pl.col('team_players_position') == 2).drop('team_players_position')
            # player_w_df = player_w_df['team_organization_code','team_id','team_table_number','team_players_id','team_players_firstname','team_players_lastname']
            # player_w_df = player_w_df.rename({'team_players_id':'Player_ID_W','team_players_firstname':'Player_Name_W','team_players_lastname':'Player_Lastname_W'})
            # pairs_ns_df = player_n_df.join(player_s_df,on=('team_id','team_organization_code','team_table_number'),how='inner')
            # pairs_ew_df = player_e_df.join(player_w_df,on=('team_id','team_organization_code','team_table_number'),how='inner')
            simultaneous_tournaments_df = simultaneous_tournaments_df.with_columns([
                # mlBridgeAugmentLib.py wants Player_ID_[NESW] to be Utf8
                pl.col('team_players_id').cast(pl.Utf8).alias('team_players_id'),
            ])
            # todo: section_name needs to be used to make unique.
            # Easier to work with a unique id for each team: team_organization_code + section_name + team_orientation + team_table_number?
            # Easier to work with a unique id for each player: team_organization_code + section_name + player_orientation + team_table_number?
            player_n_df = simultaneous_tournaments_df.filter(pl.col('team_orientation') == 'NS').filter(pl.col('team_players_position') == 1).drop('team_players_position')
            player_n_df = player_n_df['team_organization_code','team_table_number','team_players_id']
            player_n_df = player_n_df.rename({'team_players_id':'Player_ID_N'})
            player_e_df = simultaneous_tournaments_df.filter(pl.col('team_orientation') == 'EW').filter(pl.col('team_players_position') == 1).drop('team_players_position')
            player_e_df = player_e_df['team_organization_code','team_table_number','team_players_id']
            player_e_df = player_e_df.rename({'team_players_id':'Player_ID_E'})
            player_s_df = simultaneous_tournaments_df.filter(pl.col('team_orientation') == 'NS').filter(pl.col('team_players_position') == 2).drop('team_players_position')
            player_s_df = player_s_df['team_organization_code','team_table_number','team_players_id']
            player_s_df = player_s_df.rename({'team_players_id':'Player_ID_S'})
            player_w_df = simultaneous_tournaments_df.filter(pl.col('team_orientation') == 'EW').filter(pl.col('team_players_position') == 2).drop('team_players_position')
            player_w_df = player_w_df['team_organization_code','team_table_number','team_players_id']
            player_w_df = player_w_df.rename({'team_players_id':'Player_ID_W'})
            # this code will probably work for creating 'Pair_Number_(NS|EW)' columns. instead of below method?
            #pair_ns_df = simultaneous_tournaments_df.filter(pl.col('team_orientation') == 'NS')
            #pair_ns_df = pair_ns_df['team_organization_code','team_table_number']
            #pair_ns_df = pair_ns_df.rename({'team_table_number':'Pair_Number_NS'})
            #pair_ew_df = simultaneous_tournaments_df.filter(pl.col('team_orientation') == 'EW')
            #pair_ew_df = pair_ew_df['team_organization_code','team_table_number']
            #pair_ew_df = pair_ew_df.rename({'team_table_number':'Pair_Number_EW'})
            boards_df = boards_dfs['boards']
            assert boards_df.height > 0, f"No boards found for {st.session_state.tournament_id}"
            
            # Debug: Check what columns we actually have
            print(f"Boards DataFrame columns: {boards_df.columns}")
            print(f"Boards DataFrame shape: {boards_df.shape}")
            if st.session_state.debug_mode:
                st.write("**Debug: Boards DataFrame sample:**")
                st.dataframe(boards_df.head(3))
            
            boards_df = boards_df.with_columns([
                pl.lit(st.session_state.tournament_id).alias('tournament_id'),
                pl.lit(st.session_state.organization_code).alias('club_id'),
                pl.lit(st.session_state.tournament_date).alias('Date'),
                pl.lit(st.session_state.section_name).alias('Section_Name'),
                #pl.lit(st.session_state.team_id).alias('team_id'),
                pl.lit(st.session_state.player_license_number).cast(pl.Int64).alias('team_license_number'),
                pl.lit(st.session_state.player_id).cast(pl.Int64).alias('Player_ID'),
                pl.lit(st.session_state.partner_id).cast(pl.Int64).alias('Partner_ID'),
                pl.lit(st.session_state.player_direction).alias('Player_Direction'),
                pl.lit(st.session_state.pair_direction).alias('Pair_Direction'),
            ])
            
            # Debug: Check boards_df before joins
            print(f"Before joins - boards_df shape: {boards_df.shape}")
            print(f"Before joins - boards_df columns: {boards_df.columns}")
            if boards_df.height > 0:
                print(f"Before joins - sample Pair_Number values: {boards_df['Pair_Number'].unique().to_list()[:5]}")
                print(f"Before joins - sample Club_ID values: {boards_df['Club_ID'].unique().to_list()[:5]}")
                print(f"Before joins - sample club_id values: {boards_df['club_id'].unique().to_list()[:5]}")
            
            # Debug: Check player dataframes before joins
            print(f"player_n_df shape: {player_n_df.shape}")
            if player_n_df.height > 0:
                print(f"player_n_df sample team_table_number: {player_n_df['team_table_number'].unique().to_list()[:5]}")
                print(f"player_n_df sample team_organization_code: {player_n_df['team_organization_code'].unique().to_list()[:5]}")
            
            if st.session_state.pair_direction == 'NS':
                # Use LEFT joins to preserve boards even when player IDs don't match
                # This is necessary because BridgePlus and FFBridge use different numbering systems
                print(f"Joining with player_n_df...")
                boards_df = boards_df.join(player_n_df,left_on=('club_id','Pair_Number'),right_on=('team_organization_code','team_table_number'),how='left')
                print(f"After player_n_df join - boards_df shape: {boards_df.shape}")
                
                print(f"Joining with player_e_df...")
                boards_df = boards_df.join(player_e_df,left_on=('club_id','Opponent_Pair_Number'),right_on=('team_organization_code','team_table_number'),how='left')
                print(f"After player_e_df join - boards_df shape: {boards_df.shape}")
                
                print(f"Joining with player_s_df...")
                boards_df = boards_df.join(player_s_df,left_on=('club_id','Pair_Number'),right_on=('team_organization_code','team_table_number'),how='left')
                print(f"After player_s_df join - boards_df shape: {boards_df.shape}")
                
                print(f"Joining with player_w_df...")
                boards_df = boards_df.join(player_w_df,left_on=('club_id','Opponent_Pair_Number'),right_on=('team_organization_code','team_table_number'),how='left')
                print(f"After player_w_df join - boards_df shape: {boards_df.shape}")
                
                boards_df = boards_df.with_columns([
                    pl.col('Pair_Number').alias('Pair_Number_NS'),
                    pl.col('Opponent_Pair_Number').alias('Pair_Number_EW'),
                ])
                
                # Fill in player IDs from session state for the user's pair if joins didn't match
                # This handles the case where BridgePlus and FFBridge use different numbering
                # Check if joins failed (columns don't exist or are all null)
                needs_player_ids = ('Player_ID_N' not in boards_df.columns or 
                                   (boards_df.height > 0 and boards_df['Player_ID_N'].is_null().all()))
                if needs_player_ids:
                    print("Player IDs not found from joins, populating from session state...")
                    boards_df = boards_df.with_columns([
                        pl.lit(str(st.session_state.player_id if st.session_state.player_direction == 'N' else st.session_state.partner_id if st.session_state.partner_direction == 'N' else '')).alias('Player_ID_N'),
                        pl.lit(str(st.session_state.player_id if st.session_state.player_direction == 'S' else st.session_state.partner_id if st.session_state.partner_direction == 'S' else '')).alias('Player_ID_S'),
                        pl.lit('').alias('Player_ID_E'),  # Opponents - unknown
                        pl.lit('').alias('Player_ID_W'),  # Opponents - unknown
                    ])
            else:
                # Use LEFT joins to preserve boards even when player IDs don't match
                # This is necessary because BridgePlus and FFBridge use different numbering systems
                print(f"Joining with player_n_df...")
                boards_df = boards_df.join(player_n_df,left_on=('club_id','Opponent_Pair_Number'),right_on=('team_organization_code','team_table_number'),how='left')
                print(f"After player_n_df join - boards_df shape: {boards_df.shape}")
                
                print(f"Joining with player_e_df...")
                boards_df = boards_df.join(player_e_df,left_on=('club_id','Pair_Number'),right_on=('team_organization_code','team_table_number'),how='left')
                print(f"After player_e_df join - boards_df shape: {boards_df.shape}")
                
                print(f"Joining with player_s_df...")
                boards_df = boards_df.join(player_s_df,left_on=('club_id','Opponent_Pair_Number'),right_on=('team_organization_code','team_table_number'),how='left')
                print(f"After player_s_df join - boards_df shape: {boards_df.shape}")
                
                print(f"Joining with player_w_df...")
                boards_df = boards_df.join(player_w_df,left_on=('club_id','Pair_Number'),right_on=('team_organization_code','team_table_number'),how='left')
                print(f"After player_w_df join - boards_df shape: {boards_df.shape}")
                
                boards_df = boards_df.with_columns([
                    pl.col('Pair_Number').alias('Pair_Number_EW'),
                    pl.col('Opponent_Pair_Number').alias('Pair_Number_NS'),
                ])
                
                # Fill in player IDs from session state for the user's pair if joins didn't match
                # This handles the case where BridgePlus and FFBridge use different numbering
                # Check if joins failed (columns don't exist or are all null)
                needs_player_ids = ('Player_ID_E' not in boards_df.columns or 
                                   (boards_df.height > 0 and boards_df['Player_ID_E'].is_null().all()))
                if needs_player_ids:
                    print("Player IDs not found from joins, populating from session state...")
                    boards_df = boards_df.with_columns([
                        pl.lit('').alias('Player_ID_N'),  # Opponents - unknown
                        pl.lit('').alias('Player_ID_S'),  # Opponents - unknown
                        pl.lit(str(st.session_state.player_id if st.session_state.player_direction == 'E' else st.session_state.partner_id if st.session_state.partner_direction == 'E' else '')).alias('Player_ID_E'),
                        pl.lit(str(st.session_state.player_id if st.session_state.player_direction == 'W' else st.session_state.partner_id if st.session_state.partner_direction == 'W' else '')).alias('Player_ID_W'),
                    ])
            
            # Debug: Check boards_df before final join with roadsheets
            print(f"Before final join with roadsheets - boards_df shape: {boards_df.shape}")
            print(f"Before final join - df (roadsheets) shape: {df.shape}")
            if boards_df.height > 0 and df.height > 0:
                print(f"boards_df Board values: {sorted(boards_df['Board'].unique().to_list())}")
                print(f"df (roadsheets) Board values: {sorted(df['Board'].unique().to_list())}")
            
            df = boards_df.join(df, on='Board', how='left')
            print(f"After final join with roadsheets - df shape: {df.shape}")
        else:
            try:
                df = mlBridgeFFLib.convert_ffdf_api_to_mldf(dfs)
            except Exception as e:
                st.error(str(e))
                return True

        if st.session_state.debug_mode:
            st.caption(f"Final dataframe: Shape:{df.shape}")
            st.dataframe(df,selection_mode='single-row')

        if df['Contract'].is_null().all(): # ouch. e.g. Monday Simultané Octopus
            st.error("No Contract data available. Unable to proceed.")
            return True

        # Only use columns that are required by augmentation. Drop all other columns.
        df = df[
            'Date','Section_Name',
            'Board','PBN','Pair_Direction','Dealer','Vul','Declarer','Contract','Result',
            'Score_EW','Score_NS',
            'Pct_NS','Pct_EW',
            'MP_NS','MP_EW', 'MP_Top',
            'Pair_Number_NS','Pair_Number_EW',
            'Player_ID_N','Player_ID_E','Player_ID_S','Player_ID_W',
            'Player_Name_N','Player_Name_E','Player_Name_S','Player_Name_W',
        ]
        # only works if Board_We_Played is desired.
        # player_id_col = f"Player_ID_{st.session_state.player_direction}"
        # partner_id_col = f"Player_ID_{st.session_state.partner_direction}"
        # # todo: following is generic?
        # assert df[player_id_col].n_unique() == 1, f"{player_id_col} is not unique"
        # assert df[partner_id_col].n_unique() == 1, f"{partner_id_col} is not unique"
        st.session_state.session_id = st.session_state.simultane_id

        if not st.session_state.use_historical_data: # historical data is already fully augmented so skip past augmentations
            if st.session_state.do_not_cache_df:
                with st.spinner('Creating ffbridge data to dataframe...'):
                    df = augment_df(df)
            else:
                ffbridge_session_player_cache_df_filename = f'{st.session_state.cache_dir}/df-{st.session_state.session_id}-{st.session_state.player_id}.parquet'
                ffbridge_session_player_cache_df_file = pathlib.Path(ffbridge_session_player_cache_df_filename)
                if ffbridge_session_player_cache_df_file.exists():
                    df = _cached_read_parquet(str(ffbridge_session_player_cache_df_file))
                    print(f"Loaded {ffbridge_session_player_cache_df_filename}: shape:{df.shape} size:{ffbridge_session_player_cache_df_file.stat().st_size}")
                else:
                    with st.spinner('Creating ffbridge data to dataframe...'):
                        df = augment_df(df)
                    if df is not None:
                        st.session_state.df_ready = True  # main loop can notice and proceed
                    ffbridge_session_player_cache_dir = pathlib.Path(st.session_state.cache_dir)
                    ffbridge_session_player_cache_dir.mkdir(exist_ok=True)  # Creates directory if it doesn't exist
                    ffbridge_session_player_cache_df_filename = f'{st.session_state.cache_dir}/df-{st.session_state.session_id}-{st.session_state.player_id}.parquet'
                    ffbridge_session_player_cache_df_file = pathlib.Path(ffbridge_session_player_cache_df_filename)
                    df.write_parquet(ffbridge_session_player_cache_df_file)
                    print(f"Saved {ffbridge_session_player_cache_df_filename}: shape:{df.shape} size:{ffbridge_session_player_cache_df_file.stat().st_size}")
            with st.spinner('Writing column names to file...'):
                with open('df_columns.txt','w') as f:
                    for col in sorted(df.columns):
                        f.write(col+'\n')

            # personalize to player, partner, opponents, etc.
            st.session_state.df = filter_dataframe(df) #, st.session_state.group_id, st.session_state.session_id, st.session_state.player_id, st.session_state.partner_id)

            # Register DataFrame as 'self' view in the session-specific connection
            con = get_session_duckdb_connection()
            con.register('self', st.session_state.df)
            print(f"st.session_state.df:{st.session_state.df.columns}")

    print(f"=== change_game_state END: SUCCESS - player_id={st.session_state.player_id}, session_id={st.session_state.session_id} ===")
    return False


def on_game_url_input_change() -> None:
    """Handle game URL input change event"""
    st.session_state.game_url = st.session_state.game_url_input
    if change_game_state(st.session_state.player_id, None):
        st.session_state.game_url_default = ''
        reset_game_data()


@st.dialog("Select Player")
def show_player_selection_modal(filtered_options):
    """Show modal dialog with radio buttons and Select button"""
    st.write(f"Found {len(filtered_options)} match(es). Select a player:")
    
    # Radio buttons for selection
    selected_option = st.radio(
        "Players:",
        options=filtered_options,
        index=None,
        key='modal_player_radio',
        label_visibility="collapsed"
    )
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Select", disabled=selected_option is None, width="stretch", type="primary"):
            if selected_option:
                # Find the actual player_id for the selected option
                if hasattr(st.session_state, 'player_search_matches'):
                    for display_text, player_id, license_number, player_name in st.session_state.player_search_matches:
                        if display_text == selected_option:
                            # Update both the value state AND the widget key to sync the textbox
                            st.session_state.player_search_value = str(license_number)
                            st.session_state.player_search_input = str(license_number)
                            
                            # Also set the player_id for downstream processing
                            st.session_state.player_id = str(player_id)
                            
                            # Clear dialog state to dismiss dialog
                            if hasattr(st.session_state, 'player_search_matches'):
                                del st.session_state.player_search_matches
                            if hasattr(st.session_state, 'player_search_error'):
                                del st.session_state.player_search_error
                            # Clear the modal flag as well
                            if hasattr(st.session_state, 'show_player_modal'):
                                del st.session_state.show_player_modal
                            
                            # Flag for main loop to refresh after modal selection
                            st.session_state.deferred_start_report = True
                            # st.query_params["player_id"] = str(license_number)  # no-op to mark state change
                            return
                
    with col2:
        if st.button("Cancel", width="stretch"):
            # Clear dialog state immediately (like X button does)
            if hasattr(st.session_state, 'player_search_matches'):
                del st.session_state.player_search_matches
            if hasattr(st.session_state, 'player_search_error'):
                del st.session_state.player_search_error
            # Clear the modal flag as well
            if hasattr(st.session_state, 'show_player_modal'):
                del st.session_state.show_player_modal
            # Use st.stop() to await the clearing of the UI dismissal queue
            st.stop()


def player_search_input_on_change_with_query(query: str) -> None:
    """Handle player search with a specific query string"""
    
    if not query or not query.strip():
        # Clear any existing search state when input is empty
        if hasattr(st.session_state, 'player_search_matches'):
            del st.session_state.player_search_matches
        if hasattr(st.session_state, 'player_search_error'):
            del st.session_state.player_search_error
        return
    
    # Only search if we have at least 4 characters (to avoid premature searches)
    if len(query.strip()) < 4:
        return
        
    try:
        api_urls_d = {
            'search': (f"https://api.ffbridge.fr/api/v1/search-members?alive=1&search={query}", False),
        }
        dfs, api_urls_d = get_ffbridge_data_using_url_licencie(api_urls_d, show_progress=False)
        
        if len(dfs['search']) == 0:
            # Store error message in session state to persist across reruns
            st.session_state.player_search_error = f"Player number '{query}' not found. Please check the number and try again."
            # Reset player_id to None to ensure Morty instructions are shown
            st.session_state.player_id = None
            return
            
        if len(dfs['search']) > 1:
            # If input is more than 3 characters, show matches in selectbox
            if len(query.strip()) > 3:
                # Store matches for selectbox display
                matches = []
                for row in dfs['search'].iter_rows(named=True):
                    # Debug: Print available columns and values
                    if st.session_state.get('debug_mode', False):
                        print(f"Available columns: {list(row.keys())}")
                        print(f"Row data: {row}")
                    
                    # Try different possible field names for firstname/lastname
                    firstname = row.get('person_firstname', '') or row.get('firstname', '') or row.get('first_name', '')
                    lastname = row.get('person_lastname', '') or row.get('lastname', '') or row.get('last_name', '')
                    player_name = f"{firstname} {lastname}".strip()
                    
                    # Try different possible field names for license number
                    license_number = row.get('person_license_number', '') or row.get('license_number', '') or row.get('licenseNumber', '')
                    
                    # Try different possible field names for player ID
                    player_id = row.get('person_id', '') or row.get('id', '') or row.get('player_id', '')
                    
                    # Format: "First Last - number" - compact display for narrow selectbox
                    if player_name.strip() and license_number:
                        display_text = f"{player_name} - {license_number}"
                    elif player_name.strip():
                        display_text = f"{player_name} - {player_id}"
                    elif license_number:
                        display_text = f"License: {license_number}"
                    else:
                        display_text = f"Player ID: {player_id}"
                    
                    matches.append((display_text, player_id, license_number, player_name))
                
                st.session_state.player_search_matches = matches
                # Store the search query for display in selectbox (strip whitespace)
                st.session_state.player_search_query = query.strip()
                
                # Debug: Show what matches were created
                if st.session_state.get('debug_mode', False):
                    print(f"Created {len(matches)} matches:")
                    for i, (display_text, player_id, license_number, player_name) in enumerate(matches):
                        print(f"  Match {i}: '{display_text}' (ID: {player_id})")
                
                # Clear any error message since we're showing the selectbox instead
                if hasattr(st.session_state, 'player_search_error'):
                    del st.session_state.player_search_error
                # Don't reset player_id here - we want to keep current state and show modal
                # Set a flag to show modal on next run (after this search processing completes)
                st.session_state.show_player_modal = True
                return
            else:
                # For short inputs, don't show error - let user continue typing
                # Clear any previous error message since we're not showing selectbox
                if hasattr(st.session_state, 'player_search_error'):
                    del st.session_state.player_search_error
                # Don't reset player_id here either - just return
                return
            
        # Single player found - get their ID using proper Polars syntax
        try:
            player_id = dfs['search']['person_id'].to_list()[0]
        except Exception as e:
            # More informative error if column doesn't exist
            print(f"Error accessing person_id from search results: {e}")
            print(f"Available columns: {dfs['search'].columns}")
            print(f"Search dataframe:\n{dfs['search']}")
            raise Exception(f"Could not extract player_id from search results. Available columns: {dfs['search'].columns}")
        
        # Clear any previous error message and matches on successful search
        if hasattr(st.session_state, 'player_search_error'):
            del st.session_state.player_search_error
        if hasattr(st.session_state, 'player_search_matches'):
            del st.session_state.player_search_matches
        
        # Defer report start: first refresh sidebar with games, then start report
        st.session_state.player_id = str(player_id)
        try:
            populate_game_urls_for_player(st.session_state.player_id)
        except Exception as _:
            pass
        st.session_state.deferred_start_report = True
        return
        
    except Exception as e:
        # Store only the underlying error message (no prefix) for clarity
        st.session_state.player_search_error = str(e)
        # Clear any previous matches
        if hasattr(st.session_state, 'player_search_matches'):
            del st.session_state.player_search_matches
        # Reset player_id to None to ensure Morty instructions are shown
        st.session_state.player_id = None


def player_search_input_on_change() -> None:
    """Handle player search input change - delegates to helper function"""
    player_search_input = st.session_state.player_search_input
    # Sync the value state with the input
    st.session_state.player_search_value = player_search_input
    player_search_input_on_change_with_query(player_search_input)



def club_session_id_on_change() -> None:
    #st.session_state.tournament_session_ids_selectbox = None # clear tournament index whenever club index changes. todo: doesn't seem to update selectbox with new index.
    selection = st.session_state.club_session_ids_selectbox
    if selection is not None:
        session_id = int(selection.split(',')[0]) # split selectbox item on commas. only want first split.
        if change_game_state(st.session_state.player_id, session_id):
            st.session_state.session_id = None
        else:
            st.session_state.sql_query_mode = False


def create_sidebar() -> None:
    """Legacy function - use app.create_sidebar instead"""
    if 'app' in st.session_state:
        st.session_state.app.create_sidebar()
    else:
        # Fallback for backward compatibility - basic sidebar
        st.sidebar.caption(st.session_state.get('app_datetime', ''))
        st.sidebar.text_input(
            "Enter ffbridge license number", 
            on_change=player_search_input_on_change, 
            placeholder=st.session_state.get('player_license_number', ''), 
            key='player_search_input', 
            help="Enter ffbridge license number or (partial) last name."
        )


# Legacy functions removed - functionality moved to class-based approach


def initialize_website_specific() -> None:
    """Initialize website-specific settings and configurations"""

    st.session_state.assistant_logo = 'https://github.com/BSalita/ffbridge-postmortem/blob/master/assets/logo_assistant.gif?raw=true', # 🥸 todo: put into config. must have raw=true for github url.
    st.session_state.guru_logo = 'https://github.com/BSalita/ffbridge-postmortem/blob/master/assets/logo_guru.png?raw=true', # 🥷todo: put into config file. must have raw=true for github url.
    #st.session_state.game_url_default = 'https://ffbridge.fr/competitions/results/groups/7878/sessions/107118/pairs/3976783'
    #st.session_state.game_url_default = 'https://licencie.ffbridge.fr/#/resultats/simultane/34424/details/4818526?orgId=1634'
    st.session_state.game_name = 'ffbridge'
    #st.session_state.game_url = st.session_state.game_url_default
    
    # Initialize FFBridge Bearer Token from .env file
    initialize_ffbridge_bearer_token()
    
    # todo: put filenames into a .json or .toml file?
    st.session_state.rootPath = pathlib.Path('e:/bridge/data')
    st.session_state.ffbridgePath = st.session_state.rootPath.joinpath('ffbridge')
    #st.session_state.favoritesPath = pathlib.joinpath('favorites'),
    st.session_state.savedModelsPath = st.session_state.rootPath.joinpath('SavedModels')

    streamlit_chat.message(
        "Hi. I'm Morty. Your friendly postmortem chatbot. I only want to chat about ffbridge pair matchpoint games using a Mitchell movement and not shuffled.",
        key='intro_message_1',
        logo=st.session_state.assistant_logo
    )
    streamlit_chat.message(
        "I'm optimized for large screen devices such as a notebook or monitor. Do not use a smartphone.",
        key='intro_message_2',
        logo=st.session_state.assistant_logo
    )
    streamlit_chat.message(
        "To start our postmortem chat, I'll need the a player number of your ffbridge game. It will be the subject of our chat.",
        key='intro_message_3',
        logo=st.session_state.assistant_logo
    )
    streamlit_chat.message(
        "Enter the player number in the left sidebar or just re-enter the default player number. Press the enter key to begin.",
        key='intro_message_4',
        logo=st.session_state.assistant_logo
    )
    streamlit_chat.message(
        "I'm just a Proof of Concept so don't double me.",
        key='intro_message_5',
        logo=st.session_state.assistant_logo
    )
    app_info()
    return


# Everything below here is the standard mlBridge code.


# this version of perform_hand_augmentations_locked() uses self for class compatibility, older versions did not.
def perform_hand_augmentations_queue(augmenter_instance, hand_augmentation_work: Any) -> None:
    """Perform hand augmentations queue processing
    
    Args:
        augmenter_instance: The augmenter instance calling this method
        hand_augmentation_work: Work item for hand augmentation processing
    """
    if hasattr(st.session_state, 'app') and st.session_state.app:
        return st.session_state.app.perform_hand_augmentations_queue(augmenter_instance, hand_augmentation_work)
    else:
        # Fallback to original behavior
        sys.path.append(str(pathlib.Path.cwd().joinpath('streamlitlib')))
        import streamlitlib
        return streamlitlib.perform_queued_work(augmenter_instance, hand_augmentation_work, "Hand analysis")


# Legacy function - now handled by the base class
def augment_df(df: pl.DataFrame) -> pl.DataFrame:
    """Legacy function - use app.augment_df instead"""
    if 'app' in st.session_state:
        return st.session_state.app.augment_df(df)
    else:
        # Fallback for backward compatibility
        with st.spinner('Augmenting data...'):
            augmenter = AllAugmentations(df,None,sd_productions=st.session_state.single_dummy_sample_count,progress=st.progress(0),lock_func=perform_hand_augmentations_queue)
            df, hrs_cache_df = augmenter.perform_all_augmentations()
        return df


# Legacy function - now handled by the base class
def read_configs() -> Dict[str, Any]:
    """Legacy function - use app.read_configs instead"""
    if 'app' in st.session_state:
        return st.session_state.app.read_configs()
    else:
        # Fallback for backward compatibility
        st.session_state.default_favorites_file = pathlib.Path('default.favorites.json')
        st.session_state.player_id_custom_favorites_file = pathlib.Path(f'favorites/{st.session_state.player_id}.favorites.json')
        st.session_state.debug_favorites_file = pathlib.Path('favorites/debug.favorites.json')

        if st.session_state.default_favorites_file.exists():
            with open(st.session_state.default_favorites_file, 'r') as f:
                favorites = json.load(f)
            st.session_state.favorites = favorites
        else:
            st.session_state.favorites = None

        if st.session_state.player_id_custom_favorites_file.exists():
            with open(st.session_state.player_id_custom_favorites_file, 'r') as f:
                player_id_favorites = json.load(f)
            st.session_state.player_id_favorites = player_id_favorites
        else:
            st.session_state.player_id_favorites = None

        if st.session_state.debug_favorites_file.exists():
            with open(st.session_state.debug_favorites_file, 'r') as f:
                debug_favorites = json.load(f)
            st.session_state.debug_favorites = debug_favorites
        else:
            st.session_state.debug_favorites = None
        
        return getattr(st.session_state, 'favorites', {})


# Legacy function - now handled by the base class
def process_prompt_macros(sql_query: str) -> str:
    """Legacy function - use app.process_prompt_macros instead"""
    if 'app' in st.session_state:
        return st.session_state.app.process_prompt_macros(sql_query)
    else:
        # Fallback for backward compatibility
        replacements = {
            '{Player_Direction}': getattr(st.session_state, 'player_direction', None),
            '{Partner_Direction}': getattr(st.session_state, 'partner_direction', None),
            '{Pair_Direction}': getattr(st.session_state, 'pair_direction', None),
            '{Opponent_Pair_Direction}': getattr(st.session_state, 'opponent_pair_direction', None)
        }
        for old, new in replacements.items():
            if new is None:
                continue
            sql_query = sql_query.replace(old, new)
        return sql_query


# Legacy function - now handled by the base class
def write_report() -> None:
    """Legacy function - use app.write_report instead"""
    if 'app' in st.session_state:
        st.session_state.app.write_report()
    else:
        # Fallback - use standard report generation
        st.error("No app instance found for report generation")


# Legacy function - now handled by the base class
def ask_sql_query() -> None:
    """Legacy function - use app.ask_sql_query instead"""
    if 'app' in st.session_state:
        st.session_state.app.ask_sql_query()
    else:
        # Fallback for backward compatibility
        if st.session_state.show_sql_query:
            with st.container():
                with bottom():
                    st.chat_input('Enter a SQL query e.g. SELECT PBN, Contract, Result, N, S, E, W', key='main_prompt_chat_input', on_submit=chat_input_on_submit)


# Legacy function - now handled by the base class
def create_ui() -> None:
    """Legacy function - use app.create_ui instead"""
    if 'app' in st.session_state:
        st.session_state.app.create_ui()
    else:
        # Fallback for backward compatibility
        create_sidebar()
        if not st.session_state.sql_query_mode:
            if st.session_state.session_id is not None:
                write_report()
        ask_sql_query()


# Legacy function - now handled by the base class
def get_session_duckdb_connection():
    """Legacy function - use app.get_session_duckdb_connection instead"""
    if 'app' in st.session_state:
        return st.session_state.app.get_session_duckdb_connection()
    else:
        # Fallback for backward compatibility
        if 'con' not in st.session_state or st.session_state.con is None:
            st.session_state.con = duckdb.connect()
            print(f"Created new DuckDB connection for session")
        return st.session_state.con


# Legacy function - now handled by the base class
def initialize_session_state() -> None:
    """Legacy function - use app.initialize_session_state instead"""
    if 'app' not in st.session_state:
        st.session_state.app = FFBridgeApp()
    # The app will handle its own initialization


# Legacy function - now handled by the base class
def reset_game_data() -> None:
    """Legacy function - use app.reset_game_data instead"""
    if 'app' in st.session_state:
        st.session_state.app.reset_game_data()
    # Otherwise, the app will handle its own reset


def app_info() -> None:
    """Display app information"""
    st.caption(f"Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in streamlit. Data engine is polars. Query engine is duckdb. Bridge lib is endplay. Self hosted using Cloudflare Tunnel. Repo:https://github.com/BSalita/ffbridge-postmortem")
    st.caption(f"App:{st.session_state.app_datetime} Streamlit:{st.__version__} Query Params:{st.query_params.to_dict()} Environment:{os.getenv('STREAMLIT_ENV','')}")
    st.caption(f"Python:{'.'.join(map(str, sys.version_info[:3]))} pandas:{pd.__version__} polars:{pl.__version__} endplay:{endplay.__version__}")
    return


def main() -> None:
    """Main application entry point"""
    if 'app' not in st.session_state:
        st.session_state.app = FFBridgeApp()
    st.session_state.app.main()
    return


class FFBridgeApp(PostmortemBase):
    """FFBridge Streamlit application."""
    
    def __init__(self):
        super().__init__()
        # App-specific initialization
    
    def initialize_session_state(self):
        """Initialize FFBridge-specific session state."""
        # First initialize common session state
        self.initialize_common_session_state()
        
        # FFBridge-specific initialization
        if 'player_id' in st.query_params:
            player_id = st.query_params['player_id']
            if not isinstance(player_id, str):
                st.error(f'player_id must be a string {player_id}')
                st.stop()
            st.session_state.player_id = player_id
        else:
            st.session_state.player_id = None

        cache_dir = 'cache'
        pathlib.Path(cache_dir).mkdir(exist_ok=True, parents=True)
        st.session_state.cache_dir = cache_dir

        # Initialize FFBridge Bearer Token
        initialize_ffbridge_bearer_token()
        
        # Initialize website-specific components
        self.initialize_website_specific()
        self.reset_game_data()
        
    def reset_game_data(self):
        """Reset FFBridge-specific game data."""
        # First reset common game data
        self.reset_common_game_data()
        
        # FFBridge-specific defaults
        ffbridge_defaults = {
            'organization_name_default': None,
            'team_id_default': None,
            'player_license_number_default': '9500754',  # default to my license number
            'partner_license_number_default': None,
            'route_url_default': None,
        }
        
        for key, value in ffbridge_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        # FFBridge-specific session variables
        ffbridge_session_vars = {
            'organization_name': st.session_state.organization_name_default,
            'team_id': st.session_state.team_id_default,
            'player_license_number': st.session_state.player_license_number_default,
            'partner_license_number': st.session_state.partner_license_number_default,
            'route_url': st.session_state.route_url_default,
            'game_urls_d': {},
            'person_organization_id': None,
            'nb_deals': None,
        }
        
        for key, value in ffbridge_session_vars.items():
            st.session_state[key] = value

    def initialize_website_specific(self):
        """Initialize FFBridge-specific components."""
        st.session_state.assistant_logo = 'https://github.com/BSalita/ffbridge-postmortem/blob/master/assets/logo_assistant.gif?raw=true'
        st.session_state.guru_logo = 'https://github.com/BSalita/ffbridge-postmortem/blob/master/assets/logo_guru.png?raw=true'
        st.session_state.game_name = 'ffbridge'
        
        # Initialize paths
        st.session_state.rootPath = pathlib.Path('e:/bridge/data')
        st.session_state.ffbridgePath = st.session_state.rootPath.joinpath('ffbridge')
        st.session_state.savedModelsPath = st.session_state.rootPath.joinpath('SavedModels')

        # Intro messages are now displayed in create_ui() when player_id is None

    def create_ui(self):
        """Creates the main UI structure for FFBridge."""
        self.create_sidebar()
        # If a new player was entered, refresh sidebar first then start report
        if st.session_state.get('deferred_start_report', False):
            # Ensure games are available
            if st.session_state.player_id is not None:
                populate_game_urls_for_player(str(st.session_state.player_id))
                # Start report on first available game
                game_urls = st.session_state.game_urls_d.get(st.session_state.player_id, {})
                if len(game_urls) > 0:
                    st.session_state.deferred_start_report = False
                    change_game_state(str(st.session_state.player_id), None)
                    st.session_state.sql_query_mode = False
                    # Defer clearing search inputs to before widget creation
                    st.session_state.clear_player_search = True
                    # Immediately rerun so the report renders without requiring Go
                    st.rerun()
                    return
        if not st.session_state.sql_query_mode:
            # Show Morty instructions if no player is selected
            if st.session_state.player_id is None:
                # Display intro messages when no player is selected
                with st.session_state.main_section_container.container():
                    # Display any persistent error message first
                    if hasattr(st.session_state, 'player_search_error'):
                        st.error(st.session_state.player_search_error)
                    
                    streamlit_chat.message(
                        "Hi. I'm Morty. Your friendly postmortem chatbot. I only want to chat about ffbridge pair matchpoint games using a Mitchell movement and not shuffled.",
                        key='intro_message_1',
                        logo=st.session_state.assistant_logo
                    )
                    streamlit_chat.message(
                        "I'm optimized for large screen devices such as a notebook or monitor. Do not use a smartphone.",
                        key='intro_message_2',
                        logo=st.session_state.assistant_logo
                    )
                    streamlit_chat.message(
                        "To start our postmortem chat, I'll need the a player number of your ffbridge game. It will be the subject of our chat.",
                        key='intro_message_3',
                        logo=st.session_state.assistant_logo
                    )
                    streamlit_chat.message(
                        "Enter the player number in the left sidebar or just re-enter the default player number. Press the enter key to begin.",
                        key='intro_message_4',
                        logo=st.session_state.assistant_logo
                    )
                    streamlit_chat.message(
                        "I'm just a Proof of Concept so don't double me.",
                        key='intro_message_5',
                        logo=st.session_state.assistant_logo
                    )
                    app_info()
            elif st.session_state.session_id is not None:
                st.session_state.report_rendering = True
                try:
                    print(f"Starting report generation for player_id={st.session_state.player_id}, session_id={st.session_state.session_id}")
                    self.write_report()
                    print(f"Report generation completed successfully")
                except Exception as e:
                    print(f"Exception during report generation: {e}")
                    import traceback
                    traceback.print_exc()
                    st.error(f"Error generating report: {str(e)}")
                finally:
                    st.session_state.report_rendering = False
                    
        self.ask_sql_query()

    def create_sidebar(self):
        """Create FFBridge-specific sidebar."""
        
        # Process Go button input OUTSIDE sidebar context (so output goes to main window)
        if hasattr(st.session_state, 'process_go_button_input'):
            input_value = st.session_state.process_go_button_input
            del st.session_state.process_go_button_input
            
            # Processing Go button input in main context (so output goes to main window)
            
            # If it's a license number (numeric), look up the player and generate report
            if input_value.isdigit():
                try:
                    # Make API call to find the player by license number
                    api_urls_d = {
                        'search': (f"https://api.ffbridge.fr/api/v1/search-members?alive=1&search={input_value}", False),
                    }
                    dfs, api_urls_d = get_ffbridge_data_using_url_licencie(api_urls_d, show_progress=False)
                    
                    if len(dfs['search']) == 0:
                        st.error(f"License number '{input_value}' not found.")
                    elif len(dfs['search']) == 1:
                        # Exactly one player found - get their player ID
                        row = list(dfs['search'].iter_rows(named=True))[0]
                        player_id = row['person_id']  # This is the actual player ID
                        
                        # Populate sidebar first, then defer report start until after sidebar refresh
                        st.session_state.player_id = str(player_id)
                        populate_game_urls_for_player(st.session_state.player_id)
                        st.session_state.deferred_start_report = True
                        return
                    else:
                        # Multiple players found - this shouldn't happen with exact license numbers
                        st.error(f"Multiple players found for license '{input_value}'. This is unexpected.")
                        
                except Exception as e:
                    st.error(f"Error looking up license {input_value}: {str(e)}")
            else:
                # If it's a search term, trigger search to show dialog
                player_search_input_on_change_with_query(input_value)
        
        # Modal dialog just updates the textbox - user must press Enter to generate report
        
        
        st.sidebar.caption(f"Build:{st.session_state.app_datetime}")

        # Style primary buttons in the sidebar (e.g., Go) to green
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] button[kind="primary"] {
                background-color: #2e7d32 !important;
                border-color: #2e7d32 !important;
                color: #ffffff !important;
            }
            [data-testid="stSidebar"] button[kind="primary"]:hover {
                background-color: #1b5e20 !important;
                border-color: #1b5e20 !important;
                color: #ffffff !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Player search with modal dialog
        # Use a separate value state that can be updated
        # Use the stored value, or empty for truly fresh searches
        # Clear the text input value before instantiation if flagged
        if st.session_state.get('clear_player_search'):
            st.session_state.player_search_input = ''
            st.session_state.player_search_value = ''
            del st.session_state.clear_player_search
        current_value = st.session_state.get('player_search_value', '')
            
        # Simple textbox with Go button below it (Enter also triggers)
        search_input = st.sidebar.text_input(
            "Enter ffbridge license number",
            value=current_value,
            key='player_search_input',
            on_change=player_search_input_on_change,
            placeholder="Enter license number",
            help="Enter ffbridge license number or (partial) last name."
        )
        # If user pressed Enter, the on_change above already ran; if it populated a player, set deferred start
        if (st.session_state.get('player_id') and
            st.session_state.get('player_search_value') == st.session_state.get('player_search_input') and
            not st.session_state.get('session_id') and
            not st.session_state.get('deferred_start_report')):
            # Ensure games are populated and then trigger report start on next cycle
            try:
                populate_game_urls_for_player(str(st.session_state.player_id))
            except Exception:
                pass
            st.session_state.deferred_start_report = True
        
        # Show Go button directly under the input (always visible; disabled until numeric)
        is_numeric_input = bool(current_value and current_value.strip().isdigit())
        if st.sidebar.button("Go", width="stretch", type="primary", disabled=not is_numeric_input):
            # Store the input for processing outside sidebar context
            st.session_state.process_go_button_input = current_value.strip()
            st.rerun()
        # Auto-start the report once when numeric input appears
        if (is_numeric_input and
            not st.session_state.get('session_id') and
            not st.session_state.get('deferred_start_report') and
            not st.session_state.get('auto_go_triggered')):
            st.session_state.process_go_button_input = current_value.strip()
            st.session_state.auto_go_triggered = True
            st.rerun()
        
        # Show instruction when license number is populated
        if current_value and current_value.strip().isdigit():
            st.sidebar.caption("👆 Click 'Go' to generate report")
        
        # Show modal dialog if we have matches AND the flag is set (meaning search processing is complete)
        if (st.session_state.get('show_player_modal', False) and
            hasattr(st.session_state, 'player_search_matches') and 
            st.session_state.player_search_matches):
            
            # Filter matches based on current textbox content
            current_input = st.session_state.get('player_search_input', '').lower()
            match_options = [match[0] for match in st.session_state.player_search_matches]
            
            # Further filter options based on current textbox input
            if current_input and len(current_input) > 0:
                filtered_options = [opt for opt in match_options if current_input in opt.lower()]
            else:
                filtered_options = match_options
            
            # Check one more time right before showing dialog
            if filtered_options and hasattr(st.session_state, 'player_search_matches'):
                # Clear the flag since we're now showing the modal
                st.session_state.show_player_modal = False
                # Show modal dialog with player selection
                show_player_selection_modal(filtered_options)

        if st.session_state.player_id is None:
            st.sidebar.caption("Select a player or enter a license to continue.")
        
        # If a player is selected but games haven't been loaded yet, try to populate them
        if (st.session_state.player_id is not None and
            st.session_state.player_id not in st.session_state.game_urls_d):
            try:
                populate_game_urls_for_player(str(st.session_state.player_id))
            except Exception as e:
                print(f"Auto-populate games failed for player {st.session_state.player_id}: {e}")

        self.read_configs()

        is_report_running = bool(st.session_state.get('report_rendering'))
        if st.session_state.player_id in st.session_state.game_urls_d:
            st.sidebar.selectbox(
                "Choose a club game.", 
                index=0, 
                options=[f"{k}, {v['description']}" for k, v in st.session_state.game_urls_d[st.session_state.player_id].items()], 
                on_change=club_session_id_on_change, 
                key='club_session_ids_selectbox'
            )
            # Show a small verification of how many games are available
            st.sidebar.caption(f"Games found: {len(st.session_state.game_urls_d.get(st.session_state.player_id, {}))}")

        # External links (after render pass completes, even on error)
        if not is_report_running and st.session_state.get('game_url'):
            st.sidebar.link_button('View ffbridge Webpage', url=st.session_state.get('game_url', ''))
            if st.session_state.get('route_url') is not None:
                st.sidebar.link_button('View Roy Rene Webpage', url=st.session_state.route_url)
        # Download Personalized Report PDF button placeholder (below the link button)
        st.session_state.pdf_link = st.sidebar.empty()

        # Automated Postmortem Apps
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Automated Postmortem Apps**")
        st.sidebar.markdown("🔗 [ACBL Postmortem](https://acbl.postmortem.chat)")
        st.sidebar.markdown("🔗 [French ffbridge Postmortem](https://ffbridge.postmortem.chat)")
        #st.sidebar.markdown("🔗 [BridgeWebs Postmortem](https://bridgewebs.postmortem.chat)")

        # Separator above Developer Settings
        st.sidebar.markdown("---")

        # Developer Settings moved to bottom
        with st.sidebar.expander('Developer Settings', False):
            st.number_input(
                "Single Dummy Samples Count",
                min_value=1,
                max_value=100,
                value=st.session_state.single_dummy_sample_count,
                on_change=single_dummy_sample_count_on_change,
                key='single_dummy_sample_count_number_input'
            )

            if st.button('Clear Cache', help='Clear cached files'):
                clear_cache()

            if st.session_state.debug_favorites is not None:
                st.session_state.debug_player_id_names = st.session_state.debug_favorites[
                    'SelectBoxes']['Player_IDs']['options']
                if len(st.session_state.debug_player_id_names):
                    st.selectbox(
                        "Debug Player List", 
                        options=st.session_state.debug_player_id_names, 
                        placeholder=st.session_state.player_id,
                        on_change=debug_player_id_names_change, 
                        key='debug_player_id_names_selectbox'
                    )

            st.checkbox(
                'Show SQL Query',
                value=st.session_state.show_sql_query,
                key='show_sql_query_checkbox',
                on_change=sql_query_on_change,
                help='Show SQL used to query dataframes.'
            )

            st.checkbox(
                'Enable Debug Mode',
                value=st.session_state.debug_mode,
                key='debug_mode_checkbox',
                on_change=debug_mode_on_change,
                help='Show SQL used to query dataframes.'
            )

def initialize_ffbridge_bearer_token() -> None:
    """Initialize FFBridge Bearer token from .env file or environment variables"""
    
    # first, check if website is working (without bearer token) and get version number.
    url = f"https://api-lancelot.ffbridge.fr/public/version"
    response = requests.get(url)
    response.raise_for_status()
    json_data = response.json()
    version = json_data['version']
    print(f"🔍 {url} returned version: {version}")

    # First, try to get token from .env file
    load_dotenv()
    token = os.getenv('FFBRIDGE_BEARER_TOKEN_LANCELOT')
    
    if token:
        st.session_state.ffbridge_bearer_token = token
        print(f"🔑 Bearer token loaded from .env file (first 20 chars): {token[:20]}...")
    else:
        # No token in .env file, show warning
        st.error("⚠️ No Bearer token found in .env file")
        return False

    token = os.getenv('FFBRIDGE_EASI_TOKEN')
    
    if token:
        st.session_state.ffbridge_easi_token = token
        print(f"🔑 EASI token loaded from .env file (first 20 chars): {token[:20]}...")
    else:
        # No token in .env file, show warning
        st.error("⚠️ No EASI token found in .env file")
        return False

    api_urls_d = {
        'my_infos': (f"https://api.ffbridge.fr/api/v1/users/my/infos", False),
    }

    dfs, api_urls_d = get_ffbridge_data_using_url_licencie(api_urls_d, show_progress=False)
    assert len(dfs['my_infos']) == 1, f"Expected 1 row, got {len(dfs['my_infos'])}"
    # Use proper Polars syntax to get first value
    st.session_state.player_id = dfs['my_infos']['person_id'].to_list()[0] # todo: remove this?
    st.session_state.player_license_number = dfs['my_infos']['person_license_number'].to_list()[0]

        # # Try to import automation functions
        # try:
        #     from ffbridge_auth_playwright import get_bearer_token_from_env, get_bearer_token_playwright_sync
            
        #     # Try once more with explicit load_dotenv
        #     token = get_bearer_token_from_env()
        #     if token:
        #         st.session_state.ffbridge_bearer_token = token
        #         print(f"🔑 Bearer token loaded via automation module: {token[:20]}...")
        #         return True
            
        # except ImportError:
        #     print("⚠️ Browser automation module not available")
        
        #         # Test both domains and show status
        # st.info("🔍 Testing API domains and tokens...")
        
        # # Test Lancelot domain
        # lancelot_token = get_token_for_domain("api-lancelot.ffbridge.fr")
        # if lancelot_token:
        #     st.success(f"✅ Lancelot domain token ready: {lancelot_token[:20]}...")
        # else:
        #     st.error("❌ No token for api-lancelot.ffbridge.fr")
        
        # # Test API domain (via easi-token)
        # api_token = get_token_for_domain("api.ffbridge.fr")
        # if api_token:
        #     st.success(f"✅ API domain easi-token ready: {api_token[:20]}...")
        # else:
        #     st.error("❌ No easi-token for api.ffbridge.fr")
        
        # if not lancelot_token and not api_token:
        #     st.error("❌ No valid tokens found. Please refresh tokens.")
        #     return {}

    return False

# def refresh_bearer_token():
#     """Refresh Bearer token using browser automation"""
#     try:
#         from ffbridge_auth_playwright import get_bearer_token_playwright_sync
        
#         with st.spinner("🤖 Running browser automation to refresh Bearer token..."):
#             token = get_bearer_token_playwright_sync()
            
#         if token:
#             st.session_state.ffbridge_bearer_token = token
#             st.success("✅ Bearer token refreshed successfully!")
#             st.rerun()  # Refresh the page to update UI
#             return True
#         else:
#             st.error("❌ Failed to refresh Bearer token")
#             return False
            
#     except ImportError:
#         st.error("❌ Browser automation module not available. Please install playwright and python-dotenv.")
#         return False
#     except Exception as e:
#         st.error(f"❌ Error refreshing token: {e}")
#         return False

if __name__ == "__main__":
    main()
