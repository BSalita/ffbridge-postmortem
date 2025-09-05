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


import pathlib
import pandas as pd # only used for __version__ for now. might need for plotting later as pandas plotting support is better than polars.
import polars as pl
import requests
import duckdb
import json
import sys
import os
import platform
#import asyncio
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

def ShowDataFrameTable(df: pl.DataFrame, key: str, query: str = 'SELECT * FROM self', show_sql_query: bool = True) -> Optional[pl.DataFrame]:
    """Display a DataFrame table in Streamlit with optional SQL query execution
    
    Args:
        df: The Polars DataFrame to display
        key: Unique key for the Streamlit component
        query: SQL query to execute on the DataFrame
        show_sql_query: Whether to display the SQL query text
        
    Returns:
        Result DataFrame from SQL query, or None if query failed
    """
    if show_sql_query and st.session_state.show_sql_query:
        st.text(f"SQL Query: {query}")

    # if query doesn't contain 'FROM self', add 'FROM self ' to the beginning of the query.
    # can't just check for startswith 'from self'. Not universal because 'from self' can appear in subqueries or after JOIN.
    # this syntax makes easy work of adding FROM but isn't compatible with polars SQL. duckdb only.
    if 'from self' not in query.lower():
        query = 'FROM self ' + query

    # polars SQL has so many issues that it's impossible to use. disabling until 2030.
    # try:
    #     # First try using Polars SQL. However, Polars doesn't support some SQL functions: string_agg(), agg_value(), some joins are not supported.
    #     if True: # workaround issued by polars. CASE WHEN AVG() ELSE AVG() -> AVG(CASE WHEN ...)
    #         result_df = st.session_state.con.execute(query).pl()
    #     else:
    #         result_df = df.sql(query) # todo: enforce FROM self for security concerns?
    # except Exception as e:
    #     try:
    #         # If Polars fails, try DuckDB
    #         print(f"Polars SQL failed. Trying DuckDB: {e}")
    #         result_df = st.session_state.con.execute(query).pl()
    #     except Exception as e2:
    #         st.error(f"Both Polars and DuckDB SQL engines have failed. Polars error: {e}, DuckDB error: {e2}. Query: {query}")
    #         return None
    
    try:
        con = get_session_duckdb_connection()
        result_df = con.execute(query).pl()
        if show_sql_query and st.session_state.show_sql_query:
            st.text(f"Result is a dataframe of {len(result_df)} rows.")
        streamlitlib.ShowDataFrameTable(result_df, key) # requires pandas dataframe.
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


def group_id_on_change() -> None:
    """Handle group ID input change event"""
    st.session_state.sql_query_mode = False


def session_id_on_change() -> None:
    """Handle session ID input change event"""
    st.session_state.sql_query_mode = False


def debug_player_id_names_change() -> None:
    # assign changed selectbox value (debug_player_id_names_selectbox). e.g. ['2663279','Robert Salita']
    player_id_name = st.session_state.debug_player_id_names_selectbox
    change_game_state(player_id_name[0], None)


def team_id_on_change() -> None:
    """Handle team ID input change event"""
    st.session_state.sql_query_mode = False


def simultane_id_on_change() -> None:
    """Handle simultane ID input change event"""
    st.session_state.sql_query_mode = False


def teams_id_on_change() -> None:
    """Handle teams ID input change event"""
    st.session_state.sql_query_mode = False


def org_id_on_change() -> None:
    """Handle organization ID input change event"""
    st.session_state.sql_query_mode = False


def player_license_number_on_change() -> None:
    """Handle player license number input change event"""
    st.session_state.sql_query_mode = False


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
        df = pl.read_parquet(parquet_cache_file)
        
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

    match k:
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
                raise Exception(f"Failed to get data from {url}")
    
            # Create DataFrame from the JSON response. json_data can be a dict or a list.
            df = pl.DataFrame(pd.json_normalize(json_data, sep='_'))
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


def change_game_state(player_id: str, session_id: str) -> None: # todo: rename to session_id?

    print(f"Retrieving latest results for {player_id}")

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
                dfs, api_urls_d = get_ffbridge_licencie_get_urls(api_urls_d)
                st.session_state.game_urls_d[player_id] = {k:v for k,v in zip(dfs['person']['tournament_id'], dfs['person'].to_dicts())}
                st.session_state.person_organization_id = dfs['members']['seasons_organization_id'] # person's signup organization id e.g. 1212 for St Honore BC
        game_urls = st.session_state.game_urls_d[player_id]
        if game_urls is None:
            st.error(f"Player number {player_id} not found.")
            return False
        if len(game_urls) == 0:
            st.error(f"Could not find any games for {player_id}.")
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
            'organization_by_person_organization_id': (f"https://api.ffbridge.fr/api/v1/licensee-results/results/organization/{st.session_state.org_id}?date=all&person_organization_id={st.session_state.person_organization_id}&place=0&type=0", False),
            'person_by_person_organization_id': (f"https://api.ffbridge.fr/api/v1/licensee-results/results/person/{st.session_state.player_id}?date=all&person_organization_id={st.session_state.person_organization_id}&place=0&type=0", False),
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
                # e.g. "https://www.bridgeplus.com/nos-simultanes/resultats/?p=route&res=sim&tr=S202602&cl=5802079&sc=A&eq=212"
                st.session_state.route_url = f"https://www.bridgeplus.com/nos-simultanes/resultats/?p=route&res=sim&eq={st.session_state.team_number}&tr={st.session_state.tournament_id}&cl={st.session_state.organization_code}&sc={st.session_state.section_name}"
                print(f"Getting route data from: {st.session_state.route_url}")
                if False:
                    # calls internal async version which takes 60s. almost 3x faster than asyncio version below
                    #  -- but this version doesn't show progress bar -- might overwhelm server as I'm getting blacklisted(?).
                    boards_dfs = mlBridgeLib.mlBridgeBPLib.get_all_boards_for_player(st.session_state.tournament_id, st.session_state.organization_code, st.session_state.player_license_number, max_deals=36)
                else:
                    # Get boards data with progress bar by processing boards one by one using the existing function
                    boards_dfs = {'boards': None, 'score_frequency': None}
                    
                    try:
                        import asyncio
                        
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
            if st.session_state.pair_direction == 'NS':
                boards_df = boards_df.join(player_n_df,left_on=('club_id','Pair_Number'),right_on=('team_organization_code','team_table_number'),how='inner')
                boards_df = boards_df.join(player_e_df,left_on=('club_id','Opponent_Pair_Number'),right_on=('team_organization_code','team_table_number'),how='inner')
                boards_df = boards_df.join(player_s_df,left_on=('club_id','Pair_Number'),right_on=('team_organization_code','team_table_number'),how='inner')
                boards_df = boards_df.join(player_w_df,left_on=('club_id','Opponent_Pair_Number'),right_on=('team_organization_code','team_table_number'),how='inner')
                boards_df = boards_df.with_columns([
                    pl.col('Pair_Number').alias('Pair_Number_NS'),
                    pl.col('Opponent_Pair_Number').alias('Pair_Number_EW'),
                ])
            else:
                boards_df = boards_df.join(player_n_df,left_on=('club_id','Opponent_Pair_Number'),right_on=('team_organization_code','team_table_number'),how='inner')
                boards_df = boards_df.join(player_e_df,left_on=('club_id','Pair_Number'),right_on=('team_organization_code','team_table_number'),how='inner')
                boards_df = boards_df.join(player_s_df,left_on=('club_id','Opponent_Pair_Number'),right_on=('team_organization_code','team_table_number'),how='inner')
                boards_df = boards_df.join(player_w_df,left_on=('club_id','Pair_Number'),right_on=('team_organization_code','team_table_number'),how='inner')
                boards_df = boards_df.with_columns([
                    pl.col('Pair_Number').alias('Pair_Number_EW'),
                    pl.col('Opponent_Pair_Number').alias('Pair_Number_NS'),
                ])
            df = boards_df.join(df, on='Board', how='left')
        else:
            df = mlBridgeFFLib.convert_ffdf_api_to_mldf(dfs)

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
                    df = pl.read_parquet(ffbridge_session_player_cache_df_file)
                    print(f"Loaded {ffbridge_session_player_cache_df_filename}: shape:{df.shape} size:{ffbridge_session_player_cache_df_file.stat().st_size}")
                else:
                    with st.spinner('Creating ffbridge data to dataframe...'):
                        df = augment_df(df)
                    if df is not None:
                        st.rerun() # todo: not sure what is needed to recover from error:
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

    return False


def on_game_url_input_change() -> None:
    """Handle game URL input change event"""
    st.session_state.game_url = st.session_state.game_url_input
    if change_game_state(st.session_state.player_id, None):
        st.session_state.game_url_default = ''
        reset_game_data()


def player_search_input_on_change() -> None:
    # todo: looks like there's some situation where this is not called because player_search_input is already set. Need to breakpoint here to determine why st.session_state.player_id isn't updated.
    # assign changed textbox value (player_search_input) to player_id
    player_search_input = st.session_state.player_search_input
    api_urls_d = {
        'search': (f"https://api.ffbridge.fr/api/v1/search-members?alive=1&search={player_search_input}", False),
    }
    dfs, api_urls_d = get_ffbridge_data_using_url_licencie(api_urls_d, show_progress=False)
    if len(dfs['search']) == 0:
        return
    assert len(dfs['search']) == 1, f"Expected 1 row, got {len(dfs['search'])}"
    player_id = dfs['search']['person_id'][0] # todo: remove this?
    change_game_state(player_id, None)
    st.session_state.sql_query_mode = False



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
    """Create the main sidebar interface"""

    st.sidebar.caption(st.session_state.app_datetime)

    st.sidebar.text_input(
        "Enter ffbridge player license number", on_change=player_search_input_on_change, placeholder=st.session_state.player_license_number, key='player_search_input')

    if st.session_state.player_id is None:
        return

    st.sidebar.selectbox("Choose a club game.", index=0, options=[f"{k}, {v['description']}" for k, v in st.session_state.game_urls_d[st.session_state.player_id].items(
    )], on_change=club_session_id_on_change, key='club_session_ids_selectbox')  # options are event_id + event description
    
    read_configs()

    st.sidebar.link_button('View ffbridge Webpage', url=st.session_state.game_url)
    if st.session_state.route_url is not None:
        st.sidebar.link_button('View Roy Rene Webpage', url=st.session_state.route_url)
    st.session_state.pdf_link = st.sidebar.empty()

    # create_sidebar_ffbridge_licencie()? create_sidebar_ffbridge_licencie()? neither?

    with st.sidebar.expander('Developer Settings', False):

        # don't use st.sidebar... in expander.
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
            # favorite prompts selectboxes
            st.session_state.debug_player_id_names = st.session_state.debug_favorites[
                'SelectBoxes']['Player_IDs']['options']
            if len(st.session_state.debug_player_id_names):
                # changed placeholder to player_id because when selectbox gets reset, possibly due to expander auto-collapsing, we don't want an unexpected value.
                # test player_id is not None else use debug_favorites['SelectBoxes']['player_ids']['placeholder']?
                st.selectbox("Debug Player List", options=st.session_state.debug_player_id_names, placeholder=st.session_state.player_id, #.debug_favorites['SelectBoxes']['player_ids']['placeholder'],
                                        on_change=debug_player_id_names_change, key='debug_player_id_names_selectbox')

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

    return


# disabled for now. not sure if needed. maybe show in developer settings?
def create_sidebar_ffbridge_licencie() -> None:
    """Create sidebar for FFBridge licencie interface"""
    # st.session_state.simultane_id = 34424 # simultane id for the game
    # st.session_state.team_id = 4818526 # team id for the game
    # st.session_state.org_id = 1634 # e.g. 1634 is Levallois-Perret
    # st.session_state.player_id = 597539 # both player_id and person
    # st.session_state.person_organization_id = 1212 # e.g. is the person's signup organization id e.g. 1212 for St Honore BC
    print(f"{st.session_state.simultane_id=} {st.session_state.team_id=} {st.session_state.org_id=} {st.session_state.player_id=} {st.session_state.person_organization_id=}")

    # Provide a "Load Game URL" button.
    # if st.sidebar.button("Analyze Game"):
    #     st.session_state.sql_query_mode = False
    #     if change_game_state(st.session_state.player_id, None):
    #         st.session_state.game_url_default = ''
    #         reset_game_data()
    # st.sidebar.link_button('View Game Webpage', url=st.session_state.game_url)
    # st.session_state.pdf_link = st.sidebar.empty()
    st.sidebar.markdown("#### Game Retrival Settings")
    st.session_state.group_id = st.sidebar.number_input(
        'simultane_id',
        value=st.session_state.simultane_id,
        key='sidebar_simultane_id',
        on_change=simultane_id_on_change,
        help='Enter ffbridge simultane_id. e.g. 34424'
    )
    st.session_state.team_id = st.sidebar.number_input(
        'teams_id',
        value=st.session_state.team_id,
        key='sidebar_teams_id',
        on_change=teams_id_on_change,
        help='Enter ffbridge teams id. e.g. 4818526'
    )
    st.session_state.org_id = st.sidebar.number_input(
        'org_id',
        value=st.session_state.org_id,
        key='sidebar_org_id',
        on_change=org_id_on_change,
        help='Enter ffbridge org id. e.g. 1634'
    )
    st.session_state.org_id = st.sidebar.text_input(
        'player_license_number',
        value=st.session_state.player_license_number,
        key='sidebar_player_license_number',
        on_change=player_license_number_on_change,
        help='Enter ffbridge player license id. e.g. 9500754'
    )

    if st.session_state.player_id is None: # todo: not quite right. value is not updated with player_id if previously None. unsure why.
        return

    return


# disabled for now. not sure if needed. maybe show in developer settings? Only show setting as read-only?
def create_sidebar_ffbridge() -> None:
    """Create sidebar for FFBridge interface"""

    # if extract_group_id_session_id_team_id():
    #     st.error("Invalid game URL. Please enter a valid game URL.")
    #     return

    # # Provide a "Load Game URL" button.
    # if st.sidebar.button("Analyze Game"):
    #     st.session_state.sql_query_mode = False
    #     if change_game_state(st.session_state.player_id, None):
    #         st.session_state.game_url_default = ''
    #         reset_game_data()

    # When the full sidebar is to be shown:
    # --- Check if the "Analyze Game" button has been hit ---
    #if not st.session_state.analysis_started:
    # st.sidebar.link_button('View Game Webpage', url=st.session_state.game_url)
    # st.session_state.pdf_link = st.sidebar.empty()
    st.sidebar.markdown("#### Game Retrival Settings")
    st.session_state.group_id = st.sidebar.number_input(
        'Group ID',
        value=st.session_state.group_id,
        key='sidebar_group_id',
        on_change=group_id_on_change,
        help='Enter ffbridge group id. e.g. 7878 for Bridge Club St. Honore'
    )
    st.session_state.session_id = st.sidebar.number_input(
        'Session ID',
        value=st.session_state.session_id,
        key='sidebar_session_id',
        on_change=session_id_on_change,
        help='Enter ffbridge session id. e.g. 107118'
    )
    
    st.session_state.team_id = st.sidebar.number_input(
        'Pairs ID',
        value=st.session_state.team_id,
        key='sidebar_team_id',
        on_change=team_id_on_change,
        help='Enter ffbridge pairs id. e.g. 3976783'
    )
    return


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
    return


# Everything below here is the standard mlBridge code.


# this version of perform_hand_augmentations_locked() uses self for class compatibility, older versions did not.
def perform_hand_augmentations_queue(self, hand_augmentation_work: Any) -> None:
    """Perform hand augmentations queue processing
    
    Args:
        hand_augmentation_work: Work item for hand augmentation processing
    """
    return streamlitlib.perform_queued_work(self, hand_augmentation_work, "Hand analysis")


def augment_df(df: pl.DataFrame) -> pl.DataFrame:
    """Augment DataFrame with additional bridge analysis data
    
    Args:
        df: Input DataFrame containing bridge game data
        
    Returns:
        Augmented DataFrame with additional analysis columns
    """
    with st.spinner('Augmenting data...'):
        augmenter = AllAugmentations(df,None,sd_productions=st.session_state.single_dummy_sample_count,progress=st.progress(0),lock_func=perform_hand_augmentations_queue)
        df, hrs_cache_df = augmenter.perform_all_augmentations()
    # with st.spinner('Creating hand data...'):
    #     augmenter = HandAugmenter(df,{},sd_productions=st.session_state.single_dummy_sample_count,progress=st.progress(0),lock_func=perform_hand_augmentations_queue)
    #     df = augmenter.perform_hand_augmentations()
    # with st.spinner('Augmenting with result data...'):
    #     augmenter = ResultAugmenter(df,{})
    #     df = augmenter.perform_result_augmentations()
    # with st.spinner('Augmenting with contract data...'):
    #     augmenter = ScoreAugmenter(df)
    #     df = augmenter.perform_score_augmentations()
    # with st.spinner('Augmenting with DD and SD data...'):
    #     augmenter = DDSDAugmenter(df)
    #     df = augmenter.perform_dd_sd_augmentations()
    # with st.spinner('Augmenting with matchpoints and percentages data...'):
    #     augmenter = MatchPointAugmenter(df)
    #     df = augmenter.perform_matchpoint_augmentations()
    return df


def read_configs() -> Dict[str, Any]:
    """Read configuration files and return configuration dictionary
    
    Returns:
        Dictionary containing configuration settings
    """

    st.session_state.default_favorites_file = pathlib.Path(
        'default.favorites.json')
    st.session_state.player_id_custom_favorites_file = pathlib.Path(
        f'favorites/{st.session_state.player_id}.favorites.json')
    st.session_state.debug_favorites_file = pathlib.Path(
        'favorites/debug.favorites.json')

    if st.session_state.default_favorites_file.exists():
        with open(st.session_state.default_favorites_file, 'r') as f:
            favorites = json.load(f)
        st.session_state.favorites = favorites
        #st.session_state.vetted_prompts = get_vetted_prompts_from_favorites(favorites)
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

    # display missing prompts in favorites
    if 'missing_in_summarize' not in st.session_state:
        # Get the prompts from both locations
        summarize_prompts = st.session_state.favorites['Buttons']['Summarize']['prompts']
        vetted_prompts = st.session_state.favorites['SelectBoxes']['Vetted_Prompts']

        # Process the keys to ignore leading '@'
        st.session_state.summarize_keys = {p.lstrip('@') for p in summarize_prompts}
        st.session_state.vetted_keys = set(vetted_prompts.keys())

        # Find items in summarize_prompts but not in vetted_prompts. There should be none.
        st.session_state.missing_in_vetted = st.session_state.summarize_keys - st.session_state.vetted_keys
        assert len(st.session_state.missing_in_vetted) == 0, f"Oops. {st.session_state.missing_in_vetted} not in {st.session_state.vetted_keys}."

        # Find items in vetted_prompts but not in summarize_prompts. ok if there's some missing.
        st.session_state.missing_in_summarize = st.session_state.vetted_keys - st.session_state.summarize_keys

        print("\nItems in Vetted_Prompts but not in Summarize.prompts:")
        for item in st.session_state.missing_in_summarize:
            print(f"- {item}: {vetted_prompts[item]['title']}")
    return


def process_prompt_macros(sql_query: str) -> str:
    """Process SQL query macros and replace them with actual values
    
    Args:
        sql_query: Input SQL query string with macros
        
    Returns:
        Processed SQL query with macros replaced
    """
    replacements = {
        '{Player_Direction}': st.session_state.player_direction,
        '{Partner_Direction}': st.session_state.partner_direction,
        '{Pair_Direction}': st.session_state.pair_direction,
        '{Opponent_Pair_Direction}': st.session_state.opponent_pair_direction
    }
    for old, new in replacements.items():
        if new is None:
            continue
        sql_query = sql_query.replace(old, new)
    return sql_query


def write_report() -> None:
    """Write and display the bridge game analysis report"""
    # bar_format='{l_bar}{bar}' isn't working in stqdm. no way to suppress r_bar without editing stqdm source code.
    # todo: need to pass the Button title to the stqdm description. this is a hack until implemented.
    st.session_state.main_section_container = st.container(border=True)
    with st.session_state.main_section_container:
        report_title = f"Bridge Game Postmortem Report" # can't use any of '():' because of href link below.
        report_person = f"Personalized for {st.session_state.player_name} ({st.session_state.player_license_number})"
        report_creator = f"Created by https://{st.session_state.game_name}.postmortem.chat"
        report_event_info = f"{st.session_state.organization_name} {st.session_state.game_description} (event id {st.session_state.session_id}) on {datetime.strptime(st.session_state.tournament_date, '%Y-%m-%d').strftime('%d-%b-%Y')} ."
        report_game_results_webpage = f"ffbridge Results Page: {st.session_state.game_url}"
        if st.session_state.route_url is not None:
            report_roy_rene_game_results_webpage = f"Roy Rene Results Page: {st.session_state.route_url}"
        report_your_match_info = f"Your pair was {st.session_state.team_id} {st.session_state.pair_direction} in section {st.session_state.section_name}. You played {st.session_state.player_direction}. Your partner was {st.session_state.partner_name} ({st.session_state.partner_license_number}) who played {st.session_state.partner_direction}."
        #st.markdown('<div style="height: 50px;"><a name="top-of-report"></a></div>', unsafe_allow_html=True)
        st.markdown(f"### {report_title}")
        st.markdown(f"#### {report_person}")
        st.markdown(f"##### {report_creator}")
        st.markdown(f"#### {report_event_info}")
        st.markdown(f"##### {report_game_results_webpage}")
        if st.session_state.route_url is not None:
            st.markdown(f"##### {report_roy_rene_game_results_webpage}")
        st.markdown(f"#### {report_your_match_info}")
        pdf_assets = st.session_state.pdf_assets
        pdf_assets.clear()
        pdf_assets.append(f"# {report_title}")
        pdf_assets.append(f"#### {report_person}")
        pdf_assets.append(f"#### {report_creator}")
        pdf_assets.append(f"### {report_event_info}")
        pdf_assets.append(f"#### {report_game_results_webpage}")
        pdf_assets.append(f"### {report_your_match_info}")
        st.session_state.button_title = 'Summarize' # todo: generalize to all buttons!
        selected_button = st.session_state.favorites['Buttons'][st.session_state.button_title]
        vetted_prompts = st.session_state.favorites['SelectBoxes']['Vetted_Prompts']
        sql_query_count = 0
        for stats in stqdm(selected_button['prompts'], desc='Creating personalized report...'):
            assert stats[0] == '@', stats
            stat = vetted_prompts[stats[1:]]
            for i, prompt in enumerate(stat['prompts']):
                if 'sql' in prompt and prompt['sql']:
                    #print('sql:',prompt["sql"])
                    if i == 0:
                        streamlit_chat.message(f"Morty: {stat['help']}", key=f'morty_sql_query_{sql_query_count}', logo=st.session_state.assistant_logo)
                        pdf_assets.append(f"### {stat['help']}")
                    prompt_sql = prompt['sql']
                    sql_query = process_prompt_macros(prompt_sql)
                    query_df = ShowDataFrameTable(st.session_state.df, query=sql_query, key=f'sql_query_{sql_query_count}')
                    if query_df is not None:
                        pdf_assets.append(query_df)
                    sql_query_count += 1

        # As an html button (needs styling added)
        # can't use link_button() restarts page rendering. markdown() will correctly jump to href.
        # st.link_button('Go to top of report',url='#your-personalized-report')\
        # report_title_anchor = report_title.replace(' ','-').lower()
        # Go to top button using simple anchor link (centered)
        st.markdown('''
            <div style="text-align: center; margin: 20px 0;">
                <a href="#top-of-report" style="text-decoration: none;">
                    <button style="padding: 8px 16px; background-color: #ff4b4b; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px;">
                        Go to top of report
                    </button>
                </a>
            </div>
        ''', unsafe_allow_html=True)

    if 'pdf_link' in st.session_state: # shouldn't happen?
        if st.session_state.pdf_link.download_button(label="Download Personalized Report",
                data=streamlitlib.create_pdf(st.session_state.pdf_assets, title=f"Bridge Game Postmortem Report Personalized for {st.session_state.player_id}"),
                file_name = f"{st.session_state.session_id}-{st.session_state.player_id}-morty.pdf",
                disabled = len(st.session_state.pdf_assets) == 0,
                mime='application/octet-stream',
                key='personalized_report_download_button'):
            st.warning('Personalized report downloaded.')
        return


def ask_sql_query() -> None:
    """Handle SQL query input and display results"""

    if st.session_state.show_sql_query:
        with st.container():
            with bottom():
                st.chat_input('Enter a SQL query e.g. SELECT PBN, Contract, Result, N, S, E, W', key='main_prompt_chat_input', on_submit=chat_input_on_submit)


def create_ui() -> None:
    """Create the main user interface"""
    create_sidebar()
    if not st.session_state.sql_query_mode:
        #create_tab_bar()
        if st.session_state.session_id is not None:
            write_report()
    ask_sql_query()


def get_session_duckdb_connection():
    """Get or create a DuckDB connection for the current session
    
    Returns:
        duckdb.DuckDBPyConnection: Session-specific DuckDB connection
    """
    if 'con' not in st.session_state or st.session_state.con is None:
        st.session_state.con = duckdb.connect()  # In-memory database per session
        print(f"Created new DuckDB connection for session")
    
    return st.session_state.con


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables"""
    st.set_page_config(layout="wide")
    # Add this auto-scroll code
    streamlitlib.widen_scrollbars()

    if platform.system() == 'Windows': # ugh. this hack is required because torch somehow remembers the platform where the model was created. Must be a bug. Must lie to torch.
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath
    
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

    first_time_defaults = {
        'first_time': True,
        'single_dummy_sample_count': 10,
        'debug_mode': os.getenv('STREAMLIT_ENV') == 'development',
        'show_sql_query': os.getenv('STREAMLIT_ENV') == 'development',
        'use_historical_data': False,
        'do_not_cache_df': True, # todo: set to True for production
        # 'con' removed from defaults - will be created per-session below
        'con_register_name': 'self',
        'main_section_container': st.empty(),
        'app_datetime': datetime.fromtimestamp(pathlib.Path(__file__).stat().st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z'),
        'current_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'cache_dir': cache_dir,
    }
    for key, value in first_time_defaults.items():
        st.session_state[key] = value

    # Create a per-session DuckDB connection to avoid concurrency issues
    # Each user session gets its own isolated database connection
    get_session_duckdb_connection()

    initialize_website_specific()

    reset_game_data()
    return


def reset_game_data() -> None:
    """Reset game data to default values"""

    # Default values for session state variables
    reset_defaults = {
        'organization_name_default': None,
        'game_description_default': None,
        'group_id_default': None,
        'session_id_default': None,
        'section_name_default': None,
        'player_id_default': None,
        'partner_id_default': None,
        'player_license_number_default': '9500754', # default to my license number.
        'partner_license_number_default': None,
        'player_name_default': None,
        'partner_name_default': None,
        'player_direction_default': None,
        'partner_direction_default': None,
        'team_id_default': None,
        'pair_direction_default': None,
        'opponent_pair_direction_default': None,
        'route_url_default': None,
    }
    
    # Initialize default values if not already set
    for key, value in reset_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize additional session state variables that depend on defaults.
    reset_session_vars = {
        'df': None,
        'organization_name': st.session_state.organization_name_default,
        'game_description': st.session_state.game_description_default,
        'group_id': st.session_state.group_id_default,
        'session_id': st.session_state.session_id_default,
        'section_name': st.session_state.section_name_default,
        'player_id': st.session_state.player_id_default,
        'partner_id': st.session_state.partner_id_default,
        'player_license_number': st.session_state.player_license_number_default,
        'partner_license_number': st.session_state.partner_license_number_default,
        'player_name': st.session_state.player_name_default,
        'partner_name': st.session_state.partner_name_default,
        'player_direction': st.session_state.player_direction_default,
        'partner_direction': st.session_state.partner_direction_default,
        'team_id': st.session_state.team_id_default,
        'pair_direction': st.session_state.pair_direction_default,
        'opponent_pair_direction': st.session_state.opponent_pair_direction_default,
        'route_url': st.session_state.route_url_default,
        #'sidebar_loaded': False,
        'analysis_started': False,   # new flag for analysis sidebar rewrite
        'vetted_prompts': [],
        'pdf_assets': [],
        'sql_query_mode': False,
        'sql_queries': [],
        'game_urls_d': {},
        'tournament_session_urls_d': {},
    }
    
    for key, value in reset_session_vars.items():
        #if key not in st.session_state:
        st.session_state[key] = value

    return


def app_info() -> None:
    """Display application information and version details"""
    st.caption(f"Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in streamlit. Data engine is polars. Query engine is duckdb. Bridge lib is endplay. Self hosted using Cloudflare Tunnel. Repo:https://github.com/BSalita/ffbridge-postmortem")
    st.caption(
        f"App:{st.session_state.app_datetime} Python:{'.'.join(map(str, sys.version_info[:3]))} Streamlit:{st.__version__} Pandas:{pd.__version__} polars:{pl.__version__} endplay:{endplay.__version__} Query Params:{st.query_params.to_dict()}")
    return


def main() -> None:
    """Main application entry point"""
    if 'first_time' not in st.session_state:
        initialize_session_state()
        create_sidebar()
    else:
        create_ui()
    return


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
    st.session_state.player_id = dfs['my_infos']['person_id'][0] # todo: remove this?
    st.session_state.player_license_number = dfs['my_infos']['person_license_number'][0]

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
