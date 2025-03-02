# streamlit program to display French Bridge (ffbridge) game results and statistics.
# Invoke from system prompt using: streamlit run ffbridge_streamlit.py

# todo showstoppers:
# download all board results and not just the player's table. this will provide proper data for all columns e.g. matchpoints, comparisions to other tables results
# game date? do we have to get a new URL?
# there's errors in dd calculations when page is refreshed in middle of dd calculations. must be a global variable issue or re-init of dll.
# missing spinners when entering a new game url. this may just be moving code to re run area.

# todo lower priority:
# do something with packages/version/authors stuff at top of page.
# test that non-pair games are rejected.
# change player_id dtype from int to string? what about others?
# don't destroy st.container()? append new dataframes to bottom of container.

# todo ffbridge:
# given a player_id, how to get recent games?
# unblock my ip address


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
from datetime import datetime, timezone
#from dotenv import load_dotenv

from urllib.parse import urlparse
from typing import Dict, Any, List

import endplay # for __version__

# Only declared to display version information
#import fastai
import numpy as np
import pandas as pd
#import safetensors
#import sklearn
#import torch

# assumes symlinks are created in current directory.
sys.path.append(str(pathlib.Path.cwd().joinpath('streamlitlib')))  # global
sys.path.append(str(pathlib.Path.cwd().joinpath('mlBridgeLib')))  # global # Requires "./mlBridgeLib" be in extraPaths in .vscode/settings.json
sys.path.append(str(pathlib.Path.cwd().joinpath('ffbridgelib')))  # global

import ffbridgelib
import streamlitlib
#import mlBridgeLib
import mlBridgeAugmentLib # Requires "./mlBridgeLib" be in extraPaths in .vscode/settings.json
#import mlBridgeEndplayLib


def ShowDataFrameTable(df, key, query='SELECT * FROM self', show_sql_query=True):
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
        result_df = st.session_state.con.execute(query).pl()
        if show_sql_query and st.session_state.show_sql_query:
            st.text(f"Result is a dataframe of {len(result_df)} rows.")
        streamlitlib.ShowDataFrameTable(result_df, key) # requires pandas dataframe.
    except Exception as e:
        st.error(f"duckdb exception: error:{e} query:{query}")
        return None
    
    return result_df


def game_url_on_change():
    st.session_state.game_url = st.session_state.create_sidebar_game_url_on_change
    st.session_state.sql_query_mode = False


def chat_input_on_submit():
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


def single_dummy_sample_count_changed():
    st.session_state.single_dummy_sample_count = st.session_state.single_dummy_sample_count_number_input
    change_game_state(st.session_state.player_id, st.session_state.session_id)


def sql_query_on_change():
    st.session_state.sql_query_mode = False
    #st.session_state.show_sql_query = st.session_state.create_sidebar_show_sql_query_checkbox
    # if 'df' in st.session_state: # todo: is this still needed?
    #     st.session_state.df = load_historical_data()


def group_id_on_change():
    st.session_state.sql_query_mode = False
    #st.session_state.group_id = st.session_state.create_sidebar_group_id
    # if 'df' in st.session_state: # todo: is this still needed?
    #     st.session_state.df = load_historical_data()


def session_id_on_change():
    st.session_state.sql_query_mode = False
    #st.session_state.session_id = st.session_state.create_sidebar_session_id
    # if 'df' in st.session_state: # todo: is this still needed?
    #     st.session_state.df = load_historical_data()


def player_id_on_change():
    st.session_state.sql_query_mode = False
    #st.session_state.group_id = st.session_state.create_sidebar_player_id
    # if 'df' in st.session_state: # todo: is this still needed?
    #     st.session_state.df = load_historical_data()


def partner_id_on_change():
    st.session_state.sql_query_mode = False
    #st.session_state.partner_id = st.session_state.create_sidebar_partner_id
    # if 'df' in st.session_state: # todo: is this still needed?
    #     st.session_state.df = load_historical_data()


# def load_historical_data():
#     ffbridge_experimental_data_df_filename = f'ffbridge_training_data_df.parquet'
#     ffbridge_experimental_data_df_file = st.session_state.dataPath.joinpath(ffbridge_experimental_data_df_filename)
#     df = pl.read_parquet(ffbridge_experimental_data_df_file)
#     print(f"Loaded {ffbridge_experimental_data_df_filename}: shape:{df.shape} size:{ffbridge_experimental_data_df_file.stat().st_size}")
#     return df


def filter_dataframe(df, group_id, session_id, player_id, partner_id):
    # First filter for sessions containing player_id

    df = df.filter(
        pl.col('group_id').eq(group_id) &
        pl.col('session_id').eq(session_id)
    )
    
    # Set direction variables based on where player_id is found
    player_direction = None
    if player_id in df['Player_ID_N']:
        player_direction = 'N'
        partner_direction = 'S'
        pair_direction = 'NS'
        opponent_pair_direction = 'EW'
    elif player_id in df['Player_ID_E']:
        player_direction = 'E'
        partner_direction = 'W'
        pair_direction = 'EW'
        opponent_pair_direction = 'NS'
    elif player_id in df['Player_ID_S']:
        player_direction = 'S'
        partner_direction = 'N'
        pair_direction = 'NS'
        opponent_pair_direction = 'EW'
    elif player_id in df['Player_ID_W']:
        player_direction = 'W'
        partner_direction = 'E'
        pair_direction = 'EW'
        opponent_pair_direction = 'NS'

    # todo: not sure what to do here. pbns might not contain names or ids. endplay has names but not ids.
    if player_direction is None:
        df = df.with_columns(
            pl.lit(True).alias('Boards_I_Played'), # player_id could be numeric
            pl.lit(True).alias('Boards_I_Declared'), # player_id could be numeric
            pl.lit(True).alias('Boards_Partner_Declared'), # partner_id could be numeric
        )
    else:
        # Store in session state
        st.session_state.player_direction = player_direction
        st.session_state.partner_direction = partner_direction
        st.session_state.pair_direction = pair_direction
        st.session_state.opponent_pair_direction = opponent_pair_direction

        # Columns used for filtering to a specific player_id and partner_id. Needs multiple with_columns() to unnest overlapping columns.
        df = df.with_columns(
            pl.col(f'Player_ID_{player_direction}').eq(pl.lit(str(player_id))).alias('Boards_I_Played'), # player_id could be numeric
            pl.col('Declarer_ID').eq(pl.lit(str(player_id))).alias('Boards_I_Declared'), # player_id could be numeric
            pl.col('Declarer_ID').eq(pl.lit(str(partner_id))).alias('Boards_Partner_Declared'), # partner_id could be numeric
        )
    df = df.with_columns(
        pl.col('Boards_I_Played').alias('Boards_We_Played'),
        pl.col('Boards_I_Played').alias('Our_Boards'),
        (pl.col('Boards_I_Declared') | pl.col('Boards_Partner_Declared')).alias('Boards_We_Declared'),
    )
    df = df.with_columns(
        (pl.col('Boards_I_Played') & ~pl.col('Boards_We_Declared') & pl.col('Contract').ne('PASS')).alias('Boards_Opponent_Declared'),
    )

    return df


def flatten_json(nested_json: Dict) -> Dict:
    """Flatten nested JSON structure into a single level dictionary"""
    flat_dict = {}
    
    def flatten(x: Any, name: str = '') -> None:
        #print(f"flattening {name}")
        if isinstance(x, dict):
            for key, value in x.items():
                flatten(value, f"{name}_{key}" if name else key)
        elif isinstance(x, list):
            #for i, value in enumerate(x):
            #    flatten(value, f"{name}_{i}")
            flat_dict[name] = x
        else:
            flat_dict[name] = x
            
    flatten(nested_json)
    return flat_dict

def create_dataframe(data: List[Dict[str, Any]]) -> pl.DataFrame:
    """Create a Polars DataFrame from flattened JSON data"""
    try:
        # Flatten each record in the dict or list
        if isinstance(data, dict):
            #print(f"flattening dict")
            flattened_data = [flatten_json(data)]
        elif isinstance(data, list):
            #print(f"flattening list")
            flattened_data = [flatten_json(record) for record in data]
        else:
            print(f"Unsupported data type: {type(data)}")
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Create DataFrame
        df = pl.DataFrame(flattened_data)
        return df
        
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        print(f"Data structure: {type(data)}")
        print(f"First record: {json.dumps(data[0], indent=2) if data else 'Empty'}")
        raise

# obsolete?
def get_scores_data(scores_json, group_id, session_id, team_id):
    print(f"creating dataframe from scores_json")
    df = create_dataframe(scores_json)
    if df is None:
        print(f"Couldn't make dataframe from scores_json for {team_id=} {session_id=}")
        return None
    if 'board_id' not in df.columns: # todo: find out why 'board_id' doesn't exist
        print(f"No board_id for team_session_scores: {team_id} {session_id}")
        return None
    if df['lineup_segment_game_homeTeam_orientation'].ne('NS').any():
        print(f"Not a Mitchell movement. homeTeam_orientations are not all NS. Skipping: {team_id} {session_id}")
        return None
    if df['lineup_segment_game_awayTeam_orientation'].ne('EW').any():
        print(f"Not a Mitchell movement. awayTeam_orientations are not all EW. Skipping: {team_id} {session_id}")
        return None
    return df

def extract_group_id_session_id_pair_id():
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
    extracted_pair_id = int(path_parts[pair_index])
    st.session_state.group_id = extracted_group_id
    st.session_state.session_id = extracted_session_id
    st.session_state.pair_id = extracted_pair_id
    #print(f"extracted_group_id:{extracted_group_id} extracted_session_id:{extracted_session_id} extracted_pair_id:{extracted_pair_id}")
    return False


def get_ffbridge_data_using_url():

    st.session_state.game_url = st.session_state.game_url_input
    assert st.session_state.game_url is not None and st.session_state.game_url != ''

    try:

        extract_group_id_session_id_pair_id()
        # get team data
        api_team_url = f'http://localhost:8000/ffbridge.fr/competitions/results/groups/{st.session_state.group_id}/sessions/{st.session_state.session_id}/pairs/{st.session_state.pair_id}'
        #api_team_url = "https://ffbridge.fr/competitions/results/groups/7878/sessions/107118/pairs/3976783"
        #api_team_url = f"https://api-lancelot.ffbridge.fr/results/teams/{extracted_pair_id}/session/{extracted_session_id}/scores"
        #api_team_url = f'http://api-lancelot.ffbridge.fr/competitions/results/groups/{extracted_group_id}/sessions/{extracted_session_id}/pairs/{extracted_pair_id}'
        #api_team_url = "https://api-lancelot.ffbridge.fr/results/teams/3976783"
        print(f"api_team_url:{api_team_url}")
        #print(f"api_scores_url:{api_scores_url}")
        request = requests.get(api_team_url)
        request.raise_for_status()
        response_json = request.json()

        # Check if the request was successful and extract the data
        if response_json.get('success'):
            # Create DataFrame from the 'data' field
            df = pl.DataFrame(response_json['data'])
        else:
            raise ValueError(f"API request failed: {response_json}")

        # no postmortem api
        #df = pl.DataFrame(response_json)

        # Print the structure to debug
        print("DataFrame columns:", df.columns)
        #ShowDataFrameTable(df, key='team_and_session_df')

        # Update the session state
        st.session_state.group_id = st.session_state.group_id
        st.session_state.session_id = st.session_state.session_id
        st.session_state.pair_id = st.session_state.pair_id
        st.session_state.player_id = df['player1_id'][0]
        st.session_state.partner_id = df['player2_id'][0]
        st.session_state.pair_direction = df['orientation'][0]
        st.session_state.player_direction = st.session_state.pair_direction[0]
        st.session_state.partner_direction = st.session_state.pair_direction[1]
        st.session_state.opponent_pair_direction = 'EW' if st.session_state.pair_direction == 'NS' else 'NS' # opposite of pair_direction
        print(f"st.session_state.group_id:{st.session_state.group_id} st.session_state.session_id:{st.session_state.session_id} st.session_state.pair_id:{st.session_state.pair_id} st.session_state.player_id:{st.session_state.player_id} st.session_state.partner_id:{st.session_state.partner_id} st.session_state.player_direction:{st.session_state.player_direction} st.session_state.partner_direction:{st.session_state.partner_direction} st.session_state.opponent_pair_direction:{st.session_state.opponent_pair_direction}")

    except Exception as e:
        st.error(f"Error getting team or scores data: {e}")
        st.session_state.group_id = st.session_state.group_id_default
        st.session_state.session_id = st.session_state.session_id_default
        st.session_state.pair_id = st.session_state.pair_id_default
        st.session_state.player_id = st.session_state.player_id_default
        st.session_state.partner_id = st.session_state.partner_id_default
        st.session_state.player_direction = st.session_state.player_direction_default
        st.session_state.partner_direction = st.session_state.partner_direction_default
        st.session_state.opponent_pair_direction = st.session_state.opponent_pair_direction_default
        return None

    #st.session_state.df = df
    return df


def change_game_state():

    with st.spinner('Preparing Game Analysis. Takes 2 minutes total...'):
        # Use the entered URL or fallback to default.
        st.session_state.game_url = st.session_state.game_url_input.strip()
        if st.session_state.game_url is None or st.session_state.game_url.strip() == "":
            return True

        # Fetch initial data using the URL.
        df = get_ffbridge_data_using_url()

        if not st.session_state.use_historical_data: # historical data is already fully augmented so skip past augmentations
            if st.session_state.do_not_cache_df:
                with st.spinner('Creating ffbridge data to dataframe...'):
                    df = ffbridgelib.convert_ffdf_to_mldf(df) # warning: drops columns from df.
                df = augment_df(df)
            else:
                ffbridge_session_player_cache_df_filename = f'cache/df-{st.session_state.session_id}-{st.session_state.player_id}.parquet'
                ffbridge_session_player_cache_df_file = pathlib.Path(ffbridge_session_player_cache_df_filename)
                if ffbridge_session_player_cache_df_file.exists():
                    df = pl.read_parquet(ffbridge_session_player_cache_df_file)
                    print(f"Loaded {ffbridge_session_player_cache_df_filename}: shape:{df.shape} size:{ffbridge_session_player_cache_df_file.stat().st_size}")
                else:
                    with st.spinner('Creating ffbridge data to dataframe...'):
                        df = ffbridgelib.convert_ffdf_to_mldf(df) # warning: drops columns from df.
                    df = augment_df(df)
                    if df is not None:
                        st.rerun() # todo: not sure what is needed to recover from error:
                    ffbridge_session_player_cache_dir = pathlib.Path('cache')
                    ffbridge_session_player_cache_dir.mkdir(exist_ok=True)  # Creates directory if it doesn't exist
                    ffbridge_session_player_cache_df_filename = f'cache/df-{st.session_state.session_id}-{st.session_state.player_id}.parquet'
                    ffbridge_session_player_cache_df_file = pathlib.Path(ffbridge_session_player_cache_df_filename)
                    df.write_parquet(ffbridge_session_player_cache_df_file)
                    print(f"Saved {ffbridge_session_player_cache_df_filename}: shape:{df.shape} size:{ffbridge_session_player_cache_df_file.stat().st_size}")
            with st.spinner('Writing column names to file...'):
                with open('df_columns.txt','w') as f:
                    for col in sorted(df.columns):
                        f.write(col+'\n')

            # personalize to player, partner, opponents, etc.
            st.session_state.df = filter_dataframe(df, st.session_state.group_id, st.session_state.session_id, st.session_state.player_id, st.session_state.partner_id)

            # Register DataFrame as 'self' view
            st.session_state.con.register('self', st.session_state.df)
            print(f"st.session_state.df:{st.session_state.df.columns}")

    return False


def on_game_url_input_change():
    st.session_state.game_url = st.session_state.game_url_input
    change_game_state()


def create_sidebar():
    st.sidebar.caption(f"App:{st.session_state.app_datetime}") # Display application information.
    
    read_configs()

    st.session_state.game_url = st.sidebar.text_input(
        "Enter the game URL.",
        value=st.session_state.game_url_default,
        on_change=on_game_url_input_change,
        key='game_url_input'
    )

    if extract_group_id_session_id_pair_id():
        st.error("Invalid game URL. Please enter a valid game URL.")
        return

    # Provide a "Load Game URL" button.
    if st.sidebar.button("Analyze Game"):
        st.session_state.sql_query_mode = False
        change_game_state()
    # else:
    #     st.session_state.single_dummy_sample_count = st.sidebar.number_input(
    #         'Single Dummy Samples Count',
    #         value=st.session_state.single_dummy_sample_count,
    #         key='initial_sidebar_dummy_count'
    #     )
    #     st.sidebar.checkbox(
    #         'Show SQL Query',
    #         value=st.session_state.show_sql_query_default,
    #         key='initial_sidebar_show_sql_query',
    #         on_change=sql_query_on_change,
    #         help='Show SQL used to query dataframes.'
    #     )


    # When the full sidebar is to be shown:
    # --- Check if the "Analyze Game" button has been hit ---
    #if not st.session_state.analysis_started:
    st.sidebar.link_button('View Game Webpage', url=st.session_state.game_url)
    st.session_state.pdf_link = st.sidebar.empty()
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
    st.session_state.player_id = st.sidebar.number_input(
        'Player ID',
        value=st.session_state.player_id,
        key='sidebar_player_id',
        on_change=player_id_on_change,
        help='Enter ffbridge player id. e.g. 246273'
    )
    st.session_state.partner_id = st.sidebar.number_input(
        'Partner ID',
        value=st.session_state.partner_id,
        key='sidebar_partner_id',
        on_change=partner_id_on_change,
        help='Enter ffbridge partner id. e.g. 246273'
    )

    with st.sidebar.expander('Developer Settings', False):

        st.sidebar.checkbox(
            'Show SQL Query',
            value=st.session_state.show_sql_query,
            key='sidebar_show_sql_query',
            on_change=sql_query_on_change,
            help='Show SQL used to query dataframes.'
        )

        st.session_state.single_dummy_sample_count = st.sidebar.number_input(
            'Single Dummy Samples Count',
            min_value=1,
            max_value=100,
            value=st.session_state.single_dummy_sample_count,
            on_change=single_dummy_sample_count_changed,
            key='sidebar_dummy_count'
        )
    return


def initialize_website_specific():

    st.session_state.assistant_logo = 'https://github.com/BSalita/ffbridge-postmortem/blob/master/assets/logo_assistant.gif?raw=true', # 🥸 todo: put into config. must have raw=true for github url.
    st.session_state.guru_logo = 'https://github.com/BSalita/ffbridge-postmortem/blob/master/assets/logo_guru.png?raw=true', # 🥷todo: put into config file. must have raw=true for github url.
    st.session_state.game_url_default = 'https://ffbridge.fr/competitions/results/groups/7878/sessions/107118/pairs/3976783'
    st.session_state.game_name = 'ffbridge'
    st.session_state.game_url = st.session_state.game_url_default
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
        "To start our postmortem chat, I'll need the URL of your ffbridge game. It will be the subject of our chat.",
        key='intro_message_3',
        logo=st.session_state.assistant_logo
    )
    streamlit_chat.message(
        "Enter the game URL in the left sidebar or just use the default game URL. Click the Analyze Game button to begin.",
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


def perform_hand_augmentations(df, sd_productions):
    """Wrapper for backward compatibility"""
    def hand_augmentation_work(df, progress, **kwargs):
        augmenter = mlBridgeAugmentLib.HandAugmenter(
            df, 
            {}, 
            sd_productions=kwargs.get('sd_productions'),
            progress=progress
        )
        return augmenter.perform_hand_augmentations()
    
    return streamlitlib.perform_queued_work(
        df, 
        hand_augmentation_work, 
        work_description="Hand analysis",
        sd_productions=sd_productions
    )


def augment_df(df):
    with st.spinner('Creating hand data...'):
        # with safe_resource(): # perform_hand_augmentations() requires a lock because of double dummy solver dll
        #     # todo: break apart perform_hand_augmentations into dd and sd augmentations to speed up and stqdm()\
        #     progress = st.progress(0) # pass progress bar to augmenter to show progress of long running operations
        #     augmenter = mlBridgeAugmentLib.HandAugmenter(df,{},sd_productions=st.session_state.single_dummy_sample_count,progress=progress)
        #     df = augmenter.perform_hand_augmentations()
        df = perform_hand_augmentations(df, st.session_state.single_dummy_sample_count)
    with st.spinner('Augmenting with result data...'):
        augmenter = mlBridgeAugmentLib.ResultAugmenter(df,{})
        df = augmenter.perform_result_augmentations()
    with st.spinner('Augmenting with DD and SD data...'):
        augmenter = mlBridgeAugmentLib.DDSDAugmenter(df)
        df = augmenter.perform_dd_sd_augmentations()
    with st.spinner('Augmenting with matchpoints and percentages data...'):
        augmenter = mlBridgeAugmentLib.MatchPointAugmenter(df)
        df = augmenter.perform_matchpoint_augmentations()
    return df


def read_configs():

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

    if st.session_state.player_id_custom_favorites_file.exists():
        with open(st.session_state.player_id_custom_favorites_file, 'r') as f:
            player_id_favorites = json.load(f)
        st.session_state.player_id_favorites = player_id_favorites

    if st.session_state.debug_favorites_file.exists():
        with open(st.session_state.debug_favorites_file, 'r') as f:
            debug_favorites = json.load(f)
        st.session_state.debug_favorites = debug_favorites

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


def process_prompt_macros(sql_query):
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


def write_report():
    # bar_format='{l_bar}{bar}' isn't working in stqdm. no way to suppress r_bar without editing stqdm source code.
    # todo: need to pass the Button title to the stqdm description. this is a hack until implemented.
    st.session_state.main_section_container = st.container(border=True)
    with st.session_state.main_section_container:
        report_title = f"Bridge Game Postmortem Report Personalized for {st.session_state.player_name}" # can't use (st.session_state.player_id) because of href link below.
        report_creator = f"Created by https://{st.session_state.game_name}.postmortem.chat"
        report_event_info = f"{st.session_state.game_description} (event id {st.session_state.session_id})."
        report_game_results_webpage = f"Results Page: {st.session_state.game_url}"
        report_your_match_info = f"Your pair was {st.session_state.pair_id}{st.session_state.pair_direction} in section {st.session_state.section_name}. You played {st.session_state.player_direction}. Your partner was {st.session_state.partner_name} ({st.session_state.partner_id}) who played {st.session_state.partner_direction}."
        st.markdown(f"### {report_title}")
        st.markdown(f"##### {report_creator}")
        st.markdown(f"#### {report_event_info}")
        st.markdown(f"##### {report_game_results_webpage}")
        st.markdown(f"#### {report_your_match_info}")
        pdf_assets = st.session_state.pdf_assets
        pdf_assets.clear()
        pdf_assets.append(f"# {report_title}")
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

        # As a text link
        #st.markdown('[Back to Top](#your-personalized-report)')

        # As an html button (needs styling added)
        # can't use link_button() restarts page rendering. markdown() will correctly jump to href.
        # st.link_button('Go to top of report',url='#your-personalized-report')\
        report_title_anchor = report_title.replace(' ','-').lower()
        st.markdown(f'<a target="_self" href="#{report_title_anchor}"><button>Go to top of report</button></a>', unsafe_allow_html=True)

    if st.session_state.pdf_link.download_button(label="Download Personalized Report",
            data=streamlitlib.create_pdf(st.session_state.pdf_assets, title=f"Bridge Game Postmortem Report Personalized for {st.session_state.player_id}"),
            file_name = f"{st.session_state.session_id}-{st.session_state.player_id}-morty.pdf",
            disabled = len(st.session_state.pdf_assets) == 0,
            mime='application/octet-stream',
            key='personalized_report_download_button'):
        st.warning('Personalized report downloaded.')
    return


def ask_sql_query():

    if st.session_state.show_sql_query:
        with st.container():
            with bottom():
                st.chat_input('Enter a SQL query e.g. SELECT PBN, Contract, Result, N, S, E, W', key='main_prompt_chat_input', on_submit=chat_input_on_submit)


def create_ui():
    create_sidebar()
    if not st.session_state.sql_query_mode:
        #create_tab_bar()
        if st.session_state.session_id is not None:
            write_report()
    ask_sql_query()


def initialize_session_state():
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


def initialize_session_state():
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

    initialize_website_specific()
    first_time_defaults = {
        'first_time': True,
        'single_dummy_sample_count': 10,
        'show_sql_query': True, # os.getenv('STREAMLIT_ENV') == 'development',
        'use_historical_data': False,
        'do_not_cache_df': True, # todo: set to True for production
        'con': duckdb.connect(),
        'con_register_name': 'self',
        'main_section_container': st.empty(),
        'app_datetime': datetime.fromtimestamp(pathlib.Path(__file__).stat().st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z'),
        'current_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for key, value in first_time_defaults.items():
        st.session_state[key] = value

    reset_game_data()
    return


def reset_game_data():

    # Default values for session state variables
    reset_defaults = {
        'game_description_default': None,
        'group_id_default': None,
        'session_id_default': None,
        'section_name_default': None,
        'player_id_default': None,
        'partner_id_default': None,
        'player_name_default': None,
        'partner_name_default': None,
        'player_direction_default': None,
        'partner_direction_default': None,
        'pair_id_default': None,
        'pair_direction_default': None,
        'opponent_pair_direction_default': None,
    }
    
    # Initialize default values if not already set
    for key, value in reset_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize additional session state variables that depend on defaults.
    reset_session_vars = {
        'df': None,
        'game_description': st.session_state.game_description_default,
        'group_id': st.session_state.group_id_default,
        'session_id': st.session_state.session_id_default,
        'section_name': st.session_state.section_name_default,
        'player_id': st.session_state.player_id_default,
        'partner_id': st.session_state.partner_id_default,
        'player_name': st.session_state.player_name_default,
        'partner_name': st.session_state.partner_name_default,
        'player_direction': st.session_state.player_direction_default,
        'partner_direction': st.session_state.partner_direction_default,
        'pair_id': st.session_state.pair_id_default,
        'pair_direction': st.session_state.pair_direction_default,
        'opponent_pair_direction': st.session_state.opponent_pair_direction_default,
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
        if key not in st.session_state:
            st.session_state[key] = value

    return


def app_info():
    st.caption(f"Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in streamlit. Data engine is polars. Query engine is duckdb. Bridge lib is endplay. Self hosted using Cloudflare Tunnel. Repo:https://github.com/BSalita/ffbridge-postmortem")
    st.caption(
        f"App:{st.session_state.app_datetime} Python:{'.'.join(map(str, sys.version_info[:3]))} Streamlit:{st.__version__} Pandas:{pd.__version__} polars:{pl.__version__} endplay:{endplay.__version__} Query Params:{st.query_params.to_dict()}")
    return


def main():
    if 'first_time' not in st.session_state:
        initialize_session_state()
        create_sidebar()
    else:
        create_ui()
    return


if __name__ == "__main__":
    main()

