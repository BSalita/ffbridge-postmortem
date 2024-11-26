 
# streamlit program to display Bridge game deal statistics.
# Invoke from system prompt using: streamlit run ffbridge_streamlit.py

# todo showstoppers:
# fix EV stuff. lot's wrong. NV vs V, column naming, are they even correctly calculated and displayed?
# game date? do we have to get a new URL?

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
from datetime import datetime, timezone
import sys

from urllib.parse import urlparse
from typing import Dict, Any, List

import endplay # for __version__

# assumes symlinks are created in current directory.
sys.path.append(str(pathlib.Path.cwd().joinpath('streamlitlib')))  # global
sys.path.append(str(pathlib.Path.cwd().joinpath('mlBridgeLib')))  # global
sys.path.append(str(pathlib.Path.cwd().joinpath('ffbridgelib')))  # global

import ffbridgelib
import streamlitlib
#import mlBridgeLib
import mlBridgeAugmentLib
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


def app_info():
    st.caption(f"Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in streamlit. Data engine is polars. Query engine is duckdb. Bridge lib is endplay. Self hosted using Cloudflare Tunnel. Repo:https://github.com/BSalita/ffbridge-postmortem")
    st.caption(
        f"App:{st.session_state.app_datetime} Python:{'.'.join(map(str, sys.version_info[:3]))} Streamlit:{st.__version__} Pandas:{pd.__version__} polars:{pl.__version__} endplay:{endplay.__version__} Query Params:{st.query_params.to_dict()}")


def chat_input_on_submit():
    prompt = st.session_state.main_prompt_chat_input
    sql_query = process_prompt_macros(prompt)
    ShowDataFrameTable(st.session_state.df, query=sql_query, key='user_query_main_doit')


def sample_count_on_change():
    st.session_state.single_dummy_sample_count = st.session_state.create_sidebar_single_dummy_sample_count_on_change
    # if 'df' in st.session_state: # todo: is this still needed?
    #     st.session_state.df = load_historical_data()


def sql_query_on_change():
    st.session_state.show_sql_query = st.session_state.create_sidebar_show_sql_query_checkbox
    # if 'df' in st.session_state: # todo: is this still needed?
    #     st.session_state.df = load_historical_data()


def group_id_on_change():
    st.session_state.group_id = st.session_state.create_sidebar_group_id
    # if 'df' in st.session_state: # todo: is this still needed?
    #     st.session_state.df = load_historical_data()


def session_id_on_change():
    st.session_state.session_id = st.session_state.create_sidebar_session_id
    # if 'df' in st.session_state: # todo: is this still needed?
    #     st.session_state.df = load_historical_data()


def player_id_on_change():
    st.session_state.group_id = st.session_state.create_sidebar_player_id
    # if 'df' in st.session_state: # todo: is this still needed?
    #     st.session_state.df = load_historical_data()


def partner_id_on_change():
    st.session_state.partner_id = st.session_state.create_sidebar_partner_id
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


def get_team_and_scores_from_url():

    # https://ffbridge.fr/competitions/results/groups/7878/sessions/107118/ranking
    # https://ffbridge.fr/competitions/results/groups/7878/sessions/107118/pairs/3976783
    # https://api-lancelot.ffbridge.fr/results/teams/3976783/session/107118/scores

    try:

        #default_game_url = 'https://ffbridge.fr/competitions/results/groups/7878/sessions/107118/pairs/3976783'
        default_game_url = 'https://ffbridge.fr/competitions/results/groups/7878/sessions/140342/pairs/5989257'
        st.session_state.game_url = st.sidebar.text_input('Enter ffbridge results url',value=default_game_url,key='ffbridge_url')
        #print(f"st.session_state.game_url:{st.session_state.game_url}")
        st.sidebar.link_button('Open results page',url=st.session_state.game_url)

        parsed_url = urlparse(st.session_state.game_url)
        #print(f"parsed_url:{parsed_url}")
        path_parts = parsed_url.path.split('/')
        #print(f"path_parts:{path_parts}")

        # Find indices by keywords instead of fixed positions
        group_index = path_parts.index('groups') + 1
        session_index = path_parts.index('sessions') + 1
        pair_index = path_parts.index('pairs') + 1
        #print(f"group_index:{group_index} session_index:{session_index} pair_index:{pair_index}")
        
        extracted_group_id = int(path_parts[group_index])
        extracted_session_id = int(path_parts[session_index])
        extracted_pair_id = int(path_parts[pair_index])
        #print(f"extracted_group_id:{extracted_group_id} extracted_session_id:{extracted_session_id} extracted_pair_id:{extracted_pair_id}")
        
        # get team data
        api_team_url = f'https://api-lancelot.ffbridge.fr/results/teams/{extracted_pair_id}'
        #print(f"api_team_url:{api_team_url}")
        request = requests.get(api_team_url)
        #print(f"request:{request}")
        request.raise_for_status()
        team_json = request.json()
        #print(f"got team_json")
        team_df = create_dataframe(team_json)
        #print(f"team_df")
        #ShowDataFrameTable(team_df, key='team_df')
        #print(f"showedteam_df")
        player1_id = team_json['player1']['id']
        player2_id = team_json['player2']['id']
        pair_direction = team_json['orientation']
        opponent_pair_direction = 'NS' if pair_direction == 'EW' else 'EW'
        #print(f"player1_id:{player1_id} player2_id:{player2_id} pair_direction:{pair_direction} opponent_pair_direction:{opponent_pair_direction}")

        api_scores_url = f'https://api-lancelot.ffbridge.fr/results/teams/{extracted_pair_id}/session/{extracted_session_id}/scores'
        #print(f"api_scores_url:{api_scores_url}")
        request = requests.get(api_scores_url)
        #print(f"request:{request}")
        request.raise_for_status()
        scores_json = request.json() # todo: extract data and use instead of historical data.
 
        df = get_scores_data(scores_json, extracted_group_id, extracted_session_id, extracted_pair_id)

        # initialize columns which are needed for SQL queries.
        # todo: should these be initialized in ffbridgelib.convert_ffdf_to_mldf()?
        df = df.with_columns(pl.lit(extracted_group_id).cast(pl.UInt32).alias('group_id'))
        df = df.with_columns(pl.lit(extracted_session_id).cast(pl.UInt32).alias('session_id'))
        df = df.with_columns(pl.lit(extracted_pair_id).cast(pl.UInt32).alias('team_id'))
        # from 'team', add useful columns. e.g. section, startTableNumber, orientation columns
        df = df.with_columns(pl.lit(team_df['section'].first()).cast(pl.String).alias('section'))
        df = df.with_columns(pl.lit(team_df['startTableNumber'].first()).cast(pl.UInt16).alias('startTableNumber'))
        df = df.with_columns(pl.lit(team_df['orientation'].first()).cast(pl.String).alias('orientation'))
        df = df.with_columns(pl.struct([pl.col('team_id'),pl.col('session_id')]).alias('team_session_id'))
        df = df.select([pl.col('group_id','team_session_id','session_id','team_id'),pl.all().exclude('group_id','team_session_id','session_id','team_id')])
        #ShowDataFrameTable(df, key='team_and_session_df')

        # Update the session state
        st.session_state.group_id = extracted_group_id
        st.session_state.session_id = extracted_session_id
        st.session_state.pair_id = extracted_pair_id
        st.session_state.player_id = player1_id
        st.session_state.partner_id = player2_id
        st.session_state.pair_direction = pair_direction
        st.session_state.player_direction = pair_direction[0]
        st.session_state.partner_direction = pair_direction[1]
        st.session_state.opponent_pair_direction = opponent_pair_direction
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

    return df

def create_sidebar():

    st.sidebar.caption('Build:'+st.session_state.app_datetime)

    st.session_state.group_id = st.sidebar.number_input('Group ID',value=st.session_state.group_id,key='create_sidebar_group_id_on_change',on_change=group_id_on_change,help='Enter ffbridge group id. e.g. 7878 for Bridge Club St. Honore')
    st.session_state.session_id = st.sidebar.number_input('Session ID',value=st.session_state.session_id,key='create_sidebar_session_id_on_change',on_change=session_id_on_change,help='Enter ffbridge session id. e.g. 107118')
    st.session_state.player_id = st.sidebar.number_input('Player ID',value=st.session_state.player_id,key='create_sidebar_player_id_on_change',on_change=player_id_on_change,help='Enter ffbridge player id. e.g. 246273 for Robert Salita')
    st.session_state.partner_id = st.sidebar.number_input('Partner ID',value=st.session_state.partner_id,key='create_sidebar_partner_id_on_change',on_change=partner_id_on_change,help='Enter ffbridge partner id. e.g. 246273 for Robert Salita')
    st.session_state.single_dummy_sample_count = st.sidebar.number_input('Single Dummy Samples Count',value=st.session_state.single_dummy_sample_count,key='create_sidebar_single_dummy_sample_count_on_change',on_change=sample_count_on_change,help='Enter number of single dummy samples to generate.')

    # SELECT Board, Vul, ParContract, ParScore_NS, Custom_ParContract
    st.sidebar.checkbox('Show SQL Query',value=st.session_state.show_sql_query_default,key='create_sidebar_show_sql_query_checkbox',on_change=sql_query_on_change,help='Show SQL used to query dataframes.')


def read_favorites():

    if st.session_state.default_favorites_file.exists():
        with open(st.session_state.default_favorites_file, 'r') as f:
            favorites = json.load(f)
            st.session_state.favorites = favorites

    if st.session_state.player_id_custom_favorites_file.exists():
        with open(st.session_state.player_id_custom_favorites_file, 'r') as f:
            player_id_favorites = json.load(f)
            st.session_state.player_id_favorites = player_id_favorites

    if st.session_state.debug_favorites_file.exists():
        with open(st.session_state.debug_favorites_file, 'r') as f:
            debug_favorites = json.load(f)
            st.session_state.debug_favorites = debug_favorites


def load_vetted_prompts(json_file):
    sql_queries = []
    if json_file.exists():
        with open(json_file) as f:
            json_data = json.load(f)
        
        # Navigate the JSON path to get the appropriate list of prompts
        vetted_prompts = [json_data['SelectBoxes']['Vetted_Prompts'][p[1:]] for p in json_data["Buttons"]['Summarize']['prompts']]
    
    return vetted_prompts


def process_prompt_macros(sql_query):
    replacements = {
        '{Player_Direction}': st.session_state.player_direction,
        '{Partner_Direction}': st.session_state.partner_direction,
        '{Pair_Direction}': st.session_state.pair_direction,
        '{Opponent_Pair_Direction}': st.session_state.opponent_pair_direction
    }
    for old, new in replacements.items():
        sql_query = sql_query.replace(old, new)
    return sql_query


def show_dfs(vetted_prompts, pdf_assets):
    sql_query_count = 0

    # bar_format='{l_bar}{bar}' isn't working in stqdm. no way to suppress r_bar without editing stqdm source code.
    for category in stqdm(list(vetted_prompts), desc='Morty is analyzing your game...', bar_format='{l_bar}{bar}'): #[:-3]:
        #print('category:',category)
        if "prompts" in category:
            for i,prompt in enumerate(category["prompts"]):
                #print('prompt:',prompt) 
                if "sql" in prompt and prompt["sql"]:
                    if i == 0:
                        streamlit_chat.message(f"Morty: {category['help']}", key=f'morty_sql_query_{sql_query_count}', logo=st.session_state.assistant_logo)
                        pdf_assets.append(f"## {category['help']}")
                    #print('sql:',prompt["sql"])
                    prompt_sql = prompt['sql']
                    sql_query = process_prompt_macros(prompt_sql)
                    query_df = ShowDataFrameTable(st.session_state.df, query=sql_query, key=f'sql_query_{sql_query_count}')
                    if query_df is not None:
                        pdf_assets.append(query_df)
                    sql_query_count += 1
                    #break
        #break


if __name__ == '__main__':

    # first time only defaults
    if 'first_time_only_initialized' not in st.session_state:

        st.session_state.first_time_only_initialized = True
        st.set_page_config(layout="wide")
        # Add this auto-scroll code
        streamlitlib.widen_scrollbars()

        st.session_state.assistant_logo = 'https://github.com/BSalita/Bridge_Game_Postmortem_Chatbot/blob/main/assets/logo_assistant.gif?raw=true' # 🥸 todo: put into config. must have raw=true for github url.
        st.session_state.guru_logo = 'https://github.com/BSalita/Bridge_Game_Postmortem_Chatbot/blob/main/assets/logo_guru.png?raw=true' # 🥷todo: put into config file. must have raw=true for github url.

        # todo: put filenames into a .json or .toml file?
        st.session_state.rootPath = pathlib.Path('e:/bridge/data')
        st.session_state.ffbridgePath = st.session_state.rootPath.joinpath('ffbridge')
        #st.session_state.favoritesPath = pathlib.joinpath('favorites')
        st.session_state.dataPath = st.session_state.ffbridgePath.joinpath('data')
    
        st.session_state.app_datetime = datetime.fromtimestamp(pathlib.Path(__file__).stat().st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')

        app_info()
        streamlit_chat.message(f"Morty: Hi. I'm Morty, your bridge game postmortem expert.", key=f'morty_hi', logo=st.session_state.assistant_logo)
        streamlit_chat.message(f"Morty: I'm preparing a postmortem report on your game. I only understand pair games using a Mitchell movement.", key=f'morty_restrictions', logo=st.session_state.assistant_logo)
        streamlit_chat.message(f"Morty: Takes me a full minute to download and analyze your game. Please wait until report is fully rendered.", key=f'morty_in_a_minute', logo=st.session_state.assistant_logo)

        single_dummy_sample_count_default = 40 #2  # number of random deals to generate for calculating single dummy probabilities. Use smaller number for testing.
        st.session_state.single_dummy_sample_count = single_dummy_sample_count_default
        st.session_state.show_sql_query_default = True
        st.session_state.show_sql_query = st.session_state.show_sql_query_default

        st.session_state.group_id_default = 7878 # Bridge Club St. Honore
        st.session_state.session_id_default = 107118 # Robert Salita's best game at Bridge Club St. Honore
        st.session_state.pair_id_default = None
        st.session_state.player_id_default = 246273 # Robert Salita
        st.session_state.partner_id_default = None
        st.session_state.player_direction_default = 'E'
        st.session_state.partner_direction_default = 'W'
        st.session_state.pair_direction_default = 'EW'
        st.session_state.opponent_pair_direction_default = 'NS'
        st.session_state.game_url_default = None
        st.session_state.game_date_default = 'Unknown date' #pd.to_datetime(st.session_state.df['Date'].iloc[0]).strftime('%Y-%m-%d')

        st.session_state.group_id = st.session_state.group_id_default # Bridge Club St. Honore
        st.session_state.session_id = st.session_state.session_id_default # Robert Salita's best game at Bridge Club St. Honore
        st.session_state.pair_id = st.session_state.pair_id_default
        st.session_state.player_id = st.session_state.player_id_default # Robert Salita
        st.session_state.partner_id = st.session_state.partner_id_default
        st.session_state.player_direction = st.session_state.player_direction_default
        st.session_state.partner_direction = st.session_state.partner_direction_default
        st.session_state.pair_direction = st.session_state.pair_direction_default
        st.session_state.opponent_pair_direction = st.session_state.opponent_pair_direction_default
        st.session_state.game_url = st.session_state.game_url_default
        st.session_state.game_date = st.session_state.game_date_default
        st.session_state.use_historical_data = False # use historical data from file or get from url

        # Create connection
        st.session_state.con = duckdb.connect()

    if False: #'df' in st.session_state:
        create_sidebar()
    else:
        with st.spinner("Loading Game Data..."):

            # if st.session_state.use_historical_data:
            #     df = load_historical_data()
            # else:
                df = get_team_and_scores_from_url()
                create_sidebar() # update sidebar using url's parsed parameters

        with st.spinner('Looking deeply into game data...'):

            if not st.session_state.use_historical_data: # historical data is already fully augmented so skip past augmentations
                df = ffbridgelib.convert_ffdf_to_mldf(df)
                df = mlBridgeAugmentLib.perform_hand_augmentations(df,{},sd_productions=st.session_state.single_dummy_sample_count)
                df = mlBridgeAugmentLib.PerformMatchPointAndPercentAugmentations(df)
                df = mlBridgeAugmentLib.PerformResultAugmentations(df,{})
                df = mlBridgeAugmentLib.Perform_DD_SD_Augmentations(df)
                with open('df_columns.txt','w') as f:
                    for col in sorted(df.columns):
                        f.write(col+'\n')

            # personalize to player, partner, opponents, etc.
            st.session_state.df = filter_dataframe(df, st.session_state.group_id, st.session_state.session_id, st.session_state.player_id, st.session_state.partner_id)

            # Register DataFrame as 'self' view
            st.session_state.con.register('self', st.session_state.df)

            # ShowDataFrameTable(df, key='everything_df', query='SELECT Board, Pct_NS, Pct_EW, MP_NS, MP_EW FROM self')

            st.session_state.default_favorites_file = pathlib.Path(
                'default.favorites.json')
            st.session_state.player_id_custom_favorites_file = pathlib.Path(
                f'favorites/{st.session_state.player_id}.favorites.json')
            st.session_state.debug_favorites_file = pathlib.Path(
                'favorites/debug.favorites.json')
            read_favorites()
            st.session_state.vetted_prompts = load_vetted_prompts(st.session_state.default_favorites_file)

            pdf_assets = []
            pdf_assets.append(f"# Bridge Game Postmortem Report Personalized for {st.session_state.player_id}")
            pdf_assets.append(f"### Created by https://ffbridge.postmortem.chat")
            pdf_assets.append(f"## Game Date:? Session:{st.session_state.session_id} Player:{st.session_state.player_id} Partner:{st.session_state.partner_id}")

            with st.container(border=True):
                st.markdown('### Your Personalized Report')
                st.text(f'Game Date:? Session:{st.session_state.session_id} Player:{st.session_state.player_id} Partner:{st.session_state.partner_id}')
                show_dfs(st.session_state.vetted_prompts, pdf_assets)

                # As a text link
                #st.markdown('[Back to Top](#your-personalized-report)')

                # As an html button (needs styling added)
                # can't use link_button() restarts page rendering. markdown() will correctly jump to href.
                # st.link_button('Go to top of report',url='#your-personalized-report')
                st.markdown(''' <a target="_self" href="#your-personalized-report">
                                    <button>
                                        Go to top of report
                                    </button>
                                </a>''', unsafe_allow_html=True)

            if st.sidebar.download_button(label="Download Personalized Report",
                    data=streamlitlib.create_pdf(pdf_assets, title=f"Bridge Game Postmortem Report Personalized for {st.session_state.player_id}"),
                    file_name = f"{st.session_state.session_id}-{st.session_state.player_id}-morty.pdf",
                    mime='application/octet-stream'):
                st.warning('Personalized report downloaded.')

    if st.session_state.show_sql_query:
        with st.container():
            with bottom():
                st.chat_input('Enter a SQL query e.g. SELECT PBN, Contract, Result, N, S, E, W', key='main_prompt_chat_input', on_submit=chat_input_on_submit)

