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
import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv

from urllib.parse import urlparse
from typing import Dict, Any, List, Mapping, Optional, Union, Tuple, Callable
from typing_extensions import TypedDict

import endplay # for __version__

# ----------------------------
# Debug helpers (persist across reruns)
# ----------------------------

def _debug_init_state() -> None:
    if '_debug_items' not in st.session_state:
        st.session_state._debug_items = []
    if '_debug_max_items' not in st.session_state:
        st.session_state._debug_max_items = 50


def debug_capture_df(label: str, df: Any, source: Optional[str] = None) -> None:
    """Capture a dataframe for later viewing in the Debug expander."""
    if not st.session_state.get('debug_mode', False):
        return
    _debug_init_state()
    try:
        shape = getattr(df, "shape", None)
    except Exception:
        shape = None
    st.session_state._debug_items.append({
        "type": "df",
        "label": label,
        "source": source,
        "shape": shape,
        "df": df,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    # Trim
    if len(st.session_state._debug_items) > st.session_state._debug_max_items:
        st.session_state._debug_items = st.session_state._debug_items[-st.session_state._debug_max_items:]


def render_debug_expander() -> None:
    """Render a persistent debug UI section in the main page (survives reruns)."""
    if not st.session_state.get('debug_mode', False):
        return

    _debug_init_state()
    
    # Only show the debug expander if there's something to display
    has_player_row = 'player_row' in st.session_state and st.session_state.player_row is not None
    has_partner_row = 'partner_row' in st.session_state and st.session_state.partner_row is not None
    has_df = 'df' in st.session_state and st.session_state.df is not None
    has_debug_items = bool(st.session_state._debug_items)
    
    if not (has_player_row or has_partner_row or has_df or has_debug_items):
        return
    
    with st.expander("Debug", expanded=True):
        # Quick snapshots from session_state (if present)
        if has_player_row:
            st.caption("player_row (session_state)")
            st.dataframe(st.session_state.player_row, selection_mode='single-row')
        if has_partner_row:
            st.caption("partner_row (session_state)")
            st.dataframe(st.session_state.partner_row, selection_mode='single-row')
        if has_df:
            st.caption(f"df (session_state) shape: {getattr(st.session_state.df, 'shape', None)}")

        # Captured debug items (all API accesses are shown as dfs) in chronological order
        if has_debug_items:
            if has_player_row or has_partner_row or has_df:
                st.markdown("---")
            for item in st.session_state._debug_items[-20:]:
                if item.get("type") == "df":
                    src = f" | {item['source']}" if item.get("source") else ""
                    st.caption(f"{item.get('ts','')} | {item.get('label','')} shape: {item.get('shape')}{src}")
                    st.dataframe(item.get("df"), selection_mode='single-row')

# Only declared to display version information
#import fastai
import numpy as np
import pandas as pd
#import safetensors
#import sklearn
#import torch

_APP_DIR = pathlib.Path(__file__).resolve().parent
# Docker: /app/<app>; monorepo: src/<app> or src/ffbridge/<app>
_REQUIRED_LIBS = ('mlBridge', 'streamlitlib')

def _is_lib_dir(p: pathlib.Path) -> bool:
    # Reject empty leftover dirs after git removes vendored junction copies
    # (__pycache__ alone still makes Path.is_dir() true).
    return p.is_dir() and (p / '__init__.py').is_file()

_resolved_libs = []
for _name in _REQUIRED_LIBS:
    _candidates = (
        _APP_DIR / _name,
        _APP_DIR.parent / _name,
        _APP_DIR.parent.parent / _name,
    )
    _found = next((p for p in _candidates if _is_lib_dir(p)), None)
    if _found is None:
        raise FileNotFoundError(
            f"{_name} not found at " + " or ".join(str(p) for p in _candidates)
        )
    _resolved_libs.append(_found)
_path_roots = {_APP_DIR, *(_p.parent for _p in _resolved_libs)}
for _p in _path_roots:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.append(_s)
for _p in _resolved_libs:
    _s = str(_p)
    if _p.name == 'mlBridge':
        if _s not in sys.path:
            sys.path.append(_s)  # logging_config and friends
    else:
        if _s in sys.path:
            sys.path.remove(_s)
        sys.path.insert(0, _s)

import mlBridge.mlBridgeLib as mlBridgeLib
import mlBridge.mlBridgeFFLib as mlBridgeFFLib
from mlBridge import mlBridgeBPLib
import streamlitlib
import time
#import mlBridgeLib
from mlBridge.mlBridgeAugmentLib import (
    AllAugmentations,
)
from mlBridge.mlBridgePostmortemLib import PostmortemBase
#import mlBridgeEndplayLib
import ffbridge_postmortem_create as pm_create
import ffbridge_postmortem_api_client as pm_api

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

# ----------------------------
# API source (Classic vs Lancelot) infrastructure
# ----------------------------

# API constants, URL builders, and the low-level Lancelot client live in the shared
# mlBridgeFFLib (used by the Elo_Ratings project too).
classic_api_url = mlBridgeFFLib.classic_api_url
lancelot_api_url = mlBridgeFFLib.lancelot_api_url

API_SOURCE_CLASSIC = 'classic'
API_SOURCE_LANCELOT = 'lancelot'
API_SOURCE_LABELS = {
    API_SOURCE_CLASSIC: 'Classic (api.ffbridge.fr)',
    API_SOURCE_LANCELOT: 'Lancelot (api-lancelot.ffbridge.fr)',
}


@st.cache_data(ttl=300, show_spinner=False)
def probe_api_sources() -> Dict[str, Dict[str, Any]]:
    """Probe both API backends and report their health.

    Cached for 5 minutes so the sidebar doesn't re-probe on every rerun.
    Keys are API_SOURCE_CLASSIC / API_SOURCE_LANCELOT.
    """
    return mlBridgeFFLib.probe_ffbridge_health()


def auto_detect_api_source() -> str:
    """Use Lancelot unless the user explicitly selects Classic.

    A transient public-version timeout must not switch identifier namespaces.
    In particular, the Classic probe can return an auth-related 403 while the
    server is reachable but unusable for member lookup.
    """
    return API_SOURCE_LANCELOT


def get_api_source() -> str:
    return st.session_state.get('api_source', API_SOURCE_LANCELOT)


def is_lancelot_mode() -> bool:
    return get_api_source() == API_SOURCE_LANCELOT


def get_lancelot_token() -> str:
    """Return the session's Lancelot bearer token, failing fast if there is none."""
    token = st.session_state.get('ffbridge_bearer_token')
    if not token:
        raise ValueError("No Lancelot bearer token available. Set FFBRIDGE_EMAIL/FFBRIDGE_PASSWORD or FFBRIDGE_BEARER_TOKEN_LANCELOT in .env.")
    return token


def _sync_lancelot_session_token(token: str) -> None:
    st.session_state.ffbridge_bearer_token = token
    st.session_state.lancelot_token_valid = True


def _validated_lancelot_token(*, force: bool = False) -> str:
    """Return a library-validated Lancelot token and keep session_state in sync.

    Streamlit can keep a stale FFBRIDGE_BEARER_TOKEN_LANCELOT in session_state
    after startup. ensure_lancelot_auth() re-checks persons/me and refreshes
    via Firebase when the env token is rejected.
    """
    auth = pm_create.ensure_lancelot_auth(force=force)
    _sync_lancelot_session_token(auth.token)
    return auth.token


def _is_lancelot_unauthorized(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    return response is not None and response.status_code == 401


def make_lancelot_request(path: str, use_auth: bool = True) -> Any:
    """GET a Lancelot API path and return parsed JSON.

    Args:
        path: path portion (no leading slash), e.g. 'persons/search?name=x'
        use_auth: include the Lancelot bearer token (required for user/person endpoints)
    """
    token = get_lancelot_token() if use_auth else None
    print(f"Making Lancelot API request to: {lancelot_api_url(path)}")
    try:
        return mlBridgeFFLib.lancelot_get(path, token=token)
    except requests.exceptions.HTTPError as e:
        if use_auth and _is_lancelot_unauthorized(e):
            token = _validated_lancelot_token(force=True)
            return mlBridgeFFLib.lancelot_get(path, token=token)
        raise


_LANCELOT_SEARCH_SCHEMA = {
    'person_id': pl.Utf8,
    'person_firstname': pl.Utf8,
    'person_lastname': pl.Utf8,
    'person_license_number': pl.Utf8,
    'person_migration_id': pl.Int64,
}


def _lancelot_search_df(rows: List[Dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=_LANCELOT_SEARCH_SCHEMA)


def _lancelot_search_row(
    *,
    person_id: str,
    license_number: str = '',
    firstname: str = '',
    lastname: str = '',
    migration_id: Any = None,
) -> Dict[str, Any]:
    mid = None
    if migration_id is not None and str(migration_id).strip().lstrip('-').isdigit():
        mid = int(migration_id)
    return {
        'person_id': str(person_id),
        'person_firstname': firstname or '',
        'person_lastname': lastname or '',
        'person_license_number': str(license_number or ''),
        'person_migration_id': mid,
    }


def _license_lookup_from_logged_in_user(query: str) -> Optional[pl.DataFrame]:
    """Resolve the signed-in player's own license/Lancelot/Classic id without Lancelot."""
    wanted = pm_create._norm_digits(query)
    lancelot_id = st.session_state.get('logged_in_lancelot_id')
    if not lancelot_id:
        return None
    aliases = {
        pm_create._norm_digits(value)
        for value in (
            st.session_state.get('logged_in_license_number'),
            lancelot_id,
            st.session_state.get('logged_in_player_id'),
        )
        if value
    }
    if wanted not in aliases:
        return None
    return _lancelot_search_df([
        _lancelot_search_row(
            person_id=str(lancelot_id),
            license_number=str(st.session_state.get('logged_in_license_number') or ''),
            migration_id=st.session_state.get('logged_in_player_id'),
        )
    ])


def _search_df_if_license_match(
    query: str,
    *,
    person_id: Optional[str],
    license_number: Optional[str],
    migration_id: Any = None,
) -> Optional[pl.DataFrame]:
    if not person_id or not license_number:
        return None
    if pm_create._norm_digits(license_number) != pm_create._norm_digits(query):
        return None
    return _lancelot_search_df([
        _lancelot_search_row(
            person_id=str(person_id),
            license_number=str(license_number),
            migration_id=migration_id,
        )
    ])


def _license_lookup_from_index_or_api(query: str) -> Optional[pl.DataFrame]:
    """Resolve a known license from the player-session index, then the writer API."""
    try:
        indexed = pm_create._resolve_player_from_index(query)
    except FileNotFoundError:
        indexed = None
    if indexed is not None:
        found = _search_df_if_license_match(
            query,
            person_id=indexed.lancelot_id,
            license_number=indexed.license_number,
            migration_id=indexed.classic_person_id,
        )
        if found is not None:
            return found
    try:
        resolved = pm_api.resolve_player(query)
    except pm_api.FfbridgeApiClientError as e:
        print(f"search_members: API resolve failed for {query!r}: {e}")
        return None
    return _search_df_if_license_match(
        query,
        person_id=resolved.get('player_id') or resolved.get('lancelot_id'),
        license_number=resolved.get('player_license_number') or resolved.get('license_number'),
        migration_id=resolved.get('classic_person_id'),
    )


def _search_persons_lancelot(query: str) -> List[Dict[str, Any]]:
    try:
        return mlBridgeFFLib.search_persons(query, _validated_lancelot_token())
    except requests.exceptions.HTTPError as e:
        if _is_lancelot_unauthorized(e):
            return mlBridgeFFLib.search_persons(query, _validated_lancelot_token(force=True))
        raise


def search_members(query: str) -> pl.DataFrame:
    """Source-aware member search.

    Returns a normalized DataFrame with columns:
    person_id, person_firstname, person_lastname, person_license_number

    In Classic mode person_id is the Classic person_id; in Lancelot mode it is the Lancelot person id.
    Numeric Lancelot lookups use the signed-in identity or the player-session
    index first so an expired bearer token does not block a known license.
    """
    q = (query or '').strip()
    if is_lancelot_mode():
        if q.isdigit():
            local = _license_lookup_from_logged_in_user(q)
            if local is not None:
                return local
            indexed = _license_lookup_from_index_or_api(q)
            if indexed is not None:
                return indexed
        items = _search_persons_lancelot(q)
        rows = [
            _lancelot_search_row(
                person_id=str(item['id']),
                license_number=str(item.get('ffbId') or ''),
                firstname=item.get('firstName') or '',
                lastname=item.get('lastName') or '',
                migration_id=item.get('migrationId'),
            )
            for item in items
        ]
        return _lancelot_search_df(rows)
    else:
        api_urls_d = {
            'search': (classic_api_url(f"search-members?alive=1&search={q}"), False),
        }
        dfs, _ = get_ffbridge_data_using_url_licencie(api_urls_d, show_progress=False)
        return dfs['search']


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
            st.text(f"Result is a dataframe of {len(result_df)} rows.")
            streamlitlib.ShowDataFrameTable(result_df, key, height_rows=height_rows)
        except Exception as e:
            st.error(f"duckdb exception: error:{e} query:{query}")
            return None
        
        return result_df


# ----------------------------
# Sidebar <-> URL query-param sync
# ----------------------------
#
# Each entry maps a URL query-param name to the underlying st.session_state key
# that drives a sidebar widget, plus parser/serializer helpers.
#
# - parser:   str (from URL) -> typed value stored in session_state. Raise on bad input.
# - serializer: typed value -> str (for URL). Return None to omit the param from the URL.
#
# Add new sidebar options here to make them URL-syncable.

def _parse_bool_param(raw: str) -> bool:
    s = str(raw).strip().lower()
    if s in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if s in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    raise ValueError(f"invalid boolean URL param value: {raw!r}")


def _parse_api_source_param(raw: str) -> str:
    s = str(raw).strip().lower()
    if s not in (API_SOURCE_CLASSIC, API_SOURCE_LANCELOT):
        raise ValueError(f"invalid api_source URL param value: {raw!r}")
    return s


SIDEBAR_URL_PARAM_MAP: Dict[str, Dict[str, Callable[[Any], Any]]] = {
    'api_source': {
        'state_key': 'api_source',
        'parser': _parse_api_source_param,
        'serializer': lambda v: None if v is None else str(v),
    },
    'player_id': {
        'state_key': 'player_id',
        'parser': lambda v: str(v),
        'serializer': lambda v: None if v is None else str(v),
    },
    'session_id': {
        'state_key': 'session_id',
        # 'latest' means most recent game (same as omitting session_id).
        'parser': lambda v: None if str(v).strip().lower() == 'latest' else int(v),
        'serializer': lambda v: None if v is None else str(int(v)),
    },
    'single_dummy_sample_count': {
        'state_key': 'single_dummy_sample_count',
        'parser': lambda v: int(v),
        'serializer': lambda v: None if v is None else str(int(v)),
    },
    'show_sql_query': {
        'state_key': 'show_sql_query',
        'parser': _parse_bool_param,
        'serializer': lambda v: None if v is None else ('1' if bool(v) else '0'),
    },
    'debug_mode': {
        'state_key': 'debug_mode',
        'parser': _parse_bool_param,
        'serializer': lambda v: None if v is None else ('1' if bool(v) else '0'),
    },
}


def apply_url_params_to_session_state() -> None:
    """Read URL query params and write them into st.session_state for registered sidebar options.

    Called once during session initialization (after defaults are set) so URL params
    take precedence over defaults. Unknown / unparseable params are skipped with a warning.
    """
    qp = st.query_params
    for url_key, cfg in SIDEBAR_URL_PARAM_MAP.items():
        if url_key not in qp:
            continue
        raw = qp[url_key]
        try:
            value = cfg['parser'](raw)
        except Exception as e:
            print(f"Warning: ignoring URL param {url_key}={raw!r}: {e}")
            continue
        st.session_state[cfg['state_key']] = value


def sync_session_state_to_url_params() -> None:
    """Write registered sidebar options from st.session_state back to URL query params.

    Called near the end of every render so the URL always reflects current sidebar state.
    Idempotent: only writes when the serialized value differs from the current URL param.
    Values that serialize to None (e.g. None state) are removed from the URL.
    """
    qp = st.query_params
    for url_key, cfg in SIDEBAR_URL_PARAM_MAP.items():
        state_key = cfg['state_key']
        value = st.session_state.get(state_key, None)
        serialized = cfg['serializer'](value)
        current = qp.get(url_key)
        if serialized is None:
            if url_key in qp:
                del qp[url_key]
        else:
            if current != serialized:
                qp[url_key] = serialized


def api_source_on_change() -> None:
    """Handle API source selectbox change: switch backend and clear per-source state.

    Classic and Lancelot use different id spaces (person ids, session ids), so any
    player/game state from the previous source is invalid after a switch.
    """
    st.session_state.api_source = st.session_state.api_source_selectbox
    st.session_state.player_id = None
    st.session_state.session_id = None
    st.session_state.game_urls_d = {}
    st.session_state.game_url = None
    st.session_state.df = None
    st.session_state.sql_query_mode = False
    st.session_state.deferred_start_report = False
    for key in ('player_search_error', 'player_search_matches', 'show_player_modal', 'simultane_id', '_url_loaded_session_key'):
        if key in st.session_state:
            del st.session_state[key]


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


def _id_list(*values: Any) -> List[str]:
    seen: List[str] = []
    for raw in values:
        if raw is None:
            continue
        text = str(raw).strip()
        if text and text not in seen:
            seen.append(text)
    return seen


def _player_match_ids() -> List[str]:
    """License, Lancelot, and Classic ids that all name the selected player."""
    ids = _id_list(
        st.session_state.get("player_id"),
        st.session_state.get("lancelot_player_id"),
        st.session_state.get("classic_player_id"),
    )
    license_number = st.session_state.get("player_license_number")
    # reset_game_data defaults player_license_number to 9500754 as a
    # placeholder. Only treat it as this player when it is the requested
    # id or resolve/generate has bound a Lancelot id.
    if license_number and (
        str(license_number) == str(st.session_state.get("player_id") or "")
        or st.session_state.get("lancelot_player_id")
    ):
        ids = _id_list(*ids, license_number)
    return ids


def _partner_match_ids() -> List[str]:
    return _id_list(
        st.session_state.get("partner_id"),
        st.session_state.get("partner_license_number"),
    )


def filter_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Filter DataFrame to show boards played by specific player and partner
    
    Args:
        df: Input DataFrame containing board data
        
    Returns:
        Filtered DataFrame with additional boolean columns for board filtering
    """

    # Player_ID_* / lineup_* / Declarer_ID may store a license, Lancelot id,
    # or Classic id depending on cache vintage. Match every known alias.
    player_ids = _player_match_ids()
    partner_ids = _partner_match_ids()
    full_directions_d = {'N':'north', 'E':'east', 'S':'south', 'W':'west'}
    if f"lineup_{full_directions_d[st.session_state.player_direction]}Player_id" in df.columns:
        lineup_col = f'lineup_{full_directions_d[st.session_state.player_direction]}Player_id'
        df = df.with_columns(
        pl.col(lineup_col).cast(pl.Utf8).is_in(player_ids).alias('Boards_I_Played'),
        pl.col('Declarer_ID').cast(pl.Utf8).is_in(player_ids).alias('Boards_I_Declared'),
        pl.col('Declarer_ID').cast(pl.Utf8).is_in(partner_ids).alias('Boards_Partner_Declared'),
    )
    elif "Pair_Direction" in df.columns:
        # todo: better way to determine Boards_I_Played than above?
        df = df.with_columns(
            pl.col(f'Player_ID_{st.session_state.player_direction}').cast(pl.Utf8).is_in(player_ids).alias('Boards_I_Played'),
        )
        df = df.with_columns(
            pl.col('Boards_I_Played').and_(pl.col('Declarer_Direction').eq(st.session_state.player_direction)).alias('Boards_I_Declared'),
            pl.col('Boards_I_Played').and_(pl.col('Declarer_Direction').eq(st.session_state.partner_direction)).alias('Boards_Partner_Declared'),
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


def _df_from_json_normalize(json_data: Any, sep: str = '_') -> pl.DataFrame:
    """Normalize JSON with pandas, then build a Polars DataFrame in a tolerant way.

    Some FFBridge endpoints return mixed types for the same field across rows (e.g. a number sometimes
    comes back as a string like "78.022"). Using strict=False avoids hard failures during ingestion.
    """
    df_pd = pd.json_normalize(json_data, sep=sep)
    try:
        # Prefer from_pandas; it's explicit and lets us retry with cleaned dtypes.
        return pl.from_pandas(df_pd, include_index=False)
    except Exception as e:
        # pyarrow/lib conversion can fail when a column is inferred as numeric but contains strings.
        # Example: "Could not convert '78.022' with type str: tried to convert to double"
        if type(e).__name__ != 'ArrowInvalid' and 'ArrowInvalid' not in str(e):
            raise

        df_pd2 = df_pd.copy()
        for col in df_pd2.columns:
            s = df_pd2[col]
            # Only attempt cleanup on object-like columns; numeric columns are fine.
            if getattr(s.dtype, 'kind', None) == 'O' or str(s.dtype) == 'object':
                # If this column contains nested structures (lists/dicts), keep them as-is.
                # These are required for downstream explode/unnest logic (e.g. 'roadsheets', 'teams').
                try:
                    sample = [v for v in s.dropna().head(50).tolist()]
                    has_nested = any(isinstance(v, (list, dict)) for v in sample)
                except Exception:
                    has_nested = False

                if has_nested:
                    continue

                # Otherwise, normalize common French numeric formatting (comma decimal separator) before conversion.
                s_str = s.astype('string')
                s_num_candidate = pd.to_numeric(
                    s_str.str.replace(',', '.', regex=False),
                    errors='coerce'
                )
                non_null = int(s_str.notna().sum())
                num_notna = int(s_num_candidate.notna().sum())

                # If every non-null value successfully converts, keep numeric; otherwise force string dtype
                # to prevent Arrow from trying to build a floating column.
                if non_null > 0 and num_notna == non_null:
                    df_pd2[col] = s_num_candidate
                else:
                    df_pd2[col] = s_str

        # Retry after dtype cleanup. If Arrow still can't handle nested/object columns, bypass Arrow entirely.
        try:
            return pl.from_pandas(df_pd2, include_index=False)
        except Exception:
            return pl.DataFrame(df_pd2.to_dict('records'), strict=False)


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
            df = _df_from_json_normalize(json_data, sep='_')
        case 'simultaneous_deals':
            json_datas = []
            for i in range(1, st.session_state.nb_deals+1):
                deal_url = url.format(i=i)
                json_data = make_api_request_licencie(deal_url)
                if json_data is None:
                    raise Exception(f"Failed to get data from {deal_url}")
                
                assert isinstance(json_data, dict), f"Expected a dict, got {type(json_data)}"
                json_datas.append(json_data)
            
            df = _df_from_json_normalize(json_datas, sep='_')
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
            
            # Stringify nested struct/list columns so they display readable text instead of "Object"
            # The frequencies column is List[Struct[5]] with fields: ewNote, ewScore, nsNote, nsScore, count
            # We use list.eval with concat_str to build a readable string representation
            # IMPORTANT: Worked on getting frequencies to display readable text instead of "[object Object]" but gave up. Might be bug in Polars or AgGrid.
            for nested_col in ['frequencies', 'frequencies_organizations']:
                if nested_col in df.columns:
                    try:
                        # Use pure Polars expressions to convert List[Struct] to readable string
                        # Format: [EW=score/NS=score, ewNote/nsNote, n=count]; ...
                        df = df.with_columns(
                            pl.col(nested_col).list.eval(
                                pl.concat_str([
                                    pl.lit("[EW="),
                                    pl.when(pl.element().struct.field("ewScore") != "")
                                        .then(pl.element().struct.field("ewScore"))
                                        .otherwise(pl.lit("-")),
                                    pl.lit("/"),
                                    pl.when(pl.element().struct.field("nsScore") != "")
                                        .then(pl.element().struct.field("nsScore"))
                                        .otherwise(pl.lit("-")),
                                    pl.lit(", "),
                                    pl.element().struct.field("ewNote").cast(pl.Utf8),
                                    pl.lit("/"),
                                    pl.element().struct.field("nsNote").cast(pl.Utf8),
                                    pl.lit(", n="),
                                    pl.element().struct.field("count").cast(pl.Utf8),
                                    pl.lit("]"),
                                ])
                            ).list.join("; ").alias(nested_col)
                        )
                    except Exception as e:
                        print(f"Warning: Could not stringify {nested_col}: {e}")
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
            
            df = _df_from_json_normalize(json_datas, sep='_')
        case 'simultaneous_dealsNumber':
            json_data = make_api_request_licencie(url)
            if json_data is None:
                raise Exception(f"Failed to get data from {url}")
            
            assert isinstance(json_data, dict), f"Expected a dict, got {type(json_data)}"
            df = _df_from_json_normalize(json_data, sep='_')
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
            df = _df_from_json_normalize(json_data, sep='_')
            
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
            df = _df_from_json_normalize(json_data, sep='_')
            
            # Check if we got an empty result (no matches found)
            if df.is_empty():
                raise Exception(f"No data found for {k} from {url}")
            
            # Rename 'id' to 'simultane_id' if it exists
            # Some API responses don't have an 'id' column (e.g. https://api.ffbridge.fr/api/v1/simultaneous-tournaments/2991057)
            if 'id' in df.columns:
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
            df = _df_from_json_normalize(json_data, sep='_')
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
                                print(f"Column '{exploded_col_name}' contains only null values, skipping struct processing")
                        except Exception as e:
                            print(f"Error processing column '{exploded_col_name}': {e}. Skipping struct processing.")
                    else:
                        print(f"Column '{exploded_col_name}' is empty or all null, skipping struct processing")
                else:
                    print(f"Column '{exploded_col_name}' not found in DataFrame, skipping")
        case _:
            json_data = make_api_request_licencie(url)
            if json_data is None:
                raise Exception(f"Failed to get data from {url} possibly due to data not yet available. Try again in 24 hours.")
    
            # Create DataFrame from the JSON response. json_data can be a dict or a list.
            try:
                df = _df_from_json_normalize(json_data, sep='_')
            except Exception as e:
                raise Exception(f"Failed to create DataFrame from {url}. Data may not yet be available, or the API returned mixed/invalid types. {e}")
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
def _apply_lancelot_session_meta(meta: Any) -> None:
    """Copy Lancelot session metadata (dataclass or API dict) into session_state."""
    if not isinstance(meta, dict):
        meta = {
            "session_id": meta.session_id,
            "group_id": meta.group_id,
            "org_id": meta.org_id,
            "tournament_date": meta.tournament_date,
            "organization_name": meta.organization_name,
            "game_description": meta.game_description,
            "route_url": meta.route_url,
            "team_id": meta.team_id,
            "pair_direction": meta.pair_direction,
            "opponent_pair_direction": meta.opponent_pair_direction,
            "player_direction": meta.player_direction,
            "partner_direction": meta.partner_direction,
            "player_id": meta.player_id,
            "partner_id": meta.partner_id,
            "player_license_number": meta.player_license_number,
            "partner_license_number": meta.partner_license_number,
            "player_name": meta.player_name,
            "partner_name": meta.partner_name,
            "section_name": meta.section_name,
            "team_number": meta.team_number,
            "game_url": meta.game_url,
        }
    session_id = meta.get("session_id")
    if session_id is not None:
        st.session_state.session_id = int(session_id)
        st.session_state.simultane_id = int(session_id)
    st.session_state.group_id = meta.get("group_id")
    st.session_state.org_id = meta.get("org_id")
    st.session_state.tournament_date = meta.get("tournament_date") or meta.get("game_date")
    st.session_state.organization_name = meta.get("organization_name")
    st.session_state.game_description = meta.get("game_description")
    st.session_state.route_url = meta.get("route_url")
    st.session_state.team_id = meta.get("team_id")
    st.session_state.pair_direction = meta.get("pair_direction")
    st.session_state.opponent_pair_direction = meta.get("opponent_pair_direction")
    st.session_state.player_direction = meta.get("player_direction")
    st.session_state.partner_direction = meta.get("partner_direction")
    incoming_player_id = meta.get("player_id")
    incoming_license = meta.get("player_license_number")
    incoming_lancelot = (
        meta.get("cache_player_id")
        or meta.get("matched_player_id")
        or (
            incoming_player_id
            if incoming_player_id and str(incoming_player_id) != str(incoming_license or "")
            else None
        )
    )
    if incoming_lancelot and not st.session_state.get("lancelot_player_id"):
        st.session_state.lancelot_player_id = str(incoming_lancelot)
    if incoming_license:
        st.session_state.player_license_number = str(incoming_license)
    # Keep the URL/sidebar player_id as the originally requested value
    # (usually a license). Overwriting it with the Lancelot person id
    # rewrites ?player_id=9500754 to 246273 and then filter_dataframe
    # looks up 246273 as a license.
    if not st.session_state.get("player_id"):
        st.session_state.player_id = incoming_license or incoming_player_id
    st.session_state.partner_id = meta.get("partner_id")
    if meta.get("partner_license_number"):
        st.session_state.partner_license_number = meta.get("partner_license_number")
    st.session_state.player_name = meta.get("player_name")
    st.session_state.partner_name = meta.get("partner_name")
    st.session_state.section_name = meta.get("section_name")
    st.session_state.team_number = meta.get("team_number")
    st.session_state.game_url = meta.get("game_url")
    if not st.session_state.get("group_id"):
        st.session_state.group_id = meta.get("group_id")
    if not st.session_state.get("team_id"):
        st.session_state.team_id = meta.get("team_id")


def _usable_results_url(url: Any) -> Optional[str]:
    text = str(url or "").strip()
    if not text.startswith("http"):
        return None
    if "/groups/None/" in text or "/groups/none/" in text:
        return None
    return text


def _session_results_url_from_entry(entry: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not entry:
        return None
    return _usable_results_url(entry.get("results_url") or entry.get("game_url")) or (
        pm_create.ffbridge_results_page_url(
            session_id=entry.get("session_id") or st.session_state.get("session_id"),
            group_id=entry.get("group_id"),
            team_id=entry.get("team_id"),
        )
    )


def _ensure_game_results_url(df: Optional[pl.DataFrame] = None) -> Optional[str]:
    """Resolve and store the public results page for the selected session."""
    current = _usable_results_url(st.session_state.get("game_url"))
    if current:
        st.session_state.game_url = current
        return current

    session_id = st.session_state.get("session_id")
    group_id = st.session_state.get("group_id")
    team_id = st.session_state.get("team_id")
    organization_id = st.session_state.get("org_id")
    player_id = st.session_state.get("player_id")
    game_urls = st.session_state.get("game_urls_d") or {}
    entry = None
    if player_id is not None and session_id is not None:
        player_games = game_urls.get(str(player_id)) or game_urls.get(player_id) or {}
        entry = player_games.get(int(session_id)) or player_games.get(session_id)
    if entry:
        from_entry = _session_results_url_from_entry(entry)
        if from_entry:
            st.session_state.game_url = from_entry
            return from_entry
        group_id = group_id or entry.get("group_id")
        team_id = team_id or entry.get("team_id")
        organization_id = organization_id or entry.get("organization_id")

    frame = df if df is not None else st.session_state.get("df")
    if isinstance(frame, pl.DataFrame):
        if not group_id and "group_id" in frame.columns:
            group_id = frame["group_id"].drop_nulls().first()
        if not team_id and "team_id" in frame.columns:
            team_id = frame["team_id"].drop_nulls().first()

    url = pm_create.ffbridge_results_page_url(
        session_id=session_id,
        group_id=group_id,
        team_id=team_id,
    )
    if url is None and session_id is not None and not group_id:
        cache_dir = st.session_state.get("cache_dir")
        group_id = pm_create.resolve_session_group_id(
            session_id,
            organization_id=organization_id,
            cache_dir=pathlib.Path(cache_dir) if cache_dir else None,
        )
        if group_id:
            st.session_state.group_id = group_id
            url = pm_create.ffbridge_results_page_url(
                session_id=session_id,
                group_id=group_id,
                team_id=team_id,
            )
    if url:
        st.session_state.game_url = url
        if group_id:
            st.session_state.group_id = group_id
        if team_id:
            st.session_state.team_id = team_id
    return url


def get_lancelot_session_mldf(player_id: str, session_id: int, game_entry: Dict[str, Any]) -> pl.DataFrame:
    """Build the report mldf for a Lancelot session (shared create path)."""
    built = pm_create.build_lancelot_session_mldf(
        player_id,
        session_id,
        game_entry,
        token=get_lancelot_token(),
        cache_dir=pathlib.Path(st.session_state.cache_dir),
        team_progress=lambda ids: stqdm(ids, desc='Downloading team scores...'),
    )
    _apply_lancelot_session_meta(built.meta)
    if st.session_state.get('debug_mode', False):
        debug_capture_df("lancelot_ranking", built.ranking_df, source=f"results/sessions/{session_id}/ranking")
        debug_capture_df("lancelot_scores", built.scores_df, source=f"results/teams/*/session/{session_id}/scores")
    return built.df


def get_ffbridge_licencie_get_urls(api_urls_d: Dict[str, Tuple[str, bool]]) -> Tuple[Dict[str, pl.DataFrame], Dict[str, Tuple[str, bool]]]:
    """Get FFBridge data using URL configuration and display results
    
    Args:
        api_urls_d: Dictionary mapping API names to (URL, should_cache) tuples
        
    Returns:
        Tuple of (DataFrames dictionary, API URLs dictionary)
    """

    dfs, api_urls_d = get_ffbridge_data_using_url_licencie(api_urls_d)

    # Capture debug snapshots instead of rendering inline (so they persist across reruns).
    if st.session_state.get('debug_mode', False):
        for k, v in dfs.items():
            url = None
            try:
                url = api_urls_d[k][0]
            except Exception:
                url = None
            debug_capture_df(f"{k}", v, source=url)

    return dfs, api_urls_d


def _remember_resolved_player_ids(resolved: Dict[str, Any]) -> None:
    """Store every identifier namespace without changing the public player_id."""
    lancelot_id = resolved.get("player_id") or resolved.get("lancelot_id")
    license_number = resolved.get("player_license_number") or resolved.get("license_number")
    classic_id = resolved.get("classic_person_id")
    if lancelot_id:
        st.session_state.lancelot_player_id = str(lancelot_id)
    if license_number:
        st.session_state.player_license_number = str(license_number)
    if classic_id:
        st.session_state.classic_player_id = str(classic_id)


def resolve_url_player_id_param(value: str) -> str:
    """Keep the public ``?player_id=`` value and remember internal aliases.

    The sidebar and shared URLs use the FFBridge license number (e.g.
    ``9500754``). Lancelot APIs and cache files use the person id
    (``246273``). Classic APIs use the migration/person id. Rewriting the
    URL from license to Lancelot id made filter_dataframe look up 246273 as
    a license and produced an empty/wrong report.

    Lancelot mode returns ``value`` unchanged after stashing aliases.
    Classic mode still remaps a license to Classic person_id because those
    endpoints require it.
    """
    v = (value or "").strip()
    if not v or not v.isdigit():
        return value  # non-numeric -- definitely not a license number

    if is_lancelot_mode():
        try:
            resolved = pm_api.resolve_player(v)
            _remember_resolved_player_ids(resolved)
            print(
                f"resolve_url_player_id_param: keeping URL player_id={v!r}; "
                f"lancelot={resolved.get('player_id')!r} "
                f"license={resolved.get('player_license_number')!r}."
            )
        except Exception as e:
            print(f"resolve_url_player_id_param({v!r}): Lancelot resolve failed, "
                  f"falling through to direct lookup: {e}")
        return value

    try:
        search_df = search_members(v)
    except Exception as e:
        print(f"resolve_url_player_id_param({v!r}): search call failed, "
              f"falling through to direct lookup: {e}")
        return value
    if search_df is None or len(search_df) != 1:
        return value  # 0 or 2+ hits -- assume the URL value is already a person_id

    row = list(search_df.iter_rows(named=True))[0]
    # Match the field-name probing the manual flow uses (line ~2186).
    license_from_api = (
        row.get('person_license_number', '')
        or row.get('license_number', '')
        or row.get('licenseNumber', '')
        or ''
    )
    license_norm = str(license_from_api).lstrip('0')
    value_norm = v.lstrip('0')
    if license_norm and license_norm == value_norm:
        person_id = row.get('person_id')
        if person_id is not None:
            print(f"resolve_url_player_id_param: {v!r} matched license number; "
                  f"resolved to person_id={person_id}.")
            if license_from_api:
                st.session_state.player_license_number = str(license_from_api)
            st.session_state.classic_player_id = str(person_id)
            return str(person_id)
    return value


def _populate_game_urls_for_player_lancelot(player_id: str) -> bool:
    """Lancelot branch of populate_game_urls_for_player.

    Session lists come from the shared player-session index via the writer API.
    A live Lancelot login is not required to list games; it is only needed later
    to download missing board scores.
    """
    try:
        listed = pm_api.list_source_sessions(player_id)
    except pm_api.FfbridgeApiClientError as e:
        st.session_state.player_search_error = str(e)
        return False

    game_urls: Dict[int, Dict[str, Any]] = {}
    for entry in listed['sessions']:
        session_id = int(entry['session_id'])
        game_urls[session_id] = {
            'description': entry.get('description'),
            'date': entry.get('date'),
            'session_id': session_id,
            'group_id': entry.get('group_id'),
            'organization_id': entry.get('organization_id'),
            'organization_name': entry.get('organization_name') or entry.get('club'),
            'competition_label': entry.get('competition_label'),
            'session_label': entry.get('session_label'),
            'team_id': entry.get('team_id'),
            'results_url': pm_create.ffbridge_results_page_url(
                session_id=session_id,
                group_id=entry.get('group_id'),
                team_id=entry.get('team_id'),
            ),
        }

    _remember_resolved_player_ids(listed)
    canonical_id = listed['player_id']
    st.session_state.game_urls_d[canonical_id] = game_urls
    if player_id != canonical_id:
        st.session_state.game_urls_d[player_id] = game_urls
    license_number = listed.get('player_license_number')
    if license_number and str(license_number) != player_id:
        st.session_state.game_urls_d[str(license_number)] = game_urls
    ordered = {
        int(row["session_id"]): game_urls[int(row["session_id"])]
        for row in pm_create.sessions_newest_first(game_urls.values())
    }
    for key in list(st.session_state.game_urls_d):
        if st.session_state.game_urls_d[key] is game_urls:
            st.session_state.game_urls_d[key] = ordered
    st.session_state.person_organization_id = None
    return len(ordered) > 0


def populate_game_urls_for_player(player_id: str) -> bool:
    """Populate st.session_state.game_urls_d for a given player without changing session_id.
    Returns True if games were populated (length > 0), False otherwise.
    """
    if player_id in st.session_state.game_urls_d and st.session_state.game_urls_d[player_id]:
        return True
    if is_lancelot_mode():
        return _populate_game_urls_for_player_lancelot(player_id)
    api_urls_d = {
        'members': (classic_api_url(f"members/{player_id}"), False),
        'person': (classic_api_url(f"licensee-results/results/person/{player_id}?date=all&place=0&type=0"), False),
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


def _finalize_mldf_for_report(df: pl.DataFrame) -> bool:
    """Shared report-preparation tail for both API sources.

    Validates the mldf, reduces it to the columns augmentation needs, augments
    (with parquet caching), personalizes via filter_dataframe, and registers the
    result as the 'self' DuckDB view. Returns True on error, False on success.
    """
    if st.session_state.debug_mode:
        debug_capture_df("Final Dataframe", df, source="change_game_state")

    if df['Contract'].is_null().all(): # ouch. e.g. Monday Simultané Octopus
        st.error("No Contract data available. Unable to proceed.")
        return True

    st.session_state.session_id = st.session_state.simultane_id

    if not st.session_state.use_historical_data: # historical data is already fully augmented so skip past augmentations
        with st.spinner('Creating ffbridge data to dataframe...'):
            df = pm_create.augment_and_cache_mldf(
                df,
                st.session_state.session_id,
                st.session_state.player_id,
                cache_dir=pathlib.Path(st.session_state.cache_dir),
                force=False,
                sd_productions=st.session_state.single_dummy_sample_count,
                progress=st.progress(0),
                lock_func=perform_hand_augmentations_queue,
                write_cache=not st.session_state.do_not_cache_df,
            )
        if df is not None:
            st.session_state.df_ready = True
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


def _normalize_session_id_arg(session_id: Any) -> Any:
    """Treat 'latest' (case-insensitive) as None so callers pick the most recent game."""
    if isinstance(session_id, str) and session_id.strip().lower() == 'latest':
        return None
    return session_id


def _report_failure(message: Any) -> bool:
    """Persist a report error across Streamlit reruns and render it now."""
    detail = str(message)
    st.session_state.report_error = detail
    st.error(detail)
    return True


def _change_game_state_lancelot(player_id: str, session_id: Optional[int]) -> bool:
    """Lancelot branch of change_game_state. Returns True on error, False on success."""
    session_id = _normalize_session_id_arg(session_id)
    with st.spinner(f"Retrieving a list of games for {player_id} ..."):
        if not populate_game_urls_for_player(player_id):
            return _report_failure(
                st.session_state.get('player_search_error')
                or f"Could not find any games for {player_id}."
            )
        game_urls = st.session_state.game_urls_d[player_id]
        if session_id is None:
            session_id = next(iter(game_urls))  # most recent game
        session_id = int(session_id)
        if session_id not in game_urls:
            return _report_failure(
                f"Session {session_id} not found in games for {player_id}."
            )
        st.session_state.player_id = player_id

    with st.spinner('Preparing Bridge Game Postmortem Report...'):
        try:
            gen = pm_api.generate_and_wait(str(player_id), session_id=str(session_id))
        except pm_api.FfbridgeApiClientError as e:
            return _report_failure(e)
        if gen.get("status") == "error":
            return _report_failure(
                gen.get("error") or "Postmortem generate failed."
            )
        _remember_resolved_player_ids(gen)
        sid = str(gen.get("session_id") or session_id)
        results = gen.get("results") or gen.get("sessions") or []
        row = next((r for r in results if str(r.get("session_id")) == sid), results[0] if results else gen)
        meta = row.get("meta") if isinstance(row, dict) else None
        if not meta:
            meta = pm_api.postmortem_meta(str(player_id), sid)
        _apply_lancelot_session_meta(meta)
        df = pm_api.postmortem_dataframe(str(player_id), sid)
        if st.session_state.debug_mode:
            debug_capture_df("Final Dataframe", df, source="ffbridge_postmortem_api")
        st.session_state.df = filter_dataframe(df)
        st.session_state.df_ready = True
        _ensure_game_results_url(st.session_state.df)
        con = get_session_duckdb_connection()
        con.register("self", st.session_state.df)

    st.session_state.pop("report_error", None)
    print(f"=== change_game_state END: SUCCESS - player_id={st.session_state.player_id}, session_id={st.session_state.session_id} ===")
    return False


def change_game_state(player_id: str, session_id: str) -> bool: # todo: rename to session_id?

    # Keep player_id stable as a string; other parts of the UI (e.g., game_urls_d keys) depend on this.
    player_id = str(player_id) if player_id is not None else player_id
    session_id = _normalize_session_id_arg(session_id)

    print(f"=== change_game_state START: player_id={player_id}, session_id={session_id} ===")

    st.markdown('<div style="height: 50px;"><a name="top-of-report"></a></div>', unsafe_allow_html=True)

    con = get_session_duckdb_connection()

    if is_lancelot_mode():
        try:
            return _change_game_state_lancelot(player_id, session_id)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return _report_failure(f"Error preparing Lancelot report: {e}")

    with st.spinner(f"Retrieving a list of games for {player_id} ..."):
        t = time.time()
        if player_id not in st.session_state.game_urls_d:
            if True: # keeps indentation; Lancelot handling moved to _change_game_state_lancelot()
                api_urls_d = {
                    'members': (classic_api_url(f"members/{player_id}"), False),
                    'person': (classic_api_url(f"licensee-results/results/person/{player_id}?date=all&place=0&type=0"), False),
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
            'simultaneous_tournaments': (classic_api_url(f"simultaneous-tournaments/{st.session_state.simultane_id}"), False),
        }
        dfs, api_urls_d = get_ffbridge_licencie_get_urls(api_urls_d)
        simultaneous_tournaments_df = dfs['simultaneous_tournaments']
        st.session_state.player_row = simultaneous_tournaments_df.filter(
            pl.col('team_players_id').cast(pl.Int64) == int(st.session_state.player_id)
        )
        if st.session_state.get('debug_mode', False):
            debug_capture_df("player_row", st.session_state.player_row, source="simultaneous_tournaments")
        # Do NOT change player_id type here; ensure it remains a string for consistent dict-keying and widgets.
        st.session_state.player_id = str(st.session_state.player_row['team_players_id'].first())
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
        if st.session_state.get('debug_mode', False):
            debug_capture_df("partner_row", st.session_state.partner_row, source="simultaneous_tournaments")
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
            'simultaneous_roadsheets': (classic_api_url(f"simultaneous-tournaments/{st.session_state.simultane_id}/teams/{st.session_state.team_id}/roadsheets"), False),
            'simultaneous_dealsNumber': (classic_api_url(f"simultaneous-tournaments/{st.session_state.simultane_id}/teams/{st.session_state.team_id}/dealsNumber"), False),
            'simultaneous_deals': (classic_api_url(f"simultaneous-tournaments/{st.session_state.simultane_id}/teams/{st.session_state.team_id}/deals/{{i}}"), False),
            #'simultaneous_descriptions': (classic_api_url(f"simultaneous-tournaments/{st.session_state.simultane_id}/teams/{st.session_state.team_id}/deals/{{i}}/descriptions"), False),
            'simultaneous_description_by_organization_id': (classic_api_url(f"simultaneous/{st.session_state.simultane_id}/deals/{{i}}/descriptions?organization_id={st.session_state.org_id}"), False),
            'simultaneous_tournaments_by_organization_id': (classic_api_url(f"simultaneous-tournaments/{st.session_state.simultane_id}?organization_id={st.session_state.org_id}"), False),
            'my_infos': (classic_api_url("users/my/infos"), False),
            'members': (classic_api_url(f"members/{st.session_state.player_id}"), False),
            'person': (classic_api_url(f"licensee-results/results/person/{st.session_state.player_id}?date=all&place=0&type=0"), False),
            'organization_by_person_organization_id': (classic_api_url(f"licensee-results/results/organization/{st.session_state.org_id}?date=all&person_organization_id={str(st.session_state.person_organization_id or '')}&place=0&type=0"), False),
            'person_by_person_organization_id': (classic_api_url(f"licensee-results/results/person/{st.session_state.player_id}?date=all&person_organization_id={str(st.session_state.person_organization_id or '')}&place=0&type=0"), False),
        }
        dfs, api_urls = get_ffbridge_licencie_get_urls(api_urls_d)
        if st.session_state.simultaneeCode  == 'RRN':
            # RRN (Roy Rene simultaneious tournament) has no deal related columns in the simultaneous_deals dataframe.
            # so we need to get the boards from the tournament date and add the deal related columns to the simultaneous_deals dataframe.
            st.session_state.tournament_id = mlBridgeBPLib.get_teams_by_tournament_date(st.session_state.tournament_date)
            max_deals = 36 # todo: is max_deals (number of deals) available in any API at this point?
            deal_numbers = dfs['simultaneous_roadsheets']['roadsheets_deals_dealNumber'].unique().to_list()
            # uses st.session_state.player_license_number to get boards because Roy Rene website works with player_license_number to get boards.
            with st.spinner(f"Roy Rene tournaments require an extra step. Takes 1 to 3 minutes..."):
                # Get the Bridge+ club page to find the player's route link
                async def get_player_route_url():
                    # Fetch the club teams page which contains links to each pair's route
                    teams_df = await mlBridgeBPLib.get_teams_by_tournament_async(
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
                    boards_dfs = mlBridgeBPLib.get_all_boards_for_player(st.session_state.tournament_id, st.session_state.organization_code, st.session_state.player_license_number, max_deals=36)
                else:
                    # Get boards data with progress bar by processing boards one by one using the existing function
                    boards_dfs = {'boards': None, 'score_frequency': None}
                    
                    try:
                        
                        async def get_boards_with_progress():
                            # First, get the route data to see which boards this player actually played
                            # We need to find the player's team first to get the route data
                            # teams_df = await mlBridgeBPLib.get_teams_by_tournament_async(st.session_state.tournament_id, st.session_state.organization_code)
                            
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
                            async with mlBridgeBPLib.get_browser_context_async() as context:
                                try:
                                    route_results = await mlBridgeBPLib.request_board_results_dataframe_async(st.session_state.route_url, context)
                                    if len(route_results) == 0:
                                        st.warning(f"No route data found for team {st.session_state.team_number}")
                                        print(f"Route page returned empty - will try fallback: fetch boards directly")
                                        # Don't return empty yet - try fallback below
                                        played_boards = []  # Will trigger fallback
                                    else:
                                        played_boards = route_results['Board'].to_list()
                                        print(f"Found {len(played_boards)} boards played by team {st.session_state.team_number}: {played_boards}")
                                except Exception as e:
                                    print(f"Error getting route data for team {st.session_state.team_number}: {e}")
                                    print(f"Will try fallback: fetch boards directly")
                                    # Don't raise - try fallback instead
                                    played_boards = []  # Will trigger fallback
                            
                            # If no boards found in route, try fallback: fetch boards directly
                            if not played_boards:
                                print(f"No boards found in route data for team {st.session_state.team_number}")
                                print("Attempting fallback: trying to fetch boards directly using p=donne URLs")
                                print("This will try boards 1-40 (or until we find boards that exist)")
                                # Try a reasonable range of boards (typically tournaments have 20-40 boards)
                                max_deals = 40
                                played_boards = list(range(1, max_deals + 1))
                                print(f"Will try boards: {played_boards[:10]}... (up to {max_deals} boards)")
                            
                            # Create progress bar for board processing
                            progress_bar = st.progress(0)
                            progress_text = st.empty()
                            
                            # Now get board data only for the boards that were actually played using the existing function
                            all_boards = []
                            all_frequency = []
                            
                            async with mlBridgeBPLib.get_browser_context_async() as context:
                                for idx, deal_num in enumerate(played_boards):
                                    try:
                                        # Update progress
                                        progress = (idx + 1) / len(played_boards)
                                        progress_bar.progress(progress)
                                        progress_text.text(f"Processing board {idx + 1}/{len(played_boards)}: Board {deal_num}")
                                        
                                        # Try p=donne URL first (team-specific)
                                        result = None
                                        try:
                                            result = await mlBridgeBPLib.get_board_for_player_async(
                                                st.session_state.tournament_id, 
                                                st.session_state.organization_code, 
                                                st.session_state.player_license_number, 
                                                str(deal_num), 
                                                context
                                            )
                                            print(f"Board {deal_num}: get_board_for_player_async returned result with {len(result.get('boards', []))} boards")
                                        except Exception as e1:
                                            # If p=donne fails, try p=board URL (tournament-wide fallback)
                                            print(f"p=donne URL failed for board {deal_num}: {e1}")
                                            print(f"Trying fallback: p=board URL (tournament-wide board view)")
                                            try:
                                                board_url = f"https://www.bridgeplus.com/nos-simultanes/resultats/?p=board&res=sim&d={deal_num}&tr={st.session_state.tournament_id}"
                                                result = await mlBridgeBPLib.request_boards_dataframe_async(board_url, context)
                                                print(f"Board {deal_num}: p=board fallback returned result with {len(result.get('boards', []))} boards")
                                            except Exception as e2:
                                                print(f"Both p=donne and p=board URLs failed for board {deal_num}: {e2}")
                                                raise e2
                                        
                                        if result:
                                            boards_count = len(result.get('boards', []))
                                            freq_count = len(result.get('score_frequency', []))
                                            print(f"Board {deal_num}: result has {boards_count} boards, {freq_count} frequency records")
                                            
                                            if boards_count > 0:
                                                all_boards.append(result['boards'])
                                                print(f"Successfully added board {deal_num} to all_boards (total: {len(all_boards)})")
                                            else:
                                                print(f"Board {deal_num}: result['boards'] is empty, skipping")
                                                
                                            if freq_count > 0:
                                                all_frequency.append(result['score_frequency'])
                                        else:
                                            print(f"Board {deal_num}: result is None, skipping")
                                    except Exception as e:
                                        print(f"Failed to scrape board {deal_num} for player {st.session_state.player_license_number}: {e}")
                                        import traceback
                                        print(f"Traceback: {traceback.format_exc()}")
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
                            print(f"Finished processing. Total boards fetched: {len(all_boards)}")
                            if all_boards:
                                combined_boards = pl.concat(all_boards, how='vertical_relaxed')
                                print(f"Combined boards DataFrame height: {combined_boards.height}")
                            else:
                                combined_boards = pl.DataFrame()
                                print("WARNING: No boards were accumulated in all_boards list!")
                            
                            if all_frequency:
                                combined_frequency = pl.concat(all_frequency, how='vertical_relaxed')
                            else:
                                combined_frequency = pl.DataFrame()
                            
                            print(f"Returning boards_dfs with {combined_boards.height} boards")
                            return {
                                'boards': combined_boards,
                                'score_frequency': combined_frequency
                            }
                        
                        # Run the async function
                        boards_dfs = asyncio.run(get_boards_with_progress())
                        print(f"After asyncio.run: boards_dfs has {boards_dfs.get('boards', pl.DataFrame()).height} boards")
                        
                    except Exception as e:
                        st.error(f"Error getting boards for player {st.session_state.player_license_number}: {e}")
                        import traceback
                        print(f"Full traceback: {traceback.format_exc()}")
                        # Only set empty DataFrames if there was an error
                        boards_dfs = {'boards': pl.DataFrame(), 'score_frequency': pl.DataFrame()}
                        print("Set boards_dfs to empty DataFrames due to exception")

            if st.session_state.debug_mode:
                for k, v in boards_dfs.items():
                    debug_capture_df(f"boards_dfs.{k}", v, source="RRN boards scrape")

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
            # todo: looks like board data reportedly cannot be found but actually is available at: https://www.bridgeplus.com/nos-simultanes/resultats/?p=board&res=sim&d=1&tr=S202639
            if boards_df.height == 0:
                error_msg = f"No boards found for tournament {st.session_state.tournament_id}"
                error_msg += f"\n\nThis usually means:"
                error_msg += f"\n  1. The route page (p=route) returned no boards"
                error_msg += f"\n  2. The fallback (trying boards 1-40 directly) also found no boards"
                error_msg += f"\n\nPossible causes:"
                error_msg += f"\n  1. Team didn't play any boards"
                error_msg += f"\n  2. Route page structure changed (expected div.row > div.col-1 a structure)"
                error_msg += f"\n  3. Incorrect team/section/club parameters"
                error_msg += f"\n  4. Board detail pages (p=donne) are not accessible or return errors"
                error_msg += f"\n  5. Route URL: {st.session_state.get('route_url', 'Not set')}"
                error_msg += f"\n\nNote: Board detail pages (p=board) exist but require different parsing."
                error_msg += f"\nExample: https://www.bridgeplus.com/nos-simultanes/resultats/?p=board&res=sim&d=1&tr={st.session_state.tournament_id}"
                raise ValueError(error_msg)
            
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

        if _finalize_mldf_for_report(df):
            return True

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
                            # Keep the resolved license number in the textbox after the report
                            # starts (overrides the clear_player_search flag set by that path).
                            st.session_state.insert_player_search_value = str(license_number)
                            
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
                            
                            # Immediately hide the dialog visually before rerun completes
                            st.markdown(
                                """<script>
                                (function() {
                                    var dialog = parent.document.querySelector('[data-testid="stModal"]');
                                    if (dialog) dialog.style.display = 'none';
                                    var overlay = parent.document.querySelector('[data-testid="stModalOverlay"]');
                                    if (overlay) overlay.style.display = 'none';
                                })();
                                </script>""",
                                unsafe_allow_html=True
                            )
                            
                            # Rerun to apply session state changes
                            st.rerun()
                
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
            
            # Immediately hide the dialog visually before rerun completes
            st.markdown(
                """<script>
                (function() {
                    var dialog = parent.document.querySelector('[data-testid="stModal"]');
                    if (dialog) dialog.style.display = 'none';
                    var overlay = parent.document.querySelector('[data-testid="stModalOverlay"]');
                    if (overlay) overlay.style.display = 'none';
                })();
                </script>""",
                unsafe_allow_html=True
            )
            
            # Rerun to apply state changes
            st.rerun()


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
        dfs = {'search': search_members(query)}
        
        if len(dfs['search']) == 0:
            # Store error message in session state to persist across reruns
            st.session_state.player_search_error = f"No player found matching '{query}'. Please check the name or license number and try again."
            # Clear all stale state to prevent proceeding with old data
            st.session_state.player_id = None
            st.session_state.deferred_start_report = False
            # Clear stale tournament/session IDs that might cause confusing errors
            if hasattr(st.session_state, 'simultane_id'):
                del st.session_state.simultane_id
            if hasattr(st.session_state, 'session_id'):
                del st.session_state.session_id
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
            row = list(dfs['search'].iter_rows(named=True))[0]
            player_id = row['person_id']
        except Exception as e:
            # More informative error if column doesn't exist
            print(f"Error accessing person_id from search results: {e}")
            print(f"Available columns: {dfs['search'].columns}")
            print(f"Search dataframe:\n{dfs['search']}")
            raise Exception(f"Could not extract player_id from search results. Available columns: {dfs['search'].columns}")

        # Resolve the license number and insert it into the sidebar textbox
        # (replaces the name the user typed), even if loading games fails below.
        license_number = (
            row.get('person_license_number', '')
            or row.get('license_number', '')
            or row.get('licenseNumber', '')
        )
        if license_number and str(license_number) != query.strip():
            st.session_state.insert_player_search_value = str(license_number)

        # Try to populate games for this player. Clear any stale error first so the
        # no-games guard below can distinguish a fresh populate-supplied message.
        st.session_state.pop('player_search_error', None)
        if is_lancelot_mode():
            st.session_state.lancelot_player_id = str(player_id)
            if license_number:
                st.session_state.player_license_number = str(license_number)
                st.session_state.player_id = str(license_number)
            else:
                st.session_state.player_id = str(player_id)
        else:
            st.session_state.classic_player_id = str(player_id)
            st.session_state.player_id = str(player_id)
            if license_number:
                st.session_state.player_license_number = str(license_number)
        try:
            has_games = populate_game_urls_for_player(st.session_state.player_id)
        except Exception as e:
            st.session_state.player_search_error = f"Error loading games for player: {str(e)}"
            st.session_state.player_id = None
            return
        
        # Check if player has any games. Keep a more specific message if populate
        # already set one (e.g. the Lancelot logged-in-user-only limitation).
        if not has_games:
            if not st.session_state.get('player_search_error'):
                st.session_state.player_search_error = f"No games found for player '{query}'."
            st.session_state.player_id = None
            return
        
        # Clear any previous error message and matches on successful search
        if hasattr(st.session_state, 'player_search_error'):
            del st.session_state.player_search_error
        if hasattr(st.session_state, 'player_search_matches'):
            del st.session_state.player_search_matches
        
        # Store the original search query for error messages
        st.session_state.last_search_query = query
        
        # Defer report start: first refresh sidebar with games, then start report
        st.session_state.deferred_start_report = True
        return
        
    except Exception as e:
        # Store only the underlying error message (no prefix) for clarity
        st.session_state.player_search_error = str(e)
        # Clear any previous matches
        if hasattr(st.session_state, 'player_search_matches'):
            del st.session_state.player_search_matches
        # Reset player_id and deferred_start_report to prevent stale state issues
        st.session_state.player_id = None
        st.session_state.deferred_start_report = False


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
        st.sidebar.caption(f"Build:{st.session_state.get('app_datetime', '')}")
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
    
    def main(self):
        """Main application entry point - shows Morty messages on first load."""
        # Inject CSS for green Go button FIRST, before any content renders.
        # This prevents the red flash when the button first appears.
        self._inject_sidebar_button_css()
        
        if 'first_time' not in st.session_state:
            st.session_state.first_time = True
            self.initialize_session_state()
        # Always call create_ui so Morty messages show on first load when player_id is None
        self.create_ui()
    
    def _inject_sidebar_button_css(self):
        """Inject CSS to style sidebar primary buttons green."""
        st.markdown(
            """
            <style>
            /* Make primary sidebar buttons (e.g., Go) green.
               Streamlit has used different DOM attributes across versions, so we target a few. */
            [data-testid="stSidebar"] button[kind="primary"],
            [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"],
            [data-testid="stSidebar"] button[data-testid="baseButton-primary"],
            /* Most robust: our Go is a form submit, so style any button inside a sidebar form */
            [data-testid="stSidebar"] form button,
            [data-testid="stSidebar"] [data-testid="stForm"] button,
            [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button {
                background-color: #2e7d32 !important;
                border-color: #2e7d32 !important;
                color: #ffffff !important;
            }
            [data-testid="stSidebar"] button[kind="primary"]:hover,
            [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:hover,
            [data-testid="stSidebar"] button[data-testid="baseButton-primary"]:hover,
            [data-testid="stSidebar"] form button:hover,
            [data-testid="stSidebar"] [data-testid="stForm"] button:hover,
            [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button:hover {
                background-color: #1b5e20 !important;
                border-color: #1b5e20 !important;
                color: #ffffff !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    
    def ask_sql_query(self):
        """Only show SQL query textbox after the report is fully rendered."""
        # Don't show during initial load (no player selected)
        if st.session_state.get('player_id') is None:
            return
        # Don't show if no session/game selected
        if st.session_state.get('session_id') is None:
            return
        # Don't show if report data isn't available yet
        if getattr(st.session_state, 'df', None) is None:
            return
        # Don't show while report is still rendering
        if st.session_state.get('report_rendering', False):
            return
        # Don't show if show_sql_query is disabled
        if not st.session_state.get('show_sql_query', False):
            return
        # All conditions met - show the SQL query input
        self.ask_standard_sql_query()
    
    def initialize_session_state(self):
        """Initialize FFBridge-specific session state."""
        # First initialize common session state
        self.initialize_common_session_state()

        # Persist augmented dataframes to cache/df-{session_id}-{player_id}.parquet
        # (PostmortemBase defaults do_not_cache_df to True). Besides speeding up
        # revisits, the parquet cache is the data source exposed by
        # MortyBridgeBot's FFBridge postmortem tools.
        st.session_state.do_not_cache_df = False

        # Default before URL params are applied; URL ?player_id=... will override below.
        st.session_state.player_id = None

        # Lancelot is the default identifier namespace. Never switch to Classic
        # because of a transient health-probe result; URL ?api_source= and the
        # sidebar remain the explicit ways to select Classic.
        if 'api_source' not in st.session_state:
            st.session_state.api_source = auto_detect_api_source()
            print(f"Default API source: {st.session_state.api_source}")

        cache_dir = 'cache'
        pathlib.Path(cache_dir).mkdir(exist_ok=True, parents=True)
        st.session_state.cache_dir = cache_dir

        # Initialize FFBridge Bearer Token
        initialize_ffbridge_bearer_token()
        
        # Initialize website-specific components
        self.initialize_website_specific()
        self.reset_game_data()

        # Apply URL query params last so they override any defaults set above
        # (e.g. ?player_id=...&session_id=...&debug_mode=1).
        apply_url_params_to_session_state()

        # ?player_id= in shared URLs is usually a license number. Resolve
        # aliases here so generate/filter can use Lancelot/Classic ids, but
        # keep the public player_id as the requested value so the URL is
        # not rewritten from 9500754 to 246273.
        url_pid = st.session_state.get('player_id')
        if url_pid:
            resolved = resolve_url_player_id_param(str(url_pid))
            if resolved != str(url_pid) and not is_lancelot_mode():
                st.session_state.player_id = resolved
        
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
            'lancelot_player_id': None,
            'classic_player_id': None,
            'route_url': st.session_state.route_url_default,
            'game_urls_d': {},
            'person_organization_id': None,
            'nb_deals': None,
            'deferred_start_report': False,  # Flag to defer report generation until after sidebar refresh
            'report_error': None,
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
        if st.session_state.get("report_error"):
            st.error(st.session_state.report_error)
        # If a new player was entered, refresh sidebar first then start report
        if st.session_state.get('deferred_start_report', False):
            # Ensure games are available
            if st.session_state.player_id is not None:
                try:
                    populate_game_urls_for_player(str(st.session_state.player_id))
                    # Start report on first available game
                    game_urls = st.session_state.game_urls_d.get(st.session_state.player_id, {})
                    if len(game_urls) > 0:
                        st.session_state.deferred_start_report = False
                        failed = change_game_state(
                            str(st.session_state.player_id), None
                        )
                        if failed:
                            return
                        st.session_state.sql_query_mode = False
                        # Defer clearing search inputs to before widget creation
                        st.session_state.clear_player_search = True
                        # Immediately rerun so the report renders without requiring Go
                        st.rerun()
                        return
                    else:
                        # No games found for this player. Keep a more specific message if one was
                        # already set (e.g. the Lancelot logged-in-user-only limitation).
                        st.session_state.deferred_start_report = False
                        if not st.session_state.get('player_search_error'):
                            st.session_state.player_search_error = f"No games found for player ID '{st.session_state.player_id}'."
                        st.session_state.player_id = None
                except Exception as e:
                    st.session_state.deferred_start_report = False
                    error_msg = str(e)
                    # Get the original search query for better error messages
                    search_query = st.session_state.get('last_search_query', '')
                    
                    # Make error messages more user-friendly
                    if "No data found" in error_msg or "simultaneous" in error_msg.lower():
                        if search_query:
                            st.session_state.player_search_error = f"No valid game data found for '{search_query}'. The player may not exist or may not have any accessible games."
                        else:
                            st.session_state.player_search_error = "No valid game data found. The player may not exist or may not have any accessible games."
                    else:
                        st.session_state.player_search_error = f"Error loading player data: {error_msg}"
                    st.session_state.player_id = None
                    # Clear stale tournament/session state
                    if hasattr(st.session_state, 'simultane_id'):
                        del st.session_state.simultane_id
                    if hasattr(st.session_state, 'session_id'):
                        del st.session_state.session_id
                    if hasattr(st.session_state, 'last_search_query'):
                        del st.session_state.last_search_query
        # URL-driven auto-load: when ?player_id=X[&session_id=Y] is in the URL on cold load,
        # session_state has the ids but the report data (df) hasn't been fetched yet because
        # the user never clicked the game-selector dropdown. Mirror that dropdown's effect
        # by calling change_game_state once for the URL-specified pair, then rerun so the
        # report renders cleanly. When session_id is omitted or 'latest', change_game_state
        # auto-picks the player's most recent game -- same as the sidebar license path.
        # Tracking key avoids re-loading on every script rerun.
        url_session_key = (
            str(st.session_state.player_id) if st.session_state.player_id is not None else None,
            int(st.session_state.session_id) if st.session_state.session_id is not None else None,
        )
        if (url_session_key[0] is not None
                and getattr(st.session_state, 'df', None) is None
                and st.session_state.get('_url_loaded_session_key') != url_session_key):
            st.session_state._url_loaded_session_key = url_session_key
            try:
                # url_session_key[0] is the public id from the URL (license
                # or Lancelot). Aliases were stashed in initialize_session_state.
                populate_game_urls_for_player(url_session_key[0])
                failed = change_game_state(
                    url_session_key[0], url_session_key[1]
                )
                if failed:
                    return
                st.rerun()
                return
            except Exception as e:
                print(f"URL auto-load failed for {url_session_key}: {e}")
                import traceback
                traceback.print_exc()
                _report_failure(
                    f"Failed to auto-load report for player_id={url_session_key[0]}, "
                    f"session_id={url_session_key[1]}: {e}"
                )
                # Clear session_id so we fall through to the dropdown UX instead of looping.
                st.session_state.session_id = None

        if not st.session_state.sql_query_mode:
            # Show Morty instructions if no player is selected
            if st.session_state.player_id is None:
                # Display intro messages when no player is selected
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
                # Hidden machine-readable metadata for automation tools
                # (e.g. ffbridge_postmortem_generator.py). Emitted BEFORE the (slow)
                # write_report() runs so the CLI can race this tag against
                # any st.error() alert and bail out fast on a bad URL.
                _td = st.session_state.get('tournament_date') or ''
                _pid = st.session_state.get('player_id') or ''
                _sid = st.session_state.get('session_id')
                _sid_str = '' if _sid is None else str(_sid)
                st.markdown(
                    f'<span id="cli-game-meta" '
                    f'data-game-date="{_td}" '
                    f'data-player-id="{_pid}" '
                    f'data-session-id="{_sid_str}" '
                    f'style="display:none;"></span>',
                    unsafe_allow_html=True,
                )
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
        # Always render debug section (when enabled) so it doesn't disappear on reruns.
        render_debug_expander()

        # Memory footer on the main page (same pattern as Elo_Ratings apps).
        st.caption(streamlitlib.get_memory_caption_line(st))
        
        # Re-inject CSS at the end to ensure it overrides any theme styles applied during render.
        self._inject_sidebar_button_css()

        # Mirror current sidebar state into the URL bar so the page is shareable/bookmarkable
        # and reloading preserves the selected player, game, and developer settings.
        sync_session_state_to_url_params()

    def create_sidebar(self):
        """Create FFBridge-specific sidebar."""
        st.sidebar.caption(f"Build:{st.session_state.app_datetime}")
        
        # Process Go button input OUTSIDE sidebar context (so output goes to main window)
        if hasattr(st.session_state, 'process_go_button_input'):
            input_value = st.session_state.process_go_button_input
            del st.session_state.process_go_button_input
            
            # Processing Go button input in main context (so output goes to main window)
            
            # If it's a license number (numeric), look up the player and generate report
            if input_value.isdigit():
                try:
                    # Make API call to find the player by license number
                    dfs = {'search': search_members(input_value)}
                    
                    if len(dfs['search']) == 0:
                        st.error(f"License number '{input_value}' not found.")
                    elif len(dfs['search']) == 1:
                        # Exactly one player found. Keep the license as the
                        # public player_id so the URL stays shareable.
                        row = list(dfs['search'].iter_rows(named=True))[0]
                        lancelot_or_classic_id = row['person_id']
                        license_number = (
                            row.get('person_license_number')
                            or row.get('license_number')
                            or input_value
                        )
                        if is_lancelot_mode():
                            st.session_state.lancelot_player_id = str(lancelot_or_classic_id)
                            st.session_state.player_id = str(license_number or lancelot_or_classic_id)
                            if license_number:
                                st.session_state.player_license_number = str(license_number)
                        else:
                            st.session_state.classic_player_id = str(lancelot_or_classic_id)
                            st.session_state.player_id = str(lancelot_or_classic_id)
                            if license_number:
                                st.session_state.player_license_number = str(license_number)
                        
                        # Populate sidebar first, then defer report start until after sidebar refresh
                        has_games = populate_game_urls_for_player(st.session_state.player_id)
                        
                        if not has_games:
                            # A more specific message (e.g. Lancelot logged-in-user-only) may
                            # already be queued for the sidebar; don't duplicate it here.
                            if not st.session_state.get('player_search_error'):
                                st.error(f"No games found for license number '{input_value}'.")
                            st.session_state.player_id = None
                            return
                        
                        # Store the original search query for error messages
                        st.session_state.last_search_query = input_value
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

        # API source selector. Classic and Lancelot are different FFBridge
        # backends with different id spaces; switching clears player/game state.
        api_health = probe_api_sources()
        st.sidebar.selectbox(
            "API source",
            options=[API_SOURCE_LANCELOT, API_SOURCE_CLASSIC],
            index=[API_SOURCE_LANCELOT, API_SOURCE_CLASSIC].index(get_api_source()),
            format_func=lambda v: API_SOURCE_LABELS[v],
            on_change=api_source_on_change,
            key='api_source_selectbox',
            help="Choose which FFBridge API backend to use. Lancelot (default) powers the current ffbridge.fr website; Classic is the pre-2026 API.",
        )
        health_parts = []
        for source in (API_SOURCE_LANCELOT, API_SOURCE_CLASSIC):
            status = 'up' if api_health[source]['ok'] else f"unreachable ({api_health[source]['detail']})"
            health_parts.append(f"{source.capitalize()}: {status}")
        st.sidebar.caption(' | '.join(health_parts))
        if not api_health[get_api_source()]['ok']:
            st.sidebar.warning(f"The selected API source ({get_api_source()}) is currently unreachable.")

        # Player search with modal dialog
        # Initialize session state for text input if not exists (only use session state, not value= param)
        if 'player_search_input' not in st.session_state:
            st.session_state.player_search_input = ''
        
        # Insert a resolved license number into the textbox before instantiation if flagged
        # (set when a name search resolved to a single player). Takes precedence over the
        # clear flag which the report-start path also sets.
        if st.session_state.get('insert_player_search_value'):
            st.session_state.player_search_input = st.session_state.insert_player_search_value
            st.session_state.player_search_value = st.session_state.insert_player_search_value
            del st.session_state.insert_player_search_value
            st.session_state.pop('clear_player_search', None)
        # Clear the text input value before instantiation if flagged
        elif st.session_state.get('clear_player_search'):
            st.session_state.player_search_input = ''
            st.session_state.player_search_value = ''
            del st.session_state.clear_player_search
            
        # Use a form so Enter and the Go button submit the current textbox value reliably.
        # (Outside a form, Streamlit may not commit text_input into session_state until Enter/blur,
        # which can make the Go button appear "disabled".)
        with st.sidebar.form(key="player_search_form", clear_on_submit=False):
            st.text_input(
            "Enter ffbridge license number",
            key='player_search_input',
            placeholder="Enter license number",
            help="Enter ffbridge license number or (partial) last name."
        )
            submitted = st.form_submit_button("Go", type="primary", use_container_width=True)

        if submitted:
            input_value = (st.session_state.get('player_search_input') or '').strip()
            if input_value.isdigit():
                # Numeric input - store for report generation (handled at top of this function)
                st.session_state.process_go_button_input = input_value
                st.rerun()
            else:
                # Non-numeric input - trigger search/modal
                if input_value:
                    player_search_input_on_change_with_query(input_value)
                    # If a name resolved to a license number, rerun so the textbox
                    # (already instantiated this render) picks up the inserted value.
                    if st.session_state.get('insert_player_search_value'):
                        st.rerun()
        
        # Display any search error in the sidebar
        if hasattr(st.session_state, 'player_search_error'):
            st.sidebar.error(st.session_state.player_search_error)
        
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
        
        # If a player is selected but games haven't been loaded yet, try to populate them
        if (st.session_state.player_id is not None and
            st.session_state.player_id not in st.session_state.game_urls_d):
            try:
                populate_game_urls_for_player(str(st.session_state.player_id))
            except Exception as e:
                print(f"Auto-populate games failed for player {st.session_state.player_id}: {e}")

        self.read_configs()

        player_id_key = str(st.session_state.player_id) if st.session_state.player_id is not None else None
        if player_id_key is not None and player_id_key in st.session_state.game_urls_d:
            st.sidebar.selectbox(
                "Choose a club game.", 
                index=0, 
                options=[f"{k}, {v['description']}" for k, v in st.session_state.game_urls_d[player_id_key].items()], 
                on_change=club_session_id_on_change, 
                key='club_session_ids_selectbox'
            )
            # Show a small verification of how many games are available
            st.sidebar.caption(f"Games found: {len(st.session_state.game_urls_d.get(player_id_key, {}))}")

        results_url = _ensure_game_results_url()
        if results_url:
            st.sidebar.markdown(f"[FFBridge Result Page]({results_url})")
        if st.session_state.get('route_url'):
            st.sidebar.markdown(f"[Roy René Result Page]({st.session_state.route_url})")
        # Download Personalized Report PDF button placeholder (below the link button)
        st.session_state.pdf_link = st.sidebar.empty()

        # Automated Postmortem Apps
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Automated Postmortem Apps**")
        st.sidebar.markdown("🔗 [ACBL Postmortem](https://acbl.postmortem.chat)")
        st.sidebar.markdown("🔗 [French ffbridge Postmortem](https://ffbridge.postmortem.chat)")
        st.sidebar.markdown("🔗 [Calculate PBN](https://pbn.postmortem.chat)")
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
    """Load configured tokens without making startup depend on FFBridge login.

    Numeric player resolution and session listing use the shared index. Name
    search authenticates lazily through ``_validated_lancelot_token``; missing
    board downloads authenticate in the writer process.
    """
    if st.session_state.get('_lancelot_auth_initialized'):
        return
    st.session_state._lancelot_auth_initialized = True
    st.session_state.lancelot_token_valid = False
    st.session_state.logged_in_player_id = None
    st.session_state.logged_in_license_number = None
    st.session_state.logged_in_lancelot_id = None

    load_dotenv()
    st.session_state.ffbridge_bearer_token = os.getenv('FFBRIDGE_BEARER_TOKEN_LANCELOT')
    st.session_state.ffbridge_easi_token = os.getenv('FFBRIDGE_EASI_TOKEN')


if __name__ == "__main__":
    main()
