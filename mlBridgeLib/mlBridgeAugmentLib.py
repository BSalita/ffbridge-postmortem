# contains functions to augment df with additional columns
# mostly polars functions

# todo:
# don't automatically report on the most recent game. If the game is errors, it's inconvenient to selecting others.
# optimize _create_dd_columns() and calculate_final_scores()
# since some columns can be derived from other columns, we should assert that input df has at least one column in each group of mutually derivable columns.
# assert that column names don't exist in df.columns for all column creation functions.
# refactor when should *_Dcl columns be created? At end of each func, class, class of it's own?
# if a column already exists, print a message and skip creation.
# if a column already exists, generate a new column and assert that the new column is the same as the existing column.
# print column names and dtypes for all columns generated and skipped.
# print a list of mandatory columns that must be present in df.columns. many columns can be derived from other columns e.g. scoring columns.
# for each augmentation function, validate that all columns are properly generated.
# create a function which validates that all needed for the class can be derived from the input df.
# create a function which validates that all columns for the class are generated.
# create a function which validate columns for every column generated.
# use .pipe() to chain other lambdas as is done in _create_prob_taking_columns()?
# should *_Declarer be the pattern or omit _Declarer where declarer would be implied?
# change *_Declarer to *_Dcl?
# looks like Friday simultaneous doesn't have Contracts? 8-Aug-2025 did not. Is it posted with a 7+ day delay?
# create a function which creates a column of the weighted moving average of (double dummy tricks for declarer's contract minus tricks taken) per start of session. Each row of the session has the same start of session value.
# Rename to title case for column names. e.g. session_id to Session_ID

import polars as pl
import numpy as np
import warnings
from collections import defaultdict
from typing import Optional, Union, Callable, Type, Dict, List, Tuple, Any
import time

import endplay # for __version__
from endplay.parsers import pbn, lin, json
from endplay.types import Deal, Contract, Denom, Player, Penalty, Vul
from endplay.dds import calc_dd_table, calc_all_tables, par
from endplay.dealer import generate_deals

import mlBridgeLib.mlBridgeLib as mlBridgeLib
from mlBridgeLib.mlBridgeLib import (
    NESW, SHDC, NS_EW,
    PlayerDirectionToPairDirection,
    NextPosition,
    PairDirectionToOpponentPairDirection,
    score
)


# todo: use versions in mlBridgeLib
VulToEndplayVul_d = { # convert mlBridgeLib Vul to endplay Vul
    'None':Vul.none,
    'Both':Vul.both,
    'N_S':Vul.ns,
    'E_W':Vul.ew
}


DealerToEndPlayDealer_d = { # convert mlBridgeLib dealer to endplay dealer
    'N':Player.north,
    'E':Player.east,
    'S':Player.south,
    'W':Player.west
}

declarer_to_LHO_d = {
    None:None,
    'N':'E',
    'E':'S',
    'S':'W',
    'W':'N'
}


declarer_to_dummy_d = {
    None:None,
    'N':'S',
    'E':'W',
    'S':'N',
    'W':'E'
}


declarer_to_RHO_d = {
    None:None,
    'N':'W',
    'E':'N',
    'S':'E',
    'W':'S'
}


def create_hand_nesw_columns(df: pl.DataFrame) -> pl.DataFrame:    
    # create 'Hand_[NESW]' columns of type pl.String from 'PBN'
    if 'Hand_N' not in df.columns:
        for i, direction in enumerate('NESW'):
            df = df.with_columns([
                pl.col('PBN')   
              .str.slice(2)
              .str.split(' ')
              .list.get(i)
              .alias(f'Hand_{direction}')
    ])
    return df


def create_hands_lists_column(df: pl.DataFrame) -> pl.DataFrame:
    # create 'Hands' column of type pl.List(pl.List(pl.String)) from 'PBN'
    if 'Hands' not in df.columns:
        df = df.with_columns([  
            pl.col('PBN')
           .str.slice(2)
           .str.split(' ')
           .list.eval(pl.element().str.split('.'), parallel=True)
           .alias('Hands')
        ])
    return df


def create_suit_nesw_columns(df: pl.DataFrame) -> pl.DataFrame:
    # Create 'Suit_[NESW]_[SHDC]' columns of type pl.String
    if 'Suit_N_C' not in df.columns:
        for d in 'NESW':
            for i, s in enumerate('SHDC'):
                df = df.with_columns([
                    pl.col(f'Hand_{d}')
                    .str.split('.')
                   .list.get(i)
                   .alias(f'Suit_{d}_{s}')
          ])
    return df


# One Hot Encoded into binary string
def OHE_Hands(hands_bin: List[List[Tuple[Optional[str], Optional[str]]]]) -> defaultdict[str, List[Any]]:
    handsbind = defaultdict(list)
    for h in hands_bin:
        for direction,nesw in zip(NESW,h):
            assert nesw[0] is not None and nesw[1] is not None
            handsbind['_'.join(['HB',direction])].append(nesw[0])
    return handsbind


# generic function to augment metrics by suits
def Augment_Metric_By_Suits(metrics: pl.DataFrame, metric: str, dtype: pl.DataType = pl.UInt8) -> pl.DataFrame:
    """Optimized version using vectorized operations instead of map_elements."""
    # Create direction-specific columns using list access
    for d, direction in enumerate(NESW):
        # Extract direction-specific values using list indexing
        direction_expr = pl.col(metric).list.get(1).list.get(d).list.get(0).cast(dtype).alias('_'.join([metric, direction]))
        
        # Create suit-specific columns
        suit_exprs = []
        for s, suit in enumerate(SHDC):
            suit_expr = pl.col(metric).list.get(1).list.get(d).list.get(1).list.get(s).cast(dtype).alias('_'.join([metric, direction, suit]))
            suit_exprs.append(suit_expr)
        
        metrics = metrics.with_columns([direction_expr] + suit_exprs)
    
    # Create pair direction columns by summing individual directions
    for direction in NS_EW:
        pair_expr = (
            pl.col('_'.join([metric, direction[0]])) + 
            pl.col('_'.join([metric, direction[1]]))
        ).cast(dtype).alias('_'.join([metric, direction]))
        
        # Create pair suit columns
        pair_suit_exprs = []
        for s, suit in enumerate(SHDC):
            pair_suit_expr = (
                pl.col('_'.join([metric, direction[0], suit])) + 
                pl.col('_'.join([metric, direction[1], suit]))
            ).cast(dtype).alias('_'.join([metric, direction, suit]))
            pair_suit_exprs.append(pair_suit_expr)
        
        metrics = metrics.with_columns([pair_expr] + pair_suit_exprs)
    
    return metrics


def update_hrs_cache_df(hrs_cache_df: pl.DataFrame, new_df: pl.DataFrame) -> pl.DataFrame:
    # Print initial row counts
    print(f"Initial hrs_cache_df rows: {hrs_cache_df.height}")
    print(f"New data rows: {new_df.height}")
    
    # Early return if no new data to process
    if new_df.is_empty():
        print("No new data to process, returning original cache")
        return hrs_cache_df
    
    # Calculate which PBNs will be updated vs added
    existing_pbns = set(hrs_cache_df['PBN'].to_list())
    new_pbns = set(new_df['PBN'].to_list())
    
    pbns_to_update = existing_pbns & new_pbns  # intersection
    pbns_to_add = new_pbns - existing_pbns      # difference
    
    print(f"PBNs to update (existing): {len(pbns_to_update)}")
    print(f"PBNs to add (new): {len(pbns_to_add)}")
    
    # Check for duplicate PBNs in new_df with different Dealer/Vul combinations
    new_df_pbn_counts = new_df['PBN'].value_counts()
    duplicate_pbns_in_new = new_df_pbn_counts.filter(pl.col('count') > 1)
    if duplicate_pbns_in_new.height > 0:
        print(f"Note: {duplicate_pbns_in_new.height} PBNs in new data have multiple Dealer/Vul combinations")
        print(f"      This will result in more rows than unique PBNs")
    
    # Calculate expected final row count more accurately
    # The update operation replaces existing rows with matching PBNs
    # Then we add all rows from new_df that don't exist in hrs_cache_df
    expected_final_rows = hrs_cache_df.height - len(pbns_to_update) + new_df.height
    print(f"Expected final rows: {expected_final_rows} (existing: {hrs_cache_df.height} - updated: {len(pbns_to_update)} + new: {new_df.height})")

    # check for differing dtypes
    common_cols = set(hrs_cache_df.columns) & set(new_df.columns)
    dtype_diffs = {
        col: (hrs_cache_df[col].dtype, new_df[col].dtype)
        for col in common_cols
        if hrs_cache_df[col].dtype != new_df[col].dtype and new_df[col].dtype != pl.Null
    }
    assert len(dtype_diffs) == 0, f"Differing dtypes: {dtype_diffs}"

    # Update existing rows (only columns from new_df)
    hrs_cache_df = hrs_cache_df.update(new_df, on='PBN')
    
    # Add missing columns to new_df ONLY for new rows
    missing_columns = set(hrs_cache_df.columns) - set(new_df.columns)
    new_rows = new_df.join(hrs_cache_df.select('PBN'), on='PBN', how='anti')
    if new_rows.height > 0:
        new_rows = new_rows.with_columns([
            pl.lit(None).alias(col) for col in missing_columns
        ])
        hrs_cache_df = pl.concat([hrs_cache_df, new_rows.select(hrs_cache_df.columns)])
        print(f"Added {len(missing_columns)} missing columns to {new_rows.height} new rows")
    
    print(f"Final hrs_cache_df rows: {hrs_cache_df.height}")
    print(f"Operation completed successfully")
    
    return hrs_cache_df


# calculate dict of contract result scores. each column contains (non-vul,vul) scores for each trick taken. sets are always penalty doubled.
def calculate_scores() -> Tuple[Dict[Tuple, int], Dict[Tuple, int], pl.DataFrame]:

    scores_d = {}
    all_scores_d = {(None,None,None,None,None):0} # PASS

    strain_to_denom = [Denom.clubs, Denom.diamonds, Denom.hearts, Denom.spades, Denom.nt]
    for strain_char in 'SHDCN':
        strain_index = 'CDHSN'.index(strain_char) # [3,2,1,0,4]
        denom = strain_to_denom[strain_index]
        for level in range(1,8): # contract level
            for tricks in range(14):
                result = tricks-6-level
                # sets are always penalty doubled
                scores_d[(level,strain_char,tricks,False)] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.passed if result>=0 else Penalty.doubled,result=result).score(Vul.none)
                scores_d[(level,strain_char,tricks,True)] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.passed if result>=0 else Penalty.doubled,result=result).score(Vul.both)
                # calculate all possible scores
                all_scores_d[(level,strain_char,tricks,False,'')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.passed,result=result).score(Vul.none)
                all_scores_d[(level,strain_char,tricks,False,'X')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.doubled,result=result).score(Vul.none)
                all_scores_d[(level,strain_char,tricks,False,'XX')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.redoubled,result=result).score(Vul.none)
                all_scores_d[(level,strain_char,tricks,True,'')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.passed,result=result).score(Vul.both)
                all_scores_d[(level,strain_char,tricks,True,'X')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.doubled,result=result).score(Vul.both)
                all_scores_d[(level,strain_char,tricks,True,'XX')] = Contract(level=level,denom=denom,declarer=Player.north,penalty=Penalty.redoubled,result=result).score(Vul.both)

    # create score dataframe from dict
    sd = defaultdict(list)
    for suit in 'SHDCN':
        for level in range(1,8):
            for i in range(14):
                sd['_'.join(['Score',str(level)+suit])].append([scores_d[(level,suit,i,False)],scores_d[(level,suit,i,True)]])
    scores_df = pl.DataFrame(sd,orient='row')
    return all_scores_d, scores_d, scores_df


# Global cache for scores calculation
_scores_cache = None

def calculate_scores_cached() -> Tuple[Dict[Tuple, int], Dict[Tuple, int], pl.DataFrame]:
    """Cached version of calculate_scores to avoid recalculating the same scores multiple times."""
    global _scores_cache
    if _scores_cache is None:
        _scores_cache = calculate_scores()
    return _scores_cache


def display_double_dummy_deals(deals: List[Deal], dd_result_tables: List[Any], deal_index: int = 0, max_display: int = 4) -> None:
    # Display a few hands and double dummy tables
    for dd, rt in zip(deals[deal_index:deal_index+max_display], dd_result_tables[deal_index:deal_index+max_display]):
        deal_index += 1
        print(f"Deal: {deal_index}")
        print(dd)
        rt.pprint()


# todo: could save a couple seconds by creating dict of deals
def calc_double_dummy_deals(deals: List[Deal], batch_size: int = 40, output_progress: bool = False, progress: Optional[Any] = None) -> List[Any]:
    # was the wonkyness due to unique() not having maintain_order=True? Let's see if it behaves now.
    all_result_tables = []
    for i,b in enumerate(range(0,len(deals),batch_size)):
        if output_progress:
            if i % 100 == 0: # only show progress every 100 batches
                percent_complete = int(b*100/len(deals))
                if progress:
                    if hasattr(progress, 'progress'): # streamlit
                        progress.progress(percent_complete, f"{percent_complete}%: Double dummies calculated for {b} of {len(deals)} unique deals.")
                    elif hasattr(progress, 'set_description'): # tqdm
                        progress.set_description(f"{percent_complete}%: Double dummies calculated for {b} of {len(deals)} unique deals.")
                else:
                    print(f"{percent_complete}%: Double dummies calculated for {b} of {len(deals)} unique deals.")
        result_tables = calc_all_tables(deals[b:b+batch_size])
        all_result_tables.extend(result_tables)
    if output_progress: 
        if progress:
            if hasattr(progress, 'progress'): # streamlit
                progress.progress(100, f"100%: Double dummies calculated for {len(deals)} unique deals.")
                progress.empty() # hmmm, this removes the progress bar so fast that 100% message won't be seen.
            elif hasattr(progress, 'set_description'): # tqdm
                progress.set_description(f"100%: Double dummies calculated for {len(deals)} unique deals.")
        else:
            print(f"100%: Double dummies calculated for {len(deals)} unique deals.")
    return all_result_tables


# takes 10000/hour
def calculate_dd_scores(hrs_df: pl.DataFrame, hrs_cache_df: pl.DataFrame, max_adds: Optional[int] = None, output_progress: bool = True, progress: Optional[Any] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:

    # Calculate double dummy scores only
    print(f"{hrs_df.height=}")
    print(f"{hrs_cache_df.height=}")
    assert hrs_df['PBN'].null_count() == 0, "PBNs in df must be non-null"
    assert hrs_df.filter(pl.col('PBN').str.len_chars().ne(69)).height == 0, hrs_df.filter(pl.col('PBN').str.len_chars().ne(69))
    assert hrs_cache_df['PBN'].null_count() == 0, "PBNs in hrs_cache_df must be non-null"
    assert hrs_cache_df.filter(pl.col('PBN').str.len_chars().ne(69)).height == 0, hrs_cache_df.filter(pl.col('PBN').str.len_chars().ne(69))
    unique_hrs_df_pbns = set(hrs_df['PBN']) # could be non-unique PBN with difference Dealer, Vul.
    print(f"{len(unique_hrs_df_pbns)=}")
    hrs_cache_with_nulls_df = hrs_cache_df.filter(pl.col('DD_N_C').is_null())
    print(f"{len(hrs_cache_with_nulls_df)=}")
    hrs_cache_with_nulls_pbns = hrs_cache_with_nulls_df['PBN']
    print(f"{len(hrs_cache_with_nulls_pbns)=}")
    unique_hrs_cache_with_nulls_pbns = set(hrs_cache_with_nulls_pbns)
    print(f"{len(unique_hrs_cache_with_nulls_pbns)=}")
    
    hrs_cache_all_pbns = set(hrs_cache_df['PBN'])
    pbns_to_add = set(unique_hrs_df_pbns) - hrs_cache_all_pbns  # In hrs_df but NOT in hrs_cache_df
    print(f"{len(pbns_to_add)=}")
    pbns_to_replace = set(unique_hrs_df_pbns).intersection(set(unique_hrs_cache_with_nulls_pbns))  # In both, with nulls in hrs_cache_df
    print(f"{len(pbns_to_replace)=}")
    pbns_to_process = pbns_to_add.union(pbns_to_replace)
    print(f"{len(pbns_to_process)=}")
    
    if max_adds is not None:
        pbns_to_process = list(pbns_to_process)[:max_adds]
        print(f"limit: {max_adds=} {len(pbns_to_process)=}")
    
    cleaned_pbns = [Deal(pbn) for pbn in pbns_to_process]
    assert all([pbn == dpbn.to_pbn() for pbn,dpbn in zip(pbns_to_process,cleaned_pbns)]), [(pbn,dpbn.to_pbn()) for pbn,dpbn in zip(pbns_to_process,cleaned_pbns) if pbn != dpbn.to_pbn()] # usually a sort order issue which should have been fixed in previous step
    unique_dd_tables = calc_double_dummy_deals(cleaned_pbns, output_progress=output_progress, progress=progress)
    print(f"{len(unique_dd_tables)=}")
    unique_dd_tables_d = {deal.to_pbn():rt for deal,rt in zip(cleaned_pbns,unique_dd_tables)}
    print(f"{len(unique_dd_tables_d)=}")

    # Create dataframe of double dummy scores only
    d = defaultdict(list)
    dd_columns = {f'DD_{direction}_{suit}':pl.UInt8 for suit in 'SHDCN' for direction in 'NESW'}

    # Process each PBN that needs DD calculation
    for pbn in pbns_to_process:
        if pbn not in unique_dd_tables_d:
            continue
        dd_rows = sum(unique_dd_tables_d[pbn].to_list(), []) # flatten dd_table
        d['PBN'].append(pbn)
        for col,dd in zip(dd_columns, dd_rows):
            d[col].append(dd)

    # Create a DataFrame using only the keys in dictionary d while maintaining the schema from hrs_cache_df
    filtered_schema = {k: hrs_cache_df[k].dtype for k, v in hrs_cache_df.schema.items() if k in d}
    print(f"filtered_schema: {filtered_schema}")
    error_filtered_schema = {k: None for k, v in d.items() if k not in hrs_cache_df.columns}
    print(f"error_filtered_schema: {error_filtered_schema}")
    assert len(error_filtered_schema) == 0, f"error_filtered_schema: {error_filtered_schema}"
    dd_df = pl.DataFrame(d, schema=filtered_schema)
    return dd_df, unique_dd_tables_d


def calculate_par_scores(hrs_df: pl.DataFrame, hrs_cache_df: pl.DataFrame, unique_dd_tables_d: Dict[str, Any]) -> pl.DataFrame:

    # Calculate par scores using existing double dummy tables
    print(f"{hrs_df.height=}")
    print(f"{hrs_cache_df.height=}")
    print(f"Calculating par scores for {len(unique_dd_tables_d)} deals")
    
    # Assert that there are no None values in 'PBN' in hrs_cache_df
    assert hrs_cache_df['PBN'].null_count() == 0, f"Found {hrs_cache_df['PBN'].null_count()} None values in 'PBN' column of hrs_cache_df"
    
    # untested but interesting
    # # If no new DD tables were calculated, build DD tables from existing cache data
    # if len(unique_dd_tables_d) == 0:
    #     print("No new DD tables provided, building from existing cache data")
    #     # Get PBNs that have DD data but missing ParScore
    #     pbns_with_dd = hrs_cache_df.filter(
    #         pl.col('DD_N_C').is_not_null() & 
    #         pl.col('ParScore').is_null()
    #     )['PBN'].unique().to_list()
        
    #     if len(pbns_with_dd) > 0:
    #         print(f"Found {len(pbns_with_dd)} PBNs with DD data but missing ParScore")
    #         # Build DD tables from existing cache data
    #         from endplay import Deal
    #         unique_dd_tables_d = {}
    #         for pbn in pbns_with_dd:
    #             # Get DD values from cache for this PBN
    #             dd_df = hrs_cache_df.filter(pl.col('PBN') == pbn).select([
    #                 'DD_N_S', 'DD_N_H', 'DD_N_D', 'DD_N_C', 'DD_N_N',
    #                 'DD_E_S', 'DD_E_H', 'DD_E_D', 'DD_E_C', 'DD_E_N',
    #                 'DD_S_S', 'DD_S_H', 'DD_S_D', 'DD_S_C', 'DD_S_N',
    #                 'DD_W_S', 'DD_W_H', 'DD_W_D', 'DD_W_C', 'DD_W_N'
    #             ])
                
    #             if dd_df.height > 0:
    #                 dd_row = dd_df.row(0)
                    
    #                 # Convert to DD table format (4x5 matrix: NESW x SHDCN)
    #                 dd_table = []
    #                 for direction in ['N', 'E', 'S', 'W']:
    #                     row = []
    #                     for suit in ['S', 'H', 'D', 'C', 'N']:
    #                         col_name = f'DD_{direction}_{suit}'
    #                         # Find the index of this column in the selected columns
    #                         col_idx = dd_df.columns.index(col_name)
    #                         row.append(dd_row[col_idx])
    #                     dd_table.append(row)
                    
    #                 unique_dd_tables_d[pbn] = dd_table
    #         print(f"Built DD tables for {len(unique_dd_tables_d)} PBNs from cache")
    #     else:
    #         print("No PBNs found with DD data but missing ParScore")
    
    # Find PBNs that need par score calculation (only those that have DD calculations)
    unique_hrs_df_pbns = set(hrs_df['PBN'])
    print(f"{len(unique_hrs_df_pbns)=}")
    hrs_cache_with_nulls_df = hrs_cache_df.filter(pl.col('ParScore').is_null())
    print(f"{len(hrs_cache_with_nulls_df)=}")
    hrs_cache_with_nulls_pbns = set(hrs_cache_with_nulls_df['PBN'])
    print(f"{len(hrs_cache_with_nulls_pbns)=}")
    
    hrs_cache_all_pbns = set(hrs_cache_df['PBN'])
    pbns_to_add = set(unique_hrs_df_pbns) - hrs_cache_all_pbns
    print(f"{len(pbns_to_add)=}")
    pbns_to_replace = set(unique_hrs_df_pbns).intersection(hrs_cache_with_nulls_pbns)
    print(f"{len(pbns_to_replace)=}")
    pbns_to_process = pbns_to_add.union(pbns_to_replace)
    print(f"{len(pbns_to_process)=}")
    
    # Check which PBNs missing ParScore have DD tables available
    pbns_missing_par = set(hrs_cache_with_nulls_pbns)  # Use the already calculated set
    pbns_with_dd_tables = set(unique_dd_tables_d.keys())
    pbns_missing_par_and_dd = pbns_missing_par - pbns_with_dd_tables
    pbns_missing_par_with_dd = pbns_missing_par & pbns_with_dd_tables
    
    print(f"PBNs missing ParScore: {len(pbns_missing_par)}")
    print(f"PBNs with DD tables available: {len(pbns_with_dd_tables)}")
    print(f"PBNs missing ParScore with DD tables: {len(pbns_missing_par_with_dd)}")
    print(f"PBNs missing ParScore without DD tables: {len(pbns_missing_par_and_dd)}")
    
    # Only process PBNs that have DD tables available
    pbns_to_process = pbns_to_process & pbns_with_dd_tables
    print(f"PBNs to process (with DD tables): {len(pbns_to_process)}")
    
    # Get Dealer/Vul from appropriate source for each PBN type
    source_rows = []
    if pbns_to_add:
        source_rows.extend(hrs_df.filter(pl.col('PBN').is_in(list(pbns_to_add)))[['PBN','Dealer','Vul']].unique().rows())
    if pbns_to_replace:
        # Get rows from hrs_cache_df, but for None values in Dealer/Vul, get them from hrs_df
        cache_rows = hrs_cache_df.filter(pl.col('PBN').is_in(list(pbns_to_replace)))[['PBN','Dealer','Vul']].unique()
        
        # Find rows with None values in Dealer or Vul
        rows_with_none = cache_rows.filter(pl.col('Dealer').is_null() | pl.col('Vul').is_null())
        rows_without_none = cache_rows.filter(pl.col('Dealer').is_not_null() & pl.col('Vul').is_not_null())
        
        # Get Dealer/Vul values from hrs_df for rows with None values
        if rows_with_none.height > 0:
            print(f"Found {rows_with_none.height} rows with None Dealer/Vul values, getting from hrs_df")
            pbn_list = rows_with_none['PBN'].to_list()
            hrs_df_rows = hrs_df.filter(pl.col('PBN').is_in(pbn_list))[['PBN','Dealer','Vul']].unique()
            source_rows.extend(hrs_df_rows.rows())
        
        # Add rows that already have valid Dealer/Vul values
        source_rows.extend(rows_without_none.rows())
    
    # Create dataframe of par scores
    d = defaultdict(list)
    for pbn, dealer, vul in source_rows:
        if pbn not in unique_dd_tables_d:
            continue
        d['PBN'].append(pbn)
        d['Dealer'].append(dealer)
        d['Vul'].append(vul)
        parlist = par(unique_dd_tables_d[pbn], VulToEndplayVul_d[vul], DealerToEndPlayDealer_d[dealer])
        d['ParScore'].append(parlist.score)
        d['ParNumber'].append(parlist._data.number)
        contracts = [{
            "Level": str(contract.level),
            "Strain": 'SHDCN'[int(contract.denom)],
            "Doubled": contract.penalty.abbr,
            "Pair_Direction": 'NS' if contract.declarer.abbr in 'NS' else 'EW',
            "Result": contract.result
        } for contract in parlist]
        d['ParContracts'].append(contracts)

    # Create a DataFrame using only the keys in dictionary d while maintaining the schema from hrs_cache_df
    filtered_schema = {k: hrs_cache_df[k].dtype for k, v in hrs_cache_df.schema.items() if k in d}
    print(f"filtered_schema: {filtered_schema}")
    error_filtered_schema = {k: None for k, v in d.items() if k not in hrs_cache_df.columns}
    print(f"error_filtered_schema: {error_filtered_schema}")
    assert len(error_filtered_schema) == 0, f"error_filtered_schema: {error_filtered_schema}"
    par_df = pl.DataFrame(d, schema=filtered_schema)
    print(f"par_df height: {par_df.height}")
    return par_df


def constraints(deal: Deal) -> bool:
    return True


def generate_single_dummy_deals(predeal_string: str, produce: int, env: Dict[str, Any] = dict(), max_attempts: int = 1000000, seed: int = 42, show_progress: bool = True, strict: bool = True, swapping: int = 0) -> Tuple[Tuple[Deal, ...], List[Any]]:
    
    predeal = Deal(predeal_string)

    deals_t = generate_deals(
        constraints,
        predeal=predeal,
        swapping=swapping,
        show_progress=show_progress,
        produce=produce,
        seed=seed,
        max_attempts=max_attempts,
        env=env,
        strict=strict
        )

    deals = tuple(deals_t) # create a tuple before interop memory goes wonky
    
    return deals, calc_double_dummy_deals(deals)


def calculate_single_dummy_probabilities(deal: str, produce: int = 100) -> Tuple[Dict[str, pl.DataFrame], Tuple[int, Dict[Tuple[str, str, str], List[float]]]]:

    # todo: has this been obsoleted by endplay's calc_all_tables 2nd parameter?
    ns_ew_rows = {}
    SD_Tricks_df = {}
    for ns_ew in ['NS','EW']:
        s = deal[2:].split()
        if ns_ew == 'NS':
            s[1] = '...'
            s[3] = '...'
        else:
            s[0] = '...'
            s[2] = '...'
        predeal_string = 'N:'+' '.join(s)
        #print(f"predeal:{predeal_string}")

        sd_deals, sd_dd_result_tables = generate_single_dummy_deals(predeal_string, produce, show_progress=False)

        #display_double_dummy_deals(sd_deals, sd_dd_result_tables, 0, 4)
        SD_Tricks_df[ns_ew] = pl.DataFrame([[sddeal.to_pbn()]+[s for d in t.to_list() for s in d] for sddeal,t in zip(sd_deals,sd_dd_result_tables)],schema={'SD_Deal':pl.String}|{'_'.join(['SD_Tricks',d,s]):pl.UInt8 for s in 'SHDCN' for d in 'NESW'},orient='row')

        for d in 'NESW':
            for s in 'SHDCN':
                # always create 14 rows (0-13 tricks taken) for combo of direction and suit. fill never-happened with proper index and 0.0 prob value.
                #ns_ew_rows[(ns_ew,d,s)] = dd_df[d+s].to_pandas().value_counts(normalize=True).reindex(range(14), fill_value=0).tolist() # ['Fixed_Direction','Declarer_Direction','Suit']+['SD_Prob_Take_'+str(n) for n in range(14)]
                vc = {ds:p for ds,p in SD_Tricks_df[ns_ew]['_'.join(['SD_Tricks',d,s])].value_counts(normalize=True).rows()}
                index = {i:0.0 for i in range(14)} # fill values for missing probs
                ns_ew_rows[(ns_ew,d,s)] = list((index|vc).values())

    return SD_Tricks_df, (produce, ns_ew_rows)


# def append_single_dummy_results(pbns,sd_cache_d,produce=100):
#     for pbn in pbns:
#         if pbn not in sd_cache_d:
#             sd_cache_d[pbn] = calculate_single_dummy_probabilities(pbn, produce) # all combinations of declarer pair directI. ion, declarer direciton, suit, tricks taken
#     return sd_cache_d


# performs at 10000/hr
def calculate_sd_probs(df: pl.DataFrame, hrs_cache_df: pl.DataFrame, sd_productions: int = 100, max_adds=None, progress: Optional[Any] = None) -> Tuple[Dict[str, pl.DataFrame], pl.DataFrame]:

    # calculate single dummy probabilities. if already calculated use cache value else update e with new result.
    sd_d = {}
    sd_dfs_d = {}
    assert hrs_cache_df.height == hrs_cache_df.unique(subset=['PBN', 'Dealer', 'Vul']).height, "PBN+Dealer+Vul combinations in hrs_cache_df must be unique"
    
    # Calculate which PBNs to add vs replace
    # Note: Single dummy calculations are the same for a given PBN regardless of Dealer/Vul
    # So we work at the PBN level, but need to handle multiple Dealer/Vul combinations properly
    
    # Find PBNs to add (in df but not in hrs_cache_df)
    df_pbns = set(df['PBN'].to_list())
    hrs_cache_pbns = set(hrs_cache_df['PBN'].to_list())
    pbns_to_add = df_pbns - hrs_cache_pbns
    print(f"PBNs to add: {len(pbns_to_add)}")
    
    # Find PBNs to replace (in both, but with null Probs_Trials in hrs_cache_df)
    # todo: this step takes 3m. find a faster way. perhaps using join?
    pbns_to_replace = set(hrs_cache_df.filter(
        pl.col('PBN').is_in(df['PBN'].to_list()) & 
        pl.col('Probs_Trials').is_null()
    )['PBN'].to_list())
    print(f"PBNs to replace: {len(pbns_to_replace)}")
    
    # Combine all PBNs that need processing
    pbns_to_process = list(pbns_to_add.union(pbns_to_replace))
    print(f"Total unique PBNs to process: {len(pbns_to_process)}")
    if max_adds is not None:
        pbns_to_process = list(pbns_to_process)[:max_adds]
        print(f"limit: {max_adds=} {len(pbns_to_process)=}")
    cleaned_pbns = [Deal(pbn) for pbn in pbns_to_process]
    assert all([pbn == dpbn.to_pbn() for pbn,dpbn in zip(pbns_to_process,cleaned_pbns)]), [(pbn,dpbn.to_pbn()) for pbn,dpbn in zip(pbns_to_process,cleaned_pbns) if pbn != dpbn.to_pbn()] # usually a sort order issue which should have been fixed in previous step
    print(f"processing time assuming 10000/hour:{len(pbns_to_process)/10000} hours")
    for i,pbn in enumerate(pbns_to_process):
        if progress:
            percent_complete = int(i*100/len(pbns_to_process))
            if hasattr(progress, 'progress'): # streamlit
                progress.progress(percent_complete, f"{percent_complete}%: Single dummies calculated for {i} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal. This step takes 30 seconds...")
            elif hasattr(progress, 'set_description'): # tqdm
                progress.set_description(f"{percent_complete}%: Single dummies calculated for {i} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal. This step takes 30 seconds...")
        else:
            if i < 10 or i % 10000 == 0:
                percent_complete = int(i*100/len(pbns_to_process))
                print(f"{percent_complete}%: Single dummies calculated for {i} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal.")
        if not progress and (i < 10 or i % 10000 == 0):
            t = time.time()
        sd_dfs_d[pbn], sd_d[pbn] = calculate_single_dummy_probabilities(pbn, sd_productions) # all combinations of declarer pair direction, declarer direciton, suit, tricks taken
        if not progress and (i < 10 or i % 10000 == 0):
            print(f"calculate_single_dummy_probabilities: time:{time.time()-t} seconds")
        #error
    if progress:
        if hasattr(progress, 'progress'): # streamlit
            progress.progress(100, f"100%: Single dummies calculated for {len(pbns_to_process)} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal.")
        elif hasattr(progress, 'set_description'): # tqdm
            progress.set_description(f"100%: Single dummies calculated for {len(pbns_to_process)} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal.")
    else:
        print(f"100%: Single dummies calculated for {len(pbns_to_process)} of {len(pbns_to_process)} unique deals using {sd_productions} samples per deal.")

    # create single dummy trick taking probability distribution columns
    sd_probs_d = defaultdict(list)
    for pbn, v in sd_d.items():
        productions, probs_d = v
        sd_probs_d['PBN'].append(pbn)
        sd_probs_d['Probs_Trials'].append(productions)
        for (pair_direction,declarer_direction,suit),probs in probs_d.items():
            #print(pair_direction,declarer_direction,suit)
            for i,t in enumerate(probs):
                sd_probs_d['_'.join(['Probs',pair_direction,declarer_direction,suit,str(i)])].append(t)
    # st.write(sd_probs_d)
    sd_probs_df = pl.DataFrame(sd_probs_d,orient='row')

    # update sd_df with sd_probs_df # doesn't work because sd_df isn't updated unless returned.
    # if sd_df.is_empty():
    #     sd_df = sd_probs_df
    # else:
    #     assert set(sd_df['PBN']).isdisjoint(set(sd_probs_df['PBN']))
    #     assert set(sd_df.columns) == (set(sd_probs_df.columns))
    #     sd_df = pl.concat([sd_df, sd_probs_df.select(sd_df.columns)]) # must reorder columns to match sd_df

    if progress and hasattr(progress, 'empty'):
        progress.empty()

    return sd_dfs_d, sd_probs_df


def create_scores_df_with_vul(scores_df: pl.DataFrame) -> pl.DataFrame:
    # Pre-compute score columns
    score_columns = {f'Score_{level}{suit}': scores_df[f'Score_{level}{suit}']
                    for level in range(1, 8) for suit in 'CDHSN'}

    # Create a DataFrame from the score_columns dictionary
    df_scores = pl.DataFrame(score_columns)

    # Explode each column into two separate columns
    exploded_columns = []
    for col in df_scores.columns:
        exploded_columns.extend([
            pl.col(col).list.get(0).alias(f"{col}_NV"),  # Non-vulnerable score
            pl.col(col).list.get(1).alias(f"{col}_V")    # Vulnerable score
        ])

    return df_scores.with_columns(exploded_columns).drop(df_scores.columns)


# def get_cached_sd_data(pbn: str, hrs_d: Dict[str, Any]) -> Dict[str, Union[str, float]]:
#     sd_data = hrs_d[pbn]['SD'][1]
#     row_data = {'PBN': pbn}
#     for (pair_direction, declarer_direction, strain), probs in sd_data.items():
#         col_prefix = f"{pair_direction}_{declarer_direction}_{strain}"
#         for i, prob in enumerate(probs):
#             row_data[f"{col_prefix}_{i}"] = prob
#     return row_data


def calculate_sd_expected_values(df: pl.DataFrame, scores_df: pl.DataFrame) -> pl.DataFrame:

    # retrieve probabilities from cache
    #sd_probs = [get_cached_sd_data(pbn, hrs_d) for pbn in df['PBN']]

    # Create a DataFrame from the extracted sd probs (frequency distribution of tricks).
    #sd_df = pl.DataFrame(sd_probs)

    # todo: look for other places where this is called. duplicated code?
    scores_df_vuls = create_scores_df_with_vul(scores_df)

    # takes 2m for 4m rows,5m for 7m rows
    # Define the combinations
    # todo: make global function
    pair_directions = ['NS', 'EW']
    declarer_directions = 'NESW'
    strains = 'SHDCN'
    levels = range(1,8)
    tricks = range(14)
    vuls = ['NV','V']

    # Perform the multiplication
    df = df.with_columns([
        pl.col(f'Probs_{pair_direction}_{declarer_direction}_{strain}_{taken}').mul(score).alias(f'EV_{pair_direction}_{declarer_direction}_{strain}_{level}_{vul}_{taken}_{score}')
        for pair_direction in pair_directions
        for declarer_direction in pair_direction #declarer_directions
        for strain in strains
        for level in levels
        for vul in vuls
        for taken, score in zip(tricks, scores_df_vuls[f'Score_{level}{strain}_{vul}'])
    ])
    #print("Results with prob*score:")
    #display(result)

    # Add a column for the sum (expected value)
    df = df.with_columns([
        pl.sum_horizontal(pl.col(f'^EV_{pair_direction}_{declarer_direction}_{strain}_{level}_{vul}_\\d+_.*$')).alias(f'EV_{pair_direction}_{declarer_direction}_{strain}_{level}_{vul}')
        for pair_direction in pair_directions
        for declarer_direction in pair_direction #declarer_directions
        for strain in strains
        for level in levels
        for vul in vuls
    ])

    #print("\nResults with expected value:")
    return df


# Function to create columns of max values from various regexes of columns. also creates columns of the column names of the max value.
def max_horizontal_and_col(df, pattern):
    cols = df.select(pl.col(pattern)).columns
    max_expr = pl.max_horizontal(pl.col(pattern))
    col_expr = pl.when(pl.col(cols[0]) == max_expr).then(pl.lit(cols[0]))
    for col in cols[1:]:
        col_expr = col_expr.when(pl.col(col) == max_expr).then(pl.lit(col))
    return max_expr, col_expr.otherwise(pl.lit(""))


# calculate EV max scores for various regexes including all vulnerabilities. also create columns of the column names of the max values.
def create_best_contracts(df: pl.DataFrame) -> pl.DataFrame:

    # Define the combinations
    pair_directions = ['NS', 'EW']
    declarer_directions = 'NESW'
    strains = 'SHDCN'
    vulnerabilities = ['NV', 'V']

    # Dictionary to store expressions with their aliases as keys
    max_ev_dict = {}

    # all EV columns are already calculated. just need to get the max.

    # Single loop handles all EV Max, Max_Col combinations
    for v in vulnerabilities:
        # Level 4: Overall Max EV for each vulnerability
        ev_columns = f'^EV_(NS|EW)_[NESW]_[SHDCN]_[1-7]_{v}$'
        max_expr, col_expr = max_horizontal_and_col(df, ev_columns)
        max_ev_dict[f'EV_{v}_Max'] = max_expr
        max_ev_dict[f'EV_{v}_Max_Col'] = col_expr

        for pd in pair_directions:
            # Level 3: Max EV for each pair direction and vulnerability
            ev_columns = f'^EV_{pd}_[NESW]_[SHDCN]_[1-7]_{v}$'
            max_expr, col_expr = max_horizontal_and_col(df, ev_columns)
            max_ev_dict[f'EV_{pd}_{v}_Max'] = max_expr
            max_ev_dict[f'EV_{pd}_{v}_Max_Col'] = col_expr

            for dd in pd: #declarer_directions:
                # Level 2: Max EV for each pair direction, declarer direction, and vulnerability
                ev_columns = f'^EV_{pd}_{dd}_[SHDCN]_[1-7]_{v}$'
                max_expr, col_expr = max_horizontal_and_col(df, ev_columns)
                max_ev_dict[f'EV_{pd}_{dd}_{v}_Max'] = max_expr
                max_ev_dict[f'EV_{pd}_{dd}_{v}_Max_Col'] = col_expr

                for s in strains:
                    # Level 1: Max EV for each combination
                    ev_columns = f'^EV_{pd}_{dd}_{s}_[1-7]_{v}$'
                    max_expr, col_expr = max_horizontal_and_col(df, ev_columns)
                    max_ev_dict[f'EV_{pd}_{dd}_{s}_{v}_Max'] = max_expr
                    max_ev_dict[f'EV_{pd}_{dd}_{s}_{v}_Max_Col'] = col_expr

    # Create expressions list from dictionary
    t = time.time()
    all_max_ev_expr = [expr.alias(alias) for alias, expr in max_ev_dict.items()]
    print(f"create_best_contracts: all_max_ev_expr created: time:{time.time()-t} seconds")

    # Create a new DataFrame with only the new columns
    # todo: this step is inexplicably slow. appears to take 6 seconds regardless of row count?
    t = time.time()
    df = df.select(all_max_ev_expr)
    print(f"create_best_contracts: sd_ev_max_df created: time:{time.time()-t} seconds")

    return df


def convert_contract_to_contract(df: pl.DataFrame) -> pl.Series:
    # todo: use strain to strai dict instead of suit symbol. Replace replace() with replace_strict().
    # todo: implement in case 'Contract' is not in self.df.columns but BidLvl, BidSuit, Dbl, Declarer_Direction are. Or perhaps as a comparison sanity check.
    # self.df = self.df.with_columns(
    #     # easier to use discrete replaces instead of having to slice contract (nt, pass would be a complication)
    #     # first NT->N and suit symbols to SHDCN
    #     # If BidLvl is None, make Contract None
    #     pl.when(pl.col('BidLvl').is_null())
    #     .then(None)
    #     .otherwise(pl.col('BidLvl').cast(pl.String)+pl.col('BidSuit')+pl.col('Dbl')+pl.col('Declarer_Direction'))
    #     .alias('Contract'),
    # )
    return df['Contract'].str.to_uppercase().str.replace('♠','S').str.replace('♥','H').str.replace('♦','D').str.replace('♣','C').str.replace('NT','N')


# None is used instead of pl.Null because pl.Null becomes 'Null' string in pl.String columns. Not sure what's going on but the solution is to use None.
def convert_contract_to_declarer(df: pl.DataFrame) -> pl.DataFrame:
    """Optimized version using vectorized operations."""
    return df.with_columns(
        pl.when(pl.col('Contract').is_null() | (pl.col('Contract') == 'PASS'))
        .then(None)
        .otherwise(pl.col('Contract').str.slice(-1))
        .alias('Declarer_Direction')
    )


def convert_contract_to_pair_declarer(df: pl.DataFrame) -> pl.DataFrame:
    """Optimized version using vectorized operations."""
    return df.with_columns(
        pl.when(pl.col('Contract').is_null() | (pl.col('Contract') == 'PASS'))
        .then(None)
        .when(pl.col('Contract').str.slice(-1).is_in(['N', 'S']))
        .then(pl.lit('NS'))
        .otherwise(pl.lit('EW'))
        .alias('Declarer_Pair_Direction')
    )


def convert_contract_to_vul_declarer(df: pl.DataFrame) -> pl.DataFrame:
    """Optimized version using vectorized operations."""
    return df.with_columns(
        pl.when(pl.col('Contract').is_null() | (pl.col('Contract') == 'PASS'))
        .then(None)
        .when(pl.col('Declarer_Pair_Direction').eq('NS'))
        .then(pl.col('Vul_NS'))
        .when(pl.col('Declarer_Pair_Direction').eq('EW'))
        .then(pl.col('Vul_EW'))
        .alias('Vul_Declarer')
    )

def convert_contract_to_level(df: pl.DataFrame) -> pl.DataFrame:
    """Optimized version using vectorized operations."""
    return df.with_columns(
        pl.col('Contract').str.slice(0, 1).cast(pl.UInt8,strict=False).alias('BidLvl')
    )

def convert_contract_to_strain(df: pl.DataFrame) -> pl.DataFrame:
    """Optimized version using vectorized operations."""
    return df.with_columns(
        pl.when(pl.col('Contract').is_null() | (pl.col('Contract') == 'PASS'))
        .then(None)
        .otherwise(pl.col('Contract').str.slice(1, 1))
        .alias('BidSuit')
    )


def convert_contract_to_dbl(df: pl.DataFrame) -> pl.DataFrame:
    """Optimized version using vectorized operations."""
    x = df.with_columns(
        pl.when(pl.col('Contract').is_null() | (pl.col('Contract') == 'PASS'))
        .then(pl.lit(None))
        .when(pl.col('Contract').str.len_chars().eq(3))
        .then(pl.lit(''))
        .when(pl.col('Contract').str.len_chars().eq(4))
        .then(pl.lit('X'))
        .when(pl.col('Contract').str.len_chars().eq(5))
        .then(pl.lit('XX'))
        .alias('Dbl')
    )
    print(x['Dbl'].value_counts().sort('Dbl'))
    return x

def convert_player_name_to_player_names(df: pl.DataFrame) -> pl.DataFrame:
    """Optimized version using vectorized operations."""
    return df.with_columns([
        pl.concat_list([pl.col("Player_Name_N"), pl.col("Player_Name_S")]).alias("Player_Names_NS"),
        pl.concat_list([pl.col("Player_Name_E"), pl.col("Player_Name_W")]).alias("Player_Names_EW"),
    ])

def convert_declarer_to_DeclarerName(df: pl.DataFrame) -> pl.DataFrame:
    """Optimized version using vectorized operations."""
    return df.with_columns(
        pl.when(pl.col('Declarer_Direction').is_null())
        .then(None)
        .otherwise(
            pl.struct(['Declarer_Direction', 'Player_Name_N', 'Player_Name_E', 'Player_Name_S', 'Player_Name_W'])
            .map_elements(
                lambda r: None if r['Declarer_Direction'] is None else r[f"Player_Name_{r['Declarer_Direction']}"],
                return_dtype=pl.String
            )
        )
        .alias('Declarer_Name')
    )


def convert_declarer_to_DeclarerID(df: pl.DataFrame) -> pl.DataFrame:
    """Optimized version using vectorized operations."""
    return df.with_columns(
        pl.when(pl.col('Declarer_Direction').is_null())
        .then(None)
        .otherwise(
            pl.struct(['Declarer_Direction', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W'])
            .map_elements(
                lambda r: None if r['Declarer_Direction'] is None else r[f"Player_ID_{r['Declarer_Direction']}"],
                return_dtype=pl.String
            )
        )
        .alias('Declarer_ID')
    )


# convert to ml df needs to perform this.
def convert_contract_to_result(df: pl.DataFrame) -> List[Optional[int]]:
    assert False, "convert_contract_to_result not implemented. Must be done in convert_to_mldf()."
#    return [None if c is None or c == 'PASS' else 0 if c[-1] in ['=','0'] else int(c[-1]) if c[-2] == '+' else -int(c[-1]) for c in df['Contract']] # create result from contract


def convert_tricks_to_result(df: pl.DataFrame) -> pl.DataFrame:
    """Optimized version using vectorized operations."""
    return df.with_columns(
        pl.when(pl.col('Tricks').is_null() | (pl.col('Contract') == 'PASS'))
        .then(None)
        .otherwise(pl.col('Tricks') - 6 - pl.col('Contract').str.slice(0, 1).cast(pl.UInt8))
        .alias('Result')
    )


def convert_contract_to_tricks(df: pl.DataFrame) -> pl.DataFrame:
    """Optimized version using vectorized operations."""
    return df.with_columns(
        pl.when(pl.col('Contract').is_null() | (pl.col('Contract') == 'PASS') | pl.col('Result').is_null())
        .then(None)
        .otherwise(pl.col('Contract').str.slice(0, 1).cast(pl.UInt8) + 6 + pl.col('Result'))
        .alias('Tricks')
    )


# def convert_contract_to_DD_Tricks(df: pl.DataFrame) -> List[Optional[int]]:
#     return [None if c is None or c == 'PASS' else df['_'.join(['DD',d,c[1]])][i] for i,(c,d) in enumerate(zip(df['Contract'],df['Declarer_Direction']))] # extract double dummy tricks using contract and declarer as the lookup keys


# def convert_contract_to_DD_Tricks_Dummy(df: pl.DataFrame) -> List[Optional[int]]:
#     return [None if c is None or c == 'PASS' else df['_'.join(['DD',d,c[1]])][i] for i,(c,d) in enumerate(zip(df['Contract'],df['Dummy_Direction']))] # extract double dummy tricks using contract and declarer as the lookup keys


def create_dd_tricks_column_optimized(df: pl.DataFrame) -> pl.DataFrame:
    """Create DD_Tricks column using optimized vectorized operations"""
    return df.with_columns(
        pl.struct(['Declarer_Direction', 'BidSuit', pl.col('^DD_[NESW]_[CDHSN]$')])
        .map_elements(
            lambda r: None if r['Declarer_Direction'] is None else r[f"DD_{r['Declarer_Direction']}_{r['BidSuit']}"],
            return_dtype=pl.UInt8
        )
        .alias('DD_Tricks')
    )


def create_dd_tricks_dummy_column_optimized(df: pl.DataFrame) -> pl.DataFrame:
    """Create DD_Tricks_Dummy column using optimized vectorized operations"""
    return df.with_columns(
        pl.struct(['Dummy_Direction', 'BidSuit', pl.col('^DD_[NESW]_[CDHSN]$')])
        .map_elements(
            lambda r: None if r['Dummy_Direction'] is None else r[f"DD_{r['Dummy_Direction']}_{r['BidSuit']}"],
            return_dtype=pl.UInt8
        )
        .alias('DD_Tricks_Dummy')
    )


# todo: implement this
def AugmentACBLHandRecords(df: pl.DataFrame) -> pl.DataFrame:

    augmenter = FinalContractAugmenter(df)
    df = augmenter.perform_final_contract_augmentations()

    # takes 5s
    if 'game_date' in df.columns:
        t = time.time()
        df = df.with_columns(pl.Series('Date',df['game_date'].str.strptime(pl.Date,'%Y-%m-%d %H:%M:%S')))
        print(f"Time to create ACBL Date: {time.time()-t} seconds")
    # takes 5s
    if 'hand_record_id' in df.columns:
        t = time.time()
        df = df.with_columns(
            pl.col('hand_record_id').cast(pl.String),
        )
        print(f"Time to create ACBL hand_record_id: {time.time()-t} seconds")
    return df


# def Perform_Legacy_Renames(df: pl.DataFrame) -> pl.DataFrame:

#     df = df.with_columns([
#         #pl.col('Section').alias('section_name'), # will this be needed for acbl?
#         pl.col('N').alias('Player_Name_N'),
#         pl.col('S').alias('Player_Name_S'),
#         pl.col('E').alias('Player_Name_E'),
#         pl.col('W').alias('Player_Name_W'),
#         pl.col('Declarer_Name').alias('Name_Declarer'),
#         pl.col('Declarer_ID').alias('Number_Declarer'), #  todo: rename to 'Declarer_ID'?
#         pl.concat_list(['N', 'S']).alias('Player_Names_NS'),
#         pl.concat_list(['E', 'W']).alias('Player_Names_EW'),
#         # EV legacy renames
#         # pl.col('EV_Max_Col').alias('SD_Contract_Max'), # Pair direction invariant.
#         # pl.col('EV_Max_NS').alias('SD_Score_NS'),
#         # pl.col('EV_Max_EW').alias('SD_Score_EW'),
#         # pl.col('EV_Max_NS').alias('SD_Score_Max_NS'),
#         # pl.col('EV_Max_EW').alias('SD_Score_Max_EW'),
#         # (pl.col('EV_Max_NS')-pl.col('Score_NS')).alias('SD_Score_Diff_NS'),
#         # (pl.col('EV_Max_EW')-pl.col('Score_EW')).alias('SD_Score_Diff_EW'),
#         # (pl.col('EV_Max_NS')-pl.col('Score_NS')).alias('SD_Score_Max_Diff_NS'),
#         # (pl.col('EV_Max_EW')-pl.col('Score_EW')).alias('SD_Score_Max_Diff_EW'),
#         # (pl.col('EV_Max_NS')-pl.col('Pct_NS')).alias('SD_Pct_Diff_NS'),
#         # (pl.col('EV_Max_EW')-pl.col('Pct_EW')).alias('SD_Pct_Diff_EW'),
#         ])
#     return df


def DealToCards(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col(f'Suit_{direction}_{suit}').str.contains(rank).alias(f'C_{direction}{suit}{rank}')
        for direction in 'NESW'
        for suit in 'SHDC'
        for rank in 'AKQJT98765432'
    ])
    return df


def CardsToHCP(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate High Card Points (HCP) for a bridge hand dataset.
    
    Args:
    df (pl.DataFrame): Input DataFrame with columns named C_{direction}{suit}{rank}
                       where direction is N, E, S, W, suit is S, H, D, C, and rank is A, K, Q, J.
    
    Returns:
    pl.DataFrame: Input DataFrame with additional HCP columns.
    """
    hcp_d = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}

    # Step 1: Calculate HCP for each direction and suit
    hcp_suit_expr = [
        pl.sum_horizontal([pl.col(f'C_{d}{s}{r}').cast(pl.UInt8) * v for r, v in hcp_d.items()]).alias(f'HCP_{d}_{s}')
        for d in 'NESW' for s in 'SHDC'
    ]
    df = df.with_columns(hcp_suit_expr)

    # Step 2: Calculate total HCP for each direction
    hcp_direction_expr = [
        pl.sum_horizontal([pl.col(f'HCP_{d}_{s}') for s in 'SHDC']).alias(f'HCP_{d}')
        for d in 'NESW'
    ]
    df = df.with_columns(hcp_direction_expr)

    # Step 3: Calculate HCP for partnerships
    hcp_partnership_expr = [
        (pl.col('HCP_N') + pl.col('HCP_S')).alias('HCP_NS'),
        (pl.col('HCP_E') + pl.col('HCP_W')).alias('HCP_EW')
    ]
    df = df.with_columns(hcp_partnership_expr)

    # Step 4: Calculate HCP for partnerships by suit
    hcp_partnership_suit_expr = [
        (pl.col(f'HCP_N_{s}') + pl.col(f'HCP_S_{s}')).alias(f'HCP_NS_{s}')
        for s in 'SHDC'
    ] + [
        (pl.col(f'HCP_E_{s}') + pl.col(f'HCP_W_{s}')).alias(f'HCP_EW_{s}')
        for s in 'SHDC'
    ]
    df = df.with_columns(hcp_partnership_suit_expr)

    return df


def CardsToQuickTricks(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate Quick Tricks for a bridge hand dataset.
    
    Args:
    df (pl.DataFrame): Input DataFrame with Suit_{direction}_{suit} columns.
    
    Returns:
    pl.DataFrame: DataFrame with additional Quick Tricks columns.
    """
    qt_dict = {'AK': 2.0, 'AQ': 1.5, 'A': 1.0, 'KQ': 1.0, 'K': 0.5}
    
    # Calculate QT for each suit
    qt_expr = [
        pl.when(pl.col(f'Suit_{d}_{s}').str.starts_with('AK')).then(pl.lit(2.0))
        .when(pl.col(f'Suit_{d}_{s}').str.starts_with('AQ')).then(pl.lit(1.5))
        .when(pl.col(f'Suit_{d}_{s}').str.starts_with('A')).then(pl.lit(1.0))
        .when(pl.col(f'Suit_{d}_{s}').str.starts_with('KQ')).then(pl.lit(1.0))
        .when(pl.col(f'Suit_{d}_{s}').str.starts_with('K')).then(pl.lit(0.5))
        .otherwise(pl.lit(0.0)).alias(f'QT_{d}_{s}')
        for d in 'NESW' for s in 'SHDC'
    ]
    
    # Apply suit QT calculations
    df = df.with_columns(qt_expr)
    
    # Calculate QT for each direction
    direction_qt = [
        pl.sum_horizontal([pl.col(f'QT_{d}_{s}') for s in 'SHDC']).alias(f'QT_{d}')
        for d in 'NESW'
    ]
    
    # Apply direction QT calculations
    df = df.with_columns(direction_qt)
    
    # Calculate partnership QT
    partnership_qt = [
        (pl.col('QT_N') + pl.col('QT_S')).alias('QT_NS'),
        (pl.col('QT_E') + pl.col('QT_W')).alias('QT_EW')
    ]
    
    # Apply partnership QT calculations
    df = df.with_columns(partnership_qt)
    
    # Calculate partnership QT by suit
    partnership_qt_suit = [
        (pl.col(f'QT_N_{s}') + pl.col(f'QT_S_{s}')).alias(f'QT_NS_{s}')
        for s in 'SHDC'
    ] + [
        (pl.col(f'QT_E_{s}') + pl.col(f'QT_W_{s}')).alias(f'QT_EW_{s}')
        for s in 'SHDC'
    ]
    
    # Apply partnership QT by suit calculations
    return df.with_columns(partnership_qt_suit)


def calculate_LoTT(df: pl.DataFrame) -> pl.DataFrame:
    # Calculate Law of Total Tricks
    df = df.with_columns([
        (pl.col('SL_NS_C') + pl.col('SL_NS_D') + pl.col('SL_NS_H') + pl.col('SL_NS_S')).alias('LoTT_NS'),
        (pl.col('SL_EW_C') + pl.col('SL_EW_D') + pl.col('SL_EW_H') + pl.col('SL_EW_S')).alias('LoTT_EW'),
    ])
    return df

# Global column creation functions
def create_dealer_column(df: pl.DataFrame) -> pl.DataFrame:
    """Create Dealer column from Board"""
    def board_number_to_dealer(bn):
        return 'NESW'[(bn-1) & 3]
    
    return df.with_columns(
        pl.col('Board')
        .map_elements(board_number_to_dealer, return_dtype=pl.String)
        .alias('Dealer')
    )

def create_ivul_from_vul(df: pl.DataFrame) -> pl.DataFrame:
    """Create iVul column from Vul column"""
    def vul_to_ivul(vul: str) -> int:
        return ['None','N_S','E_W','Both'].index(vul)
    
    return df.with_columns(
        pl.col('Vul')
        .map_elements(vul_to_ivul, return_dtype=pl.UInt8)
        .alias('iVul')
    )

def create_ivul_from_board(df: pl.DataFrame) -> pl.DataFrame:
    """Create iVul column from Board column"""
    def board_number_to_vul(bn: int) -> int:
        bn -= 1
        return range(bn//4, bn//4+4)[bn & 3] & 3
    
    return df.with_columns(
        pl.col('Board')
        .map_elements(board_number_to_vul, return_dtype=pl.UInt8)
        .alias('iVul')
    )

def create_vul_from_ivul(df: pl.DataFrame) -> pl.DataFrame:
    """Create Vul column from iVul column"""
    def ivul_to_vul(ivul: int) -> str:
        return ['None','N_S','E_W','Both'][ivul]
    
    return df.with_columns(
        pl.col('iVul')
        .map_elements(ivul_to_vul, return_dtype=pl.String)
        .alias('Vul')
    )

def create_vul_ns_ew_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Create Vul_NS and Vul_EW columns from Vul column"""
    return df.with_columns([
        pl.Series('Vul_NS', df['Vul'].is_in(['N_S','Both']), pl.Boolean),
        pl.Series('Vul_EW', df['Vul'].is_in(['E_W','Both']), pl.Boolean)
    ])

def create_suit_length_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Create SL_[NESW]_[SHDC] columns from Suit_[NESW]_[SHDC] columns"""
    sl_nesw_columns = [
        pl.col(f"Suit_{direction}_{suit}").str.len_chars().alias(f"SL_{direction}_{suit}")
        for direction in "NESW"
        for suit in "SHDC"
    ]
    return df.with_columns(sl_nesw_columns)

def create_pair_suit_length_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Create SL_(NS|EW)_[SHDC] columns from SL_[NESW]_[SHDC] columns"""
    sl_ns_ew_columns = [
        pl.sum_horizontal(f"SL_{pair[0]}_{suit}", f"SL_{pair[1]}_{suit}").alias(f"SL_{pair}_{suit}")
        for pair in ['NS', 'EW']
        for suit in "SHDC"
    ]
    return df.with_columns(sl_ns_ew_columns)

def create_suit_length_arrays_for_direction(df: pl.DataFrame, direction: str) -> pl.DataFrame:
    """Create suit length array columns for a specific direction using vectorized operations."""
    # Build base list columns
    cdhs_expr = pl.concat_list([
        pl.col(f"SL_{direction}_C"),
        pl.col(f"SL_{direction}_D"),
        pl.col(f"SL_{direction}_H"),
        pl.col(f"SL_{direction}_S"),
    ]).alias(f"SL_{direction}_CDHS")

    cdhs_sj_expr = (
        pl.col(f"SL_{direction}_C").cast(pl.String) + pl.lit("-") +
        pl.col(f"SL_{direction}_D").cast(pl.String) + pl.lit("-") +
        pl.col(f"SL_{direction}_H").cast(pl.String) + pl.lit("-") +
        pl.col(f"SL_{direction}_S").cast(pl.String)
    ).alias(f"SL_{direction}_CDHS_SJ")

    ml_expr = pl.concat_list([
        pl.col(f"SL_{direction}_C"),
        pl.col(f"SL_{direction}_D"),
        pl.col(f"SL_{direction}_H"),
        pl.col(f"SL_{direction}_S"),
    ]).list.sort(descending=True).alias(f"SL_{direction}_ML")

    # Vectorized ML_I (no list.argsort; sort structs by -value then take idx)
    ml_i_expr = (
        pl.concat_list([
            pl.struct([(-pl.col(f"SL_{direction}_C").cast(pl.Int8)).alias("val_neg"), pl.lit(0).alias("idx")]),
            pl.struct([(-pl.col(f"SL_{direction}_D").cast(pl.Int8)).alias("val_neg"), pl.lit(1).alias("idx")]),
            pl.struct([(-pl.col(f"SL_{direction}_H").cast(pl.Int8)).alias("val_neg"), pl.lit(2).alias("idx")]),
            pl.struct([(-pl.col(f"SL_{direction}_S").cast(pl.Int8)).alias("val_neg"), pl.lit(3).alias("idx")]),
        ])
        .list.sort()  # lexicographic on (val_neg, idx)
        .list.eval(pl.element().struct.field("idx"))
        .alias(f"SL_{direction}_ML_I")
    )

    # Materialize base first, then derive SJ strings
    df = df.with_columns([cdhs_expr, cdhs_sj_expr, ml_expr, ml_i_expr])

    ml_sj_expr = (
        pl.col(f"SL_{direction}_ML").list.eval(pl.element().cast(pl.Utf8)).list.join("-")
        .alias(f"SL_{direction}_ML_SJ")
    )
    ml_i_sj_expr = (
        pl.col(f"SL_{direction}_ML_I").list.eval(pl.element().cast(pl.Utf8)).list.join("-")
        .alias(f"SL_{direction}_ML_I_SJ")
    )

    df = df.with_columns([ml_sj_expr, ml_i_sj_expr])
    
    return df

def create_distribution_point_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Create DP_[NESW]_[SHDC] columns from SL_[NESW]_[SHDC] columns"""
    dp_columns = [
        pl.when(pl.col(f"SL_{direction}_{suit}") == 0).then(3)
        .when(pl.col(f"SL_{direction}_{suit}") == 1).then(2)
        .when(pl.col(f"SL_{direction}_{suit}") == 2).then(1)
        .otherwise(0)
        .alias(f"DP_{direction}_{suit}")
        for direction in "NESW"
        for suit in "SHDC"
    ]
    return df.with_columns(dp_columns)

def create_total_point_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Create Total_Points columns from HCP and DP columns"""
    # Individual suit total points
    df = df.with_columns([
        (pl.col(f'HCP_{d}_{s}')+pl.col(f'DP_{d}_{s}')).alias(f'Total_Points_{d}_{s}')
        for d in 'NESW'
        for s in 'SHDC'
    ])
    
    # Direction total points
    df = df.with_columns([
        (pl.sum_horizontal([f'Total_Points_{d}_{s}' for s in 'SHDC'])).alias(f'Total_Points_{d}')
        for d in 'NESW'
    ])
    
    # Pair total points
    df = df.with_columns([
        (pl.col('Total_Points_N')+pl.col('Total_Points_S')).alias('Total_Points_NS'),
        (pl.col('Total_Points_E')+pl.col('Total_Points_W')).alias('Total_Points_EW'),
    ])
    
    return df

def create_contract_type_column(df: pl.DataFrame) -> pl.DataFrame:
    """Create ContractType column from Contract, BidLvl, and BidSuit columns"""
    return df.with_columns(
        pl.when(pl.col('Contract').eq('PASS')).then(pl.lit("Pass"))
        .when(pl.col('BidLvl').eq(5) & pl.col('BidSuit').is_in(['C', 'D'])).then(pl.lit("Game"))
        .when(pl.col('BidLvl').is_in([4,5]) & pl.col('BidSuit').is_in(['H', 'S'])).then(pl.lit("Game"))
        .when(pl.col('BidLvl').is_in([3,4,5]) & pl.col('BidSuit').eq('N')).then(pl.lit("Game"))
        .when(pl.col('BidLvl').eq(6)).then(pl.lit("SSlam"))
        .when(pl.col('BidLvl').eq(7)).then(pl.lit("GSlam"))
        .otherwise(pl.lit("Partial"))
        .alias('ContractType')
    )

def create_player_names(df: pl.DataFrame) -> pl.DataFrame:
    """Create Player_Names_(NS|EW) using optimized vectorized operations."""
    df = convert_player_name_to_player_names(df)
    return df

def create_declarer_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Create Declarer_Name and Declarer_ID columns using optimized vectorized operations."""
    df = convert_declarer_to_DeclarerName(df)
    df = convert_declarer_to_DeclarerID(df)
    return df

def create_mp_top_column(df: pl.DataFrame) -> pl.DataFrame:
    """Create MP_Top column from Score, session_id, PBN, and Board columns"""
    return df.with_columns(
        pl.col('Score').count().over(['session_id','PBN','Board']).sub(1).alias('MP_Top')
    )

def create_matchpoint_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Create MP_NS and MP_EW columns from Score_NS and Score_EW columns"""
    return df.with_columns([
        pl.col('Score_NS').rank(method='average', descending=False).sub(1)
            .over(['session_id', 'PBN', 'Board']).alias('MP_NS'),
        pl.col('Score_EW').rank(method='average', descending=False).sub(1)
            .over(['session_id', 'PBN', 'Board']).alias('MP_EW'),
    ])

def create_percentage_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Create Pct_NS and Pct_EW columns from MP_NS, MP_EW, and MP_Top columns"""
    return df.with_columns([
        (pl.col('MP_NS') / pl.col('MP_Top')).alias('Pct_NS'),
        (pl.col('MP_EW') / pl.col('MP_Top')).alias('Pct_EW')
    ])

def create_declarer_pct_column(df: pl.DataFrame) -> pl.DataFrame:
    """Create Declarer_Pct column from Declarer_Pair_Direction, Pct_NS, and Pct_EW columns"""
    return df.with_columns(
        pl.when(pl.col('Declarer_Pair_Direction').eq('NS'))
        .then('Pct_NS')
        .otherwise('Pct_EW')
        .alias('Declarer_Pct')
    )

def create_max_suit_length_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Create SL_Max_[NS|EW] columns from SL_[NS|EW]_[SHDC] columns using vectorized operations."""
    # Create max suit length columns for each pair direction
    for pair_direction in ['NS', 'EW']:
        # Get the suit length columns for this pair
        suit_cols = [f"SL_{pair_direction}_{suit}" for suit in 'SHDC']
        
        # Find the maximum suit length and its corresponding suit
        max_expr = pl.max_horizontal(suit_cols).alias(f'SL_Max_{pair_direction}')
        
        # Find which suit has the maximum length
        max_col_expr = (
            pl.when(pl.col(suit_cols[0]) == pl.max_horizontal(suit_cols)).then(pl.lit(suit_cols[0]))
            .when(pl.col(suit_cols[1]) == pl.max_horizontal(suit_cols)).then(pl.lit(suit_cols[1]))
            .when(pl.col(suit_cols[2]) == pl.max_horizontal(suit_cols)).then(pl.lit(suit_cols[2]))
            .otherwise(pl.lit(suit_cols[3]))
            .alias(f'SL_Max_{pair_direction}_Col')
        )
        
        df = df.with_columns([max_expr, max_col_expr])
    
    return df

def create_quality_indicator_columns(df: pl.DataFrame, suit_quality_criteria: dict, stopper_criteria: dict) -> pl.DataFrame:
    """Create quality indicator columns from SL and HCP columns"""
    series_expressions = [
        pl.Series(
            f"{series_type}_{direction}_{suit}",
            criteria(
                df[f"SL_{direction}_{suit}"],
                df[f"HCP_{direction}_{suit}"]
            ),
            pl.Boolean
        )
        for direction in "NESW"
        for suit in "SHDC"
        for series_type, criteria in {**suit_quality_criteria, **stopper_criteria}.items()
    ]
    
    df = df.with_columns(series_expressions)
    df = df.with_columns([
        pl.lit(False).alias(f"Forcing_One_Round"),
        pl.lit(False).alias(f"Opponents_Cannot_Play_Undoubled_Below_2N"),
        pl.lit(False).alias(f"Forcing_To_2N"),
        pl.lit(False).alias(f"Forcing_To_3N"),
    ])
    return df

def create_balanced_indicator_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Create Balanced_[NESW] columns from SL_[NESW]_ML_SJ, SL_[NESW]_C, and SL_[NESW]_D columns"""
    return df.with_columns([
        pl.Series(
            f"Balanced_{direction}",
            df[f"SL_{direction}_ML_SJ"].is_in(['4-3-3-3','4-4-3-2']) |
            (df[f"SL_{direction}_ML_SJ"].is_in(['5-3-3-2','5-4-2-2']) & 
             (df[f"SL_{direction}_C"].eq(5) | df[f"SL_{direction}_D"].eq(5))),
            pl.Boolean
        )
        for direction in 'NESW'
    ])

def create_result_column_from_tricks(df: pl.DataFrame) -> pl.DataFrame:
    """Create Result column from Tricks column using optimized vectorized operations."""
    return convert_tricks_to_result(df)

def create_result_column_from_contract(df: pl.DataFrame) -> pl.DataFrame:
    """Create Result column from Contract column"""
    return df.with_columns(
        pl.Series('Result', convert_contract_to_result(df), pl.Int8, strict=False)
    )

def create_tricks_column_from_contract(df: pl.DataFrame) -> pl.DataFrame:
    """Create Tricks column from Contract column using optimized vectorized operations."""
    return convert_contract_to_tricks(df)


def create_dd_score_columns_optimized(df: pl.DataFrame, scores_d: Dict) -> pl.DataFrame:
    """Create DD_Score columns using optimized vectorized operations"""
    # Create DD_Score_Refs column
    df = df.with_columns(
        (pl.lit('DD_Score_')+pl.col('BidLvl').cast(pl.String)+pl.col('BidSuit')+pl.lit('_')+pl.col('Declarer_Direction')).alias('DD_Score_Refs'),
    )
    
    # Create scores for columns: DD_Score_[1-7][CDHSN]_[NESW]
    df = df.with_columns([
        pl.struct([f"DD_{direction}_{strain}", f"Vul_{pair_direction}"])
        .map_elements(
            lambda r, lvl=level, strn=strain, dir=direction, pdir=pair_direction: 
                scores_d.get((lvl, strn, r[f"DD_{dir}_{strn}"], r[f"Vul_{pdir}"]), None),
            return_dtype=pl.Int16
        )
        .alias(f"DD_Score_{level}{strain}_{direction}")
        for level in range(1, 8)
        for strain in mlBridgeLib.CDHSN
        for direction, pair_direction in [('N','NS'), ('E','EW'), ('S','NS'), ('W','EW')]
    ])

    # Create list of column names: DD_Score_[1-7][CDHSN]_[NESW]
    dd_score_columns = [f"DD_Score_{level}{strain}_{direction}" 
                        for level in range(1, 8)
                        for strain in mlBridgeLib.CDHSN  
                        for direction in mlBridgeLib.NESW]
    
    # Create DD_Score_Declarer by selecting the DD_Score_[1-7][CDHSN]_[NESW] column for the given Declarer_Direction
    df = df.with_columns([
        pl.struct(['BidLvl', 'BidSuit', 'Declarer_Direction'] + dd_score_columns)
        .map_elements(
            lambda r: None if r['Declarer_Direction'] is None else r[f"DD_Score_{r['BidLvl']}{r['BidSuit']}_{r['Declarer_Direction']}"],
            return_dtype=pl.Int16
        )
        .alias('DD_Score_Declarer')
    ])

    # Create DD_Score_NS and DD_Score_EW columns
    df = df.with_columns([
        pl.when(pl.col('Declarer_Pair_Direction').eq('NS'))
        .then(pl.col('DD_Score_Declarer'))
        .when(pl.col('Declarer_Pair_Direction').eq('EW'))
        .then(pl.col('DD_Score_Declarer').neg())
        .otherwise(0)
        .alias('DD_Score_NS'),
        
        pl.when(pl.col('Declarer_Pair_Direction').eq('EW'))
        .then(pl.col('DD_Score_Declarer'))
        .when(pl.col('Declarer_Pair_Direction').eq('NS'))
        .then(pl.col('DD_Score_Declarer').neg())
        .otherwise(0)
        .alias('DD_Score_EW')
    ])
    
    return df

def create_direction_summary_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Create direction summary columns (DP_[NESW], DP_[NS|EW], DP_[NS|EW]_[SHDC]) from DP_[NESW]_[SHDC] columns"""
    # Direction total DPs
    df = df.with_columns([
        (pl.col(f'DP_{d}_S')+pl.col(f'DP_{d}_H')+pl.col(f'DP_{d}_D')+pl.col(f'DP_{d}_C')).alias(f'DP_{d}')
        for d in "NESW"
    ])
    
    # Pair total DPs
    df = df.with_columns([
        (pl.col('DP_N')+pl.col('DP_S')).alias('DP_NS'),
        (pl.col('DP_E')+pl.col('DP_W')).alias('DP_EW'),
    ])
    
    # Pair suit DPs
    df = df.with_columns([
        (pl.col(f'DP_N_{s}') + pl.col(f'DP_S_{s}')).alias(f'DP_NS_{s}')
        for s in 'SHDC'
    ] + [
        (pl.col(f'DP_E_{s}') + pl.col(f'DP_W_{s}')).alias(f'DP_EW_{s}')
        for s in 'SHDC'
    ])
    
    return df

import polars as pl
import math
from collections import defaultdict

# todo: gpt5 offered: Optional: JIT the loop with numba for 10–100x speedup on large datasets.
def compute_elo_pair_matchpoint_ratings(
    df: pl.DataFrame,
    *,
    initial_rating: float = 1500.0,
    k_base: float = 24.0,
    provisional_boost_until: int = 100  # boards per pair-direction
) -> pl.DataFrame:
    # Ensure schema
    need_cols = {"Date", "Pair_Number_NS", "Pair_Number_EW", "Pct_NS"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Sort to preserve temporal order
    sort_keys = [c for c in ["Date", "session_id", "Round", "Board"] if c in df.columns]
    if not sort_keys:
        sort_keys = ["Date"]
    df_sorted = df.sort(sort_keys)

    n_rows = df_sorted.height
    ns_pairs = df_sorted["Pair_Number_NS"].to_numpy()
    ew_pairs = df_sorted["Pair_Number_EW"].to_numpy()
    pct_ns = df_sorted["Pct_NS"].to_numpy()

    # Section-size scaling vector
    scale = np.ones(n_rows, dtype=np.float64)
    if "Section_Pairs" in df_sorted.columns:
        pairs = df_sorted["Section_Pairs"].cast(pl.Float64).to_numpy()
    elif "Num_Pairs" in df_sorted.columns:
        pairs = df_sorted["Num_Pairs"].cast(pl.Float64).to_numpy()
    elif "MP_Top" in df_sorted.columns:
        pairs = (df_sorted["MP_Top"].cast(pl.Float64).to_numpy() / 2.0) + 1.0
    else:
        pairs = None
    if pairs is not None:
        with np.errstate(invalid="ignore"):
            valid = pairs > 1.0
        scale_valid = np.sqrt(np.maximum(pairs[valid] - 1.0, 1.0) / 11.0)
        scale[valid] = scale_valid

    # Map pair ids to contiguous indices per direction
    ns_unique = df_sorted["Pair_Number_NS"].unique().to_list()
    ew_unique = df_sorted["Pair_Number_EW"].unique().to_list()
    ns_index = {pid: i for i, pid in enumerate(ns_unique)}
    ew_index = {pid: i for i, pid in enumerate(ew_unique)}

    ratings_ns = np.full(len(ns_unique), initial_rating, dtype=np.float64)
    ratings_ew = np.full(len(ew_unique), initial_rating, dtype=np.float64)
    counts_ns = np.zeros(len(ns_unique), dtype=np.int32)
    counts_ew = np.zeros(len(ew_unique), dtype=np.int32)

    # Outputs
    r_ns_before_arr = np.empty(n_rows, dtype=np.float64)
    r_ew_before_arr = np.empty(n_rows, dtype=np.float64)
    e_ns_arr = np.empty(n_rows, dtype=np.float64)
    e_ew_arr = np.empty(n_rows, dtype=np.float64)
    r_ns_after_arr = np.empty(n_rows, dtype=np.float64)
    r_ew_after_arr = np.empty(n_rows, dtype=np.float64)
    n_ns_arr = np.empty(n_rows, dtype=np.int32)
    n_ew_arr = np.empty(n_rows, dtype=np.int32)
    delta_before_arr = np.empty(n_rows, dtype=np.float64)
    delta_after_arr = np.empty(n_rows, dtype=np.float64)

    for i in range(n_rows):
        ns = ns_pairs[i]
        ew = ew_pairs[i]
        idx_ns = ns_index[ns]
        idx_ew = ew_index[ew]

        r_ns = ratings_ns[idx_ns]
        r_ew = ratings_ew[idx_ew]

        # Expected
        e_ns = 1.0 / (1.0 + 10.0 ** (-(r_ns - r_ew) / 400.0))
        e_ew = 1.0 - e_ns

        # K with provisional boost and section size scale
        k_ns = k_base * (1.5 if counts_ns[idx_ns] < provisional_boost_until else 1.0) * scale[i]
        k_ew = k_base * (1.5 if counts_ew[idx_ew] < provisional_boost_until else 1.0) * scale[i]

        r_ns_before = r_ns
        r_ew_before = r_ew

        s_ns = pct_ns[i]
        if s_ns is not None and not (isinstance(s_ns, float) and np.isnan(s_ns)):
            s_ns_f = float(s_ns)
            s_ew_f = 1.0 - s_ns_f
            r_ns = r_ns + k_ns * (s_ns_f - e_ns)
            r_ew = r_ew + k_ew * (s_ew_f - e_ew)
            ratings_ns[idx_ns] = r_ns
            ratings_ew[idx_ew] = r_ew
            counts_ns[idx_ns] += 1
            counts_ew[idx_ew] += 1

        # Save outputs
        r_ns_before_arr[i] = r_ns_before
        r_ew_before_arr[i] = r_ew_before
        e_ns_arr[i] = e_ns
        e_ew_arr[i] = e_ew
        r_ns_after_arr[i] = r_ns
        r_ew_after_arr[i] = r_ew
        n_ns_arr[i] = counts_ns[idx_ns]
        n_ew_arr[i] = counts_ew[idx_ew]
        delta_before_arr[i] = r_ns_before - r_ew_before
        delta_after_arr[i] = r_ns - r_ew

    out_df = pl.DataFrame({
        "Elo_R_NS_Before": r_ns_before_arr,
        "Elo_R_EW_Before": r_ew_before_arr,
        "Elo_E_Pair_NS": e_ns_arr,
        "Elo_E_Pair_EW": e_ew_arr,
        "Elo_R_NS": r_ns_after_arr,
        "Elo_R_EW": r_ew_after_arr,
        "Elo_N_NS": n_ns_arr,
        "Elo_N_EW": n_ew_arr,
        "Elo_Delta_Before": delta_before_arr,
        "Elo_Delta_After": delta_after_arr,
    })
    return df_sorted.hstack(out_df)


def compute_elo_player_matchpoint_ratings(
    df: pl.DataFrame,
    *,
    initial_rating: float = 1500.0,
    k_base: float = 24.0,
    provisional_boost_until: int = 100,  # boards per player-direction
) -> pl.DataFrame:
    """
    Compute leakage-safe player Elo-style ratings for duplicate boards.

    Required columns:
      - 'Date'
      - 'Player_ID_N', 'Player_ID_S', 'Player_ID_E', 'Player_ID_W'
      - 'Pct_NS' in [0, 1]

    Returns the original DataFrame with appended columns (aligned by row):
      - R_N_NS_Before, R_S_NS_Before, R_E_EW_Before, R_W_EW_Before
      - NS_side_Before, EW_side_Before, Elo_Delta_Before
      - E_NS, E_EW (expected scores from pre-board ratings)
      - R_N_NS_after, R_S_NS_after, R_E_EW_after, R_W_EW_after
      - N_N_NS, N_S_NS, N_E_EW, N_W_EW (played counts per player-direction)
    """
    need_cols = {
        "Date",
        "Player_ID_N",
        "Player_ID_S",
        "Player_ID_E",
        "Player_ID_W",
        "Pct_NS",
    }
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Sort by available stable keys to preserve temporal order
    sort_keys = [c for c in ["Date", "session_id", "Round", "Board"] if c in df.columns]
    if not sort_keys:
        sort_keys = ["Date"]
    df_sorted = df.sort(sort_keys)

    # Extract arrays
    n_rows = df_sorted.height
    pid_n_arr = df_sorted["Player_ID_N"].to_numpy()
    pid_s_arr = df_sorted["Player_ID_S"].to_numpy()
    pid_e_arr = df_sorted["Player_ID_E"].to_numpy()
    pid_w_arr = df_sorted["Player_ID_W"].to_numpy()
    pct_ns = df_sorted["Pct_NS"].to_numpy()

    # Section-size scaling vector (same precedence as pair Elo)
    scale = np.ones(n_rows, dtype=np.float64)
    if "Section_Pairs" in df_sorted.columns:
        pairs = df_sorted["Section_Pairs"].cast(pl.Float64).to_numpy()
    elif "Num_Pairs" in df_sorted.columns:
        pairs = df_sorted["Num_Pairs"].cast(pl.Float64).to_numpy()
    elif "MP_Top" in df_sorted.columns:
        pairs = (df_sorted["MP_Top"].cast(pl.Float64).to_numpy() / 2.0) + 1.0
    else:
        pairs = None
    if pairs is not None:
        with np.errstate(invalid="ignore"):
            valid = pairs > 1.0
        scale_valid = np.sqrt(np.maximum(pairs[valid] - 1.0, 1.0) / 11.0)
        scale[valid] = scale_valid

    # Map player ids to contiguous indices per direction
    ns_player_unique = pl.concat([
        df_sorted["Player_ID_N"],
        df_sorted["Player_ID_S"],
    ]).unique().to_list()
    ew_player_unique = pl.concat([
        df_sorted["Player_ID_E"],
        df_sorted["Player_ID_W"],
    ]).unique().to_list()

    idx_ns_map = {pid: i for i, pid in enumerate(ns_player_unique)}
    idx_ew_map = {pid: i for i, pid in enumerate(ew_player_unique)}

    ratings_ns = np.full(len(ns_player_unique), initial_rating, dtype=np.float64)
    ratings_ew = np.full(len(ew_player_unique), initial_rating, dtype=np.float64)
    counts_ns = np.zeros(len(ns_player_unique), dtype=np.int32)
    counts_ew = np.zeros(len(ew_player_unique), dtype=np.int32)

    # Outputs
    R_N_Before = np.empty(n_rows, dtype=np.float64)
    R_S_Before = np.empty(n_rows, dtype=np.float64)
    R_E_Before = np.empty(n_rows, dtype=np.float64)
    R_W_Before = np.empty(n_rows, dtype=np.float64)
    NS_side_Before = np.empty(n_rows, dtype=np.float64)
    EW_side_Before = np.empty(n_rows, dtype=np.float64)
    E_NS = np.empty(n_rows, dtype=np.float64)
    E_EW = np.empty(n_rows, dtype=np.float64)
    R_N_after = np.empty(n_rows, dtype=np.float64)
    R_S_after = np.empty(n_rows, dtype=np.float64)
    R_E_after = np.empty(n_rows, dtype=np.float64)
    R_W_after = np.empty(n_rows, dtype=np.float64)
    N_N = np.empty(n_rows, dtype=np.int32)
    N_S = np.empty(n_rows, dtype=np.int32)
    N_E = np.empty(n_rows, dtype=np.int32)
    N_W = np.empty(n_rows, dtype=np.int32)
    Elo_Delta_Before = np.empty(n_rows, dtype=np.float64)

    for i in range(n_rows):
        pid_n = pid_n_arr[i]
        pid_s = pid_s_arr[i]
        pid_e = pid_e_arr[i]
        pid_w = pid_w_arr[i]

        idx_n = idx_ns_map[pid_n]
        idx_s = idx_ns_map[pid_s]
        idx_e = idx_ew_map[pid_e]
        idx_w = idx_ew_map[pid_w]

        r_n_ns = ratings_ns[idx_n]
        r_s_ns = ratings_ns[idx_s]
        r_e_ew = ratings_ew[idx_e]
        r_w_ew = ratings_ew[idx_w]

        ns_before = (r_n_ns + r_s_ns) / 2.0
        ew_before = (r_e_ew + r_w_ew) / 2.0

        e_ns = 1.0 / (1.0 + 10.0 ** (-(ns_before - ew_before) / 400.0))
        e_ew = 1.0 - e_ns

        # K per player with provisional boost and section-size scale
        k_n = k_base * (1.5 if counts_ns[idx_n] < provisional_boost_until else 1.0) * scale[i]
        k_s = k_base * (1.5 if counts_ns[idx_s] < provisional_boost_until else 1.0) * scale[i]
        k_e = k_base * (1.5 if counts_ew[idx_e] < provisional_boost_until else 1.0) * scale[i]
        k_w = k_base * (1.5 if counts_ew[idx_w] < provisional_boost_until else 1.0) * scale[i]

        r_n_before = r_n_ns
        r_s_before = r_s_ns
        r_e_before = r_e_ew
        r_w_before = r_w_ew

        s_ns = pct_ns[i]
        if s_ns is not None and not (isinstance(s_ns, float) and np.isnan(s_ns)):
            delta = float(s_ns) - e_ns
            r_n_ns = r_n_ns + 0.5 * k_n * delta
            r_s_ns = r_s_ns + 0.5 * k_s * delta
            r_e_ew = r_e_ew - 0.5 * k_e * delta
            r_w_ew = r_w_ew - 0.5 * k_w * delta
            ratings_ns[idx_n] = r_n_ns
            ratings_ns[idx_s] = r_s_ns
            ratings_ew[idx_e] = r_e_ew
            ratings_ew[idx_w] = r_w_ew
            counts_ns[idx_n] += 1
            counts_ns[idx_s] += 1
            counts_ew[idx_e] += 1
            counts_ew[idx_w] += 1

        # Save outputs
        R_N_Before[i] = r_n_before
        R_S_Before[i] = r_s_before
        R_E_Before[i] = r_e_before
        R_W_Before[i] = r_w_before
        NS_side_Before[i] = ns_before
        EW_side_Before[i] = ew_before
        E_NS[i] = e_ns
        E_EW[i] = e_ew
        R_N_after[i] = r_n_ns
        R_S_after[i] = r_s_ns
        R_E_after[i] = r_e_ew
        R_W_after[i] = r_w_ew
        N_N[i] = counts_ns[idx_n]
        N_S[i] = counts_ns[idx_s]
        N_E[i] = counts_ew[idx_e]
        N_W[i] = counts_ew[idx_w]
        Elo_Delta_Before[i] = ns_before - ew_before

    out_df = pl.DataFrame({
        "Elo_R_N_Before": R_N_Before,
        "Elo_R_E_Before": R_E_Before,
        "Elo_R_S_Before": R_S_Before,
        "Elo_R_W_Before": R_W_Before,
        "Elo_E_Players_NS": E_NS,
        "Elo_E_Players_EW": E_EW,
        "Elo_R_N": R_N_after,
        "Elo_R_E": R_E_after,
        "Elo_R_S": R_S_after,
        "Elo_R_W": R_W_after,
        "Elo_N_N": N_N,
        "Elo_N_E": N_E,
        "Elo_N_S": N_S,
        "Elo_N_W": N_W,
    })
    return df_sorted.hstack(out_df)


def compute_event_start_end_elo_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add constant per-session Elo columns for each seat and pair.
    Adds both EventStart (pre-board at first appearance) and EventEnd
    (post-update at last appearance) values for players and pairs, and
    propagates them across all rows in the session for each entity.
    """
    t = time.time()
    sort_keys = [c for c in ["Date", "session_id", "Round", "Board"] if c in df.columns]
    if not sort_keys:
        sort_keys = ["Date"]
    df = df.sort(sort_keys)

    # Player start
    seat_before_cols = {
        "N": ("Player_ID_N", "Elo_R_N_Before", "Elo_R_Player_N_EventStart"),
        "S": ("Player_ID_S", "Elo_R_S_Before", "Elo_R_Player_S_EventStart"),
        "E": ("Player_ID_E", "Elo_R_E_Before", "Elo_R_Player_E_EventStart"),
        "W": ("Player_ID_W", "Elo_R_W_Before", "Elo_R_Player_W_EventStart"),
    }
    for _, (pid_col, before_col, out_col) in seat_before_cols.items():
        if pid_col in df.columns and before_col in df.columns:
            df = df.with_columns(
                pl.col(before_col).first().over(["session_id", pid_col]).alias(out_col)
            )

    # Pair start
    pair_before_cols = {
        "NS": ("Pair_Number_NS", "Elo_R_NS_Before", "Elo_R_Pair_NS_EventStart"),
        "EW": ("Pair_Number_EW", "Elo_R_EW_Before", "Elo_R_Pair_EW_EventStart"),
    }
    for _, (pair_col, before_col, out_col) in pair_before_cols.items():
        if pair_col in df.columns and before_col in df.columns:
            df = df.with_columns(
                pl.col(before_col).first().over(["session_id", pair_col]).alias(out_col)
            )

    # Player end
    seat_after_cols = {
        "N": ("Player_ID_N", "Elo_R_N", "Elo_R_Player_N_EventEnd"),
        "S": ("Player_ID_S", "Elo_R_S", "Elo_R_Player_S_EventEnd"),
        "E": ("Player_ID_E", "Elo_R_E", "Elo_R_Player_E_EventEnd"),
        "W": ("Player_ID_W", "Elo_R_W", "Elo_R_Player_W_EventEnd"),
    }
    for _, (pid_col, after_col, out_col) in seat_after_cols.items():
        if pid_col in df.columns and after_col in df.columns:
            df = df.with_columns(
                pl.col(after_col).last().over(["session_id", pid_col]).alias(out_col)
            )

    # Pair end
    pair_after_cols = {
        "NS": ("Pair_Number_NS", "Elo_R_NS", "Elo_R_Pair_NS_EventEnd"),
        "EW": ("Pair_Number_EW", "Elo_R_EW", "Elo_R_Pair_EW_EventEnd"),
    }
    for _, (pair_col, after_col, out_col) in pair_after_cols.items():
        if pair_col in df.columns and after_col in df.columns:
            df = df.with_columns(
                pl.col(after_col).last().over(["session_id", pair_col]).alias(out_col)
            )

    print(f"Event start/end Elo columns: {time.time()-t:.3f} seconds")
    return df


class DealAugmenter:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def _time_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        t = time.time()
        result = func(*args, **kwargs)
        print(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _add_default_columns(self) -> None:
        if 'group_id' not in self.df.columns:
            self.df = self.df.with_columns(pl.lit(0).alias('group_id'))
        if 'session_id' not in self.df.columns:
            self.df = self.df.with_columns(pl.lit(0).alias('session_id'))
        if 'section_name' not in self.df.columns:
            self.df = self.df.with_columns(pl.lit('').alias('section_name'))

    def _create_dealer(self) -> None:
        # Assert required columns exist
        assert 'Board' in self.df.columns, "Required column 'Board' not found in DataFrame"
        
        t = time.time()
        if 'Dealer' not in self.df.columns:
            self.df = create_dealer_column(self.df)
        print(f"create Dealer: time:{time.time()-t} seconds")
        
        # Assert column was created
        assert 'Dealer' in self.df.columns, "Column 'Dealer' was not created"

    def _create_vulnerability(self) -> None:
        # Assert required columns exist for iVul creation
        assert ('Vul' in self.df.columns or 'Board' in self.df.columns), "Required column 'Vul' or 'Board' not found in DataFrame"
        
        t = time.time()
        if 'iVul' not in self.df.columns:
            if 'Vul' in self.df.columns:
                self.df = create_ivul_from_vul(self.df)
            else:
                self.df = create_ivul_from_board(self.df)
        print(f"create iVul: time:{time.time()-t} seconds")
        
        # Assert iVul column was created
        assert 'iVul' in self.df.columns, "Column 'iVul' was not created"
        
        t = time.time()
        if 'Vul' not in self.df.columns:
            self.df = create_vul_from_ivul(self.df)
        print(f"create Vul from iVul: time:{time.time()-t} seconds")
        
        assert 'Vul' in self.df.columns, "Column 'Vul' was not created"
        
        t = time.time()
        if 'Vul_NS' not in self.df.columns:
            self.df = create_vul_ns_ew_columns(self.df)
        print(f"create Vul_NS/EW: time:{time.time()-t} seconds")
        
        # Assert Vul_NS and Vul_EW columns were created
        assert 'Vul_NS' in self.df.columns, "Column 'Vul_NS' was not created"
        assert 'Vul_EW' in self.df.columns, "Column 'Vul_EW' was not created"

    def _create_hand_columns(self) -> None:
        self.df = self._time_operation("create_hand_nesw_columns", create_hand_nesw_columns, self.df)
        self.df = self._time_operation("create_suit_nesw_columns", create_suit_nesw_columns, self.df)
        self.df = self._time_operation("create_hands_lists_column", create_hands_lists_column, self.df)

    def perform_deal_augmentations(self) -> pl.DataFrame:
        """Main method to perform all deal augmentations"""
        t_start = time.time()
        print(f"Starting deal augmentations")
        
        self._add_default_columns()
        self._create_dealer()
        self._create_vulnerability()
        self._create_hand_columns()
        
        print(f"Deal augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df


class HandAugmenter:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.suit_quality_criteria = {
            "Biddable": lambda sl, hcp: sl.ge(5) | (sl.eq(4) & hcp.ge(3)),
            "Rebiddable": lambda sl, hcp: sl.ge(6) | (sl.eq(5) & hcp.ge(3)),
            "Twice_Rebiddable": lambda sl, hcp: sl.ge(7) | (sl.eq(6) & hcp.ge(3)),
            "Strong_Rebiddable": lambda sl, hcp: sl.ge(6) & hcp.ge(9),
            "Solid": lambda sl, hcp: hcp.ge(9),  # todo: 6 card requires ten
        }
        self.stopper_criteria = {
            "At_Best_Partial_Stop_In": lambda sl, hcp: (sl + hcp).lt(4),
            "Partial_Stop_In": lambda sl, hcp: (sl + hcp).ge(4),
            "Likely_Stop_In": lambda sl, hcp: (sl + hcp).ge(5),
            "Stop_In": lambda sl, hcp: hcp.ge(4) | (sl + hcp).ge(6),
            "At_Best_Stop_In": lambda sl, hcp: (sl + hcp).ge(7),
            "Two_Stops_In": lambda sl, hcp: (sl + hcp).ge(8),
        }

    def _time_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        t = time.time()
        result = func(*args, **kwargs)
        print(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _create_cards(self) -> None:
        # Assert required columns exist
        assert 'PBN' in self.df.columns, "Required column 'PBN' not found in DataFrame"
        
        if 'C_NSA' not in self.df.columns:
            self.df = self._time_operation("create C_NSA", DealToCards, self.df)
        
        # Assert columns were created
        for direction in "NESW":
            for suit in "SHDC":
                for rank in 'AKQJT98765432':
                    assert f'C_{direction}{suit}{rank}' in self.df.columns, f"Column 'C_{direction}{suit}{rank}' was not created"

    def _create_hcp(self) -> None:
        # Assert required columns exist
        for direction in "NESW":
            for suit in "SHDC":
                for rank in 'AKQJT98765432':
                    assert f'C_{direction}{suit}{rank}' in self.df.columns, f"Required column 'C_{direction}{suit}{rank}' not found in DataFrame"
        
        if 'HCP_N_C' not in self.df.columns:
            self.df = self._time_operation("create HCP", CardsToHCP, self.df)
        
        # Assert columns were created
        for direction in "NESW":
            for suit in "SHDC":
                assert f'HCP_{direction}_{suit}' in self.df.columns, f"Column 'HCP_{direction}_{suit}' was not created"

    def _create_quick_tricks(self) -> None:
        # Assert required columns exist
        for direction in "NESW":
            for suit in "SHDC":
                assert f'Suit_{direction}_{suit}' in self.df.columns, f"Required column 'Suit_{direction}_{suit}' not found in DataFrame"
        
        if 'QT_N_C' not in self.df.columns:
            self.df = self._time_operation("create QT", CardsToQuickTricks, self.df)
        
        # Assert columns were created
        for direction in "NESW":
            for suit in "SHDC":
                assert f'QT_{direction}_{suit}' in self.df.columns, f"Column 'QT_{direction}_{suit}' was not created"

    def _create_suit_lengths(self) -> None:
        # Assert required columns exist
        for direction in "NESW":
            for suit in "SHDC":
                assert f'Suit_{direction}_{suit}' in self.df.columns, f"Required column 'Suit_{direction}_{suit}' not found in DataFrame"
        
        t = time.time()
        if 'SL_N_C' not in self.df.columns:
            self.df = create_suit_length_columns(self.df)
        print(f"create SL_[NESW]_[SHDC]: time:{time.time()-t} seconds")
        
        # Assert columns were created
        for direction in "NESW":
            for suit in "SHDC":
                assert f'SL_{direction}_{suit}' in self.df.columns, f"Column 'SL_{direction}_{suit}' was not created"

    def _create_pair_suit_lengths(self) -> None:
        # Assert required columns exist
        for pair in ['NS', 'EW']:
            for suit in "SHDC":
                assert f'SL_{pair[0]}_{suit}' in self.df.columns, f"Required column 'SL_{pair[0]}_{suit}' not found in DataFrame"
                assert f'SL_{pair[1]}_{suit}' in self.df.columns, f"Required column 'SL_{pair[1]}_{suit}' not found in DataFrame"
        
        t = time.time()
        if 'SL_NS_C' not in self.df.columns:
            self.df = create_pair_suit_length_columns(self.df)
        print(f"create SL_(NS|EW)_[SHDC]: time:{time.time()-t} seconds")
        
        # Assert columns were created
        for pair in ['NS', 'EW']:
            for suit in "SHDC":
                assert f'SL_{pair}_{suit}' in self.df.columns, f"Column 'SL_{pair}_{suit}' was not created"

    def _create_suit_length_arrays(self) -> None:
        # Assert required columns exist
        for direction in 'NESW':
            for suit in 'CDHS':
                assert f'SL_{direction}_{suit}' in self.df.columns, f"Required column 'SL_{direction}_{suit}' not found in DataFrame"
        
        if 'SL_N_CDHS' not in self.df.columns:
            for d in 'NESW':
                t = time.time()
                self.df = create_suit_length_arrays_for_direction(self.df, d)
                print(f"create SL_{d} arrays: time:{time.time()-t} seconds")
        
        # Assert columns were created
        for direction in 'NESW':
            assert f'SL_{direction}_CDHS' in self.df.columns, f"Column 'SL_{direction}_CDHS' was not created"
            assert f'SL_{direction}_CDHS_SJ' in self.df.columns, f"Column 'SL_{direction}_CDHS_SJ' was not created"
            assert f'SL_{direction}_ML' in self.df.columns, f"Column 'SL_{direction}_ML' was not created"
            assert f'SL_{direction}_ML_SJ' in self.df.columns, f"Column 'SL_{direction}_ML_SJ' was not created"
            assert f'SL_{direction}_ML_I' in self.df.columns, f"Column 'SL_{direction}_ML_I' was not created"
            assert f'SL_{direction}_ML_I_SJ' in self.df.columns, f"Column 'SL_{direction}_ML_I_SJ' was not created"

    def _create_distribution_points(self) -> None:
        # Assert required columns exist
        for direction in "NESW":
            for suit in "SHDC":
                assert f'SL_{direction}_{suit}' in self.df.columns, f"Required column 'SL_{direction}_{suit}' not found in DataFrame"
        
        t = time.time()
        if 'DP_N_C' not in self.df.columns:
            # Calculate individual suit DPs
            self.df = create_distribution_point_columns(self.df)
            self.df = create_direction_summary_columns(self.df)
        print(f"create DP columns: time:{time.time()-t} seconds")
        
        # Assert columns were created
        for direction in "NESW":
            for suit in "SHDC":
                assert f'DP_{direction}_{suit}' in self.df.columns, f"Column 'DP_{direction}_{suit}' was not created"
            assert f'DP_{direction}' in self.df.columns, f"Column 'DP_{direction}' was not created"
        assert 'DP_NS' in self.df.columns, "Column 'DP_NS' was not created"
        assert 'DP_EW' in self.df.columns, "Column 'DP_EW' was not created"
        for suit in 'SHDC':
            assert f'DP_NS_{suit}' in self.df.columns, f"Column 'DP_NS_{suit}' was not created"
            assert f'DP_EW_{suit}' in self.df.columns, f"Column 'DP_EW_{suit}' was not created"

    def _create_total_points(self) -> None:
        # Assert required columns exist
        for direction in 'NESW':
            for suit in 'SHDC':
                assert f'HCP_{direction}_{suit}' in self.df.columns, f"Required column 'HCP_{direction}_{suit}' not found in DataFrame"
                assert f'DP_{direction}_{suit}' in self.df.columns, f"Required column 'DP_{direction}_{suit}' not found in DataFrame"
        
        t = time.time()
        if 'Total_Points_N_C' not in self.df.columns:
            print("Todo: Don't forget to adjust Total_Points for singleton king and doubleton queen.")
            self.df = create_total_point_columns(self.df)
        print(f"create Total_Points: time:{time.time()-t} seconds")
        
        # Assert columns were created
        for direction in 'NESW':
            for suit in 'SHDC':
                assert f'Total_Points_{direction}_{suit}' in self.df.columns, f"Column 'Total_Points_{direction}_{suit}' was not created"
            assert f'Total_Points_{direction}' in self.df.columns, f"Column 'Total_Points_{direction}' was not created"
        assert 'Total_Points_NS' in self.df.columns, "Column 'Total_Points_NS' was not created"
        assert 'Total_Points_EW' in self.df.columns, "Column 'Total_Points_EW' was not created"

    def _create_max_suit_lengths(self) -> None:
        # Assert required columns exist
        for direction in ['NS', 'EW']:
            for suit in ['S', 'H', 'D', 'C']:
                assert f'SL_{direction[0]}_{suit}' in self.df.columns, f"Required column 'SL_{direction[0]}_{suit}' not found in DataFrame"
                assert f'SL_{direction[1]}_{suit}' in self.df.columns, f"Required column 'SL_{direction[1]}_{suit}' not found in DataFrame"
        
        if 'SL_Max_NS' not in self.df.columns:
            t = time.time()
            self.df = create_max_suit_length_columns(self.df)
            print(f"create SL_Max columns: time:{time.time()-t} seconds")
        
        # Assert columns were created
        for direction in ['NS', 'EW']:
            assert f'SL_Max_{direction}' in self.df.columns, f"Column 'SL_Max_{direction}' was not created"

    def _create_quality_indicators(self) -> None:
        # Assert required columns exist
        for direction in "NESW":
            for suit in "SHDC":
                assert f'SL_{direction}_{suit}' in self.df.columns, f"Required column 'SL_{direction}_{suit}' not found in DataFrame"
                assert f'HCP_{direction}_{suit}' in self.df.columns, f"Required column 'HCP_{direction}_{suit}' not found in DataFrame"
        
        t = time.time()
        self.df = create_quality_indicator_columns(self.df, self.suit_quality_criteria, self.stopper_criteria)
        print(f"create quality indicators: time:{time.time()-t} seconds")
        
        # Assert columns were created
        for direction in "NESW":
            for suit in "SHDC":
                for series_type in {**self.suit_quality_criteria, **self.stopper_criteria}.keys():
                    assert f'{series_type}_{direction}_{suit}' in self.df.columns, f"Column '{series_type}_{direction}_{suit}' was not created"
        assert 'Forcing_One_Round' in self.df.columns, "Column 'Forcing_One_Round' was not created"
        assert 'Opponents_Cannot_Play_Undoubled_Below_2N' in self.df.columns, "Column 'Opponents_Cannot_Play_Undoubled_Below_2N' was not created"
        assert 'Forcing_To_2N' in self.df.columns, "Column 'Forcing_To_2N' was not created"
        assert 'Forcing_To_3N' in self.df.columns, "Column 'Forcing_To_3N' was not created"

    def _create_balanced_indicators(self) -> None:
        # Assert required columns exist
        for direction in 'NESW':
            assert f'SL_{direction}_ML_SJ' in self.df.columns, f"Required column 'SL_{direction}_ML_SJ' not found in DataFrame"
            assert f'SL_{direction}_C' in self.df.columns, f"Required column 'SL_{direction}_C' not found in DataFrame"
            assert f'SL_{direction}_D' in self.df.columns, f"Required column 'SL_{direction}_D' not found in DataFrame"
        
        t = time.time()
        self.df = create_balanced_indicator_columns(self.df)
        print(f"create balanced indicators: time:{time.time()-t} seconds")
        
        # Assert columns were created
        for direction in 'NESW':
            assert f'Balanced_{direction}' in self.df.columns, f"Column 'Balanced_{direction}' was not created"

    def perform_hand_augmentations(self) -> pl.DataFrame:
        """Main method to perform all hand augmentations"""
        t_start = time.time()
        print(f"Starting hand augmentations")
        
        self._create_cards()
        self._create_hcp()
        self._create_quick_tricks()
        self._create_suit_lengths()
        self._create_pair_suit_lengths()
        self._create_suit_length_arrays()
        self._create_distribution_points()
        self._create_total_points()
        self._create_max_suit_lengths()
        self._create_quality_indicators()
        self._create_balanced_indicators()
        
        print(f"Hand augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df


class DD_SD_Augmenter:
    def __init__(self, df: pl.DataFrame, hrs_cache_df: pl.DataFrame, sd_productions: int = 40, max_adds: Optional[int] = None, output_progress: Optional[bool] = True, progress: Optional[Any] = None, lock_func: Optional[Callable[..., pl.DataFrame]] = None):
        self.df = df
        self.hrs_cache_df = hrs_cache_df
        self.sd_productions = sd_productions
        self.max_adds = max_adds
        self.output_progress = output_progress
        self.progress = progress
        self.lock_func = lock_func

    def _time_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        t = time.time()
        result = func(*args, **kwargs)
        print(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _process_scores_and_tricks(self) -> pl.DataFrame:
        all_scores_d, scores_d, scores_df = self._time_operation("calculate_scores", calculate_scores)
        
        # Calculate double dummy scores first
        dd_df, unique_dd_tables_d = self._time_operation(
            "calculate_dd_scores", 
            calculate_dd_scores, 
            self.df, self.hrs_cache_df, self.max_adds, self.output_progress, self.progress
        )

        if not dd_df.is_empty():
            self.hrs_cache_df = update_hrs_cache_df(self.hrs_cache_df, dd_df)
        
        # Calculate par scores using the double dummy results
        par_df = self._time_operation(
            "calculate_par_scores",
            calculate_par_scores,
            self.df, self.hrs_cache_df, unique_dd_tables_d
        )

        if not par_df.is_empty():
            self.hrs_cache_df = update_hrs_cache_df(self.hrs_cache_df, par_df)

        sd_dfs_d, sd_df = self._time_operation(
            "calculate_sd_probs",
            calculate_sd_probs,
            self.df, self.hrs_cache_df, self.sd_productions, self.max_adds, self.progress
        )

        if not sd_df.is_empty():
            self.hrs_cache_df = update_hrs_cache_df(self.hrs_cache_df, sd_df)

        self.df = self.df.join(self.hrs_cache_df, on=['PBN','Dealer','Vul'], how='inner') # on='PBN', how='left' or on=['PBN','Dealer','Vul'], how='inner'

        # create DD_(NS|EW)_[SHDCN] which is the max of NS or EW for each strain
        self.df = self.df.with_columns(
            pl.max_horizontal(f"DD_{pair[0]}_{strain}",f"DD_{pair[1]}_{strain}").alias(f"DD_{pair}_{strain}")
            for pair in ['NS','EW']
            for strain in "SHDCN"
        )

        self.df = self._time_operation(
            "calculate_sd_expected_values",
            calculate_sd_expected_values,
            self.df, scores_df
        )

        best_contracts_df = create_best_contracts(self.df)
        assert self.df.height == best_contracts_df.height, f"{self.df.height} != {best_contracts_df.height}"
        self.df = pl.concat([self.df, best_contracts_df], how='horizontal')
        del best_contracts_df        

        return self.df, self.hrs_cache_df #, scores_df

    def perform_dd_sd_augmentations(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        if self.lock_func is None:
            self.df, self.hrs_cache_df = self.perform_dd_sd_augmentations_queue_up()
        else:
            self.df, self.hrs_cache_df = self.lock_func(self, self.perform_dd_sd_augmentations_queue_up)
        return self.df, self.hrs_cache_df

    def perform_dd_sd_augmentations_queue_up(self) -> pl.DataFrame:
        """Main method to perform all double dummy and single dummy augmentations"""
        t_start = time.time()
        print(f"Starting DD/SD trick augmentations")
        
        self.df, self.hrs_cache_df = self._process_scores_and_tricks()
        
        print(f"DD/SD trick augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df, self.hrs_cache_df


class AllContractsAugmenter:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def _time_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        t = time.time()
        result = func(*args, **kwargs)
        print(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _create_ct_types(self) -> None:
        if 'CT_N_C' not in self.df.columns:
            ct_columns = [
                pl.when(pl.col(f"DD_{direction}_{strain}") < 7).then(pl.lit("Pass"))
                .when((pl.col(f"DD_{direction}_{strain}") == 11) & (strain in ['C', 'D'])).then(pl.lit("Game"))
                .when((pl.col(f"DD_{direction}_{strain}").is_in([10,11])) & (strain in ['H', 'S'])).then(pl.lit("Game"))
                .when((pl.col(f"DD_{direction}_{strain}").is_in([9,10,11])) & (strain == 'N')).then(pl.lit("Game"))
                .when(pl.col(f"DD_{direction}_{strain}") == 12).then(pl.lit("SSlam"))
                .when(pl.col(f"DD_{direction}_{strain}") == 13).then(pl.lit("GSlam"))
                .otherwise(pl.lit("Partial"))
                .alias(f"CT_{direction}_{strain}")
                for direction in "NESW"
                for strain in "SHDCN"
            ]
            self.df = self._time_operation(
                "create CT columns",
                lambda df: df.with_columns(ct_columns),
                self.df
            )

    # create CT boolean columns from CT columns
    def _create_ct_booleans(self) -> None:
        if 'CT_N_C_Game' not in self.df.columns:
            ct_boolean_columns = [
                pl.col(f"CT_{direction}_{strain}").eq(pl.lit(contract_type))
                .alias(f"CT_{direction}_{strain}_{contract_type}")
                for direction in "NESW"
                for strain in "SHDCN"
                for contract_type in ["Pass","Game","SSlam","GSlam","Partial"]
            ]
            self.df = self._time_operation(
                "create CT boolean columns",
                lambda df: df.with_columns(ct_boolean_columns),
                self.df
            )
            
        # Create CT boolean columns for pair directions (NS and EW)
        if 'CT_NS_C_Game' not in self.df.columns:
            ct_pair_boolean_columns = [
                (pl.col(f"CT_{pair_direction[0]}_{strain}_{contract_type}") | 
                 pl.col(f"CT_{pair_direction[1]}_{strain}_{contract_type}"))
                .alias(f"CT_{pair_direction}_{strain}_{contract_type}")
                for pair_direction in ["NS", "EW"]
                for strain in "SHDCN"
                for contract_type in ["Pass","Game","SSlam","GSlam","Partial"]
            ]
            self.df = self._time_operation(
                "create CT pair boolean columns",
                lambda df: df.with_columns(ct_pair_boolean_columns),
                self.df
            )

    def perform_all_contracts_augmentations(self) -> pl.DataFrame:
        """Main method to perform AllContracts augmentations"""
        t_start = time.time()
        print(f"Starting AllContracts augmentations")

        self._create_ct_types()
        self._create_ct_booleans()

        print(f"AllContracts augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df


class FinalContractAugmenter:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        # note that polars exprs are allowed in dict values.
        self.vul_conditions = {
            'NS': pl.col('Vul').is_in(['N_S', 'Both']),
            'EW': pl.col('Vul').is_in(['E_W', 'Both'])
        }

    def _time_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        t = time.time()
        result = func(*args, **kwargs)
        print(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _process_contract_columns(self) -> None:
        assert 'Player_Name_N' in self.df.columns
        self.df = self._time_operation(
            "convert_contract_to_contract",
            lambda df: df.with_columns(
                pl.Series('Contract', convert_contract_to_contract(df), pl.String, strict=False)
            ),
            self.df
        )
        assert 'Player_Name_N' in self.df.columns

        # todo: move this section to contract established class
        self.df = self._time_operation(
            "convert_contract_parts",
            lambda df: convert_contract_to_dbl(
                convert_contract_to_strain(
                    convert_contract_to_level(
                        convert_contract_to_vul_declarer(
                            convert_contract_to_pair_declarer(
                                convert_contract_to_declarer(df)
                            )
                        )
                    )
                )
            ),
            self.df
        )

        # todo: move this to table established class? to_ml() func?
        assert 'LHO_Direction' not in self.df.columns
        assert 'Dummy_Direction' not in self.df.columns
        assert 'RHO_Direction' not in self.df.columns
        print(self.df['Declarer_Direction'].value_counts())
        self.df = self._time_operation(
            "convert_declarer_to_directions",
            lambda df: df.with_columns([
                pl.col('Declarer_Direction').replace_strict(declarer_to_LHO_d).alias('LHO_Direction'),
                pl.col('Declarer_Direction').replace_strict(declarer_to_dummy_d).alias('Dummy_Direction'),
                pl.col('Declarer_Direction').replace_strict(declarer_to_RHO_d).alias('RHO_Direction'),
            ]),
                self.df
            )

    # create ContractType column using final contract
    def _create_contract_types(self) -> None:
        # Assert required columns exist
        assert 'Contract' in self.df.columns, "Required column 'Contract' not found in DataFrame"
        assert 'BidLvl' in self.df.columns, "Required column 'BidLvl' not found in DataFrame"
        assert 'BidSuit' in self.df.columns, "Required column 'BidSuit' not found in DataFrame"
        
        self.df = self._time_operation(
            "create_contract_types",
            create_contract_type_column,
            self.df
        )
        
        # Assert column was created
        assert 'ContractType' in self.df.columns, "Column 'ContractType' was not created"

    def _create_player_names(self) -> None:
        assert 'Player_Name_N' in self.df.columns, "Required column 'Player_Name_N' not found in DataFrame"
        assert 'Player_Name_E' in self.df.columns, "Required column 'Player_Name_E' not found in DataFrame"
        assert 'Player_Name_S' in self.df.columns, "Required column 'Player_Name_S' not found in DataFrame"
        assert 'Player_Name_W' in self.df.columns, "Required column 'Player_Name_W' not found in DataFrame"
        
        self.df = self._time_operation(
            "create_player_names",
            create_player_names,
            self.df
        )
        
        # Assert column was created
        assert 'Player_Names_NS' in self.df.columns, "Column 'Player_Names_NS' was not created"
        assert 'Player_Names_EW' in self.df.columns, "Column 'Player_Names_EW' was not created"

    # todo: move this to contract established class
    def _create_declarer_columns(self) -> None:
        # Assert required columns exist
        assert 'Declarer_Direction' in self.df.columns, "Required column 'Declarer_Direction' not found in DataFrame"
        assert 'Player_Name_N' in self.df.columns, "Required column 'Player_Name_N' not found in DataFrame"
        assert 'Player_Name_E' in self.df.columns, "Required column 'Player_Name_E' not found in DataFrame"
        assert 'Player_Name_S' in self.df.columns, "Required column 'Player_Name_S' not found in DataFrame"
        assert 'Player_Name_W' in self.df.columns, "Required column 'Player_Name_W' not found in DataFrame"
        assert 'Player_ID_N' in self.df.columns, "Required column 'Player_ID_N' not found in DataFrame"
        assert 'Player_ID_E' in self.df.columns, "Required column 'Player_ID_E' not found in DataFrame"
        assert 'Player_ID_S' in self.df.columns, "Required column 'Player_ID_S' not found in DataFrame"
        assert 'Player_ID_W' in self.df.columns, "Required column 'Player_ID_W' not found in DataFrame"
        
        self.df = self._time_operation(
            "convert_declarer_columns",
            create_declarer_columns,
            self.df
        )
        
        # Assert columns were created
        assert 'Declarer_Name' in self.df.columns, "Column 'Declarer_Name' was not created"
        assert 'Declarer_ID' in self.df.columns, "Column 'Declarer_ID' was not created"

    # todo: move this to contract established class
    def _create_result_columns(self) -> None:
        # Assert required columns exist
        assert 'Contract' in self.df.columns, "Required column 'Contract' not found in DataFrame"
        
        if 'Result' not in self.df.columns:
            if 'Tricks' in self.df.columns:
                self.df = self._time_operation(
                    "convert_tricks_to_result",
                    create_result_column_from_tricks,
                    self.df
                )
            else:
                # todo: create assert that result is in Contract.
                self.df = self._time_operation(
                    "convert_contract_to_result",
                    create_result_column_from_contract,
                    self.df
                )

        if 'Tricks' not in self.df.columns:
            self.df = self._time_operation(
                "convert_contract_to_tricks",
                create_tricks_column_from_contract,
                self.df
            )
        
        # Assert columns were created
        assert 'Result' in self.df.columns, "Column 'Result' was not created"
        assert 'Tricks' in self.df.columns, "Column 'Tricks' was not created"

    # # todo: move this to contract established class
    # def _create_dd_columns(self) -> None:
    #     if 'DD_Tricks' not in self.df.columns:
    #         self.df = self._time_operation(
    #             "convert_contract_to_DD_Tricks",
    #             lambda df: df.with_columns([
    #                 pl.Series('DD_Tricks', convert_contract_to_DD_Tricks(df), pl.UInt8, strict=False),
    #                 pl.Series('DD_Tricks_Dummy', convert_contract_to_DD_Tricks_Dummy(df), pl.UInt8, strict=False),
    #             ]),
    #             self.df
    #         )

    #     # todo: move this to its own def.
    #     if 'DD_Score_NS' not in self.df.columns:
    #         self.df = self._time_operation(
    #             "convert_contract_to_DD_Score_Ref",
    #             convert_contract_to_DD_Score_Ref,
    #             self.df
    #         )

    # todo: move this to contract established class
    def _create_dd_columns(self) -> None:
        # Cache the scores calculation - this is static and expensive
        if not hasattr(self, '_cached_scores'):
            self._cached_scores = calculate_scores_cached()
        
        all_scores_d, scores_d, scores_df = self._cached_scores
        
        if 'DD_Tricks' not in self.df.columns:
            self.df = self._time_operation(
                "create_dd_tricks_column_optimized",
                create_dd_tricks_column_optimized,
                self.df
            )

        if 'DD_Tricks_Dummy' not in self.df.columns:
            self.df = self._time_operation(
                "create_dd_tricks_dummy_column_optimized",
                create_dd_tricks_dummy_column_optimized,
                self.df
            )

        # todo: move this to its own def.
        if 'DD_Score_NS' not in self.df.columns:
            self.df = self._time_operation(
                "create_dd_score_columns_optimized",
                create_dd_score_columns_optimized,
                self.df,
                scores_d
            )

    # todo: move this to contract established class
    def _create_ev_columns(self) -> None:
        max_expressions = []
        for pd in ['NS', 'EW']:
            max_expressions.extend(self._create_ev_expressions_for_pair(pd))

        self.df = self.df.with_columns(max_expressions)

        ev_columns = f'^EV_Max_(NS|EW)$'
        max_expr, col_expr = max_horizontal_and_col(self.df, ev_columns)
        self.df = self.df.with_columns([
            max_expr.alias('EV_Max'),
            col_expr.alias('EV_Max_Col'),
        ])
        
        self.df = self._time_operation(
            "create_ev_columns",
            lambda df: df.with_columns([
                pl.when(pl.col('Declarer_Pair_Direction').eq('NS')).then(pl.col('EV_Max_NS')).otherwise(pl.col('EV_Max_EW')).alias('EV_Max_Declarer'),
                pl.when(pl.col('Declarer_Pair_Direction').eq('NS')).then(pl.col('EV_Max_Col_NS')).otherwise(pl.col('EV_Max_Col_EW')).alias('EV_Max_Col_Declarer'),
            ]),
            self.df
        )


    # todo: move this to contract established class
    def _create_ev_expressions_for_pair(self, pd: str) -> List:
        expressions = []
        expressions.extend(self._create_basic_ev_expressions(pd))
        
        for dd in pd:
            expressions.extend(self._create_declarer_ev_expressions(pd, dd))
            
            for s in 'SHDCN':
                expressions.extend(self._create_strain_ev_expressions(pd, dd, s))
                
                for l in range(1, 8):
                    expressions.extend(self._create_level_ev_expressions(pd, dd, s, l))
        
        return expressions

    # todo: move this to contract established class
    def _create_basic_ev_expressions(self, pd: str) -> List:
        return [
            pl.when(self.vul_conditions[pd])
              .then(pl.col(f'EV_{pd}_V_Max'))
              .otherwise(pl.col(f'EV_{pd}_NV_Max'))
              .alias(f'EV_Max_{pd}'),
            pl.when(self.vul_conditions[pd])
              .then(pl.col(f'EV_{pd}_V_Max_Col'))
              .otherwise(pl.col(f'EV_{pd}_NV_Max_Col'))
              .alias(f'EV_Max_Col_{pd}')
        ]

    # todo: move this to contract established class
    def _create_declarer_ev_expressions(self, pd: str, dd: str) -> List:
        return [
            pl.when(self.vul_conditions[pd])
              .then(pl.col(f'EV_{pd}_{dd}_V_Max'))
              .otherwise(pl.col(f'EV_{pd}_{dd}_NV_Max'))
              .alias(f'EV_{pd}_{dd}_Max'),
            pl.when(self.vul_conditions[pd])
              .then(pl.col(f'EV_{pd}_{dd}_V_Max_Col'))
              .otherwise(pl.col(f'EV_{pd}_{dd}_NV_Max_Col'))
              .alias(f'EV_{pd}_{dd}_Max_Col')
        ]

    # todo: move this to contract established class
    def _create_strain_ev_expressions(self, pd: str, dd: str, s: str) -> List:
        return [
            pl.when(self.vul_conditions[pd])
              .then(pl.col(f'EV_{pd}_{dd}_{s}_V_Max'))
              .otherwise(pl.col(f'EV_{pd}_{dd}_{s}_NV_Max'))
              .alias(f'EV_{pd}_{dd}_{s}_Max'),
            pl.when(self.vul_conditions[pd])
              .then(pl.col(f'EV_{pd}_{dd}_{s}_V_Max_Col'))
              .otherwise(pl.col(f'EV_{pd}_{dd}_{s}_NV_Max_Col'))
              .alias(f'EV_{pd}_{dd}_{s}_Max_Col')
        ]

    # todo: move this to contract established class
    def _create_level_ev_expressions(self, pd: str, dd: str, s: str, l: int) -> List:
        return [
            pl.when(self.vul_conditions[pd])
            .then(pl.col(f'EV_{pd}_{dd}_{s}_{l}_V'))
            .otherwise(pl.col(f'EV_{pd}_{dd}_{s}_{l}_NV'))
            .alias(f'EV_{pd}_{dd}_{s}_{l}')
        ]

    def _create_score_columns(self) -> None:

        if 'Score' not in self.df.columns:
            if 'Score_NS' in self.df.columns:
                assert 'Score_EW' in self.df.columns, "Score_EW does not exist but Score_NS does."
                self.df = self._time_operation(
                    "convert_score_nsew_to_score",
                    lambda df: df.with_columns([
                        pl.when(pl.col('Declarer_Pair_Direction').eq('NS'))
                        .then(pl.col('Score_NS'))
                        .otherwise(pl.col('Score_EW')) # assuming Score_EW can be a score, 0 (PASS) or None?
                        .alias('Score'),
                    ]),
                    self.df
                )
            else:
                # neither 'Score' nor 'Score_NS' exist.
                all_scores_d, scores_d, scores_df = calculate_scores()
                self.df = self._time_operation(
                    "convert_contract_to_score",
                    lambda df: df.with_columns([
                        pl.struct(['BidLvl', 'BidSuit', 'Tricks', 'Vul_Declarer', 'Dbl'])
                            .map_elements(lambda x: all_scores_d.get(tuple(x.values()),None), # default of None should only occur in the case of director's adjustment.
                                        return_dtype=pl.Int16)
                            .alias('Score'),
                    ]),
                    self.df
                )

        if 'Score_NS' not in self.df.columns:
            self.df = self._time_operation(
                "convert_score_to_score",
                # lambda df: df.with_columns([
                #     pl.col('Score').alias('Score_NS'),
                #     pl.col('Score').neg().alias('Score_EW')
                # ]),
                lambda df: df.with_columns([
                    pl.when(pl.col('Declarer_Pair_Direction').eq('NS'))
                    .then(pl.col('Score'))
                    .otherwise(-pl.col('Score'))
                    .alias('Score_NS'),
                    pl.when(pl.col('Declarer_Pair_Direction').eq('EW'))
                    .then(pl.col('Score'))
                    .otherwise(-pl.col('Score'))
                    .alias('Score_EW')
                ]),
                self.df
            )

    def _create_score_diff_columns(self) -> None:
        # First create the initial diff columns
        self.df = self._time_operation(
            "create_basic_diff_columns",
            lambda df: df.with_columns([
                pl.Series('Par_Diff_NS', (df['Score_NS']-df['Par_NS']), pl.Int16),
                pl.Series('Par_Diff_EW', (df['Score_EW']-df['Par_EW']), pl.Int16),
                pl.Series('DD_Tricks_Diff', (df['Tricks'].cast(pl.Int8)-df['DD_Tricks'].cast(pl.Int8)), pl.Int8, strict=False),
                pl.Series('EV_Max_Diff_NS', df['Score_NS'] - df['EV_Max_NS'], pl.Float32),
                pl.Series('EV_Max_Diff_EW', df['Score_EW'] - df['EV_Max_EW'], pl.Float32),
            ]),
            self.df
        )

        # todo: move into above lambda.
        # Then create Par_Diff_EW using the now-existing Par_Diff_NS
        self.df = self._time_operation(
            "create_parscore_diff_ew",
            lambda df: df.with_columns([
                pl.Series('Par_Diff_EW', -df['Par_Diff_NS'], pl.Int16)
            ]),
            self.df
        )

    # todo: would be interesting to enhance this for any contracts and then move into all contract class
    def _create_lott(self) -> None:
        if 'LoTT' not in self.df.columns:
            self.df = self._time_operation("create LoTT", calculate_LoTT, self.df)

    # def _perform_legacy_renames(self) -> None:
    #     self.df = self._time_operation(
    #         "perform legacy renames",
    #         Perform_Legacy_Renames,
    #         self.df
    #     )


    def _create_position_columns(self) -> None:
        # these augmentations should not already exist.
        assert 'Direction_OnLead' not in self.df.columns
        assert 'Opponent_Pair_Direction' not in self.df.columns
        assert 'Direction_Dummy' not in self.df.columns
        assert 'OnLead' not in self.df.columns
        assert 'Direction_NotOnLead' not in self.df.columns
        assert 'Dummy' not in self.df.columns
        assert 'Defender_Par_GE' not in self.df.columns
        assert 'EV_Score_Col_Declarer' not in self.df.columns
        assert 'Score_Declarer' not in self.df.columns
        assert 'Par_Declarer' not in self.df.columns
 
        self.df = self._time_operation(
            "create position columns",
            lambda df: df.with_columns([
                pl.struct(['Declarer_Direction', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(
                    lambda r: None if r['Declarer_Direction'] is None else r[f'Player_ID_{r["Declarer_Direction"]}'],
                    return_dtype=pl.String
                ).alias('Declarer'),
            ])
            .with_columns([
                pl.col('Declarer_Direction').replace_strict(NextPosition).alias('Direction_OnLead'),
            ])
            .with_columns([
                pl.concat_str([
                    pl.lit('EV'),
                    pl.col('Declarer_Pair_Direction'),
                    pl.col('Declarer_Direction'),
                    pl.col('BidSuit'),
                    pl.col('BidLvl').cast(pl.String),
                ], separator='_').alias('EV_Score_Col_Declarer'),
                
                # pl.when(pl.col('Declarer_Pair_Direction').eq(pl.lit('NS')))
                # .then(pl.col('Score_NS'))
                # .otherwise(pl.col('Score_EW'))
                # .alias('Score_Declarer'),

                pl.struct(['Contract','Declarer_Pair_Direction', 'Score_NS', 'Score_EW']).map_elements(
                    lambda r: 0 if r['Contract'] == 'PASS' else None if r['Declarer_Pair_Direction'] is None else r[f'Score_{r["Declarer_Pair_Direction"]}'],
                    return_dtype=pl.Int16
                ).alias('Score_Declarer'),
                
                pl.when(pl.col('Declarer_Pair_Direction').eq(pl.lit('NS')))
                .then(pl.col('Par_NS'))
                .otherwise(pl.col('Par_EW'))
                .alias('Par_Declarer'),          
            ])
            .with_columns([
                pl.col('Declarer_Pair_Direction').replace_strict(PairDirectionToOpponentPairDirection).alias('Opponent_Pair_Direction'),
                pl.col('Direction_OnLead').replace_strict(NextPosition).alias('Direction_Dummy'),
                pl.struct(['Direction_OnLead', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(
                    lambda r: None if r['Direction_OnLead'] is None else r[f'Player_ID_{r["Direction_OnLead"]}'],
                    return_dtype=pl.String
                ).alias('OnLead'),
            ])
            .with_columns([
                pl.col('Direction_Dummy').replace_strict(NextPosition).alias('Direction_NotOnLead'),
                pl.struct(['Direction_Dummy', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(
                    lambda r: None if r['Direction_Dummy'] is None else r[f'Player_ID_{r["Direction_Dummy"]}'],
                    return_dtype=pl.String
                ).alias('Dummy'),
                pl.col('Score_Declarer').le(pl.col('Par_Declarer')).alias('Defender_Par_GE')
            ])
            .with_columns([
                pl.struct(['Direction_NotOnLead', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']).map_elements(
                    lambda r: None if r['Direction_NotOnLead'] is None else r[f"Player_ID_{r["Direction_NotOnLead"]}"],
                    return_dtype=pl.String
                ).alias('NotOnLead')
            ]),
            self.df
        )

    def _create_prob_taking_columns(self) -> None:
        t0 = time.time()
        # Valid (pair, declarer) combinations
        pd_pairs = [("NS", "N"), ("NS", "S"), ("EW", "E"), ("EW", "W")]
        suits = list("CDHSN")

        def build_prob_expr(current_t: int) -> pl.Expr:
            # Weighted sum: select matching Probs_{pair}_{decl}_{suit}_{t}
            terms = []
            for pair, decl in pd_pairs:
                for s in suits:
                    prob_col = f"Probs_{pair}_{decl}_{s}_{current_t}"
                    # Only add term if the probability column exists to avoid ColumnNotFound errors
                    if prob_col in self.df.columns:
                        terms.append(
                            pl.when(
                                (pl.col("Declarer_Pair_Direction") == pair)
                                & (pl.col("Declarer_Direction") == decl)
                                & (pl.col("BidSuit") == s)
                            )
                            .then(pl.col(prob_col))
                            .otherwise(0.0)
                        )
            # If no matching columns exist, return Nulls
            if not terms:
                return pl.lit(None).alias(f"Prob_Taking_{current_t}")
            return (
                pl.when(pl.col("BidSuit").is_null())
                .then(None)
                .otherwise(pl.sum_horizontal(terms))
                .alias(f"Prob_Taking_{current_t}")
            )

        # Create all Prob_Taking_{t} columns in one with_columns
        self.df = self.df.with_columns([
            build_prob_expr(t) for t in range(14)
        ])
        print(f"create prob_taking columns: time:{time.time()-t0} seconds")

    # def _create_prob_taking_columns(self) -> None:
    #     self.df = self._time_operation(
    #         "create prob_taking columns",
    #         lambda df: df.with_columns([
    #             *[
    #                 # Note: this is how to create a column name to dynamically choose a value from multiple columns on a row-by-row basis.
    #                 # For each t (0 ... 13), build a new column which looks up the value
    #                 # from the column whose name is dynamically built from the row values.
    #                 # If BidSuit is None, return None.
    #                 pl.struct([
    #                     pl.col("Declarer_Pair_Direction"),
    #                     pl.col("Declarer_Direction"),
    #                     pl.col("BidSuit"),
    #                     pl.col("^Probs_.*$")
    #                 ]).map_elements(
    #                     # crazy, crazy. current_t is needed because map_elements is a lambda and not a function. otherwise t is always 13!
    #                     lambda row, current_t=t: None if row["BidSuit"] is None 
    #                                               else row[f'Probs_{row["Declarer_Pair_Direction"]}_{row["Declarer_Direction"]}_{row["BidSuit"]}_{current_t}'],
    #                     return_dtype=pl.Float32
    #                 ).alias(f'Prob_Taking_{t}') # todo: short form of 'Declarer_SD_Probs_Taking_{t}'
    #                 for t in range(14)
    #             ]
    #         ]),
    #         self.df
    #     )

    def _create_board_result_columns(self) -> None:
        print(self.df.filter(pl.col('Result').is_null() | pl.col('Tricks').is_null())
              ['Contract','Declarer_Direction','Vul_Declarer','iVul','Score_NS','BidLvl','Result','Tricks'])
        
        all_scores_d, scores_d, scores_df = calculate_scores() # todo: put this in __init__?
        
        t = time.time()
        self.df = self.df.with_columns([
            # Optimized EV_Score_Declarer using when/otherwise instead of map_elements
            pl.when(pl.col('EV_Score_Col_Declarer').is_null())
            .then(None)
            .otherwise(
                # Create a struct with all EV columns that might be referenced
                pl.struct(['EV_Score_Col_Declarer'] + 
                         [f'EV_{pd}_{d}_{s}_{l}' for pd in ['NS','EW'] for d in pd for s in 'SHDCN' for l in range(1,8)])
                .map_elements(lambda x: x[x['EV_Score_Col_Declarer']] if x['EV_Score_Col_Declarer'] is not None else None, return_dtype=pl.Float32)
            )
            .alias('EV_Score_Declarer'),
            
            # Computed_Score_Declarer using pre-computed scores dictionary
            pl.struct(['BidLvl', 'BidSuit', 'Tricks', 'Vul_Declarer', 'Dbl'])
                .map_elements(lambda x: all_scores_d.get(tuple(x.values()),None), # default becomes 0. ok? should only occur in the case of null (PASS).
                            return_dtype=pl.Int16)
                .alias('Computed_Score_Declarer'),

            # Computed_Score_Declarer2 using score function - FIXED: external function call removed
            # Note: This computation is now handled by the Computed_Score_Declarer above using the pre-computed scores dictionary
            # The external score function call has been removed to avoid struct reference issues
            pl.lit(None).alias('Computed_Score_Declarer2'),
        ])
        print(f"create board result columns: time:{time.time()-t} seconds")
        # Note: Computed_Score_Declarer2 has been removed due to external function call issues
        # The Computed_Score_Declarer using the pre-computed scores dictionary is the primary implementation

    def _create_trick_columns(self) -> None:
        t = time.time()
        self.df = self.df.with_columns([
            (pl.col('Result') > 0).alias('OverTricks'),
            (pl.col('Result') == 0).alias('JustMade'),
            (pl.col('Result') < 0).alias('UnderTricks'),
            # pl.col('Tricks').alias('Tricks_Declarer'),
            #(pl.col('Tricks') - pl.col('DD_Tricks')).alias('Tricks_DD_Diff_Declarer'),
        ])
        print(f"create trick columns: time:{time.time()-t} seconds")

    def _create_rating_columns(self) -> None:
        self.df = self._time_operation(
            "create rating columns",
            lambda df: df.with_columns([
                pl.col('DD_Tricks_Diff') # was Tricks_DD_Diff_Declarer
                    .mean()
                    .over('Declarer_ID')
                    .alias('Declarer_Rating'),

                pl.col('Defender_Par_GE')
                    .cast(pl.Float32)
                    .mean()
                    .over('OnLead')
                    .alias('Defender_OnLead_Rating'),

                pl.col('Defender_Par_GE')
                    .cast(pl.Float32)
                    .mean()
                    .over('NotOnLead')
                    .alias('Defender_NotOnLead_Rating')
            ]),
            self.df
        )

    def perform_final_contract_augmentations(self) -> pl.DataFrame:
        """Main method to perform final contract augmentations"""
        t_start = time.time()
        print(f"Starting final contract augmentations")

        self._process_contract_columns() # 5s
        self._create_contract_types() # 29s
        self._create_player_names() # 1s # do sooner?
        self._create_declarer_columns() # 70s # do sooner?
        self._create_result_columns()
        self._create_score_columns()
        self._create_dd_columns() # 1m30s
        self._create_ev_columns() # 2s
        self._create_score_diff_columns()
        self._create_lott() # 7s # todo: would be interesting to create lott for all contracts and then move into AllContractsAugmenter
        #self._perform_legacy_renames() # 6s
        self._create_position_columns() # 1m30s
        self._create_prob_taking_columns() # 7s
        self._create_board_result_columns() # 10m
        self._create_trick_columns() # 0s
        self._create_rating_columns() # 3s

        print(f"Final contract augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df


class MatchPointAugmenter:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.discrete_score_columns = [] # ['DD_Score_NS', 'EV_Max_NS'] # calculate matchpoints for these columns which change with each row's Score_NS
        self.dd_score_columns = [f'DD_Score_{l}{s}_{d}' for d in 'NESW' for s in 'SHDCN' for l in range(1,8)]
        self.ev_score_columns = [f'EV_{pd}_{d}_{s}_{l}' for pd in ['NS','EW'] for d in pd for s in 'SHDCN' for l in range(1,8)]
        self.all_score_columns = self.discrete_score_columns + self.dd_score_columns + self.ev_score_columns

    def _time_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        t = time.time()
        result = func(*args, **kwargs)
        print(f"{operation_name}: time:{time.time()-t} seconds")
        return result

    def _create_mp_top(self) -> None:
        # Assert required columns exist
        assert 'Score' in self.df.columns, "Required column 'Score' not found in DataFrame"
        assert 'session_id' in self.df.columns, "Required column 'session_id' not found in DataFrame"
        assert 'PBN' in self.df.columns, "Required column 'PBN' not found in DataFrame"
        assert 'Board' in self.df.columns, "Required column 'Board' not found in DataFrame"
        
        t = time.time()
        if 'MP_Top' not in self.df.columns:
            self.df = create_mp_top_column(self.df)
        print(f"create MP_Top: time:{time.time()-t} seconds")
        
        # Assert column was created
        assert 'MP_Top' in self.df.columns, "Column 'MP_Top' was not created"

    def _calculate_matchpoints(self) -> None:
        # Assert required columns exist
        assert 'Score_NS' in self.df.columns, "Required column 'Score_NS' not found in DataFrame"
        assert 'Score_EW' in self.df.columns, "Required column 'Score_EW' not found in DataFrame"
        assert 'session_id' in self.df.columns, "Required column 'session_id' not found in DataFrame"
        assert 'PBN' in self.df.columns, "Required column 'PBN' not found in DataFrame"
        assert 'Board' in self.df.columns, "Required column 'Board' not found in DataFrame"
        
        t = time.time()
        if 'MP_NS' not in self.df.columns:
            self.df = create_matchpoint_columns(self.df)
        print(f"calculate matchpoints MP_(NS|EW): time:{time.time()-t} seconds")
        
        # Assert columns were created
        assert 'MP_NS' in self.df.columns, "Column 'MP_NS' was not created"
        assert 'MP_EW' in self.df.columns, "Column 'MP_EW' was not created"

    def _calculate_percentages(self) -> None:
        # Assert required columns exist
        assert 'MP_NS' in self.df.columns, "Required column 'MP_NS' not found in DataFrame"
        assert 'MP_EW' in self.df.columns, "Required column 'MP_EW' not found in DataFrame"
        assert 'MP_Top' in self.df.columns, "Required column 'MP_Top' not found in DataFrame"
        
        t = time.time()
        if 'Pct_NS' not in self.df.columns:
            self.df = create_percentage_columns(self.df)
        print(f"calculate matchpoints percentages: time:{time.time()-t} seconds")
        
        # Assert columns were created
        assert 'Pct_NS' in self.df.columns, "Column 'Pct_NS' was not created"
        assert 'Pct_EW' in self.df.columns, "Column 'Pct_EW' was not created"

    def _create_declarer_pct(self) -> None:
        # Assert required columns exist
        assert 'Declarer_Pair_Direction' in self.df.columns, "Required column 'Declarer_Pair_Direction' not found in DataFrame"
        assert 'Pct_NS' in self.df.columns, "Required column 'Pct_NS' not found in DataFrame"
        assert 'Pct_EW' in self.df.columns, "Required column 'Pct_EW' not found in DataFrame"
        
        t = time.time()
        if 'Declarer_Pct' not in self.df.columns:
            self.df = create_declarer_pct_column(self.df)
        print(f"create Declarer_Pct: time:{time.time()-t} seconds")
        
        # Assert column was created
        assert 'Declarer_Pct' in self.df.columns, "Column 'Declarer_Pct' was not created"

    def _calculate_matchpoints_group(self, series_list: list[pl.Series]) -> pl.Series:
        col_values = series_list[0]
        score_ns_values = series_list[1]
        if col_values.is_null().sum() > 0: # todo: use null_count()?
            print(f"Warning: Null values in col_values: {col_values.is_null().sum()}") # todo: use null_count()?
        #if score_ns_values.is_null().sum() > 0:
        #    print(f"Warning: Null values in score_ns_values: {score_ns_values.is_null().sum()}") # todo: use null_count()?
        # todo: is there a more proper way to handle null values in col_values and score_ns_values?
        score_ns_values = score_ns_values.fill_null(0.0) # todo: why do some have nulls? sitout? adjusted score?
        col_values = col_values.fill_null(0.0) # todo: why do some have nulls? sitout? adjusted score?
        return pl.Series([
            sum(1.0 if val > score else 0.5 if val == score else 0.0 
                for score in score_ns_values)
            for val in col_values
        ])

    def _calculate_all_score_matchpoints(self) -> None:
        """Calculate matchpoints for all score columns in batches to prevent memory issues."""
        print(f"Processing {len(self.all_score_columns)} score columns in batches...")
        
        # Process columns in smaller batches to prevent memory explosion
        batch_size = 10  # Process 10 columns at a time
        all_columns = self.all_score_columns + ['DD_Score_NS', 'DD_Score_EW', 'Par_NS', 'Par_EW']
        
        for i in range(0, len(all_columns), batch_size):
            batch = all_columns[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(all_columns) + batch_size - 1)//batch_size}: {len(batch)} columns")
            
            self.df = self._time_operation(
                f"create score matchpoints batch {i//batch_size + 1}",
                lambda df: df.with_columns([
                    # Calculate and sum in one operation per column
                    pl.when(pl.col(col) > pl.col('Score_NS' if '_NS' in col or col[-1] in 'NS' else 'Score_EW'))
                    .then(1.0)
                    .when(pl.col(col) == pl.col('Score_NS' if '_NS' in col or col[-1] in 'NS' else 'Score_EW'))
                    .then(0.5)
                    .otherwise(0.0)
                    .sum().over(['session_id', 'PBN', 'Board'])
                    .alias(f'MP_{col}')
                    for col in batch
                ]),
                self.df
            )
        
        # Calculate matchpoints for declarer columns
        declarer_columns = ['DD_Score_Declarer', 'Par_Declarer', 'EV_Score_Declarer', 'EV_Max_Declarer']
        self.df = self._time_operation(
            "create declarer score matchpoints",
            lambda df: df.with_columns([
                # Calculate matchpoints for declarer columns using Score_Declarer
                pl.when(pl.col(col) > pl.col('Score_Declarer'))
                .then(1.0)
                .when(pl.col(col) == pl.col('Score_Declarer'))
                .then(0.5)
                .otherwise(0.0)
                .sum().over(['session_id', 'PBN', 'Board'])
                .alias(f'MP_{col}')
                for col in declarer_columns
            ]),
            self.df
        )
        
        # Calculate matchpoints for max columns
        max_columns = ['EV_Max_NS', 'EV_Max_EW']
        self.df = self._time_operation(
            "create max score matchpoints",
            lambda df: df.with_columns([
                # Calculate matchpoints for max columns using Score_NS/Score_EW
                pl.when(pl.col(col) > pl.col('Score_NS' if col.endswith('_NS') else 'Score_EW'))
                .then(1.0)
                .when(pl.col(col) == pl.col('Score_NS' if col.endswith('_NS') else 'Score_EW'))
                .then(0.5)
                .otherwise(0.0)
                .sum().over(['session_id', 'PBN', 'Board'])
                .alias(f'MP_{col}')
                for col in max_columns
            ]),
            self.df
        )

    def _calculate_mp_pct_from_new_score(self, col: str) -> pl.Series:
        """Calculate matchpoint percentage from MP column."""
        return pl.col(f'MP_{col}') / (pl.col('MP_Top') + 1)

    def _calculate_elo_pair_matchpoint_ratings(self) -> None:
        """Calculate Elo pair matchpoint ratings."""
        t = time.time()
        self.df = compute_elo_pair_matchpoint_ratings(self.df)
        print(f"calculate Elo pair matchpoint ratings: time:{time.time()-t} seconds")

    def _calculate_elo_player_matchpoint_ratings(self) -> None:
        """Calculate Elo player matchpoint ratings."""
        t = time.time()
        self.df = compute_elo_player_matchpoint_ratings(self.df)
        print(f"calculate Elo player matchpoint ratings: time:{time.time()-t} seconds")

    def _calculate_event_start_end_elo_columns(self) -> None:
        """Calculate Elo start/end columns."""
        t = time.time()
        self.df = compute_event_start_end_elo_columns(self.df)
        print(f"calculate event start end Elo columns: time:{time.time()-t} seconds")

    def _calculate_final_scores(self) -> None:
        """Calculate final scores and percentages using optimized vectorized operations."""
        t = time.time()
        self.df = self._calculate_final_scores_internal(self.df)
        print(f"calculate final scores: time:{time.time()-t} seconds")

    def _calculate_final_scores_internal(self, df: pl.DataFrame) -> pl.DataFrame:
        """Internal implementation of final scores calculation."""
        # Split into smaller, more efficient functions
        self._calculate_dd_score_percentages()
        self._calculate_par_percentages()
        self._calculate_declarer_percentages()
        self._calculate_max_scores()
        self._calculate_difference_scores()
        
        return self.df

    def _calculate_dd_score_percentages(self) -> None:
        """Calculate DD Score percentages using optimized vectorized operations."""
        t = time.time()
        # Preserve original order since join_asof requires sorting
        self.df = self.df.with_row_index(name='__row_nr')
        
        # Use an asof-join against per-board score frequency tables.
        # This avoids Python lambdas and list.eval cross-column references.
        for pair in ['NS', 'EW']:
            score_col = f'Score_{pair}'
            dd_score_col = f'DD_Score_{pair}'
            assert score_col in self.df.columns, f"Required column {score_col} not found"
            assert dd_score_col in self.df.columns, f"Required column {dd_score_col} not found"

            # Build per-board frequency and cumulative frequency table for actual scores
            freq_name = f'__freq_{pair.lower()}'
            cum_name = f'__cum_{pair.lower()}'
            key_name = f'__score_key_{pair}'

            lookup = (
                self.df.select(['Board', score_col])
                .group_by(['Board', score_col])
                .agg(pl.count().alias(freq_name))
                .rename({score_col: key_name})
                .sort(['Board', key_name])
                .with_columns([
                    pl.col(freq_name).cum_sum().over('Board').alias(cum_name)
                ])
            )

            # Ensure DD score dtype matches right key dtype for asof join
            right_key_dtype = lookup.schema[key_name]
            dd_tmp = f'__dd_tmp_{pair}'
            self.df = self.df.with_columns([
                pl.col(dd_score_col).cast(right_key_dtype).alias(dd_tmp)
            ])

            # Sort left side by group and asof key to satisfy join_asof requirements
            self.df = self.df.sort(['Board', dd_tmp])

            # Asof join to get cumulative count up to the largest score <= DD
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Sortedness of columns cannot be checked when 'by' groups provided",
                    category=UserWarning,
                )
                self.df = self.df.join_asof(
                    lookup,
                    left_on=dd_tmp,
                    right_on=key_name,
                    by='Board',
                    strategy='backward'
                )

            # Compute wins and ties from the joined stats
            mp_col = f'MP_{dd_score_col}'
            self.df = self.df.with_columns([
                # wins: count of scores strictly less than DD
                pl.when(pl.col(cum_name).is_null())
                .then(0.0)
                .otherwise(
                    pl.when(pl.col(dd_tmp) == pl.col(key_name))
                    .then((pl.col(cum_name) - pl.col(freq_name)).cast(pl.Float64))
                    .otherwise(pl.col(cum_name).cast(pl.Float64))
                ).alias(f'{mp_col}_wins'),

                # ties: 0.5 per equal score
                pl.when((pl.col(dd_tmp) == pl.col(key_name)) & pl.col(freq_name).is_not_null())
                .then((pl.col(freq_name) * 0.5).cast(pl.Float64))
                .otherwise(0.0)
                .alias(f'{mp_col}_ties')
            ])

            # Total MP and percentage
            self.df = self.df.with_columns([
                (pl.col(f'{mp_col}_wins') + pl.col(f'{mp_col}_ties')).alias(mp_col),
                (pl.col(mp_col) / (pl.col('MP_Top') + 1)).alias(f'DD_Score_Pct_{pair}')
            ])

            # Drop temporaries for this pair
            drop_cols = [c for c in [dd_tmp, key_name, freq_name, cum_name, f'{mp_col}_wins', f'{mp_col}_ties'] if c in self.df.columns]
            if drop_cols:
                self.df = self.df.drop(drop_cols)

        # Restore original order
        if '__row_nr' in self.df.columns:
            self.df = self.df.sort('__row_nr').drop('__row_nr')
        
        print(f"DD score percentages: {time.time()-t:.3f} seconds")

    def _calculate_par_percentages(self) -> None:
        """Calculate Par percentages using optimized vectorized operations."""
        t = time.time()
        
        # Calculate Par percentages using vectorized operations
        for pair in ['NS', 'EW']:
            par_col = f'Par_{pair}'
            score_col = f'Score_{pair}'
            
            # Calculate Par matchpoints using vectorized when/otherwise
            self.df = self.df.with_columns([
                pl.when(pl.col(par_col) > pl.col(score_col)).then(1.0)
                .when(pl.col(par_col) == pl.col(score_col)).then(0.5)
                .otherwise(0.0)
                .alias(f'MP_Rank_{par_col}')
            ])
            
            # Sum over board and calculate percentage
            self.df = self.df.with_columns([
                pl.col(f'MP_Rank_{par_col}').sum().over('Board').alias(f'MP_{par_col}'),
                ((pl.col(f'MP_Rank_{par_col}').sum().over('Board')) / (pl.col('MP_Top') + 1)).alias(f'Par_Pct_{pair}')
            ])
            
            # Clean up intermediate column
            self.df = self.df.drop(f'MP_Rank_{par_col}')
        
        print(f"Par percentages: {time.time()-t:.3f} seconds")

    def _calculate_declarer_percentages(self) -> None:
        """Calculate declarer orientation scores using existing helper method."""
        t = time.time()
        
        # Calculate declarer orientation scores using existing helper method
        self.df = self.df.with_columns([
            self._calculate_mp_pct_from_new_score('DD_Score_Declarer').alias('MP_DD_Pct_Declarer'),
            self._calculate_mp_pct_from_new_score('Par_Declarer').alias('MP_Par_Pct_Declarer'),
            self._calculate_mp_pct_from_new_score('EV_Score_Declarer').alias('MP_EV_Pct_Declarer'),
            self._calculate_mp_pct_from_new_score('EV_Max_Declarer').alias('MP_EV_Max_Pct_Declarer')
        ])
        
        print(f"Declarer percentages: {time.time()-t:.3f} seconds")

    def _calculate_max_scores(self) -> None:
        """Calculate max scores and their percentages."""
        t = time.time()
        
        # Calculate remaining scores and percentages in a single operation
        # Combine all operations to reduce DataFrame operations
        self.df = self.df.with_columns([
            # Max DD scores
            pl.max_horizontal(f'^MP_DD_Score_[1-7][SHDCN]_[NS]$').alias('MP_DD_Score_Max_NS'),
            pl.max_horizontal(f'^MP_DD_Score_[1-7][SHDCN]_[EW]$').alias('MP_DD_Score_Max_EW'),
            
            # Max EV scores
            pl.max_horizontal(f'^MP_EV_NS_[NS]_[SHDCN]_[1-7]$').alias('MP_EV_Max_NS'),
            pl.max_horizontal(f'^MP_EV_EW_[EW]_[SHDCN]_[1-7]$').alias('MP_EV_Max_EW'),
        ])
        
        # Calculate percentages for max scores
        self.df = self.df.with_columns([
            pl.when(pl.col('MP_Top') > 0)
            .then(pl.coalesce([pl.col('MP_DD_Score_Max_NS').cast(pl.Float64), pl.lit(0.0)]) / pl.col('MP_Top'))
            .otherwise(0.0)
            .alias('DD_Pct_Max_NS'),
            pl.when(pl.col('MP_Top') > 0)
            .then(pl.coalesce([pl.col('MP_DD_Score_Max_EW').cast(pl.Float64), pl.lit(0.0)]) / pl.col('MP_Top'))
            .otherwise(0.0)
            .alias('DD_Pct_Max_EW'),
            self._calculate_mp_pct_from_new_score('EV_Max_NS').alias('EV_Pct_Max_NS'),
            self._calculate_mp_pct_from_new_score('EV_Max_EW').alias('EV_Pct_Max_EW'),
        ])
        
        print(f"Max scores: {time.time()-t:.3f} seconds")

    def _calculate_difference_scores(self) -> None:
        """Calculate difference columns and clean up temporary columns."""
        t = time.time()
        
        # Calculate difference columns in a single operation
        zero = pl.lit(0.0)
        self.df = self.df.with_columns([
            (pl.coalesce([pl.col('Pct_NS').cast(pl.Float64), zero]) - pl.coalesce([pl.col('EV_Pct_Max_NS').cast(pl.Float64), zero])).alias('EV_Pct_Max_Diff_NS'),
            (pl.coalesce([pl.col('Pct_EW').cast(pl.Float64), zero]) - pl.coalesce([pl.col('EV_Pct_Max_EW').cast(pl.Float64), zero])).alias('EV_Pct_Max_Diff_EW'),
            (pl.coalesce([pl.col('Pct_NS').cast(pl.Float64), zero]) - pl.coalesce([pl.col('DD_Pct_Max_NS').cast(pl.Float64), zero])).alias('DD_Pct_Max_Diff_NS'),
            (pl.coalesce([pl.col('Pct_EW').cast(pl.Float64), zero]) - pl.coalesce([pl.col('DD_Pct_Max_EW').cast(pl.Float64), zero])).alias('DD_Pct_Max_Diff_EW'),
            (pl.coalesce([pl.col('DD_Pct_Max_NS').cast(pl.Float64), zero]) - pl.coalesce([pl.col('EV_Pct_Max_NS').cast(pl.Float64), zero])).alias('DD_EV_Pct_Max_Diff_NS'),
            (pl.coalesce([pl.col('DD_Pct_Max_EW').cast(pl.Float64), zero]) - pl.coalesce([pl.col('EV_Pct_Max_EW').cast(pl.Float64), zero])).alias('DD_EV_Pct_Max_Diff_EW'),
        ])
        
        print(f"Difference scores: {time.time()-t:.3f} seconds")

    def _add_event_start_end_elo_columns(self) -> None:
        """Add constant per-session Elo columns for each seat and pair.
        Adds both EventStart (pre-board at first appearance) and EventEnd
        (post-update at last appearance) values for players and pairs, and
        propagates them across all rows in the session for each entity.
        """
        t = time.time()
        # Ensure sorting to define session start order
        sort_keys = [c for c in ["Date", "session_id", "Round", "Board"] if c in self.df.columns]
        if not sort_keys:
            sort_keys = ["Date"]
        self.df = self.df.sort(sort_keys)

        # Player Elo at session start (direction-agnostic per person, but taken from seat)
        seat_before_cols = {
            "N": ("Player_ID_N", "Elo_R_N_Before", "Elo_R_Player_N_EventStart"),
            "S": ("Player_ID_S", "Elo_R_S_Before", "Elo_R_Player_S_EventStart"),
            "E": ("Player_ID_E", "Elo_R_E_Before", "Elo_R_Player_E_EventStart"),
            "W": ("Player_ID_W", "Elo_R_W_Before", "Elo_R_Player_W_EventStart"),
        }
        for seat, (pid_col, before_col, out_col) in seat_before_cols.items():
            if pid_col in self.df.columns and before_col in self.df.columns:
                self.df = self.df.with_columns(
                    pl.col(before_col).first().over(["session_id", pid_col]).alias(out_col)
                )

        # Pair Elo at session start (directional pairing id)
        pair_before_cols = {
            "NS": ("Pair_Number_NS", "Elo_R_NS_Before", "Elo_R_Pair_NS_EventStart"),
            "EW": ("Pair_Number_EW", "Elo_R_EW_Before", "Elo_R_Pair_EW_EventStart"),
        }
        for side, (pair_col, before_col, out_col) in pair_before_cols.items():
            if pair_col in self.df.columns and before_col in self.df.columns:
                self.df = self.df.with_columns(
                    pl.col(before_col).first().over(["session_id", pair_col]).alias(out_col)
                )

        # Player Elo at session end (post-update at last appearance)
        seat_after_cols = {
            "N": ("Player_ID_N", "Elo_R_N", "Elo_R_Player_N_EventEnd"),
            "S": ("Player_ID_S", "Elo_R_S", "Elo_R_Player_S_EventEnd"),
            "E": ("Player_ID_E", "Elo_R_E", "Elo_R_Player_E_EventEnd"),
            "W": ("Player_ID_W", "Elo_R_W", "Elo_R_Player_W_EventEnd"),
        }
        for seat, (pid_col, after_col, out_col) in seat_after_cols.items():
            if pid_col in self.df.columns and after_col in self.df.columns:
                self.df = self.df.with_columns(
                    pl.col(after_col).last().over(["session_id", pid_col]).alias(out_col)
                )

        # Pair Elo at session end (directional pairing id)
        pair_after_cols = {
            "NS": ("Pair_Number_NS", "Elo_R_NS", "Elo_R_Pair_NS_EventEnd"),
            "EW": ("Pair_Number_EW", "Elo_R_EW", "Elo_R_Pair_EW_EventEnd"),
        }
        for side, (pair_col, after_col, out_col) in pair_after_cols.items():
            if pair_col in self.df.columns and after_col in self.df.columns:
                self.df = self.df.with_columns(
                    pl.col(after_col).last().over(["session_id", pair_col]).alias(out_col)
                )

        print(f"Event start/end Elo columns: {time.time()-t:.3f} seconds")

    def perform_matchpoint_augmentations(self) -> pl.DataFrame:
        t_start = time.time()
        print(f"Starting matchpoint augmentations")
        
        self._create_mp_top() # 5s
        self._calculate_matchpoints() # 12s
        self._calculate_percentages() # 1s
        self._create_declarer_pct() # 1s
        self._calculate_all_score_matchpoints() # 3m
        self._calculate_final_scores() # ?
        self._calculate_elo_pair_matchpoint_ratings() # 1s
        self._calculate_elo_player_matchpoint_ratings() # 1s
        self._calculate_event_start_end_elo_columns()

        print(f"Matchpoint augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df


# todo: IMP augmentations are not implemented yet
class IMPAugmenter:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def perform_imp_augmentations(self) -> pl.DataFrame:
        t_start = time.time()
        print(f"Starting IMP augmentations")
        
        print(f"IMP augmentations complete: {time.time() - t_start:.2f} seconds")
        return self.df


class AllHandRecordAugmentations:
    def __init__(self, df: pl.DataFrame, 
                 hrs_cache_df: Optional[pl.DataFrame] = None, 
                 sd_productions: int = 40, 
                 max_adds: Optional[int] = None,
                 output_progress: Optional[bool] = True,
                 progress: Optional[Any] = None,
                 lock_func: Optional[Callable[..., pl.DataFrame]] = None):
        """Initialize the AllAugmentations class with a DataFrame and optional parameters.
        
        Args:
            df: The input DataFrame to augment
            hrs_cache_df: dataframe of cached computes
            sd_productions: Number of single dummy productions to generate
            max_adds: Maximum number of adds to generate
            output_progress: Whether to output progress
            progress: Optional progress indicator object
            lock_func: Optional function for thread safety
        """
        self.df = df
        self.hrs_cache_df = hrs_cache_df
        self.sd_productions = sd_productions
        self.max_adds = max_adds
        self.output_progress = output_progress
        self.progress = progress
        self.lock_func = lock_func

        # instance initialization

        # Double dummy tricks for each player and strain
        dd_cols = {f"DD_{p}_{s}": pl.UInt8 for p in 'NESW' for s in 'CDHSN'}

        # Single dummy probabilities. Note that the order of declarers and suits must match the original schema.
        # The declarer order was found to be N, S, W, E from inspecting the original schema.
        probs_cols = {f"Probs_{pair}_{declarer}_{s}_{i}": pl.Float64 for pair in ['NS', 'EW'] for declarer in 'NESW' for s in 'CDHSN' for i in range(14)}

        # Columns that appear after DD columns and before probability columns
        schema_cols = {
            'PBN': pl.String,
            'Dealer': pl.String,
            'Vul': pl.String,
            **dd_cols,
            'ParScore': pl.Int16,
            'ParNumber': pl.Int8,
            'ParContracts': pl.List(pl.Struct({
                'Level': pl.String, 
                'Strain': pl.String, 
                'Doubled': pl.String, 
                'Pair_Direction': pl.String, 
                'Result': pl.Int16
            })),
            'Probs_Trials': pl.Int64,
            **probs_cols,
        }

        # Convert the dict to a Polars Schema for a valid comparison
        hrs_cache_df_schema = pl.Schema(schema_cols)

        if hrs_cache_df is None:
            self.hrs_cache_df = pl.DataFrame(schema=hrs_cache_df_schema)
        else:
            assert set(self.hrs_cache_df.schema.items()) == set(hrs_cache_df_schema.items()), f"hrs_cache_df schema {self.hrs_cache_df.schema} does not match expected schema {hrs_cache_df_schema}"
        
    def perform_all_hand_record_augmentations(self) -> pl.DataFrame:
        """Execute all hand record augmentation steps. Input is a fully cleaned hand record DataFrame.
        
        Returns:
            The fully augmented hand record DataFrame.
        """
        t_start = time.time()
        print(f"Starting all hand record augmentations on DataFrame with {len(self.df)} rows")
        
        # Step 1: Deal-level augmentations
        deal_augmenter = DealAugmenter(self.df)
        self.df = deal_augmenter.perform_deal_augmentations()
        
        # Step 2: Hand augmentations
        result_augmenter = HandAugmenter(self.df)
        self.df = result_augmenter.perform_hand_augmentations()

        # Step 3: Double dummy and single dummy augmentations
        dd_sd_augmenter = DD_SD_Augmenter(
            self.df, 
            self.hrs_cache_df,  
            self.sd_productions, 
            self.max_adds, 
            self.output_progress,
            self.progress,
            self.lock_func
        )
        self.df, self.hrs_cache_df = dd_sd_augmenter.perform_dd_sd_augmentations()

        # todo: move this somewhere more sensible.
        self.df = self.df.with_columns(pl.col('ParScore').alias('Par_NS'))
        self.df = self.df.with_columns(pl.col('ParScore').neg().alias('Par_EW'))
        
        # todo: move this somewhere more sensible.
        # Create DD columns for pair directions and strains e.g. DD_NS_S
        dd_pair_columns = [
            pl.max_horizontal(f'DD_{pair_direction[0]}_{strain}', f'DD_{pair_direction[1]}_{strain}').alias(f'DD_{pair_direction}_{strain}')
            for pair_direction in ['NS', 'EW']
            for strain in 'SHDCN'
        ]
        self.df = self.df.with_columns(dd_pair_columns)
        
        # Step 4: All contract augmentations
        hand_augmenter = AllContractsAugmenter(self.df)
        self.df = hand_augmenter.perform_all_contracts_augmentations()

        print(f"All hand records augmentations completed in {time.time() - t_start:.2f} seconds")
        
        return self.df, self.hrs_cache_df


class AllBoardResultsAugmentations:
    def __init__(self, df: pl.DataFrame):
        self.df = df


    def perform_all_board_results_augmentations(self) -> pl.DataFrame:
        """Execute all board results augmentation steps. Input is a fully augmented hand record DataFrame.
        Only relies on columns within brs_df and not any in hrs_df.

        Returns:
            The fully joined and augmented hand record and board results DataFrame.
        """
        t_start = time.time()
        print(f"Starting all board results augmentations on DataFrame with {len(self.df)} rows")
        
        # Step 5: Final contract augmentations
        hand_augmenter = FinalContractAugmenter(self.df)
        self.df = hand_augmenter.perform_final_contract_augmentations()
        
        # Step 6: Matchpoint augmentations
        matchpoint_augmenter = MatchPointAugmenter(self.df)
        self.df = matchpoint_augmenter.perform_matchpoint_augmentations()
        
        # Step 7: IMP augmentations (not implemented yet)
        imp_augmenter = IMPAugmenter(self.df)
        self.df = imp_augmenter.perform_imp_augmentations()
        
        print(f"All board results augmentations completed in {time.time() - t_start:.2f} seconds")
        
        return self.df


class AllAugmentations:
    def __init__(self, df: pl.DataFrame, hrs_cache_df: Optional[pl.DataFrame] = None, sd_productions: int = 40, max_adds: Optional[int] = None, output_progress: Optional[bool] = True, progress: Optional[Any] = None, lock_func: Optional[Callable[..., pl.DataFrame]] = None):
        self.df = df
        self.hrs_cache_df = hrs_cache_df
        self.sd_productions = sd_productions
        self.max_adds = max_adds
        self.output_progress = output_progress
        self.progress = progress
        self.lock_func = lock_func

    def perform_all_augmentations(self) -> pl.DataFrame:
        """Execute all augmentation steps.
        
        Returns:
            The fully joined and augmented hand record and board results DataFrame.
        """
        t_start = time.time()
        print(f"Starting all augmentations on DataFrame with {len(self.df)} rows")

        hand_record_augmenter = AllHandRecordAugmentations(self.df, self.hrs_cache_df, self.sd_productions, self.max_adds, self.output_progress, self.progress, self.lock_func)
        self.df, self.hrs_cache_df = hand_record_augmenter.perform_all_hand_record_augmentations()
        board_results_augmenter = AllBoardResultsAugmentations(self.df)
        self.df = board_results_augmenter.perform_all_board_results_augmentations()

        print(f"All augmentations completed in {time.time() - t_start:.2f} seconds")

        return self.df, self.hrs_cache_df


# def read_parquet_sample(file_path, n_rows=1000, method='head'):
#     """
#     Read a sample of rows from a parquet file.
    
#     Args:
#         file_path: Path to parquet file
#         n_rows: Number of rows to sample
#         method: 'head' for first n rows, 'sample' for random sample, 'tail' for last n rows
    
#     Returns:
#         DataFrame with sampled rows
#     """
#     if method == 'head':
#         return pl.scan_parquet(file_path).limit(n_rows).collect()
#     elif method == 'sample':
#         return pl.scan_parquet(file_path).sample(n_rows).collect()
#     elif method == 'tail':
#         return pl.scan_parquet(file_path).tail(n_rows).collect()
#     else:
#         raise ValueError("method must be 'head', 'sample', or 'tail'")

# def read_parquet_slice(file_path, offset=0, length=1000):
#     """
#     Read a specific slice of rows from a parquet file.
    
#     Args:
#         file_path: Path to parquet file
#         offset: Starting row (0-indexed)
#         length: Number of rows to read
    
#     Returns:
#         DataFrame with sliced rows
#     """
#     return pl.scan_parquet(file_path).slice(offset, length).collect()

# def read_parquet_filtered(file_path, filters=None, n_rows=None):
#     """
#     Read parquet file with filters and optional row limit.
    
#     Args:
#         file_path: Path to parquet file
#         filters: Polars expression for filtering
#         n_rows: Optional limit on number of rows
    
#     Returns:
#         Filtered DataFrame
#     """
#     lazy_df = pl.scan_parquet(file_path)
    
#     if filters is not None:
#         lazy_df = lazy_df.filter(filters)
    
#     if n_rows is not None:
#         lazy_df = lazy_df.limit(n_rows)
    
#     return lazy_df.collect()

# def read_parquet_every_nth(file_path, n=10):
#     """
#     Read every nth row from a parquet file.
    
#     Args:
#         file_path: Path to parquet file
#         n: Take every nth row
    
#     Returns:
#         DataFrame with every nth row
#     """
#     return (pl.scan_parquet(file_path)
#             .with_row_index()
#             .filter(pl.col("index") % n == 0)
#             .drop("index")
#             .collect())

# def read_parquet_by_percentage(file_path, percentage=0.1):
#     """
#     Read a percentage of rows from a parquet file using random sampling.
    
#     Args:
#         file_path: Path to parquet file
#         percentage: Percentage of rows to sample (0.0 to 1.0)
    
#     Returns:
#         DataFrame with sampled rows
#     """
#     return pl.scan_parquet(file_path).sample(fraction=percentage).collect()

# def read_parquet_lazy_info(file_path):
#     """
#     Get information about a parquet file without reading data.
    
#     Args:
#         file_path: Path to parquet file
    
#     Returns:
#         Dictionary with file info
#     """
#     lazy_df = pl.scan_parquet(file_path)
#     schema = lazy_df.collect_schema()
    
#     # Get row count (this does scan the file but doesn't load data)
#     row_count = lazy_df.select(pl.len()).collect().item()
    
#     return {
#         'columns': schema.names(),
#         'dtypes': {name: str(dtype) for name, dtype in schema.items()},
#         'row_count': row_count,
#         'column_count': len(schema)
#     }

# def read_parquet_lazy_select(file_path, columns=None, filters=None, sample_n=None, sample_fraction=None):
#     """
#     Read parquet file using lazy evaluation with column selection and optional operations.
#     This is the PREFERRED approach for reading parquet files.
    
#     Args:
#         file_path: Path to parquet file
#         columns: List of column names to select
#         filters: Polars expression for filtering
#         sample_n: Number of rows to sample
#         sample_fraction: Fraction of rows to sample (0.0 to 1.0)
    
#     Returns:
#         DataFrame with selected columns and applied operations
#     """
#     lazy_df = pl.scan_parquet(file_path)
    
#     # Apply column selection (projection pushdown)
#     if columns is not None:
#         lazy_df = lazy_df.select(columns)
    
#     # Apply filters (predicate pushdown)
#     if filters is not None:
#         lazy_df = lazy_df.filter(filters)
    
#     # Apply sampling
#     if sample_n is not None:
#         lazy_df = lazy_df.sample(n=sample_n)
#     elif sample_fraction is not None:
#         lazy_df = lazy_df.sample(fraction=sample_fraction)
    
#     # Execute the optimized query plan
#     return lazy_df.collect()

# def read_parquet_with_sampling(file_path, columns=None, n_rows=None, method='limit', seed=None):
#     """
#     Read parquet file with sampling, handling LazyFrame compatibility issues.
    
#     Args:
#         file_path: Path to parquet file
#         columns: List of column names to select
#         n_rows: Number of rows to sample/limit
#         method: 'limit' (first n rows), 'collect_sample' (random), 'systematic' (every nth)
#         seed: Random seed for sampling
    
#     Returns:
#         DataFrame with sampled rows
#     """
#     lazy_df = pl.scan_parquet(file_path)
    
#     if columns is not None:
#         lazy_df = lazy_df.select(columns)
    
#     if n_rows is None:
#         return lazy_df.collect()
    
#     if method == 'limit':
#         # Fastest - first n rows
#         return lazy_df.limit(n_rows).collect()
    
#     elif method == 'collect_sample':
#         # True random sampling - requires collecting all data first
#         df = lazy_df.collect()
#         return df.sample(n=n_rows, seed=seed)
    
#     elif method == 'systematic':
#         # Every nth row - memory efficient
#         total_rows = lazy_df.select(pl.len()).collect().item()
#         skip = max(1, total_rows // n_rows)
#         return (
#             lazy_df
#             .with_row_index()
#             .filter(pl.col("index") % skip == 0)
#             .drop("index")
#             .limit(n_rows)
#             .collect()
#         )
    
#     else:
#         raise ValueError("method must be 'limit', 'collect_sample', or 'systematic'")

