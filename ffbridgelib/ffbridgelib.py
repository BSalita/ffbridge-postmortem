import polars as pl
from endplay.types import Deal # only used to correct and validate pbns. pbn == Deal(pbn).to_pbn()

def BoardNumberToDealer(bn):
    return 'NESW'[(bn-1) & 3]

def BoardNumberToVul(bn):
    bn -= 1
    return range(bn//4, bn//4+4)[bn & 3] & 3

def PbnToN(bd):
    hands = bd[2:].split(' ')
    d = bd[0]
    match d:
        case 'N':
            pbn = bd
        case 'E':
            pbn = 'N:'+' '.join([hands[1],hands[2],hands[3],hands[0]])
        case 'S':
            pbn = 'N:'+' '.join([hands[2],hands[3],hands[0],hands[1]])
        case 'W':
            pbn = 'N:'+' '.join([hands[3],hands[0],hands[1],hands[2]])
        case _:
            raise ValueError(f"Invalid dealer: {d}")
    dpbn = Deal(pbn).to_pbn()
    if pbn != dpbn:
        print(f"Invalid PBN: {pbn} != {dpbn}") # often a sort order issue.
    return dpbn

def convert_ffdf_to_mldf(ffdf):

    # assignments are broken into parts for polars compatibility (could be parallelized).
    df = ffdf.select([
        pl.col('group_id'),
        pl.col('board_id'),
        pl.col('team_session_id'),
        pl.col('team_id'),
        pl.col('session_id'),
        pl.col('boardNumber').alias('Board'),
        #pl.col('board_frequencies'),
        # flatten the board_frequencies column into multiple columns
        # todo: need to generalize this to work for any json column with a list of structs. either do it here or in previous step.
        pl.col('board_frequencies').list.eval(pl.element().struct.field('nsScore')).alias('Scores_List_NS'),
        pl.col('board_frequencies').list.eval(pl.element().struct.field('nsNote')).alias('Pcts_List_NS'),
        pl.col('board_frequencies').list.eval(pl.element().struct.field('ewScore')).alias('Scores_List_EW'),
        pl.col('board_frequencies').list.eval(pl.element().struct.field('ewNote')).alias('Pcts_List_EW'),
        pl.col('board_frequencies').list.eval(pl.element().struct.field('count')).alias('Score_Freq_List'),
        # todo: have not solved the issue of which direction is dealer using ffbridge. I'm missing something but don't know what.
        # derive dealer from boardNumber which works for standard bridge boards but isn't guaranteed to work for all boards and events.
        pl.col('boardNumber')
            .map_elements(BoardNumberToDealer,return_dtype=pl.String)
            #.alias('DealerFromBoardNumber'),
            .alias('Dealer'),
        #pl.col('board_deal').str.slice(0, 1).alias('Dealer'),  # Use first character of 'board_deal' to create Dealer column.
        pl.col('board_deal')
            .map_elements(PbnToN,return_dtype=pl.String)
            .alias('PBN'),
        pl.col('boardNumber')
            .map_elements(BoardNumberToVul,return_dtype=pl.UInt8)
            .replace_strict({
                0: 'None',
                1: 'N_S',
                2: 'E_W',
                3: 'Both'
        })
        .alias('Vul'),
        pl.concat_str([
            pl.col('contract').str.replace('NT', 'N').str.slice(0,2),
            pl.col('declarer'),
            pl.col('contract').str.replace('NT', 'N').str.slice(2)
        ]).alias('Contract'),
        pl.when(pl.col('result').str.starts_with('+'))
            .then(pl.col('result').str.slice(1))  # Remove '+'
            .when(pl.col('result').str.starts_with('-'))
            .then(pl.col('result'))
            .otherwise(pl.lit('0'))  # Replace '=' with '0'
            .cast(pl.Int16)
            .alias('Result'),
        # not liking that only one of the two columns has a value. I prefer to have both with opposite signs.
        # although this may be an issue for director adjustments.
        pl.when(pl.col('nsScore').str.contains(r'^\d'))
            .then(pl.col('nsScore'))
            .when(pl.col('ewScore').str.contains(r'^\d'))
            .then(pl.lit('-')+pl.col('ewScore'))
            .otherwise(pl.lit('0'))
            .cast(pl.Int16)
            .alias('Score'),
        (pl.col('nsNote')/100.0).alias('Pct_NS'),
        (pl.col('ewNote')/100.0).alias('Pct_EW'),
        pl.col('player1_id'),
        pl.col('player1_firstName'),
        pl.col('player1_lastName'),
        pl.col('player2_id'),
        pl.col('player2_firstName'),
        pl.col('player2_lastName'),
        (pl.col('lineup_northPlayer_firstName')+pl.lit(' ')+pl.col('lineup_northPlayer_lastName')).alias('Player_Name_N'),
        (pl.col('lineup_eastPlayer_firstName')+pl.lit(' ')+pl.col('lineup_eastPlayer_lastName')).alias('Player_Name_E'),
        (pl.col('lineup_southPlayer_firstName')+pl.lit(' ')+pl.col('lineup_southPlayer_lastName')).alias('Player_Name_S'),
        (pl.col('lineup_westPlayer_firstName')+pl.lit(' ')+pl.col('lineup_westPlayer_lastName')).alias('Player_Name_W'),
        pl.col('lineup_northPlayer_id').cast(pl.String).alias('Player_ID_N'),
        pl.col('lineup_eastPlayer_id').cast(pl.String).alias('Player_ID_E'),
        pl.col('lineup_southPlayer_id').cast(pl.String).alias('Player_ID_S'),
        pl.col('lineup_westPlayer_id').cast(pl.String).alias('Player_ID_W'),
        pl.col('lineup_segment_game_homeTeam_section').alias('section_id_home'),
        pl.col('lineup_segment_game_homeTeam_orientation').alias('Pair_Direction_Home'),
        pl.col('lineup_segment_game_homeTeam_startTableNumber').alias('Pair_Number_Home'),
        pl.col('lineup_segment_game_awayTeam_section').alias('section_id_away'),
        pl.col('lineup_segment_game_awayTeam_orientation').alias('Pair_Direction_Away'),
        pl.col('lineup_segment_game_awayTeam_startTableNumber').alias('Pair_Number_Away'),
    ])
    assert all(df['section_id_home'] == df['section_id_away'])
    assert all(df['Pair_Direction_Home'].is_in(['NS',''])), df['Pair_Direction_Home'].value_counts() # '' is sitout
    assert all(df['Pair_Direction_Away'].is_in(['EW',''])), df['Pair_Direction_Away'].value_counts() # '' is sitout
    df = df.with_columns(
        pl.col('section_id_home').alias('section_name'),
        pl.col('Score_Freq_List').list.sum().sub(1).alias('MP_Top'),
        pl.col('Pair_Direction_Home').alias('Pair_Direction_NS'),
        pl.col('Pair_Direction_Away').alias('Pair_Direction_EW'),
        pl.col('Pair_Number_Home').alias('Pair_Number_NS'),
        pl.col('Pair_Number_Away').alias('Pair_Number_EW'),
        pl.col('section_id_home')+pl.lit('_')+pl.col('Pair_Direction_Home')+pl.col('Pair_Number_Home').cast(pl.Utf8).alias('Pair_ID_NS'),
        pl.col('section_id_away')+pl.lit('_')+pl.col('Pair_Direction_Away')+pl.col('Pair_Number_Away').cast(pl.Utf8).alias('Pair_ID_EW'),
    )
    df = df.with_columns(
        pl.struct(['Scores_List_NS', 'Scores_List_EW', 'Score_Freq_List'])\
            .map_elements(lambda x: [int(score_ns) if len(score_ns) else int('-'+score_ew) for score_ns, score_ew, freq in zip(x['Scores_List_NS'], x['Scores_List_EW'], x['Score_Freq_List']) for _ in range(freq)],return_dtype=pl.List(pl.Int16))
            .alias('Expanded_Scores_List')
    )
    df = df.with_columns(
        (pl.col('Pct_NS')*pl.col('MP_Top')).round(2).alias('MP_NS'),
        (pl.col('Pct_EW')*pl.col('MP_Top')).round(2).alias('MP_EW'),
    )

    return df
