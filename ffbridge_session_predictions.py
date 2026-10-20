"""Session predictions and pair rankings for the FFBridge postmortem.

Club rank is the place among pairs at the same club. Predicted club rank
is the place of the pair's predicted session percentage among those pairs.
Pairs with the same predicted percentage share a place. ``Global_Scratch_Rank`` is
the official ``rank`` on FFBridge's session ranking. ``Global_Scratch_Rank_Pred`` is
where the predicted percentage would sit among the other pairs' published
percentages in the national field.

Every diff is actual minus predicted: ``Rank_Diff`` is ``Global_Scratch_Rank`` minus
``Global_Scratch_Rank_Pred`` and ``Pct_Diff`` is the published percentage minus the
predicted one.

``Pct_NS_Pred`` is the percentage model's North-South score on the board,
from the deal, the four players' pre-session ratings, and their records
before the session date. East-West receive the complement.

``Contract_Pred`` and ``Declarer_Direction_Pred`` are one choice: the
contract that maximizes the contract model's probability times the declarer
model's probability for that contract's declarer.
"""

from __future__ import annotations

import functools
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

_HOST_MODELS_DIR = Path(r"E:\bridge\data\ffbridge\SavedModels")
# postmortem_start.ps1 stages the host checkpoints onto the /data mount.
_STAGED_MODELS_DIR = Path("/data/_wslc_host/ffbridge-SavedModels")
_PUBLISHED_MODELS_DIR = Path("/data/ffbridge/SavedModels")
_ELO_CACHE_DIRS = (
    Path("/data/ffbridge/elo_cache"),
    Path(__file__).resolve().parent.parent / "data" / "ffbridge" / "elo_cache",
    Path(r"E:\bridge\data\ffbridge\elo_cache"),
)
TARGETS = ("Declarer_Direction", "Contract", "Pct_NS")
PLAYER_HISTORY_NAME = "ffbridge_prediction_player_history.parquet"
PAIR_HISTORY_NAME = "ffbridge_prediction_pair_history.parquet"
RANKING_COLUMNS = (
    "Pair",
    "Orientation",
    "Club_Rank",
    "Club_Rank_Pred",
    "Global_Scratch_Rank",
    "Global_Scratch_Rank_Pred",
    "Rank_Diff",
    "Global_Scratch_Pct",
    "Global_Scratch_Pct_Pred",
    "Pct_Diff",
    "Handicap_Pct",
    "Handicap_Club_Rank",
    "Handicap_Global_Rank",
    "Boards",
)
BOARD_COLUMNS = (
    "Board",
    "Pair_NS",
    "Pair_EW",
    "Contract",
    "Contract_Pred",
    "Declarer_Direction",
    "Declarer_Direction_Pred",
    "Pct_NS",
    "Pct_NS_Pred",
    "Pct_Diff",
)
# Same ordered codes as ffbridge_prediction_data.CT_CODES. The postmortem
# stores the words; the checkpoints were trained on the numbers.
_CT_CODES = {"Pass": 0.0, "Partial": 1.0, "Game": 2.0, "SSlam": 3.0, "GSlam": 4.0}
_PREDICTION_SQL_MARKERS = (
    "pct_ns_pred",
    "contract_pred",
    "declarer_direction_pred",
    "global_scratch_",
    "global_rank",
    "club_rank_pred",
    "rank_diff",
)


def _checkpoints_present(directory: Path) -> bool:
    return any(
        (directory / f"{_model_name(target)}_schema.json").is_file()
        for target in TARGETS
    )


def models_dir() -> Path:
    """First directory that actually contains the club checkpoints.

    A configured path that does not exist is skipped. The container cannot
    see ``E:\\bridge\\data``, so the staged mount is checked as well.
    """
    configured = os.environ.get("FFBRIDGE_SAVED_MODELS")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend((_STAGED_MODELS_DIR, _PUBLISHED_MODELS_DIR, _HOST_MODELS_DIR))
    for directory in candidates:
        if _checkpoints_present(directory):
            return directory
    return Path(configured) if configured else _HOST_MODELS_DIR


def _elo_results_path() -> Optional[Path]:
    configured = os.environ.get("FFBRIDGE_ELO_RESULTS")
    if configured:
        path = Path(configured)
        return path if path.is_file() else None
    found = [
        path
        for directory in _ELO_CACHE_DIRS
        if directory.is_dir()
        for path in directory.glob("*Lancelot_API*.results.parquet")
    ]
    return max(found, key=lambda path: path.stat().st_mtime) if found else None


def _id_token(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.strip_chars()
        .str.replace(r"\.0$", "")
    )


def _attach_elo_features(frame: pl.DataFrame, session_id: Any) -> pl.DataFrame:
    """Attach the pre-session seat and pair Elo used by the trained models.

    ``scratch_pair_elo`` is the mean of the two players' *after* ratings, so
    it contains this session's result. The pair rating is the mean of the two
    before ratings instead, as in training.
    """
    path = _elo_results_path()
    if path is None:
        raise FileNotFoundError("FFBridge Elo results parquet was not found")
    required_ids = [f"Player_ID_{seat}" for seat in "NESW"]
    missing = [name for name in required_ids if name not in frame.columns]
    if missing:
        raise ValueError(f"Postmortem is missing player ids required for Elo: {missing}")

    available = set(pl.scan_parquet(path).collect_schema().names())
    # Postmortem seat ids are Lancelot ids. The Elo file's player*_id is a
    # different person id, so a lookup on that column alone misses every seat
    # and the model then sees the training-mean Elo at every table.
    id_pairs = [
        pair
        for pair in (
            ("player1_lancelot_id", "player2_lancelot_id"),
            ("player1_id", "player2_id"),
            ("player1_classic_person_id", "player2_classic_person_id"),
        )
        if pair[0] in available and pair[1] in available
    ]
    if not id_pairs:
        raise ValueError("FFBridge Elo results have no player id columns")
    results = (
        pl.scan_parquet(path)
        .filter(pl.col("tournament_id").cast(pl.Utf8) == str(session_id))
        .select(
            pl.col("player1_scratch_elo_before").cast(pl.Float32).alias("e1"),
            pl.col("player2_scratch_elo_before").cast(pl.Float32).alias("e2"),
            *[_id_token(name).alias(name) for pair in id_pairs for name in pair],
        )
        .with_columns(((pl.col("e1") + pl.col("e2")) / 2.0).alias("pair_elo"))
        .collect()
    )
    if results.is_empty():
        raise ValueError(f"FFBridge Elo has no pre-session ratings for session {session_id}")

    players = (
        pl.concat(
            [
                results.select(pl.col(column).alias("id"), pl.col(elo).alias("elo"))
                for left, right in id_pairs
                for column, elo in ((left, "e1"), (right, "e2"))
            ]
        )
        .filter(pl.col("id") != "")
        .group_by("id")
        .agg(pl.col("elo").mean())
    )
    player_ratings = dict(players.iter_rows())
    pairs = (
        pl.concat(
            [
                results.select(
                    _pair_key(pl.col(left), pl.col(right)).alias("key"),
                    pl.col("pair_elo"),
                )
                for left, right in id_pairs
            ]
        )
        .filter(pl.col("key").is_not_null())
        .group_by("key")
        .agg(pl.col("pair_elo").mean())
    )
    pair_ratings = dict(pairs.iter_rows())
    out = frame.with_columns(
        *[
            _id_token(f"Player_ID_{seat}")
            .replace_strict(player_ratings, default=None)
            .cast(pl.Float32)
            .alias(f"Elo_{seat}")
            for seat in "NESW"
        ],
        _pair_key(pl.col("Player_ID_N"), pl.col("Player_ID_S"))
        .replace_strict(pair_ratings, default=None)
        .cast(pl.Float32)
        .alias("Elo_NS"),
        _pair_key(pl.col("Player_ID_E"), pl.col("Player_ID_W"))
        .replace_strict(pair_ratings, default=None)
        .cast(pl.Float32)
        .alias("Elo_EW"),
    )
    if all(out[f"Elo_{seat}"].null_count() == out.height for seat in "NESW"):
        raise ValueError(
            f"FFBridge Elo ids did not match the players in session {session_id}"
        )
    return out


@functools.lru_cache(maxsize=2)
def _classic_ids(path: Path, modified: float) -> Dict[str, str]:
    """Lancelot person id to the Classic id the history tables are keyed on."""
    scan = pl.scan_parquet(path)
    pairs = (
        pl.concat(
            [
                scan.select(
                    _id_token(f"player{index}_lancelot_id").alias("lancelot"),
                    _id_token(f"player{index}_id").alias("classic"),
                )
                for index in (1, 2)
            ]
        )
        .filter((pl.col("lancelot") != "") & (pl.col("classic") != ""))
        .unique()
        .collect()
    )
    conflicts = pairs.group_by("lancelot").len().filter(pl.col("len") > 1).height
    if conflicts:
        raise ValueError(f"{conflicts} Lancelot ids map to more than one Classic id in {path.name}")
    return dict(pairs.iter_rows())


@functools.lru_cache(maxsize=2)
def _history_tables(directory: Path, modified: Tuple[float, float]) -> Tuple[pl.DataFrame, pl.DataFrame]:
    return (
        pl.read_parquet(directory / PLAYER_HISTORY_NAME),
        pl.read_parquet(directory / PAIR_HISTORY_NAME),
    )


def _add_model_inputs(frame: pl.DataFrame, required: set, directory: Path) -> pl.DataFrame:
    """Suit lengths and player history on a copy used only as model input."""
    from mlBridge.ffbridge_prediction_features import (
        HISTORY_FEATURES,
        SHAPE_FEATURES,
        attach_history,
        shape_features,
    )

    out = frame
    if required & (set(SHAPE_FEATURES) | {"SL_N_Longest"}):
        out = shape_features(out)
    if "SL_N_Longest" in required:
        # Training reads this from the hand file's SL_N_ML_SJ, an integer.
        # The postmortem's SL_N_ML_SJ is the shape string ("4-3-3-3").
        out = out.with_columns(
            pl.max_horizontal(f"SL_N_{suit}" for suit in "SHDC").cast(pl.Float32).alias("SL_N_Longest")
        )
    if required & set(HISTORY_FEATURES):
        elo_path = _elo_results_path()
        if elo_path is None:
            raise FileNotFoundError("FFBridge Elo results parquet was not found")
        classic = _classic_ids(elo_path, elo_path.stat().st_mtime)
        paths = (directory / PLAYER_HISTORY_NAME, directory / PAIR_HISTORY_NAME)
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"History tables not found in {directory}: {missing}")
        players, pairs = _history_tables(directory, tuple(path.stat().st_mtime for path in paths))
        out = out.with_columns(
            _id_token(f"Player_ID_{seat}").replace_strict(classic, default=None).alias(f"_classic_{seat}")
            for seat in "NESW"
        )
        if all(out[f"_classic_{seat}"].null_count() == out.height for seat in "NESW"):
            raise ValueError("No postmortem player id has a Classic id in the Elo results")
        out = attach_history(out, players, pairs, {seat: f"_classic_{seat}" for seat in "NESW"})
    return out


def _schema_classes(schema: Dict[str, Any]) -> List[str]:
    class_to_idx = schema["class_to_idx"]
    return sorted(class_to_idx, key=class_to_idx.get)


def sql_needs_predictions(sql: str) -> bool:
    lowered = sql.lower()
    return any(marker in lowered for marker in _PREDICTION_SQL_MARKERS)


def _model_name(target: str) -> str:
    return f"ffbridge_club_predicted_{target.lower()}_torch_model"


def _dtype_from_schema(dtype_name: str) -> pl.DataType:
    text = str(dtype_name)
    if "Float64" in text:
        return pl.Float64
    if "Float" in text:
        return pl.Float32
    if "UInt" in text:
        return pl.UInt32
    if "Int" in text:
        return pl.Int32
    if "Boolean" in text or "Bool" in text:
        return pl.Boolean
    if "Date" in text:
        return pl.Date
    return pl.Utf8


def _align_features(
    frame: pl.DataFrame,
    feature_dtypes: Dict[str, Any],
    required: List[str],
    scaling_params: Optional[Dict[str, Any]] = None,
) -> pl.DataFrame:
    """Cast model inputs. Contract-type words become the trained numeric codes.

    Missing ratings take the schema mean, as in training. ``Elo_Gap`` is then
    taken from the filled pair ratings, and the ``*_Missing`` flags must
    already be on the frame (see ``_add_elo_missing_flags``).
    """
    missing = [column for column in required if column != "Elo_Gap" and column not in frame.columns]
    if missing:
        raise ValueError(f"Model input is missing {missing}")
    aligned = frame
    for column in required:
        if column == "Elo_Gap":
            continue
        expected = _dtype_from_schema(str(feature_dtypes.get(column, "Float32")))
        dtype = aligned.schema[column]
        if column.startswith("Elo_") and expected in (pl.Float32, pl.Float64):
            mean = (scaling_params or {}).get(column, {}).get("mean")
            if mean is None:
                raise ValueError(f"Model schema has no neutral value for {column}")
            aligned = aligned.with_columns(
                pl.col(column)
                .cast(expected, strict=False)
                .fill_null(float(mean))
                .fill_nan(float(mean))
                .alias(column)
            )
            continue
        if expected in (pl.Float32, pl.Float64) and dtype in (pl.Utf8, pl.String):
            aligned = aligned.with_columns(
                pl.col(column)
                .cast(pl.Utf8)
                .replace_strict(_CT_CODES, default=None)
                .cast(pl.Float32)
                .fill_null(0.0)
                .alias(column)
            )
            continue
        if dtype != expected and expected == pl.Utf8:
            aligned = aligned.with_columns(pl.col(column).cast(pl.Utf8, strict=False))
        elif dtype != expected and expected in (pl.Float32, pl.Float64) and dtype.is_numeric():
            aligned = aligned.with_columns(pl.col(column).cast(expected))
    if "Elo_Gap" in required:
        if "Elo_NS" not in required or "Elo_EW" not in required:
            raise ValueError("Model uses Elo_Gap without Elo_NS and Elo_EW")
        expected = _dtype_from_schema(str(feature_dtypes.get("Elo_Gap", "Float32")))
        aligned = aligned.with_columns(
            (pl.col("Elo_NS") - pl.col("Elo_EW")).cast(expected).alias("Elo_Gap")
        )
    return aligned


def _add_elo_missing_flags(frame: pl.DataFrame) -> pl.DataFrame:
    """1.0 where a pair rating is missing, taken before any fill."""
    return frame.with_columns(
        pl.col(f"Elo_{pair}").is_null().cast(pl.Float32).alias(f"Elo_{pair}_Missing")
        for pair in ("NS", "EW")
    )


def _predict_targets(
    frame: pl.DataFrame,
    session_id: Any = None,
) -> Tuple[pl.DataFrame, Optional[str]]:
    """Add model outputs. Missing checkpoints leave the prediction columns null."""
    directory = models_dir()
    missing = [
        target
        for target in TARGETS
        if not (directory / f"{_model_name(target)}_schema.json").is_file()
    ]
    out = frame
    if missing:
        warning = (
            f"Prediction models not found in {directory} "
            f"({', '.join(missing)}). Rankings use published scores only."
        )
        return _null_prediction_columns(out), warning
    schemas = {
        target: json.loads(
            (directory / f"{_model_name(target)}_schema.json").read_text(encoding="utf-8")
        )
        for target in TARGETS
    }
    requires_elo = any(
        str(name).startswith("Elo_")
        for schema in schemas.values()
        for name in schema.get("numerical_feature_cols") or []
    )
    elo_columns = {f"Elo_{seat}" for seat in "NESW"} | {"Elo_NS", "Elo_EW"}
    if requires_elo and not elo_columns.issubset(out.columns):
        try:
            out = _attach_elo_features(out, session_id)
        except Exception as exc:
            return _null_prediction_columns(out), f"Prediction Elo unavailable: {exc}"
    if requires_elo:
        out = _add_elo_missing_flags(out)
    try:
        from mlBridge.ffbridge_prediction_features import joint_contract_and_declarer
        from mlBridge.mlBridgeAiLib import predict_model
    except Exception as exc:
        return _null_prediction_columns(out), f"Prediction library unavailable: {exc}"

    all_required = {
        str(name)
        for schema in schemas.values()
        for name in (schema.get("numerical_feature_cols") or []) + (schema.get("categorical_feature_cols") or [])
    }
    try:
        model_input = _add_model_inputs(out, all_required, directory)
    except Exception as exc:
        return _null_prediction_columns(out), f"Prediction features unavailable: {exc}"

    warnings: List[str] = []
    probabilities: Dict[str, Any] = {}
    for target in TARGETS:
        schema = schemas[target]
        feature_dtypes = schema.get("feature_dtypes") or {}
        scaling_params = schema.get("scaling_params") or {}
        required = list(schema.get("numerical_feature_cols") or []) + list(
            schema.get("categorical_feature_cols") or []
        )
        classifier = target != "Pct_NS"
        try:
            aligned = _align_features(model_input, feature_dtypes, required, scaling_params)
            predicted = predict_model(
                directory,
                _model_name(target),
                aligned,
                device="cpu",
                return_probs=classifier,
                strict_validation=False,
                parity_drop_non_finite=False,
            )
        except Exception as exc:
            warnings.append(f"{target}: {exc}")
            continue
        pred_column = f"{target}_Pred"
        if pred_column not in predicted.columns or predicted.height != out.height:
            warnings.append(f"{target}: prediction row count did not match the session")
            continue
        if classifier:
            classes = _schema_classes(schema)
            probabilities[target] = (
                classes,
                predicted[f"{target}_Probs"].list.to_array(len(classes)).to_numpy(),
            )
        if pred_column in out.columns:
            out = out.drop(pred_column)
        out = out.hstack(predicted.select(pred_column))
    if "Contract" in probabilities and "Declarer_Direction" in probabilities:
        contracts, directions = joint_contract_and_declarer(
            *probabilities["Contract"], *probabilities["Declarer_Direction"]
        )
        out = out.with_columns(
            pl.Series("Contract_Pred", contracts, dtype=pl.Utf8),
            pl.Series("Declarer_Direction_Pred", directions, dtype=pl.Utf8),
        )
    out = _assign_ns_prediction(out)
    warning = "; ".join(warnings) if warnings else None
    return out, warning


def _null_prediction_columns(frame: pl.DataFrame) -> pl.DataFrame:
    columns = {
        "Declarer_Direction_Pred": pl.Utf8,
        "Contract_Pred": pl.Utf8,
        "Pct_NS_Pred": pl.Float64,
        "Pct_Diff": pl.Float64,
    }
    exprs = [
        pl.lit(None).cast(dtype).alias(name)
        for name, dtype in columns.items()
        if name not in frame.columns
    ]
    return frame.with_columns(exprs) if exprs else frame


def _assign_ns_prediction(frame: pl.DataFrame) -> pl.DataFrame:
    """Use the NS percentage model as is.

    The scores are national percentages, and clubs often seat their stronger
    pairs in one direction. Centering per board or per session forces the NS
    and EW lines to the same average and ranked 2026 club sessions worse.
    """
    if "Pct_NS_Pred" not in frame.columns:
        return _null_prediction_columns(frame)
    prediction = pl.col("Pct_NS_Pred").cast(pl.Float64, strict=False).clip(0.0, 1.0)
    actual = (
        pl.col("Pct_NS").cast(pl.Float64, strict=False)
        if "Pct_NS" in frame.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    return frame.with_columns(
        prediction.alias("Pct_NS_Pred"),
        (actual - prediction).alias("Pct_Diff"),
    )


_SCORE_COLUMNS = ("sessionScore", "session_score", "percentage")


def _ranking_frame(payload: Any) -> Optional[pl.DataFrame]:
    if isinstance(payload, dict):
        payload = payload.get("items") or payload.get("ranking") or [payload]
    if not payload:
        return None
    return pl.json_normalize(payload, separator="_", infer_schema_length=None)


def ranking_is_published(ranking: Optional[pl.DataFrame]) -> bool:
    """FFBridge lists every pair at 0% and 1st place until it publishes results."""
    if ranking is None or ranking.is_empty():
        return False
    score_col = _first_present(ranking, _SCORE_COLUMNS)
    if score_col is None:
        return False
    scores = ranking[score_col].cast(pl.Float64, strict=False).fill_null(0.0)
    return bool((scores != 0.0).any())


def _fetch_ranking(session_id: Any) -> Any:
    import mlBridge.mlBridgeFFLib as mlBridgeFFLib

    return mlBridgeFFLib.lancelot_get(f"results/sessions/{int(session_id)}/ranking")


def load_ranking(session_id: Any, cache_dir: Path) -> Tuple[Optional[pl.DataFrame], Optional[str]]:
    """The national ranking and a warning when FFBridge has not published it.

    A ranking cached before publication is downloaded again, and the cache is
    rewritten once the published scores arrive.
    """
    path = Path(cache_dir) / "rankings" / f"{session_id}.json"
    if path.is_file():
        ranking = _ranking_frame(json.loads(path.read_text(encoding="utf-8")))
        if ranking_is_published(ranking):
            return ranking, None
    payload = _fetch_ranking(session_id)
    ranking = _ranking_frame(payload)
    if not ranking_is_published(ranking):
        return None, (
            f"FFBridge has not published results for session {session_id} yet, "
            "so actual percentages and ranks are not shown."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ranking, None


def _text(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8)


def _first_present(frame: pl.DataFrame, names: Tuple[str, ...]) -> Optional[str]:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _canonical_key(left: Any, right: Any) -> Optional[str]:
    """Order-independent id pair. Empty when either player id is missing."""
    def token(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        return text

    first, second = token(left), token(right)
    if not first or not second:
        return None
    return first + "|" + second if first <= second else second + "|" + first


def _pair_key(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    def token(expr: pl.Expr) -> pl.Expr:
        return (
            expr.cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.strip_chars()
            .str.replace(r"\.0$", "")
        )

    first, second = token(left), token(right)
    combined = (
        pl.when(first <= second)
        .then(first + pl.lit("|") + second)
        .otherwise(second + pl.lit("|") + first)
    )
    return pl.when((first == "") | (second == "")).then(pl.lit(None)).otherwise(combined)


def _pair_label(section: pl.Expr, table: pl.Expr, left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return (
        pl.coalesce([section.fill_null(""), pl.lit("")])
        + table.cast(pl.Utf8)
        + pl.lit(" - ")
        + left.fill_null("")
        + pl.lit(" - ")
        + right.fill_null("")
    )


def _rank_against(score: Optional[float], field: List[float]) -> Optional[int]:
    if score is None:
        return None
    better = sum(1 for other in field if other > score)
    return better + 1


def _mean(values: List[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def ranking_rows(boards: pl.DataFrame, ranking: Optional[pl.DataFrame]) -> pl.DataFrame:
    """One row per ranked pair: club ranks, then global ranks, then percentages.

    The actual percentages and ranks are scratch, like the predictions. In a
    handicapped session FFBridge's published figures follow as Handicap_*.
    """
    official = _official_pairs(ranking)
    predicted = _predicted_pair_scores(boards)
    if official.is_empty() and predicted.is_empty():
        return pl.DataFrame({name: [] for name in RANKING_COLUMNS})

    national_scores = [
        float(value)
        for value in (official["Global_Scratch_Pct"].drop_nulls().to_list() if official.height else [])
    ]
    rows: List[Dict[str, Any]] = []
    club_groups: Dict[str, List[Dict[str, Any]]] = {}
    keys_seen = set()

    def add_row(raw: Dict[str, Any]) -> None:
        key = (raw.get("Club"), raw.get("Orientation"), raw.get("Table"))
        if key in keys_seen:
            return
        keys_seen.add(key)
        club_groups.setdefault(str(raw.get("Club")), []).append(raw)
        rows.append(raw)

    home_club = None
    if official.height:
        officials = official.to_dicts()
        matches = [_match_prediction(predicted, raw) for raw in officials]
        # The postmortem is one club's boards. A row from another club can
        # still match by table number when its player ids are missing.
        matched_clubs = Counter(raw.get("Club") for raw, pred in zip(officials, matches) if pred is not None)
        home_club = matched_clubs.most_common(1)[0][0] if matched_clubs else None
        for raw, pred in zip(officials, matches):
            if raw.get("Club") != home_club:
                pred = None
            raw["Global_Scratch_Pct_Pred"] = None if pred is None else pred.get("Global_Scratch_Pct_Pred")
            raw["Boards"] = None if pred is None else pred.get("Boards")
            if pred is not None and pred.get("Pair"):
                raw["Pair"] = pred["Pair"]
            add_row(raw)
    else:
        for raw in predicted.to_dicts():
            raw["Global_Scratch_Rank"] = None
            raw["Club"] = raw.get("Section") or ""
            raw["Global_Scratch_Pct"] = None
            add_row(raw)

    for raw in rows:
        for pct_name, rank_name in (("Global_Scratch_Pct", "Club_Rank"), ("Handicap_Pct", "Handicap_Club_Rank")):
            club_field = [
                float(item[pct_name])
                for item in club_groups.get(str(raw.get("Club")), [])
                if item.get(pct_name) is not None
            ]
            raw[rank_name] = _rank_against(
                None if raw.get(pct_name) is None else float(raw[pct_name]),
                club_field,
            )
        pred_score = raw.get("Global_Scratch_Pct_Pred")
        if pred_score is None:
            raw["Club_Rank_Pred"] = None
        else:
            # Rank the displayed percentage. Smaller leftovers from playing a
            # slightly different set of the same deals must not become a
            # distinct place for every table.
            shown = round(float(pred_score), 2)
            club_pred_field = [
                round(float(item["Global_Scratch_Pct_Pred"]), 2)
                for item in club_groups.get(str(raw.get("Club")), [])
                if item.get("Global_Scratch_Pct_Pred") is not None
            ]
            raw["Club_Rank_Pred"] = _rank_against(shown, club_pred_field)
        if pred_score is None:
            raw["Global_Scratch_Rank_Pred"] = None
            raw["Rank_Diff"] = None
            raw["Pct_Diff"] = None
            continue
        pred_score = float(pred_score)
        national_pred_field = []
        for item in rows:
            if item is raw:
                national_pred_field.append(pred_score)
            elif item.get("Global_Scratch_Pct") is not None:
                national_pred_field.append(float(item["Global_Scratch_Pct"]))
        raw["Global_Scratch_Rank_Pred"] = _rank_against(pred_score, national_pred_field or national_scores)
        official_rank = raw.get("Global_Scratch_Rank")
        raw["Rank_Diff"] = (
            None if official_rank is None or raw["Global_Scratch_Rank_Pred"] is None
            else int(official_rank) - int(raw["Global_Scratch_Rank_Pred"])
        )
        actual = raw.get("Global_Scratch_Pct")
        raw["Pct_Diff"] = None if actual is None else round(float(actual) - pred_score, 2)
        raw["Global_Scratch_Pct_Pred"] = round(pred_score, 2)

    # Other clubs were needed to rank against the national field, but the
    # postmortem reports only the home club's pairs.
    if home_club is not None:
        rows = [raw for raw in rows if raw.get("Club") == home_club]
    ranked = pl.DataFrame(rows) if rows else pl.DataFrame({name: [] for name in RANKING_COLUMNS})
    for name in (*RANKING_COLUMNS, "Section", "Table", "Club"):
        if name not in ranked.columns:
            ranked = ranked.with_columns(pl.lit(None).alias(name))
    ordered = [
        name
        for name in (*RANKING_COLUMNS, "Section", "Table", "Club", "Left_Id", "Right_Id")
        if name in ranked.columns
    ]
    return ranked.select(ordered).sort(
        ["Club_Rank", "Global_Scratch_Rank", "Club_Rank_Pred"],
        nulls_last=True,
    )


def _official_pairs(ranking: Optional[pl.DataFrame]) -> pl.DataFrame:
    if ranking is None or ranking.is_empty():
        return pl.DataFrame()
    score_col = _first_present(ranking, ("sessionScore", "session_score", "percentage"))
    rank_col = _first_present(ranking, ("rank", "general_rank", "generalRank"))
    club_col = _first_present(ranking, ("simultaneousId", "club_code"))
    orientation_col = _first_present(ranking, ("orientation", "team_orientation"))
    table_col = _first_present(ranking, ("tableNumber", "team_startTableNumber", "table_number"))
    section_col = _first_present(ranking, ("section", "team_section"))
    if score_col is None or orientation_col is None or table_col is None:
        return pl.DataFrame()
    first_left = _first_present(ranking, ("team_player1_firstName", "team_player1_firstname"))
    last_left = _first_present(ranking, ("team_player1_lastName", "team_player1_lastname"))
    first_right = _first_present(ranking, ("team_player2_firstName", "team_player2_firstname"))
    last_right = _first_present(ranking, ("team_player2_lastName", "team_player2_lastname"))
    left = (
        (pl.col(first_left).fill_null("") + pl.lit(" ") + pl.col(last_left).fill_null("")).str.strip_chars()
        if first_left and last_left
        else pl.lit("")
    )
    right = (
        (pl.col(first_right).fill_null("") + pl.lit(" ") + pl.col(last_right).fill_null("")).str.strip_chars()
        if first_right and last_right
        else pl.lit("")
    )
    section = pl.col(section_col).cast(pl.Utf8) if section_col else pl.lit("")
    table = pl.col(table_col).cast(pl.Utf8)
    id_left = _first_present(ranking, ("team_player1_id", "player1_id"))
    id_right = _first_present(ranking, ("team_player2_id", "player2_id"))
    left_id = pl.col(id_left).cast(pl.Utf8) if id_left else pl.lit(None).cast(pl.Utf8)
    right_id = pl.col(id_right).cast(pl.Utf8) if id_right else pl.lit(None).cast(pl.Utf8)
    score = pl.col(score_col).cast(pl.Float64, strict=False)
    rank = pl.col(rank_col).cast(pl.Int64, strict=False) if rank_col else pl.lit(None).cast(pl.Int64)
    # In a handicapped session sessionScore and rank include the IV bonus
    # (totalBonus, percentage points) and the predictions do not.
    bonus_col = _first_present(ranking, ("totalBonus",))
    unbonused_col = _first_present(ranking, ("totalScoreWithoutHandicap",))
    unbonused = (
        pl.col(unbonused_col).cast(pl.Float64, strict=False) if unbonused_col else pl.lit(None, dtype=pl.Float64)
    )
    bonus = pl.col(bonus_col).cast(pl.Float64, strict=False) if bonus_col else pl.lit(None, dtype=pl.Float64)
    handicapped = ranking.select(
        ((bonus.fill_null(0.0) > 0.0) | unbonused.is_not_null()).any()
    ).item()
    if handicapped:
        scratch = pl.coalesce([unbonused, score - bonus]).round(2)
        unbonused_rank_col = _first_present(ranking, ("rankWithoutHandicap",))
        scratch_rank = pl.coalesce([
            pl.col(unbonused_rank_col).cast(pl.Int64, strict=False) if unbonused_rank_col else pl.lit(None).cast(pl.Int64),
            scratch.rank("min", descending=True).cast(pl.Int64),
        ])
        handicap, handicap_rank = score, rank
    else:
        scratch, scratch_rank = score, rank
        handicap, handicap_rank = pl.lit(None, dtype=pl.Float64), pl.lit(None).cast(pl.Int64)
    return ranking.select(
        _pair_label(section, table, left, right).alias("Pair"),
        pl.col(orientation_col).cast(pl.Utf8).alias("Orientation"),
        scratch.alias("Global_Scratch_Pct"),
        scratch_rank.alias("Global_Scratch_Rank"),
        handicap.alias("Handicap_Pct"),
        handicap_rank.alias("Handicap_Global_Rank"),
        (pl.col(club_col).cast(pl.Utf8) if club_col else pl.lit("")).alias("Club"),
        section.alias("Section"),
        table.alias("Table"),
        left_id.alias("Left_Id"),
        right_id.alias("Right_Id"),
    )


def _predicted_pair_scores(boards: pl.DataFrame) -> pl.DataFrame:
    """One predicted session percentage per pair.

    North-South get the mean of ``Pct_NS_Pred``. East-West get the complement,
    which is their score on the same deals.
    """
    frames = []
    for orientation, number_col, name_a, name_b, pct_expr in (
        ("NS", "Pair_Number_NS", "Player_Name_N", "Player_Name_S", pl.col("Pct_NS_Pred")),
        ("EW", "Pair_Number_EW", "Player_Name_E", "Player_Name_W", 1.0 - pl.col("Pct_NS_Pred")),
    ):
        if number_col not in boards.columns:
            continue
        section_col = "section_name" if "section_name" in boards.columns else (
            "Section_Name" if "Section_Name" in boards.columns else None
        )
        section = pl.col(section_col).cast(pl.Utf8) if section_col else pl.lit("")
        left = pl.col(name_a) if name_a in boards.columns else pl.lit("")
        right = pl.col(name_b) if name_b in boards.columns else pl.lit("")
        id_a, id_b = ("Player_ID_N", "Player_ID_S") if orientation == "NS" else ("Player_ID_E", "Player_ID_W")
        pred = pct_expr if "Pct_NS_Pred" in boards.columns else pl.lit(None, dtype=pl.Float64)
        pair_key = (
            _pair_key(pl.col(id_a), pl.col(id_b))
            if id_a in boards.columns and id_b in boards.columns
            else pl.lit(None).cast(pl.Utf8)
        )
        work = boards.with_columns(
            section.fill_null("").alias("Section"),
            pl.col(number_col).cast(pl.Utf8).alias("Table"),
            pred.alias("_pred"),
            left.alias("_left"),
            right.alias("_right"),
            pair_key.alias("Pair_Key"),
        )
        piece = (
            work.group_by(["Section", "Table", "Pair_Key"])
            .agg(
                pl.col("_pred").mean().alias("_pred"),
                pl.len().alias("Boards"),
                pl.col("_left").drop_nulls().first().alias("_left"),
                pl.col("_right").drop_nulls().first().alias("_right"),
            )
            .with_columns(
                pl.lit(orientation).alias("Orientation"),
                _pair_label(pl.col("Section"), pl.col("Table"), pl.col("_left"), pl.col("_right")).alias("Pair"),
                (pl.col("_pred") * 100.0).alias("Global_Scratch_Pct_Pred"),
            )
            .select("Pair", "Orientation", "Section", "Table", "Pair_Key", "Global_Scratch_Pct_Pred", "Boards")
        )
        frames.append(piece)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _match_prediction(predicted: pl.DataFrame, official: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if predicted.is_empty():
        return None
    key = _canonical_key(official.get("Left_Id"), official.get("Right_Id"))
    if key and "Pair_Key" in predicted.columns:
        matched = predicted.filter(
            (pl.col("Orientation") == official.get("Orientation"))
            & (pl.col("Pair_Key") == key)
        )
        if not matched.is_empty():
            return matched.to_dicts()[0]
        # The national ranking contains the same table numbers at many clubs.
        # When player ids are available, a failed id match means this is not
        # the club represented by the postmortem; table number must not attach
        # that club's prediction to an unrelated pair elsewhere.
        return None
    same_table = predicted.filter(
        (pl.col("Orientation") == official.get("Orientation"))
        & (pl.col("Table") == str(official.get("Table")))
        & (pl.col("Section").fill_null("") == str(official.get("Section") or ""))
    )
    if same_table.is_empty():
        same_table = predicted.filter(
            (pl.col("Orientation") == official.get("Orientation"))
            & (pl.col("Table") == str(official.get("Table")))
        )
    # The same table number is used at every club. Attaching an arbitrary
    # match would give another club's prediction to this pair.
    if same_table.height != 1:
        return None
    return same_table.to_dicts()[0]


def board_prediction_rows(boards: pl.DataFrame) -> pl.DataFrame:
    frame = boards
    if "Declarer_Direction" not in frame.columns and "Declarer" in frame.columns:
        frame = frame.with_columns(pl.col("Declarer").cast(pl.Utf8).alias("Declarer_Direction"))
    for name, dtype in (
        ("Contract_Pred", pl.Utf8),
        ("Declarer_Direction_Pred", pl.Utf8),
        ("Pct_NS_Pred", pl.Float64),
        ("Pct_Diff", pl.Float64),
    ):
        if name not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(name))
    section = pl.col("section_name") if "section_name" in frame.columns else pl.lit("")
    pair_ns = _pair_label(
        section.cast(pl.Utf8),
        pl.col("Pair_Number_NS").cast(pl.Utf8) if "Pair_Number_NS" in frame.columns else pl.lit(""),
        pl.col("Player_Name_N") if "Player_Name_N" in frame.columns else pl.lit(""),
        pl.col("Player_Name_S") if "Player_Name_S" in frame.columns else pl.lit(""),
    )
    pair_ew = _pair_label(
        section.cast(pl.Utf8),
        pl.col("Pair_Number_EW").cast(pl.Utf8) if "Pair_Number_EW" in frame.columns else pl.lit(""),
        pl.col("Player_Name_E") if "Player_Name_E" in frame.columns else pl.lit(""),
        pl.col("Player_Name_W") if "Player_Name_W" in frame.columns else pl.lit(""),
    )
    selected = frame.with_columns(pair_ns.alias("Pair_NS"), pair_ew.alias("Pair_EW"))
    keep = [name for name in BOARD_COLUMNS if name in selected.columns]
    if "Board" in selected.columns:
        selected = selected.sort("Board")
    return selected.select(keep)


def _file_built_at(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.is_file():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")


def prediction_provenance(directory: Path) -> Dict[str, Any]:
    """UTC build times of every artifact the predictions depend on.

    The checkpoints and history tables are rebuilt by the nightly training
    run, so the same session can be predicted differently from one day to
    the next. ``model_version`` is the newest checkpoint time.
    """
    models = {target: _file_built_at(directory / f"{_model_name(target)}.pth") for target in TARGETS}
    stamps = [stamp for stamp in models.values() if stamp]
    return {
        "model_version": max(stamps) if stamps else None,
        "models_built_at": models,
        "player_history_built_at": _file_built_at(directory / PLAYER_HISTORY_NAME),
        "pair_history_built_at": _file_built_at(directory / PAIR_HISTORY_NAME),
        "elo_results_built_at": _file_built_at(_elo_results_path()),
    }


def apply_session_predictions(
    boards: pl.DataFrame,
    session_id: Any,
    cache_dir: Path,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Attach board predictions and broadcast pair ranks onto every board row."""
    predicted, warning = _predict_targets(boards, session_id)
    if "Declarer_Direction" not in predicted.columns and "Declarer" in predicted.columns:
        predicted = predicted.with_columns(pl.col("Declarer").cast(pl.Utf8).alias("Declarer_Direction"))
    ranking, ranking_warning = load_ranking(session_id, Path(cache_dir))
    ranks = ranking_rows(predicted, ranking)
    predicted = _broadcast_ranks(predicted, ranks)
    info = {
        "warning": " ".join(text for text in (warning, ranking_warning) if text) or None,
        "models_dir": str(models_dir()),
        "provenance": prediction_provenance(models_dir()),
        "predictions_available": warning is None and "Pct_NS_Pred" in predicted.columns and predicted["Pct_NS_Pred"].null_count() < predicted.height,
        "results_published": ranking is not None,
        "ranking_pairs": ranks.height,
        "ranks": ranks,
    }
    return predicted, info


def _broadcast_ranks(boards: pl.DataFrame, ranks: pl.DataFrame) -> pl.DataFrame:
    out = boards
    stale = [
        f"{name}_{orientation}"
        for orientation in ("NS", "EW")
        for name in RANKING_COLUMNS
        if name != "Orientation" and f"{name}_{orientation}" in out.columns
    ]
    if stale:
        out = out.drop(stale)
    section_col = "section_name" if "section_name" in out.columns else (
        "Section_Name" if "Section_Name" in out.columns else None
    )
    id_columns = {"NS": ("Player_ID_N", "Player_ID_S"), "EW": ("Player_ID_E", "Player_ID_W")}
    for orientation, number_col in (("NS", "Pair_Number_NS"), ("EW", "Pair_Number_EW")):
        if number_col not in out.columns or ranks.is_empty():
            continue
        value_columns = [
            pl.col(name).alias(f"{name}_{orientation}")
            for name in RANKING_COLUMNS
            if name != "Orientation"
        ]
        side = ranks.filter(pl.col("Orientation") == orientation)
        has_ids = "Left_Id" in side.columns and "Right_Id" in side.columns
        id_a, id_b = id_columns[orientation]
        if has_ids and id_a in out.columns and id_b in out.columns:
            side = side.select(
                _pair_key(pl.col("Left_Id"), pl.col("Right_Id")).alias("_pair_key"),
                *value_columns,
            )
            out = out.with_columns(_pair_key(pl.col(id_a), pl.col(id_b)).alias("_pair_key"))
            out = out.join(side, on="_pair_key", how="left").drop("_pair_key")
            continue
        side = side.select(
            pl.col("Section").alias("_section"),
            pl.col("Table").alias("_table"),
            *value_columns,
        )
        counts = side.group_by(["_section", "_table"]).len()
        unique = counts.filter(pl.col("len") == 1).select("_section", "_table")
        side = side.join(unique, on=["_section", "_table"], how="inner")
        out = out.with_columns(
            (pl.col(section_col).cast(pl.Utf8) if section_col else pl.lit("")).alias("_section"),
            pl.col(number_col).cast(pl.Utf8).alias("_table"),
        )
        out = out.join(side, on=["_section", "_table"], how="left").drop("_section", "_table")
    return out


def prediction_payload(
    boards: pl.DataFrame,
    session_id: Any,
    cache_dir: Path,
    *,
    view: str,
    limit: int,
) -> Dict[str, Any]:
    predicted, info = apply_session_predictions(boards, session_id, cache_dir)
    if view == "rankings":
        frame = info["ranks"].select(RANKING_COLUMNS)
    else:
        frame = board_prediction_rows(predicted)
    frame = frame.head(limit)
    return {
        "view": view,
        "columns": frame.columns,
        "rows": frame.to_dicts(),
        "row_count": frame.height,
        "predictions_available": info["predictions_available"],
        "warning": info["warning"],
        "models_dir": info["models_dir"],
        "provenance": info["provenance"],
    }
