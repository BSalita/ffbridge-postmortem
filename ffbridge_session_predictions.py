"""Session predictions and pair rankings for the FFBridge postmortem.

Club rank is the place among pairs at the same club. Our global rank is where
a pair's predicted session percentage would sit in the published national
field. FFBridge's global rank is the official ``rank`` on the session ranking.
``Rank_Diff`` is our global rank minus that official rank.

Board percentages use the predicted declarer. When the predicted declarer is
North or South, ``Pct_NS_Pred`` is the predicted declarer percentage. When the
predicted declarer is East or West, North-South receive the complement.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

DEFAULT_MODELS_DIR = Path(
    os.environ.get("FFBRIDGE_SAVED_MODELS", r"E:\bridge\data\ffbridge\SavedModels")
)
TARGETS = ("Declarer_Direction", "Contract", "Declarer_Pct")
RANKING_COLUMNS = (
    "Pair",
    "Orientation",
    "Club_Rank",
    "Club_Rank_Pred",
    "Our_Global_Rank",
    "FFBridge_Global_Rank",
    "Rank_Diff",
    "Global_Pct",
    "Global_Pct_Pred",
    "Pct_Diff",
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
_PREDICTION_SQL_MARKERS = (
    "pct_ns_pred",
    "contract_pred",
    "declarer_direction_pred",
    "our_global_rank",
    "ffbridge_global_rank",
    "club_rank_pred",
    "rank_diff",
)


def models_dir() -> Path:
    return DEFAULT_MODELS_DIR


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


def _predict_targets(frame: pl.DataFrame) -> Tuple[pl.DataFrame, Optional[str]]:
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
    try:
        from mlBridge.mlBridgeAiLib import predict_model
    except Exception as exc:
        return _null_prediction_columns(out), f"Prediction library unavailable: {exc}"

    warnings: List[str] = []
    for target in TARGETS:
        schema_path = directory / f"{_model_name(target)}_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        feature_dtypes = schema.get("feature_dtypes") or {}
        required = list(schema.get("numerical_feature_cols") or []) + list(
            schema.get("categorical_feature_cols") or []
        )
        aligned = out
        for column in required:
            expected = _dtype_from_schema(str(feature_dtypes.get(column, "Float32")))
            if column not in aligned.columns:
                aligned = aligned.with_columns(pl.lit(None).cast(expected).alias(column))
            elif aligned.schema[column] != expected and expected == pl.Utf8:
                aligned = aligned.with_columns(pl.col(column).cast(pl.Utf8, strict=False))
        try:
            predicted = predict_model(
                directory,
                _model_name(target),
                aligned,
                device="cpu",
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
        if pred_column in out.columns:
            out = out.drop(pred_column)
        out = out.hstack(predicted.select(pred_column))
    out = _assign_ns_prediction(out)
    warning = "; ".join(warnings) if warnings else None
    return out, warning


def _null_prediction_columns(frame: pl.DataFrame) -> pl.DataFrame:
    columns = {
        "Declarer_Direction_Pred": pl.Utf8,
        "Contract_Pred": pl.Utf8,
        "Declarer_Pct_Pred": pl.Float64,
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
    if "Declarer_Pct_Pred" not in frame.columns:
        return _null_prediction_columns(frame)
    direction = pl.col("Declarer_Direction_Pred").cast(pl.Utf8)
    declarer_pct = pl.col("Declarer_Pct_Pred").cast(pl.Float64, strict=False)
    ns_pred = (
        pl.when(direction.is_in(["N", "S"]))
        .then(declarer_pct)
        .when(direction.is_in(["E", "W"]))
        .then(1.0 - declarer_pct)
        .otherwise(0.5)
    )
    actual = pl.col("Pct_NS").cast(pl.Float64, strict=False) if "Pct_NS" in frame.columns else pl.lit(None)
    return frame.with_columns(
        ns_pred.alias("Pct_NS_Pred"),
        (ns_pred - actual).alias("Pct_Diff"),
    )


def load_ranking(session_id: Any, cache_dir: Path) -> Optional[pl.DataFrame]:
    path = Path(cache_dir) / "rankings" / f"{session_id}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("items") or payload.get("ranking") or [payload]
    if not payload:
        return None
    return pl.json_normalize(payload, separator="_", infer_schema_length=None)


def _text(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8)


def _first_present(frame: pl.DataFrame, names: Tuple[str, ...]) -> Optional[str]:
    for name in names:
        if name in frame.columns:
            return name
    return None


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
    """One row per ranked pair, club rank then our global rank then FFBridge."""
    official = _official_pairs(ranking)
    predicted = _predicted_pair_scores(boards)
    if official.is_empty() and predicted.is_empty():
        return pl.DataFrame({name: [] for name in RANKING_COLUMNS})

    national_scores = [
        float(value)
        for value in (official["Global_Pct"].drop_nulls().to_list() if official.height else [])
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

    if official.height:
        for raw in official.to_dicts():
            pred = _match_prediction(predicted, raw)
            raw["Global_Pct_Pred"] = None if pred is None else pred.get("Global_Pct_Pred")
            raw["Boards"] = None if pred is None else pred.get("Boards")
            if pred is not None and pred.get("Pair"):
                raw["Pair"] = pred["Pair"]
            add_row(raw)
    else:
        for raw in predicted.to_dicts():
            raw["FFBridge_Global_Rank"] = None
            raw["Club"] = raw.get("Section") or ""
            raw["Global_Pct"] = raw.get("Global_Pct_Pred")
            add_row(raw)

    for raw in rows:
        club_field = [
            float(item["Global_Pct"])
            for item in club_groups.get(str(raw.get("Club")), [])
            if item.get("Global_Pct") is not None
        ]
        raw["Club_Rank"] = _rank_against(
            None if raw.get("Global_Pct") is None else float(raw["Global_Pct"]),
            club_field,
        )
        pred_score = raw.get("Global_Pct_Pred")
        if pred_score is None:
            raw["Club_Rank_Pred"] = None
            raw["Our_Global_Rank"] = None
            raw["Rank_Diff"] = None
            raw["Pct_Diff"] = None
            continue
        pred_score = float(pred_score)
        club_pred_field = []
        for item in club_groups.get(str(raw.get("Club")), []):
            score = item.get("Global_Pct_Pred")
            if score is None:
                score = item.get("Global_Pct")
            if score is not None:
                club_pred_field.append(float(score))
        raw["Club_Rank_Pred"] = _rank_against(pred_score, club_pred_field)
        national_pred_field = []
        for item in rows:
            if item is raw:
                national_pred_field.append(pred_score)
            elif item.get("Global_Pct") is not None:
                national_pred_field.append(float(item["Global_Pct"]))
        raw["Our_Global_Rank"] = _rank_against(pred_score, national_pred_field or national_scores)
        official_rank = raw.get("FFBridge_Global_Rank")
        raw["Rank_Diff"] = (
            None if official_rank is None or raw["Our_Global_Rank"] is None
            else int(raw["Our_Global_Rank"]) - int(official_rank)
        )
        actual = raw.get("Global_Pct")
        raw["Pct_Diff"] = None if actual is None else round(pred_score - float(actual), 2)
        raw["Global_Pct_Pred"] = round(pred_score, 2)

    ranked = pl.DataFrame(rows) if rows else pl.DataFrame({name: [] for name in RANKING_COLUMNS})
    for name in (*RANKING_COLUMNS, "Section", "Table", "Club"):
        if name not in ranked.columns:
            ranked = ranked.with_columns(pl.lit(None).alias(name))
    ordered = [name for name in (*RANKING_COLUMNS, "Section", "Table", "Club") if name in ranked.columns]
    return ranked.select(ordered).sort(
        ["Club_Rank", "FFBridge_Global_Rank"],
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
    return ranking.select(
        _pair_label(section, table, left, right).alias("Pair"),
        pl.col(orientation_col).cast(pl.Utf8).alias("Orientation"),
        pl.col(score_col).cast(pl.Float64, strict=False).alias("Global_Pct"),
        (
            pl.col(rank_col).cast(pl.Int64, strict=False)
            if rank_col
            else pl.lit(None).cast(pl.Int64)
        ).alias("FFBridge_Global_Rank"),
        (pl.col(club_col).cast(pl.Utf8) if club_col else pl.lit("")).alias("Club"),
        section.alias("Section"),
        table.alias("Table"),
    )


def _predicted_pair_scores(boards: pl.DataFrame) -> pl.DataFrame:
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
        pred = pct_expr if "Pct_NS_Pred" in boards.columns else pl.lit(None, dtype=pl.Float64)
        work = boards.with_columns(
            section.fill_null("").alias("Section"),
            pl.col(number_col).cast(pl.Utf8).alias("Table"),
            pred.alias("_pred"),
            left.alias("_left"),
            right.alias("_right"),
        )
        piece = (
            work.group_by(["Section", "Table"])
            .agg(
                pl.col("_pred").mean().alias("_pred"),
                pl.len().alias("Boards"),
                pl.col("_left").drop_nulls().first().alias("_left"),
                pl.col("_right").drop_nulls().first().alias("_right"),
            )
            .with_columns(
                pl.lit(orientation).alias("Orientation"),
                _pair_label(pl.col("Section"), pl.col("Table"), pl.col("_left"), pl.col("_right")).alias("Pair"),
                (pl.col("_pred") * 100.0).alias("Global_Pct_Pred"),
            )
            .select("Pair", "Orientation", "Section", "Table", "Global_Pct_Pred", "Boards")
        )
        frames.append(piece)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _match_prediction(predicted: pl.DataFrame, official: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if predicted.is_empty():
        return None
    matched = predicted.filter(
        (pl.col("Orientation") == official.get("Orientation"))
        & (pl.col("Table") == str(official.get("Table")))
        & (pl.col("Section").fill_null("") == str(official.get("Section") or ""))
    )
    if matched.is_empty():
        matched = predicted.filter(
            (pl.col("Orientation") == official.get("Orientation"))
            & (pl.col("Table") == str(official.get("Table")))
        )
    if matched.is_empty():
        return None
    return matched.to_dicts()[0]


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


def apply_session_predictions(
    boards: pl.DataFrame,
    session_id: Any,
    cache_dir: Path,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Attach board predictions and broadcast pair ranks onto every board row."""
    predicted, warning = _predict_targets(boards)
    if "Declarer_Direction" not in predicted.columns and "Declarer" in predicted.columns:
        predicted = predicted.with_columns(pl.col("Declarer").cast(pl.Utf8).alias("Declarer_Direction"))
    ranking = load_ranking(session_id, Path(cache_dir))
    ranks = ranking_rows(predicted, ranking)
    predicted = _broadcast_ranks(predicted, ranks)
    info = {
        "warning": warning,
        "models_dir": str(models_dir()),
        "predictions_available": warning is None and "Pct_NS_Pred" in predicted.columns and predicted["Pct_NS_Pred"].null_count() < predicted.height,
        "ranking_pairs": ranks.height,
    }
    return predicted, info


def _broadcast_ranks(boards: pl.DataFrame, ranks: pl.DataFrame) -> pl.DataFrame:
    out = boards
    section_col = "section_name" if "section_name" in out.columns else (
        "Section_Name" if "Section_Name" in out.columns else None
    )
    for orientation, number_col in (("NS", "Pair_Number_NS"), ("EW", "Pair_Number_EW")):
        if number_col not in out.columns or ranks.is_empty():
            continue
        side = ranks.filter(pl.col("Orientation") == orientation).select(
            pl.col("Section").alias("_section"),
            pl.col("Table").alias("_table"),
            *[pl.col(name).alias(f"{name}_{orientation}") for name in RANKING_COLUMNS if name != "Orientation"],
        )
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
        frame = ranking_rows(predicted, load_ranking(session_id, Path(cache_dir))).select(RANKING_COLUMNS)
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
    }
