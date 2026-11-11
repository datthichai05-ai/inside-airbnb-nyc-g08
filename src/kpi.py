"""Tính 7 KPI từ dữ liệu đã QA, không chạy lại QA hoặc gọi LLM.

    python src/kpi.py --config configs/nyc.yaml
    python src/kpi.py --kpis 1 3 5

KPI 6–7 đọc llm_clean.csv.gz từ QA POST. Ô kết quả trống là NaN.
Cảnh báo tỷ lệ khi n<30, coverage<80%; so sánh giá chính cần n≥10.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

try:
    from . import qa
except ImportError:
    import qa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOM_TYPES = ("Entire home/apt", "Private room")
PRICE_MIN_N, RATE_WARNING_N, LOW_COVERAGE_PCT = 10, 30, 80
OUTPUTS = (
    "kpi_01_snapshot_listing_price.csv",
    "kpi_02_wifi_kitchen_declaration.csv",
    "kpi_03_three_night_availability.csv",
    "kpi_04_price_trend.csv",
    "kpi_05_superhost_trend.csv",
    "kpi_06_negative_review_rate_by_room_type.csv",
    "kpi_07_aspect_complaint_rate.csv",
)


def _require(frame, columns):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing QA columns: {missing}")


def _flag(frame, column, *, nullable=False):
    """Khôi phục Boolean CSV: chuỗi 'False' phải là False."""
    _require(frame, [column])
    values = frame[column].astype("string")
    allowed = values.isin(["True", "False"])
    if nullable:
        allowed |= values.isna() | values.eq("").fillna(False)
    if not allowed.all():
        raise ValueError(f"Invalid QA Boolean: {column}")
    return values.map({"True": True, "False": False}).astype("boolean")


def _analysis_rows(frame, required=(), keys=("qa_snapshot", "id")):
    _require(frame, [*keys, *required, "analysis_row", "key_status", "duplicate_status"])
    if frame[["key_status", "duplicate_status"]].eq("conflict").any().any():
        raise ValueError("Conflicting QA keys block the related KPI")
    rows = frame.loc[_flag(frame, "analysis_row")].copy()
    if rows.duplicated(list(keys)).any():
        raise ValueError(f"Non-unique analysis key: {keys}")
    for column in set(keys) & {"id", "listing_id"}:
        if not rows[column].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError(f"{column} must remain a nonempty string")
    rows = rows.rename(columns={"qa_snapshot": "snapshot"})
    if "id" in keys:
        _require(rows, ["neighbourhood_group_cleansed"])
        rows["borough"] = rows.neighbourhood_group_cleansed
    return rows


def _pct(numerator, denominator):
    if isinstance(denominator, pd.Series):
        denominator = denominator.astype(float)
        if isinstance(numerator, pd.Series):
            numerator = numerator.astype(float)
        return 100 * numerator / denominator.where(denominator.gt(0))
    return 100 * numerator / denominator if denominator > 0 else math.nan


def _ordered(frame, snapshots, groups):
    if not snapshots or len(set(snapshots)) != len(snapshots):
        raise ValueError("Snapshots must be nonempty and unique")
    if not set(frame.snapshot) <= set(snapshots):
        raise ValueError("Data contain snapshots outside the configuration")
    order = {snapshot: i for i, snapshot in enumerate(snapshots)}
    return (frame.assign(_order=frame.snapshot.map(order))
            .sort_values(["_order", *groups]).drop(columns="_order").reset_index(drop=True))


def _checked(result, *fractions):
    """Chỉ kiểm tra bất biến đầu ra; không lặp lại quy tắc chuẩn hóa của QA."""
    counts = result.filter(regex=r"^(n_|reviews_|total_reviews_)")
    bad_counts = counts.notna() & (counts.lt(0) | counts.mod(1).ne(0))
    if bad_counts.any().any():
        raise ValueError("Counts must be nonnegative integers")
    for column in result:
        if column.endswith("_pct") or "_pct_" in column:
            if not result[column].dropna().between(0, 100).all():
                raise ValueError(f"Rate outside [0,100]: {column}")
    for numerator, denominator in fractions:
        if result[numerator].gt(result[denominator]).any():
            raise ValueError(f"{numerator} exceeds {denominator}")
    return result


def _price_summary(frame, groups, eligibility):
    """Dùng chung KPI 1/4; mẫu số giữ listing thiếu giá, median giữ outlier hợp lệ."""
    _require(frame, ["price_num", "price_status", eligibility])
    prices = pd.to_numeric(frame.price_num, errors="coerce").astype(float)
    positive = prices.gt(0) & prices.lt(math.inf)
    valid = _flag(frame, eligibility) & frame.price_status.eq("valid") & positive
    missing = frame.price_status.eq("missing")
    invalid = ~missing & (~frame.price_status.eq("valid") | ~positive)
    iqr = (_flag(frame, "price_iqr_flag") & valid if "price_iqr_flag" in frame
           else pd.Series(pd.NA, index=frame.index, dtype="Int64"))
    values = frame.assign(_price=prices.where(valid), _valid=valid, _missing=missing, _invalid=invalid, _iqr=iqr)
    result = values.groupby(groups, dropna=False).agg(
        median_price_usd=("_price", "median"), n_listings_total=("id", "size"),
        n_price_valid=("_valid", "sum"), n_price_missing=("_missing", "sum"),
        n_price_invalid=("_invalid", "sum"),
        n_price_iqr_flag=("_iqr", lambda flags: flags.sum(min_count=1))).reset_index()
    result["n_price_excluded"] = result.n_listings_total - result.n_price_valid
    result["price_coverage_pct"] = _pct(result.n_price_valid, result.n_listings_total)
    result["low_n_flag"] = result.n_price_valid.lt(PRICE_MIN_N)
    result["low_coverage_flag"] = result.price_coverage_pct.lt(LOW_COVERAGE_PCT)
    result["iqr_warning_flag"] = result.n_price_iqr_flag.gt(0)
    warning = result.low_n_flag | result.low_coverage_flag | result.iqr_warning_flag.fillna(False)
    result["status"] = "ok"
    result.loc[warning, "status"] = "warning"
    result.loc[result.n_price_valid.eq(0), "status"] = "no_valid_price"
    return _checked(result, ("n_price_valid", "n_listings_total"))


def calculate_kpi_01(listings):
    frame = _analysis_rows(listings, ["room_type", "kpi1_group_eligible"])
    if frame.snapshot.nunique() > 1:
        raise ValueError("KPI1 requires the first configured snapshot only")
    frame = frame.loc[_flag(frame, "kpi1_group_eligible")]
    return _price_summary(frame, ["snapshot", "borough", "room_type"], "kpi1_eligible")


def calculate_kpi_02(listings):
    frame = _analysis_rows(listings, ["room_type", "amenities", "amenities_status", "kpi2_eligible",
                                      "has_wifi", "has_kitchen", "borough_valid", "room_type_status"])
    valid = _flag(frame, "kpi2_eligible") & frame.amenities_status.eq("valid")
    wifi, kitchen = _flag(frame, "has_wifi", nullable=True), _flag(frame, "has_kitchen", nullable=True)
    if (valid & (wifi.isna() | kitchen.isna())).any():
        raise ValueError("Valid amenities require verified Wi-Fi/Kitchen flags")
    # QA đã xác nhận amenities hợp lệ. Chỉ nhận diện [] để đếm, không parse lại JSON.
    empty = valid & frame.amenities.str.fullmatch(r"\s*\[\s*\]\s*", na=False)
    frame = frame.assign(
        _valid=valid, _both=valid & wifi & kitchen, _empty=empty,
        _missing=frame.amenities_status.eq("missing"),
        _invalid=~frame.amenities_status.isin(["valid", "missing"]),
        _group_warning=~_flag(frame, "borough_valid") | frame.room_type_status.ne("valid"))
    result = frame.groupby(["snapshot", "borough", "room_type"], dropna=False).agg(
        n_listings_total=("id", "size"), n_valid_amenities=("_valid", "sum"),
        n_missing_amenities=("_missing", "sum"), n_invalid_amenities=("_invalid", "sum"),
        n_empty_valid=("_empty", "sum"), n_both=("_both", "sum"),
        grouping_warning_flag=("_group_warning", "any")).reset_index()
    result["n_amenities_excluded"] = result.n_listings_total - result.n_valid_amenities
    result["wifi_kitchen_pct"] = _pct(result.n_both, result.n_valid_amenities)
    result["amenities_coverage_pct"] = _pct(result.n_valid_amenities, result.n_listings_total)
    result["map_version"], result["pocket_wifi_policy"] = qa.RULE_VERSION, "exclude_pocket_wifi"
    result["low_n_flag"] = result.n_valid_amenities.lt(RATE_WARNING_N)
    result["low_coverage_flag"] = result.amenities_coverage_pct.lt(LOW_COVERAGE_PCT)
    result["status"] = "ok"
    result.loc[result.low_n_flag | result.low_coverage_flag | result.grouping_warning_flag, "status"] = "warning"
    result.loc[result.n_valid_amenities.eq(0), "status"] = "no_valid_amenities"
    return _checked(result, ("n_both", "n_valid_amenities"), ("n_valid_amenities", "n_listings_total"))


def calculate_kpi_03(calendar, listings, snapshot):
    parents = _analysis_rows(listings, ["room_type", "kpi3_listing_eligible"])
    days = _analysis_rows(calendar, ["qa_snapshot", "available_state", "constraints_state",
                                     "kpi3_day_eligible", "listing_room_type"],
                          keys=("listing_id", "date_iso"))
    if not parents.snapshot.eq(snapshot).all() or not days.snapshot.eq(snapshot).all():
        raise ValueError("KPI3 calendar/listings snapshot mismatch")
    parents = parents.loc[_flag(parents, "kpi3_listing_eligible"), ["id", "borough", "room_type"]]
    start = date.fromisoformat(snapshot) + timedelta(days=1)
    dates = [(start + timedelta(days=i)).isoformat() for i in range(90)]
    days = days.loc[_flag(days, "kpi3_day_eligible")].copy()
    if not days.date_iso.isin(dates).all():
        raise ValueError("KPI3 eligible dates are outside the 90-night window")
    days = days.merge(parents.rename(columns={"id": "listing_id"}), on="listing_id",
                      how="left", validate="many_to_one", indicator=True)
    if not days._merge.eq("both").all() or not days.listing_room_type.eq(days.room_type).all():
        raise ValueError("KPI3 listing/room mapping mismatch")
    if not days[["available_state", "constraints_state"]].isin(["T", "F", "U"]).all().all():
        raise ValueError("KPI3 states must be T/F/U")
    # Dựng đủ 90 ngày để ngày thiếu vẫn là U, không bị bỏ khỏi 88 ngày nhận phòng.
    index = pd.MultiIndex.from_product([parents.id, dates], names=["listing_id", "date_iso"])
    daily = days.set_index(["listing_id", "date_iso"])[["available_state", "constraints_state"]].reindex(index)
    # shift(-1/-2) lấy hai đêm kế tiếp của cùng listing, không dịch constraints.
    available = daily.available_state.fillna("U")
    nights = available.groupby(level="listing_id", sort=False)
    conditions = pd.DataFrame({"night_1": available, "night_2": nights.shift(-1),
                               "night_3": nights.shift(-2), "arrival_constraint": daily.constraints_state.fillna("U")})
    runs = pd.Series("U", index=index)
    runs.loc[conditions.eq("F").any(axis=1)] = "F"
    runs.loc[conditions.eq("T").all(axis=1)] = "T"
    runs = runs.loc[index.get_level_values("date_iso").isin(dates[:-2])]
    # Một chuỗi T là đủ; chỉ F nếu cả 88 chuỗi F; còn lại giữ U.
    states = pd.DataFrame({"_true": runs.eq("T").groupby(level="listing_id").any(),
                           "_false": runs.eq("F").groupby(level="listing_id").all()})
    states["_unknown"] = ~(states._true | states._false)
    states["_missing"] = daily.available_state.isna().groupby(level="listing_id").sum()
    states["_invalid"] = daily.constraints_state.eq("U").groupby(level="listing_id").sum()
    base = parents.merge(states, left_on="id", right_index=True, validate="one_to_one")
    combined = pd.concat([base.assign(group_level="all", borough="", room_type=""),
                          base.assign(group_level="borough_room_type")], ignore_index=True)
    result = combined.groupby(["group_level", "borough", "room_type"], dropna=False).agg(
        n_listings_total=("id", "size"), n_true=("_true", "sum"), n_false=("_false", "sum"),
        n_unknown=("_unknown", "sum"), n_missing_listing_days=("_missing", "sum"),
        n_invalid_constraint_rows=("_invalid", "sum")).reset_index()
    if parents.empty:
        result.loc[0] = ["all", "", "", 0, 0, 0, 0, 0, 0]
    result.insert(0, "snapshot", snapshot)
    result.insert(1, "window_start", dates[0])
    result.insert(2, "window_end", dates[-1])
    result["n_evaluable"] = result.n_true + result.n_false
    if not (result.n_evaluable + result.n_unknown).eq(result.n_listings_total).all():
        raise ValueError("KPI3: T+F+U must equal N_total")
    result["three_night_share_pct"] = _pct(result.n_true, result.n_evaluable)
    result["coverage_pct"] = _pct(result.n_evaluable, result.n_listings_total)
    result["lower_bound_pct"] = _pct(result.n_true, result.n_listings_total)
    result["upper_bound_pct"] = _pct(result.n_true + result.n_unknown, result.n_listings_total)
    result["constraint_policy"], result["denominator_policy"] = "arrival_day", "evaluable"
    result["low_n_flag"] = result.n_evaluable.lt(RATE_WARNING_N)
    result["low_coverage_flag"] = result.coverage_pct.lt(LOW_COVERAGE_PCT)
    warning = (result.n_unknown.gt(0) | result.n_missing_listing_days.gt(0) |
               result.n_invalid_constraint_rows.gt(0) | result.low_n_flag | result.low_coverage_flag)
    result["status"] = "ok"
    result.loc[warning, "status"] = "warning"
    result.loc[result.n_evaluable.eq(0), "status"] = "no_evaluable_listings"
    return _checked(result, ("n_true", "n_evaluable"), ("n_evaluable", "n_listings_total"))


def calculate_kpi_04(listings, snapshots):
    frame = _analysis_rows(listings, ["neighbourhood_cleansed", "neighbourhood_valid", "kpi4_eligible"])
    frame = frame.loc[_flag(frame, "neighbourhood_valid")].copy()
    frame["neighbourhood"] = frame.neighbourhood_cleansed
    groups = ["borough", "neighbourhood"]
    result = _ordered(_price_summary(frame, ["snapshot", *groups], "kpi4_eligible"), snapshots, groups)
    # Giữ đủ mốc của mỗi khu phố: shift() phải gặp NaN tại mốc thiếu, không nhảy mốc.
    labels = list(frame.groupby(groups, dropna=False).groups)
    index = pd.MultiIndex.from_tuples([(s, *g) for g in labels for s in snapshots], names=["snapshot", *groups])
    result = result.set_index(["snapshot", *groups]).reindex(index).reset_index()
    counts = [c for c in result if c.startswith("n_") and (c != "n_price_iqr_flag" or "price_iqr_flag" in frame)]
    result[counts] = result[counts].fillna(0).astype(int)
    result["price_coverage_pct"] = _pct(result.n_price_valid, result.n_listings_total)
    result["low_n_flag"] = result.n_price_valid.lt(PRICE_MIN_N)
    result["low_coverage_flag"] = result.price_coverage_pct.lt(LOW_COVERAGE_PCT)
    result["iqr_warning_flag"] = result.n_price_iqr_flag.gt(0)
    grouped = result.groupby(groups, dropna=False, sort=False)
    previous = grouped.median_price_usd.shift()
    baseline = grouped.median_price_usd.transform(lambda prices: prices.iloc[0])
    result["previous_snapshot"] = result.snapshot.map(dict(zip(snapshots[1:], snapshots[:-1])))
    result["base_snapshot"] = snapshots[0]
    result["delta_prev_usd"] = result.median_price_usd - previous
    result["pct_change_prev"] = 100 * (result.median_price_usd / previous - 1)
    result["pct_change_base"] = 100 * (result.median_price_usd / baseline - 1)
    previous_n = grouped.n_price_valid.shift()
    baseline_n = grouped.n_price_valid.transform(lambda counts: counts.iloc[0])
    result["sample_ok_pair"] = result.n_price_valid.ge(PRICE_MIN_N) & previous_n.ge(PRICE_MIN_N)
    result["sample_ok_base"] = result.n_price_valid.ge(PRICE_MIN_N) & baseline_n.ge(PRICE_MIN_N)
    # Có trường quote là dấu hiệu khác ngữ cảnh giá; không dùng quote để thay giá QA.
    quote_columns = [c for c in frame if c.startswith("price_quote_")]
    has_quote = frame[quote_columns].fillna("").ne("").any(axis=1)
    contexts = has_quote.groupby(frame.snapshot).any().map({True: "quote_fields_present", False: "listing_price_only"})
    result["price_source_context"] = result.snapshot.map(contexts)
    boundary = result.previous_snapshot.notna() & result.price_source_context.ne(result.previous_snapshot.map(contexts))
    base_boundary = result.price_source_context.ne(contexts.get(snapshots[0]))
    result["price_context_status"] = "limited_cross_sectional"
    result.loc[boundary | base_boundary, "price_context_status"] = "limited_price_quote_boundary"
    result["main_chart_eligible"] = result.sample_ok_pair & ~boundary & previous.notna() & result.median_price_usd.notna()
    result["source"], result["price_source_column"] = "listings_clean.csv.gz", "price_num (QA of listings.price)"
    result["status"] = "limited_price_context"
    result.loc[~result.sample_ok_pair, "status"] = "limited_sample_or_no_previous"
    result.loc[result.n_price_valid.eq(0), "status"] = "no_valid_price"
    return _checked(_ordered(result, snapshots, groups), ("n_price_valid", "n_listings_total"))


def calculate_kpi_05(listings, snapshots):
    frame = _analysis_rows(listings, ["superhost_state", "superhost_status", "kpi5_eligible"])
    if not frame.superhost_state.isin(["T", "F", "U"]).all():
        raise ValueError("Superhost states must be T/F/U")
    valid = _flag(frame, "kpi5_eligible")
    frame = frame.assign(
        _valid=valid, _true=valid & frame.superhost_state.eq("T"), _false=valid & frame.superhost_state.eq("F"),
        _missing=frame.superhost_status.eq("missing"), _invalid=~valid & frame.superhost_status.ne("missing"))
    groups = ["aggregation_level", "borough"]
    combined = pd.concat([frame.assign(aggregation_level="all", borough=""),
                          frame.assign(aggregation_level="borough")], ignore_index=True)
    result = combined.groupby(["snapshot", *groups], dropna=False).agg(
        n_listings_total=("id", "size"), n_status_valid=("_valid", "sum"),
        n_superhost=("_true", "sum"), n_non_superhost=("_false", "sum"),
        n_status_missing=("_missing", "sum"), n_status_invalid=("_invalid", "sum")).reset_index()
    result = _ordered(result, snapshots, groups)
    labels = [("all", ""), *[("borough", b) for b in sorted(frame.borough.unique())]]
    index = pd.MultiIndex.from_tuples([(s, *g) for g in labels for s in snapshots], names=["snapshot", *groups])
    result = result.set_index(["snapshot", *groups]).reindex(index).reset_index()
    counts = [c for c in result if c.startswith("n_")]
    result[counts] = result[counts].fillna(0).astype(int)
    if not (result.n_superhost + result.n_non_superhost).eq(result.n_status_valid).all():
        raise ValueError("KPI5: T+F must equal N_valid")
    if not (result.n_status_valid + result.n_status_missing + result.n_status_invalid).eq(result.n_listings_total).all():
        raise ValueError("KPI5 status counts must sum to N_total")
    result["superhost_pct"] = _pct(result.n_superhost, result.n_status_valid)
    result["status_coverage_pct"] = _pct(result.n_status_valid, result.n_listings_total)
    rates = result.groupby(groups, dropna=False, sort=False).superhost_pct
    result["previous_snapshot"] = result.snapshot.map(dict(zip(snapshots[1:], snapshots[:-1])))
    result["change_prev_pp"] = result.superhost_pct - rates.shift()
    result["change_base_pp"] = result.superhost_pct - rates.transform(lambda values: values.iloc[0])
    result["one_listing_step_pp"] = _pct(1, result.n_status_valid)
    result["low_n_flag"] = result.n_status_valid.lt(RATE_WARNING_N)
    result["status"] = "ok"
    result.loc[result.low_n_flag | result.status_coverage_pct.lt(LOW_COVERAGE_PCT), "status"] = "warning"
    result.loc[result.n_status_valid.eq(0), "status"] = "no_valid_status"
    return _checked(_ordered(result, snapshots, groups), ("n_superhost", "n_status_valid"),
                    ("n_status_valid", "n_listings_total"))


def _review_base(predictions):
    # QA POST đã kiểm manifest, mapping, version và hash; ở đây chỉ kiểm giao diện.
    _require(predictions, ["review_id", "qa_snapshot", "listing_room_type", "sentiment_label",
                           "kpi6_eligible", "kpi7_base_eligible"])
    if predictions.review_id.duplicated().any():
        raise ValueError("Review IDs must be unique in the QA view")
    if not predictions.review_id.map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("Review IDs must remain nonempty strings")
    if predictions.qa_snapshot.nunique() > 1:
        raise ValueError("LLM view must use one snapshot")
    eligible = _flag(predictions, "kpi6_eligible")
    if not eligible.eq(_flag(predictions, "kpi7_base_eligible")).all():
        raise ValueError("KPI6/7 must share the same review base")
    base = predictions.loc[eligible & predictions.listing_room_type.isin(ROOM_TYPES)]
    if not base.sentiment_label.isin(["positive", "negative", "mixed"]).all():
        raise ValueError("Eligible reviews require valid sentiment labels")
    return base


def calculate_kpi_06(predictions):
    base = _review_base(predictions)
    rows = []
    for room in ROOM_TYPES:
        group = base.loc[base.listing_room_type.eq(room)]
        # Mixed vẫn thuộc mẫu số, chỉ negative thuộc tử số.
        total, negative = len(group), int(group.sentiment_label.eq("negative").sum())
        rows.append((room, total, negative, _pct(negative, total)))
    result = pd.DataFrame(rows, columns=["room_type", "total_reviews_in_group",
                                        "reviews_negative_sentiment", "negative_sentiment_pct_of_group"])
    return _checked(result, ("reviews_negative_sentiment", "total_reviews_in_group"))


def calculate_kpi_07(predictions):
    base = _review_base(predictions)
    rows = []
    for room in ROOM_TYPES:
        group = base.loc[base.listing_room_type.eq(room)]
        for aspect in qa.ASPECTS:
            flags = [f"{aspect}_positive", f"{aspect}_negative"]
            checks = [f"{aspect}_pair_valid", f"{aspect}_eligible", f"{aspect}_room_complete"]
            _require(predictions, [*flags, *checks])
            verified = all(_flag(group, column).all() for column in checks)
            labels = group[flags].astype("string")
            if not verified or not labels.isin(["", "1"]).all().all():
                # Dừng trước khi ghi CSV chính thức; không đổi lỗi nhãn thành 'không nhắc'.
                raise ValueError(f"BLOCKED KPI07: unverified {room} / {aspect}; no CSV generated")
            positive, negative = labels[flags[0]].eq("1"), labels[flags[1]].eq("1")
            # OR đếm một review vừa khen vừa chê đúng một lần trong mẫu số.
            evaluating, complaints = int((positive | negative).sum()), int(negative.sum())
            rows.append((room, aspect, len(group), evaluating, complaints, _pct(complaints, evaluating)))
    result = pd.DataFrame(rows, columns=["room_type", "aspect", "total_reviews_in_group",
                                        "reviews_evaluating_aspect", "reviews_negative_on_aspect",
                                        "negative_pct_among_aspect_reviews"])
    return _checked(result, ("reviews_negative_on_aspect", "reviews_evaluating_aspect"),
                    ("reviews_evaluating_aspect", "total_reviews_in_group"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "nyc.yaml")
    parser.add_argument("--kpis", nargs="+", type=int, choices=range(1, 8), default=list(range(1, 8)))
    parser.add_argument("--llm-view", type=Path, help="llm_clean view from QA POST")
    args = parser.parse_args(argv)
    config = qa.load_config(args.config)
    snapshots, selected = config.snapshots, set(args.kpis)
    if 3 in selected and len(snapshots) < 3:
        raise ValueError("KPI3 needs at least three snapshots")
    folder = PROJECT_ROOT / "data" / "processed"
    report = qa.read_csv(PROJECT_ROOT / "reports" / "qa_report.csv")
    _require(report, ["snapshot", "status", "kpi", "dataset", "rule_id"])
    blocked = report.loc[report.snapshot.isin(["all", *snapshots]) & report.status.eq("blocked")]
    needed = set(snapshots) if selected & {2, 4, 5} else set()
    if 1 in selected:
        needed.add(snapshots[0])
    if 3 in selected:
        needed.add(snapshots[2])
    listings = {}
    for snapshot in snapshots:
        if snapshot in needed:
            frame = qa.read_csv(folder / snapshot / "listings_clean.csv.gz")
            _require(frame, ["qa_snapshot"])
            if not frame.qa_snapshot.eq(snapshot).all():
                raise ValueError(f"Listings snapshot mismatch: {snapshot}")
            listings[snapshot] = frame
    combined = pd.concat(list(listings.values()), ignore_index=True) if listings else pd.DataFrame()
    results = {}
    # Chỉ chặn KPI nguồn đang được tính; thiếu LLM không cản KPI 1–5.
    for row in blocked.itertuples():
        if set(row.kpi.split(",")) & {str(k) for k in selected & {1, 2, 3, 4, 5}}:
            raise ValueError(f"QA BLOCKED: {row.dataset} / {row.rule_id}")
    if 1 in selected:
        results[1] = calculate_kpi_01(listings[snapshots[0]])
    if 2 in selected:
        results[2] = _ordered(calculate_kpi_02(combined), snapshots, ["borough", "room_type"])
    if 3 in selected:
        calendar = qa.read_csv(folder / snapshots[2] / "calendar_clean.csv.gz")
        results[3] = calculate_kpi_03(calendar, listings[snapshots[2]], snapshots[2])
    if 4 in selected:
        results[4] = calculate_kpi_04(combined, snapshots)
    if 5 in selected:
        results[5] = calculate_kpi_05(combined, snapshots)
    for number, result in results.items():
        qa.write_csv(result, folder / OUTPUTS[number - 1])
        print(f"KPI{number:02d}: OK; rows={len(result)}; {OUTPUTS[number - 1]}")
    if not selected & {6, 7}:
        return 0
    llm_path = args.llm_view or folder / snapshots[-1] / "llm_clean.csv.gz"
    if not llm_path.is_file():
        if args.llm_view:
            raise FileNotFoundError(llm_path)
        print("WAITING_LLM: KPI06/07 need last-snapshot llm_clean.csv.gz from QA POST")
        return 0
    # Lỗi aspect chỉ chặn KPI7; xung đột identity/schema chặn cả KPI6/7.
    for row in blocked.itertuples():
        if row.dataset != "predictions" or row.rule_id in {"DUPLICATE_CONFLICT", "SCHEMA"}:
            if set(row.kpi.split(",")) & {str(k) for k in selected & {6, 7}}:
                raise ValueError(f"QA BLOCKED: {row.dataset} / {row.rule_id}")
    view = qa.read_csv(llm_path)
    _require(view, ["qa_snapshot", "review_id"])
    if len(view) != 1000 or not view.qa_snapshot.eq(snapshots[-1]).all():
        raise ValueError("QA LLM view must have 1,000 reviews at the last configured snapshot")
    for number, calculate in [(6, calculate_kpi_06), (7, calculate_kpi_07)]:
        if number in selected:
            result = calculate(view)
            qa.write_csv(result, folder / OUTPUTS[number - 1])
            print(f"KPI{number:02d}: OK; rows={len(result)}; {OUTPUTS[number - 1]}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, pd.errors.ParserError) as exc:
        print(f"KPI failed: {exc}", file=sys.stderr)
        sys.exit(1)

