"""QA chung; giu dong/cot nguon, chi them cot phan tich va dieu kien KPI.

    python src/qa.py --config configs/nyc.yaml --stage pre
    python src/qa.py --stage post --predictions path/to/predictions.csv --version v1

Hop dong CSV prediction (khong goi LLM): snapshot, review_id, listing_id,
version, model, prompt_version, source_text_sha256 (SHA256 comments UTF-8 goc),
request_status=success, parse_status=valid, analysis_status=complete, sentiment,
va 10 cot <aspect>_positive/negative. Chi chap nhan '1' hoac o rong cho aspect;
o rong chi hop le khi request/parse/analysis thanh cong va du schema.
--version chon ro mot version, khong tu lay prediction moi nhat.

Script KPI phai dung analysis_row va co *_eligible; duplicate conflict duoc
bao blocked, khong duoc cong bo KPI lien quan truoc khi giai quyet conflict.
Pham vi KPI1=YAML snapshots[0], KPI3=snapshots[2], KPI6/7=snapshots[-1];
KPI2/4/5 dung tung snapshot, khong lay giao ID cua panel.
KPI3 chi dung constraints_state tai ngay check-in, available_state ca 3 dem.
qa_calendar khong xet chuoi 3 dem hay tinh trang thai listing T/F/U.
kpi3_day_eligible la ngay co khoa/mapping hop le, van giu cac trang thai U.
KPI6 dung kpi6_eligible (gom mixed); KPI7 dung cung base, chi tinh room/aspect
khi <aspect>_room_complete=True va mau so la OR hai cot bang 1. Mau so 0
phai tra NaN o buoc KPI, khong phai 0%; khong co cong thuc tong hop o day.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import sys
import unicodedata
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

try:
    from .download import PROJECT_ROOT, load_assignments, load_config
except ImportError:
    from download import PROJECT_ROOT, load_assignments, load_config

# HƯỚNG DẪN ĐỌC FILE:
# (1) Hàm nhỏ: kiểm tra kiểu dữ liệu, khóa, trùng, ghi báo cáo.
# (2) QA trước LLM: listings -> calendar -> reviews -> nguồn review được chọn.
# (3) QA sau LLM: xác minh prediction -> tạo cờ cho KPI 6, 7.
# LƯU Ý: qa.py chỉ chuẩn bị dữ liệu/cờ hợp lệ, KHÔNG tính kết quả KPI.
RULE_VERSION = "g08-2026-11-11.1"
ASPECTS = ("cleanliness", "host_friendliness", "noise", "amenities", "location")
MISSING = {"", "na", "n/a", "nan", "null", "none", "<na>"}
ROOM_TYPES = {"Entire home/apt", "Private room", "Shared room", "Hotel room"}
USD = re.compile(r"(?:[+-]?\$?|\$[+-]?)(?:[0-9]+|[1-9][0-9]{0,2}(?:,[0-9]{3})+)(?:\.[0-9]{1,2})?\Z")
WIFI = re.compile(r"(?:fast )?wi[ -]?fi(?: *- *[0-9]+(?:\.[0-9]+)? mbps)?\Z")
REPORT_COLUMNS = [
    "stage", "rule_id", "dataset", "snapshot", "kpi", "condition", "reason",
    "action", "affected", "n_before", "n_after", "n_eligible", "coverage_pct",
    "status", "notes", "source", "source_sha256", "rule_version", "kpi_before",
    "kpi_after", "delta", "kpi_na_reason",
]


# Chuẩn hóa ô nhập trước khi xét thiếu / hợp lệ; không sửa cột gốc.
def token(value) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def missing(value) -> bool:
    return token(value).casefold() in MISSING


# Dấu vân tay tệp đầu vào: giúp đối chiếu chính xác phiên bản dữ liệu.
def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# Đọc CSV chặt chẽ: kiểm tra độ dài hàng trước khi pandas tải dữ liệu.
# Đọc mọi cột dưới dạng chuỗi để giữ nguyên ID, không tự đổi ô trống thành NaN.
def read_csv(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if not header or len(set(header)) != len(header):
            raise ValueError(f"{path}: empty/duplicate CSV header")
        # pandas co the tu lay o du lam index hoac dien o thieu; khong de mat nguon.
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"{path}: wrong CSV width at line {reader.line_num}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    frame.attrs.update(source=str(path), source_sha256=sha256(path))
    return frame


# Chỉ THÊM cột QA; báo lỗi nếu vô tình ghi đè cột nguồn.
def add_columns(frame: pd.DataFrame, **columns) -> None:
    overlap = set(columns) & set(frame.columns)
    if overlap:
        raise ValueError(f"Source/derived column collision: {sorted(overlap)}")
    for name, values in columns.items():
        frame[name] = values


def field(frame: pd.DataFrame, name: str) -> pd.Series:
    # Cot nguon thieu khong duoc tao gia; series rong chi dung cho QA.
    return frame[name] if name in frame else pd.Series("", index=frame.index)


# Mỗi quy tắc QA được ghi thành một hàng trong reports/qa_report.csv.
# n_before == n_after vì ta giữ mọi dòng nguồn; *_eligible phản ánh số dùng được.
# kpi_before/after để trống: file này không tính chỉ số KPI.
def record(report: list, frame: pd.DataFrame, rule: str, dataset: str,
           snapshot: str, kpi: str, condition: str, reason: str, action: str,
           affected: int, *, eligible=None, total=None, status=None, notes="",
           stage="pre") -> None:
    before = len(frame) if total is None else int(total)
    report.append(dict(
        stage=stage, rule_id=rule, dataset=dataset, snapshot=snapshot, kpi=kpi,
        condition=condition, reason=reason, action=action, affected=int(affected),
        n_before=before, n_after=before,
        n_eligible=eligible, coverage_pct=(100 * eligible / before
                                        if eligible is not None and before else None),
        status=status or ("warning" if affected else "pass"), notes=notes,
        source=frame.attrs.get("source", ""),
        source_sha256=frame.attrs.get("source_sha256", ""),
        rule_version=RULE_VERSION, kpi_before=None, kpi_after=None, delta=None,
        kpi_na_reason="QA only; KPI aggregation not run; no comparable before/after KPI",
    ))


# Thiếu cột bắt buộc => báo blocked; không tạo dữ liệu giả để thay thế.
def schema(frame, required, report, dataset, snapshot, kpi, *, stage="pre"):
    absent = sorted(set(required) - set(frame.columns))
    record(report, frame, "SCHEMA", dataset, snapshot, kpi, "required columns present",
           "missing columns" if absent else "schema available", "FLAG", len(absent),
           status="blocked" if absent else "pass", notes=",".join(absent), stage=stage)
    return not absent


def id_status(value):
    if missing(value):
        return "missing"
    # Khong qua float, khong mat leading zeros hay ID vuot 2**53.
    return "valid" if re.fullmatch(r"[0-9]+", str(value)) and str(value).strip("0") else "domain_error"


# 3 việc: (1) kiểm tra khóa; (2) phát hiện bản ghi trùng hệt;
# (3) phát hiện cùng khóa nhưng nội dung khác nhau (conflict).
# analysis_row=True chỉ cho bản ghi có khóa hợp lệ, không phải bản sao dư,
# không nằm trong nhóm khóa xung đột. Dữ liệu gốc vẫn được giữ đủ.
def mark_keys(frame, keys, report, dataset, snapshot, kpi, *, stage="pre"):
    original = list(frame.columns)
    bad = pd.Series(False, index=frame.index)
    key_state = pd.Series("valid", index=frame.index)
    key_reason = pd.Series("", index=frame.index)
    for key in keys:
        values = field(frame, key)
        states = values.map(id_status) if key in {"id", "listing_id", "review_id"} else (
            values.map(lambda v: iso_date(v)[1]) if key == "date" else
            values.map(lambda v: "missing" if missing(v) else "valid"))
        invalid = states.ne("valid")
        key_reason = key_reason + states.where(invalid, "").map(lambda s: f"{key}:{s};" if s else "")
        key_state = key_state.mask(invalid, states)
        bad |= invalid
    # Nếu tệp thiếu bất kỳ cột khóa nào, không thể tin vào phép nối dữ liệu.
    if not set(keys) <= set(original):
        bad[:] = True
        exact = conflict = pd.Series(False, index=frame.index)
        repeated = exact
    else:
        # Bản sao hoàn toàn: giữ 1 bản đại diện để phân tích.
        # Cùng khóa mà khác bất kỳ cột nguồn nào: flag conflict cho mọi bản liên quan.
        exact = frame.duplicated(original, keep="first")
        key_values = frame[keys].copy()
        if "date" in keys:
            parsed = key_values.date.map(iso_date)
            bad |= parsed.map(lambda pair: pair[1] != "valid").astype(bool)
            key_values["date"] = [pair[0] if pair[1] == "valid" else raw
                                  for pair, raw in zip(parsed, key_values.date)]
        # Chi bo ban sao hoan toan de kiem tra noi dung xung dot cua cung khoa.
        distinct = key_values.loc[~exact]
        conflict_keys = distinct.loc[distinct.duplicated(keys, keep=False)]
        conflict = pd.Series(pd.MultiIndex.from_frame(key_values).isin(
            pd.MultiIndex.from_frame(conflict_keys)), index=frame.index)
        repeated = key_values.duplicated(keys, keep=False)
    add_columns(frame, key_status=key_state.mask(conflict, "conflict"),
                key_reason=key_reason.mask(conflict, key_reason + "conflicting_key;"),
                duplicate_status=pd.Series("unique", index=frame.index)
                .mask(repeated & ~conflict, "exact_duplicate").mask(conflict, "conflict"),
                analysis_row=~(bad | exact | conflict))
    record(report, frame, "KEY_INVALID", dataset, snapshot, kpi, "key present and valid",
           "missing/invalid key", "EXCLUDE_FROM_KPI", bad.sum(), stage=stage)
    record(report, frame, "DUPLICATE_EXACT", dataset, snapshot, kpi, "all source columns identical",
           "extra identical copies", "EXCLUDE_FROM_KPI", (exact & ~conflict).sum(),
           eligible=int(frame.analysis_row.sum()), notes="Only one representative in analysis; all source rows kept", stage=stage)
    record(report, frame, "DUPLICATE_CONFLICT", dataset, snapshot, kpi, "same key, different contents",
           "conflicting key blocks related computation/merge", "EXCLUDE_FROM_KPI", conflict.sum(),
           status="blocked" if conflict.any() else "pass", stage=stage)
    return frame


# Giá: trả về (giá_số, trạng_thái). Giá 0/âm không dùng cho KPI giá.
# Không chặn giá cao chỉ vì bất thường; outlier xử lý riêng bằng cờ IQR.
def price(value):
    value = token(value)
    if missing(value):
        return math.nan, "missing"
    if re.fullmatch(r"(?:[+-]?\$?|\$[+-]?)(?:inf(?:inity)?|nan)", value, re.I):
        return math.nan, "domain_error"
    if not USD.fullmatch(value):
        return math.nan, "parse_error"
    number = float(value.replace("$", "").replace(",", ""))
    return (number, "valid") if math.isfinite(number) and number > 0 else (math.nan, "domain_error")


# Số đêm phải là số nguyên dương; Decimal tránh lỗi làm tròn float.
def positive_int(value):
    value = token(value)
    if missing(value):
        return math.nan, "missing"
    try:
        number = Decimal(value)
    except InvalidOperation:
        return math.nan, "parse_error"
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        return math.nan, "domain_error"
    if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value):
        return math.nan, "parse_error"
    return int(number), "valid"


# Chỉ t/f trong nguồn mới có nghĩa xác định; U = không xác định.
# Không suy đoán dữ liệu thiếu thành False.
def boolean(value):
    value = token(value).casefold()
    if missing(value):
        return "U", "missing"
    return ({"t": "T", "f": "F"}[value], "valid") if value in {"t", "f"} else ("U", "domain_error")


def iso_date(value):
    value = token(value)
    if missing(value):
        return "", "missing"
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError
        return value, "valid"
    except ValueError:
        return "", "parse_error"


# Amenities là danh sách JSON, không tìm chuỗi con tùy tiện.
# Tên tiện nghi được chuẩn hóa để nhận ra Wi-Fi / Kitchen nhất quán.
def amenities(value):
    if missing(value):
        return "missing", pd.NA, pd.NA
    try:
        items = json.loads(value)
    except (ValueError, TypeError):
        return "parse_error", pd.NA, pd.NA
    if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
        return "domain_error", pd.NA, pd.NA
    labels = {" ".join(unicodedata.normalize("NFKC", item).casefold()
                      .translate(str.maketrans({c: "-" for c in "‐‑‒–—−"})).split())
              for item in items}
    return "valid", any(WIFI.fullmatch(item) for item in labels), "kitchen" in labels


# Một hàm chung dùng parser để tạo 2 cột: giá trị đã đổi + lý do hợp lệ/lỗi.
def normalize_pair(frame, source, first, second, parser):
    pairs = [parser(value) for value in field(frame, source)]
    add_columns(frame, **{first: pd.Series([p[0] for p in pairs], index=frame.index, dtype=None if pairs else object),
                          second: pd.Series([p[1] for p in pairs], index=frame.index, dtype=str)})


def report_states(frame, column, report, dataset, snapshot, kpi):
    for state in ("missing", "parse_error", "domain_error", "unknown"):
        record(report, frame, f"{column.upper()}_{state.upper()}", dataset, snapshot, kpi,
               f"{column}={state}", state, "EXCLUDE_FROM_KPI",
               frame[column].eq(state).sum(), eligible=int(frame[column].eq("valid").sum()))


# QA listings: chuẩn hóa giá, tiện nghi, loại phòng, khu vực và superhost.
# Mỗi KPI sử dụng cờ eligible riêng; không loại toàn bộ dòng vì lỗi ở cột khác.
def qa_listings(source, snapshot, report):
    frame = source.copy()
    schema(frame, ("id", "price", "amenities", "host_is_superhost", "room_type",
                   "neighbourhood_cleansed", "neighbourhood_group_cleansed"), report,
           "listings", snapshot, "1,2,3,4,5,6,7")
    mark_keys(frame, ["id"], report, "listings", snapshot, "1,2,3,4,5,6,7")
    add_columns(frame, qa_snapshot=snapshot)
    normalize_pair(frame, "price", "price_num", "price_status", price)
    normalize_pair(frame, "host_is_superhost", "superhost_state", "superhost_status", boolean)
    items = [amenities(value) for value in field(frame, "amenities")]
    add_columns(frame, amenities_status=[p[0] for p in items],
                has_wifi=pd.array([p[1] for p in items], dtype="boolean"),
                has_kitchen=pd.array([p[2] for p in items], dtype="boolean"),
                room_type_status=field(frame, "room_type").map(
                    lambda v: "missing" if missing(v) else "valid" if v in ROOM_TYPES else "domain_error"),
                neighbourhood_valid=~field(frame, "neighbourhood_cleansed").map(missing).astype(bool),
                borough_valid=~field(frame, "neighbourhood_group_cleansed").map(missing).astype(bool))
    usable = frame.analysis_row
    room = frame.room_type_status.eq("valid")
    prices = frame.loc[usable & frame.price_status.eq("valid"), "price_num"]
    # IQR = Q3-Q1; dùng 1.5*IQR để CẢNH BÁO giá quá khác biệt.
    # Chỉ tính ngưỡng từ giá hợp lệ; cờ IQR KHÔNG làm kpi*_eligible=False.
    q1, q3 = prices.quantile([.25, .75])
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    # Ví dụ: KPI1 cần giá + borough + room hợp lệ; KPI4 cần giá + neighbourhood.
    # KPI2 cần amenities; KPI5 cần trạng thái superhost xác định.
    add_columns(frame, price_iqr_flag=frame.price_status.eq("valid") &
                ((frame.price_num < lower) | (frame.price_num > upper)),
                amenities_valid=frame.amenities_status.eq("valid"),
                kpi1_eligible=usable & room & frame.borough_valid & frame.price_status.eq("valid"),
                kpi1_group_eligible=usable & room & frame.borough_valid,
                kpi2_eligible=usable & frame.amenities_status.eq("valid"),
                kpi3_listing_eligible=usable,
                kpi4_eligible=usable & frame.neighbourhood_valid & frame.price_status.eq("valid"),
                kpi5_eligible=usable & frame.superhost_status.eq("valid"))
    for column, kpi in (("price_status", "1,4"), ("amenities_status", "2"),
                        ("superhost_status", "5"), ("room_type_status", "1,6,7")):
        report_states(frame, column, report, "listings", snapshot, kpi)
    record(report, frame, "PRICE_IQR", "listings", snapshot, "1,4", "outside Q1/Q3 +/-1.5*IQR",
           "valid price outlier", "FLAG", frame.price_iqr_flag.sum(),
           eligible=int(frame.kpi4_eligible.sum()),
           notes=f"Q1={q1}; Q3={q3}; lower={lower}; upper={upper}; no price cap; outliers retained")
    for column in ("neighbourhood_valid", "borough_valid"):
        record(report, frame, column.upper(), "listings", snapshot, "1,4", column,
               "missing grouping label", "EXCLUDE_FROM_KPI", (~frame[column]).sum())
    for kpi in (1, 2, 4, 5):
        record(report, frame, f"KPI{kpi}_ELIGIBILITY", "listings", snapshot, str(kpi),
               f"kpi{kpi}_eligible", "KPI-specific view; no shared row deletion", "EXCLUDE_FROM_KPI",
               (usable & ~frame[f"kpi{kpi}_eligible"]).sum(),
               eligible=int(frame[f"kpi{kpi}_eligible"].sum()))
    return frame


# Nối Calendar/Reviews với Listings của CÙNG snapshot để lấy loại phòng.
# many_to_one = mỗi listing_id chỉ được khớp tối đa 1 dòng listings.
# Left merge giữ nguyên số dòng bên trái; ID mồ côi và conflict phải báo rõ.
def attach_room(frame, listings, report, dataset, snapshot, kpi):
    """Left merge dung snapshot; parent conflict khong bi che thanh orphan."""
    if not listings.qa_snapshot.eq(snapshot).all():
        raise ValueError("Listings snapshot mismatch")
    parent = listings.loc[listings.analysis_row, ["room_type_status"]].copy()
    parent["id"] = field(listings, "id").loc[parent.index]
    parent["listing_room_type"] = field(listings, "room_type").loc[parent.index]
    parent = parent.rename(columns={"id": "listing_id", "room_type_status": "listing_room_status"})
    overlap = (set(parent.columns) - {"listing_id"}) & set(frame.columns)
    if overlap or "_qa_merge" in frame:
        raise ValueError(f"Merge derived column collision: {overlap}")
    before = len(frame)
    attrs = frame.attrs.copy()
    merged = frame.merge(parent, how="left", on="listing_id", validate="many_to_one", indicator="_qa_merge", sort=False)
    if len(merged) != before:
        raise ValueError("Listings merge multiplied rows")
    merged.attrs = attrs
    # Nếu parent có ID xung đột, phân biệt parent_conflict với orphan (không tìm thấy).
    conflict_ids = set(field(listings.loc[listings.key_status.eq("conflict")], "id"))
    invalid_id = field(merged, "listing_id").map(id_status).ne("valid")
    status = pd.Series("matched", index=merged.index).mask(merged._qa_merge.eq("left_only"), "orphan")
    status = status.mask(field(merged, "listing_id").isin(conflict_ids), "parent_conflict").mask(invalid_id, "invalid_id")
    add_columns(merged, listing_merge_status=status)
    merged = merged.drop(columns="_qa_merge")
    for state in ("orphan", "parent_conflict", "invalid_id"):
        record(report, merged, f"LISTING_MERGE_{state.upper()}", dataset, snapshot, kpi,
               f"listing_merge_status={state}", "same-snapshot parent unavailable",
               "EXCLUDE_FROM_KPI", status.eq(state).sum(), eligible=int(status.eq("matched").sum()),
               status="blocked" if state == "parent_conflict" and status.eq(state).any() else None,
               notes=f"left many_to_one; rows before={before}, after={len(merged)}")
    return merged


# QA calendar chỉ kiểm tra dữ liệu THEO TỪNG NGÀY, chưa tính KPI3.
# KPI3 sẽ tự tìm 3 ngày liên tiếp và chỉ xét minimum/maximum ở ngày check-in.
def qa_calendar(source, listings, snapshot, report):
    frame = source.copy()
    required = ("listing_id", "date", "available", "minimum_nights", "maximum_nights")
    schema(frame, required, report, "calendar", snapshot, "3")
    mark_keys(frame, ["listing_id", "date"], report, "calendar", snapshot, "3")
    # Cot listing_id thieu: van giu schema nguon, merge khong the thuc hien.
    if "listing_id" in frame:
        frame = attach_room(frame, listings, report, "calendar", snapshot, "3")
    else:
        add_columns(frame, listing_merge_status="invalid_id")
    add_columns(frame, qa_snapshot=snapshot)
    normalize_pair(frame, "date", "date_iso", "date_status", iso_date)
    normalize_pair(frame, "available", "available_state", "available_status", boolean)
    normalize_pair(frame, "minimum_nights", "minimum_nights_num", "minimum_nights_status", positive_int)
    normalize_pair(frame, "maximum_nights", "maximum_nights_num", "maximum_nights_status", positive_int)
    # Ràng buộc 3 đêm: T = min<=3<=max; F = ràng buộc hợp lệ nhưng không cho 3 đêm;
    # U = thiếu/sai min hoặc max (kể cả min>max). Không tráo hai giới hạn.
    valid = frame.minimum_nights_status.eq("valid") & frame.maximum_nights_status.eq("valid")
    reversed_bounds = valid & (frame.minimum_nights_num > frame.maximum_nights_num)
    state = pd.Series("unknown", index=frame.index).mask(valid, "valid").mask(reversed_bounds, "domain_error")
    constraints = pd.Series("U", index=frame.index).mask(state.eq("valid"), "F")
    constraints = constraints.mask(state.eq("valid") & (frame.minimum_nights_num <= 3)
                                   & (frame.maximum_nights_num >= 3), "T")
    # Cửa sổ 90 đêm: từ ngày sau snapshot đến snapshot + 90 ngày (bao gồm).
    start, end = date.fromisoformat(snapshot) + timedelta(days=1), date.fromisoformat(snapshot) + timedelta(days=90)
    in_window = frame.date_status.eq("valid") & frame.date_iso.between(start.isoformat(), end.isoformat())
    # eligible chỉ đòi hỏi ngày/khóa/nối hợp lệ: vẫn giữ ngày có available hoặc constraints = U.
    # Tránh làm sai mẫu số KPI3 do âm thầm bỏ trạng thái không xác định.
    add_columns(frame, constraints_status=state, constraints_state=constraints,
                min_gt_max_flag=reversed_bounds, in_90_night_window=in_window,
                kpi3_day_eligible=frame.analysis_row & frame.date_status.eq("valid") &
                frame.listing_merge_status.eq("matched") & in_window)
    # CHÚ Ý KHI RÀ SOÁT: hàm report_states đang ghi EXCLUDE_FROM_KPI cho cả
    # constraints_status=unknown, nhưng kpi3_day_eligible vẫn giữ dòng U.
    # Đây là cách ghi action chưa nhất quán, nên chỉnh ở lượt sửa logic.
    for column in ("date_status", "available_status", "minimum_nights_status", "maximum_nights_status", "constraints_status"):
        report_states(frame, column, report, "calendar", snapshot, "3")
    # Tương tự: min>max được gắn U, không phải xóa dòng calendar.
    # Nhãn action EXCLUDE_FROM_KPI bên dưới cần đổi thành FLAG/UNKNOWN cho nhất quán.
    record(report, frame, "MIN_GT_MAX", "calendar", snapshot, "3", "minimum_nights > maximum_nights",
           "invalid bounds; constraints U; never swap", "EXCLUDE_FROM_KPI", reversed_bounds.sum())
    record(report, frame, "CONSTRAINTS_KNOWN_FALSE", "calendar", snapshot, "3", "valid min>3 or max<3",
           "valid constraint does not allow 3 nights", "FLAG", constraints.eq("F").sum(), status="pass")
    # Ghi ngay thieu o Listings; khong tao hay dien them dong Calendar.
    # Đối chiếu đủ 90 ngày của mỗi listing; chỉ ghi số ngày thiếu, KHÔNG sinh dòng bù.
    dates = { (start + timedelta(days=i)).isoformat() for i in range(90) }
    present = frame.loc[in_window & field(frame, "listing_id").map(id_status).eq("valid")]
    grouped = present.groupby("listing_id")["date_iso"].agg(set).to_dict() if "listing_id" in frame else {}
    missing_dates = [sorted(dates - grouped.get(identifier, set())) for identifier in field(listings, "id")]
    add_columns(listings, calendar_missing_nights_90=[len(items) for items in missing_dates],
                calendar_missing_dates_90=[json.dumps(items) for items in missing_dates])
    eligible_listings = listings.analysis_row
    absent = int(listings.loc[eligible_listings, "calendar_missing_nights_90"].sum())
    record(report, frame, "CALENDAR_MISSING_NIGHTS_90", "calendar", snapshot, "3", "snapshot+1 through snapshot+90",
           "absent listing-date observations; remain unknown", "FLAG", absent,
           total=int(eligible_listings.sum()) * 90, eligible=int(eligible_listings.sum()) * 90 - absent,
           notes="Counts observations, not removed rows; invalid-date/conflicting rows are separately flagged")
    return frame


# QA reviews: khóa review, ngày, nội dung, cặp review_id/listing_id được giao.
# Chỉ review được chọn trong manifest và có nguồn đáng tin cậy mới đủ điều kiện LLM.
def qa_reviews(source, listings, manifest, snapshot, report):
    frame = source.copy()
    schema(frame, ("id", "listing_id", "comments", "date"), report, "reviews", snapshot, "6,7")
    mark_keys(frame, ["id"], report, "reviews", snapshot, "6,7")
    if "listing_id" in frame:
        frame = attach_room(frame, listings, report, "reviews", snapshot, "6,7")
    else:
        add_columns(frame, listing_merge_status="invalid_id", listing_room_status="missing", listing_room_type="")
    add_columns(frame, qa_snapshot=snapshot)
    normalize_pair(frame, "date", "review_date_iso", "review_date_status", iso_date)
    frame.loc[frame.review_date_status.eq("valid") & frame.review_date_iso.gt(snapshot), "review_date_status"] = "domain_error"
    mapping = manifest.set_index("review_id")
    ids = field(frame, "id")
    expected = ids.map(mapping.listing_id)
    selected = ids.isin(mapping.index[mapping.llm_selected])
    add_columns(frame, review_text_status=field(frame, "comments").map(lambda v: "missing" if missing(v) else "valid"),
                manifest_mapping_status=pd.Series("matched", index=frame.index)
                .mask(expected.isna(), "unassigned").mask(expected.notna() & expected.ne(field(frame, "listing_id")), "mismatch"),
                llm_selected=selected, manual_selected=ids.isin(mapping.index[mapping.manual_selected]))
    add_columns(frame, llm_source_eligible=frame.analysis_row & selected &
                frame.listing_merge_status.eq("matched") & frame.listing_room_status.eq("valid") &
                frame.manifest_mapping_status.eq("matched") & frame.review_date_status.eq("valid") &
                frame.review_text_status.eq("valid"))
    for column in ("review_date_status", "review_text_status"):
        report_states(frame, column, report, "reviews", snapshot, "6,7")
    record(report, frame, "REVIEW_MANIFEST_MAPPING", "reviews", snapshot, "6,7", "review/listing pair matches assignment",
           "unassigned or mismatched review", "EXCLUDE_FROM_KPI", frame.manifest_mapping_status.ne("matched").sum())
    return frame


# Manifest = danh sách ID do nhóm/giảng viên đã chốt.
# Kiểm tra đủ 1.000 review LLM, 100 nhãn tay và nhãn tay là tập con;
# sai thì DỪNG, tuyệt đối không tự lấy review khác để đủ số lượng.
def qa_manifest(config, assignments, report):
    frame = read_csv(config.assignment_paths["review_ids"])
    if not schema(frame, ("review_id", "listing_id", "Mau_LLM", "Nhan_tay"), report, "manifest", "all", "6,7"):
        raise ValueError("Review manifest lacks sample flags")
    # load_assignments da kiem unique ID, listing mapping va regions.
    invalid_tokens = ~frame.Mau_LLM.isin(["", "x"]) | ~frame.Nhan_tay.isin(["", "x"])
    add_columns(frame, llm_selected=frame.Mau_LLM.eq("x"), manual_selected=frame.Nhan_tay.eq("x"))
    selected, manual = int(frame.llm_selected.sum()), int(frame.manual_selected.sum())
    subset_error = frame.manual_selected & ~frame.llm_selected
    bad_ids = frame.review_id.map(id_status).ne("valid") | frame.listing_id.map(id_status).ne("valid")
    valid = selected == 1000 and manual == 100 and not (invalid_tokens | subset_error | bad_ids).any()
    record(report, frame, "LLM_MANIFEST", "manifest", "all", "6,7", "1000 LLM; 100 manual subset; valid string IDs",
           "preassigned sample only", "FLAG", int((invalid_tokens | subset_error | bad_ids).sum()) + int(selected != 1000) + int(manual != 100),
           eligible=selected, status="pass" if valid else "blocked",
           notes=f"selected={selected}; manual={manual}; subset_errors={int(subset_error.sum())}; assignment_sha256={json.dumps(assignments.fingerprints, sort_keys=True)}")
    if not valid:
        raise ValueError("Manifest invalid; do not resample")
    return frame


# Tạo đúng roster 1.000 ID từ manifest, dù review nào thiếu cũng giữ dòng đó.
# Hash comments dùng để kiểm tra prediction thuộc đúng văn bản review ban đầu.
def llm_source_view(reviews, manifest, report, snapshot):
    roster = manifest.loc[manifest.llm_selected, ["review_id", "listing_id", "manual_selected"]].copy()
    # Conflict review khong duoc chon ban dau/cuoi; roster van giu selected ID.
    right = reviews.loc[reviews.analysis_row, ["id", "comments", "review_date_iso", "listing_room_type",
                                              "llm_source_eligible", "listing_id"]].rename(
                                                  columns={"id": "review_id", "listing_id": "source_listing_id"})
    view = roster.merge(right, how="left", on="review_id", validate="one_to_one", indicator=True)
    if len(view) != len(roster):
        raise ValueError("LLM source merge multiplied rows")
    view.attrs = reviews.attrs.copy()
    view["source_valid"] = view.llm_source_eligible.eq(True) & view.listing_id.eq(view.source_listing_id)
    view["qa_snapshot"] = snapshot
    view["source_text_sha256"] = view.comments.map(lambda v: hashlib.sha256(v.encode("utf-8")).hexdigest() if isinstance(v, str) else "")
    record(report, view, "LLM_SOURCE_COVERAGE", "llm_source", snapshot, "6,7", "selected source is unique, mapped, dated, nonempty with valid room",
           "source verification before LLM", "EXCLUDE_FROM_KPI", (~view.source_valid).sum(),
           eligible=int(view.source_valid.sum()), notes=f"selected={len(view)}; matched={int(view._merge.eq('both').sum())}")
    return view.drop(columns="_merge")


# QA POST-LLM: kiểm tra identity, version, nguồn review, sentiment và 5 cặp aspect.
# Không yêu cầu aspect phải có '1': cả hai ô trống vẫn hợp lệ khi LLM chạy thành công.
# Không trộn dự đoán từ nhiều phiên bản/prompt.
def qa_predictions(source, roster, version, report, snapshot):
    """Tra ve tat ca prediction va view 1000 ID; khong tinh KPI6/7."""
    frame = source.copy()
    aspects = [f"{a}_{polarity}" for a in ASPECTS for polarity in ("positive", "negative")]
    required = ["snapshot", "review_id", "listing_id", "version", "model", "prompt_version",
                "source_text_sha256", "request_status", "parse_status", "analysis_status", "sentiment", *aspects]
    schema(frame, [c for c in required if c not in aspects], report, "predictions", snapshot, "6,7", stage="post")
    absent_aspects = sorted(set(aspects) - set(frame.columns))
    record(report, frame, "ASPECT_SCHEMA", "predictions", snapshot, "7", "ten independent aspect columns present",
           "missing aspect fields invalidate affected pairs only", "EXCLUDE_FROM_KPI", len(absent_aspects),
           status="blocked" if absent_aspects else "pass", notes=",".join(absent_aspects), stage="post")
    identity = {"snapshot", "review_id", "listing_id", "version", "source_text_sha256"}
    if not identity <= set(frame.columns):
        raise ValueError(f"Prediction identity/provenance schema missing: {sorted(identity - set(frame.columns))}")
    mark_keys(frame, ["snapshot", "review_id", "version"], report, "predictions", snapshot, "6,7", stage="post")
    normalize_pair(frame, "sentiment", "sentiment_label", "sentiment_status",
                   lambda v: ("", "missing") if missing(v) else (token(v), "valid")
                   if token(v) in {"positive", "negative", "mixed"} else ("", "domain_error"))
    add_columns(frame, request_valid=field(frame, "request_status").eq("success"),
                parse_valid=field(frame, "parse_status").eq("valid"),
                analysis_valid=field(frame, "analysis_status").eq("complete"),
                provenance_valid=~field(frame, "model").map(missing).astype(bool) & ~field(frame, "prompt_version").map(missing).astype(bool),
                selected_version=field(frame, "version").eq(version) & field(frame, "snapshot").eq(snapshot))
    # Mỗi aspect có hai cờ độc lập positive/negative nhận '1' hoặc trống.
    # Cặp không đúng schema / lỗi gọi LLM => không được coi trống là 'không nhắc tới'.
    for aspect in ASPECTS:
        cols = [f"{aspect}_{p}" for p in ("positive", "negative")]
        valid = pd.Series(set(cols) <= set(frame.columns), index=frame.index)
        for col in cols:
            valid &= field(frame, col).isin(["", "1"])
        add_columns(frame, **{f"{aspect}_pair_valid": valid & frame.request_valid & frame.parse_valid & frame.analysis_valid})
        record(report, frame, f"ASPECT_PAIR_INVALID_{aspect}", "predictions", snapshot, "7",
               "pair is 1/blank with complete successful analysis", "unverified aspect is not valid absence of evidence",
               "EXCLUDE_FROM_KPI", (~frame[f"{aspect}_pair_valid"]).sum(), stage="post")
    # Chon version ro rang; ban sao hoan toan chi lay mot, conflict khong merge.
    # Chỉ ghép những prediction có khóa không xung đột, đúng snapshot và version.
    right = frame.loc[frame.analysis_row & frame.selected_version].copy()
    add_columns(right, prediction_row=right.index)
    view = roster.merge(right, how="left", on="review_id", suffixes=("_source", "_prediction"),
                        validate="one_to_one", indicator=True)
    if len(view) != len(roster):
        raise ValueError("Prediction merge multiplied sample")
    view.attrs = frame.attrs.copy()
    matched = view._merge.eq("both")
    # Provenance hợp lệ khi cả listing_id và SHA256 nội dung khớp nguồn.
    # KPI6 dùng sentiment positive/negative/mixed; KPI7 cùng tập review cơ sở.
    provenance = matched & view.get("listing_id_prediction", pd.Series("", index=view.index)).eq(view.listing_id_source) & view.get(
        "source_text_sha256_prediction", pd.Series("", index=view.index)).eq(view.source_text_sha256_source)
    base = (view.source_valid & provenance & view.request_valid.eq(True) & view.parse_valid.eq(True)
            & view.analysis_valid.eq(True) & view.provenance_valid.eq(True))
    add_columns(view, kpi6_eligible=base & view.sentiment_status.eq("valid"))
    add_columns(view, kpi7_base_eligible=view.kpi6_eligible)
    add_columns(frame, source_match_status="unmatched", kpi6_eligible=False, kpi7_base_eligible=False)
    # Phản ánh kết quả ghép từ view 1.000 ID về đúng hàng trong tệp predictions.
    for idx, row in view.loc[matched].iterrows():
        target = int(row.prediction_row)
        frame.loc[target, "source_match_status"] = "matched" if provenance.loc[idx] else "mismatch"
        frame.loc[target, ["kpi6_eligible", "kpi7_base_eligible"]] = bool(view.loc[idx, "kpi6_eligible"])
    # KPI7: xét đủ dữ liệu theo từng cặp (room_type, aspect), không cộng nhầm
    # nhãn chưa kiểm chứng vào mẫu số. Mẫu số tính ở script KPI là số review
    # có positive==1 HOẶC negative==1; mẫu số 0 phải ra NaN, không phải 0%.
    for aspect in ASPECTS:
        col = f"{aspect}_pair_valid"
        add_columns(view, **{f"{aspect}_eligible": view.kpi7_base_eligible & view[col].eq(True),
                             f"{aspect}_room_complete": False})
        add_columns(frame, **{f"{aspect}_eligible": False, f"{aspect}_room_complete": False})
        # Mot nhan chua xac minh cung chan room x aspect, khong thu hep mau am tham.
        # room_complete được đánh giá TRÊN tập KPI6 hợp lệ (base),
        # không tự động bao gồm review bị loại khỏi base vì thiếu prediction.
        for room, group in view.loc[view.kpi7_base_eligible].groupby("listing_room_type"):
            complete = bool(group[col].eq(True).all())
            mask = view.listing_room_type.eq(room)
            view.loc[mask, f"{aspect}_room_complete"] = complete
            rows = view.loc[mask & matched, "prediction_row"].astype(int)
            frame.loc[rows, f"{aspect}_room_complete"] = complete
            record(report, frame, f"ASPECT_COMPLETE_{room}_{aspect}", "predictions", snapshot, "7",
                   "every base review in room has verified aspect pair", "incomplete pair blocks room/aspect",
                   "EXCLUDE_FROM_KPI", (~group[col].eq(True)).sum(), total=len(group),
                   eligible=int(group[col].eq(True).sum()), status="pass" if complete else "blocked", stage="post")
        valid_rows = view.loc[view[f"{aspect}_eligible"], "prediction_row"].astype(int)
        frame.loc[valid_rows, f"{aspect}_eligible"] = True
    for rule, bad in (
        ("REQUEST_ERROR", ~frame.request_valid), ("PARSE_ERROR", ~frame.parse_valid),
        ("ANALYSIS_ERROR", ~frame.analysis_valid), ("SENTIMENT_INVALID", frame.sentiment_status.ne("valid")),
        ("PROVENANCE_INVALID", ~frame.provenance_valid),
        ("PREDICTION_UNMATCHED", frame.selected_version & frame.source_match_status.ne("matched")),
    ):
        record(report, frame, rule, "predictions", snapshot, "6,7", rule.lower(),
               "invalid prediction value/state", "EXCLUDE_FROM_KPI", bad.sum(), stage="post")
    record(report, view, "LLM_PREDICTION_COVERAGE", "predictions", snapshot, "6,7", "selected version matched to preassigned sample",
           "shared verified sentiment base; aspect errors independent", "EXCLUDE_FROM_KPI", (~view.kpi6_eligible).sum(),
           eligible=int(view.kpi6_eligible.sum()), stage="post",
           notes=f"version={version}; selected={len(view)}; predicted={int(matched.sum())}; valid={int(view.kpi6_eligible.sum())}; error={int((matched & ~view.kpi6_eligible).sum())}; unmatched={int((~matched).sum())}")
    return frame, view.drop(columns="_merge")


# Kết quả làm sạch chỉ ghi vào data/processed; raw và interim không bị sửa.
def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, encoding="utf-8", na_rep="", lineterminator="\n", compression=compression)


# Chạy POST giữ các quy tắc PRE, thay riêng phần báo cáo POST cũ nếu có.
def write_report(rows, path, stage):
    current = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    if stage == "post" and path.is_file():
        previous = pd.read_csv(path, keep_default_na=False)
        current = pd.concat([previous.loc[previous.stage.eq("pre")], current], ignore_index=True)
    write_csv(current, path)


# Điều phối chính: đọc config + manifest rồi chạy 1 trong 2 giai đoạn.
# PRE: xử lý đủ 4 snapshot / ba tệp nguồn. POST: chỉ kiểm dự đoán LLM.
def run(config_path, *, stage="pre", predictions=None, version=None, root=PROJECT_ROOT):
    """Tra ve cac DataFrame; CLI chi ghi mot lan moi bang. Raw/interim chi doc."""
    config = load_config(Path(config_path))
    report, tables = [], {}
    report_path = Path(root) / "reports" / "qa_report.csv"
    try:
        assignments = load_assignments(config)
        manifest = qa_manifest(config, assignments, report)
        # Thứ tự quan trọng: listings trước để calendar/reviews nối loại phòng.
        if stage == "pre":
            for snapshot in config.snapshots:
                folder = Path(root) / "data" / "interim" / snapshot
                sources = {name: read_csv(folder / f"{name}_filtered.csv.gz")
                           for name in ("listings", "calendar", "reviews")}
                listings = qa_listings(sources["listings"], snapshot, report)
                calendar = qa_calendar(sources["calendar"], listings, snapshot, report)
                reviews = qa_reviews(sources["reviews"], listings, manifest, snapshot, report)
                for name, expected, key in (("listings", assignments.listing_areas, "id"),
                                            ("reviews", assignments.review_listings, "id")):
                    absent = set(expected) - set(field(sources[name], key))
                    record(report, sources[name], "ASSIGNED_IDS_ABSENT", name, snapshot, "1,2,3,4,5" if name == "listings" else "6,7",
                           "assigned IDs absent in this snapshot", "no invented rows or cross-snapshot fill", "FLAG", len(absent),
                           total=len(expected), eligible=len(expected) - len(absent), notes=json.dumps(sorted(absent)))
                tables[snapshot] = dict(listings=listings, calendar=calendar, reviews=reviews)
                for name, frame in tables[snapshot].items():
                    write_csv(frame, Path(root) / "data" / "processed" / snapshot / f"{name}_clean.csv.gz")
                print(f"{snapshot}: listings={len(listings)}, calendar={len(calendar)}, reviews={len(reviews)}")
            last = config.snapshots[-1]
            tables["llm_source"] = llm_source_view(tables[last]["reviews"], manifest, report, last)
            record(report, tables["llm_source"], "PREDICTIONS_PENDING", "predictions", last, "6,7",
                   "post stage optional until LLM results exist", "pre-LLM QA successful without predictions", "FLAG", 0,
                   status="pass", notes="Run --stage post --predictions FILE --version VERSION after LLM")
        # POST không gọi mô hình; dùng CSV kết quả do bước LLM xuất ra.
        else:
            # Post thay phan LLM report, giu nguyen ket qua QA nguon.
            if not report_path.is_file():
                raise ValueError("Run --stage pre before --stage post")
            report.clear()
            last = config.snapshots[-1]
            folder = Path(root) / "data" / "interim" / last
            scratch = []
            listings = qa_listings(read_csv(folder / "listings_filtered.csv.gz"), last, scratch)
            reviews = qa_reviews(read_csv(folder / "reviews_filtered.csv.gz"), listings, manifest, last, scratch)
            roster = llm_source_view(reviews, manifest, scratch, last)
            if predictions is None:
                record(report, roster, "PREDICTIONS_PENDING", "predictions", last, "6,7", "predictions provided",
                       "LLM results not available", "FLAG", len(roster), status="warning", stage="post")
            else:
                if not version:
                    raise ValueError("--version is required with --predictions")
                clean, view = qa_predictions(read_csv(Path(predictions)), roster, version, report, last)
                tables["predictions"], tables["llm"] = clean, view
                write_csv(clean, Path(root) / "data" / "processed" / last / "predictions_clean.csv.gz")
                write_csv(view, Path(root) / "data" / "processed" / last / "llm_clean.csv.gz")
    # Dù lỗi sau khi đã phát hiện quy tắc, vẫn cố ghi báo cáo QA đã thu thập.
    finally:
        if report:
            write_report(report, report_path, stage)
    return tables, pd.DataFrame(report, columns=REPORT_COLUMNS)


# Điểm vào dòng lệnh; trả 1 nếu có blocked/lỗi để pipeline biết cần dừng.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "nyc.yaml")
    parser.add_argument("--stage", choices=("pre", "post"), default="pre")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--version")
    args = parser.parse_args(argv)
    if args.stage == "pre" and (args.predictions or args.version):
        parser.error("--predictions/--version require --stage post")
    try:
        _, report = run(args.config, stage=args.stage, predictions=args.predictions, version=args.version)
        blocked = int(report.status.eq("blocked").sum())
        print(f"reports/qa_report.csv: {len(report)} current-stage rules; blocked={blocked}")
        return int(blocked > 0)
    except (ValueError, OSError, KeyError, pd.errors.ParserError) as exc:
        print(f"QA failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
