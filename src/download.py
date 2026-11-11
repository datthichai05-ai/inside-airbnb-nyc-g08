"""Tai va loc du lieu Inside Airbnb theo configs/nyc.yaml, chua thuc hien QA.

Chay tai thu muc goc project:
    python src/download.py --dry-run
    python src/download.py --snapshot 2026-06-14
    python src/download.py --filter-only
    python src/download.py --filter-only --force-filter
    python src/download.py --download-only
    python src/download.py

Raw: data/raw/<snapshot>/{listings,calendar,reviews}.csv.gz.
Interim: data/interim/<snapshot>/{listings,calendar,reviews}_filtered.csv.gz.
filter_summary.json luu thong ke va dau vet dau vao de tranh dung lai ket qua cu.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "nyc.yaml"
BASE_URL = "https://data.insideairbnb.com"
FILES = ("listings.csv.gz", "calendar.csv.gz", "reviews.csv.gz")
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0
TIMEOUT = (10, 60)  # Thoi gian ket noi va cho du lieu (giay)
CHUNK_SIZE = 1024 * 1024
FILTER_VERSION = 1
# csv.reader streams one record at a time, including long, multiline comments.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


class DownloadError(Exception):
    """Loi du lieu tai ve khong hop le."""


@dataclass(frozen=True)
class Config:
    city: str
    data_root: str  # Duong dan URL Inside Airbnb, khong phai thu muc cuc bo.
    snapshots: list[str]
    assignment_paths: dict[str, Path]


@dataclass(frozen=True)
class Assignments:
    listing_areas: dict[str, str]
    review_listings: dict[str, str]
    regions: dict[str, tuple[float, float, float, float]]
    fingerprints: dict[str, str]


def load_config(path: Path, *, download_only: bool = False) -> Config:
    """Doc va kiem tra cac khoa can thiet trong YAML."""
    if not path.is_file():
        raise ValueError(f"Khong tim thay file cau hinh: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("YAML phai la mot dictionary.")

    city = config.get("city")
    data_root = config.get("data_root")
    raw_snapshots = config.get("snapshots")

    if not isinstance(city, str) or not city.strip():
        raise ValueError("Thieu 'city' trong YAML.")
    if not isinstance(data_root, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_-]+)*", data_root):
        raise ValueError("'data_root' khong hop le; vi du: united-states/ny/new-york-city")
    if not isinstance(raw_snapshots, list) or not raw_snapshots:
        raise ValueError("'snapshots' phai la danh sach ngay khong rong.")

    snapshots = []
    for value in raw_snapshots:
        # PyYAML co the tu chuyen ngay khong co dau nhay thanh datetime.date.
        snapshot = value.isoformat() if isinstance(value, date) else str(value)
        try:
            if date.fromisoformat(snapshot).isoformat() != snapshot:
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"Snapshot sai dinh dang YYYY-MM-DD: {snapshot}") from exc
        if snapshot in snapshots:
            raise ValueError(f"Snapshot bi lap: {snapshot}")
        snapshots.append(snapshot)

    assignment_paths = {}
    if not download_only:
        assignments = config.get("assignments")
        if not isinstance(assignments, dict):
            raise ValueError("Thieu mapping 'assignments' trong YAML (listing_ids, review_ids, regions).")
        for key in ("listing_ids", "review_ids", "regions"):
            value = assignments.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Thieu duong dan assignments.{key} trong YAML.")
            # Duong dan tuong doi duoc tinh tu thu muc chua YAML.
            assignment_paths[key] = (path.resolve().parent / value).resolve()
    return Config(city, data_root, snapshots, assignment_paths)


def column_indexes(header: list[str], required: tuple[str, ...], path: Path) -> dict[str, int]:
    for name in required:
        if header.count(name) != 1:
            raise ValueError(f"{path}: can dung mot cot '{name}', cac cot hien co: {header}")
    return {name: header.index(name) for name in required}


def read_assignment(path: Path, required: tuple[str, ...]) -> tuple[list[dict[str, str]], str]:
    if not path.is_file():
        raise ValueError(f"Khong tim thay file phan cong: {path}")
    content = path.read_bytes()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
    header = reader.fieldnames or []
    column_indexes(header, required, path)
    if len(set(header)) != len(header):
        raise ValueError(f"{path}: ten cot bi trung lap.")
    rows = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path}: sai so cot tai dong {reader.line_num}.")
        if any(not row[key].strip() for key in required):
            raise ValueError(f"{path}: gia tri phan cong rong tai dong {reader.line_num}.")
        rows.append(row)
    if not rows:
        raise ValueError(f"File phan cong rong: {path}")
    return rows, hashlib.sha256(content).hexdigest()


def load_assignments(config: Config) -> Assignments:
    schemas = {
        "listing_ids": ("listing_id", "analysis_area"),
        "review_ids": ("review_id", "listing_id"),
        "regions": ("analysis_area", "lon_min", "lat_min", "lon_max", "lat_max"),
    }
    rows, fingerprints = {}, {}
    for key, columns in schemas.items():
        rows[key], fingerprints[key] = read_assignment(config.assignment_paths[key], columns)

    def unique_mapping(key: str, id_column: str, value_column: str) -> dict[str, str]:
        result = {}
        for row in rows[key]:
            identifier = row[id_column]
            if identifier in result:
                raise ValueError(f"{config.assignment_paths[key]}: {id_column} bi lap: {identifier}")
            result[identifier] = row[value_column]
        return result

    listing_areas = unique_mapping("listing_ids", "listing_id", "analysis_area")
    review_listings = unique_mapping("review_ids", "review_id", "listing_id")
    unknown_listings = set(review_listings.values()) - listing_areas.keys()
    if unknown_listings:
        raise ValueError(f"Review phan cong tro toi listing ngoai danh sach: {sorted(unknown_listings)}")

    regions = {}
    for row in rows["regions"]:
        area = row["analysis_area"]
        try:
            bounds = tuple(float(row[key]) for key in schemas["regions"][1:])
        except ValueError as exc:
            raise ValueError(f"Regions {area}: toa do khong phai so.") from exc
        xmin, ymin, xmax, ymax = bounds
        if (area in regions or not all(math.isfinite(value) for value in bounds)
                or not -180 <= xmin < xmax <= 180 or not -90 <= ymin < ymax <= 90):
            raise ValueError(f"Regions {area}: ma vung lap hoac bounding box khong hop le.")
        regions[area] = bounds
    unknown_areas = set(listing_areas.values()) - regions.keys()
    if unknown_areas:
        raise ValueError(f"analysis_area khong co trong regions: {sorted(unknown_areas)}")
    return Assignments(listing_areas, review_listings, regions, fingerprints)


def valid_gzip(path: Path) -> bool:
    """Doc het gzip de kiem tra header, CRC va du lieu khong bi cat cut."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with gzip.open(path, "rb") as file:
            while file.read(CHUNK_SIZE):
                pass
        return True
    except (OSError, EOFError, zlib.error):
        return False


def download_file(session: requests.Session, url: str, destination: Path, *, force: bool) -> bool:
    """Tra ve True neu tai moi, False neu bo qua file da hop le."""
    if not force and valid_gzip(destination):
        print(f"  [SKIP] {destination.name} (da ton tai, gzip hop le)")
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    if force:
        part.unlink(missing_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "UET-IAI-InsideAirbnb-Student-Project/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"

        try:
            with session.get(url, headers=headers, stream=True, timeout=TIMEOUT) as response:
                if response.status_code == 416 and offset and valid_gzip(part):
                    part.replace(destination)
                    print(f"  [OK] {destination.name} (hoan tat tu .part)")
                    return True
                if response.status_code == 416:
                    part.unlink(missing_ok=True)
                    raise DownloadError("May chu tu choi phan Range; se tai lai tu dau")

                response.raise_for_status()
                if response.status_code not in (200, 206):
                    raise DownloadError(f"HTTP khong mong doi: {response.status_code}")

                # May chu co the khong ho tro Range va tra ve 200: tai lai tu dau.
                append = bool(offset and response.status_code == 206)
                if response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.match(r"bytes (\d+)-\d+/(?:\d+|\*)$", content_range)
                    if not match or int(match.group(1)) != offset:
                        part.unlink(missing_ok=True)
                        raise DownloadError("Content-Range khong dung offset")

                expected = response.headers.get("Content-Length")
                written = 0
                with part.open("ab" if append else "wb") as file:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            file.write(chunk)
                            written += len(chunk)
                if expected is not None and written != int(expected):
                    raise DownloadError(f"Thieu byte: nhan {written}, can {expected}")

            if not valid_gzip(part):
                part.unlink(missing_ok=True)
                raise DownloadError("File tai ve khong phai gzip hop le (hoac bi cat cut)")

            part.replace(destination)
            print(f"  [OK] {destination.name} ({destination.stat().st_size:,} bytes)")
            return True

        except (requests.RequestException, DownloadError, OSError) as exc:
            # Khong retry voi URL khong ton tai / khong co quyen truy cap.
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                if exc.response.status_code in (400, 401, 403, 404, 410):
                    raise DownloadError(f"Khong tai duoc {url}: HTTP {exc.response.status_code}") from exc
            print(f"  [RETRY {attempt}/{MAX_RETRIES}] {destination.name}: {exc}")
            if attempt == MAX_RETRIES:
                raise DownloadError(f"Tai that bai: {url}") from exc
            time.sleep(BACKOFF_SECONDS * attempt)

    raise DownloadError(f"Tai that bai: {url}")


def validate_filtered(path: Path, header: list[str], key_index: int,
                      wanted: set[str], expected_rows: int) -> None:
    """Kiem tra gzip, schema, so dong va khoa truoc khi cong bo file interim."""
    count = 0
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != header:
            raise DownloadError(f"{path}: thu tu/ten cot dau ra khong khop nguon.")
        for row in reader:
            if len(row) != len(header) or row[key_index] not in wanted:
                raise DownloadError(f"{path}: sai so cot hoac co ID ngoai phan cong.")
            count += 1
    if count != expected_rows:
        raise DownloadError(f"{path}: so dong dau ra {count}, can {expected_rows}.")


def region_status(row: list[str], header: list[str], area: str, assignments: Assignments) -> str:
    """Doi chieu bounding box duoc giao, khong loai/chinh sua dong du lieu."""
    try:
        lon = float(row[header.index("longitude")])
        lat = float(row[header.index("latitude")])
    except ValueError:
        return "uncheckable"
    if not math.isfinite(lon) or not math.isfinite(lat):
        return "uncheckable"
    xmin, ymin, xmax, ymax = assignments.regions[area]
    # Hai bien deu duoc tinh; chi doi chieu vung da giao, khong gan lai vung.
    return "matched" if xmin <= lon <= xmax and ymin <= lat <= ymax else "outside"


def filter_csv(source: Path, destination: Path, table: str, assignments: Assignments,
               present_listings: set[str]) -> dict:
    """Loc streaming, moi gia tri CSV van la chuoi; bo nho khong tang theo raw."""
    wanted = set(assignments.review_listings if table == "reviews" else assignments.listing_areas)
    key = "listing_id" if table == "calendar" else "id"
    required = ("id", "listing_id") if table == "reviews" else (key,)
    stats = {"input_rows": 0, "output_rows": 0}
    found, absent_parents = set(), set()
    region_ids = {status: set() for status in ("outside", "uncheckable")}
    region_counts = {status: 0 for status in ("matched", "outside", "uncheckable")}
    absent_parent_rows = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    try:
        with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, [])
            indexes = column_indexes(header, required, source)
            stats["columns"] = header
            with gzip.open(part, "wt", encoding="utf-8", newline="", compresslevel=6) as output:
                writer = csv.writer(output)
                writer.writerow(header)
                for row in reader:
                    stats["input_rows"] += 1
                    if len(row) != len(header):
                        raise DownloadError(f"{source}: sai so cot tai dong {reader.line_num}.")
                    identifier = row[indexes[key]]
                    if identifier not in wanted:
                        continue
                    if table == "reviews":
                        parent = row[indexes["listing_id"]]
                        expected = assignments.review_listings[identifier]
                        if parent != expected:
                            raise DownloadError(
                                f"{source}: review_id={identifier} co listing_id={parent}, "
                                f"phan cong yeu cau {expected}; khong tu thay doi/bo qua cap sai.")
                    elif table == "calendar":
                        parent = identifier
                    else:
                        status = region_status(row, header, assignments.listing_areas[identifier], assignments)
                        region_counts[status] += 1
                        if status in region_ids:
                            region_ids[status].add(identifier)
                    if table != "listings" and parent not in present_listings:
                        absent_parents.add(parent)
                        absent_parent_rows += 1
                    writer.writerow(row)
                    stats["output_rows"] += 1
                    found.add(identifier)
        stats["missing_assigned_ids"] = sorted(wanted - found)
        if table == "listings":
            stats["regions"] = {"rows": region_counts,
                                "outside_listing_ids": sorted(region_ids["outside"]),
                                "uncheckable_listing_ids": sorted(region_ids["uncheckable"])}
        else:
            # Van giu dong dung phan cong neu listing vang mat trong snapshot.
            stats["listing_ids_absent_from_listings"] = sorted(absent_parents)
            stats["rows_with_listing_absent"] = absent_parent_rows
        validate_filtered(part, header, indexes[key], wanted, stats["output_rows"])
        part.replace(destination)
        return stats
    finally:
        # Khong de lai gzip interim bi ghi do; raw .part van duoc giu de resume.
        part.unlink(missing_ok=True)


def file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cached_summary(path: Path, inputs: dict, outputs: dict[str, Path]) -> dict | None:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["inputs"] != inputs:
            return None
        if summary["outputs"] != {name: file_signature(output) for name, output in outputs.items()}:
            return None
        if set(summary["tables"]) != set(outputs):
            return None
        for table in outputs:
            stats = summary["tables"][table]
            if (not isinstance(stats["missing_assigned_ids"], list)
                    or not isinstance(stats["columns"], list)
                    or not 0 <= stats["output_rows"] <= stats["input_rows"]):
                return None
            if table == "listings":
                if not all(isinstance(stats["regions"]["rows"][key], int)
                           for key in ("matched", "outside", "uncheckable")):
                    return None
            elif (not isinstance(stats["listing_ids_absent_from_listings"], list)
                  or not isinstance(stats["rows_with_listing_absent"], int)):
                return None
        if all(valid_gzip(output) for output in outputs.values()):
            return summary
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def print_stats(table: str, stats: dict, *, skipped: bool = False) -> None:
    action = "SKIP FILTER" if skipped else "FILTER"
    print(f"  [{action}] {table}: {stats['input_rows']:,} -> {stats['output_rows']:,} dong; "
          f"{len(stats['columns'])} cot; ID phan cong vang mat: {len(stats['missing_assigned_ids']):,}")
    if table == "listings":
        counts = stats["regions"]["rows"]
        print(f"    Regions: khop={counts['matched']}, ngoai vung={counts['outside']}, "
              f"khong doi chieu duoc={counts['uncheckable']} (khong loai dong)")
    else:
        print(f"    Listing vang mat trong listings cung snapshot: "
              f"{len(stats['listing_ids_absent_from_listings'])} ID / {stats['rows_with_listing_absent']} dong")


def filter_snapshot(raw_dir: Path, interim_dir: Path, assignments: Assignments, *, force: bool) -> dict:
    sources = {filename.split(".")[0]: raw_dir / filename for filename in FILES}
    outputs = {table: interim_dir / f"{table}_filtered.csv.gz" for table in sources}
    for source in sources.values():
        if not source.is_file():
            raise DownloadError(f"Thieu raw: {source}; hay tai truoc khi dung --filter-only.")
    inputs = {"filter_version": FILTER_VERSION, "assignments_sha256": assignments.fingerprints,
              "sources": {table: file_signature(source) for table, source in sources.items()}}
    summary_path = interim_dir / "filter_summary.json"
    summary = None if force else cached_summary(summary_path, inputs, outputs)
    if summary is not None:
        for table, stats in summary["tables"].items():
            print_stats(table, stats, skipped=True)
        return summary

    tables = {}
    present_listings = set()
    for table, source in sources.items():
        tables[table] = filter_csv(source, outputs[table], table, assignments, present_listings)
        if table == "listings":
            present_listings = assignments.listing_areas.keys() - set(tables[table]["missing_assigned_ids"])
        print_stats(table, tables[table])
    if inputs["sources"] != {table: file_signature(source) for table, source in sources.items()}:
        raise DownloadError("Raw thay doi trong luc loc; hay chay lai de tao interim nhat quan.")
    summary = {"inputs": inputs, "outputs": {table: file_signature(output) for table, output in outputs.items()},
               "tables": tables}
    part = summary_path.with_name(summary_path.name + ".part")
    try:
        part.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        part.replace(summary_path)
    finally:
        part.unlink(missing_ok=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Tai va loc du lieu Inside Airbnb theo YAML (chua QA)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Duong dan YAML (mac dinh: configs/nyc.yaml)")
    parser.add_argument("--snapshot", help="Chi xu ly mot moc YYYY-MM-DD trong YAML")
    parser.add_argument("--dry-run", action="store_true", help="Chi in ke hoach, khong tai/ghi file")
    parser.add_argument("--force", action="store_true", help="Tai lai raw (neu co tai) va loc lai")
    parser.add_argument("--force-filter", action="store_true", help="Loc lai interim, khong ep tai lai raw")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--filter-only", action="store_true", help="Chi loc raw da co, khong ket noi mang")
    mode.add_argument("--download-only", action="store_true", help="Chi tai raw, giu cach chay cu")
    args = parser.parse_args()
    if args.download_only and args.force_filter:
        parser.error("--force-filter khong dung cung --download-only")

    try:
        config = load_config(args.config, download_only=args.download_only)
        assignments = None if args.download_only else load_assignments(config)
    except (ValueError, yaml.YAMLError, OSError, csv.Error) as exc:
        parser.error(str(exc))

    snapshots = config.snapshots
    if args.snapshot:
        if args.snapshot not in snapshots:
            parser.error(f"Snapshot {args.snapshot} khong nam trong configs YAML")
        snapshots = [args.snapshot]

    print(f"Thanh pho: {config.city} | Snapshots: {', '.join(snapshots)}")
    if assignments is not None:
        print(f"Phan cong: {len(assignments.listing_areas)} listing ID, "
              f"{len(assignments.review_listings)} review ID, {len(assignments.regions)} vung toa do")
    failures = []
    with requests.Session() as session:
        for snapshot in snapshots:
            print(f"\n[{snapshot}]")
            raw_dir = PROJECT_ROOT / "data" / "raw" / snapshot
            interim_dir = PROJECT_ROOT / "data" / "interim" / snapshot
            download_failed = False
            if not args.filter_only:
                for filename in FILES:
                    url = f"{BASE_URL}/{config.data_root}/{snapshot}/data/{filename}"
                    destination = raw_dir / filename
                    if args.dry_run:
                        print(f"  [DOWNLOAD{' FORCE' if args.force else ' / SKIP IF VALID'}] "
                              f"{url}\n    -> {destination}")
                        continue
                    try:
                        download_file(session, url, destination, force=args.force)
                    except (DownloadError, OSError) as exc:
                        print(f"  [FAIL] {exc}")
                        failures.append(f"{snapshot}/{filename}")
                        download_failed = True
            if assignments is not None and not download_failed:
                if args.dry_run:
                    for filename in FILES:
                        output = interim_dir / filename.replace(".csv.gz", "_filtered.csv.gz")
                        print(f"  [FILTER{' FORCE' if args.force or args.force_filter else ' / SKIP IF CURRENT'}] "
                              f"{raw_dir / filename}\n    -> {output}")
                    print(f"    Thong ke -> {interim_dir / 'filter_summary.json'}")
                    continue
                try:
                    filter_snapshot(raw_dir, interim_dir, assignments, force=args.force or args.force_filter)
                except (DownloadError, OSError, EOFError, zlib.error, csv.Error, ValueError) as exc:
                    print(f"  [FAIL FILTER] {snapshot}: {exc}")
                    failures.append(f"{snapshot}/filter")

    if failures:
        print("\nFile loi: " + ", ".join(failures))
        return 1
    print("\nHoan thanh." if not args.dry_run else "\nDry-run hoan thanh (khong tai/ghi file).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
