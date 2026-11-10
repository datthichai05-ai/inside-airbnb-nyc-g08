"""Tai du lieu goc Inside Airbnb theo configs/nyc.yaml.

Chay tai thu muc goc project:
    python src/download.py --dry-run
    python src/download.py --snapshot 2026-06-14
    python src/download.py

Cac tep duoc luu tai data/raw/<snapshot>/{listings,calendar,reviews}.csv.gz.
"""

from __future__ import annotations

import argparse
import gzip
import re
import time
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


class DownloadError(Exception):
    """Loi du lieu tai ve khong hop le."""


def load_config(path: Path) -> tuple[str, str, list[str]]:
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

    return city, data_root, snapshots


def valid_gzip(path: Path) -> bool:
    """Doc het gzip de kiem tra header, CRC va du lieu khong bi cat cut."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with gzip.open(path, "rb") as file:
            while file.read(CHUNK_SIZE):
                pass
        return True
    except (OSError, EOFError):
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Tai du lieu Inside Airbnb theo YAML")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Duong dan YAML (mac dinh: configs/nyc.yaml)")
    parser.add_argument("--snapshot", help="Chi tai mot moc YYYY-MM-DD trong YAML")
    parser.add_argument("--dry-run", action="store_true", help="Chi in URL, khong tai")
    parser.add_argument("--force", action="store_true", help="Tai lai, ghi de file sau khi kiem tra")
    args = parser.parse_args()

    try:
        city, data_root, snapshots = load_config(args.config)
    except (ValueError, yaml.YAMLError, OSError) as exc:
        parser.error(str(exc))

    if args.snapshot:
        if args.snapshot not in snapshots:
            parser.error(f"Snapshot {args.snapshot} khong nam trong configs YAML")
        snapshots = [args.snapshot]

    print(f"Thanh pho: {city} | Snapshots: {', '.join(snapshots)}")
    failures = []
    with requests.Session() as session:
        for snapshot in snapshots:
            print(f"\n[{snapshot}]")
            for filename in FILES:
                url = f"{BASE_URL}/{data_root}/{snapshot}/data/{filename}"
                destination = PROJECT_ROOT / "data" / "raw" / snapshot / filename
                if args.dry_run:
                    print(f"  {url}\n    -> {destination}")
                    continue
                try:
                    download_file(session, url, destination, force=args.force)
                except DownloadError as exc:
                    print(f"  [FAIL] {exc}")
                    failures.append(f"{snapshot}/{filename}")

    if failures:
        print("\nFile loi: " + ", ".join(failures))
        return 1
    print("\nHoan thanh." if not args.dry_run else "\nDry-run hoan thanh (chua tai du lieu).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
