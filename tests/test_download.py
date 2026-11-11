"""Selection/download invariants; no business QA or KPI calculations."""

import csv
import gzip
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from src import download


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "wt" if path.suffix == ".gz" else "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def read_csv(path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        return list(csv.reader(stream))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="airbnb-download-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = io.StringIO()
        redirect = redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.config_dir = self.root / "configs"
        self.config_dir.mkdir()
        self.config_path = self.config_dir / "city.yaml"
        self.write_config(["2024-02-29"])
        write_csv(self.config_dir / "listings.csv", ["listing_id", "analysis_area"],
                  [["0001", "A01"], ["9007199254740993123", "A02"], ["0003", "A01"]])
        write_csv(self.config_dir / "reviews.csv", ["review_id", "listing_id", "Mau_LLM", "Nhan_tay"],
                  [["0007", "0001", "", ""], ["9007199254740993999", "9007199254740993123", "x", ""],
                   ["0009", "0003", "", "x"], ["0010", "0001", "", ""]])
        write_csv(self.config_dir / "regions.csv", ["analysis_area", "lon_min", "lat_min", "lon_max", "lat_max"],
                  [["A01", "-75", "40", "-73", "42"], ["A02", "0", "0", "1", "1"]])
        self.config = download.load_config(self.config_path)
        self.assignments = download.load_assignments(self.config)
        self.raw = self.root / "data" / "raw" / "2024-02-29"
        self.interim = self.root / "data" / "interim" / "2024-02-29"
        comment = 'đúng, "nội dung"\r\n  giữ nguyên\n' + "x" * 140000
        self.fixtures = {
            "listings": (["price", "id", "longitude", "latitude", "name"],
                         [["", "0001", "-74", "41", "NA"],
                          ["$-100.000", "9007199254740993123", "-74", "41", "  text  "],
                          ["$9", "1", "-74", "41", "not assigned"],
                          ["", "0001", "-74", "41", "NA"]]),
            "calendar": (["date", "available", "listing_id", "price", "minimum_nights"],
                         [["1900-01-01", "f", "0001", "", "-9"],
                          ["2099-12-31", "t", "9007199254740993123", "$9999999", "NA"],
                          ["2024-02-29", "f", "0003", "$0.00", "0"],
                          ["2024-02-29", "t", "1", "$9", "1"],
                          ["1900-01-01", "f", "0001", "", "-9"]]),
            "reviews": (["comments", "id", "listing_id", "reviewer_id"],
                        [[comment, "0007", "0001", "00001"],
                         ["NA", "9007199254740993999", "9007199254740993123", "9007199254740993998"],
                         ["parent absent", "0009", "0003", ""],
                         ["not assigned", "7", "0001", "1"],
                         [comment, "0007", "0001", "00001"]]),
        }
        self.write_raw(self.raw)

    def write_config(self, snapshots):
        config = {"city": "test-city", "data_root": "country/state/city", "snapshots": snapshots,
                  "assignments": {"listing_ids": "listings.csv", "review_ids": "reviews.csv", "regions": "regions.csv"}}
        self.config_path.write_text(download.yaml.safe_dump(config), encoding="utf-8")

    def write_raw(self, raw_dir):
        for table, (header, rows) in self.fixtures.items():
            write_csv(raw_dir / f"{table}.csv.gz", header, rows)

    def filter(self, *, force=False):
        return download.filter_snapshot(self.raw, self.interim, self.assignments, force=force)

    def run_main(self, *arguments):
        with patch.object(download, "PROJECT_ROOT", self.root), \
                patch("sys.argv", ["download.py", "--config", str(self.config_path), *arguments]), \
                patch.object(download.requests.Session, "get", side_effect=AssertionError("Unexpected network")):
            return download.main()

    def test_exact_values_columns_order_and_source_duplicates(self):
        summary = self.filter()
        expected_indexes = {"listings": [0, 1, 3], "calendar": [0, 1, 2, 4], "reviews": [0, 1, 2, 4]}
        for table, indexes in expected_indexes.items():
            with self.subTest(table=table):
                header, rows = self.fixtures[table]
                actual = read_csv(self.interim / f"{table}_filtered.csv.gz")
                self.assertEqual(actual, [header] + [rows[index] for index in indexes])
                self.assertNotIn("snapshot", actual[0])
                self.assertEqual(summary["tables"][table]["input_rows"], len(rows))
                self.assertEqual(summary["tables"][table]["output_rows"], len(indexes))
        self.assertEqual(summary["tables"]["listings"]["missing_assigned_ids"], ["0003"])
        self.assertEqual(summary["tables"]["reviews"]["missing_assigned_ids"], ["0010"])

    def test_region_and_absent_parent_statistics_do_not_remove_rows(self):
        summary = self.filter()
        self.assertEqual(summary["tables"]["listings"]["regions"]["rows"],
                         {"matched": 2, "outside": 1, "uncheckable": 0})
        for table in ("calendar", "reviews"):
            self.assertEqual(summary["tables"][table]["listing_ids_absent_from_listings"], ["0003"])
            self.assertEqual(summary["tables"][table]["rows_with_listing_absent"], 1)

    def test_invalid_or_missing_coordinates_are_retained(self):
        header, rows = self.fixtures["listings"]
        for longitude in ("NaN", "", "NA", "inf"):
            with self.subTest(longitude=longitude):
                rows[0][2] = longitude
                write_csv(self.raw / "listings.csv.gz", header, rows)
                stats = self.filter(force=True)["tables"]["listings"]
                self.assertEqual(stats["output_rows"], 3)
                self.assertEqual(stats["regions"]["rows"]["uncheckable"], 1)
        write_csv(self.raw / "listings.csv.gz", ["id"], [["0001"]])
        self.assertEqual(self.filter(force=True)["tables"]["listings"]["regions"]["rows"]["uncheckable"], 1)

    def test_source_snapshot_column_is_preserved(self):
        write_csv(self.raw / "listings.csv.gz", ["id", "snapshot"], [["0001", "original value"]])
        self.filter()
        self.assertEqual(read_csv(self.interim / "listings_filtered.csv.gz"),
                         [["id", "snapshot"], ["0001", "original value"]])

    def test_review_pair_mismatch_keeps_previous_good_output_and_removes_part(self):
        self.filter()
        target = self.interim / "reviews_filtered.csv.gz"
        previous = target.read_bytes()
        write_csv(self.raw / "reviews.csv.gz", ["id", "listing_id", "comments"], [["0007", "0003", "original"]])
        with self.assertRaisesRegex(download.DownloadError, "review_id=0007.*listing_id=0003"):
            self.filter(force=True)
        self.assertEqual(target.read_bytes(), previous)
        self.assertFalse(target.with_name(target.name + ".part").exists())

    def test_corrupt_raw_is_not_published_as_interim(self):
        self.filter()
        target = self.interim / "reviews_filtered.csv.gz"
        previous = target.read_bytes()
        source = self.raw / "reviews.csv.gz"
        source.write_bytes(source.read_bytes()[:-8])
        with self.assertRaises(EOFError):
            self.filter(force=True)
        self.assertEqual(target.read_bytes(), previous)
        self.assertFalse(target.with_name(target.name + ".part").exists())

    def test_empty_selection_still_has_original_header(self):
        header = ["id", "arbitrary", "price"]
        write_csv(self.raw / "listings.csv.gz", header, [["not-assigned", "", "NA"]])
        stats = self.filter()["tables"]["listings"]
        self.assertEqual(read_csv(self.interim / "listings_filtered.csv.gz"), [header])
        self.assertEqual(stats["output_rows"], 0)
        self.assertEqual(len(stats["missing_assigned_ids"]), 3)

    def test_bad_raw_schema_and_row_width_fail_clearly(self):
        for header, rows in ((["listing_id"], [["0001"]]), (["id", "name"], [["0001"]])):
            with self.subTest(header=header):
                write_csv(self.raw / "listings.csv.gz", header, rows)
                with self.assertRaises((ValueError, download.DownloadError)):
                    self.filter()
                self.assertFalse((self.interim / "listings_filtered.csv.gz").exists())
                self.assertFalse((self.interim / "listings_filtered.csv.gz.part").exists())

    def test_rerun_skips_filter_and_force_refilters(self):
        self.filter()
        before = {path.name: download.file_signature(path) for path in self.interim.iterdir()}
        with patch.object(download, "filter_csv", side_effect=AssertionError("Unexpected refilter")):
            self.filter()
        self.assertEqual(before, {path.name: download.file_signature(path) for path in self.interim.iterdir()})
        with patch.object(download, "filter_csv", wraps=download.filter_csv) as filtering:
            self.filter(force=True)
            self.assertEqual(filtering.call_count, 3)

    def test_corrupt_output_or_summary_is_regenerated(self):
        self.filter()
        (self.interim / "calendar_filtered.csv.gz").write_bytes(b"incomplete gzip")
        self.filter()
        self.assertTrue(download.valid_gzip(self.interim / "calendar_filtered.csv.gz"))
        for invalid in ("{", "[]", '{"inputs": null}'):
            (self.interim / "filter_summary.json").write_text(invalid, encoding="utf-8")
            with patch.object(download, "filter_csv", wraps=download.filter_csv) as filtering:
                self.filter()
                self.assertEqual(filtering.call_count, 3)
        summary_path = self.interim / "filter_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        del summary["tables"]["calendar"]["rows_with_listing_absent"]
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        with patch.object(download, "filter_csv", wraps=download.filter_csv) as filtering:
            self.filter()
            self.assertEqual(filtering.call_count, 3)

    def test_changed_source_or_assignment_invalidates_cache(self):
        self.filter()
        write_csv(self.raw / "calendar.csv.gz", ["listing_id", "date"], [["0001", "new day"]])
        with patch.object(download, "filter_csv", wraps=download.filter_csv) as filtering:
            self.filter()
            self.assertEqual(filtering.call_count, 3)
        write_csv(self.config_dir / "reviews.csv", ["review_id", "listing_id"], [["0009", "0003"]])
        self.assignments = download.load_assignments(self.config)
        self.filter()
        self.assertEqual(read_csv(self.interim / "reviews_filtered.csv.gz")[1:], [self.fixtures["reviews"][1][2]])

    def test_assignment_files_headers_duplicates_and_links_are_validated(self):
        (self.config_dir / "listings.csv").unlink()
        with self.assertRaisesRegex(ValueError, "Khong tim thay file phan cong"):
            download.load_assignments(self.config)
        for header, rows, message in (
                (["id", "analysis_area"], [["0001", "A01"]], "listing_id"),
                (["listing_id", "analysis_area"], [["0001", "A01"], ["0001", "A01"]], "bi lap"),
                (["listing_id", "analysis_area"], [["0001", "A01"]], "ngoai danh sach"),
                (["listing_id", "analysis_area"], [["0001", ""]], "rong")):
            with self.subTest(message=message):
                write_csv(self.config_dir / "listings.csv", header, rows)
                with self.assertRaisesRegex(ValueError, message):
                    download.load_assignments(self.config)

    def test_unknown_or_invalid_regions_fail(self):
        for rows, message in (
                ([["A01", "-75", "40", "-73", "42"]], "analysis_area"),
                ([["A01", "-73", "40", "-75", "42"]], "bounding box"),
                ([["A01", "NaN", "40", "-73", "42"]], "bounding box")):
            with self.subTest(rows=rows):
                write_csv(self.config_dir / "regions.csv", ["analysis_area", "lon_min", "lat_min", "lon_max", "lat_max"], rows)
                with self.assertRaisesRegex(ValueError, message):
                    download.load_assignments(self.config)

    def test_filter_only_and_default_rerun_never_request_existing_raw(self):
        self.assertEqual(self.run_main("--filter-only"), 0)
        before = {path.name: download.file_signature(path) for path in self.interim.iterdir()}
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(before, {path.name: download.file_signature(path) for path in self.interim.iterdir()})

    def test_force_filter_does_not_redownload(self):
        self.filter()
        with patch.object(download, "filter_csv", wraps=download.filter_csv) as filtering:
            self.assertEqual(self.run_main("--force-filter"), 0)
            self.assertEqual(filtering.call_count, 3)

    def test_filter_only_missing_raw_reports_failure(self):
        (self.raw / "calendar.csv.gz").unlink()
        self.assertEqual(self.run_main("--filter-only"), 1)
        self.assertIn("Thieu raw:", self.output.getvalue())
        self.assertFalse(self.interim.exists())

    def test_failed_download_does_not_filter_stale_raw(self):
        for failure in (download.DownloadError("HTTP 404"), PermissionError("raw inaccessible")):
            with self.subTest(failure=failure), \
                    patch.object(download, "download_file", side_effect=failure), \
                    patch.object(download, "filter_snapshot") as filtering:
                self.assertEqual(self.run_main("--force"), 1)
                filtering.assert_not_called()

    def test_dry_run_writes_nothing_even_with_force(self):
        before = {str(path): download.file_signature(path) for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(self.run_main("--dry-run", "--force"), 0)
        after = {str(path): download.file_signature(path) for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(self.interim.exists())
        self.assertIn("https://data.insideairbnb.com/country/state/city/2024-02-29/data/", self.output.getvalue())

    def test_new_yaml_snapshots_and_single_snapshot_selection(self):
        self.write_config(["2024-02-29", "2024-03-01"])
        self.write_raw(self.raw.parent / "2024-03-01")
        self.assertEqual(self.run_main("--filter-only", "--snapshot", "2024-03-01"), 0)
        self.assertFalse(self.interim.exists())
        self.assertTrue((self.interim.parent / "2024-03-01" / "filter_summary.json").is_file())
        self.write_config(["2024-02-29"])
        self.assertEqual(self.run_main("--filter-only"), 0)
        self.assertTrue((self.interim / "filter_summary.json").is_file())

    def test_download_only_supports_old_yaml_without_assignments(self):
        self.config_path.write_text("city: test\ndata_root: country/city\nsnapshots: [2024-02-29]\n", encoding="utf-8")
        self.assertEqual(self.run_main("--download-only"), 0)
        self.assertFalse(self.interim.exists())

    def test_snapshot_outside_yaml_is_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.run_main("--snapshot", "2024-03-01")
        self.assertEqual(error.exception.code, 2)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="airbnb-http-tests-")
        self.addCleanup(temporary.cleanup)
        self.destination = Path(temporary.name) / "raw" / "listings.csv.gz"
        self.destination.parent.mkdir()
        self.part = self.destination.with_name(self.destination.name + ".part")
        self.payload = gzip.compress(b"id,name\n0001,NA\n")
        self.session = MagicMock()
        self.sleep = patch.object(download.time, "sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)
        redirect = redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def response(self, status, body=b"", **headers):
        response = MagicMock()
        response.status_code = status
        response.headers = headers
        response.__enter__.return_value = response
        response.iter_content.return_value = [body]
        if status >= 400:
            http_response = download.requests.Response()
            http_response.status_code = status
            response.raise_for_status.side_effect = download.requests.HTTPError(str(status), response=http_response)
        return response

    def run_download(self, *, force=False):
        return download.download_file(self.session, "https://example.invalid/data.csv.gz", self.destination, force=force)

    def test_valid_raw_is_skipped(self):
        self.destination.write_bytes(self.payload)
        self.assertFalse(self.run_download())
        self.session.get.assert_not_called()

    def test_timeout_retry_and_force_download(self):
        self.destination.write_bytes(self.payload)
        self.session.get.side_effect = [download.requests.Timeout("timeout"), self.response(200, self.payload)]
        self.assertTrue(self.run_download(force=True))
        self.assertEqual(self.session.get.call_count, 2)
        self.assertEqual(self.session.get.call_args.kwargs["timeout"], download.TIMEOUT)
        self.assertTrue(self.session.get.call_args.kwargs["stream"])
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assertFalse(self.part.exists())

    def test_resume_range(self):
        offset = 9
        self.part.write_bytes(self.payload[:offset])
        self.session.get.return_value = self.response(206, self.payload[offset:], **{
            "Content-Range": f"bytes {offset}-{len(self.payload) - 1}/{len(self.payload)}",
            "Content-Length": str(len(self.payload) - offset)})
        self.assertTrue(self.run_download())
        self.assertEqual(self.session.get.call_args.kwargs["headers"]["Range"], f"bytes={offset}-")
        self.assertEqual(self.destination.read_bytes(), self.payload)

    def test_server_ignoring_range_restarts_safely(self):
        self.part.write_bytes(self.payload[:9])
        self.session.get.return_value = self.response(200, self.payload)
        self.assertTrue(self.run_download())
        self.assertEqual(self.destination.read_bytes(), self.payload)

    def test_complete_part_is_promoted_on_416(self):
        self.part.write_bytes(self.payload)
        self.session.get.return_value = self.response(416)
        self.assertTrue(self.run_download())
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assertFalse(self.part.exists())

    def test_404_fails_without_retry_or_snapshot_substitution(self):
        self.destination.write_bytes(self.payload)
        self.session.get.return_value = self.response(404)
        with self.assertRaisesRegex(download.DownloadError, "HTTP 404"):
            self.run_download(force=True)
        self.assertEqual(self.session.get.call_count, 1)
        self.assertEqual(self.destination.read_bytes(), self.payload)

    def test_invalid_download_is_not_published(self):
        self.session.get.return_value = self.response(200, b"not gzip")
        with self.assertRaises(download.DownloadError):
            self.run_download()
        self.assertEqual(self.session.get.call_count, download.MAX_RETRIES)
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.part.exists())

    def test_gzip_integrity_checks_full_stream(self):
        for payload in (b"", b"not gzip", self.payload[:-8], self.payload[:-1] + b"\xff"):
            with self.subTest(payload=payload):
                self.destination.write_bytes(payload)
                self.assertFalse(download.valid_gzip(self.destination))


if __name__ == "__main__":
    unittest.main()
