from __future__ import annotations

import _bootstrap  # noqa: F401

import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from behavior_lab.datasets.nber_best_offer.source_inventory import SourceInventoryError, public_summary, run_source_inventory
from behavior_lab.datasets.nber_best_offer.source_schema import read_codebook_sheets


class NberSourceInventoryTests(unittest.TestCase):
    def test_inventory_keeps_raw_rows_out_of_committed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            official = root / "official"
            raw = root / "raw"
            manifest_path = root / "repo" / "datasets" / "manifests" / "nber_best_offer_downloads.yaml"
            doc_path = root / "repo" / "docs" / "runs" / "NBER_SOURCE_INVENTORY.md"
            official.mkdir()

            self._write_gzip_csv(
                official / "anon_bo_lists.csv.gz",
                [
                    ["listing_id", "seller_id", "start_time", "price"],
                    ["listing-1", "seller-secret-1", "2012-05-01T00:00:00", "100"],
                    ["listing-2", "seller-secret-2", "2012-05-02T00:00:00", "120"],
                    ["listing-3", "seller-secret-3", "2012-05-03T00:00:00", "140"],
                    ["malformed", "too-short"],
                ],
            )
            self._write_gzip_csv(
                official / "anon_bo_threads.csv.gz",
                [
                    ["thread_id", "buyer_id", "seller_id", "event_time", "amount"],
                    ["thread-1", "buyer-secret-1", "seller-secret-1", "2012-05-01T01:00:00", "90"],
                    ["thread-2", "buyer-secret-2", "seller-secret-2", "2012-05-02T01:00:00", "95"],
                    ["thread-3", "buyer-secret-3", "seller-secret-3", "2012-05-03T01:00:00", "105"],
                ],
            )
            self._write_minimal_xlsx(official / "Codebook.xlsx")

            sources = [
                {
                    "logical_name": "anon_bo_lists",
                    "filename": "anon_bo_lists.csv.gz",
                    "url": (official / "anon_bo_lists.csv.gz").as_uri(),
                    "kind": "csv_gzip",
                },
                {
                    "logical_name": "anon_bo_threads",
                    "filename": "anon_bo_threads.csv.gz",
                    "url": (official / "anon_bo_threads.csv.gz").as_uri(),
                    "kind": "csv_gzip",
                },
                {
                    "logical_name": "codebook",
                    "filename": "Codebook.xlsx",
                    "url": (official / "Codebook.xlsx").as_uri(),
                    "kind": "xlsx",
                },
            ]

            manifest = run_source_inventory(
                raw_dir=raw,
                manifest_path=manifest_path,
                doc_path=doc_path,
                official_sources=sources,
                first_sample_rows=2,
                reservoir_rows=2,
                chronological_rows_per_slice=1,
                timeout_seconds=10,
                download=True,
            )

            list_file = next(item for item in manifest["files"] if item["logical_name"] == "anon_bo_lists")
            self.assertEqual(list_file["rows"]["accepted"], 3)
            self.assertEqual(list_file["rows"]["rejected"], 1)
            self.assertTrue(list_file["gzip_integrity"]["valid"])
            self.assertTrue(Path(list_file["sample_files"]["first_rows"]["path"]).exists())
            self.assertIn("chronological", list_file["sample_files"])

            committed_text = manifest_path.read_text(encoding="utf-8") + "\n" + doc_path.read_text(encoding="utf-8")
            self.assertNotIn("seller-secret-1", committed_text)
            self.assertNotIn("buyer-secret-1", committed_text)
            self.assertIn("seller_id", committed_text)

            external_inventory = json.loads(Path(manifest["external_inventory_path"]).read_text(encoding="utf-8"))
            private_lists = next(item for item in external_inventory["files"] if item["logical_name"] == "anon_bo_lists")
            self.assertEqual(private_lists["first_10_valid_rows"][0]["seller_id"], "seller-secret-1")

            summary = json.dumps(public_summary(manifest), sort_keys=True)
            self.assertNotIn("seller-secret-1", summary)
            self.assertNotIn("buyer-secret-1", summary)
            self.assertFalse(json.loads(summary)["raw_rows_printed"])
            sheets = read_codebook_sheets(official / "Codebook.xlsx")
            self.assertEqual(sheets["Codebook"][0][:2], ["variable", "description"])

    def test_report_inventory_requires_explicit_download_when_files_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            official = root / "official"
            raw = root / "raw"
            official.mkdir()
            self._write_gzip_csv(official / "anon_bo_lists.csv.gz", [["a"], ["b"]])
            sources = [
                {
                    "logical_name": "anon_bo_lists",
                    "filename": "anon_bo_lists.csv.gz",
                    "url": (official / "anon_bo_lists.csv.gz").as_uri(),
                    "kind": "csv_gzip",
                }
            ]
            with self.assertRaises(SourceInventoryError):
                run_source_inventory(
                    raw_dir=raw,
                    manifest_path=root / "repo" / "manifest.yaml",
                    doc_path=root / "repo" / "inventory.md",
                    official_sources=sources,
                    write_outputs=False,
                )
            self.assertFalse((raw / "anon_bo_lists.csv.gz").exists())

    def _write_gzip_csv(self, path: Path, rows: list[list[str]]) -> None:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)

    def _write_minimal_xlsx(self, path: Path) -> None:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "[Content_Types].xml",
                """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>""",
            )
            archive.writestr(
                "xl/workbook.xml",
                """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Codebook" sheetId="1" r:id="rId1"/></sheets>
</workbook>""",
            )
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>""",
            )
            archive.writestr(
                "xl/worksheets/sheet1.xml",
                """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1">
      <c r="A1" t="inlineStr"><is><t>variable</t></is></c>
      <c r="B1" t="inlineStr"><is><t>description</t></is></c>
    </row>
    <row r="2">
      <c r="A2" t="inlineStr"><is><t>seller_id</t></is></c>
      <c r="B2" t="inlineStr"><is><t>synthetic definition</t></is></c>
    </row>
  </sheetData>
</worksheet>""",
            )


if __name__ == "__main__":
    unittest.main()
