from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET


REAL_TRANSFORMATION_VERSION = "nber_best_offer_real_normalization.v1"

REAL_LISTING_COLUMNS = [
    "anon_item_id",
    "anon_title_code",
    "anon_product_id",
    "anon_leaf_categ_id",
    "anon_slr_id",
    "anon_buyer_id",
    "auct_start_dt",
    "fdbk_score_start",
    "fdbk_pstv_start",
    "auct_end_dt",
    "start_price_usd",
    "photo_count",
    "to_lst_cnt",
    "bo_lst_cnt",
    "count1",
    "ref_price1",
    "count2",
    "ref_price2",
    "count3",
    "ref_price3",
    "item_cndtn_id",
    "view_item_count",
    "wtchr_count",
    "meta_categ_id",
    "item_price",
    "bo_ck_yn",
    "ship_time_slowest",
    "ship_time_fastest",
    "ship_time_chosen",
    "decline_price",
    "accept_price",
    "bin_rev",
    "lstg_gen_type_id",
    "store",
    "ref_price4",
    "count4",
    "slr_us",
    "buyer_us",
]

REAL_THREAD_COLUMNS = [
    "anon_item_id",
    "anon_thread_id",
    "anon_byr_id",
    "anon_slr_id",
    "src_cre_dt",
    "fdbk_score_src",
    "fdbk_pstv_src",
    "offr_type_id",
    "status_id",
    "offr_price",
    "src_cre_date",
    "response_time",
    "slr_hist",
    "byr_hist",
    "any_mssg",
    "byr_us",
]

OFFER_TYPE_MAP = {
    "0": {"actor": "buyer", "action": "offer"},
    "1": {"actor": "buyer", "action": "counter"},
    "2": {"actor": "seller", "action": "counter"},
}

STATUS_MAP = {
    "0": "expired",
    "1": "accepted",
    "2": "declined",
    "6": "auto_declined",
    "7": "countered",
    "8": "declined_other_buyer_accepted",
    "9": "auto_accepted",
}


class NberSchemaError(ValueError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_mapping_path() -> Path:
    return repo_root() / "datasets" / "manifests" / "nber_best_offer_real_mapping.yaml"


def load_real_mapping(path: str | Path | None = None) -> dict[str, Any]:
    mapping_path = Path(path) if path is not None else default_mapping_path()
    text = mapping_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _load_yaml_mapping_fallback(text)


def mapping_hash(path: str | Path | None = None) -> str:
    mapping_path = Path(path) if path is not None else default_mapping_path()
    return sha256_file(mapping_path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_real_headers(*, listings: list[str] | None = None, threads: list[str] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {"valid": True, "files": {}}
    if listings is not None:
        report["files"]["anon_bo_lists.csv"] = _compare_header(listings, REAL_LISTING_COLUMNS)
    if threads is not None:
        report["files"]["anon_bo_threads.csv"] = _compare_header(threads, REAL_THREAD_COLUMNS)
    report["valid"] = all(item["valid"] for item in report["files"].values())
    return report


def read_csv_header(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def inspect_schema(*, codebook_path: str | Path | None = None, mapping_path: str | Path | None = None) -> dict[str, Any]:
    mapping = load_real_mapping(mapping_path)
    report: dict[str, Any] = {
        "schema_version": mapping["schema_version"],
        "mapping_hash": mapping_hash(mapping_path),
        "expected_headers": {
            "anon_bo_lists.csv": REAL_LISTING_COLUMNS,
            "anon_bo_threads.csv": REAL_THREAD_COLUMNS,
        },
        "unresolved_semantics": mapping.get("unresolved_semantics", []),
    }
    if codebook_path is not None:
        sheets = read_codebook_sheets(codebook_path)
        report["codebook"] = {
            "path": str(Path(codebook_path).resolve()),
            "sha256": sha256_file(codebook_path),
            "sheets": {name: {"rows": len(rows), "columns": max((len(row) for row in rows), default=0)} for name, rows in sheets.items()},
        }
    return report


def _load_yaml_mapping_fallback(text: str) -> dict[str, Any]:
    """Read the committed schema manifest without requiring PyYAML.

    The repository permits normal YAML for human-maintained research manifests,
    but the package should still import in a minimal environment. This fallback
    extracts the fields needed by code paths and tests; the full manifest remains
    the source of record for human review.
    """

    schema_version_match = re.search(r"^schema_version:\s*(.+)$", text, flags=re.MULTILINE)
    return {
        "schema_version": schema_version_match.group(1).strip() if schema_version_match else "unknown",
        "files": {
            "anon_bo_lists.csv": {"header": _extract_inline_columns(text, "anon_bo_lists.csv") or REAL_LISTING_COLUMNS},
            "anon_bo_threads.csv": {"header": _extract_inline_columns(text, "fixture_columns") or REAL_THREAD_COLUMNS},
        },
        "code_maps": {
            "offr_type_id": OFFER_TYPE_MAP,
            "status_id": {key: {"status": value} for key, value in STATUS_MAP.items()},
        },
        "unresolved_semantics": _extract_unresolved_lines(text),
    }


def _extract_inline_columns(text: str, key: str) -> list[str] | None:
    pattern = rf"{re.escape(key)}[^\n]*\n(?:[^\n]*\n)*?\s+(?:columns|fixture_columns):\s*\[([^\]]+)\]"
    match = re.search(pattern, text)
    if not match and key == "fixture_columns":
        match = re.search(r"fixture_columns:\s*\[([^\]]+)\]", text)
    if not match:
        return None
    return [item.strip().strip('"').strip("'") for item in match.group(1).split(",") if item.strip()]


def _extract_unresolved_lines(text: str) -> list[str]:
    if "unresolved:" not in text:
        return []
    lines = []
    capture = False
    for line in text.splitlines():
        if line.strip() == "unresolved:":
            capture = True
            continue
        if capture:
            stripped = line.strip()
            if not stripped.startswith("- "):
                break
            lines.append(stripped[2:].strip('"'))
    return lines


def read_codebook_sheets(path: str | Path) -> dict[str, list[list[str]]]:
    workbook = Path(path)
    namespaces = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with ZipFile(workbook) as archive:
        shared_strings = _read_shared_strings(archive, namespaces)
        workbook_xml = ET.fromstring(archive.read("xl/workbook.xml"))
        rels_xml = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rels = {item.attrib["Id"]: item.attrib["Target"] for item in rels_xml}
        sheets: dict[str, list[list[str]]] = {}
        for sheet in workbook_xml.findall("main:sheets/main:sheet", namespaces):
            rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = "xl/" + rels[rel_id].lstrip("/")
            sheets[sheet.attrib["name"]] = _read_sheet_rows(archive, target, shared_strings, namespaces)
        return sheets


def _compare_header(actual: list[str], expected: list[str]) -> dict[str, Any]:
    return {
        "valid": actual == expected,
        "actual_count": len(actual),
        "expected_count": len(expected),
        "missing": [column for column in expected if column not in actual],
        "unexpected": [column for column in actual if column not in expected],
        "order_mismatch": actual != expected and sorted(actual) == sorted(expected),
    }


def _read_shared_strings(archive: ZipFile, namespaces: dict[str, str]) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall("main:si", namespaces):
        strings.append("".join(text.text or "" for text in item.findall(".//main:t", namespaces)))
    return strings


def _read_sheet_rows(archive: ZipFile, path: str, strings: list[str], namespaces: dict[str, str]) -> list[list[str]]:
    root = ET.fromstring(archive.read(path))
    rows = []
    for row in root.findall("main:sheetData/main:row", namespaces):
        values: list[str] = []
        for cell in row.findall("main:c", namespaces):
            index = _column_index(cell.attrib.get("r", "A1"))
            while len(values) <= index:
                values.append("")
            raw = cell.find("main:v", namespaces)
            value = ""
            if cell.attrib.get("t") == "inlineStr":
                value = "".join(text.text or "" for text in cell.findall(".//main:t", namespaces))
            elif raw is not None and raw.text is not None:
                value = strings[int(raw.text)] if cell.attrib.get("t") == "s" else raw.text
            values[index] = value
        rows.append(values)
    return rows


def _column_index(reference: str) -> int:
    letters = "".join(char for char in reference if char.isalpha())
    value = 0
    for letter in letters.upper():
        value = value * 26 + (ord(letter) - 64)
    return value - 1
