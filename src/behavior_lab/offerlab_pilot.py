from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

from behavior_lab.core import parse_time, stable_hash, utc_now
from behavior_lab.ledger import DuplicateRecordError, ImmutableLedger


PILOT_SCHEMA_VERSION = "offerlab_seller_pilot.v1"
PILOT_IMPORT_RECORD_TYPE = "offerlab_seller_pilot_import"
PILOT_ROW_RECORD_TYPE = "offerlab_seller_pilot_row"
DEFAULT_PILOT_DATA_ROOT = Path(r"C:\OfferLabData\seller_pilots")

DATASET_FILENAMES = {
    "listings",
    "offers",
    "orders",
    "fees",
    "shipping_costs",
    "cost_basis",
    "cancellations_unpaid",
    "returns_refunds",
    "inventory",
    "traffic",
}

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "listings": {
        "listing_id",
        "event_time",
        "available_at",
        "asking_price_amount",
        "currency",
        "category",
        "listing_status",
    },
    "offers": {
        "offer_id",
        "listing_id",
        "event_time",
        "available_at",
        "offer_amount",
        "currency",
        "offer_state",
    },
    "orders": {
        "order_id",
        "listing_id",
        "event_time",
        "available_at",
        "sale_price_amount",
        "currency",
        "order_status",
    },
    "fees": {"fee_id", "order_id", "event_time", "available_at", "fee_amount", "currency", "fee_type"},
    "shipping_costs": {
        "shipping_id",
        "order_id",
        "event_time",
        "available_at",
        "shipping_cost_amount",
        "currency",
    },
    "cost_basis": {"cost_basis_id", "listing_id", "event_time", "available_at", "unit_cost_amount", "currency"},
    "cancellations_unpaid": {
        "cancellation_id",
        "event_time",
        "available_at",
        "event_type",
        "currency",
    },
    "returns_refunds": {"return_id", "order_id", "event_time", "available_at", "refund_amount", "currency"},
    "inventory": {"inventory_id", "listing_id", "event_time", "available_at", "quantity_available"},
    "traffic": {"traffic_id", "listing_id", "event_time", "available_at", "impressions", "views"},
}

OPTIONAL_COLUMNS: dict[str, set[str]] = {
    "listings": {"sku", "title", "condition", "listed_at", "ended_at", "quantity_listed"},
    "offers": {
        "buyer_id_hash",
        "seller_response",
        "seller_response_time",
        "seller_response_amount",
        "decision_history_available_at",
        "expires_at",
    },
    "orders": {"offer_id", "paid_at", "completed_at", "return_window_matured_at", "quantity"},
    "fees": {"listing_id"},
    "shipping_costs": {"listing_id", "carrier"},
    "cost_basis": {"sku", "cost_source"},
    "cancellations_unpaid": {"order_id", "listing_id", "offer_id", "amount"},
    "returns_refunds": {
        "listing_id",
        "return_opened_at",
        "return_closed_at",
        "return_window_matured_at",
        "return_status",
    },
    "inventory": {"sku", "inventory_age_days", "warehouse_location"},
    "traffic": {"period_start", "period_end", "watchers"},
}

MONEY_COLUMNS = {
    "asking_price_amount",
    "offer_amount",
    "seller_response_amount",
    "sale_price_amount",
    "fee_amount",
    "shipping_cost_amount",
    "unit_cost_amount",
    "amount",
    "refund_amount",
}
INTEGER_COLUMNS = {"quantity_listed", "quantity", "quantity_available", "impressions", "views", "watchers"}
TIME_COLUMNS = {
    "event_time",
    "available_at",
    "listed_at",
    "ended_at",
    "seller_response_time",
    "decision_history_available_at",
    "expires_at",
    "paid_at",
    "completed_at",
    "return_window_matured_at",
    "return_opened_at",
    "return_closed_at",
    "period_start",
    "period_end",
}


class OfferLabPilotError(ValueError):
    pass


@dataclass(frozen=True)
class PilotImportResult:
    pilot_id: str
    import_id: str
    import_hash: str
    imported_rows: int
    skipped_existing: bool
    ledger: str
    data_root: str


def default_data_root() -> Path:
    return Path(str(Path.cwd().anchor or "C:\\")) / DEFAULT_PILOT_DATA_ROOT.relative_to(DEFAULT_PILOT_DATA_ROOT.anchor)


def write_template(output_dir: str | Path | None = None) -> dict[str, Any]:
    destination = Path(output_dir) if output_dir is not None else default_data_root() / "_templates" / PILOT_SCHEMA_VERSION
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "pilot_id": "replace_with_external_pilot_id",
        "notes": [
            "Copy this directory outside the repository before adding seller data.",
            "Every source column must appear in the columns mapping. Use canonical column names when possible.",
            "Use null/blank for unknown costs; OfferLab never imputes cost basis.",
        ],
        "datasets": {},
    }
    for dataset in sorted(DATASET_FILENAMES):
        columns = _canonical_columns(dataset)
        path = destination / f"{dataset}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
        manifest["datasets"][dataset] = {
            "file": f"{dataset}.csv",
            "columns": {column: column for column in columns},
        }
    manifest_path = destination / "pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "output_dir": str(destination.resolve()),
        "manifest": str(manifest_path.resolve()),
        "datasets": sorted(DATASET_FILENAMES),
        "read_only": True,
    }


def inspect_input(input_dir: str | Path) -> dict[str, Any]:
    source_dir = Path(input_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise OfferLabPilotError(f"INPUT_DIR does not exist or is not a directory: {source_dir}")
    manifest = _load_manifest(source_dir)
    files = _discover_dataset_files(source_dir, manifest)
    datasets: dict[str, Any] = {}
    errors: list[str] = []
    warnings: list[str] = []
    for dataset in sorted(DATASET_FILENAMES):
        file_path = files.get(dataset)
        if file_path is None:
            warnings.append(f"{dataset}: file not found")
            datasets[dataset] = {"present": False}
            continue
        rows, source_columns = _read_rows(file_path)
        mapping = _mapping_for(dataset, source_columns, manifest)
        validation = _validate_mapping(dataset, source_columns, mapping)
        errors.extend(f"{dataset}: {error}" for error in validation["errors"])
        warnings.extend(f"{dataset}: {warning}" for warning in validation["warnings"])
        datasets[dataset] = {
            "present": True,
            "file": str(file_path.resolve()),
            "format": _format_for(file_path),
            "rows": len(rows),
            "source_sha256": _file_sha256(file_path),
            "source_columns": source_columns,
            "column_mapping": mapping,
            "required_columns": sorted(REQUIRED_COLUMNS[dataset]),
            "optional_columns": sorted(OPTIONAL_COLUMNS[dataset]),
            "missing_required_columns": validation["missing_required_columns"],
            "unmapped_source_columns": validation["unmapped_source_columns"],
        }
    input_inside_repo = _path_is_inside_repo(source_dir)
    if input_inside_repo:
        warnings.append("INPUT_DIR is inside the repository; import will refuse seller data from this location")
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "input_dir": str(source_dir.resolve()),
        "manifest": str((source_dir / "pilot_manifest.json").resolve()) if (source_dir / "pilot_manifest.json").exists() else None,
        "input_inside_repository": input_inside_repo,
        "datasets": datasets,
        "errors": errors,
        "warnings": warnings,
        "ready_to_import": not errors,
        "read_only": True,
        "executes_seller_actions": False,
    }


def import_pilot(input_dir: str | Path, *, data_root: str | Path | None = None, pilot_id: str | None = None) -> PilotImportResult:
    source_dir = Path(input_dir)
    if _path_is_inside_repo(source_dir):
        raise OfferLabPilotError("Refusing to import seller data from inside the repository")
    inspection = inspect_input(source_dir)
    if inspection["errors"]:
        raise OfferLabPilotError("Input is not ready to import: " + "; ".join(inspection["errors"]))
    manifest = _load_manifest(source_dir)
    actual_pilot_id = _pilot_id(source_dir, manifest, pilot_id)
    root = Path(data_root) if data_root is not None else default_data_root()
    if _path_is_inside_repo(root):
        raise OfferLabPilotError("Refusing to write seller pilot ledgers inside the repository")
    ledger = ImmutableLedger(root / actual_pilot_id / "ledger.jsonl")
    files = _discover_dataset_files(source_dir, manifest)
    row_payloads: list[dict[str, Any]] = []
    source_files = []
    for dataset, file_path in sorted(files.items()):
        rows, source_columns = _read_rows(file_path)
        mapping = _mapping_for(dataset, source_columns, manifest)
        validation = _validate_mapping(dataset, source_columns, mapping)
        if validation["errors"]:
            raise OfferLabPilotError(f"{dataset} is not ready to import: {'; '.join(validation['errors'])}")
        source_files.append(
            {
                "dataset": dataset,
                "path_name": file_path.name,
                "format": _format_for(file_path),
                "sha256": _file_sha256(file_path),
                "rows": len(rows),
                "source_columns": source_columns,
                "column_mapping": mapping,
            }
        )
        for row_number, row in enumerate(rows, start=1):
            canonical = _canonicalize_row(dataset, row, mapping, row_number=row_number, file_path=file_path)
            row_hash = stable_hash(
                {
                    "schema_version": PILOT_SCHEMA_VERSION,
                    "dataset": dataset,
                    "source_file_sha256": source_files[-1]["sha256"],
                    "source_row_number": row_number,
                    "canonical": canonical,
                }
            )
            row_payloads.append(
                {
                    "schema_version": PILOT_SCHEMA_VERSION,
                    "pilot_id": actual_pilot_id,
                    "dataset": dataset,
                    "source_file_name": file_path.name,
                    "source_file_sha256": source_files[-1]["sha256"],
                    "source_row_number": row_number,
                    "source_column_mapping": mapping,
                    "canonical": canonical,
                    "row_hash": row_hash,
                }
            )
    import_body = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "pilot_id": actual_pilot_id,
        "source_dir_name": source_dir.name,
        "source_files": source_files,
        "row_hashes": [row["row_hash"] for row in row_payloads],
    }
    import_hash = stable_hash(import_body)
    import_id = f"pilot_import_{import_hash[:16]}"
    import_payload = {
        **import_body,
        "import_id": import_id,
        "import_hash": import_hash,
        "imported_at": utc_now(),
        "read_only": True,
        "executes_seller_actions": False,
    }
    existing = ledger.find_record(f"offerlab_pilot_import_{import_hash}", PILOT_IMPORT_RECORD_TYPE)
    if existing is not None:
        ledger.verify_hash_chain()
        return PilotImportResult(
            pilot_id=actual_pilot_id,
            import_id=import_id,
            import_hash=import_hash,
            imported_rows=0,
            skipped_existing=True,
            ledger=str(ledger.path.resolve()),
            data_root=str(root.resolve()),
        )
    entries: list[tuple[str, Any, str | None]] = [
        (PILOT_IMPORT_RECORD_TYPE, import_payload, f"offerlab_pilot_import_{import_hash}")
    ]
    entries.extend(
        (
            PILOT_ROW_RECORD_TYPE,
            {**row, "import_id": import_id, "import_hash": import_hash},
            f"offerlab_pilot_row_{actual_pilot_id}_{import_hash[:12]}_{index:08d}",
        )
        for index, row in enumerate(row_payloads, start=1)
    )
    try:
        ledger.append_many_guarded(entries, unique_record_ids=True)
    except DuplicateRecordError as exc:
        raise OfferLabPilotError(f"Import version collision: {import_hash}") from exc
    ledger.verify_hash_chain()
    return PilotImportResult(
        pilot_id=actual_pilot_id,
        import_id=import_id,
        import_hash=import_hash,
        imported_rows=len(row_payloads),
        skipped_existing=False,
        ledger=str(ledger.path.resolve()),
        data_root=str(root.resolve()),
    )


def audit_pilot(pilot_id: str, *, data_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(data_root) if data_root is not None else default_data_root()
    ledger = ImmutableLedger(root / pilot_id / "ledger.jsonl")
    imports = ledger.payloads(PILOT_IMPORT_RECORD_TYPE)
    if not imports:
        raise OfferLabPilotError(f"No imports found for pilot_id {pilot_id!r}")
    latest_import = imports[-1]
    rows = [
        record
        for record in ledger.payloads(PILOT_ROW_RECORD_TYPE)
        if record.get("pilot_id") == pilot_id and record.get("import_id") == latest_import["import_id"]
    ]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row["canonical"])
    listings = {str(row["listing_id"]): row for row in by_dataset["listings"]}
    offers = by_dataset["offers"]
    orders = by_dataset["orders"]
    fees_by_order = _sum_by(by_dataset["fees"], "order_id", "fee_amount")
    shipping_by_order = _sum_by(by_dataset["shipping_costs"], "order_id", "shipping_cost_amount")
    refunds_by_order = _sum_by(by_dataset["returns_refunds"], "order_id", "refund_amount")
    cost_by_listing = _latest_costs(by_dataset["cost_basis"])
    cancellations = by_dataset["cancellations_unpaid"]
    returns = by_dataset["returns_refunds"]
    inventory_by_listing = {str(row["listing_id"]): row for row in by_dataset["inventory"]}
    orders_by_offer = {str(row.get("offer_id")): row for row in orders if row.get("offer_id")}
    orders_by_listing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for order in orders:
        orders_by_listing[str(order["listing_id"])].append(order)

    offer_rows = []
    response_latencies = []
    for offer in offers:
        linked_order = orders_by_offer.get(str(offer["offer_id"]))
        if linked_order is None:
            listing_orders = orders_by_listing.get(str(offer["listing_id"]), [])
            linked_order = listing_orders[0] if len(listing_orders) == 1 else None
        response_time = offer.get("seller_response_time")
        if response_time:
            response_latencies.append(_hours_between(str(offer["event_time"]), str(response_time)))
        offer_rows.append(
            {
                "offer": offer,
                "accepted": _offer_accepted(offer),
                "order": linked_order,
                "buyer_paid": bool(linked_order and _buyer_paid(linked_order)),
                "order_completed": bool(linked_order and _order_completed(linked_order)),
                "return_window_matured": bool(linked_order and _return_window_matured(linked_order, returns)),
            }
        )

    margin_rows = []
    incomplete_outcomes = []
    missing_cost_basis = []
    missing_fee = []
    missing_shipping = []
    for order in orders:
        listing = listings.get(str(order["listing_id"]))
        cost = cost_by_listing.get(str(order["listing_id"]))
        fee = fees_by_order.get(str(order["order_id"]))
        shipping = shipping_by_order.get(str(order["order_id"]))
        mature = _buyer_paid(order) and _order_completed(order) and _return_window_matured(order, returns)
        if cost is None:
            missing_cost_basis.append(str(order["listing_id"]))
        if fee is None:
            missing_fee.append(str(order["order_id"]))
        if shipping is None:
            missing_shipping.append(str(order["order_id"]))
        if not mature:
            incomplete_outcomes.append(str(order["order_id"]))
            continue
        if cost is None or fee is None:
            continue
        sale_price = _float_or_none(order.get("sale_price_amount"))
        if sale_price is None:
            continue
        quantity = int(order.get("quantity") or 1)
        refund = refunds_by_order.get(str(order["order_id"]), 0.0)
        ship = shipping or 0.0
        margin = sale_price - fee - ship - refund - (cost * quantity)
        asking = _float_or_none(listing.get("asking_price_amount")) if listing else None
        margin_rows.append(
            {
                "order_id": order["order_id"],
                "listing_id": order["listing_id"],
                "category": listing.get("category") if listing else "unknown",
                "inventory_age_bucket": _inventory_age_bucket(inventory_by_listing.get(str(order["listing_id"]))),
                "sale_price": sale_price,
                "asking_price": asking,
                "realized_to_asking_ratio": round(sale_price / asking, 4) if asking else None,
                "mature_contribution_margin": round(margin, 2),
            }
        )

    accepted = sum(1 for item in offer_rows if item["accepted"])
    paid = sum(1 for item in offer_rows if item["buyer_paid"])
    completed = sum(1 for item in offer_rows if item["order_completed"])
    return_matured = sum(1 for item in offer_rows if item["return_window_matured"])
    decision_history_known = sum(1 for offer in offers if offer.get("seller_response") and offer.get("seller_response_time"))
    paid_orders = [order for order in orders if _buyer_paid(order)]
    completed_orders = [order for order in orders if _order_completed(order)]
    readiness = _readiness_gate(
        mature_margin_count=len(margin_rows),
        cost_coverage=_coverage(len(orders) - len(set(missing_cost_basis)), len(orders)),
        fee_coverage=_coverage(len(orders) - len(set(missing_fee)), len(orders)),
        decision_history_coverage=_coverage(decision_history_known, len(offers)),
        return_window_coverage=_coverage(sum(1 for order in completed_orders if _return_window_matured(order, returns)), len(completed_orders)),
    )
    ledger.verify_hash_chain()
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "pilot_id": pilot_id,
        "ledger": str(ledger.path.resolve()),
        "ledger_valid": True,
        "latest_import_id": latest_import["import_id"],
        "latest_import_hash": latest_import["import_hash"],
        "read_only": True,
        "executes_seller_actions": False,
        "counts": {dataset: len(by_dataset.get(dataset, [])) for dataset in sorted(DATASET_FILENAMES)},
        "offer_funnel": {
            "offers_received": len(offers),
            "seller_accepted": accepted,
            "buyer_paid": paid,
            "order_completed": completed,
            "return_window_matured": return_matured,
            "mature_margin_count": len(margin_rows),
        },
        "rates": {
            "acceptance_rate": _coverage(accepted, len(offers)),
            "payment_rate_after_acceptance": _coverage(paid, accepted),
            "completion_rate_after_payment": _coverage(len([order for order in paid_orders if _order_completed(order)]), len(paid_orders)),
        },
        "response_latency_hours": {
            "count": len(response_latencies),
            "average": round(mean(response_latencies), 4) if response_latencies else None,
            "median": round(median(response_latencies), 4) if response_latencies else None,
        },
        "realized_price_vs_asking": {
            "orders_with_ratio": len([row for row in margin_rows if row["realized_to_asking_ratio"] is not None]),
            "average_ratio": _average([row["realized_to_asking_ratio"] for row in margin_rows]),
        },
        "mature_contribution_margin": {
            "orders": len(margin_rows),
            "total": round(sum(row["mature_contribution_margin"] for row in margin_rows), 2),
            "average": _average([row["mature_contribution_margin"] for row in margin_rows]),
        },
        "cancellation_return_effects": {
            "cancellations": sum(1 for row in cancellations if str(row.get("event_type")).lower() == "cancellation"),
            "unpaid_orders": sum(1 for row in cancellations if str(row.get("event_type")).lower() == "unpaid_order"),
            "returns_or_refunds": len(returns),
            "total_refunds": round(sum(_float_or_none(row.get("refund_amount")) or 0.0 for row in returns), 2),
        },
        "breakdowns": {
            "by_category": _margin_breakdown(margin_rows, "category"),
            "by_inventory_age": _margin_breakdown(margin_rows, "inventory_age_bucket"),
        },
        "data_quality_gaps": {
            "missing_cost_basis_listing_ids": sorted(set(missing_cost_basis))[:25],
            "orders_missing_actual_fees": sorted(set(missing_fee))[:25],
            "orders_missing_shipping_costs": sorted(set(missing_shipping))[:25],
            "incomplete_outcome_order_ids": sorted(set(incomplete_outcomes))[:25],
            "unmapped_source_columns": [],
            "never_imputed_costs": True,
        },
        "readiness_gate": readiness,
        "shadow_evaluation_possible": readiness["passed"],
    }


def _canonical_columns(dataset: str) -> list[str]:
    return sorted(REQUIRED_COLUMNS[dataset] | OPTIONAL_COLUMNS[dataset])


def _load_manifest(source_dir: Path) -> dict[str, Any]:
    path = source_dir / "pilot_manifest.json"
    if not path.exists():
        return {"schema_version": PILOT_SCHEMA_VERSION, "datasets": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OfferLabPilotError("pilot_manifest.json must contain a JSON object")
    if payload.get("schema_version") != PILOT_SCHEMA_VERSION:
        raise OfferLabPilotError(f"pilot_manifest.json schema_version must be {PILOT_SCHEMA_VERSION}")
    if not isinstance(payload.get("datasets"), dict):
        raise OfferLabPilotError("pilot_manifest.json datasets must be an object")
    return payload


def _discover_dataset_files(source_dir: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    datasets = manifest.get("datasets", {})
    for dataset in DATASET_FILENAMES:
        entry = datasets.get(dataset)
        if isinstance(entry, dict) and entry.get("file"):
            path = source_dir / str(entry["file"])
            if path.exists():
                files[dataset] = path
            continue
        matches = [source_dir / f"{dataset}{suffix}" for suffix in [".csv", ".json", ".jsonl", ".parquet"]]
        for match in matches:
            if match.exists():
                files[dataset] = match
                break
    return files


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    fmt = _format_for(path)
    if fmt == "csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
            return rows, list(reader.fieldnames or [])
    if fmt == "jsonl":
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise OfferLabPilotError(f"{path.name} line {line_number} must be a JSON object")
            rows.append(payload)
        return rows, _union_columns(rows)
    if fmt == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            payload = payload["rows"]
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise OfferLabPilotError(f"{path.name} must contain a JSON array of objects or an object with rows")
        rows = [dict(row) for row in payload]
        return rows, _union_columns(rows)
    if fmt == "parquet":
        return _read_parquet(path)
    raise OfferLabPilotError(f"Unsupported file type: {path.suffix}")


def _read_parquet(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise OfferLabPilotError("Parquet input requires pyarrow or pandas to be installed") from exc
        frame = pd.read_parquet(path)
        rows = frame.to_dict(orient="records")
        return rows, [str(column) for column in frame.columns]
    table = pq.read_table(path)
    rows = [dict(row) for row in table.to_pylist()]
    return rows, [str(column) for column in table.column_names]


def _format_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".parquet":
        return "parquet"
    return suffix.lstrip(".")


def _union_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen = set()
    for row in rows:
        for column in row:
            if column not in seen:
                seen.add(column)
                columns.append(str(column))
    return columns


def _mapping_for(dataset: str, source_columns: list[str], manifest: dict[str, Any]) -> dict[str, str]:
    entry = manifest.get("datasets", {}).get(dataset, {})
    columns = entry.get("columns") if isinstance(entry, dict) else None
    if columns is not None:
        if not isinstance(columns, dict):
            raise OfferLabPilotError(f"{dataset} manifest columns must be an object")
        return {str(source): str(target) for source, target in columns.items()}
    canonical = set(_canonical_columns(dataset))
    return {column: column for column in source_columns if column in canonical}


def _validate_mapping(dataset: str, source_columns: list[str], mapping: dict[str, str]) -> dict[str, Any]:
    canonical = set(_canonical_columns(dataset))
    source_set = set(source_columns)
    mapped_sources = set(mapping)
    unmapped = sorted(source_set - mapped_sources)
    unknown_sources = sorted(mapped_sources - source_set)
    invalid_targets = sorted({target for target in mapping.values() if target != "ignore" and target not in canonical})
    mapped_targets = {target for source, target in mapping.items() if source in source_set and target != "ignore"}
    missing_required = sorted(REQUIRED_COLUMNS[dataset] - mapped_targets)
    errors = []
    if unmapped:
        errors.append(f"unmapped source columns: {unmapped}")
    if invalid_targets:
        errors.append(f"invalid canonical targets: {invalid_targets}")
    if missing_required:
        errors.append(f"missing required canonical columns: {missing_required}")
    warnings = []
    if unknown_sources:
        warnings.append(f"manifest maps columns not present in source: {unknown_sources}")
    return {
        "errors": errors,
        "warnings": warnings,
        "missing_required_columns": missing_required,
        "unmapped_source_columns": unmapped,
    }


def _canonicalize_row(dataset: str, row: dict[str, Any], mapping: dict[str, str], *, row_number: int, file_path: Path) -> dict[str, Any]:
    canonical = {column: None for column in _canonical_columns(dataset)}
    for source, target in mapping.items():
        if target == "ignore" or source not in row:
            continue
        canonical[target] = _clean_value(row[source])
    for required in REQUIRED_COLUMNS[dataset]:
        if canonical.get(required) in {None, ""}:
            raise OfferLabPilotError(f"{file_path.name} row {row_number}: missing required {required}")
    _validate_times(dataset, canonical, row_number=row_number, file_path=file_path)
    _validate_money(dataset, canonical, row_number=row_number, file_path=file_path)
    _validate_integer_columns(canonical, row_number=row_number, file_path=file_path)
    return canonical


def _clean_value(value: Any) -> Any:
    if value == "":
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _validate_times(dataset: str, row: dict[str, Any], *, row_number: int, file_path: Path) -> None:
    for column in TIME_COLUMNS:
        if row.get(column) in {None, ""}:
            continue
        try:
            parse_time(str(row[column]))
        except ValueError as exc:
            raise OfferLabPilotError(f"{file_path.name} row {row_number}: {column} must be timezone-aware ISO-8601") from exc
    event = parse_time(str(row["event_time"]))
    available = parse_time(str(row["available_at"]))
    if available < event:
        raise OfferLabPilotError(f"{file_path.name} row {row_number}: available_at may not be before event_time")
    if dataset == "offers" and row.get("seller_response_time"):
        response = parse_time(str(row["seller_response_time"]))
        if response < event:
            raise OfferLabPilotError(f"{file_path.name} row {row_number}: seller_response_time may not be before event_time")


def _validate_money(dataset: str, row: dict[str, Any], *, row_number: int, file_path: Path) -> None:
    currency = row.get("currency")
    has_money = any(row.get(column) not in {None, ""} for column in MONEY_COLUMNS & set(row))
    if "currency" in REQUIRED_COLUMNS[dataset] or currency not in {None, ""} or has_money:
        if not isinstance(currency, str) or len(currency) != 3 or currency.upper() != currency or not currency.isalpha():
            raise OfferLabPilotError(f"{file_path.name} row {row_number}: currency must be a 3-letter uppercase ISO code")
    for column in MONEY_COLUMNS & set(row):
        value = row.get(column)
        if value in {None, ""}:
            continue
        number = _float_or_none(value)
        if number is None or number < 0:
            raise OfferLabPilotError(f"{file_path.name} row {row_number}: {column} must be a non-negative amount")


def _validate_integer_columns(row: dict[str, Any], *, row_number: int, file_path: Path) -> None:
    for column in INTEGER_COLUMNS & set(row):
        value = row.get(column)
        if value in {None, ""}:
            continue
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise OfferLabPilotError(f"{file_path.name} row {row_number}: {column} must be an integer") from exc
        if number < 0:
            raise OfferLabPilotError(f"{file_path.name} row {row_number}: {column} may not be negative")
        row[column] = number
    if "inventory_age_days" in row and row.get("inventory_age_days") not in {None, ""}:
        age = _float_or_none(row.get("inventory_age_days"))
        if age is None or age < 0:
            raise OfferLabPilotError(f"{file_path.name} row {row_number}: inventory_age_days may not be negative")


def _pilot_id(source_dir: Path, manifest: dict[str, Any], explicit: str | None) -> str:
    value = explicit or manifest.get("pilot_id")
    if value and value != "replace_with_external_pilot_id":
        text = str(value)
    else:
        text = f"pilot_{stable_hash({'source_dir_name': source_dir.name})[:12]}"
    if not text.replace("_", "").replace("-", "").isalnum():
        raise OfferLabPilotError("pilot_id may contain only letters, numbers, underscores, and hyphens")
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_is_inside_repo(path: Path) -> bool:
    repo_roots = [root for root in {_repo_root(Path.cwd()), _repo_root(Path(__file__).resolve())} if root is not None]
    for repo in repo_roots:
        try:
            path.resolve().relative_to(repo.resolve())
        except ValueError:
            continue
        return True
    return False


def _repo_root(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _sum_by(rows: list[dict[str, Any]], key: str, amount_key: str) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        row_key = row.get(key)
        amount = _float_or_none(row.get(amount_key))
        if row_key and amount is not None:
            totals[str(row_key)] += amount
    return dict(totals)


def _latest_costs(rows: list[dict[str, Any]]) -> dict[str, float]:
    costs: dict[str, tuple[datetime, float]] = {}
    for row in rows:
        listing_id = row.get("listing_id")
        cost = _float_or_none(row.get("unit_cost_amount"))
        if not listing_id or cost is None:
            continue
        available = parse_time(str(row["available_at"]))
        existing = costs.get(str(listing_id))
        if existing is None or available >= existing[0]:
            costs[str(listing_id)] = (available, cost)
    return {listing_id: cost for listing_id, (_, cost) in costs.items()}


def _offer_accepted(offer: dict[str, Any]) -> bool:
    values = {str(offer.get("offer_state", "")).lower(), str(offer.get("seller_response", "")).lower()}
    return bool(values & {"accepted", "accept", "seller_accepted"})


def _buyer_paid(order: dict[str, Any]) -> bool:
    status = str(order.get("order_status", "")).lower()
    return bool(order.get("paid_at")) or status in {"paid", "completed", "shipped", "delivered"}


def _order_completed(order: dict[str, Any]) -> bool:
    status = str(order.get("order_status", "")).lower()
    return bool(order.get("completed_at")) or status in {"completed", "delivered"}


def _return_window_matured(order: dict[str, Any], returns: list[dict[str, Any]]) -> bool:
    maturity = order.get("return_window_matured_at")
    if maturity and parse_time(str(maturity)) <= datetime.now(timezone.utc):
        return True
    order_id = str(order.get("order_id"))
    for row in returns:
        if str(row.get("order_id")) == order_id and row.get("return_window_matured_at"):
            if parse_time(str(row["return_window_matured_at"])) <= datetime.now(timezone.utc):
                return True
    return False


def _hours_between(start: str, end: str) -> float:
    return round((parse_time(end) - parse_time(start)).total_seconds() / 3600.0, 4)


def _readiness_gate(
    *,
    mature_margin_count: int,
    cost_coverage: float,
    fee_coverage: float,
    decision_history_coverage: float,
    return_window_coverage: float,
) -> dict[str, Any]:
    thresholds = {
        "minimum_mature_margin_outcomes": 30,
        "minimum_cost_coverage": 0.95,
        "minimum_fee_coverage": 0.95,
        "minimum_decision_history_coverage": 0.8,
        "minimum_return_window_coverage": 0.8,
    }
    checks = {
        "sufficient_mature_outcomes": mature_margin_count >= thresholds["minimum_mature_margin_outcomes"],
        "cost_coverage": cost_coverage >= thresholds["minimum_cost_coverage"],
        "fee_coverage": fee_coverage >= thresholds["minimum_fee_coverage"],
        "decision_history_coverage": decision_history_coverage >= thresholds["minimum_decision_history_coverage"],
        "return_window_coverage": return_window_coverage >= thresholds["minimum_return_window_coverage"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "mature_margin_outcomes": mature_margin_count,
            "cost_coverage": cost_coverage,
            "fee_coverage": fee_coverage,
            "decision_history_coverage": decision_history_coverage,
            "return_window_coverage": return_window_coverage,
        },
        "thresholds": thresholds,
    }


def _margin_breakdown(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    output = []
    for bucket, items in sorted(grouped.items()):
        output.append(
            {
                "bucket": bucket,
                "orders": len(items),
                "average_mature_contribution_margin": _average(
                    [row["mature_contribution_margin"] for row in items]
                ),
                "total_mature_contribution_margin": round(
                    sum(row["mature_contribution_margin"] for row in items), 2
                ),
            }
        )
    return output


def _inventory_age_bucket(row: dict[str, Any] | None) -> str:
    if not row or row.get("inventory_age_days") in {None, ""}:
        return "unknown"
    age = float(row["inventory_age_days"])
    if age < 30:
        return "0_to_29_days"
    if age < 90:
        return "30_to_89_days"
    if age < 180:
        return "90_to_179_days"
    return "180_plus_days"


def _average(values: list[float | None]) -> float | None:
    concrete = [float(value) for value in values if value is not None]
    if not concrete:
        return None
    return round(sum(concrete) / len(concrete), 4)


def _coverage(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
