from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from behavior_lab.datasets.nber_best_offer.normalize import read_jsonl
from behavior_lab.datasets.nber_best_offer.schema import FORBIDDEN_FUTURE_FIELDS


class NberTaskError(ValueError):
    pass


def build_tasks(normalized_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    root = Path(normalized_dir)
    if (root / "manifest.json").exists() and (root / "tables").exists():
        return build_real_tasks_from_records(
            _read_partitioned_table(root, "listings"),
            _read_partitioned_table(root, "negotiation_turns"),
        )
    listings = {str(row["listing_id"]): row for row in read_jsonl(root / "listings.jsonl")}
    turns = sorted(read_jsonl(root / "negotiation_turns.jsonl"), key=lambda row: (str(row["thread_id"]), int(row["turn_index"])))
    threads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for turn in turns:
        threads[str(turn["thread_id"])].append(turn)
    return {
        "seller_next_action": seller_next_action(listings, threads),
        "buyer_response_to_counter": buyer_response_to_counter(listings, threads),
        "agreement": agreement_task(listings, threads),
        "final_price_ratio": final_price_ratio_task(listings, threads),
        "response_latency": response_latency_task(listings, threads),
    }


def build_real_tasks_from_records(listing_rows: list[dict[str, Any]], turn_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    listings = {str(row["listing_id"]): row for row in listing_rows}
    turns = sorted(turn_rows, key=lambda row: (str(row["thread_id"]), int(row["turn_index"])))
    threads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for turn in turns:
        threads[str(turn["thread_id"])].append(turn)
    return {
        "seller_next_action": seller_next_action_real(listings, threads),
        "buyer_response_to_counter": buyer_response_to_counter_real(listings, threads),
        "agreement": agreement_task(listings, threads),
        "final_price_ratio": final_price_ratio_task(listings, threads),
        "response_latency": response_latency_task(listings, threads),
    }


def seller_next_action(listings: dict[str, dict[str, Any]], threads: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for thread_id, turns in threads.items():
        for index, turn in enumerate(turns[:-1]):
            next_turn = turns[index + 1]
            if turn["actor"] != "buyer" or next_turn["actor"] != "seller":
                continue
            listing = listings[str(turn["listing_id"])]
            rows.append(
                _snapshot(
                    task="seller_next_action",
                    label=_seller_label(next_turn),
                    listing=listing,
                    turn=turn,
                    history=turns[: index + 1],
                    row_id=f"{thread_id}:{turn['turn_index']}:seller_next_action",
                )
            )
    return rows


def seller_next_action_real(listings: dict[str, dict[str, Any]], threads: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for thread_id, turns in threads.items():
        for index, turn in enumerate(turns):
            if turn["actor"] != "buyer" or turn["action"] not in {"offer", "counter"}:
                continue
            next_turn = turns[index + 1] if index + 1 < len(turns) else None
            label = _real_status_label(turn, counter_actor="seller", next_turn=next_turn)
            if label is None:
                continue
            listing = listings[str(turn["listing_id"])]
            rows.append(
                _snapshot(
                    task="seller_next_action",
                    label=label,
                    listing=listing,
                    turn=turn,
                    history=turns[: index + 1],
                    row_id=f"{thread_id}:{turn['turn_index']}:seller_next_action",
                )
            )
    return rows


def buyer_response_to_counter(listings: dict[str, dict[str, Any]], threads: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for thread_id, turns in threads.items():
        for index, turn in enumerate(turns[:-1]):
            next_turn = turns[index + 1]
            if turn["actor"] != "seller" or turn["action"] != "counter" or next_turn["actor"] != "buyer":
                continue
            listing = listings[str(turn["listing_id"])]
            rows.append(
                _snapshot(
                    task="buyer_response_to_counter",
                    label=_buyer_label(next_turn),
                    listing=listing,
                    turn=turn,
                    history=turns[: index + 1],
                    row_id=f"{thread_id}:{turn['turn_index']}:buyer_response",
                )
            )
    return rows


def buyer_response_to_counter_real(listings: dict[str, dict[str, Any]], threads: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for thread_id, turns in threads.items():
        for index, turn in enumerate(turns):
            if turn["actor"] != "seller" or turn["action"] != "counter":
                continue
            next_turn = turns[index + 1] if index + 1 < len(turns) else None
            label = _real_status_label(turn, counter_actor="buyer", next_turn=next_turn)
            if label is None:
                continue
            listing = listings[str(turn["listing_id"])]
            rows.append(
                _snapshot(
                    task="buyer_response_to_counter",
                    label=label,
                    listing=listing,
                    turn=turn,
                    history=turns[: index + 1],
                    row_id=f"{thread_id}:{turn['turn_index']}:buyer_response",
                )
            )
    return rows


def agreement_task(listings: dict[str, dict[str, Any]], threads: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for thread_id, turns in threads.items():
        label = agreement_label(turns)
        # A truncated observation window is not evidence of failed agreement.
        # Censored threads remain available to dataset audits but do not become
        # supervised negative labels.
        if label is None:
            continue
        first = turns[0]
        listing = listings[str(first["listing_id"])]
        rows.append(
            _snapshot(
                task="agreement",
                label=label,
                listing=listing,
                turn=first,
                history=[first],
                row_id=f"{thread_id}:agreement",
            )
        )
    return rows


def agreement_label(turns: list[dict[str, Any]]) -> str | None:
    if any(_status_id(turn.get("status_id")) in {1, 9} for turn in turns):
        return "1"
    if any(_status_id(turn.get("status_id")) == 8 for turn in turns):
        return None
    if any(str(turn.get("action", "")).lower() == "accept" for turn in turns):
        return "1"
    if not turns:
        return None
    last = turns[-1]
    if _status_id(last.get("status_id")) in {0, 2, 6}:
        return "0"
    action = str(last.get("action", "")).lower()
    status = str(last.get("status", "")).lower()
    terminal_negative_actions = {"decline", "expire", "leave", "reject", "quit"}
    terminal_negative_statuses = {"declined", "expired", "rejected", "closed_no_agreement"}
    if action in terminal_negative_actions or status in terminal_negative_statuses:
        return "0"
    return None


def final_price_ratio_task(listings: dict[str, dict[str, Any]], threads: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for thread_id, turns in threads.items():
        accepted = next(
            (
                turn
                for turn in turns
                if (turn["action"] == "accept" or _status_id(turn.get("status_id")) in {1, 9}) and turn["amount"] is not None
            ),
            None,
        )
        if accepted is None:
            continue
        first = turns[0]
        listing = listings[str(first["listing_id"])]
        rows.append(
            _snapshot(
                task="final_price_ratio",
                label=round(float(accepted["amount"]) / float(listing["listing_price"]), 6),
                listing=listing,
                turn=first,
                history=[first],
                row_id=f"{thread_id}:final_price_ratio",
            )
        )
    return rows


def response_latency_task(listings: dict[str, dict[str, Any]], threads: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for thread_id, turns in threads.items():
        for index, turn in enumerate(turns[:-1]):
            next_turn = turns[index + 1]
            listing = listings[str(turn["listing_id"])]
            latency = (_parse_time(str(next_turn["event_time"])) - _parse_time(str(turn["event_time"]))).total_seconds()
            rows.append(
                _snapshot(
                    task="response_latency",
                    label=latency,
                    listing=listing,
                    turn=turn,
                    history=turns[: index + 1],
                    row_id=f"{thread_id}:{turn['turn_index']}:response_latency",
                )
            )
    return rows


def assert_no_future_leakage(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        feature_names = set(row.get("features", {}))
        if feature_names & FORBIDDEN_FUTURE_FIELDS:
            return False
        history = row.get("observed_history", [])
        if any("future_" in str(key) or str(key) in FORBIDDEN_FUTURE_FIELDS for item in history for key in item):
            return False
        if any(str(item.get("status", "")).lower() in {"accepted", "declined", "expired", "finalized"} for item in history):
            return False
    return True


def _snapshot(*, task: str, label: Any, listing: dict[str, Any], turn: dict[str, Any], history: list[dict[str, Any]], row_id: str) -> dict[str, Any]:
    features = {
        "category": listing["category"],
        "condition": listing["condition"],
        "listing_price": listing["listing_price"],
        "reference_price": _safe_reference_price(listing),
        "current_actor": turn["actor"],
        "current_action": turn["action"],
        "current_amount": turn["amount"],
        "offer_to_asking_ratio": (float(turn["amount"]) / float(listing["listing_price"])) if turn.get("amount") else None,
        "round_number": turn["turn_index"],
        "event_time": turn["event_time"],
        "prior_turn_count": len(history) - 1,
        "prior_counter_count": sum(1 for item in history[:-1] if item["action"] == "counter"),
    }
    return {
        "row_id": row_id,
        "task": task,
        "label": label,
        "features": features,
        "observed_history": [_sanitize_history_turn(item) for item in history],
        "thread_id": turn["thread_id"],
        "listing_id": listing["listing_id"],
        "seller_id": listing["seller_id"],
        "buyer_id": turn.get("buyer_id"),
        "timestamp": turn["event_time"],
    }


def _seller_label(turn: dict[str, Any]) -> str:
    if turn["action"] == "accept":
        return "accept"
    if turn["action"] == "counter":
        return "counter"
    if turn["action"] == "decline":
        return "decline"
    return "expire"


def _buyer_label(turn: dict[str, Any]) -> str:
    if turn["action"] == "accept":
        return "accept"
    if turn["action"] == "counter":
        return "counter"
    if turn["action"] == "expire":
        return "expire"
    return "leave"


def _real_status_label(turn: dict[str, Any], *, counter_actor: str, next_turn: dict[str, Any] | None) -> str | None:
    status_id = _status_id(turn.get("status_id"))
    if status_id in {1, 9}:
        return "accept"
    if status_id in {2, 6}:
        return "decline"
    if status_id == 0:
        return "expire"
    if status_id == 7:
        if next_turn is not None and next_turn.get("actor") == counter_actor and next_turn.get("action") == "counter":
            return "counter"
        return None
    if status_id == 8:
        return None
    return None


def _status_id(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return None


def _safe_reference_price(listing: dict[str, Any]) -> Any:
    if "reference_price_unavailable_reason" in listing or "excluded_reference_price_ref_price4" in listing:
        return None
    return listing.get("reference_price")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _sanitize_history_turn(turn: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_index": turn["turn_index"],
        "actor": turn["actor"],
        "action": turn["action"],
        "amount": turn["amount"],
        "event_time": turn["event_time"],
    }


def _read_partitioned_table(root: Path, table_name: str) -> list[dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    table = manifest["tables"][table_name]
    rows: list[dict[str, Any]] = []
    for partition in table.get("partitions", []):
        path = Path(partition["path"])
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            rows.extend(pq.read_table(path).to_pylist())
        else:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        rows.append(json.loads(line))
    return rows
