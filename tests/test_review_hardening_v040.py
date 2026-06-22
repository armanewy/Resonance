from __future__ import annotations

import _bootstrap  # noqa: F401

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from behavior_lab.benchmarks.splits import (
    assert_disjoint_groups,
    chronological_group_purged_split,
)
from behavior_lab.core import stable_hash
from behavior_lab.data_sources.cache import ContentAddressedCache
from behavior_lab.data_sources.registry import AuthorizationEvidence, default_registry
from behavior_lab.datasets.nber_best_offer.tasks import agreement_label, agreement_task
from behavior_lab.offerlab_models.common import model_lineage, reserve_hidden_submission
from behavior_lab.offerlab_models.transfer import evaluate_transfer_ablation
from behavior_lab.offerlab_research import (
    AppendOnlyResearchStore,
    OfferLabResearchAPI,
    ResearchBudgetError,
    ResearchPermissionError,
)
from tools.ebay_api_probe import EbayApiProbe, StaticProbeClient
from tools.ebay_api_probe.probe import ROLE_REQUESTS


def _rows() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    def row(index: int, label: str = "accept") -> dict[str, object]:
        return {
            "row_id": f"r{index}",
            "thread_id": f"t{index}",
            "listing_id": f"l{index}",
            "task": "seller_next_action",
            "timestamp": f"2020-01-{index:02d}T00:00:00+00:00",
            "label": label,
            "features": {
                "category": "parts",
                "condition": "used",
                "listing_price": 100.0,
                "current_actor": "buyer",
                "current_action": "offer",
                "current_amount": 70.0 + index,
                "offer_to_asking_ratio": (70.0 + index) / 100.0,
                "round_number": 1,
                "prior_turn_count": 0,
                "prior_counter_count": 0,
            },
            "observed_history": [],
        }

    train = [row(1, "accept"), row(2, "counter"), row(3, "accept")]
    development = [row(4, "counter"), row(5, "accept")]
    hidden = [row(6, "accept"), row(7, "counter")]
    return train, development, hidden


def _proposal() -> dict[str, object]:
    return {
        "proposal_id": "p1",
        "terms": ["offer_to_asking_ratio"],
        "target_label": "accept",
        "falsification": "Fails if development log loss is not competitive.",
    }


class ReviewHardeningV040Tests(unittest.TestCase):
    def test_development_budget_persists_and_crash_consumes_reservation(self) -> None:
        train, development, hidden = _rows()
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "research.jsonl"
            api = OfferLabResearchAPI(
                campaign_id="persistent-dev",
                training_rows=train,
                development_rows=development,
                hidden_rows=hidden,
                development_evaluations=1,
                store=AppendOnlyResearchStore(store_path),
            )
            api.register_formula(_proposal())

            def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
                raise RuntimeError("simulated evaluator crash")

            api._evaluate = fail  # type: ignore[method-assign]
            with self.assertRaises(RuntimeError):
                api.evaluate_development("p1")
            reopened = OfferLabResearchAPI(
                campaign_id="persistent-dev",
                training_rows=train,
                development_rows=development,
                hidden_rows=hidden,
                development_evaluations=1,
                store=AppendOnlyResearchStore(store_path),
            )
            self.assertEqual(reopened.development_evaluations_remaining, 0)
            with self.assertRaises(ResearchBudgetError):
                reopened.evaluate_development("p1")

    def test_hidden_budget_is_reserved_before_evaluation_and_bound_to_artifact(self) -> None:
        train, development, hidden = _rows()
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "research.jsonl"
            api = OfferLabResearchAPI(
                campaign_id="persistent-hidden",
                training_rows=train,
                development_rows=development,
                hidden_rows=hidden,
                development_evaluations=2,
                hidden_submissions=1,
                store=AppendOnlyResearchStore(store_path),
            )
            api.register_formula(_proposal())
            api.evaluate_development("p1")
            original = api._evaluate

            def fail_hidden(proposal: object, rows: object, *, split: str) -> dict[str, object]:
                if split == "hidden":
                    raise RuntimeError("simulated hidden crash")
                return original(proposal, rows, split=split)  # type: ignore[arg-type]

            api._evaluate = fail_hidden  # type: ignore[method-assign]
            with self.assertRaises(RuntimeError):
                api.submit_hidden_once("p1", lockbox_id="named-lockbox")
            reopened = OfferLabResearchAPI(
                campaign_id="persistent-hidden",
                training_rows=train,
                development_rows=development,
                hidden_rows=hidden,
                development_evaluations=2,
                hidden_submissions=1,
                store=AppendOnlyResearchStore(store_path),
            )
            self.assertEqual(reopened.hidden_submissions_remaining, 0)
            with self.assertRaises(ResearchBudgetError):
                reopened.submit_hidden_once("p1", lockbox_id="renamed-lockbox")
            with self.assertRaises(ResearchBudgetError):
                reserve_hidden_submission(
                    store_path=store_path,
                    namespace="predictive_suite",
                    requested_lockbox_id="other-path",
                    target="seller_next_action",
                    hidden_rows=hidden,
                    artifact_id="artifact",
                )

    def test_model_suite_reservation_blocks_api_hidden_replay(self) -> None:
        train, development, hidden = _rows()
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "research.jsonl"
            reserve_hidden_submission(
                store_path=store_path,
                namespace="formula_suite",
                requested_lockbox_id="formula-first",
                target="seller_next_action_formula",
                hidden_rows=hidden,
                artifact_id="formula-artifact",
            )
            api = OfferLabResearchAPI(
                campaign_id="cross-path-hidden",
                training_rows=train,
                development_rows=development,
                hidden_rows=hidden,
                development_evaluations=2,
                hidden_submissions=1,
                store=AppendOnlyResearchStore(store_path),
            )
            api.register_formula(_proposal())
            api.evaluate_development("p1")
            with self.assertRaises(ResearchBudgetError):
                api.submit_hidden_once("p1", lockbox_id="api-after-formula")

    def test_legacy_hidden_case_hash_blocks_cross_path_replay(self) -> None:
        train, development, hidden = _rows()
        case_set_hash = stable_hash(
            sorted(
                stable_hash(
                    {
                        "task": row.get("task"),
                        "timestamp": row.get("timestamp"),
                        "features": row.get("features", {}),
                        "observed_history": row.get("observed_history", []),
                    }
                )
                for row in hidden
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "research.jsonl"
            AppendOnlyResearchStore(store_path).append(
                "hidden_submitted",
                {
                    "campaign_id": "legacy-hidden",
                    "result": {
                        "lockbox_id": "legacy-lockbox",
                        "hidden_case_set_hash": case_set_hash,
                    },
                },
            )
            with self.assertRaises(ResearchBudgetError):
                reserve_hidden_submission(
                    store_path=store_path,
                    namespace="predictive_suite",
                    requested_lockbox_id="predictive-after-legacy",
                    target="seller_next_action",
                    hidden_rows=hidden,
                    artifact_id="artifact",
                )

            api = OfferLabResearchAPI(
                campaign_id="legacy-hidden-api",
                training_rows=train,
                development_rows=development,
                hidden_rows=hidden,
                development_evaluations=2,
                hidden_submissions=1,
                store=AppendOnlyResearchStore(store_path),
            )
            api.register_formula(_proposal())
            api.evaluate_development("p1")
            with self.assertRaises(ResearchBudgetError):
                api.submit_hidden_once("p1", lockbox_id="api-after-legacy")

    def test_campaign_metadata_reset_and_overlapping_hidden_subset_are_rejected(self) -> None:
        train, development, hidden = _rows()
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "research.jsonl"
            store = AppendOnlyResearchStore(store_path)
            first = OfferLabResearchAPI(
                campaign_id="campaign-a",
                training_rows=train,
                development_rows=development,
                hidden_rows=hidden,
                store=store,
            )
            first.register_formula(_proposal())
            first.evaluate_development("p1")
            first.submit_hidden_once("p1", lockbox_id="first")

            changed = [dict(row) for row in hidden]
            changed[0] = dict(changed[0])
            changed[0]["label"] = "counter"
            with self.assertRaises(ResearchPermissionError):
                OfferLabResearchAPI(
                    campaign_id="campaign-a",
                    training_rows=train,
                    development_rows=development,
                    hidden_rows=changed,
                    store=AppendOnlyResearchStore(store_path),
                )

            renamed_hidden = []
            for index, row in enumerate(hidden):
                item = dict(row)
                item["row_id"] = f"renamed-row-{index}"
                item["thread_id"] = f"renamed-thread-{index}"
                item["listing_id"] = f"renamed-listing-{index}"
                renamed_hidden.append(item)
            second = OfferLabResearchAPI(
                campaign_id="campaign-b",
                training_rows=train,
                development_rows=development,
                hidden_rows=renamed_hidden,
                store=AppendOnlyResearchStore(store_path),
            )
            second.register_formula(_proposal())
            second.evaluate_development("p1")
            with self.assertRaises(ResearchBudgetError):
                second.submit_hidden_once("p1", lockbox_id="metadata-reset")

    def test_hidden_source_identity_blocks_feature_extraction_replay(self) -> None:
        _train, _development, hidden = _rows()
        changed_hidden = []
        for row in hidden:
            item = dict(row)
            item["features"] = dict(item["features"])  # type: ignore[index]
            item["features"]["current_amount"] = float(item["features"]["current_amount"]) + 1.0  # type: ignore[index]
            item["features"]["offer_to_asking_ratio"] = float(item["features"]["current_amount"]) / 100.0  # type: ignore[index]
            changed_hidden.append(item)
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "research.jsonl"
            reserve_hidden_submission(
                store_path=store_path,
                namespace="predictive_suite",
                requested_lockbox_id="first",
                target="seller_next_action",
                hidden_rows=hidden,
                artifact_id="artifact-a",
            )
            with self.assertRaises(ResearchBudgetError):
                reserve_hidden_submission(
                    store_path=store_path,
                    namespace="predictive_suite",
                    requested_lockbox_id="second",
                    target="seller_next_action",
                    hidden_rows=changed_hidden,
                    artifact_id="artifact-b",
                )

    def test_hidden_submission_rejects_post_development_proposal_mutation(self) -> None:
        train, development, hidden = _rows()
        api = OfferLabResearchAPI(
            campaign_id="artifact-binding",
            training_rows=train,
            development_rows=development,
            hidden_rows=hidden,
            store=AppendOnlyResearchStore(Path(tempfile.mkdtemp()) / "research.jsonl"),
        )
        api.register_formula(_proposal())
        api.evaluate_development("p1")
        api.proposals["p1"] = replace(api.proposals["p1"], terms=["round_number"])
        with self.assertRaises(ResearchPermissionError):
            api.submit_hidden_once("p1", lockbox_id="artifact")

    def test_chronological_group_split_purges_cross_boundary_threads(self) -> None:
        rows = []
        for index in range(1, 11):
            group = "cross" if index in {6, 7} else f"g{index}"
            rows.append(
                {
                    "row_id": f"r{index}",
                    "thread_id": group,
                    "timestamp": f"2020-01-{index:02d}T00:00:00+00:00",
                }
            )
        split = chronological_group_purged_split(
            rows,
            time_key="timestamp",
            group_key="thread_id",
        )
        self.assertIn("cross", split.purged_group_ids)
        self.assertEqual(split.purged_rows, 2)
        self.assertTrue(assert_disjoint_groups(split, group_key="thread_id"))
        all_ids = {row["row_id"] for row in split.train + split.development + split.hidden}
        self.assertNotIn("r6", all_ids)
        self.assertNotIn("r7", all_ids)

    def test_censored_negotiation_is_not_labeled_as_failed_agreement(self) -> None:
        censored = [
            {
                "thread_id": "t1",
                "listing_id": "l1",
                "buyer_id": "b1",
                "seller_id": "s1",
                "turn_index": 1,
                "actor": "buyer",
                "action": "offer",
                "amount": 70.0,
                "status": "submitted",
                "event_time": "2020-01-01T00:00:00+00:00",
            }
        ]
        self.assertIsNone(agreement_label(censored))
        rows = agreement_task(
            {
                "l1": {
                    "listing_id": "l1",
                    "seller_id": "s1",
                    "category": "parts",
                    "condition": "used",
                    "listing_price": 100.0,
                    "reference_price": 95.0,
                }
            },
            {"t1": censored},
        )
        self.assertEqual(rows, [])

    def test_transfer_default_is_explicitly_not_run(self) -> None:
        report = evaluate_transfer_ablation()
        self.assertEqual(report["status"], "not_run")
        self.assertIsNone(report["base_hidden_loss"])
        self.assertFalse(report["retained"])

    def test_authorized_data_requires_hashed_ledger_evidence(self) -> None:
        registry = default_registry()
        denied = registry.check("current_ebay_authorized_data", "production_export")
        self.assertFalse(denied.allowed)
        evidence = AuthorizationEvidence.create(
            source_id="current_ebay_authorized_data",
            authorization_id="auth-1",
            owner_subject_hash="owner-hash",
            authorized_at="2026-06-21T12:00:00+00:00",
            scopes=["sell.fulfillment.readonly"],
            ledger_record_hash="ledger-record-hash",
        )
        allowed = registry.check(
            "current_ebay_authorized_data",
            "production_export",
            authorization_evidence=evidence,
        )
        self.assertTrue(allowed.allowed)
        self.assertTrue(allowed.authorization_evidence_verified)
        tampered = evidence.to_dict()
        tampered["scopes"] = ["sell.finances"]
        self.assertFalse(
            registry.check(
                "current_ebay_authorized_data",
                "production_export",
                authorization_evidence=tampered,
            ).allowed
        )

    def test_model_lineage_hashes_actual_feature_values(self) -> None:
        train, _development, _hidden = _rows()
        first = model_lineage("m", train, feature_contract=["offer_to_asking_ratio"])
        changed = [dict(row) for row in train]
        changed[0] = dict(changed[0])
        changed[0]["features"] = dict(changed[0]["features"])
        changed[0]["features"]["offer_to_asking_ratio"] = 0.01
        second = model_lineage("m", changed, feature_contract=["offer_to_asking_ratio"])
        self.assertNotEqual(first["training_rows_hash"], second["training_rows_hash"])
        self.assertNotEqual(
            first["training_feature_values_hash"],
            second["training_feature_values_hash"],
        )

    def test_content_cache_handles_concurrent_same_object_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            source.write_bytes(b"same-content" * 1000)
            cache = ContentAddressedCache(root / "cache")
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(lambda _index: cache.add_file(source), range(16)))
            self.assertEqual(len({result.sha256 for result in results}), 1)
            destination = Path(results[0].path)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            manifest_lines = [
                line
                for line in (root / "cache" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(manifest_lines), 1)

    def test_unrelated_ebay_visibility_is_observed_not_assumed(self) -> None:
        responses = {
            "seller_owned_best_offers": {"status": 200},
            "buyer_participated_best_offers": {"status": 200},
            ROLE_REQUESTS["unrelated_public_ended"]: {"status": 200, "bestOffers": [{"price": {"value": "70"}}]},
        }
        report = EbayApiProbe(StaticProbeClient(responses)).run(
            scopes=["https://api.ebay.com/oauth/api_scope"],
            seller_owned_listing_id="a",
            buyer_participated_listing_id="b",
            unrelated_listing_id="c",
            authorized_production_user_token=True,
        )
        self.assertEqual(report["unrelated_visibility_observation"], "accessible")
        self.assertEqual(
            report["permission_matrix"][ROLE_REQUESTS["unrelated_public_ended"]]["observed_result"],
            "accessible",
        )


if __name__ == "__main__":
    unittest.main()
