from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from behavior_lab.bridge import (
    CAMPAIGN_001_ID,
    import_snapshot_file,
    prepare_snapshot_file,
    validate_snapshot_file,
    write_campaign_001_template,
)
from behavior_lab.benchmarks.contracts import validate_manifest_file
from behavior_lab.campaign001_collector import (
    DEFAULT_DATA_DIR,
    amend_capture,
    finalize_capture,
    invalidate_capture,
    load_script,
    missed_capture,
    resume_capture,
    start_capture,
    status_capture,
)
from behavior_lab.discovery import DiscoveryLoop
from behavior_lab.evaluation import evaluate_model, paired_compare, pareto_frontier
from behavior_lab.gym import TARGET, WorldGym
from behavior_lab.ledger import ImmutableLedger
from behavior_lab.data_sources.registry import default_registry
from behavior_lab.datasets.auctionnet.strategy import compare_strategies
from behavior_lab.datasets.craigslist_bargain.parser import evaluate_parser
from behavior_lab.datasets.criteo_uplift.uplift import simple_uplift_report
from behavior_lab.datasets.nber_best_offer.acquire import fetch_codebook, fetch_full
from behavior_lab.datasets.nber_best_offer.audit import audit as nber_audit
from behavior_lab.datasets.nber_best_offer.audit import benchmark as nber_benchmark
from behavior_lab.datasets.nber_best_offer.inventory import inventory_path
from behavior_lab.datasets.nber_best_offer.normalize import build_sample_dataset, normalize_dataset
from behavior_lab.datasets.nber_best_offer.real_normalize import full_normalization_status, inspect_real_source_schema, normalize_real_dataset
from behavior_lab.datasets.nber_best_offer.replication import replication_check, validate_replication_targets
from behavior_lab.datasets.nber_best_offer.source_inventory import inventory_official_sources, public_summary, run_source_inventory
from behavior_lab.datasets.nber_best_offer.source_schema import inspect_schema
from behavior_lab.datasets.open_bandit.ope import evaluate_policy
from behavior_lab.offerlab import (
    ingest_offerlab_snapshots,
    profit_audit,
    profit_audit_report,
    recommend_offer_action,
    write_profit_audit_report,
    write_campaign_002_template,
    load_offerlab_snapshots,
)
from behavior_lab.offerlab_pilot import (
    audit_pilot,
    import_pilot,
    inspect_input as inspect_pilot_input,
    onboard_input as onboard_pilot_input,
    shadow_report_pilot,
    write_template as write_pilot_template,
)
from behavior_lab.offerlab_models import run_sample_research_suite
from behavior_lab.offerlab_models.benchmark_v1 import BenchmarkPaths, run_offerlab_benchmark_v1
from behavior_lab.offerlab_models.benchmark_v2 import BenchmarkV2Paths as BenchmarkV2BuildPaths
from behavior_lab.offerlab_models.benchmark_v2 import build_offerlab_benchmark_v2
from behavior_lab.offerlab_models.benchmark_v2_runner import BenchmarkV2Paths as BenchmarkV2RunnerPaths
from behavior_lab.offerlab_models.benchmark_v2_runner import run_offerlab_benchmark_v2
from behavior_lab.offerlab_models.benchmark_v2_integration import BenchmarkV2IntegrationPaths
from behavior_lab.offerlab_models.benchmark_v2_integration import run_offerlab_benchmark_v2_integration
from behavior_lab.research_api import ResearchAPI
from behavior_lab.runner import BatchConfig, SyntheticBatchRunner
from behavior_lab.stress import LabStressTester
from behavior_lab.worlds import make_world


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value may not be negative")
    return number


def command_seed_world(args: argparse.Namespace) -> None:
    world = make_world(args.world, seed=args.seed)
    gym = WorldGym(args.data_dir, world=world)
    added = gym.seed(args.episodes)
    _print_json({"data_dir": str(Path(args.data_dir).resolve()), "world": world.name, "episodes_added": added})


def command_verify_ledger(args: argparse.Namespace) -> None:
    ledger = ImmutableLedger(Path(args.data_dir) / "ledger.jsonl")
    _print_json({"ledger": str(ledger.path.resolve()), "valid": ledger.verify_hash_chain(), "records": len(ledger.scan())})


def command_run_loop(args: argparse.Namespace) -> None:
    world = make_world(args.world, seed=args.seed)
    gym = WorldGym(args.data_dir, world=world)
    if not gym.decision_episodes():
        gym.seed(args.episodes)
    report = DiscoveryLoop(gym).run(
        iterations=args.iterations,
        offline_trials_per_iteration=args.offline_trials,
        prospective_episodes=args.prospective_episodes,
    )
    _print_json(report)


def command_demo(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    if args.reset and data_dir.exists():
        shutil.rmtree(data_dir)
    world = make_world(args.world, seed=args.seed)
    gym = WorldGym(data_dir, world=world)
    if not gym.decision_episodes():
        gym.seed(args.episodes)

    campaign_id = "demo_initial"
    api = ResearchAPI(gym, campaign_id=campaign_id)
    splits = gym.splits(campaign_id)
    models = api.fit_model_zoo()
    dev_metrics = [evaluate_model(model, splits["development"], split="development", include_details=True) for model in models]
    dev_metrics.sort(key=lambda item: item.log_loss)
    best_model = next(model for model in models if model.model_id == dev_metrics[0].model_id)
    pairwise = paired_compare(models[0], best_model, splits["development"])
    proposal = api.propose_experiment([model.model_id for model in models[:4]])
    experiment = api.run_offline_experiment(proposal, trials=args.offline_trials)

    loop_report = DiscoveryLoop(gym).run(
        iterations=args.iterations,
        offline_trials_per_iteration=args.offline_trials,
        prospective_episodes=args.prospective_episodes,
    )
    gym.ledger.verify_hash_chain()
    _print_json(
        {
            "data_dir": str(data_dir.resolve()),
            "world": world.name,
            "wave_1": {
                "campaign_id": campaign_id,
                "episodes": len(gym.decision_episodes()),
                "splits": {key: len(value) for key, value in splits.items()},
                "best_development": asdict(dev_metrics[0]),
            },
            "wave_2": {
                "models_fit": len(models),
                "pareto_frontier": pareto_frontier(dev_metrics),
                "paired_compare_base_vs_best": pairwise,
            },
            "wave_3": experiment,
            "wave_4": loop_report,
            "ledger_records": len(gym.ledger.scan()),
            "ledger_valid": True,
        }
    )


def command_stress_test(args: argparse.Namespace) -> None:
    tester = LabStressTester()
    data_dir = Path(args.data_dir)
    if args.matrix:
        _print_json(tester.run_world_matrix(data_dir, episodes=args.episodes, seed=args.seed))
    else:
        _print_json(tester.run(data_dir, episodes=args.episodes, seed=args.seed, world=args.world))


def command_batch_stress(args: argparse.Namespace) -> None:
    config = BatchConfig(
        worlds=_parse_csv_strings(args.worlds),
        seeds=_parse_csv_ints(args.seeds),
        episode_counts=_parse_csv_ints(args.episode_counts),
    )
    _print_json(SyntheticBatchRunner(args.data_dir).run(config))


def command_campaign_template(args: argparse.Namespace) -> None:
    template = write_campaign_001_template(args.output)
    _print_json({"output": str(Path(args.output).resolve()), "campaign_id": template["campaign_id"]})


def command_bridge_hash(args: argparse.Namespace) -> None:
    snapshots = prepare_snapshot_file(args.input, args.output)
    _print_json(
        {
            "input": str(Path(args.input).resolve()),
            "output": str(Path(args.output).resolve()),
            "snapshots": len(snapshots),
            "source_hashes": [snapshot["source_hash"] for snapshot in snapshots],
        }
    )


def command_bridge_validate(args: argparse.Namespace) -> None:
    _print_json(validate_snapshot_file(args.input, campaign_id=args.campaign_id))


def command_bridge_import(args: argparse.Namespace) -> None:
    result = import_snapshot_file(args.input, data_dir=args.data_dir, campaign_id=args.campaign_id)
    _print_json(asdict(result))


def command_campaign_001_capture_start(args: argparse.Namespace) -> None:
    phase = "pilot" if args.pilot else None
    _print_json(start_capture(args.data_dir, script=load_script(args.script), collection_phase=phase))


def command_campaign_001_capture_finalize(args: argparse.Namespace) -> None:
    _print_json(finalize_capture(args.episode_id, args.data_dir, script=load_script(args.script)))


def command_campaign_001_capture_resume(args: argparse.Namespace) -> None:
    _print_json(resume_capture(args.data_dir, episode_id=args.episode_id, script=load_script(args.script)))


def command_campaign_001_capture_missed(args: argparse.Namespace) -> None:
    phase = "pilot" if args.pilot else None
    _print_json(missed_capture(args.data_dir, script=load_script(args.script), collection_phase=phase))


def command_campaign_001_capture_status(args: argparse.Namespace) -> None:
    _print_json(status_capture(args.data_dir))


def command_campaign_001_capture_amend(args: argparse.Namespace) -> None:
    value: Any
    try:
        value = json.loads(args.value)
    except json.JSONDecodeError:
        value = args.value
    _print_json(amend_capture(args.episode_id, args.field, value, args.reason, args.data_dir))


def command_campaign_001_capture_invalidate(args: argparse.Namespace) -> None:
    _print_json(invalidate_capture(args.episode_id, args.reason, args.data_dir))


def command_offerlab_template(args: argparse.Namespace) -> None:
    template = write_campaign_002_template(args.output)
    _print_json({"output": str(Path(args.output).resolve()), "campaign_id": template["campaign_id"]})


def command_offerlab_ingest(args: argparse.Namespace) -> None:
    _print_json(asdict(ingest_offerlab_snapshots(args.input, data_dir=args.data_dir)))


def command_offerlab_audit(args: argparse.Namespace) -> None:
    _print_json(profit_audit(args.data_dir))


def command_offerlab_report(args: argparse.Namespace) -> None:
    if args.output:
        _print_json(write_profit_audit_report(args.data_dir, args.output))
    else:
        _print_json(profit_audit_report(args.data_dir))


def command_offerlab_recommend(args: argparse.Namespace) -> None:
    snapshots = load_offerlab_snapshots(args.input)
    if len(snapshots) != 1:
        raise SystemExit("offerlab-recommend requires exactly one snapshot")
    config = None
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    _print_json(recommend_offer_action(snapshots[0], data_dir=args.data_dir, config=config))


def command_offerlab_pilot_template(args: argparse.Namespace) -> None:
    _print_json(write_pilot_template(args.output_dir))


def command_offerlab_pilot_inspect(args: argparse.Namespace) -> None:
    _print_json(inspect_pilot_input(args.input_dir))


def command_offerlab_pilot_onboard(args: argparse.Namespace) -> None:
    _print_json(onboard_pilot_input(args.input_dir, output_path=args.output))


def command_offerlab_pilot_import(args: argparse.Namespace) -> None:
    _print_json(asdict(import_pilot(args.input_dir, data_root=args.data_root, pilot_id=args.pilot_id)))


def command_offerlab_pilot_audit(args: argparse.Namespace) -> None:
    _print_json(audit_pilot(args.pilot_id, data_root=args.data_root))


def command_offerlab_pilot_shadow_report(args: argparse.Namespace) -> None:
    _print_json(shadow_report_pilot(args.pilot_id, data_root=args.data_root, output_path=args.output))


def command_offerlab_models_sample(args: argparse.Namespace) -> None:
    _print_json(run_sample_research_suite())


def command_offerlab_models_benchmark_v1(args: argparse.Namespace) -> None:
    raise SystemExit(
        "OfferLab Benchmark v1 is frozen and hidden-spent. "
        "Do not rerun it; create Benchmark v2 with fresh hidden cases instead."
    )


def command_offerlab_models_benchmark_v2_build(args: argparse.Namespace) -> None:
    _print_json(
        build_offerlab_benchmark_v2(
            BenchmarkV2BuildPaths(
                normalized_dir=Path(args.normalized_dir),
                output_dir=Path(args.output_dir),
                protocol_path=Path(args.protocol),
                v1_final_manifest_path=Path(args.v1_final_manifest),
                external_v1_hidden_tokens_path=Path(args.external_v1_hidden_tokens) if args.external_v1_hidden_tokens else None,
            ),
            require_full_release=not args.allow_bounded_test_input,
            partition_rows=args.partition_rows,
        )
    )


def command_offerlab_models_benchmark_v2(args: argparse.Namespace) -> None:
    _print_json(
        run_offerlab_benchmark_v2(
            BenchmarkV2RunnerPaths(
                normalized_dir=Path(args.normalized_dir),
                output_path=Path(args.output),
                doc_path=Path(args.doc),
                model_cards_dir=Path(args.model_cards_dir),
                protocol_path=Path(args.protocol),
            ),
            batch_size=args.batch_size,
            allow_hidden_submission=args.submit_hidden,
        )
    )


def command_offerlab_models_benchmark_v2_integrate(args: argparse.Namespace) -> None:
    _print_json(
        run_offerlab_benchmark_v2_integration(
            BenchmarkV2IntegrationPaths(
                normalized_dir=Path(args.normalized_dir),
                benchmark_dir=Path(args.benchmark_dir),
                output_path=Path(args.output),
                preregistration_path=Path(args.preregistration),
                pre_hidden_output_path=Path(args.pre_hidden_output),
                doc_path=Path(args.doc),
                pre_hidden_doc_path=Path(args.pre_hidden_doc),
                model_cards_dir=Path(args.model_cards_dir),
                protocol_path=Path(args.protocol),
                v1_final_manifest_path=Path(args.v1_final_manifest),
                external_v1_hidden_tokens_path=Path(args.external_v1_hidden_tokens) if args.external_v1_hidden_tokens else None,
            ),
            batch_size=args.batch_size,
            partition_rows=args.partition_rows,
            allow_bounded_test_input=args.allow_bounded_test_input,
            submit_hidden=args.submit_hidden,
        )
    )


def command_money_offerlab_evaluate(args: argparse.Namespace) -> None:
    from behavior_lab.labs.offerlab_money import evaluate

    _print_json(
        evaluate(
            args.pilot_id,
            data_root=args.data_root,
            money_root=args.money_root,
            evaluation_timestamp=args.evaluation_timestamp,
            output_path=args.output,
        )
    )


def command_money_offerlab_report(args: argparse.Namespace) -> None:
    from behavior_lab.labs.offerlab_money import report

    _print_json(report(args.pilot_id, data_root=args.data_root, money_root=args.money_root, output_path=args.output))


def command_money_weather_edge_backfill(args: argparse.Namespace) -> None:
    from behavior_lab.labs.weather_edge import FixtureWeatherEdgeProvider, backfill

    provider = FixtureWeatherEdgeProvider.from_json(args.fixture)
    payload = backfill(provider, args.storage_root, as_of=args.as_of)
    _write_json_output(args.output, payload)
    _print_json(payload)


def command_money_weather_edge_paper_cycle(args: argparse.Namespace) -> None:
    from behavior_lab.labs.weather_edge import FixtureWeatherEdgeProvider, paper_cycle

    provider = FixtureWeatherEdgeProvider.from_json(args.fixture)
    payload = paper_cycle(provider, args.storage_root, as_of=args.as_of)
    _write_json_output(args.output, payload)
    _print_json(payload)


def command_money_weather_edge_report(args: argparse.Namespace) -> None:
    from behavior_lab.labs.weather_edge import FixtureWeatherEdgeProvider, report

    provider = FixtureWeatherEdgeProvider.from_json(args.fixture) if args.fixture else None
    payload = report(args.storage_root, provider=provider, as_of=args.as_of)
    _write_json_output(args.output, payload)
    _print_json(payload)


def command_money_etf_risk_backfill(args: argparse.Namespace) -> None:
    from behavior_lab.labs.etf_risk import ETFRiskConfig
    from behavior_lab.labs.etf_risk.commands import backfill
    from behavior_lab.money.integration import fixture_etf_provider

    _require_demo_fixture(args.demo_fixture)
    provider, _sessions = fixture_etf_provider(session_count=args.session_count)
    payload = backfill(
        provider,
        ledger_path=args.ledger_path,
        config=ETFRiskConfig(min_history_trading_days=args.min_history_days),
        start=args.start,
        end=args.end,
    )
    _write_json_output(args.output, payload)
    _print_json(payload)


def command_money_etf_risk_paper_cycle(args: argparse.Namespace) -> None:
    from behavior_lab.labs.etf_risk import ETFRiskConfig
    from behavior_lab.labs.etf_risk.commands import paper_cycle
    from behavior_lab.money.integration import fixture_etf_provider

    _require_demo_fixture(args.demo_fixture)
    provider, sessions = fixture_etf_provider(session_count=args.session_count)
    decision_cutoff = args.decision_cutoff or f"{sessions[-2]}T21:10:00+00:00"
    payload = paper_cycle(
        provider,
        ledger_path=args.ledger_path,
        config=ETFRiskConfig(min_history_trading_days=args.min_history_days),
        decision_cutoff=decision_cutoff,
        strategy_id=args.strategy_id,
    )
    _write_json_output(args.output, payload)
    _print_json(payload)


def command_money_etf_risk_report(args: argparse.Namespace) -> None:
    from behavior_lab.labs.etf_risk import ETFRiskConfig
    from behavior_lab.labs.etf_risk.commands import report
    from behavior_lab.money.integration import fixture_etf_provider

    _require_demo_fixture(args.demo_fixture)
    provider, _sessions = fixture_etf_provider(session_count=args.session_count)
    payload = report(
        provider,
        ledger_path=args.ledger_path,
        config=ETFRiskConfig(min_history_trading_days=args.min_history_days),
        start=args.start,
        end=args.end,
    )
    _write_json_output(args.output, payload)
    _print_json(payload)


def command_money_wave2_integration_report(args: argparse.Namespace) -> None:
    from behavior_lab.money.integration import run_wave2_integration_proof

    _print_json(
        run_wave2_integration_proof(
            output_dir=args.output_dir,
            workspace=args.workspace,
            generated_at=args.generated_at,
        )
    )


def command_money_autopilot(args: argparse.Namespace) -> None:
    from behavior_lab.money.autopilot import MoneyAutopilot

    autopilot = MoneyAutopilot.from_path(args.portfolio)
    command = getattr(args, "autopilot_command", None) or "run-once"
    if command == "status":
        _print_json(autopilot.status())
    elif command == "run-once":
        _print_json(autopilot.run_once())
    elif command == "approvals":
        _print_json(autopilot.approvals())
    elif command == "weekly-report":
        _print_json(autopilot.weekly_report(write_event=True))
    elif command == "pause":
        _print_json(autopilot.pause(args.contract_id))
    elif command == "resume":
        _print_json(autopilot.resume(args.contract_id))
    else:
        raise SystemExit(f"unsupported autopilot command: {command}")


def command_money_canary(args: argparse.Namespace) -> None:
    from behavior_lab.money.canary import CanaryOptions, MoneyCanaryManager

    manager = MoneyCanaryManager(args.state_dir)
    command = args.canary_command
    if command == "start":
        payload = manager.start(
            args.contract_id,
            CanaryOptions(
                lab=args.lab,
                as_of=args.as_of,
                strategy_version=args.strategy_version,
                source_version=args.source_version,
                seller_pilot_ready=args.seller_pilot_ready,
            ),
        )
    elif command == "resume":
        payload = manager.resume(args.canary_id, as_of=args.as_of, strategy_version=args.strategy_version)
    elif command == "status":
        payload = manager.status(args.canary_id)
    elif command == "report":
        payload = manager.report(args.canary_id)
    elif command == "invalidate":
        payload = manager.invalidate(args.canary_id, reason=args.reason, as_of=args.as_of)
    else:
        raise SystemExit(f"unsupported canary command: {command}")
    _write_json_output(getattr(args, "output", None), payload)
    _print_json(payload)


def command_money_tournament(args: argparse.Namespace) -> None:
    from behavior_lab.money.tournament import run_financial_tournament

    payload = run_financial_tournament(
        output_dir=args.output_dir,
        docs_dir=args.docs_dir,
        workspace=args.workspace,
        generated_at=args.generated_at,
    )
    _print_json(payload)


def command_money_operations(args: argparse.Namespace) -> None:
    from behavior_lab.money.operations import MoneyOperations

    operations = MoneyOperations(args.state_dir)
    command = args.operations_command
    if command == "start":
        payload = operations.start(
            as_of=args.as_of,
            seller_readiness_report=args.seller_readiness_report,
            release_commit=args.release_commit,
        )
    elif command == "status":
        payload = operations.status()
    elif command == "doctor":
        payload = operations.doctor()
    elif command == "weekly-report":
        payload = operations.weekly_report()
    elif command == "stop":
        payload = operations.stop()
    elif command == "recover":
        payload = operations.recover(as_of=args.as_of)
    else:
        raise SystemExit(f"unsupported operations command: {command}")
    _write_json_output(getattr(args, "output", None), payload)
    _print_json(payload)


def command_money_contract_scout(args: argparse.Namespace) -> None:
    from behavior_lab.money.contract_scout import ContractScout, load_proposals

    scout = ContractScout(args.state_dir, operations_state_dir=getattr(args, "operations_state_dir", None))
    command = args.contract_scout_command
    if command == "run":
        payload = scout.run(
            proposals=load_proposals(args.proposals_json) if args.proposals_json else None,
            search_budget=args.search_budget,
            llm_budget_usd=args.llm_budget_usd,
            include_seed_families=not args.no_seed_families,
        )
    elif command == "proposals":
        payload = scout.proposals()
    elif command == "approve":
        payload = scout.approve(args.proposal_id)
    elif command == "reject":
        payload = scout.reject(args.proposal_id, reason=args.reason)
    elif command == "report":
        payload = scout.report()
    else:
        raise SystemExit(f"unsupported contract-scout command: {command}")
    _write_json_output(getattr(args, "output", None), payload)
    _print_json(payload)


def command_money_data_mesh(args: argparse.Namespace) -> None:
    from behavior_lab.finance_data.data_mesh import FinancialDataMesh, load_fixture, load_manifest, load_manifests

    mesh = FinancialDataMesh(args.state_dir)
    command = args.data_mesh_command
    if command == "validate-manifest":
        payload = mesh.validate_manifest(load_manifest(args.manifest))
    elif command == "trial":
        payload = mesh.trial_manifest(load_manifest(args.manifest), fixture_payload=load_fixture(args.fixture), fixture_name=Path(args.fixture).name)
    elif command == "activate":
        payload = mesh.activate_manifest(load_manifest(args.manifest), fixture_payload=load_fixture(args.fixture), fixture_name=Path(args.fixture).name)
    elif command == "acquire":
        fixtures = _load_json_file(args.fixtures_json) if args.fixtures_json else {}
        source_catalog = _load_json_file(args.source_catalog_json) if args.source_catalog_json else []
        payload = mesh.acquire(
            contract_proposals=_load_json_file(args.contracts_json),
            manifests=load_manifests(args.manifests_json),
            fixtures_by_source=fixtures,
            source_catalog=source_catalog,
            search_budget=args.search_budget,
            llm_budget_usd=args.llm_budget_usd,
        )
    elif command == "repair":
        payload = mesh.repair_source(
            args.source_id,
            failure=_load_json_file(args.failure_json),
            candidate_manifest=load_manifest(args.candidate_manifest),
            fixture_payload=load_fixture(args.fixture),
        )
    elif command == "backfill-plan":
        payload = mesh.backfill_plan(
            source_id=args.source_id,
            start_date=args.start_date,
            end_date=args.end_date,
            chunk_days=args.chunk_days,
            completed_chunk_ids=_parse_csv_strings(args.completed_chunks) if args.completed_chunks else [],
        )
    elif command == "connector-audit":
        payload = mesh.audit_generated_connector(
            source_id=args.source_id,
            manifest_hash=args.manifest_hash,
            code=Path(args.code_file).read_text(encoding="utf-8"),
        )
    elif command == "classify":
        payload = mesh.classify_source_value(source_id=args.source_id, metrics=_load_json_file(args.metrics_json))
    elif command == "catalog":
        payload = mesh.catalog()
    else:
        raise SystemExit(f"unsupported data-mesh command: {command}")
    _write_json_output(getattr(args, "output", None), payload)
    _print_json(payload)


def _write_json_output(output: str | None, payload: dict[str, Any]) -> None:
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json_file(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_demo_fixture(enabled: bool) -> None:
    if not enabled:
        raise SystemExit("ETF Risk CLI currently requires --demo-fixture until an authorized provider adapter is wired.")


def command_data_source_list(args: argparse.Namespace) -> None:
    _print_json({"sources": default_registry().list()})


def command_data_source_inspect(args: argparse.Namespace) -> None:
    _print_json(default_registry().inspect(args.source_id))


def command_data_source_verify(args: argparse.Namespace) -> None:
    _print_json(default_registry().verify_lineage(args.source_id, args.use))


def command_data_source_permissions(args: argparse.Namespace) -> None:
    _print_json(default_registry().permissions(args.source_id))


def command_benchmark_validate_manifest(args: argparse.Namespace) -> None:
    _print_json(validate_manifest_file(args.input))


def command_nber_fetch(args: argparse.Namespace) -> None:
    if args.full:
        _print_json(fetch_full(cache_dir=args.cache_dir, url=args.url, explicit=True).to_dict())
    else:
        _print_json(fetch_codebook(cache_dir=args.cache_dir).to_dict())


def command_nber_inventory(args: argparse.Namespace) -> None:
    _print_json(inventory_path(args.input))


def command_nber_source_inventory(args: argparse.Namespace) -> None:
    raw_dir = args.raw_dir or str(Path(os.environ.get("OFFERLAB_DATA_ROOT", r"C:\OfferLabData")) / "raw" / "nber_best_offer")
    if args.write_report:
        manifest = run_source_inventory(
            raw_dir=raw_dir,
            manifest_path=args.manifest,
            doc_path=args.doc,
            first_sample_rows=args.first_sample_rows,
            reservoir_rows=args.reservoir_rows,
            chronological_rows_per_slice=args.chronological_rows_per_slice,
            timeout_seconds=args.timeout_seconds,
            download=args.download,
        )
        _print_json(public_summary(manifest))
    else:
        _print_json(
            inventory_official_sources(
                raw_dir=raw_dir,
                download=args.download,
                sample_dir=args.sample_dir,
                reservoir_size=args.reservoir_rows,
            )
        )


def command_nber_build_sample(args: argparse.Namespace) -> None:
    _print_json(build_sample_dataset(args.output_dir))


def command_nber_normalize(args: argparse.Namespace) -> None:
    _print_json(normalize_dataset(args.input_dir, args.output_dir))


def command_nber_inspect_schema(args: argparse.Namespace) -> None:
    if args.raw_dir:
        _print_json(inspect_real_source_schema(args.raw_dir))
    else:
        _print_json(inspect_schema(codebook_path=args.codebook))


def command_nber_normalize_real(args: argparse.Namespace) -> None:
    raw_dir = args.raw_dir or str(Path(os.environ.get("OFFERLAB_DATA_ROOT", r"C:\OfferLabData")) / "raw" / "nber_best_offer")
    output_dir = args.output_dir or str(Path(os.environ.get("OFFERLAB_DATA_ROOT", r"C:\OfferLabData")) / "processed" / "nber_best_offer_full")
    _print_json(
        normalize_real_dataset(
            raw_dir,
            output_dir,
            limit_threads=args.limit_threads,
            full=args.full,
            bucket_count=args.bucket_count,
            partition_rows=args.partition_rows,
            seed=args.seed,
            resume=args.resume,
            stop_after_thread_pass=args.stop_after_thread_pass,
        )
    )


def command_nber_full_status(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or str(Path(os.environ.get("OFFERLAB_DATA_ROOT", r"C:\OfferLabData")) / "processed" / "nber_best_offer_full")
    _print_json(full_normalization_status(output_dir))


def command_nber_replication_check(args: argparse.Namespace) -> None:
    if args.normalized_dir:
        _print_json(replication_check(args.normalized_dir, targets_path=args.targets))
    else:
        _print_json(validate_replication_targets(args.targets))


def command_nber_benchmark(args: argparse.Namespace) -> None:
    _print_json(nber_benchmark(args.normalized_dir))


def command_nber_audit(args: argparse.Namespace) -> None:
    _print_json(nber_audit(args.normalized_dir, output_path=args.output))


def command_benchmark_suite_permissions(args: argparse.Namespace) -> None:
    registry = default_registry()
    sources = ["nber_ebay_best_offer", "open_bandit_dataset", "criteo_uplift", "auctionnet", "craigslist_bargain"]
    _print_json({source: registry.permissions(source) for source in sources})


def command_benchmark_suite_run(args: argparse.Namespace) -> None:
    open_bandit_logs = [
        {"action": "a", "propensity": 0.5, "reward": 1.0},
        {"action": "b", "propensity": 0.5, "reward": 0.0},
        {"action": "a", "propensity": 0.5, "reward": 1.0},
        {"action": "b", "propensity": 0.5, "reward": 1.0},
    ]
    open_bandit = evaluate_policy(open_bandit_logs, lambda _row: {"a": 0.5, "b": 0.5})
    criteo = simple_uplift_report(
        [
            {"treatment": 0, "conversion": 0},
            {"treatment": 0, "conversion": 0},
            {"treatment": 1, "conversion": 1},
            {"treatment": 1, "conversion": 0},
        ]
    )
    craigslist = evaluate_parser(
        [
            {"text": "Would you take $80 if I pick up tonight?", "offer_amount": 80.0, "act": "propose"},
            {"text": "I can meet you at $95, final offer.", "offer_amount": 95.0, "act": "counter"},
            {"text": "Deal, I accept.", "offer_amount": None, "act": "accept"},
        ]
    )
    registry = default_registry()
    _print_json(
        {
            "DIRECT_EVIDENCE": {
                "source_id": "nber_ebay_best_offer",
                "status": "run nber-best-offer benchmark on normalized NBER data",
                "production_export_permission": registry.check("nber_ebay_best_offer", "production_export").to_dict(),
            },
            "EVALUATOR_VALIDATION": open_bandit,
            "CAUSAL_VALIDATION": criteo,
            "SIMULATION": compare_strategies(),
            "LANGUAGE_EXTRACTION": craigslist,
            "ARTIFACT_LINEAGE": {
                "production_export": registry.verify_lineage(
                    ["nber_ebay_best_offer", "open_bandit_dataset", "criteo_uplift", "auctionnet", "craigslist_bargain"],
                    "production_export",
                )
            },
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Behavior Discovery Lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed-world", help="Seed a hidden synthetic world into the ledger")
    seed.add_argument("--data-dir", default=".behavior_lab")
    seed.add_argument("--world", default="habit")
    seed.add_argument("--episodes", type=_positive, default=200)
    seed.add_argument("--seed", type=int, default=7)
    seed.set_defaults(func=command_seed_world)

    loop = subparsers.add_parser("run-loop", help="Run the campaign-safe offline discovery loop")
    loop.add_argument("--data-dir", default=".behavior_lab")
    loop.add_argument("--world", default="habit")
    loop.add_argument("--episodes", type=_positive, default=200)
    loop.add_argument("--iterations", type=_positive, default=3)
    loop.add_argument("--offline-trials", type=_positive, default=8)
    loop.add_argument("--prospective-episodes", type=_nonnegative, default=40)
    loop.add_argument("--seed", type=int, default=7)
    loop.set_defaults(func=command_run_loop)

    verify = subparsers.add_parser("verify-ledger", help="Verify the append-only hash chain")
    verify.add_argument("--data-dir", default=".behavior_lab")
    verify.set_defaults(func=command_verify_ledger)

    stress = subparsers.add_parser(
        "stress-test",
        help="Audit chronology, redaction, baselines, formula discovery, and intervention direction",
    )
    stress.add_argument("--data-dir", default=".stress_lab")
    stress.add_argument("--world", default="habit")
    stress.add_argument("--episodes", type=_positive, default=160)
    stress.add_argument("--seed", type=int, default=17)
    stress.add_argument("--matrix", action="store_true", help="Run the audit across all synthetic hidden worlds")
    stress.set_defaults(func=command_stress_test)

    batch = subparsers.add_parser("batch-stress", help="Run locked/idempotent synthetic stress batches")
    batch.add_argument("--data-dir", default="runs/batch")
    batch.add_argument("--worlds", default="habit,two_mode,threshold,nonstationary,confounded")
    batch.add_argument("--seeds", default="11,23,47,89,131")
    batch.add_argument("--episode-counts", default="100,300,1000")
    batch.set_defaults(func=command_batch_stress)

    data_source = subparsers.add_parser("data-source", help="Inspect registered external dataset permissions")
    data_source_subparsers = data_source.add_subparsers(dest="data_source_command", required=True)

    data_source_list = data_source_subparsers.add_parser("list", help="List registered data sources")
    data_source_list.set_defaults(func=command_data_source_list)

    data_source_inspect = data_source_subparsers.add_parser("inspect", help="Inspect one registered data source")
    data_source_inspect.add_argument("source_id")
    data_source_inspect.set_defaults(func=command_data_source_inspect)

    data_source_verify = data_source_subparsers.add_parser("verify", help="Verify source lineage for a requested use")
    data_source_verify.add_argument("source_id", nargs="+")
    data_source_verify.add_argument("--use", default="production_export")
    data_source_verify.set_defaults(func=command_data_source_verify)

    data_source_permissions = data_source_subparsers.add_parser("permissions", help="Show allowed uses for one source")
    data_source_permissions.add_argument("source_id")
    data_source_permissions.set_defaults(func=command_data_source_permissions)

    money = subparsers.add_parser("money", help="Paper-only financial decision lab utilities")
    money_subparsers = money.add_subparsers(dest="money_command", required=True)

    money_offerlab = money_subparsers.add_parser("offerlab", help="OfferLab Money paper shadow decisions")
    money_offerlab_subparsers = money_offerlab.add_subparsers(dest="money_offerlab_command", required=True)
    money_offerlab_evaluate = money_offerlab_subparsers.add_parser("evaluate", help="Evaluate a seller pilot into paper money ledger entries")
    money_offerlab_evaluate.add_argument("pilot_id", metavar="PILOT_ID")
    money_offerlab_evaluate.add_argument("--data-root", help="External seller pilot ledger root")
    money_offerlab_evaluate.add_argument("--money-root", help="Money ledger root for this pilot")
    money_offerlab_evaluate.add_argument("--evaluation-timestamp")
    money_offerlab_evaluate.add_argument("--output")
    money_offerlab_evaluate.set_defaults(func=command_money_offerlab_evaluate)
    money_offerlab_report = money_offerlab_subparsers.add_parser("report", help="Report paper money ledger entries for one seller pilot")
    money_offerlab_report.add_argument("pilot_id", metavar="PILOT_ID")
    money_offerlab_report.add_argument("--data-root", help="External seller pilot ledger root")
    money_offerlab_report.add_argument("--money-root", help="Money ledger root for this pilot")
    money_offerlab_report.add_argument("--output")
    money_offerlab_report.set_defaults(func=command_money_offerlab_report)

    money_weather = money_subparsers.add_parser("weather-edge", help="Multicity Weather Edge paper lab")
    money_weather_subparsers = money_weather.add_subparsers(dest="money_weather_edge_command", required=True)
    money_weather_backfill = money_weather_subparsers.add_parser("backfill", help="Run fixture/provider historical Weather Edge backfill")
    money_weather_backfill.add_argument("--fixture", required=True, help="Fixture JSON for a read-only Weather Edge provider")
    money_weather_backfill.add_argument("--storage-root", required=True)
    money_weather_backfill.add_argument("--as-of")
    money_weather_backfill.add_argument("--output")
    money_weather_backfill.set_defaults(func=command_money_weather_edge_backfill)
    money_weather_cycle = money_weather_subparsers.add_parser("paper-cycle", help="Run one paper-only Weather Edge cycle")
    money_weather_cycle.add_argument("--fixture", required=True, help="Fixture JSON for a read-only Weather Edge provider")
    money_weather_cycle.add_argument("--storage-root", required=True)
    money_weather_cycle.add_argument("--as-of", required=True)
    money_weather_cycle.add_argument("--output")
    money_weather_cycle.set_defaults(func=command_money_weather_edge_paper_cycle)
    money_weather_report = money_weather_subparsers.add_parser("report", help="Report Weather Edge paper ledger state")
    money_weather_report.add_argument("--storage-root", required=True)
    money_weather_report.add_argument("--fixture", help="Optional fixture JSON for evidence-gate historical counts")
    money_weather_report.add_argument("--as-of")
    money_weather_report.add_argument("--output")
    money_weather_report.set_defaults(func=command_money_weather_edge_report)

    money_etf = money_subparsers.add_parser("etf-risk", help="Paper-only broad ETF risk lab")
    money_etf_subparsers = money_etf.add_subparsers(dest="money_etf_risk_command", required=True)
    money_etf_backfill = money_etf_subparsers.add_parser("backfill", help="Run ETF Risk walk-forward backfill with a safe fixture provider")
    money_etf_backfill.add_argument("--ledger-path", required=True)
    money_etf_backfill.add_argument("--demo-fixture", action="store_true", help="Use the built-in offline authorized fixture provider")
    money_etf_backfill.add_argument("--session-count", type=_positive, default=90)
    money_etf_backfill.add_argument("--min-history-days", type=_positive, default=35)
    money_etf_backfill.add_argument("--start")
    money_etf_backfill.add_argument("--end")
    money_etf_backfill.add_argument("--output")
    money_etf_backfill.set_defaults(func=command_money_etf_risk_backfill)
    money_etf_cycle = money_etf_subparsers.add_parser("paper-cycle", help="Run one ETF Risk paper-only decision with a safe fixture provider")
    money_etf_cycle.add_argument("--ledger-path", required=True)
    money_etf_cycle.add_argument("--demo-fixture", action="store_true", help="Use the built-in offline authorized fixture provider")
    money_etf_cycle.add_argument("--session-count", type=_positive, default=90)
    money_etf_cycle.add_argument("--min-history-days", type=_positive, default=35)
    money_etf_cycle.add_argument("--decision-cutoff")
    money_etf_cycle.add_argument("--strategy-id")
    money_etf_cycle.add_argument("--output")
    money_etf_cycle.set_defaults(func=command_money_etf_risk_paper_cycle)
    money_etf_report = money_etf_subparsers.add_parser("report", help="Report ETF Risk paper ledger state")
    money_etf_report.add_argument("--ledger-path", required=True)
    money_etf_report.add_argument("--demo-fixture", action="store_true", help="Use the built-in offline authorized fixture provider")
    money_etf_report.add_argument("--session-count", type=_positive, default=90)
    money_etf_report.add_argument("--min-history-days", type=_positive, default=35)
    money_etf_report.add_argument("--start")
    money_etf_report.add_argument("--end")
    money_etf_report.add_argument("--output")
    money_etf_report.set_defaults(func=command_money_etf_risk_report)

    money_wave2 = money_subparsers.add_parser("wave2-integration", help="Generate Finance Wave 2 integration proof reports")
    money_wave2_subparsers = money_wave2.add_subparsers(dest="money_wave2_command", required=True)
    money_wave2_report = money_wave2_subparsers.add_parser("report", help="Write wave_2_integration JSON and HTML reports")
    money_wave2_report.add_argument("--output-dir", default="reports/finance")
    money_wave2_report.add_argument("--workspace")
    money_wave2_report.add_argument("--generated-at", default="2026-07-04T00:00:00+00:00")
    money_wave2_report.set_defaults(func=command_money_wave2_integration_report)

    money_autopilot = money_subparsers.add_parser("autopilot", help="Run the local paper-finance portfolio autopilot")
    money_autopilot.add_argument("--portfolio", default="money-lab.yaml")
    money_autopilot.set_defaults(func=command_money_autopilot)
    money_autopilot_subparsers = money_autopilot.add_subparsers(dest="autopilot_command")

    money_autopilot_status = money_autopilot_subparsers.add_parser("status", help="Show resumable autopilot status")
    money_autopilot_status.add_argument("--portfolio", default="money-lab.yaml")
    money_autopilot_status.set_defaults(func=command_money_autopilot)

    money_autopilot_run_once = money_autopilot_subparsers.add_parser("run-once", help="Run one local paper autopilot cycle")
    money_autopilot_run_once.add_argument("--portfolio", default="money-lab.yaml")
    money_autopilot_run_once.set_defaults(func=command_money_autopilot)

    money_autopilot_approvals = money_autopilot_subparsers.add_parser("approvals", help="List waiting approval items")
    money_autopilot_approvals.add_argument("--portfolio", default="money-lab.yaml")
    money_autopilot_approvals.set_defaults(func=command_money_autopilot)

    money_autopilot_weekly = money_autopilot_subparsers.add_parser("weekly-report", help="Write and print a paper weekly report")
    money_autopilot_weekly.add_argument("--portfolio", default="money-lab.yaml")
    money_autopilot_weekly.set_defaults(func=command_money_autopilot)

    money_autopilot_pause = money_autopilot_subparsers.add_parser("pause", help="Pause one contract")
    money_autopilot_pause.add_argument("contract_id", metavar="CONTRACT_ID")
    money_autopilot_pause.add_argument("--portfolio", default="money-lab.yaml")
    money_autopilot_pause.set_defaults(func=command_money_autopilot)

    money_autopilot_resume = money_autopilot_subparsers.add_parser("resume", help="Resume one contract")
    money_autopilot_resume.add_argument("contract_id", metavar="CONTRACT_ID")
    money_autopilot_resume.add_argument("--portfolio", default="money-lab.yaml")
    money_autopilot_resume.set_defaults(func=command_money_autopilot)

    money_contract_scout = money_subparsers.add_parser("contract-scout", help="Scout paper-only financial decision contracts")
    money_contract_scout_subparsers = money_contract_scout.add_subparsers(dest="contract_scout_command", required=True)

    contract_scout_run = money_contract_scout_subparsers.add_parser("run", help="Run the autonomous paper contract scout")
    contract_scout_run.add_argument("--state-dir", default=".money_contract_scout")
    contract_scout_run.add_argument("--operations-state-dir", help="Optional money operations state directory to inspect active, paused, and blocked contracts")
    contract_scout_run.add_argument("--proposals-json", help="Optional structured proposal list from a bounded research agent")
    contract_scout_run.add_argument("--search-budget", type=int, default=8)
    contract_scout_run.add_argument("--llm-budget-usd", type=float, default=0.0)
    contract_scout_run.add_argument("--no-seed-families", action="store_true")
    contract_scout_run.add_argument("--output")
    contract_scout_run.set_defaults(func=command_money_contract_scout)

    for scout_command in ("proposals", "report"):
        scout_parser = money_contract_scout_subparsers.add_parser(scout_command, help=f"Show contract scout {scout_command}")
        scout_parser.add_argument("--state-dir", default=".money_contract_scout")
        scout_parser.add_argument("--operations-state-dir", help="Optional money operations state directory to inspect active, paused, and blocked contracts")
        scout_parser.add_argument("--output")
        scout_parser.set_defaults(func=command_money_contract_scout)

    contract_scout_approve = money_contract_scout_subparsers.add_parser("approve", help="Approve an eligible proposal for experimental paper portfolio consideration")
    contract_scout_approve.add_argument("proposal_id", metavar="PROPOSAL_ID")
    contract_scout_approve.add_argument("--state-dir", default=".money_contract_scout")
    contract_scout_approve.add_argument("--output")
    contract_scout_approve.set_defaults(func=command_money_contract_scout)

    contract_scout_reject = money_contract_scout_subparsers.add_parser("reject", help="Manually reject a stored contract proposal")
    contract_scout_reject.add_argument("proposal_id", metavar="PROPOSAL_ID")
    contract_scout_reject.add_argument("--state-dir", default=".money_contract_scout")
    contract_scout_reject.add_argument("--reason", default="manual_rejection")
    contract_scout_reject.add_argument("--output")
    contract_scout_reject.set_defaults(func=command_money_contract_scout)

    money_data_mesh = money_subparsers.add_parser("data-mesh", help="Manage the experimental financial data mesh")
    money_data_mesh_subparsers = money_data_mesh.add_subparsers(dest="data_mesh_command", required=True)

    data_mesh_validate = money_data_mesh_subparsers.add_parser("validate-manifest", help="Validate one declarative source manifest")
    data_mesh_validate.add_argument("--state-dir", default=".money_data_mesh")
    data_mesh_validate.add_argument("--manifest", required=True)
    data_mesh_validate.add_argument("--output")
    data_mesh_validate.set_defaults(func=command_money_data_mesh)

    for data_mesh_command in ("trial", "activate"):
        data_mesh_parser = money_data_mesh_subparsers.add_parser(data_mesh_command, help=f"{data_mesh_command} one declarative source manifest with a fixture")
        data_mesh_parser.add_argument("--state-dir", default=".money_data_mesh")
        data_mesh_parser.add_argument("--manifest", required=True)
        data_mesh_parser.add_argument("--fixture", required=True)
        data_mesh_parser.add_argument("--output")
        data_mesh_parser.set_defaults(func=command_money_data_mesh)

    data_mesh_acquire = money_data_mesh_subparsers.add_parser("acquire", help="Acquire missing source families into the experimental catalog")
    data_mesh_acquire.add_argument("--state-dir", default=".money_data_mesh")
    data_mesh_acquire.add_argument("--contracts-json", required=True)
    data_mesh_acquire.add_argument("--manifests-json", required=True)
    data_mesh_acquire.add_argument("--fixtures-json")
    data_mesh_acquire.add_argument("--source-catalog-json")
    data_mesh_acquire.add_argument("--search-budget", type=int, default=8)
    data_mesh_acquire.add_argument("--llm-budget-usd", type=float, default=0.0)
    data_mesh_acquire.add_argument("--output")
    data_mesh_acquire.set_defaults(func=command_money_data_mesh)

    data_mesh_repair = money_data_mesh_subparsers.add_parser("repair", help="Repair or substitute an experimental source version")
    data_mesh_repair.add_argument("source_id", metavar="SOURCE_ID")
    data_mesh_repair.add_argument("--state-dir", default=".money_data_mesh")
    data_mesh_repair.add_argument("--failure-json", required=True)
    data_mesh_repair.add_argument("--candidate-manifest", required=True)
    data_mesh_repair.add_argument("--fixture", required=True)
    data_mesh_repair.add_argument("--output")
    data_mesh_repair.set_defaults(func=command_money_data_mesh)

    data_mesh_backfill = money_data_mesh_subparsers.add_parser("backfill-plan", help="Plan a resumable progressive source backfill")
    data_mesh_backfill.add_argument("source_id", metavar="SOURCE_ID")
    data_mesh_backfill.add_argument("--state-dir", default=".money_data_mesh")
    data_mesh_backfill.add_argument("--start-date", required=True)
    data_mesh_backfill.add_argument("--end-date", required=True)
    data_mesh_backfill.add_argument("--chunk-days", type=_positive, default=7)
    data_mesh_backfill.add_argument("--completed-chunks")
    data_mesh_backfill.add_argument("--output")
    data_mesh_backfill.set_defaults(func=command_money_data_mesh)

    data_mesh_connector = money_data_mesh_subparsers.add_parser("connector-audit", help="Audit generated connector code in sandbox-only mode")
    data_mesh_connector.add_argument("source_id", metavar="SOURCE_ID")
    data_mesh_connector.add_argument("--state-dir", default=".money_data_mesh")
    data_mesh_connector.add_argument("--manifest-hash", required=True)
    data_mesh_connector.add_argument("--code-file", required=True)
    data_mesh_connector.add_argument("--output")
    data_mesh_connector.set_defaults(func=command_money_data_mesh)

    data_mesh_classify = money_data_mesh_subparsers.add_parser("classify", help="Classify experimental source marginal value")
    data_mesh_classify.add_argument("source_id", metavar="SOURCE_ID")
    data_mesh_classify.add_argument("--state-dir", default=".money_data_mesh")
    data_mesh_classify.add_argument("--metrics-json", required=True)
    data_mesh_classify.add_argument("--output")
    data_mesh_classify.set_defaults(func=command_money_data_mesh)

    data_mesh_catalog = money_data_mesh_subparsers.add_parser("catalog", help="List experimental data mesh sources")
    data_mesh_catalog.add_argument("--state-dir", default=".money_data_mesh")
    data_mesh_catalog.add_argument("--output")
    data_mesh_catalog.set_defaults(func=command_money_data_mesh)

    money_canary = money_subparsers.add_parser("canary", help="Manage immutable prospective paper canaries")
    money_canary_subparsers = money_canary.add_subparsers(dest="canary_command", required=True)

    money_canary_start = money_canary_subparsers.add_parser("start", help="Start an immutable prospective paper canary")
    money_canary_start.add_argument("contract_id", metavar="CONTRACT_ID")
    money_canary_start.add_argument("--state-dir", default=".money_canaries")
    money_canary_start.add_argument("--lab", choices=sorted({"offerlab_seller_pilot", "weather_edge", "etf_risk"}))
    money_canary_start.add_argument("--as-of")
    money_canary_start.add_argument("--strategy-version", default="fixture_frozen_v1")
    money_canary_start.add_argument("--source-version", default="fixture_source_v1")
    money_canary_start.add_argument("--seller-pilot-ready", action="store_true")
    money_canary_start.add_argument("--output")
    money_canary_start.set_defaults(func=command_money_canary)

    money_canary_resume = money_canary_subparsers.add_parser("resume", help="Append a daily/weekly canary snapshot")
    money_canary_resume.add_argument("canary_id", metavar="CANARY_ID")
    money_canary_resume.add_argument("--state-dir", default=".money_canaries")
    money_canary_resume.add_argument("--as-of")
    money_canary_resume.add_argument("--strategy-version")
    money_canary_resume.add_argument("--output")
    money_canary_resume.set_defaults(func=command_money_canary)

    money_canary_status = money_canary_subparsers.add_parser("status", help="Show prospective canary status")
    money_canary_status.add_argument("canary_id", metavar="CANARY_ID")
    money_canary_status.add_argument("--state-dir", default=".money_canaries")
    money_canary_status.add_argument("--output")
    money_canary_status.set_defaults(func=command_money_canary)

    money_canary_report = money_canary_subparsers.add_parser("report", help="Render current canary evidence report")
    money_canary_report.add_argument("canary_id", metavar="CANARY_ID")
    money_canary_report.add_argument("--state-dir", default=".money_canaries")
    money_canary_report.add_argument("--output")
    money_canary_report.set_defaults(func=command_money_canary)

    money_canary_invalidate = money_canary_subparsers.add_parser("invalidate", help="Invalidate a canary without rewriting prior snapshots")
    money_canary_invalidate.add_argument("canary_id", metavar="CANARY_ID")
    money_canary_invalidate.add_argument("--state-dir", default=".money_canaries")
    money_canary_invalidate.add_argument("--reason", required=True)
    money_canary_invalidate.add_argument("--as-of")
    money_canary_invalidate.add_argument("--output")
    money_canary_invalidate.set_defaults(func=command_money_canary)

    money_tournament = money_subparsers.add_parser("tournament", help="Run the paper-only financial evidence tournament")
    money_tournament_subparsers = money_tournament.add_subparsers(dest="money_tournament_command", required=True)
    money_tournament_run = money_tournament_subparsers.add_parser("run", help="Write FINANCIAL_TOURNAMENT reports and wedge decision")
    money_tournament_run.add_argument("--output-dir", default="reports/finance")
    money_tournament_run.add_argument("--docs-dir", default="docs/finance")
    money_tournament_run.add_argument("--workspace")
    money_tournament_run.add_argument("--generated-at", default="2026-07-05T12:00:00+00:00")
    money_tournament_run.set_defaults(func=command_money_tournament)

    money_operations = money_subparsers.add_parser("operations", help="Operate the frozen paper/shadow finance release")
    money_operations_subparsers = money_operations.add_subparsers(dest="operations_command", required=True)

    money_operations_start = money_operations_subparsers.add_parser("start", help="Start the frozen financial evidence release")
    money_operations_start.add_argument("--state-dir", default=".money_operations")
    money_operations_start.add_argument("--as-of")
    money_operations_start.add_argument("--seller-readiness-report")
    money_operations_start.add_argument("--release-commit")
    money_operations_start.add_argument("--output")
    money_operations_start.set_defaults(func=command_money_operations)

    for operation_name in ("status", "doctor", "weekly-report", "stop"):
        operation = money_operations_subparsers.add_parser(operation_name, help=f"Run money operations {operation_name}")
        operation.add_argument("--state-dir", default=".money_operations")
        operation.add_argument("--output")
        operation.set_defaults(func=command_money_operations)

    money_operations_recover = money_operations_subparsers.add_parser("recover", help="Recover missed paper/shadow cycles after restart")
    money_operations_recover.add_argument("--state-dir", default=".money_operations")
    money_operations_recover.add_argument("--as-of")
    money_operations_recover.add_argument("--output")
    money_operations_recover.set_defaults(func=command_money_operations)

    benchmark_parser = subparsers.add_parser("benchmark", help="Federated benchmark utilities")
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_command", required=True)
    benchmark_manifest = benchmark_subparsers.add_parser("validate-manifest", help="Validate a benchmark manifest JSON file")
    benchmark_manifest.add_argument("--input", required=True)
    benchmark_manifest.set_defaults(func=command_benchmark_validate_manifest)

    nber = subparsers.add_parser("nber-best-offer", help="NBER eBay Best Offer benchmark tools")
    nber_subparsers = nber.add_subparsers(dest="nber_command", required=True)

    nber_fetch = nber_subparsers.add_parser("fetch", help="Record or explicitly download NBER Best Offer data")
    nber_fetch.add_argument("--cache-dir", default=".dataset_cache")
    nber_fetch.add_argument("--codebook", action="store_true", help="Record codebook/source discovery without full download")
    nber_fetch.add_argument("--full", action="store_true", help="Explicitly download a full official source file")
    nber_fetch.add_argument("--url", help="Official NBER file URL for --full")
    nber_fetch.set_defaults(func=command_nber_fetch)

    nber_inventory = nber_subparsers.add_parser("inventory", help="Inventory a CSV or CSV.GZ file")
    nber_inventory.add_argument("--input", required=True)
    nber_inventory.set_defaults(func=command_nber_inventory)

    nber_source_inventory = nber_subparsers.add_parser("source-inventory", help="Inventory official NBER source files without normalization")
    nber_source_inventory.add_argument("--raw-dir", default=None, help="Defaults to OFFERLAB_DATA_ROOT/raw/nber_best_offer")
    nber_source_inventory.add_argument("--download", action="store_true", help="Download missing official files before inventory")
    nber_source_inventory.add_argument("--sample-dir", help="Optional external directory for redacted samples")
    nber_source_inventory.add_argument("--reservoir-rows", type=_nonnegative, default=10_000)
    nber_source_inventory.add_argument("--write-report", action="store_true", help="Write committed metadata report paths; does not download unless --download is supplied")
    nber_source_inventory.add_argument("--manifest", default="datasets/manifests/nber_best_offer_downloads.yaml")
    nber_source_inventory.add_argument("--doc", default="docs/runs/NBER_SOURCE_INVENTORY.md")
    nber_source_inventory.add_argument("--first-sample-rows", type=_positive, default=100)
    nber_source_inventory.add_argument("--chronological-rows-per-slice", type=_positive, default=100)
    nber_source_inventory.add_argument("--timeout-seconds", type=_positive, default=120)
    nber_source_inventory.set_defaults(func=command_nber_source_inventory)

    nber_sample = nber_subparsers.add_parser("build-sample", help="Build a tiny deterministic NBER-format sample")
    nber_sample.add_argument("--output-dir", default="runs/nber_sample/raw")
    nber_sample.set_defaults(func=command_nber_build_sample)

    nber_normalize = nber_subparsers.add_parser("normalize", help="Normalize NBER CSV/CSV.GZ files into partitioned JSONL")
    nber_normalize.add_argument("--input-dir", required=True)
    nber_normalize.add_argument("--output-dir", required=True)
    nber_normalize.set_defaults(func=command_nber_normalize)

    nber_inspect_schema = nber_subparsers.add_parser("inspect-schema", help="Inspect official NBER real-source schema")
    nber_inspect_schema.add_argument("--codebook", help="Optional path to Codebook.xlsx")
    nber_inspect_schema.add_argument("--raw-dir", help="Optional raw directory to validate actual CSV headers")
    nber_inspect_schema.set_defaults(func=command_nber_inspect_schema)

    nber_normalize_real = nber_subparsers.add_parser("normalize-real", help="Normalize official NBER real source with thread-linked listing extraction")
    nber_normalize_real.add_argument("--raw-dir", default=None, help="Defaults to OFFERLAB_DATA_ROOT/raw/nber_best_offer")
    nber_normalize_real.add_argument("--output-dir", default=None, help="Defaults to OFFERLAB_DATA_ROOT/processed/nber_best_offer_full")
    nber_normalize_real.add_argument("--limit-threads", type=_positive)
    nber_normalize_real.add_argument("--full", action="store_true")
    nber_normalize_real.add_argument("--resume", action="store_true", help="Reuse verified partition checkpoints and completed output files")
    nber_normalize_real.add_argument("--bucket-count", type=_positive, default=32)
    nber_normalize_real.add_argument("--partition-rows", type=_positive, default=50_000)
    nber_normalize_real.add_argument("--seed", type=int, default=20240621)
    nber_normalize_real.add_argument("--stop-after-thread-pass", action="store_true", help=argparse.SUPPRESS)
    nber_normalize_real.set_defaults(func=command_nber_normalize_real)

    nber_full_status = nber_subparsers.add_parser("full-status", help="Report full NBER normalization progress, checkpoints, and manifest integrity")
    nber_full_status.add_argument("--output-dir", default=None, help="Defaults to OFFERLAB_DATA_ROOT/processed/nber_best_offer_full")
    nber_full_status.set_defaults(func=command_nber_full_status)

    nber_replication = nber_subparsers.add_parser("replication-check", help="Validate or run the frozen NBER replication contract")
    nber_replication.add_argument("--normalized-dir", help="Run checks against a normalized real-source manifest")
    nber_replication.add_argument("--targets", help="Optional replication target manifest")
    nber_replication.set_defaults(func=command_nber_replication_check)

    nber_bench = nber_subparsers.add_parser("benchmark", help="Run leakage-safe baseline leaderboards")
    nber_bench.add_argument("--normalized-dir", required=True)
    nber_bench.set_defaults(func=command_nber_benchmark)

    nber_audit_parser = nber_subparsers.add_parser("audit", help="Run NBER adversarial audit checks")
    nber_audit_parser.add_argument("--normalized-dir", required=True)
    nber_audit_parser.add_argument("--output")
    nber_audit_parser.set_defaults(func=command_nber_audit)

    suite = subparsers.add_parser("benchmark-suite", help="Run wider-net validation suite smoke checks")
    suite_subparsers = suite.add_subparsers(dest="suite_command", required=True)
    suite_run = suite_subparsers.add_parser("run", help="Run Open Bandit and Criteo smoke benchmarks")
    suite_run.set_defaults(func=command_benchmark_suite_run)
    suite_report = suite_subparsers.add_parser("report", help="Alias for run")
    suite_report.set_defaults(func=command_benchmark_suite_run)
    suite_permissions = suite_subparsers.add_parser("permissions", help="Show cross-dataset production-export permissions")
    suite_permissions.set_defaults(func=command_benchmark_suite_permissions)

    template = subparsers.add_parser("campaign-001-template", help="Write a raw manual-entry template for Campaign 001")
    template.add_argument("--output", default="campaigns/campaign_001_task_initiation/manual_entry_template.json")
    template.set_defaults(func=command_campaign_template)

    bridge_hash = subparsers.add_parser("bridge-hash", help="Add source_hash values to raw Behavior Lab snapshot exports")
    bridge_hash.add_argument("--input", required=True)
    bridge_hash.add_argument("--output", required=True)
    bridge_hash.set_defaults(func=command_bridge_hash)

    bridge_validate = subparsers.add_parser("bridge-validate", help="Validate immutable Behavior Lab campaign snapshots")
    bridge_validate.add_argument("--input", required=True)
    bridge_validate.add_argument("--campaign-id", default=CAMPAIGN_001_ID)
    bridge_validate.set_defaults(func=command_bridge_validate)

    bridge_import = subparsers.add_parser("bridge-import", help="Import validated Behavior Lab snapshots into an append-only ledger")
    bridge_import.add_argument("--input", required=True)
    bridge_import.add_argument("--data-dir", required=True)
    bridge_import.add_argument("--campaign-id", default=CAMPAIGN_001_ID)
    bridge_import.set_defaults(func=command_bridge_import)

    capture = subparsers.add_parser("campaign-001-capture", help="Local Campaign 001 episode collector")
    capture_subparsers = capture.add_subparsers(dest="capture_command", required=True)

    capture_start = capture_subparsers.add_parser("start", help="Seal a pre-decision Campaign 001 episode")
    capture_start.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    capture_start.add_argument("--script", help="JSON object for deterministic/manual-free capture")
    capture_start.add_argument("--pilot", action="store_true", help="Mark this episode as part of the five-episode pilot")
    capture_start.set_defaults(func=command_campaign_001_capture_start)

    capture_finalize = capture_subparsers.add_parser("finalize", help="Finalize outcomes and import a bridge export")
    capture_finalize.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    capture_finalize.add_argument("--episode-id", required=True)
    capture_finalize.add_argument("--script", help="JSON object containing protected outcomes")
    capture_finalize.set_defaults(func=command_campaign_001_capture_finalize)

    capture_resume = capture_subparsers.add_parser("resume", help="List or finalize resumable local episodes")
    capture_resume.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    capture_resume.add_argument("--episode-id")
    capture_resume.add_argument("--script", help="JSON object containing protected outcomes")
    capture_resume.set_defaults(func=command_campaign_001_capture_resume)

    capture_missed = capture_subparsers.add_parser("missed", help="Record an eligible task missed before capture")
    capture_missed.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    capture_missed.add_argument("--script", help="JSON object describing the missed eligible task")
    capture_missed.add_argument("--pilot", action="store_true", help="Mark this missed episode as part of the pilot")
    capture_missed.set_defaults(func=command_campaign_001_capture_missed)

    capture_status = capture_subparsers.add_parser("status", help="Show operational collector status only")
    capture_status.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    capture_status.set_defaults(func=command_campaign_001_capture_status)

    capture_amend = capture_subparsers.add_parser("amend", help="Append a correction note without changing sealed data")
    capture_amend.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    capture_amend.add_argument("--episode-id", required=True)
    capture_amend.add_argument("--field", required=True)
    capture_amend.add_argument("--value", required=True)
    capture_amend.add_argument("--reason", required=True)
    capture_amend.set_defaults(func=command_campaign_001_capture_amend)

    capture_invalidate = capture_subparsers.add_parser("invalidate", help="Invalidate an unfinished local capture")
    capture_invalidate.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    capture_invalidate.add_argument("--episode-id", required=True)
    capture_invalidate.add_argument("--reason", required=True)
    capture_invalidate.set_defaults(func=command_campaign_001_capture_invalidate)

    offer_template = subparsers.add_parser("offerlab-template", help="Write a Campaign 002 eBay offer snapshot template")
    offer_template.add_argument("--output", default="campaigns/campaign_002_ebay_seller_offers/examples/offer_snapshot_template.json")
    offer_template.set_defaults(func=command_offerlab_template)

    offer_ingest = subparsers.add_parser("offerlab-ingest", help="Ingest normalized read-only eBay offer snapshots")
    offer_ingest.add_argument("--input", required=True)
    offer_ingest.add_argument("--data-dir", default="data/campaign_002_ebay_seller_offers")
    offer_ingest.set_defaults(func=command_offerlab_ingest)

    offer_audit = subparsers.add_parser("offerlab-audit", help="Summarize realized margin from ingested OfferLab history")
    offer_audit.add_argument("--data-dir", default="data/campaign_002_ebay_seller_offers")
    offer_audit.set_defaults(func=command_offerlab_audit)

    offer_report = subparsers.add_parser("offerlab-report", help="Write the read-only OfferLab profit-audit report")
    offer_report.add_argument("--data-dir", default="data/campaign_002_ebay_seller_offers")
    offer_report.add_argument("--output", help="Optional .md or .json report path")
    offer_report.set_defaults(func=command_offerlab_report)

    offer_recommend = subparsers.add_parser("offerlab-recommend", help="Read-only economic recommendation for one offer snapshot")
    offer_recommend.add_argument("--input", required=True)
    offer_recommend.add_argument("--data-dir", default=None)
    offer_recommend.add_argument("--config", help="Optional JSON economics config with fee, holding cost, and return risk")
    offer_recommend.set_defaults(func=command_offerlab_recommend)

    offer_pilot = subparsers.add_parser("offerlab-pilot", help="Local-only seller pilot import and audit kit")
    offer_pilot_subparsers = offer_pilot.add_subparsers(dest="offerlab_pilot_command", required=True)

    offer_pilot_template = offer_pilot_subparsers.add_parser("template", help="Write seller CSV templates and an explicit column manifest")
    offer_pilot_template.add_argument("--output-dir", help="Template output directory; defaults outside the repository")
    offer_pilot_template.set_defaults(func=command_offerlab_pilot_template)

    offer_pilot_inspect = offer_pilot_subparsers.add_parser("inspect", help="Inspect seller pilot files without importing them")
    offer_pilot_inspect.add_argument("input_dir", metavar="INPUT_DIR")
    offer_pilot_inspect.set_defaults(func=command_offerlab_pilot_inspect)

    offer_pilot_onboard = offer_pilot_subparsers.add_parser("onboard", help="Guide local seller export mapping and readiness checks")
    offer_pilot_onboard.add_argument("input_dir", metavar="INPUT_DIR")
    offer_pilot_onboard.add_argument("--output", help="Optional JSON output path for the data-readiness report")
    offer_pilot_onboard.set_defaults(func=command_offerlab_pilot_onboard)

    offer_pilot_import = offer_pilot_subparsers.add_parser("import", help="Import seller pilot files into a local external ledger")
    offer_pilot_import.add_argument("input_dir", metavar="INPUT_DIR")
    offer_pilot_import.add_argument("--data-root", help="External seller pilot ledger root; defaults to C:\\OfferLabData\\seller_pilots")
    offer_pilot_import.add_argument("--pilot-id", help="Override pilot_id from the manifest")
    offer_pilot_import.set_defaults(func=command_offerlab_pilot_import)

    offer_pilot_audit = offer_pilot_subparsers.add_parser("audit", help="Audit the latest imported version for one seller pilot")
    offer_pilot_audit.add_argument("pilot_id", metavar="PILOT_ID")
    offer_pilot_audit.add_argument("--data-root", help="External seller pilot ledger root; defaults to C:\\OfferLabData\\seller_pilots")
    offer_pilot_audit.set_defaults(func=command_offerlab_pilot_audit)
    offer_pilot_shadow = offer_pilot_subparsers.add_parser("shadow-report", help="Build an isolated read-only seller-pilot shadow report")
    offer_pilot_shadow.add_argument("pilot_id", metavar="PILOT_ID")
    offer_pilot_shadow.add_argument("--data-root", help="External seller pilot ledger root; defaults to C:\\OfferLabData\\seller_pilots")
    offer_pilot_shadow.add_argument("--output", help="Optional JSON output path for the shadow report")
    offer_pilot_shadow.set_defaults(func=command_offerlab_pilot_shadow_report)

    offer_models = subparsers.add_parser("offerlab-models", help="Run research-only OfferLab model leaderboards")
    offer_models_subparsers = offer_models.add_subparsers(dest="offerlab_models_command", required=True)
    offer_models_sample = offer_models_subparsers.add_parser("sample", help="Run the deterministic NBER-format sample model suite")
    offer_models_sample.set_defaults(func=command_offerlab_models_sample)
    offer_models_benchmark = offer_models_subparsers.add_parser("benchmark-v1", help="Retired: Benchmark v1 is frozen and hidden-spent")
    offer_models_benchmark.add_argument("--normalized-dir", required=True)
    offer_models_benchmark.add_argument("--output", default="reports/offerlab_benchmark_v1.json")
    offer_models_benchmark.add_argument("--doc", default="docs/runs/OFFERLAB_BENCHMARK_V1_RESULTS.md")
    offer_models_benchmark.add_argument("--model-cards-dir", default="docs/model_cards/offerlab_benchmark_v1")
    offer_models_benchmark.add_argument("--protocol", default="datasets/manifests/offerlab_benchmark_v1.yaml")
    offer_models_benchmark.add_argument("--lockbox-store", required=True, help="External durable JSONL event store for one-shot hidden submissions")
    offer_models_benchmark.add_argument("--row-cap", type=_positive, default=500)
    offer_models_benchmark.add_argument("--seed", type=int, default=20240621)
    offer_models_benchmark.set_defaults(func=command_offerlab_models_benchmark_v1)
    offer_models_benchmark_v2 = offer_models_subparsers.add_parser("benchmark-v2-build", help="Build Benchmark v2 tasks and split manifests without training")
    offer_models_benchmark_v2.add_argument("--normalized-dir", required=True)
    offer_models_benchmark_v2.add_argument("--output-dir", required=True)
    offer_models_benchmark_v2.add_argument("--protocol", default="datasets/manifests/offerlab_benchmark_v2.yaml")
    offer_models_benchmark_v2.add_argument("--v1-final-manifest", default="reports/offerlab_benchmark_v1_final_manifest.json")
    offer_models_benchmark_v2.add_argument("--external-v1-hidden-tokens", required=True)
    offer_models_benchmark_v2.add_argument("--partition-rows", type=_positive, default=50_000)
    offer_models_benchmark_v2.add_argument("--allow-bounded-test-input", action="store_true", help=argparse.SUPPRESS)
    offer_models_benchmark_v2.set_defaults(func=command_offerlab_models_benchmark_v2_build)
    offer_models_benchmark_v2_runner = offer_models_subparsers.add_parser("benchmark-v2", help="Run Benchmark v2 pre-hidden development model runner")
    offer_models_benchmark_v2_runner.add_argument("--normalized-dir", required=True)
    offer_models_benchmark_v2_runner.add_argument("--output", default="reports/offerlab_benchmark_v2_pre_hidden.json")
    offer_models_benchmark_v2_runner.add_argument("--doc", default="docs/runs/OFFERLAB_BENCHMARK_V2_PRE_HIDDEN.md")
    offer_models_benchmark_v2_runner.add_argument("--model-cards-dir", default="docs/model_cards/offerlab_benchmark_v2")
    offer_models_benchmark_v2_runner.add_argument("--protocol", default="datasets/manifests/offerlab_benchmark_v2.yaml")
    offer_models_benchmark_v2_runner.add_argument("--batch-size", type=_positive, default=10_000)
    offer_models_benchmark_v2_runner.add_argument("--submit-hidden", action="store_true")
    offer_models_benchmark_v2_runner.set_defaults(func=command_offerlab_models_benchmark_v2)
    offer_models_benchmark_v2_integrate = offer_models_subparsers.add_parser("benchmark-v2-integrate", help="Run the Benchmark v2 integration gate and preregistration artifact")
    offer_models_benchmark_v2_integrate.add_argument("--normalized-dir", required=True)
    offer_models_benchmark_v2_integrate.add_argument("--benchmark-dir", required=True)
    offer_models_benchmark_v2_integrate.add_argument("--output", default="reports/offerlab_benchmark_v2.json")
    offer_models_benchmark_v2_integrate.add_argument("--preregistration", default="reports/offerlab_benchmark_v2_preregistration.json")
    offer_models_benchmark_v2_integrate.add_argument("--pre-hidden-output", default="reports/offerlab_benchmark_v2_pre_hidden.json")
    offer_models_benchmark_v2_integrate.add_argument("--doc", default="docs/runs/OFFERLAB_BENCHMARK_V2_INTEGRATION.md")
    offer_models_benchmark_v2_integrate.add_argument("--pre-hidden-doc", default="docs/runs/OFFERLAB_BENCHMARK_V2_PRE_HIDDEN.md")
    offer_models_benchmark_v2_integrate.add_argument("--model-cards-dir", default="docs/model_cards/offerlab_benchmark_v2")
    offer_models_benchmark_v2_integrate.add_argument("--protocol", default="datasets/manifests/offerlab_benchmark_v2.yaml")
    offer_models_benchmark_v2_integrate.add_argument("--v1-final-manifest", default="reports/offerlab_benchmark_v1_final_manifest.json")
    offer_models_benchmark_v2_integrate.add_argument("--external-v1-hidden-tokens")
    offer_models_benchmark_v2_integrate.add_argument("--partition-rows", type=_positive, default=50_000)
    offer_models_benchmark_v2_integrate.add_argument("--batch-size", type=_positive, default=10_000)
    offer_models_benchmark_v2_integrate.add_argument("--submit-hidden", action="store_true")
    offer_models_benchmark_v2_integrate.add_argument("--allow-bounded-test-input", action="store_true", help=argparse.SUPPRESS)
    offer_models_benchmark_v2_integrate.set_defaults(func=command_offerlab_models_benchmark_v2_integrate)

    demo = subparsers.add_parser("demo", help="Run all waves end-to-end with campaign-safe lockboxes")
    demo.add_argument("--data-dir", default=".demo")
    demo.add_argument("--world", default="habit")
    demo.add_argument("--episodes", type=_positive, default=180)
    demo.add_argument("--iterations", type=_positive, default=3)
    demo.add_argument("--offline-trials", type=_positive, default=8)
    demo.add_argument("--prospective-episodes", type=_nonnegative, default=40)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--reset", action=argparse.BooleanOptionalAction, default=True)
    demo.set_defaults(func=command_demo)
    return parser


def main(argv: list[str] | None = None) -> None:
    raw_args = sys.argv[1:] if argv is None else list(argv)
    if len(raw_args) >= 2 and raw_args[0] == "offerlab-models" and raw_args[1] == "benchmark-v1":
        raise SystemExit(
            "OfferLab Benchmark v1 is frozen and hidden-spent. "
            "Do not rerun it; create Benchmark v2 with fresh hidden cases instead."
        )
    parser = build_parser()
    args = parser.parse_args(raw_args)
    args.func(args)


def _parse_csv_strings(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("CSV list may not be empty")
    return items


def _parse_csv_ints(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("CSV integer list may not be empty")
    return items


if __name__ == "__main__":
    main()
