# FINANCE_OPERATIONS_RELEASE Audit

Verdict: PASS

Audited commit: 602090a15162264eac5adc357cc0a668bdf6ca76

## Retested Prior Blockers

- Forged OfferLab seller readiness report with `readiness_gate.passed=True`,
  `canary_start_allowed=True`, valid fees/shipping/orders, and invalid
  `cost_basis.unit_cost_amount` did not start the seller canary. Operations
  recorded `seller_readiness.passed=False` and the seller canary manifest entry
  stayed `status=blocked`.
- Invalidated Weather Edge canary with elapsed minimum duration kept
  `final_evidence_report.available=False` and `real_money_allowed=False`.

## Commands Run

```powershell
python -m pytest tests/audit/test_finance_operations_release_regressions.py -q
python -m pytest tests/money/test_operations.py tests/money/test_canary.py tests/test_offerlab_pilot_onboard.py -q
$env:PYTHONPATH='src'; python -m behavior_lab.cli money operations start --state-dir "$env:TEMP\bdl_cli_audit_ops_src" --as-of 2026-07-01T12:00:00+00:00
$env:PYTHONPATH='src'; python -m behavior_lab.cli money operations status --state-dir "$env:TEMP\bdl_cli_audit_ops_src"
$env:PYTHONPATH='src'; python -m behavior_lab.cli money operations doctor --state-dir "$env:TEMP\bdl_cli_audit_ops_src"
$env:PYTHONPATH='src'; python -m behavior_lab.cli money operations weekly-report --state-dir "$env:TEMP\bdl_cli_audit_ops_src"
$env:PYTHONPATH='src'; python -m behavior_lab.cli money operations recover --state-dir "$env:TEMP\bdl_cli_audit_ops_src" --as-of 2026-07-09T12:00:00+00:00
$env:PYTHONPATH='src'; python -m behavior_lab.cli money operations stop --state-dir "$env:TEMP\bdl_cli_audit_ops_src"
python -m pytest -q
```

## Notes

`doctor` reports the intentionally blocked seller canary as a `missing_canary`
error when no seller readiness report is provided. This did not block Weather
Edge or ETF Risk recovery: `recover` resumed both active paper canaries and
skipped the seller canary as `canary_not_started`.
