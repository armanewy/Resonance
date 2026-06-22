# Financial Data Mesh 681c04e Audit

Verdict: PASS after audit fixes.

Commit audited: `681c04e5fa8b86e788abc84877fc54e0d9a6ee79`

Feature: Wave 2 self-expanding experimental financial data mesh.

Scope:

- `src/behavior_lab/finance_data/data_mesh.py`
- `src/behavior_lab/finance_data/__init__.py`
- `src/behavior_lab/cli.py` data-mesh commands
- `tests/finance_data/test_data_mesh.py`
- `docs/finance/FINANCIAL_DATA_MESH.md`

## Findings

### FDM-681C04E-001: P1, fixed

`FinancialDataMesh.repair_source()` coerced `candidate_manifest` to `DeclarativeSourceManifest` before calling activation. In the audited commit this happened at `src/behavior_lab/finance_data/data_mesh.py:445-446`. Unknown nested authority fields could be dropped by dataclass coercion, bypass `_raw_manifest_rejection_reasons()`, and allow an experimental repair switch that the raw manifest validator would have rejected.

Fix: repair candidates now stay on the raw-manifest activation path, so nested authority fields are rejected before coercion. Regression coverage was added in `tests/finance_data/test_data_mesh.py:190`.

### FDM-681C04E-002: P1, fixed

Malformed manifests and malformed feed fixtures could escape the append-only audit trail. `trial_manifest()` did not catch manifest coercion errors, and `_run_fixture_trial()` let adapter parser exceptions, such as malformed RSS/XML, propagate instead of recording a failed trial. This violated the requirement that validation and fixture trials be fail-closed and auditable.

Fix: malformed manifests append `data_mesh_manifest_trial_blocked`, malformed fixture parsing returns a failed trial, and trial payloads now include retrieval/parser provenance without generated-code execution. Regression coverage was added in `tests/finance_data/test_data_mesh.py:207` and `tests/finance_data/test_data_mesh.py:219`.

### FDM-681C04E-003: P2, fixed

Secret detection and redaction were not aligned for secret-shaped keys. A payload such as `{"api_key": "plain-secret-value"}` was not reliably classified as a secret exposure and could preserve the value in audit payloads.

Fix: secret detection now checks both keys and values, and redaction replaces values under secret-shaped keys. Regression coverage was added in `tests/finance_data/test_data_mesh.py:177`.

### FDM-681C04E-004: P1, fixed

Generated connector audit rejected obvious trading and environment markers but could accept generated code with file, database, network, or dynamic-execution capabilities. Because accepted connector audit payloads are used as a sandbox-only safety signal, these capabilities must fail closed.

Fix: generated connector static markers now reject common file, DB, network, parent-environment, and dynamic-execution access patterns. Regression coverage was added in `tests/finance_data/test_data_mesh.py:267`.

### FDM-681C04E-005: P3, residual

The `catalog` payload includes `state_dir` as an absolute local path from `FinancialDataMesh.catalog()` (`src/behavior_lab/finance_data/data_mesh.py:586` after fixes). This can expose a local filesystem path through CLI output. It does not expose credentials and does not activate sources or mutate production state, so it is not a correctness blocker.

## Audit Questions

1. Experimental-only boundaries: PASS after fixes. Activation, acquisition, repair, backfill, connector audit, and value classification remain experimental/paper-only and set production flags false.
2. Authority fields silently dropped during coercion: PASS after fix. Repair candidates no longer bypass raw manifest rejection.
3. Unclear licenses, credentials, generated connectors, ambiguous timestamps, unbounded rate limits, current-only revisions: PASS. These reject or require approval and do not activate.
4. Append-only and auditable operations: PASS after fix. Malformed trial inputs now append blocked/failed records instead of raising outside the store.
5. Generic adapter parser fail-closed/provenance: PASS after fix. Malformed fixtures fail trial and preserve parser/retrieval provenance without raw fixture paths.
6. Generated connector sandbox-only: PASS after fix. Malicious file/DB/network/environment access is rejected and no connector code is executed.
7. Source repair preserves versions and avoids non-experimental switching: PASS after fix. Repairs append diagnosis and repair events, preserve old/new metadata, and do not switch outside the experimental catalog.
8. CLI local paths/secrets: PASS with P3 residual. Secrets are redacted; `catalog` still reports a local `state_dir` path.
9. Wave 2 required tests: PASS after added adversarial tests. Coverage includes manifest-only activation, schema drift repair, dead source substitution, progressive backfill, rate limiting, revision leakage, ambiguous timestamps, unclear license, malicious generated connector, redundant pruning, nested repair authority, malformed trial, malformed parser, and secret-key redaction.
10. Full suite: PASS.

## Tests Run

- `python -m pytest tests\finance_data\test_data_mesh.py -q`: 18 passed.
- `python -m pytest tests\finance_data -q`: 26 passed.
- `python -m pytest -q`: passed with exit code 0; `python -m pytest --collect-only -q` counted 432 tests.
