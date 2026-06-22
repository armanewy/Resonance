# Experimental Financial Data Mesh

The data mesh is the Wave 2 acquisition substrate for paper-only financial research.
It does not register production sources, write production observations, place trades,
mutate seller accounts, accept licenses, or allocate money.

## Boundaries

- Declarative manifests are tried before generated code.
- Passing sources enter only the experimental catalog.
- Generated connector candidates are audited as sandbox-only artifacts.
- Parent environment inheritance and production database writes are prohibited.
- Unclear licensing, credentials, generated-code requirements, and production
  activation requests block activation.
- Every validation, trial, activation, repair, backfill plan, and value
  classification is appended to `data_mesh.jsonl`.

## Supported Manifest Shapes

The validator recognizes these generic public-data shapes:

- JSON and CSV APIs
- Static timestamped public files
- Socrata
- CKAN
- ArcGIS FeatureServer
- SDMX
- RSS/Atom
- GeoJSON
- GTFS and GTFS-Realtime

Each manifest must declare the official publisher, endpoint, pagination, event
and availability timestamps, timezone, units, geography, cadence, revision
behavior, missing-value behavior, license, rate limits, normalized series, and
quality checks.

## CLI

```powershell
python -m behavior_lab.cli money data-mesh validate-manifest --manifest manifest.json
python -m behavior_lab.cli money data-mesh activate --manifest manifest.json --fixture fixture.json
python -m behavior_lab.cli money data-mesh catalog
```

The output always reports `production_source_activation: false` or
`production_state_mutated: false` for operations that might otherwise be confused
with production registration.
