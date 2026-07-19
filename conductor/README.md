# Canonical conductor system

The v6 conductor has one active registry: `conductor/manifest.json`.

## Stable tracks

| ID | Scope |
|---|---|
| C00 | repository intake, mapping, and agent bootstrap |
| C01 | directional EVoP contract and conformance |
| C02 | Perspective Acceptability Frontier |
| C03 | evidence, model validation, and case contracts |
| C04 | case suite, housing exemplar, and regime discovery |
| C05 | pipeline reproducibility and artifact governance |
| C06 | preprint and publication strategy |
| C07 | `vop_poc_nz` modernisation |
| C08 | `voiage` productionisation |
| C09 | cross-repository boundaries and adapters |
| C10 | documentation, release truth, and supply chain |
| C11 | performance and backend decisions |
| C12 | GitHub project, issues, and agent operations |

Dependencies are defined in the manifest, not inferred from track filenames.

## Local state

Initialise:

```bash
python scripts/track_state.py /path/to/repo --pack-root /path/to/pack --init
python scripts/conductor_status.py /path/to/repo --pack-root /path/to/pack
```

Start a track:

```bash
python scripts/track_state.py /path/to/repo \
  --pack-root /path/to/pack \
  --track C01 \
  --status in_progress \
  --note "Reconcile live EVoP implementation"
```

Complete a track only with evidence:

```bash
python scripts/track_state.py /path/to/repo \
  --pack-root /path/to/pack \
  --track C01 \
  --status completed \
  --evidence tests/perspective/test_conformance_v1.py \
  --commit <sha>
```

## Registry validation

```bash
python scripts/conductor_registry.py /path/to/pack
python scripts/issue_registry.py /path/to/pack --check
```

Regenerate issue Markdown only when intentionally changing the canonical backlog:

```bash
python scripts/issue_registry.py /path/to/pack --clean
```

The validator rejects duplicate IDs/titles/files, unknown dependencies, and dependency cycles.

## Legacy material

`conductor/legacy_track_map.json` records old v1–v5 filenames and titles. Legacy tracks may be archived locally after their decisions and evidence are mapped; they are not active in v6.
