# Server-retirement archives

The final results remain directly readable in `../h200_memory_tiles/`.
These compressed archives additionally preserve files that were originally
excluded from Git because they were temporary or superseded:

- `completed_run_diagnostics.tar.gz`: all 312 worker logs, all 312 exact job
  configurations, and the completed run's console log.
- `superseded_latency_run.tar.gz`: the stopped, earlier latency-only run,
  including its partial results and source snapshots. This is historical data,
  not part of the final 312-case experiment.
- `validation_runs.tar.gz`: the environment-check script, benchmark smoke and
  pilot runs, memory/tile smoke runs, and the resource-limit validation records
  that were stored under `/tmp`.

Archive contents were read back successfully before committing. The archive
paths preserve their original experiment names; absolute server paths inside
job JSON files are historical provenance, not portable execution paths.

`../artifact_inventory.json` records SHA-256 hashes and byte counts for the
preserved result artifacts, documentation, scripts, and dependency configuration.
It excludes itself and the intentionally uncommitted `AGENTS.md` deletion.
