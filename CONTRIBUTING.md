# Contributing

## Tests

From the repository root:

```bash
python3 -m unittest discover -s tests
```

Stdlib only. No extra install. Add a test next to the code you change. The suite must stay green.

## Honesty rules

These are enforced in code and in tests. Keep them.

1. **Never present an estimate as a measurement.** Token fields from the file are measured. Character-count stand-ins are estimated. A dollar figure prints only when the file had both a model id and token counts (`CostBreakdown.priced`). Unknown stays unknown.
2. **Never let a partial scan render as clean.** Unreadable roots, skipped files, malformed lines, a hit of the 20-second parse budget, or a failed pipeline stage produce `PARTIAL SCAN — NOT A CLEAN RESULT`. That verdict wins over clusters.
3. **Degrade to a labeled unknown rather than crash.** Missing directories, permission errors, and broken JSONL become NOT COUNTED. Exit 0 unless argparse itself rejects the flags.

Repetition is distinct sessions, not turns. Default `min_runs` stays 3 unless a caller passes another value.

## New harness support

A new agent is not a label change. It needs:

1. A root under `_HARNESS_ROOTS` in `gh/discover.py`.
2. Parse coverage for that file shape in `gh/parse.py`.
3. Fixtures under `tests/fixtures/` (see `claude_clean.jsonl`, `codex_clean.jsonl`, and `tests/fixtures/cursor_home/`).
4. Tests that discover, parse, and render against those fixtures, including the empty-and-broken cases.

Keep `resources/python/gh/` in sync with `gh/` when you change library code. The Play loads the copy under `resources/`.
