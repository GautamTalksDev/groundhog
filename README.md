# groundhog

Groundhog reads your local Claude Code / Codex session history and tells you which chores you keep paying to redo — with evidence.

**One command (Play):**

```bash
rote play run https://play.modiqo.ai/<owner>/groundhog
```

**Or the CLI:**

```bash
python3 groundhog.py
```

## Privacy

| | |
|---|---|
| **Reads** | Local session files under your home directory only |
| **Writes** | Nothing, unless you pass `--out PATH` (CLI) — the Play declares no writes |
| **Sends** | Nothing — zero network, no accounts, no uploads |

Local only. Reads, never writes (by default), never sends.

See [PLAY.md](./PLAY.md) for the stranger-facing Play card.

## Supported harnesses

| Tool | Locations checked |
|------|-------------------|
| Claude Code | `~/.claude/projects/` |
| Codex | `~/.codex/sessions/`, `~/.codex/history/` |

## What it can't see

- Chat UIs that don't write JSONL session files locally
- Sessions older than `--days` (default 14)
- Token counts the harness never recorded (those costs are labeled **estimated** or **unknown**, never presented as measured)
- Anything outside the paths above
- Codex/Claude installs on another machine or OS user

The **NOT COUNTED** section of the report always lists what was missing, skipped, or estimated.

## Options (CLI)

| Flag | Default | Description |
|------|---------|-------------|
| `--days` | `14` | Look-back window in days |
| `--format` | `text` | `text` or `json` |
| `--out` | stdout | Write report to this path |
| `--top` | `3` | Number of top chores to show |
| `--min-runs` | `3` | Minimum repeats to surface a chore |
| `--redact` / `--no-redact` | redact on | Scrub secret-like strings; truncate evidence |
| `--verbose` | off | Progress on stderr |

## Requirements

Python **3.8+**, standard library only. No `pip install`. No `requirements.txt`.
(3.8 is the floor: Ubuntu 20.04 / Debian 11 era interpreters are supported; 3.7 is not.)
