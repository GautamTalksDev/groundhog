# Privacy

Groundhog reads local session transcripts on this computer. It does not send them anywhere. You can verify every claim in this file with the grep commands at the bottom.

## Directories read, and why

Discovery only walks these roots under `$HOME` (`gh/discover.py`, `_HARNESS_ROOTS`):

| Path | Why |
|------|-----|
| `~/.claude/projects/` | Claude Code writes one JSONL transcript per session here. |
| `~/.codex/sessions/` | Codex session transcripts. |
| `~/.codex/history/` | Older Codex history in the same JSONL shape. |
| `~/.cursor/projects/<dir>/agent-transcripts/` | Cursor agent transcripts. `<dir>` is a project folder name. |

Inside those trees, only files whose name ends in `.jsonl` are opened, and only if the file mtime is inside the look-back window. Non-JSONL files in the same directories are ignored. The walk does not descend through directory symlinks (`os.walk(..., followlinks=False)`). A `.jsonl` that is itself a symlink is resolved with `Path.resolve()`. If the real path falls outside the harness root it was found under, Groundhog refuses it, counts it, and names it in NOT COUNTED as `1 file skipped (symlink points outside the history directory)`. The target is not parsed.

`--suggest` (CLI only) re-opens those same discovered session files to recover tool-use names for a scaffold. It does not add new roots.

## What is held in memory

Parse streams each JSONL file one line at a time. It does not load the raw file as one string. It does keep a normalized `Session` for every file that produced usable turns: session id, harness, project label, timestamps, every extracted turn (role, text, token fields, model), and every extracted tool call (name, path string, command string).

Intent extraction then copies substantive user asks into `Intent` records (raw text, normalized text, session token snapshot). Assistant turns stay on the `Session` and are used when estimating tokens from character counts if usage fields are missing.

When you run the Play, each step also writes a JSON artifact into that run's `artifacts/` directory so the next step can read it. `artifacts/parse.json` contains the normalized sessions, including turn text. Those files belong to the rote run. They are still on this machine. Session history files themselves are never modified.

## What is never read

- Your project trees, home config, SSH keys, `.env` files, browsers, and password stores. Those paths are not in `_HARNESS_ROOTS`.
- Bytes of files the agent Read or Wrote. Tool-use blocks contribute a path string and a command string that were already in the transcript.
- Credential stores. If a secret was pasted into a chat, it may already live in the JSONL. Groundhog will see that string because it is in the session file. It does not go looking for secrets anywhere else.

Turn text is parsed for both user and assistant roles when the JSON has extractable text. Clustering uses user asks that pass the intent filters (length cap 1200 characters, substantive, not a bare acknowledgement). Evidence quotes on the report come from those user asks.

## Nothing leaves the machine

There is no HTTP client, no `urllib`, no `socket` connect, no `requests`, and no `fetch` in this repository's analysis code. Model prices are read from the bundled `prices.json`. The Play front matter sets `requires_endpoints: []`.

A network stack on the computer can still exist. Groundhog's own code does not use it.

## Writes

Session files are opened for reading.

The CLI writes a report only when you pass `--out PATH`. Otherwise it prints to stdout. If that write fails, it prints to stdout and says so on stderr.

The Play writes JSON artifacts under the rote run's `artifacts/` directory (`write_artifact` in `resources/python/step_io.py`). That is how the nine steps pass data. It is not a write into your Claude, Codex, or Cursor trees.

## Redaction

`--redact` is on by default (`--no-redact` turns it off). In `gh/redact.py`:

- Spans matching `sk-…`, `ghp_…`, `AKIA…`, `Bearer …`, and long base64-looking tokens (40+ characters) are replaced with `<redacted>`.
- Evidence quotes are then truncated to 120 characters.
- The full rendered report is run through the same secret patterns once more, without extra truncation.

Limits, which are real:

- Short passwords, most JWTs, and any secret that does not match those regexes stay visible.
- Redaction is applied at render time. In-memory session objects and Play `artifacts/parse.json` still hold the original turn text.
- `--no-redact` plus `--out` writes whatever was in the transcript into the report file you named.

## How to verify this yourself

Run these from the repository root. They should find no network client imports and should show the roots, the read-only walk, and the two write sites.

```bash
# Network clients: expect no matches in analysis code
grep -RInE '^(import |from )(urllib|http\.client|httpx|requests|aiohttp|socket|ssl)(\.| |$)' \
  --include='*.py' gh groundhog.py resources/python

grep -RInE 'urlopen|http\.client|socket\.socket|requests\.|aiohttp|fetch\(' \
  --include='*.py' --include='*.ts' gh groundhog.py resources main.ts

# Discovery roots and symlink policy
grep -n -A6 '_HARNESS_ROOTS' gh/discover.py
grep -n 'followlinks' gh/discover.py
grep -n '_resolved_inside\|SKIP_SYMLINK_OUTSIDE' gh/discover.py

# CLI writes only via --out
grep -n 'args.out' groundhog.py

# Play step artifacts (rote run directory, not session trees)
grep -n 'write_artifact' resources/python/step_io.py resources/python/steps/*.py

# Redaction patterns and evidence cap
grep -n -A20 '_SECRET_PATTERNS\|EVIDENCE_LIMIT' gh/redact.py

# Bundled prices, not a live API
ls prices.json && grep -n 'prices.json' gh/cost.py
```

Then run Groundhog against an empty `HOME` and confirm it still exits 0 and lists the four checked paths:

```bash
HOME=$(mktemp -d) python3 groundhog.py --days 14
```
