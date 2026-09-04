# Groundhog

Groundhog reads your **local** Claude Code and Codex session history on this machine and tells you which chores you keep paying an agent to redo. It **reads those files only, writes nothing unless you pass an output path, and sends nothing** — no network calls, no accounts, no API keys, no uploads.

## What you get

A one-screen report of your top repeated chores, with:

- how often you asked
- rough token and dollar cost (labeled measured / estimated / unknown)
- short evidence quotes from your own asks
- a **NOT COUNTED** section listing anything missing or skipped

## One command

```bash
rote play run https://play.modiqo.ai/<owner>/groundhog
```

Or locally after checkout:

```bash
python3 groundhog.py
# or
rote play run ./main.ts days=14 top=3 min_runs=3 redact=true
```

## Parameters

| Name | Default | Meaning |
|------|---------|---------|
| `days` | `14` | Look-back window |
| `top` | `3` | How many chores to show |
| `min_runs` | `3` | Minimum repeats to surface |
| `redact` | `true` | Scrub secret-like strings; truncate evidence |

## What it can't see

- Sessions outside `~/.claude/projects/`, `~/.codex/sessions/`, `~/.codex/history/`
- History older than `days`
- Token counts a harness never recorded (costs then say **estimated** or **unknown**)
- Any chat UI that does not write local JSONL session files
- Another machine or OS user

## Trust claims (match the code)

- **Read-only** — no declared writes; analysis never modifies your session files
- **No auth / no credentials** — nothing to sign in to
- **No network** — prices come from a bundled `prices.json`, not a live API
- **Requires** — `python3` only (stdlib; no `pip install`)
