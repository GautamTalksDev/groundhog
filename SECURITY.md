# Security

Groundhog is a local reader for agent session transcripts. Those files already hold whatever you and the agent typed, including secrets you pasted. The job of this tool is to rank repeated work without making that worse: no extra privilege, no extra copies except the ones you ask for, no network.

Report a vulnerability privately at
[GitHub security advisories](https://github.com/GautamTalksDev/groundhog/security/advisories/new).
Do not open a public issue for a leak or an unexpected file read.

## Threat model

**Assets.** Session JSONL under the four discovery roots. The rendered report (stdout or `--out`). Play artifacts in a rote run directory. The process's memory while a scan runs.

**Attackers.** A malicious session file already on disk. A JSONL symlink planted under a harness root. A consumer of a `--out` report or a Play artifact who should not see transcript text. Anyone hoping the tool will call the network or spawn a shell from transcript content.

**Non-goals.** Groundhog does not protect the session files from the OS user who already owns them. If you can read `~/.cursor/projects`, you can read those chats without this tool.

## Least privilege

The process runs as the invoking user. There is no setuid path, no sudo, and no declared elevation. Session files are opened with `open(..., "r")`. Discovery lists directories and stats files. The analysis never modifies a transcript.

`--out PATH` writes a report to a path you pass. That path is not sandboxed. It can overwrite any file your user can write. Do not point it at a session tree unless you mean to.

## No network egress

The analysis code imports none of `urllib`, `http.client`, `socket`, `ssl`, `requests`, or `aiohttp`. `main.ts` does not call `fetch`. Prices are loaded from `prices.json` on disk. `requires_endpoints` is empty. Nothing in the process adds a network call at runtime either.

Groundhog will not phone home if the machine is online. It also will not fetch a model price list.

## Secret handling and redaction limits

Transcripts are untrusted input that often contain keys. Default render redacts a small set of patterns (`sk-`, `ghp_`, `AKIA`, `Bearer …`, long base64-looking runs) and truncates evidence to 120 characters. See [PRIVACY.md](PRIVACY.md).

Redaction is not encryption and is not a secret scanner. It will miss many credentials. Play artifacts and in-memory `Session` objects keep original text. `--no-redact` is an explicit choice to print that text.

JSONL lines are `json.loads`'d. Transcript strings are never `eval`'d and never passed to a shell. Shell commands that appear in tool-use blocks are stored as strings for rediscovery and ranking. They are not executed.

## Path traversal and symlinks

Input paths are not a user flag. Discovery starts at `_HARNESS_ROOTS` under `Path.home()` and walks downward.

`os.walk(..., followlinks=False, onerror=...)` does not recurse into directory symlinks. A directory the walk cannot open is recorded, named in NOT COUNTED, and forces `PARTIAL SCAN — NOT A CLEAN RESULT`. A `.jsonl` file that is itself a symlink is resolved with `Path.resolve()` and kept only when that real path is inside the harness root it was found under (`_resolved_inside` in `gh/discover.py`). A link that escapes the root is refused before `stat`/`open` of the target. It is counted and named in NOT COUNTED as `1 file skipped (symlink points outside the history directory)`. The scan is partial. The target path is not printed.

A symlink whose target still sits inside the same history directory is treated as a normal session file.

`--out` can write anywhere the user can write. Play `write_artifact` writes under the path rote passes in (`artifacts/...`).

## Resource limits

Parse takes a monotonic deadline of 20 seconds (`TIME_BUDGET_SECONDS` in `groundhog.py`, `TIME_BUDGET` in `resources/python/steps/parse.py`). Files are ordered smallest first. Each file is read line by line. Hitting the deadline sets `truncated=True`, skips the rest, and forces the `PARTIAL SCAN — NOT A CLEAN RESULT` verdict.

The Play also sets `timeout_ms: 60000` on every step in `main.ts`. That is a separate rote kill switch. The 20-second budget is the one the parser itself enforces.

There is no extra cap on process RSS. A single enormous JSONL line still becomes one Python string.

## Dependency posture

Python 3 standard library only. No PyPI package, no lockfile, no native extension. The supply-chain surface of this repo is the stdlib you already trust plus this source. `prices.json` is data, not code.

The Play wrapper is TypeScript front matter plus a short presentation script that prints the report step's stdout. It does not add npm library dependencies inside this repository.

## OWASP Top 10:2025

The current list is [OWASP Top 10:2025](https://owasp.org/Top10/2025/0x00_2025-Introduction/), announced November 2025 and finalized January 2026. There is no 2026 revision. SSRF is not a standalone category. It was rolled into A01 Broken Access Control.

Groundhog is not a web application. The two mappings below are the ones that actually describe this codebase.

**A03:2025 Software Supply Chain Failures.** This category debuted at #3. On the [official A03 page](https://owasp.org/Top10/2025/A03_2025-Software_Supply_Chain_Failures/) it has the highest average incidence rate in the 2025 data (5.72%) and only 11 mapped CVEs. OWASP's own note is that this class is hard to test for: there is often no signature to scan. Groundhog's control is structural. Python 3 standard library only. No PyPI package, no lockfile, no native extension. The supply-chain surface is the CPython runtime on the machine plus this source. `prices.json` is data, not code.

**A10:2025 Mishandling of Exceptional Conditions.** This is a new category for 2025. It covers failing open, poor error handling, and landing in an inconsistent state when something unusual happens. That is Groundhog's design thesis. A partial scan never renders as a clean null. Missing directories, broken JSONL, unreadable files, unreadable subdirectories under a harness root, a hit of the 20-second parse budget, and failed pipeline stages become labeled NOT COUNTED. Estimates are never presented as measurements. Unknown stays unknown. Rendering emptiness as safety is an A10 failure. The five verdict classes exist so that cannot happen here.

The remaining 2025 categories, briefly:

| ID | Applies? |
|----|----------|
| A01 Broken Access Control | Weakly. The process can read any transcript the OS user can read, and `--out` can write any path that user can write. SSRF sits in this category in 2025. Groundhog has no outbound requests, so that slice does not apply. |
| A02 Security Misconfiguration | Default is `--redact` on. `--no-redact` is opt-in. |
| A04 Cryptographic Failures | Does not apply. No encryption of transcripts, no token storage, no TLS client. |
| A05 Injection | JSON parsing only. Transcript text is not executed. `--out` is a file write, not a query. |
| A06 Insecure Design | Applies as this threat model: reading secrets that already sit in chat logs, then printing a subset. |
| A07 Authentication Failures | Does not apply. No accounts. |
| A08 Software or Data Integrity Failures | No update channel inside the tool. You install a Play version or a git checkout. |
| A09 Security Logging & Alerting Failures | Does not apply as a service. Groundhog does not keep its own log. A report file you requested is your responsibility. |

## OWASP LLM Top 10

Groundhog is not an LLM. It does not call a model.

| ID | Applies? |
|----|----------|
| LLM01 Prompt Injection | Does not apply to Groundhog itself. Session text is not sent to a model. |
| LLM02 Sensitive Information Disclosure | Applies. The input is chat history. Default redaction is heuristic. Artifacts and `--out` can hold raw text. |
| LLM03 Supply Chain | Stdlib only, as above. |
| LLM04 Data and Model Poisoning | Does not apply. No training, no model weights. A crafted JSONL can skew *this* report. It cannot retrain anything. |
| LLM05 Improper Output Handling | The report is text. `--suggest` emits Markdown derived from transcripts. Treat both as untrusted if you feed them to another tool. |
| LLM06 Excessive Agency | Does not apply. Groundhog does not invoke agent tools. |
| LLM07 System Prompt Leakage | Does not apply. |
| LLM08 Vector and Embedding Weaknesses | Does not apply. Clustering is lexical, in-process. |
| LLM09 Misinformation | Applies as honesty rules. Same control as A10:2025: estimates are labeled, a partial scan cannot render as a clean null, unknowns stay unknown. |
| LLM10 Unbounded Consumption | The 20-second parse deadline and smallest-first order bound work. A hostile huge file can still consume that window and a large line can consume memory. |

## Reporting a vulnerability

1. Open a [private advisory](https://github.com/GautamTalksDev/groundhog/security/advisories/new).
2. Include the Groundhog version (`main.ts` `metadata.version` or the Play URI), the command you ran, and a minimal JSONL that shows the bug.
3. Give a reasonable window before any public write-up.

A missing redaction pattern is in scope if it leaks from default `--redact` output. A secret that lives only in the original session file, and never in the report, is already on disk and is out of scope unless Groundhog read it from outside the discovery roots. A file symlink that escapes a harness root and is parsed would be in scope; as of 0.1.0 that path is refused and named.
