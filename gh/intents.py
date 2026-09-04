"""Extract and normalize task intents from session turns."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from gh.parse import Session, Turn, unwrap_cursor_text


@dataclass
class Intent:
    """One substantive user ask, normalized for clustering."""

    session_id: str
    harness: str
    project: str
    timestamp: Optional[str]
    raw_text: str
    normalized: str
    session_turn_count: int
    session_tokens: Optional[int]
    # Session-level usage snapshot for costing (None = absent in file).
    session_input_tokens: Optional[int] = None
    session_output_tokens: Optional[int] = None
    session_cache_read_tokens: Optional[int] = None
    session_model: Optional[str] = None
    session_text_chars: int = 0


# Pure acknowledgements / nudges — never intents on their own.
_CONTINUATIONS = frozenset(
    {
        "yes",
        "y",
        "yeah",
        "yep",
        "ok",
        "okay",
        "k",
        "sure",
        "continue",
        "go on",
        "go ahead",
        "thanks",
        "thank you",
        "thx",
        "no",
        "nope",
        "nah",
        "try again",
        "fix it",
        "do it",
        "please",
        "please continue",
        "lgtm",
        "looks good",
        "done",
        "ship it",
    }
)

# Later turns starting this way are corrections, not new tasks.
_CORRECTION_PREFIXES = (
    "actually",
    "wait",
    "no wait",
    "never mind",
    "nevermind",
    "i meant",
    "i mean",
    "instead",
    "rather",
    "change that",
    "update that",
    "make it",
    "make that",
    "fix that",
    "fix the typo",
    "also just",
    "one more thing",
    "sorry",
    "my bad",
    "on second thought",
    "don't",
    "do not",
    "remove that",
    "undo",
)

# Signal that the user is asking for work (not dumping a blob).
_IMPERATIVE_VERBS = frozenset(
    {
        "add",
        "build",
        "change",
        "check",
        "clean",
        "create",
        "debug",
        "delete",
        "deploy",
        "document",
        "explain",
        "extract",
        "find",
        "fix",
        "generate",
        "implement",
        "improve",
        "investigate",
        "migrate",
        "move",
        "optimize",
        "parse",
        "patch",
        "refactor",
        "remove",
        "rename",
        "replace",
        "rewrite",
        "run",
        "set",
        "ship",
        "show",
        "simplify",
        "summarize",
        "test",
        "update",
        "upgrade",
        "wire",
        "write",
        "compare",
        "review",
        "resolve",
        "handle",
        "support",
        "convert",
        "install",
        "configure",
        "setup",
        "make",
        "help",
        "please",
        "can",
        "could",
        "would",
        "need",
        "want",
        "look",
        "inspect",
        "trace",
        "profile",
        "benchmark",
        "commit",
        "push",
        "merge",
        "rebase",
        "draft",
        "design",
        "plan",
        "scaffold",
        "stub",
        "mock",
        "assert",
        "ensure",
        "verify",
        "validate",
        "audit",
        "rank",
        "cluster",
        "normalize",
        "discover",
        "render",
        "cost",
        "redact",
        "publish",
        "bump",
        "polish",
    }
)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "please",
        "also",
        "like",
        "get",
        "got",
        "using",
        "use",
        "used",
    }
)

_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# Absolute / home / Windows paths only — not relative segments like discover/parse.
_RE_PATH = re.compile(
    r"(?<![A-Za-z0-9])(?:~/|/|[A-Za-z]:\\)[^\s'\"`]+"
)
_RE_SHA = re.compile(r"\b[0-9a-f]{7,40}\b", re.IGNORECASE)
_RE_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_RE_NUM = re.compile(r"\b\d{3,}\b")
_RE_QUOTED = re.compile(r"([\"'])(?:\\.|(?!\1).)*\1")
_RE_PUNCT = re.compile(r"[^\w\s<>]+", re.UNICODE)
_RE_SPACE = re.compile(r"\s+")

# Truncate the comparison string only — raw_text stays full for evidence.
_NORMALIZED_MAX_CHARS = 400
_NORMALIZED_MAX_TOKENS = 30

# A task request is short. Discursive replies (analysis, quoted prompts)
# are not intents even when they contain an imperative verb.
_MAX_INTENT_CHARS = 1200

# A bold span covering a phrase/sentence, not a single **word**.
_RE_EMPHASIS_BLOCK = re.compile(r"\*\*[^*\n]{12,}\*\*")
# Opening fence followed by a newline = pasted block, not inline ```code```.
_RE_FENCED_PROMPT = re.compile(r"```[^\n]*\n")

# Shared prompt templates otherwise dominate the TF-IDF vocabulary and
# merge unrelated work (GASKET + Keyring clustered on "implement the
# plan as specified"). Applied when building `normalized` only;
# raw_text is left intact for evidence.
BOILERPLATE_PHRASES = [
    "implement the plan as specified",
    "it is attached for your reference",
    "do not edit the plan file itself",
    "do not edit the plan",
    "do not edit files",
    "to-do's from the plan have already been created",
    "todos from the plan have already been created",
    "do not create them again",
    "mark them as in_progress as you work, starting with the first one",
    "mark them as in_progress as you work",
    "don't stop until you have completed all the to-dos",
    "don't stop until you have completed all the todos",
    "dont stop until you have completed all the to-dos",
    "read-only exploration",
    "read only exploration",
    "return exact",
    "focus on:",
    "find and summarize:",
]

_META_PREFIXES = (
    "fair —",
    "fair—",
    "fair -",
    "fair,",
    "you're right",
    "you’re right",
    "youre right",
    "you are right",
    "sorry",
    "i mixed",
    "actually",
)

_FILE_FETCH_PHRASES = (
    "send me this file please",
    "send me these files please",
    "send me this file",
    "send me these files",
    "send this file please",
    "please send me this file",
    "please send me",
    "send me this",
    "send me",
)

# "In /home/foo/bar, …" / "In Cursor, …" / "At <path>, …" / "Inside <path>, …"
_RE_LOCATIVE = re.compile(
    r"^(?:in|at|inside)\s+"
    r"(?:cursor\b|(?:~/|/|[A-Za-z]:\\)[^\s,]+)\s*,\s*",
    re.IGNORECASE,
)
_RE_FILENAME_TOKEN = re.compile(
    r"^[\w.+-]+\.[A-Za-z0-9]{1,10}$"
)
# Same chore across repos must match; project names live on Intent.project.
_RE_CHECKPOINT = re.compile(
    r"\bcp-?\d+(?:\.\d+)?\b|\bcheckpoint\s+\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)
_RE_SLASH_TOKEN = re.compile(r"[^\s]+/[^\s]+")
_RE_REPO_SEGMENT = re.compile(
    r"\b(?:crates|packages|apps|internal|cmd|pkg|src)"
    r"(?:/[A-Za-z0-9._-]+|\s+[A-Za-z0-9._-]*[./-][A-Za-z0-9._-]*)",
    re.IGNORECASE,
)

_STACK_HINTS = (
    "traceback (most recent call last)",
    "exception in thread",
    "caused by:",
    "fatal error:",
    "panic:",
    "stack trace",
    'file "/',
    "file '/",
    "at com.",
    "at org.",
)

_STACK_START = re.compile(
    r"(?im)^(traceback \(most recent call last\)|fatal error:|panic:|"
    r"exception in thread|stack trace)"
)

_CODE_LINE_HINTS = (
    "import ",
    "from ",
    "def ",
    "class ",
    "function ",
    "const ",
    "let ",
    "var ",
    "public ",
    "private ",
    "package ",
    "#!/",
    "```",
)


def extract_intents(
    sessions: list[Session],
    projects: Optional[list[str]] = None,
) -> list[Intent]:
    """Pull substantive user asks from sessions; skip corrections/noise."""
    names = _project_names(sessions, projects)
    intents: list[Intent] = []
    for session in sessions:
        usage = _session_usage(session)
        turn_count = len(session.turns)
        seen_first = False
        for turn in session.turns:
            if turn.role != "user":
                continue
            text, tagged_ts = unwrap_cursor_text(turn.text)
            if not is_substantive(text):
                continue
            if not seen_first:
                intents.append(
                    _to_intent(
                        session,
                        turn,
                        turn_count,
                        usage,
                        text,
                        tagged_ts,
                        projects=names,
                    )
                )
                seen_first = True
                continue
            if is_new_task(text):
                intents.append(
                    _to_intent(
                        session,
                        turn,
                        turn_count,
                        usage,
                        text,
                        tagged_ts,
                        projects=names,
                    )
                )
    return intents


def is_substantive(text: str) -> bool:
    """True if text looks like a real user ask, not noise."""
    raw = text or ""
    if len(raw.strip()) > _MAX_INTENT_CHARS:
        return False
    cleaned = strip_leading_locative(raw.strip())
    stripped = _strip_boilerplate_phrases(cleaned)
    if _is_conversational_reply(stripped):
        return False
    collapsed = _RE_SPACE.sub(" ", stripped).strip()
    if len(collapsed) < 25:
        return False
    if _is_continuation(collapsed):
        return False
    if _is_meta_comment(collapsed):
        return False
    if _is_file_fetch_request(collapsed):
        return False
    if _is_blob_without_imperative(stripped):
        return False
    return True


def is_new_task(text: str) -> bool:
    """True if a later user turn starts a new task rather than a correction."""
    cleaned = (text or "").strip()
    if not is_substantive(cleaned):
        return False
    lower = cleaned.lower()
    for prefix in _CORRECTION_PREFIXES:
        if lower.startswith(prefix):
            # Exception: long standalone ask that merely opens with a hedge
            # still counts if it clearly states a new imperative goal.
            if len(cleaned) < 80:
                return False
            # "Actually, rewrite the billing client from scratch" — new task.
            if not _has_imperative(cleaned):
                return False
    return _has_imperative(cleaned)


def normalize_intent(
    text: str,
    projects: Optional[Iterable[str]] = None,
) -> str:
    """Normalize ask text for comparison / clustering."""
    s = strip_leading_locative(text or "")
    s = _strip_boilerplate_phrases(s)
    if len(s) > _NORMALIZED_MAX_CHARS:
        s = s[:_NORMALIZED_MAX_CHARS]
    s = s.lower()
    # Neutralize before punctuation split so "cp-5" and "crates/foo" stay one token.
    s = _RE_CHECKPOINT.sub(" <checkpoint> ", s)
    s = _RE_URL.sub(" <url> ", s)
    # Paths are already on Intent.project — a <path> token inflates
    # similarity between unrelated tasks in different repos.
    s = _RE_PATH.sub(" ", s)
    s = _replace_subpaths(s)
    s = _replace_project_names(s, projects)
    s = _RE_UUID.sub(" <id> ", s)
    s = _RE_SHA.sub(" <sha> ", s)
    s = _RE_QUOTED.sub(" <str> ", s)
    s = _RE_NUM.sub(" <num> ", s)
    s = _RE_PUNCT.sub(" ", s)
    s = _RE_SPACE.sub(" ", s).strip()
    tokens = [t for t in s.split(" ") if t and t not in _STOPWORDS]
    if len(tokens) > _NORMALIZED_MAX_TOKENS:
        tokens = tokens[:_NORMALIZED_MAX_TOKENS]
    return " ".join(tokens)


def strip_leading_locative(text: str) -> str:
    """Drop a leading ``In <path>,`` / ``In Cursor,`` / ``At`` / ``Inside``."""
    return _RE_LOCATIVE.sub("", text or "", count=1)


def _project_names(
    sessions: list[Session],
    extra: Optional[Iterable[str]] = None,
) -> list[str]:
    names = {s.project for s in sessions if s.project}
    if extra:
        names.update(p for p in extra if p)
    return sorted(names)


def _replace_project_names(
    text: str, projects: Optional[Iterable[str]]
) -> str:
    """Swap known project names for ``<project>``.

    The same chore repeated across different repos must be recognizable as
    the same chore, and project names are already tracked as a field.
    """
    s = text
    names = sorted(
        {p.strip() for p in (projects or []) if p and len(p.strip()) >= 3},
        key=len,
        reverse=True,
    )
    for name in names:
        s = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])",
            " <project> ",
            s,
            flags=re.IGNORECASE,
        )
    return s


def _replace_subpaths(text: str) -> str:
    s = _RE_SLASH_TOKEN.sub(" <subpath> ", text)
    s = _RE_REPO_SEGMENT.sub(" <subpath> ", s)
    return s


def _strip_boilerplate_phrases(text: str) -> str:
    s = (
        (text or "")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
    )
    for phrase in sorted(BOILERPLATE_PHRASES, key=len, reverse=True):
        s = re.sub(re.escape(phrase), " ", s, flags=re.IGNORECASE)
    return s


def _to_intent(
    session: Session,
    turn: Turn,
    turn_count: int,
    usage: dict,
    text: Optional[str] = None,
    tagged_ts: Optional[str] = None,
    projects: Optional[Iterable[str]] = None,
) -> Intent:
    query = (text if text is not None else turn.text).strip()
    return Intent(
        session_id=session.session_id,
        harness=session.harness,
        project=session.project,
        timestamp=tagged_ts or turn.timestamp or session.started_at,
        raw_text=query,
        normalized=normalize_intent(query, projects=projects),
        session_turn_count=turn_count,
        session_tokens=usage["session_tokens"],
        session_input_tokens=usage["input_tokens"],
        session_output_tokens=usage["output_tokens"],
        session_cache_read_tokens=usage["cache_read_tokens"],
        session_model=usage["model"],
        session_text_chars=usage["text_chars"],
    )


def _session_usage(session: Session) -> dict:
    """Aggregate per-session token fields; None means absent (not zero)."""
    input_total = 0
    output_total = 0
    cache_total = 0
    saw_input = False
    saw_output = False
    saw_cache = False
    model: Optional[str] = None
    text_chars = 0

    for turn in session.turns:
        text_chars += len(turn.text or "")
        if turn.model and not model:
            model = turn.model
        if turn.input_tokens is not None:
            input_total += turn.input_tokens
            saw_input = True
        if turn.output_tokens is not None:
            output_total += turn.output_tokens
            saw_output = True
        if turn.cache_read_tokens is not None:
            cache_total += turn.cache_read_tokens
            saw_cache = True

    session_tokens: Optional[int]
    if saw_input or saw_output or saw_cache:
        session_tokens = (
            (input_total if saw_input else 0)
            + (output_total if saw_output else 0)
            + (cache_total if saw_cache else 0)
        )
    else:
        session_tokens = None

    return {
        "session_tokens": session_tokens,
        "input_tokens": input_total if saw_input else None,
        "output_tokens": output_total if saw_output else None,
        "cache_read_tokens": cache_total if saw_cache else None,
        "model": model,
        "text_chars": text_chars,
    }


def _session_token_sum(session: Session) -> Optional[int]:
    return _session_usage(session)["session_tokens"]


def _is_meta_comment(text: str) -> bool:
    lower = (text or "").strip().lower()
    lower = lower.replace("\u2019", "'").replace("\u2013", "-")
    for prefix in _META_PREFIXES:
        if lower.startswith(prefix):
            return True
    return False


def _is_file_fetch_request(text: str) -> bool:
    """True for a filename list plus a bare 'send me this file' request."""
    s = _RE_SPACE.sub(" ", (text or "").strip().lower())
    stripped_any = False
    for phrase in _FILE_FETCH_PHRASES:
        if phrase in s:
            s = s.replace(phrase, " ")
            stripped_any = True
    s = s.replace(",", " ")
    s = _RE_SPACE.sub(" ", s).strip()
    tokens = [t.strip(".,;:") for t in s.split() if t.strip(".,;:")]
    if not tokens:
        return stripped_any
    filler = {"and", "or", "the", "file", "files", "this", "these", "please", "a"}
    content = [t for t in tokens if t not in filler]
    if not content:
        return stripped_any
    return all(_RE_FILENAME_TOKEN.match(t) for t in content)


def _is_continuation(text: str) -> bool:
    compact = _RE_SPACE.sub(" ", text.strip().lower())
    compact = _RE_PUNCT.sub("", compact).strip()
    return compact in _CONTINUATIONS


def _is_conversational_reply(text: str) -> bool:
    """True for a discursive reply to the assistant, not a task request.

    Task requests are short and imperative. Replies are long analysis,
    markdown-formatted status, or a pasted prompt in a fenced block.
    """
    raw = text or ""
    stripped = raw.lstrip()
    if stripped.startswith("**"):
        return True
    if _RE_EMPHASIS_BLOCK.search(raw):
        return True
    if _is_multi_paragraph_analysis(raw):
        return True
    if _RE_FENCED_PROMPT.search(raw):
        return True
    return False


def _is_multi_paragraph_analysis(text: str) -> bool:
    paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()
    ]
    # Punctuation-only leftovers after boilerplate stripping are not analysis.
    substantive = [
        p for p in paragraphs if re.search(r"[A-Za-z]{3,}", p)
    ]
    if len(substantive) >= 3:
        return True
    if len(substantive) >= 2:
        long_paras = [p for p in substantive if len(p) >= 180]
        if len(long_paras) >= 2:
            return True
    return False


def _has_imperative(text: str) -> bool:
    # Normalize lightly for verb detection (keep words intact).
    s = text.lower()
    s = _RE_URL.sub(" ", s)
    s = _RE_PATH.sub(" ", s)
    s = _RE_PUNCT.sub(" ", s)
    tokens = [t for t in s.split() if t]
    if not tokens:
        return False
    # Leading modal/softener then verb: "can you fix...", "please add..."
    for i, tok in enumerate(tokens[:6]):
        if tok in _IMPERATIVE_VERBS:
            return True
        if i >= 3 and tok not in {"you", "me", "us", "to", "the", "a", "an"}:
            break
    # Also accept verb anywhere in short task sentences.
    if len(tokens) <= 24 and any(t in _IMPERATIVE_VERBS for t in tokens):
        return True
    return False


def _prose_preamble(text: str) -> str:
    """Text before a pasted traceback/code dump, if any."""
    match = _STACK_START.search(text)
    if match and match.start() > 0:
        return text[: match.start()].strip()
    lines = text.splitlines()
    prose: list[str] = []
    for ln in lines:
        low = ln.strip().lower()
        if not low:
            if prose:
                break
            continue
        if any(low.startswith(h.strip()) or h in low[:30] for h in _CODE_LINE_HINTS):
            break
        if re.match(r"^file ['\"].+, line \d+", low):
            break
        if re.match(r"^\w+(error|exception):", low):
            break
        prose.append(ln)
        if len(prose) >= 3:
            break
    return "\n".join(prose).strip()


def _is_blob_without_imperative(text: str) -> bool:
    """Detect pasted stack traces / file dumps with no ask."""
    lower = text.lower()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    stack_hits = sum(1 for hint in _STACK_HINTS if hint in lower)
    code_hits = 0
    for ln in lines:
        low = ln.lower()
        if any(h in low[:40] for h in _CODE_LINE_HINTS):
            code_hits += 1
        if re.match(r'^file [\'"].+[\'"], line \d+', low):
            stack_hits += 1
        if re.match(r"^\w+error:", low) or re.match(r"^\w+exception:", low):
            stack_hits += 1

    looks_like_stack = stack_hits >= 2 or (
        stack_hits >= 1 and len(lines) >= 5
    )
    looks_like_file = (
        len(lines) >= 8
        and code_hits >= max(3, len(lines) // 3)
        and len(text) >= 200
    )
    if not (looks_like_stack or looks_like_file):
        return False
    # Only the human preamble can rescue a blob ("fix this: <traceback>").
    preamble = _prose_preamble(text)
    if preamble and _has_imperative(preamble):
        return False
    return True
