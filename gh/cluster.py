"""Group repeated intents with stdlib TF-IDF clustering."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from gh.intents import Intent, strip_leading_locative
from gh.parse import unwrap_cursor_text

# Cosine threshold for joining an existing cluster.
# Higher → fewer, tighter clusters (precision). Lower → more merges (recall).
# Prefer false negatives over false positives: three correct beats thirty noisy.
SIMILARITY_THRESHOLD = 0.45

# Post-cluster cohesion floor. Clusters looser than this are dropped.
MIN_COHESION = 0.35

# Stranger-facing label length — truncate on a word boundary.
_LABEL_LIMIT = 70

# Leading reply / status openers make poor chore labels.
_LABEL_PENALTY_PREFIXES = (
    "confirmed",
    "fair —",
    "fair—",
    "fair -",
    "fair,",
    "fair ",
    "you're right",
    "you’re right",
    "youre right",
    "sorry",
    "thanks",
    "thank you",
    "okay",
    "ok,",
    "yes,",
    "yeah,",
)

# Leading verbs that read as an ask (subset; used only for label choice).
_LABEL_VERBS = frozenset(
    {
        "add",
        "build",
        "change",
        "check",
        "clean",
        "compare",
        "configure",
        "create",
        "debug",
        "delete",
        "deploy",
        "design",
        "document",
        "explore",
        "explain",
        "extract",
        "find",
        "fix",
        "generate",
        "harden",
        "implement",
        "improve",
        "inspect",
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
        "scaffold",
        "set",
        "ship",
        "show",
        "simplify",
        "summarize",
        "test",
        "update",
        "upgrade",
        "verify",
        "wire",
        "write",
        "review",
        "resolve",
        "support",
        "install",
        "make",
        "help",
        "please",
        "can",
        "could",
        "would",
        "need",
        "look",
        "read",
        "re-run",
        "rerun",
    }
)

# Placeholders / ultra-generic tokens do not count as content words in labels.
_NON_CONTENT = frozenset(
    {
        "<path>",
        "<url>",
        "<sha>",
        "<id>",
        "<num>",
        "<str>",
        "please",
        "also",
        "just",
        "help",
        "need",
        "want",
    }
)

# Light synonym/stem fold applied only inside TF-IDF so near-paraphrases
# share mass without a stemmer dependency. Precision still comes from the
# cosine threshold + post-filters.
_TOKEN_FOLD = {
    "authentication": "auth",
    "authenticate": "auth",
    "authorisation": "auth",
    "authorization": "auth",
    "tests": "test",
    "testing": "test",
    "suites": "suite",
    "failures": "fail",
    "failing": "fail",
    "failed": "fail",
    "failure": "fail",
    "repair": "fix",
    "repairs": "fix",
    "repairing": "fix",
    "fixed": "fix",
    "fixes": "fix",
    "fixing": "fix",
    "rewrite": "write",
    "rewriting": "write",
    "rewrote": "write",
    "writing": "write",
    "written": "write",
    "explains": "explain",
    "explained": "explain",
    "understands": "understand",
    "understanding": "understand",
    "scaffolding": "scaffold",
    "modules": "module",
    "flags": "flag",
    "records": "record",
    "helpers": "helper",
    "intents": "intent",
    "paths": "path",
    "urls": "url",
    "shas": "sha",
    "reports": "report",
    "attacks": "attack",
    "fixtures": "fixture",
    "sessions": "session",
    "projects": "project",
}


@dataclass
class Cluster:
    """A group of near-duplicate intents representing one repeated chore."""

    id: str
    members: list[Intent]
    label: str
    projects: set[str]
    first_seen: Optional[str]
    last_seen: Optional[str]
    run_count: int
    cohesion: float
    # Unique session_id values among members. Repetition is sessions, not turns.
    distinct_sessions: int = 0
    # Internal: running TF-IDF centroid (term → weight). Not for callers.
    _centroid: dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.distinct_sessions = _distinct_session_count(self.members)


def cluster_intents(
    intents: list[Intent],
    *,
    min_runs: int = 3,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[Cluster]:
    """Greedy single-pass agglomeration over TF-IDF vectors of normalized text."""
    if not intents:
        return []

    vectors = _tfidf_vectors(intents)
    clusters: list[Cluster] = []

    for intent, vec in zip(intents, vectors):
        best_idx = -1
        best_sim = -1.0
        for i, cluster in enumerate(clusters):
            sim = _cosine(vec, cluster._centroid)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_idx >= 0 and best_sim >= similarity_threshold:
            _add_to_cluster(clusters[best_idx], intent, vec)
        else:
            clusters.append(_new_cluster(len(clusters) + 1, intent, vec))

    # Finalize labels / cohesion, then apply precision filters.
    finalized: list[Cluster] = []
    for cluster in clusters:
        _finalize(cluster, vectors, intents)
        finalized.append(cluster)

    filtered = _precision_filter(finalized, min_runs=min_runs)
    filtered.sort(
        key=lambda c: (-c.distinct_sessions, -c.cohesion, c.label.lower())
    )
    # Re-number for stable dump output after filtering/sorting.
    for i, cluster in enumerate(filtered, 1):
        cluster.id = f"c{i}"
    return filtered


def _tfidf_vectors(intents: list[Intent]) -> list[dict[str, float]]:
    docs = [_tokens(intent.normalized) for intent in intents]
    n_docs = len(docs)
    df: dict[str, int] = {}
    for tokens in docs:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    idf: dict[str, float] = {}
    for term, count in df.items():
        # Smoothed IDF so rare terms dominate without dividing by zero.
        idf[term] = math.log((1.0 + n_docs) / (1.0 + count)) + 1.0

    vectors: list[dict[str, float]] = []
    for tokens in docs:
        tf: dict[str, int] = {}
        for term in tokens:
            tf[term] = tf.get(term, 0) + 1
        length = float(len(tokens)) or 1.0
        vec = {
            term: (count / length) * idf[term]
            for term, count in tf.items()
        }
        vectors.append(_l2_normalize(vec))
    return vectors


def _tokens(normalized: str) -> list[str]:
    out: list[str] = []
    for raw in (normalized or "").split():
        term = _TOKEN_FOLD.get(raw, raw)
        # Cheap plural fold when no explicit synonym matched.
        if (
            term == raw
            and len(term) > 4
            and term.endswith("s")
            and not term.endswith("ss")
        ):
            stem = term[:-1]
            term = _TOKEN_FOLD.get(stem, stem)
        out.append(term)
    return out

def _l2_normalize(vec: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm <= 0.0:
        return {}
    return {k: v / norm for k, v in vec.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # Iterate the smaller dict.
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _new_cluster(seq: int, intent: Intent, vec: dict[str, float]) -> Cluster:
    return Cluster(
        id=f"c{seq}",
        members=[intent],
        label=clean_label(intent.raw_text),
        projects={intent.project},
        first_seen=intent.timestamp,
        last_seen=intent.timestamp,
        run_count=1,
        cohesion=1.0,
        distinct_sessions=1,
        _centroid=dict(vec),
    )


def _add_to_cluster(
    cluster: Cluster, intent: Intent, vec: dict[str, float]
) -> None:
    n = len(cluster.members)
    # Incremental mean centroid.
    centroid: dict[str, float] = {}
    keys = set(cluster._centroid) | set(vec)
    for key in keys:
        old = cluster._centroid.get(key, 0.0)
        new = vec.get(key, 0.0)
        centroid[key] = (old * n + new) / (n + 1)
    cluster._centroid = _l2_normalize(centroid)
    cluster.members.append(intent)
    cluster.projects.add(intent.project)
    cluster.run_count = len(cluster.members)
    cluster.distinct_sessions = _distinct_session_count(cluster.members)
    ts = intent.timestamp
    if ts:
        if cluster.first_seen is None or ts < cluster.first_seen:
            cluster.first_seen = ts
        if cluster.last_seen is None or ts > cluster.last_seen:
            cluster.last_seen = ts


def _finalize(
    cluster: Cluster,
    all_vectors: list[dict[str, float]],
    all_intents: list[Intent],
) -> None:
    # Map members back to vectors via identity on object + fallback index.
    intent_index = {id(intent): i for i, intent in enumerate(all_intents)}
    member_vecs: list[dict[str, float]] = []
    for member in cluster.members:
        idx = intent_index.get(id(member))
        if idx is None:
            member_vecs.append({})
        else:
            member_vecs.append(all_vectors[idx])

    cluster.cohesion = _mean_pairwise(member_vecs)
    cluster.label = choose_cluster_label(cluster.members, member_vecs)
    cluster.run_count = len(cluster.members)
    cluster.distinct_sessions = _distinct_session_count(cluster.members)
    cluster.projects = {m.project for m in cluster.members}

    timestamps = [m.timestamp for m in cluster.members if m.timestamp]
    if timestamps:
        cluster.first_seen = min(timestamps)
        cluster.last_seen = max(timestamps)


def _mean_pairwise(vectors: list[dict[str, float]]) -> float:
    n = len(vectors)
    if n <= 1:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _cosine(vectors[i], vectors[j])
            pairs += 1
    return total / pairs if pairs else 1.0


def _medoid(
    members: list[Intent], vectors: list[dict[str, float]]
) -> Optional[Intent]:
    if not members:
        return None
    if len(members) == 1:
        return members[0]
    best_i = 0
    best_score = -1.0
    for i, vec in enumerate(vectors):
        score = sum(_cosine(vec, other) for j, other in enumerate(vectors) if i != j)
        if score > best_score:
            best_score = score
            best_i = i
    return members[best_i]


def _precision_filter(
    clusters: list[Cluster], *, min_runs: int
) -> list[Cluster]:
    if not clusters:
        return []

    cohesions = sorted(c.cohesion for c in clusters)
    median_cohesion = cohesions[len(cohesions) // 2]

    kept: list[Cluster] = []
    for cluster in clusters:
        if cluster.distinct_sessions < min_runs:
            continue
        if cluster.cohesion < MIN_COHESION:
            continue
        if (
            len(cluster.projects) > 3
            and cluster.cohesion < median_cohesion
        ):
            # Generic-language false positive spanning many repos.
            continue
        if _content_word_count(cluster.label) < 4:
            continue
        kept.append(cluster)
    return kept


def _distinct_session_count(members: list[Intent]) -> int:
    return len({m.session_id for m in members})


def _content_word_count(label: str) -> int:
    tokens = [
        t
        for t in label.lower().replace("/", " ").split()
        if t and t not in _NON_CONTENT and not t.startswith("<")
    ]
    # Strip light punctuation leftovers.
    cleaned = []
    for tok in tokens:
        tok = "".join(ch for ch in tok if ch.isalnum() or ch in "_-")
        if tok and tok not in _NON_CONTENT:
            cleaned.append(tok)
    return len(cleaned)


def choose_cluster_label(
    members: list[Intent], vectors: list[dict[str, float]]
) -> str:
    """Pick a stranger-facing chore label from cluster members.

    Prefer an imperative ask. Among imperative members, prefer the TF-IDF
    medoid. If nothing reads as a task, fall back to the medoid as before.
    Clustering membership is unchanged — this only picks the display string.
    """
    if not members:
        return ""
    medoid = _medoid(members, vectors) or members[0]
    scored = [(label_imperative_score(m.raw_text), m) for m in members]
    imperatives = [m for score, m in scored if score > 0]
    if imperatives:
        if medoid in imperatives:
            chosen = medoid
        else:
            chosen = max(
                imperatives,
                key=lambda m: label_imperative_score(m.raw_text),
            )
    else:
        chosen = medoid
    return clean_label(chosen.raw_text)


def label_imperative_score(text: str) -> float:
    """Higher = better chore label. Negative = conversational / status reply."""
    raw = (text or "").lstrip()
    if not raw:
        return -100.0
    if raw.startswith("**") or raw.startswith("*"):
        return -50.0
    if raw[0] in "\"'`“”‘’":
        return -50.0
    lower = raw.lower()
    for prefix in _LABEL_PENALTY_PREFIXES:
        if lower.startswith(prefix):
            return -50.0
    probe = strip_leading_locative(raw)
    probe = probe.lstrip("*_\"'`“”‘’ \t")
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", probe.lower())
    if not tokens:
        return -10.0
    for i, tok in enumerate(tokens[:5]):
        if tok in _LABEL_VERBS:
            return 10.0 - (0.1 * i)
        if tok in {"please", "you", "me", "us", "to", "the", "a", "an"}:
            continue
        break
    return 0.0


def clean_label(text: str, limit: int = _LABEL_LIMIT) -> str:
    """Strip markdown/locatives and truncate on a word boundary."""
    s, _ = unwrap_cursor_text(text or "")
    s = s.lstrip()
    # Leading markdown emphasis (**bold**, *italics*, _).
    while s.startswith("**"):
        s = s[2:]
    s = s.lstrip("*_ \t")
    # Drop conversational openers left after markdown strip.
    lower = s.lower()
    for prefix in _LABEL_PENALTY_PREFIXES:
        if lower.startswith(prefix):
            s = s[len(prefix) :].lstrip(" .,;:—-")
            lower = s.lower()
            break
    # Unwrap a leading quoted clause before nibbling quote chars.
    quoted = re.match(r'^[\"“]([^\"”]+)[\"”]\s*', s)
    if quoted:
        s = (quoted.group(1) + " " + s[quoted.end() :]).strip()
    while s and s[0] in "\"'`“”‘’":
        s = s[1:]
    s = strip_leading_locative(s)
    # Plan-card scaffolding: keep the headline before "Prompt:".
    prompt_at = re.search(r"\s+Prompt:\s*", s, flags=re.IGNORECASE)
    if prompt_at and prompt_at.start() >= 12:
        s = s[: prompt_at.start()].rstrip()
    s = s.replace("**", "")
    s = " ".join(s.split())
    if len(s) <= limit:
        return s
    cut = s[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    cut = cut.rstrip(".,;:!?—- ")
    if not cut:
        cut = s[: max(1, limit - 1)]
    return cut + "…"


def _truncate(text: str, limit: int) -> str:
    return clean_label(text, limit=limit)
