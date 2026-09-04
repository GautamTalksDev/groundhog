"""Group repeated intents with stdlib TF-IDF clustering."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from gh.intents import Intent

# Cosine threshold for joining an existing cluster.
# Higher → fewer, tighter clusters (precision). Lower → more merges (recall).
# Prefer false negatives over false positives: three correct beats thirty noisy.
SIMILARITY_THRESHOLD = 0.45

# Post-cluster cohesion floor. Clusters looser than this are dropped.
MIN_COHESION = 0.35

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
    # Internal: running TF-IDF centroid (term → weight). Not for callers.
    _centroid: dict[str, float] = field(default_factory=dict, repr=False)


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
    filtered.sort(key=lambda c: (-c.run_count, -c.cohesion, c.label.lower()))
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
        label=_truncate(intent.raw_text, 90),
        projects={intent.project},
        first_seen=intent.timestamp,
        last_seen=intent.timestamp,
        run_count=1,
        cohesion=1.0,
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
    medoid = _medoid(cluster.members, member_vecs)
    cluster.label = _truncate(medoid.raw_text if medoid else cluster.label, 90)
    cluster.run_count = len(cluster.members)
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
        if cluster.run_count < min_runs:
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


def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
