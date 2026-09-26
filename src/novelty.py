"""
Novelty scoring pipeline.

    relevance(S)  = cosine_similarity(embed(S), embed(fixed_content))
    novelty(S)    = 1 - max(cosine_similarity(embed(S), embed(p)) for p in pool)

    if relevance(S) < RELEVANCE_FLOOR:
        final_score = 0.0          # irrelevant -> no reward, regardless of novelty
    else:
        final_score = clip(novelty(S), 0.0, 1.0)

Rationale for this shape (see README for the full writeup):
  - Novelty is measured *relative to the existing pool*, not in the abstract --
    a submission is only "novel" insofar as nobody has said it yet.
  - Relevance is a hard gate, not a blended multiplier. A blended score (e.g.
    0.5*novelty + 0.5*relevance) would let a wildly off-topic but unique
    submission still earn partial credit. The task explicitly requires that
    "high novelty, but low relevance submissions are not rewarded" -- a hard
    floor is the only shape that guarantees this in all cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.embeddings import TextEmbedder, cosine, get_embedder

RELEVANCE_FLOOR = 0.07  # tuned against the synthetic dataset; see README "Threshold tuning"
GEMINI_RELEVANCE_FLOOR = 0.773


def relevance_floor_for_backend(backend: str) -> float:
    floors = {
        "local": RELEVANCE_FLOOR,
        "gemini": GEMINI_RELEVANCE_FLOOR,
    }
    normalized_backend = backend.strip().lower()
    try:
        return floors[normalized_backend]
    except KeyError as error:
        raise ValueError(
            f"Unknown embedding backend={backend!r}. Use 'local' or 'gemini'."
        ) from error


def submission_text(submission: dict) -> str:
    """Concatenate the user-provided fields into a single string for embedding.
    Stance is included as a short tag; headline/body carry the semantic weight."""
    return f"{submission['headline']}. {submission['body']} [stance: {submission['stance']}]"


@dataclass
class ScoreResult:
    submission_id: str
    relevance: float
    novelty: float
    final_score: float
    nearest_neighbor_id: str | None


def build_corpus(pool: list[dict], fixed_content: str) -> list[str]:
    """Corpus used to fit the embedder's vocabulary: every pool submission's
    text plus the fixed content itself."""
    return [submission_text(s) for s in pool] + [fixed_content]


def score_submission(
    new_submission: dict,
    pool: list[dict],
    fixed_content: str,
    embedder=None,  # TextEmbedder or GeminiEmbedder; see src/embeddings.py
    relevance_floor: float | None = None,
) -> ScoreResult:
    """
    Score `new_submission` for novelty against `pool`, gated by relevance to
    `fixed_content`. If `embedder` is not provided, one is fit fresh on
    pool + fixed_content + new_submission (fine for demo/eval scale; for
    production, fit once and reuse -- see README).
    """
    if embedder is None:
        corpus = build_corpus(pool, fixed_content) + [submission_text(new_submission)]
        embedder = get_embedder(corpus)  # backend chosen via EMBEDDING_BACKEND env var
    if relevance_floor is None:
        relevance_floor = relevance_floor_for_backend(
            getattr(embedder, "backend", "local")
        )

    new_vec = embedder.embed(submission_text(new_submission))
    fixed_vec = embedder.embed(fixed_content)

    relevance = cosine(new_vec, fixed_vec)

    if not pool:
        novelty = 1.0
        nearest_id = None
    else:
        sims = []
        for p in pool:
            p_vec = embedder.embed(submission_text(p))
            sims.append((cosine(new_vec, p_vec), p["id"]))
        max_sim, nearest_id = max(sims, key=lambda x: x[0])
        novelty = 1.0 - max_sim

    novelty = float(np.clip(novelty, 0.0, 1.0))

    if relevance < relevance_floor:
        final_score = 0.0
    else:
        final_score = novelty

    return ScoreResult(
        submission_id=new_submission["id"],
        relevance=round(relevance, 4),
        novelty=round(novelty, 4),
        final_score=round(final_score, 4),
        nearest_neighbor_id=nearest_id,
    )
