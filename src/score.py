"""
CLI: score every submission in data/submissions.json against "the rest of the
pool" (leave-one-out), and print a summary table grouped by expected band.

Usage:
    python -m src.score
"""

import json
from pathlib import Path

from src.embeddings import get_embedder
from src.novelty import (
    build_corpus,
    relevance_floor_for_backend,
    score_submission,
    submission_text,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data():
    fixed_content = json.loads((DATA_DIR / "fixed_content.json").read_text())["text"]
    submissions = json.loads((DATA_DIR / "submissions.json").read_text())
    return fixed_content, submissions


def main():
    fixed_content, submissions = load_data()

    # Fit one embedder on the full corpus (all submissions + fixed content)
    # so every leave-one-out comparison uses a consistent vocabulary.
    # Backend (local TF-IDF vs. live Gemini API) is chosen by the
    # EMBEDDING_BACKEND env var -- see src/embeddings.py.
    import os
    backend = os.environ.get("EMBEDDING_BACKEND", "local").strip().lower()
    print(f"[embedding backend: {backend}]")
    relevance_floor = relevance_floor_for_backend(backend)

    corpus = build_corpus(submissions, fixed_content)
    embedder = get_embedder(corpus)

    results = []
    for i, sub in enumerate(submissions):
        pool = submissions[:i] + submissions[i + 1:]
        result = score_submission(
            sub,
            pool,
            fixed_content,
            embedder=embedder,
            relevance_floor=relevance_floor,
        )
        results.append((sub.get("expected_novelty_band", "?"), result))

    print(f"{'band':10} {'id':12} {'relevance':>10} {'novelty':>8} {'final':>8}  nearest")
    for band, r in sorted(results, key=lambda x: x[0]):
        print(f"{band:10} {r.submission_id:12} {r.relevance:10.3f} {r.novelty:8.3f} {r.final_score:8.3f}  {r.nearest_neighbor_id}")

    # Per-band summary
    print("\n--- Band averages (final_score) ---")
    by_band = {}
    for band, r in results:
        by_band.setdefault(band, []).append(r.final_score)
    for band, scores in sorted(by_band.items()):
        avg = sum(scores) / len(scores)
        print(f"{band:10} avg={avg:.3f} n={len(scores)}")


if __name__ == "__main__":
    main()
