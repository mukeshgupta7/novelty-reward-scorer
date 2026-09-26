"""
Automated tests demonstrating:
  1. High-novelty, on-topic submissions ARE rewarded (final_score high).
  2. Near-duplicate / low-novelty submissions are NOT rewarded highly.
  3. Off-topic submissions are NOT rewarded, REGARDLESS of novelty
     (the explicit "high novelty but low relevance -> not rewarded" requirement).

Uses the pre-generated data/submissions.json, which is labeled with an
`expected_novelty_band` per item (low / medium / high / offtopic) so we can
assert against known-good expectations rather than eyeballing numbers.
"""

import json
from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from src.embeddings import TextEmbedder
from src.novelty import (
    build_corpus,
    score_submission,
    submission_text,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def dataset():
    fixed_content = json.loads((DATA_DIR / "fixed_content.json").read_text())["text"]
    submissions = json.loads((DATA_DIR / "submissions.json").read_text())
    return fixed_content, submissions


@pytest.fixture(scope="module")
def embedder(dataset):
    fixed_content, submissions = dataset
    corpus = build_corpus(submissions, fixed_content)
    return TextEmbedder().fit(corpus)


def _score_all(dataset, embedder):
    """Leave-one-out score every submission against the rest of the pool."""
    fixed_content, submissions = dataset
    results = {}
    for i, sub in enumerate(submissions):
        pool = submissions[:i] + submissions[i + 1:]
        results[sub["id"]] = score_submission(sub, pool, fixed_content, embedder=embedder)
    return results


def test_offtopic_submissions_never_rewarded(dataset, embedder):
    """Core requirement: high novelty but low relevance -> final_score == 0.0."""
    fixed_content, submissions = dataset
    results = _score_all(dataset, embedder)

    offtopic_ids = [s["id"] for s in submissions if s["expected_novelty_band"] == "offtopic"]
    assert offtopic_ids, "expected at least one off-topic submission in the dataset"

    for oid in offtopic_ids:
        r = results[oid]
        # Off-topic items should have high raw novelty (nobody else is talking
        # about pasta/hiking/phones in this pool) -- proving the relevance
        # gate, not a lack of novelty, is what's blocking the reward.
        assert r.novelty > 0.7, f"{oid} expected high novelty (proves gate isn't just low novelty), got {r.novelty}"
        assert r.final_score == 0.0, f"{oid} is off-topic and must score 0.0, got {r.final_score}"


def test_high_novelty_ontopic_submissions_rewarded(dataset, embedder):
    """Genuinely distinct, on-topic takes should score well above near-duplicates."""
    results = _score_all(dataset, embedder)
    fixed_content, submissions = dataset

    high_scores = [results[s["id"]].final_score for s in submissions if s["expected_novelty_band"] == "high"]
    low_scores = [results[s["id"]].final_score for s in submissions if s["expected_novelty_band"] == "low"]

    avg_high = sum(high_scores) / len(high_scores)
    avg_low = sum(low_scores) / len(low_scores)

    assert avg_high > avg_low, (
        f"high-novelty band should score above low-novelty (near-duplicate) band on average: "
        f"avg_high={avg_high:.3f} avg_low={avg_low:.3f}"
    )
    # Majority of "high" band items should clear a meaningful reward bar.
    rewarded = [s for s in high_scores if s > 0.5]
    assert len(rewarded) >= len(high_scores) * 0.6, (
        f"expected at least 60% of high-novelty submissions to score > 0.5, "
        f"got {len(rewarded)}/{len(high_scores)}"
    )


def test_near_duplicate_submissions_not_highly_rewarded(dataset, embedder):
    """Near-duplicate ('low' band) submissions should mostly score low --
    they add little new information relative to what's already in the pool."""
    results = _score_all(dataset, embedder)
    fixed_content, submissions = dataset

    low_scores = [results[s["id"]].final_score for s in submissions if s["expected_novelty_band"] == "low"]
    avg_low = sum(low_scores) / len(low_scores)

    assert avg_low < 0.5, f"near-duplicate submissions should average below 0.5, got {avg_low:.3f}"


def test_relevance_gate_overrides_novelty_score():
    """Direct unit-level test of the gating rule, independent of the dataset:
    a synthetic, clearly off-topic-but-unique submission must score 0.0 even
    though nothing in the pool resembles it (so raw novelty would be ~1.0)."""
    fixed_content = "City council approved new protected bike lanes downtown to improve cyclist safety."
    pool = [
        {"id": "p1", "headline": "Bike lanes help safety", "body": "Protected lanes reduce injuries for cyclists downtown.", "stance": "Support"},
        {"id": "p2", "headline": "Parking loss hurts shops", "body": "Removing parking spots downtown will hurt small businesses.", "stance": "Oppose"},
    ]
    offtopic_novel_submission = {
        "id": "new-offtopic",
        "headline": "My new sourdough starter",
        "body": "I've been feeding my sourdough starter daily and it finally doubled in size overnight.",
        "stance": "Neutral",
    }

    corpus = build_corpus(pool, fixed_content) + [
        f"{offtopic_novel_submission['headline']}. {offtopic_novel_submission['body']} [stance: {offtopic_novel_submission['stance']}]"
    ]
    embedder = TextEmbedder().fit(corpus)

    result = score_submission(offtopic_novel_submission, pool, fixed_content, embedder=embedder)

    assert result.novelty > 0.8, "sanity check: this submission should look highly novel against the pool"
    assert result.final_score == 0.0, "off-topic content must not be rewarded even when maximally novel"


@pytest.mark.parametrize(
    ("relevance", "should_pass_gate"),
    [(0.758, False), (0.787, True)],
)
def test_gemini_relevance_floor_separates_observed_sample_scores(
    relevance, should_pass_gate
):
    fixed_content = "fixed article"
    submission = {
        "id": "new",
        "headline": "New submission",
        "body": "A comment to score.",
        "stance": "Neutral",
    }
    pool = [{
        "id": "existing",
        "headline": "Existing submission",
        "body": "A different comment.",
        "stance": "Support",
    }]
    submission_embedding = np.array([relevance, sqrt(1 - relevance**2)])
    query_text = submission_text(submission)

    class FixedEmbedder:
        backend = "gemini"

        def embed(self, text):
            if text == fixed_content:
                return np.array([1.0, 0.0])
            if text == query_text:
                return submission_embedding
            return np.array([0.0, 1.0])

    result = score_submission(
        submission,
        pool,
        fixed_content,
        embedder=FixedEmbedder(),
    )

    if should_pass_gate:
        assert result.final_score == pytest.approx(result.novelty)
    else:
        assert result.novelty > 0.0
        assert result.final_score == 0.0


def test_score_is_always_in_unit_interval(dataset, embedder):
    results = _score_all(dataset, embedder)
    for r in results.values():
        assert 0.0 <= r.final_score <= 1.0
        assert 0.0 <= r.novelty <= 1.0
        assert 0.0 <= r.relevance <= 1.0
