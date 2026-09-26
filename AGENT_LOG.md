# Coding Agent Disclosure

Per the hackathon requirement to disclose coding-agent collaboration, this
project's initial scoring pipeline was developed with Claude (Anthropic); the
Gemini SDK migration and Streamlit demo were developed with GitHub Copilot.
I set the project goals, reviewed the proposed changes, and ran the checks.

## Agent Interaction

I asked Claude to design and implement the scoring pipeline, then reviewed
its behavior and requested fixes for low relevance scores. I later asked
GitHub Copilot to migrate Gemini to `google-genai` and add a separate
Streamlit demo without changing CLI behavior. I ran the offline tests and
syntax checks; no live Gemini API call was made.

## Engineering Decisions

- Kept character n-gram TF-IDF as the offline default after neural embeddings
  proved impractical in the constrained environment. Added Gemini as an
  optional backend behind the same embedder interface.
- Defined novelty relative to the existing submission pool and used a hard
  relevance gate so off-topic submissions cannot earn a reward.
- Kept the Streamlit demo as a separate entry point so the CLI workflows
  remain independent of the app's backend selection.

## Problem and Fix

Gemini initially rewarded off-topic submissions because the CLI and app used
the TF-IDF relevance cutoff (`0.07`). Cosine-score ranges differ by backend.
Added a separate provisional Gemini cutoff (`0.773`), based on the observed
gap between off-topic (`0.758`) and on-topic (`0.787`) sample scores, and
applied it in both interfaces. Regression tests cover that sample boundary;
the cutoff still needs validation on held-out, human-labeled data.

## Evaluation

Five local tests cover off-topic gating, novelty ordering, duplicate handling,
and score bounds. The test suite passes with the offline TF-IDF backend.
Python syntax and Gemini SDK imports were checked; a live Gemini API call was
not tested.

The evaluation uses synthetic submissions, and the relevance threshold was
tuned on that same dataset. These checks confirm the implemented rules on the
demo data, but do not establish generalization. A held-out, human-labeled set
and separate Gemini evaluation are needed for that.

Run the tests with `python -m pytest tests -v`.
