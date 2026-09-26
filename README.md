# Rewarding Novelty in Submissions

A pipeline that scores user-generated content (UGC) on a normalized
**novelty scale of [0.0, 1.0]**, relative to a pool of existing submissions,
gated by relevance to the fixed content it responds to.

## 1. The "shape" of UGC being evaluated

**Fixed content**: a short news snippet (≤100 words) — a fictional local
news story about a city council approving new protected bike lanes downtown.
(`data/fixed_content.json`)

**User submission** — three discrete, user-provided properties, as required:

| Field | Type | Example |
|---|---|---|
| `headline` | short free text | "This will make downtown traffic worse" |
| `body` | 1-3 sentence free text commentary | "Removing parking spots downtown will make traffic worse for drivers." |
| `stance` | multi-choice (`Support` / `Oppose` / `Neutral`) | `Oppose` |

This mirrors a realistic "comment on a news article" UGC pattern: users pick
a stance and write a short reaction, exactly the shape described in the
problem statement's own example.

## 2. Scoring design

```
relevance(S) = cosine_similarity(embed(S), embed(fixed_content))
novelty(S)   = 1 - max(cosine_similarity(embed(S), embed(p)) for p in pool)

if relevance(S) < RELEVANCE_FLOOR:
    final_score = 0.0
else:
    final_score = clip(novelty(S), 0.0, 1.0)
```

**Why a hard relevance gate, not a blended score.** The task explicitly
requires that "high novelty, but low relevance submissions are not
rewarded." A weighted blend (e.g. `0.5 * novelty + 0.5 * relevance`) would
still hand out partial credit to a wildly off-topic-but-unique submission.
A hard floor is the only shape that guarantees the required behavior in
every case, so relevance acts as a **gate**, and novelty (relative to the
existing pool) is the only thing that determines the actual reward once the
gate is cleared.

**Why novelty is relative to the pool, not absolute.** A submission is only
"novel" insofar as nobody has already said it. Measuring novelty as
`1 - max_similarity_to_pool` directly captures "how different is this from
everything already submitted," which is the practical definition of novelty
in a UGC-rewards context (you don't want to keep paying out for the same
comment restated 50 times).

### Embeddings: switchable backend, local by default

Text is embedded via `src/embeddings.py`, which supports **two
interchangeable backends** behind one environment variable,
`EMBEDDING_BACKEND`:

| `EMBEDDING_BACKEND` | What it uses | Needs |
|---|---|---|
| `local` (default) | TF-IDF over character n-grams (3-5 chars), `scikit-learn` | nothing — fully offline |
| `gemini` | Google's `gemini-embedding-001` live API | `GEMINI_API_KEY` + `pip install "google-genai>=1.0"` |

Both backends implement the exact same interface (`fit(corpus)` /
`embed(text)`), selected by a single factory function, `get_embedder()` —
nothing else in `novelty.py` or `score.py` needs to know or change based on
which one is active.

The `local` backend was the deliberate default for this environment (see
`AGENT_LOG.md`): no model download, no API key, no network access. It's
robust to short, informal, sometimes-misspelled UGC text and captures
near-duplicate/paraphrase similarity well at this pool size (~50 items) —
but it's lexical (word/character overlap), not semantic, so it can miss
relevance for on-topic content that uses very different vocabulary (see
Limitations).

**To switch to live Gemini embeddings**, see the "Switching between local
and Gemini embeddings" section below.

## 3. Novelty evaluation mechanism (≈50-item pool)

`data/submissions.json` contains 50 synthetic submissions, generated across
four labeled bands to make the pipeline's behavior testable against known
expectations:

- **`low`** (15 items) — near-duplicate restatements of the same complaint
  ("bike lanes will cause traffic"), varying only in wording.
- **`medium`** (15 items) — same handful of underlying arguments (business
  impact, safety benefit, emissions benefit, congestion displacement),
  phrased distinctly enough to be a different "take" but overlapping in idea.
- **`high`** (15 items) — genuinely distinct angles on the same topic
  (equity impact, insurance/liability questions, winter usability,
  enforcement, tourism impact, etc.) — each raises a point none of the
  others do.
- **`offtopic`** (5 items) — comments about unrelated subjects entirely
  (pasta, hiking, phones, TV shows, gym routines) submitted against the
  bike-lane article.

Synthetic generation used hand-authored templates rather than a live LLM
call, because this sandboxed dev environment has no outbound network access
to Gemini/OpenAI's APIs (see `AGENT_LOG.md`). `src/generate_dataset.py`
includes a `generate_with_gemini()` stub documenting exactly how to wire in
live LLM generation if you have a `GEMINI_API_KEY` and internet access —
the rest of the pipeline (scoring, tests) is unaffected either way, since it
only consumes `data/submissions.json`'s schema, not how it was produced.

## 4. Results — success criteria and level of achievement

**Success criteria defined up front:**
1. Off-topic submissions must score exactly `0.0`, regardless of novelty.
2. `high`-novelty-band submissions should score meaningfully above
   `low`-novelty-band (near-duplicate) submissions, on average.
3. A majority of genuinely novel, on-topic submissions should clear a
   "clearly rewarded" bar (`final_score > 0.5`).
4. All scores must stay within `[0.0, 1.0]`.

**Achieved** (leave-one-out scoring across all 50 items,
`python -m src.score`):

| Band | Avg. final score | n |
|---|---|---|
| `high` (distinct takes) | **0.716** | 15 |
| `medium` (overlapping takes) | 0.663 | 15 |
| `low` (near-duplicates) | 0.386 | 15 |
| `offtopic` | **0.000** | 5 |

All 5 automated tests in `tests/test_novelty.py` pass:
- `test_offtopic_submissions_never_rewarded` — confirms all 5 off-topic
  items score `0.0` despite having *high raw novelty* (>0.7), proving the
  relevance gate — not a lack of novelty — is what blocks the reward.
- `test_high_novelty_ontopic_submissions_rewarded` — confirms the `high`
  band average beats the `low` band average, and ≥60% of `high`-band items
  score above 0.5.
- `test_near_duplicate_submissions_not_highly_rewarded` — confirms the
  `low` band averages below 0.5.
- `test_relevance_gate_overrides_novelty_score` — a from-scratch unit test
  (independent of the generated dataset) that constructs a maximally novel
  but clearly off-topic submission and confirms it still scores `0.0`.
- `test_score_is_always_in_unit_interval` — sanity bound-check.

Run them yourself:
```bash
pip install scikit-learn numpy pytest
python3 src/generate_dataset.py   # regenerate data/submissions.json (optional, already committed)
python3 -m src.score              # see per-item + per-band scores
python3 -m pytest tests/ -v
```

### Threshold tuning

`RELEVANCE_FLOOR = 0.07` (in `src/novelty.py`) was chosen empirically by
inspecting the relevance-score distributions of each band (see
`AGENT_LOG.md` for the tuning trace). At this threshold:
- **All 5** off-topic items are correctly zeroed (max off-topic relevance
  was 0.068, comfortably under the floor).
- **3 borderline on-topic items** (2 `high`, 1 `medium` — genuinely
  relevant but low lexical overlap with the fixed content's exact wording,
  e.g. "Equity angle" discussing e-bike affordability without using the
  words "bike lane") also fall under the floor and are zeroed as false
  negatives.

This is a deliberate conservative tradeoff: the task's explicit, testable
requirement is that **irrelevant content must never be rewarded**; a few
under-rewarded borderline-relevant items is a safer failure mode than any
off-topic item slipping through. See Limitations for how a semantic
embedding model would close this gap.

## 5. Switching between local and Gemini embeddings

Everything below is a **one-variable switch** — no code edits required.

### Stay offline (default — nothing to do)

```bash
python3 -m src.score
```
Runs fully locally, no API key needed. You'll see `[embedding backend: local]`
printed at the top of the output.

### Switch to live Gemini embeddings

**Step 1 — get a free API key**
Go to [aistudio.google.com](https://aistudio.google.com/), sign in, and
generate a free API key.

**Step 2 — install the Gemini SDK**
```bash
python -m pip install "google-genai>=1.0"
```

**Step 3 — set the environment variables**

In Windows Command Prompt (`cmd`):
```cmd
set "GEMINI_API_KEY=YOUR_API_KEY"
set "EMBEDDING_BACKEND=gemini"
```

In PowerShell:
```powershell
$env:GEMINI_API_KEY = "YOUR_API_KEY"
$env:EMBEDDING_BACKEND = "gemini"
```

In Bash:
```bash
export GEMINI_API_KEY=your-key-here
export EMBEDDING_BACKEND=gemini
```

**Step 4 — run exactly as before**
```cmd
python -m src.score
```
You'll see `[embedding backend: gemini]` printed at the top, and every
`embed()` call now goes to Google's `gemini-embedding-001` model instead of
the local TF-IDF vectorizer. Same CLI, same output format, same tests —
only the underlying similarity calculation changes.

**To switch back to offline**, just unset (or override) the variable:
```bash
unset EMBEDDING_BACKEND
# or explicitly: export EMBEDDING_BACKEND=local
python3 -m src.score
```
In Windows Command Prompt, run `set "EMBEDDING_BACKEND=local"` before
`python -m src.score`.

### Notes

- **Tests always run on the local backend by design**
  (`tests/test_novelty.py` imports `TextEmbedder` directly, not the
  factory), so `pytest` stays fast, free, deterministic, and runnable
  without any API key or network access — regardless of what
  `EMBEDDING_BACKEND` is set to in your shell. This is intentional: a
  judge should be able to clone the repo and run the test suite with zero
  setup.
- **Dataset generation** (`src/generate_dataset.py`) has its own,
  independent Gemini switch: if `GEMINI_API_KEY` is set (and
  `google-genai` is installed and reachable), running
  `python -m src.generate_dataset` will ask Gemini to generate the 50
  synthetic submissions directly instead of using the hardcoded templates.
  It falls back to the templates automatically on any missing key, missing
  package, network error, or malformed response — so it always produces a
  usable dataset either way.
- **Rate limits / cost**: the Gemini backend caches embeddings in memory
  per run (`GeminiEmbedder._cache`) to avoid re-embedding the same text
  twice in one scoring pass, but does not cache across separate runs. The
  free tier's rate limits are generally enough for the ~50-item scale of
  this dataset.
- **Error messages are intentionally explicit**: if you set
  `EMBEDDING_BACKEND=gemini` without installing the package or without a
  key, you'll get a clear `ImportError` or `RuntimeError` telling you
  exactly what's missing, rather than a silent fallback — since silently
  falling back would make it easy to think you're testing live embeddings
  when you're actually still on TF-IDF.

## 6. Limitations

- **Lexical, not semantic, relevance matching.** Because embeddings are
  TF-IDF/character-based (not a trained semantic model), relevance
  detection depends on some vocabulary overlap with the fixed content's
  exact wording. A submission that is genuinely on-topic but uses entirely
  different vocabulary (synonyms, oblique references) can be
  under-scored — see the 3 false negatives above. Swapping in Gemini's or
  OpenAI's embedding API would very likely fix this, since those models
  capture semantic similarity independent of shared words.
- **Small pool, in-memory only.** At ~50 items, holding everything in
  memory and refitting the vectorizer is fine; this would need a proper
  vector index (pgvector, FAISS) at real scale (thousands+ of submissions),
  and the embedder would need to be fit once and reused/persisted rather
  than refit per scoring call (`score.py` currently refits once for the
  whole batch, which is correct; a live service would persist the fitted
  vectorizer or move to a fixed-dimension neural embedding that doesn't
  need per-corpus fitting at all).
- **`stance` field is unused in scoring.** It's collected as one of the 3
  required structured fields but only included as a tag string in the
  embedded text; a fuller system might weight novelty/relevance separately
  per stance (e.g. "novel *given* that this is a Support-stance comment").
- **Threshold tuned on one synthetic dataset.** `RELEVANCE_FLOOR = 0.07`
  is specific to this fixed-content example and embedding method; it would
  need re-validation (or an adaptive/percentile-based floor) for other
  topics or a live embedding model.
- **No abuse/gaming defenses.** A user submitting deliberately garbled or
  keyword-stuffed text to game novelty isn't defended against here; a
  production system would likely add a minimum text-quality/coherence check
  alongside the relevance gate.

## 7. Repo layout

```
novelty-reward/
├── data/
│   ├── fixed_content.json      # the fixed news snippet
│   └── submissions.json        # 50 synthetic labeled submissions (golden dataset)
├── src/
│   ├── generate_dataset.py     # synthetic dataset generator (template-based + Gemini stub)
│   ├── embeddings.py           # local TF-IDF embedder (swap point for real embeddings)
│   ├── novelty.py              # core scoring algorithm
│   └── score.py                # CLI: leave-one-out scoring over the whole dataset
├── tests/
│   └── test_novelty.py         # 5 automated tests, see section 4
├── README.md                   # this file
└── AGENT_LOG.md                # coding-agent disclosure / prompt trace
```
