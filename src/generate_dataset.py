"""
Generates a synthetic pool of ~50 user-generated submissions responding to the
fixed content in data/fixed_content.json.

Each submission has the required 3 discrete user-provided properties:
    - headline   (short text)
    - body       (1-3 sentence commentary)
    - stance     (multi-choice: Support / Oppose / Neutral)

Submissions are deliberately generated across four bands so we can validate
the scoring pipeline against known-good expectations in tests:

    "low"      -> near-duplicates of each other (should score LOW novelty)
    "medium"   -> paraphrases / same idea, different wording (MEDIUM novelty)
    "high"     -> genuinely distinct takes on the topic (HIGH novelty)
    "offtopic" -> irrelevant to the fixed content (should be REJECTED via the
                  relevance floor regardless of how "novel" they are)

LLM usage note
---------------
This script is written so that if you have a GEMINI_API_KEY (see
https://aistudio.google.com/) it will call Gemini to generate more varied,
naturalistic text for each band. Without a key (e.g. offline / sandboxed
environments where the Gemini API isn't reachable), it falls back to a
deterministic template-based generator so the pipeline still runs end-to-end.
The fallback is what was actually used to produce data/submissions.json in
this repo, since this dev environment has no outbound access to Google's API.
"""

import json
import os
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIXED_CONTENT_PATH = DATA_DIR / "fixed_content.json"
SUBMISSIONS_PATH = DATA_DIR / "submissions.json"

STANCES = ["Support", "Oppose", "Neutral"]

random.seed(42)

# ---------------------------------------------------------------------------
# Template banks per novelty band. All "low" and "medium" entries are built
# from small variations of a shared idea so they cluster together in
# embedding space; "high" entries each express a structurally different
# argument/angle; "offtopic" entries are about unrelated subjects entirely.
# ---------------------------------------------------------------------------

LOW_BASE_HEADLINE = "Bike lanes will just cause more traffic"
LOW_BASE_BODY = (
    "This is a bad idea because removing parking spots will make traffic "
    "worse for everyone driving downtown."
)
LOW_VARIANTS = [
    ("Bike lanes will just cause more traffic",
     "This is a bad idea because removing parking spots will make traffic worse for everyone driving downtown."),
    ("These bike lanes will make traffic worse",
     "Bad idea overall, since taking away parking spaces will make downtown traffic worse for drivers."),
    ("More traffic because of bike lanes",
     "I think this is a bad plan; losing parking will worsen the traffic situation for everyone who drives downtown."),
    ("Bike lanes = worse traffic",
     "Not a good idea. Removing the parking spots is going to make traffic worse downtown for drivers."),
    ("This will make downtown traffic worse",
     "Bad call, honestly. Cutting parking spots downtown will just make the traffic worse for drivers."),
    ("Traffic will get worse with these lanes",
     "It's a bad idea because you're removing parking and that will make the downtown traffic worse for cars."),
    ("Worse traffic incoming from bike lanes",
     "Bad idea in my view — fewer parking spots downtown means worse traffic for everyone who drives."),
    ("Bike lanes will hurt traffic flow",
     "This seems like a bad idea since removing parking downtown will make traffic worse for drivers overall."),
    ("Downtown traffic will worsen",
     "Bad idea because removing parking spots downtown is going to make the traffic worse for drivers."),
    ("Traffic congestion will increase",
     "This is a bad idea — losing parking spots downtown will make the traffic worse for everyone driving."),
    ("Bike lanes are a bad traffic move",
     "Bad idea overall since removing the downtown parking will worsen traffic for drivers."),
    ("Cars will suffer from these lanes",
     "Not great — removing parking spots downtown will make the traffic worse for everyone who drives."),
    ("This traffic plan is flawed",
     "Bad idea, taking away parking downtown is just going to make traffic worse for drivers."),
    ("Bike lanes hurt drivers",
     "Bad idea since removing downtown parking spots will make traffic worse for everyone driving."),
    ("Parking loss means traffic gain",
     "This is a bad idea; removing parking spots downtown will make the traffic situation worse for drivers."),
]

MEDIUM_VARIANTS = [
    ("Small businesses will lose customers",
     "Oppose", "Shoppers who drive downtown may just stop coming if parking gets harder to find near stores."),
    ("Losing foot traffic to bike lanes",
     "Oppose", "Store owners near the new lanes are worried customers will avoid downtown once parking shrinks."),
    ("Downtown shops could take a hit",
     "Neutral", "It's plausible fewer parking spaces means fewer drive-in customers for small downtown retailers."),
    ("Retailers fear the parking cuts",
     "Oppose", "Local shop owners say losing curbside parking could push regular customers to suburban strip malls."),
    ("Will shoppers still come downtown?",
     "Neutral", "With less parking, some customers who drive in for quick errands might just skip downtown altogether."),
    ("Business worries about the redesign",
     "Oppose", "Several small business owners are concerned the lane changes will quietly erode their walk-in traffic."),
    ("Safety gains are worth it",
     "Support", "Fewer cyclist injuries matters more than a few lost parking spots near downtown storefronts."),
    ("Protecting cyclists should come first",
     "Support", "The injury data makes this an easy call — rider safety should outweigh convenience parking downtown."),
    ("Cyclist safety is the real priority",
     "Support", "Given how many cyclists get hurt downtown, protected lanes seem like an overdue safety upgrade."),
    ("Emissions cuts matter more than parking",
     "Support", "Reducing car trips downtown by making biking safer is a reasonable tradeoff for a few parking spots."),
    ("Climate benefits outweigh the downside",
     "Support", "Cutting emissions from downtown car trips seems worth the inconvenience of losing some parking."),
    ("This is mostly about emissions, not traffic",
     "Support", "The bigger win here is fewer downtown car trips and lower emissions, not the parking argument."),
    ("Congestion will just shift elsewhere",
     "Neutral", "Cutting lanes downtown for bikes could just push the same car congestion onto side streets instead."),
    ("Traffic doesn't disappear, it moves",
     "Neutral", "Removing driving lanes downtown for bikes likely reroutes the same congestion to nearby residential streets."),
    ("Side streets will absorb the overflow",
     "Oppose", "If downtown lanes shrink for bikes, all that car traffic has to spill onto quieter side streets nearby."),
]

HIGH_VARIANTS = [
    ("Who actually gets a vote here?",
     "Neutral", "The council approved this after limited public hearings — the real question is whether residents outside downtown had any say at all."),
    ("$40M could fix transit instead",
     "Oppose", "For $40M, the city could subsidize bus fares citywide for years — bike lanes feel like a narrow use of a big budget."),
    ("Winter usability is being ignored",
     "Oppose", "Nobody's addressing how these lanes function once snow season hits; plowing bike infrastructure is often an afterthought here."),
    ("Delivery trucks need a real plan",
     "Neutral", "Commercial delivery access downtown isn't mentioned at all — trucks double-parking in bike lanes could become the new normal."),
    ("This mirrors a failed pilot from 2019",
     "Oppose", "A similar corridor redesign was tried and quietly reversed a few years back after merchant pushback; what's different this time?"),
    ("Equity angle nobody's raising",
     "Support", "Lower-income residents who can't afford e-bikes or cargo bikes may benefit least from this, even though they're subsidizing it via taxes."),
    ("Insurance and liability shift",
     "Neutral", "Once the city builds protected lanes, does liability for cyclist-vehicle collisions shift toward the city instead of drivers? That's untested legally."),
    ("Compare this to peer cities' data",
     "Support", "Cities that added similar protected lanes saw injury drops within 18 months — the timeline here roughly matches that pattern."),
    ("Construction disruption is the real cost",
     "Oppose", "Two years of construction chaos downtown could do more damage to small businesses than the permanent parking loss ever will."),
    ("Bike theft infrastructure gap",
     "Neutral", "None of this addresses secure bike parking downtown — safer lanes won't matter if riders still can't safely park at their destination."),
    ("Tourism impact is underexplored",
     "Neutral", "Downtown draws a lot of weekend tourist traffic by car; nobody's modeled whether reduced parking hurts weekend tourism revenue."),
    ("This could be a template for other districts",
     "Support", "If this works, it's a testable model the city could roll out to the two other congested commercial corridors it has."),
    ("The real fight is about curb space, not bikes",
     "Neutral", "This is less about cyclists and more a proxy war over who controls scarce curb space: delivery, parking, or bikes."),
    ("Enforcement is the missing piece",
     "Oppose", "Protected lanes only work if the city actually enforces them; right now enforcement of existing bike lanes downtown is basically nonexistent."),
    ("Property values near the corridor",
     "Support", "Cities with similar redesigns often saw nearby property values rise over time as the area became more walkable and attractive."),
]

OFFTOPIC_VARIANTS = [
    ("My favorite pasta recipe",
     "Neutral", "I made a great carbonara last night using guanciale instead of bacon, and it completely changed the flavor."),
    ("New phone review",
     "Support", "The battery life on the latest flagship phone is impressive, easily lasting a full day of heavy use."),
    ("Weekend hiking trip",
     "Neutral", "We hiked up to the ridge trail this weekend and the fall colors were absolutely stunning up there."),
    ("Thoughts on the new season of my show",
     "Oppose", "The latest season felt rushed compared to earlier ones, especially the finale which wrapped up too quickly."),
    ("Gym routine update",
     "Support", "Switched to a push-pull-legs split this month and I'm already noticing better recovery between sessions."),
]


def _build_low_band():
    entries = []
    for i, (headline, body) in enumerate(LOW_VARIANTS):
        entries.append({
            "id": f"low-{i:02d}",
            "headline": headline,
            "body": body,
            "stance": "Oppose",
            "expected_novelty_band": "low",
        })
    return entries


def _build_medium_band():
    entries = []
    for i, (headline, stance, body) in enumerate(MEDIUM_VARIANTS):
        entries.append({
            "id": f"medium-{i:02d}",
            "headline": headline,
            "body": body,
            "stance": stance,
            "expected_novelty_band": "medium",
        })
    return entries


def _build_high_band():
    entries = []
    for i, (headline, stance, body) in enumerate(HIGH_VARIANTS):
        entries.append({
            "id": f"high-{i:02d}",
            "headline": headline,
            "body": body,
            "stance": stance,
            "expected_novelty_band": "high",
        })
    return entries


def _build_offtopic_band():
    entries = []
    for i, (headline, stance, body) in enumerate(OFFTOPIC_VARIANTS):
        entries.append({
            "id": f"offtopic-{i:02d}",
            "headline": headline,
            "body": body,
            "stance": stance,
            "expected_novelty_band": "offtopic",
        })
    return entries


GEMINI_GENERATION_PROMPT = """\
You are generating a synthetic test dataset for a content-novelty scoring
system. The fixed content being commented on is:

"{fixed_content}"

Generate a JSON array of exactly 50 objects, each representing a synthetic
user comment on this content, with this exact schema:
  - "headline": short string, the user's own short title for their comment
  - "body": 1-3 sentences of commentary
  - "stance": one of "Support", "Oppose", "Neutral"
  - "expected_novelty_band": one of "low", "medium", "high", "offtopic"

Distribution required: 15 "low" (near-duplicate restatements of the same
1-2 complaints, varying only in wording), 15 "medium" (a handful of distinct
underlying arguments, phrased differently), 15 "high" (each expresses a
genuinely distinct angle/argument nothing else in the set raises), and 5
"offtopic" (comments about completely unrelated subjects, submitted as if
by mistake or spam).

Return ONLY the raw JSON array, no markdown fences, no commentary.
"""


def generate_with_gemini(fixed_content: str):
    """
    Live-LLM path: if GEMINI_API_KEY is set (and google-genai is
    installed + the API is reachable), calls Gemini to generate the 50
    synthetic submissions instead of using the hardcoded templates below.

    Falls back to the template-based generator (returns None) on any
    missing key, missing package, network error, or malformed response --
    so `main()` always produces a usable dataset either way.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
    except ImportError:
        print("google-genai not installed; falling back to templates. "
              "Install with: pip install google-genai")
        return None

    try:
        client = genai.Client(api_key=api_key)
        model_name = "gemini-2.5-flash"
        prompt = GEMINI_GENERATION_PROMPT.format(fixed_content=fixed_content)
        response = client.models.generate_content(model=model_name, contents=prompt)
        raw = response.text.strip()
        # Strip markdown code fences if the model added them despite instructions.
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[len("json"):]
        dataset = json.loads(raw)

        required_keys = {"headline", "body", "stance", "expected_novelty_band"}
        for i, item in enumerate(dataset):
            if not required_keys.issubset(item.keys()):
                raise ValueError(f"item {i} missing required keys: {item}")
            item.setdefault("id", f"gemini-{i:02d}")

        print(f"Generated {len(dataset)} submissions via Gemini ({model_name}).")
        return dataset

    except Exception as e:  # network error, bad JSON, rate limit, etc.
        print(f"Gemini generation failed ({e!r}); falling back to templates.")
        return None


def main():
    fixed_content_text = json.loads(FIXED_CONTENT_PATH.read_text())["text"]
    dataset = generate_with_gemini(fixed_content_text)
    if dataset is None:
        dataset = (
            _build_low_band()
            + _build_medium_band()
            + _build_high_band()
            + _build_offtopic_band()
        )
        random.shuffle(dataset)

    SUBMISSIONS_PATH.write_text(json.dumps(dataset, indent=2))
    print(f"Wrote {len(dataset)} submissions to {SUBMISSIONS_PATH}")
    band_counts = {}
    for d in dataset:
        band_counts[d["expected_novelty_band"]] = band_counts.get(d["expected_novelty_band"], 0) + 1
    print("Band breakdown:", band_counts)


if __name__ == "__main__":
    main()
