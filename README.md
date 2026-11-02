# Defect Triage

[![CI](https://github.com/chinthanshetty07/defect-triage/actions/workflows/ci.yml/badge.svg)](https://github.com/chinthanshetty07/defect-triage/actions/workflows/ci.yml)

Classifies free-text rolling-stock defect reports ("BC presure low on coch C3, urgent") by **subsystem** and
**severity**, routes safety-critical ones to a safety desk, shows the technician the most similar past cases and
how they were fixed, and sends anything the model is unsure about to a person.

It combines a classical NLP baseline (scikit-learn, pandas, NumPy) with an optional GenAI classifier (Claude
structured outputs, given retrieved past cases as examples), and evaluates both the way an operator would care
about: **how many genuinely high-severity faults were caught**, not just overall accuracy.

An independent learning project in the kind of engineering domain Cyient works in (rail and transportation). It
is not affiliated with Cyient or any operator, and **all reports are synthetic or hand-written**.

## Pipeline

```
report -> TF-IDF (words + char n-grams) -> logistic regression -> subsystem, severity, confidence
       -> optional: Claude with the 4 most similar past reports as examples (falls back to the baseline)
       -> safety rules (escalate only, never downgrade) -> queue + needs_human_review
       -> similar past cases with their fixes
```

- **Character n-grams** make the baseline robust to how maintainers type: "brk", "presure", "coch", "dr".
- **Safety rules only escalate.** Brakes, train protection (ATP/TPWS) and running gear are always high; red-flag
  words (smoke, burning smell, crack, arcing, stuck open…) escalate any report. A rule never lowers what the
  model said, because a missed brake fault costs far more than a technician checking a door that was fine.
- **Confidence gate.** Below 0.6 confidence the report goes to a person instead of being auto-routed.
- **Time-based split.** The model trains on older reports and is tested on newer ones.

## Results (offline, `python -m triage.evaluate`)

| Test set | Subsystem accuracy | Severity accuracy | High-severity recall (model) | High-severity recall (with rules) | Auto-routed |
|---|---|---|---|---|---|
| Templated synthetic reports (225, time split) | 98.7% | 83.1% | 96.8% | 100% | 87% |
| **Hand-written, unseen phrasings (40)** | **42.5%** | **30.0%** | **44.4%** | **72.2%** | **15%** |

The first row looks great and means little: the generator's phrasings repeat, so the test set is close to the
training set. The second row is the honest one. Forty reports written in wording the generator never produced
("train overshot platform despite full service brake", "sparks seen from roof equipment") cut the baseline to
42.5%. Two things still work as intended:

- the safety rules lift high-severity recall from 44% to 72%, and
- the confidence gate sends 85% of those unfamiliar reports to a person rather than auto-routing them.

That gap is exactly where a language model should help: it knows that "retardation poor" is about brakes and
"collector head shoe" is a pantograph part without having seen those words in training. The Claude path is
built and tested (with a fake client, including timeouts, refusals and malformed output), but **I haven't
measured it yet**, because that needs an API key and costs money. To run it:

```bash
export ANTHROPIC_API_KEY=...        # never committed
python -m triage.evaluate --llm 40  # Claude on the 40 unseen-phrasing reports
```

The interesting question it answers: does retrieval-augmented Claude close the gap on unseen phrasings
without losing the baseline's recall on safety-critical faults?

## Run it

```bash
pip install -r requirements-dev.txt
pytest -q                                       # 14 tests
python -m triage.evaluate                       # writes reports/metrics.json
python -m triage.cli "BC presure low on coch C3, urgent"
```

## What I'd do next

- Measure the Claude path on the unseen set and compare cost per report against the accuracy gain.
- Replace the generator with real (anonymised) reports; the evaluation code doesn't change.
- Per-class thresholds: be stricter on safety-critical classes than on lighting.
- A fine-tuned small transformer as a middle option between TF-IDF and an API model.

Built with Claude Code; the tests and the unseen-phrasing set are how I checked what it wrote.
