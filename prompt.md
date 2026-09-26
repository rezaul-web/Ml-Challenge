# ROLE
You are acting as the lead ML architect and pair-programmer for a 3-person team competing in the 
Amazon ML Challenge 2026 (Business Entity Resolution track). I will supervise; you will do all 
analysis, design, documentation, and code generation. Treat this as a real production-grade 
submission, not a toy notebook — it will be reviewed by judges alongside our methodology doc.

The full problem statement is attached/pasted below (or in the referenced file). Read it fully 
before doing anything else.

# CONTEXT (do not skip)
- Task: Entity Resolution across 3 noisy business-record sources (Source 1 = deduplicated 
  reference; Source 2 & 3 = records to match against it).
- Output 1: matching_results.tsv (scored on leaderboard).
- Output 2: candidate_pairs.tsv (the exact candidate set fed to the final model at inference — 
  used to audit blocking quality, not separately scored but required).
- Metric: macro-averaged F_0.5 per Source 1 entity (precision weighted 2x over recall). 
  Singletons (no true match) count fully — predicting empty correctly = 1.0, any false match = 0.0.
- Test set introduces a THIRD country (France) unseen in training — pipeline must generalize, 
  not hardcode {US, India}.
- Hard constraints: 
  - No external data/API lookups (disqualification risk) — pure ML/text-similarity on provided 
    data only.
  - Final model must be MIT/Apache-2.0 licensed and ≤8B parameters.
  - Output format is strict TSV with exact column names, no duplicate IDs, IDs must exist in 
    test set, every Source 1 entity must appear exactly once.
  - A `utils/validate_submission.py` exists — our pipeline must pass it before we consider 
    ourselves "done."
  - Final zip needs: output/, code/business_entity_resolution/src+README+requirements.txt, and 
    a filled Documentation_template.md.

# WHAT I NEED FROM YOU — IN THIS ORDER

## Phase 1 — Problem Analysis
Before proposing anything, produce a short written analysis covering:
- Restate the problem in your own words, including what makes it hard (noisy names/addresses, 
  no shared IDs, open-set country field, precision-heavy metric, scale).
- Identify the two distinct sub-problems: (a) blocking/candidate generation, (b) pairwise/set 
  matching classification — and why they need different objectives (recall-max vs precision-max).
- Flag every constraint above that will shape architecture choices (e.g. license+size cap rules 
  out big proprietary embedding APIs; F_0.5 means we should tune classification threshold toward 
  precision; France-in-test means no hardcoded country branching).
- Call out anything ambiguous or risky in the spec that we should decide on explicitly (e.g. how 
  to handle many-to-many matches, how to pick a validation split).

## Phase 2 — Solution Design (propose, don't just default to the obvious)
Give me an optimised end-to-end approach, with brief justification for each choice, covering:
1. **Preprocessing/normalization** for business_name and business_address (per-country aware, 
   but not hardcoded — e.g. legal-suffix normalization tables, transliteration handling, 
   abbreviation expansion).
2. **Blocking / candidate generation strategy** — must scale to billions in theory but at least 
   to the given dataset in practice. Propose concrete techniques (e.g. token/n-gram blocking keys, 
   MinHash/LSH, TF-IDF + approximate nearest neighbor, sorted neighborhood on normalized name+city) 
   and how we'll measure/report recall ceiling and reduction ratio.
3. **Feature engineering** for the matching stage — string similarity families (Jaccard, 
   Levenshtein/edit distance, token-sort/token-set ratio, TF-IDF cosine, phonetic codes), address 
   component similarity, country-match features, and any structural/graph features from the 
   candidate set.
4. **Matching model** — propose a specific, justified choice of a lightweight, open-license 
   (MIT/Apache-2.0), ≤8B-parameter model or classical ML approach (e.g. gradient boosting on 
   engineered features vs a small open sentence-embedding model + classifier vs hybrid). Explain 
   the precision/recall tradeoff and how you'll tune the decision threshold specifically for F_0.5.
5. **Handling singletons and many-to-many matches** explicitly.
6. **Generalization to the France test split** — how the pipeline avoids overfitting to 
   US/India-specific patterns.
7. **Validation strategy** — how we'll hold out part of train_ground_truth.tsv to estimate F_0.5 
   before submitting.

Present this as a concise written proposal I can review before code generation begins.

## Phase 3 — Documentation Deliverable
Generate a `TECH_STACK_AND_APPROACH.md` file (this is separate from the final 
Documentation_template.md) containing:
- Chosen tech stack (languages, libraries, model, versions) with license notes for the final model.
- Architecture diagram (as a text/mermaid diagram) of the full pipeline: raw data → 
  preprocessing → blocking → feature engineering → matching model → post-processing → 
  validation → output files.
- Rationale summary for each major design decision from Phase 2.
- A glossary of the noise patterns we're explicitly handling.

## Phase 4 — Modular Breakdown for a 3-Person Team
Split the codebase into independently developable, clearly-interfaced modules so 3 people can 
work in parallel with minimal merge conflicts. For each module, specify: its responsibility, 
its input/output contract (function signatures / file formats), and its dependencies on other 
modules. Suggested split (adjust as you see fit, but keep it 3-way):
- **Module A — Data & Preprocessing**: ingestion, cleaning/normalization of name & address 
  fields, train/validation split creation, shared data schemas/utilities.
- **Module B — Blocking / Candidate Generation**: everything that produces candidate_pairs.tsv, 
  plus recall/reduction-ratio evaluation against ground truth.
- **Module C — Matching Model & Scoring**: feature engineering on candidates, model 
  training/inference, threshold tuning for F_0.5, generation of matching_results.tsv, and the 
  local F_0.5 scorer + integration with validate_submission.py.
Also define a thin **orchestration layer / config** (owned by whoever finishes first, or shared) 
that wires the 3 modules into one reproducible `run_pipeline.py`.

## Phase 5 — Structured Implementation Plan
Produce a step-by-step build plan (as a checklist/table) mapping each module to concrete tasks, 
in dependency order, with rough effort estimates. Note which tasks can start in parallel on day 1 
vs which block on another module's interface being defined.

## Phase 6 — Code Generation
Only after Phases 1–5 are written and I've had a chance to see them, generate the actual runnable 
code, module by module, matching the plan above:
- Clean, well-commented, production-quality Python.
- Config-driven (no hardcoded paths/countries/thresholds — use a config file).
- Include the local F_0.5 evaluation script and a script to run `validate_submission.py` checks.
- Include `requirements.txt` with pinned versions and a `README.md` with exact run instructions 
  from raw data to final output/ files.
- Make sure the final chosen model is verifiably MIT/Apache-2.0 and ≤8B parameters — state the 
  exact model name/version and license source.

# OUTPUT EXPECTATIONS
- Do not skip straight to code. Complete Phases 1–2 first and present them for my review.
- Be explicit about tradeoffs and why you picked one technique over an equally valid alternative.
- Every design choice must respect the constraints section above — flag it loudly if any proposal 
  risks violating fair-play, license, or size rules.
- Optimise for the actual scored metric (macro F_0.5 with singleton credit), not generic 
  accuracy/F1.