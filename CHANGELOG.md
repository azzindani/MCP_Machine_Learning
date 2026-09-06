# Changelog

All notable changes to this project will be documented in this file.

This file starts at `0.1.2`. Releases before it are described on their
[GitHub release pages](https://github.com/azzindani/MCP_Machine_Learning/releases)
rather than restated here, since reconstructing them after the fact would be a
guess dressed as a record.

---

## [Unreleased]

Twelve commits since `0.1.2`, most of them driven by a tool user's written
review of a 38,576-row credit-risk sweep. The review trained three models, took
the best at 0.9628, and then noticed its top three features were all recorded
*after* the loan resolved. Every tool involved had been honest; none of them had
been useful about it.

### Fixed — sweep round 24, "believe the description"

- **`search_columns`' `dtype` filter did not filter.** The value was compared
  against four literal group names in an if/elif chain with no else, so anything
  else matched no branch, filtered nothing, and the whole frame came back under
  `success: true`. On `Ad_Data.csv`, `dtype="float64"` returned all 16 columns —
  `Date`, `product` and `phase` included — and `dtype="object"` did the same,
  while `has_nulls=True` correctly returned 1.

  `float64` is not an exotic input: it is the string `inspect_dataset` prints in
  its own `dtype` field, so it is exactly what a caller reads off one tool and
  hands to the next.

  **The sibling settled it.** MCP_Data_Analyst exposes `search_columns` with the
  same name and the same description and answered `dtype="float64"` with the
  four numeric columns, because that repo hit this first and fixed it. Two
  identically-described tools disagreed, both said `success: true`, and nothing
  told the caller which one they were holding. The alias table is now ported
  here, an unlisted value is refused with a hint naming the vocabulary, and an
  alias that widens the filter says so — `float64` means `numeric`, which also
  matches integer columns, and a count quietly including them would disagree
  with the word the caller typed. The description names the vocabulary, none of
  which was discoverable before.

  One deliberate divergence, asserted by a test so it cannot become accidental:
  this tier keeps `bool` as its own group where the sibling sorts booleans into
  numeric or object.

### Added

- **Leakage detection on every tool that takes a target.** `train_classifier`,
  `train_regressor`, `train_with_cv`, `compare_models`, `check_data_quality` and
  `evaluate_model` now name features that may already contain the outcome, with
  the evidence for each: how well one feature separates the classes alone,
  whether its *missingness* tracks the target — `last_payment_date` is null
  exactly when nothing was ever repaid — and whether it is named like a
  post-outcome field. The last is labelled a hint and nothing more, because a
  column called `total_payment` might be a budget rather than a settlement.

  The existing guard could not have caught this: it fires at 0.999 and looks for
  a feature that determines the target exactly. 0.9628 is nowhere near that, and
  no single column determined the outcome — the leak was statistical, not
  functional. A check tuned for "obviously impossible" misses "quietly
  meaningless", and the second is the one that ships.

- **`check_data_quality(target_column=…)`.** The review asked for the warning
  here by name, so that it arrives before a model is fit rather than after three
  have been. Suspects are kept out of `alerts` and out of `quality_score`: a
  score that moves depending on whether the caller named a target would be a
  number about the question rather than the data. With no target the response
  says the check did not run, rather than staying silent — a report that scores
  96 and mentions no leakage reads as "none found".

- **`evaluate_model` reports leakage with the score it just produced.** The note
  carries the number it doubts. The check runs on the raw frame, before the
  encoding loop's `.fillna(-1)` turns a null into a number and erases the
  missingness signal for good. `training_leakage_warning` travels from the
  manifest, for the path that skips every other warning: train, `export_model`,
  and someone else evaluates it on a fresh test file.

- **Split provenance in the manifest** — test size, seed, stratification, CV
  folds, and whether the split was time-ordered. A score is a claim about unseen
  data and is only as good as the split behind it; nothing in the manifest let a
  reader tell which kind they had.

- **`read_model_report(top_n, skip_encoding_map)`**, with `skip_encoding_map`
  defaulting to **True**. The report used to return a 28,000-entry encoding map
  inline.

### Fixed

- **`export_model` deleted the training record it was meant to ship.** With no
  `output_dir` the destination manifest *is* the training manifest, and it was
  replaced wholesale with an export descriptor — losing `split`,
  `encoding_map_path`, `feature_defaults`, `hyperparameters`, `leakage_warning`,
  `n_classes`, `scaler` and `model_key`. The snapshot guard above it skips the
  same-path case, so there was not even a backup. It returned `success: true`,
  the manifest it left behind was valid JSON with plausible contents, and the
  loss was invisible unless you knew which keys had been there a moment before.
  An export descriptor is extra information about a file, not a replacement for
  its provenance.

- **A smoke assertion could not tell "unreadable" from "absent"** and reported
  both as absent. Fixing it to print the manifest's actual keys is what exposed
  the export defect above, in one line, after three green CI runs had passed
  over it.

- **`compare_models` saved the score and not how it was produced.**
  `_medium_train.py` wrote manifests with no split provenance at all.

- **The manifest stopped being one column's encoding map.** Above 200
  categories the map moves to a `.encoding_map.json` sidecar. The `.pkl` keeps
  the full metadata, so a model shipped on its own still predicts.

- **Three scorers in two repos, none naming its denominator.** One file scored
  5.6, 41 and 89 depending on who asked. `shared/quality.py` is now one module,
  byte-identical with MCP_Data_Analyst's copy, and the score arrives with its
  parts: `{completeness, validity, uniqueness, drift}`.

- **The docstring gate measured the first line and claimed to measure the
  docstring**, so a long second line passed a cap meant to protect every
  client's `tools/list`.

- **The memory canary measured pages macOS had compressed away**, then measured
  pages it had evicted. The margin went into the allocation, not the assertion —
  lowering the ceiling would have turned CI green by weakening the guard.

### Changed

- The README listed `filter_rows` and `merge_datasets` as ml-medium tools.
  Both exist in the engine; neither is registered, so a caller who read the
  table burned a loop iteration on a tool that was never there. Removed, with a
  pointer to the sibling server that does have them.
