# Changelog

All notable changes to this project will be documented in this file.

This file starts at `0.1.2`. Releases before it are described on their
[GitHub release pages](https://github.com/azzindani/MCP_Machine_Learning/releases)
rather than restated here, since reconstructing them after the fact would be a
guess dressed as a record.

---

## [Unreleased]

### Fixed — a prediction names its class

- `get_predictions`, `predict_single` and `batch_predict` answer a classifier
  trained on a text target with the class name (`prediction: "Facebook Ads"`),
  the code beside it (`class_code`), and `class_labels` in code order, which
  also names each probability; `predict_single` keys its probabilities by
  name and `batch_predict` writes names to the CSV and its distribution. They
  answered bare codes (`prediction: 1`, `probabilities: {'0': 0.0, '1': 1.0}`).
  A numeric target is unchanged.
- `evaluate_model` encodes a text target with the model's own map. It fitted
  a new encoder to the evaluation file, so a file holding one class scored
  every row against the wrong code.

### Added — upload URLs, off by default

- With `MCP_UPLOAD_URLS=1` and `MCP_UPLOAD_BASE_URL`, a path from the caller's
  sandbox is refused with a single-use URL minted for that file, and one
  `curl -T` from the sandbox writes it to the inbox -- no bytes through the
  model. The token is HMAC-signed (`MCP_UPLOAD_SECRET`), fixes the file name,
  expires in 15 minutes, writes once, and is capped at `MCP_MAX_UPLOAD_MB`; a
  forged, re-signed, expired or spent one writes nothing. The route,
  `/upload/<token>`, is mounted at the root outside the API key and answers
  404 while uploads are off.

### Added — one endpoint, four tools

- `/mcp` serves the whole surface as four domain tools -- `ml_data`,
  `ml_train`, `ml_predict`, `ml_report` -- each an `action` (one of the 33
  tier tools, by its own name) plus an `args` object whose properties say
  which actions take them, in the Pipeline server's shape. A model connected
  to every tier read 33 tool names on every turn; here it reads four. Each
  action runs the tier tool itself (`shared/domain_tools.py`, byte-identical
  with MCP_Data_Analyst's and MCP_Microsoft_Office's), so validation, inline
  files, model signing and answers are identical. Discovery for `/mcp` goes
  to the OAuth bridge. The tier endpoints keep serving unchanged.

### Added — a file too big for one call arrives in parts

- Each part is an inline file with `part=<i>/<n>;sha256=<of the whole file>`
  in its header. Parts wait in the inbox's hidden `.parts` folder, arrive in
  any order, and a part sent again replaces itself; until the last one lands
  the tool answers `tool_ran: false` with the parts still missing. The joined
  file must match its SHA-256 or nothing is kept. Capped at
  `MCP_MAX_UPLOAD_MB` (default 100); an upload left unfinished for an hour is
  dropped.

### Added — a file's bytes may go wherever a path goes

- `data:text/csv;name=sales.csv;base64,<bytes>` in place of a path is saved to
  `MCP_OUTPUT_DIR/inbox` under its name before the tool runs, and the tool
  reads that path. A wrapper on every tier does it, so no tool's schema changes
  and no tool echoes the bytes back. Capped at `MCP_MAX_INLINE_MB` (default 10)
  before decoding; a name cannot leave the inbox; the same bytes sent twice are
  one file, and a taken name is never overwritten. For a caller whose file is
  in its own sandbox -- a claude.ai upload -- with no link to give; the
  caller's-side refusal names this route first.

### Added — a file on the caller's side reaches the server, or the refusal says how

- A Google Drive, Docs/Sheets/Slides, Dropbox, GitHub or GitLab share link is
  rewritten to the address that serves the file (a Sheet to its CSV export,
  the tab in `gid` when named). As the browser shows them, each answered with
  a web page, which was saved as `data.csv` and parsed.
- A web page served where a file was asked for is refused, not written: a
  share link that is not public answers with a sign-in page even at the right
  address. A URL may serve a page only when its name says it is one.
- A path from the caller's side -- `/mnt/user-data/…` (a claude.ai upload),
  `/home/claude/…`, `/mnt/data/…`, a Windows drive or `/Users/…` on a server
  that is neither -- is refused as what it is, naming the routes in that work
  on this server, instead of "outside the folders this server can use".

### Security — five output paths still wrote wherever they were pointed

- The confinement below covered what tools read and missed five places they
  write. `batch_predict` resolved its output with a bare `Path(...).resolve()`,
  so a relative path landed beside the process and an absolute one anywhere.
  `export_model`, `run_preprocessing` and the filter and merge outputs wrapped
  `resolve_path` in `except ValueError: use the raw path` -- written before
  confinement existed, and since the refusal is a `ValueError` here, it turned
  every refused output into a write. All five now refuse, with nothing written,
  and a relative output lands in the data folder. Found by driving the deployed
  server directly: `batch_predict(output_path="sweep/preds.csv")` tried to
  create `/app/sweep`.
- A refused path is now an answer in the usual failure shape (`success: false`,
  `op`, `error`, `hint`) from every tool. Clustering and anomaly labels,
  `split_dataset`, the HTML reports and model outputs let the refusal escape,
  so the caller saw "Error executing tool" with no hint. Nothing was written
  either way.

### Security — a deployed server reads and writes only inside the folders it serves

- Every tool resolved its path with `Path(raw).resolve()`, so any authenticated
  caller of an HTTP deployment could name any file the container could read: a
  dataset anywhere, a model file (a pickle, loaded as code), or
  `/proc/self/environ` with the API keys. Model paths, the HTML layout/theme
  outputs and several preprocessing outputs skipped the resolver entirely.
- With `MCP_CONFINE_PATHS` on (set by the HTTP transport and compose) a path
  must lie inside `MCP_OUTPUT_DIR`, the workspace root or `MCP_ALLOWED_ROOTS`,
  judged after symlinks resolve. A relative path is read from the data folder.
  A local stdio install is unchanged, except that `~` now expands.
- A workspace `base_dir` outside the served folders is refused, and a workspace
  *name* such as `../../etc` is always refused, confined or not.

---

## [0.2.0] — 2026-09-07

Source-only release: no wheel and no container image are published. Build the
image from the `Dockerfile` here, or install from the tag.

### Added

- **Every dispatch parameter declares an `enum`.** `model`, `models`, `task`,
  `method`, `algorithm` and `format` publish their legal values in
  `tools/list`, rendered from `ALLOWED_CLASSIFIERS` / `ALLOWED_REGRESSORS` and
  the runtime's own tables rather than a second copy of them.
- The enum advertises rather than enforces, so `train_regressor(model="lr")`
  still answers *"'lr' is a `train_classifier()` model. Pick one listed above,
  or call `train_classifier()`"* instead of pydantic's generic literal error.

### Fixed

- **`dry_run` withheld the leakage warning.** Both `train_classifier` and
  `train_regressor` returned from the dry-run branch before leakage detection
  ran, so the one call a caller makes to check a setup before spending the
  compute was the one call that would not tell them the target was in the
  features. `leakage_suspects` and `leakage_note` are now computed and returned
  on the dry-run path too.
- **`plot_learning_curve` accepted any `task`** and silently fell through to
  regression.
- **`detect_outliers` and `drop_column` now take their sibling repo's
  spelling**, so a vocabulary learned in Data_Analyst carries over.

### Changed

- 1,951 tests.

---

## [0.2.0] — 2026-09-07 · part two: the tool-user review

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
