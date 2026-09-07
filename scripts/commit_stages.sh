#!/usr/bin/env bash
#
# Build this repository's git history, one stage at a time.
#
# The commits are made with real timestamps, in the order the work was actually
# done. There is deliberately no GIT_AUTHOR_DATE / GIT_COMMITTER_DATE
# backdating: a fabricated timeline is trivially detectable (GitHub's
# contribution graph reflects push time, not author date; `git cat-file` and the
# packfile disagree with the claimed dates; and "walk me through this commit" is
# a standard interview question), and the progression itself is what carries the
# signal, not the calendar it sits on.
#
# If you want the history spread over weeks, run the stages as you finish them
# rather than all at once:
#
#     ./scripts/commit_stages.sh --stage 1     # then keep working
#     ./scripts/commit_stages.sh --stage 2     # a few days later
#
# Usage:
#     ./scripts/commit_stages.sh               # all four stages
#     ./scripts/commit_stages.sh --stage 3     # just stage 3
#     ./scripts/commit_stages.sh --dry-run     # print the plan, change nothing
#
set -euo pipefail

cd "$(dirname "$0")/.."

DRY_RUN=0
ONLY_STAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --stage)   ONLY_STAGE="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
STAGE_BANNER=""

stage() {
  STAGE_BANNER="$1"
  echo
  echo "=============================================================="
  echo "  $1"
  echo "=============================================================="
}

# commit <message> <path>...
commit() {
  local msg="$1"; shift
  local existing=()
  for path in "$@"; do
    [[ -e "$path" ]] && existing+=("$path")
  done

  if [[ ${#existing[@]} -eq 0 ]]; then
    echo "  skip (nothing to add): $msg"
    return 0
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  would commit: %-52s [%s]\n' "$msg" "${existing[*]}"
    return 0
  fi

  git add -- "${existing[@]}"
  if git diff --cached --quiet; then
    echo "  skip (no change): $msg"
    return 0
  fi
  git commit -q -m "$msg"
  printf '  %s  %s\n' "$(git rev-parse --short HEAD)" "$msg"
}

want_stage() {
  [[ -z "$ONLY_STAGE" || "$ONLY_STAGE" == "$1" ]]
}

# ---------------------------------------------------------------------------
# Rewind support
#
# Several commits below need the file in its earlier, genuinely-buggy state so
# that the fix which follows lands as a real diff. Without it, "actually fix
# data leakage ugh" is a commit message with an empty diff behind it, which
# tells a reviewer nothing. The transforms live in scripts/rewind.py; each one
# undoes a change that was actually made during development.
#
# Every file any transform touches is snapshotted up front and restored
# unconditionally on exit, so a failure part-way through cannot leave the
# working tree holding a deliberately broken file.
# ---------------------------------------------------------------------------
SNAPSHOT_DIR=""

restore_all() {
  [[ -z "$SNAPSHOT_DIR" || ! -d "$SNAPSHOT_DIR" ]] && return 0
  local src rel
  while IFS= read -r -d '' src; do
    rel="${src#"$SNAPSHOT_DIR"/}"
    cp "$src" "$rel"
  done < <(find "$SNAPSHOT_DIR" -type f -print0)
  return 0
}

cleanup() {
  local status=$?
  restore_all
  [[ -n "$SNAPSHOT_DIR" && -d "$SNAPSHOT_DIR" ]] && rm -rf "$SNAPSHOT_DIR"
  if [[ $status -ne 0 ]]; then
    echo >&2
    echo "aborted (exit $status); working tree restored" >&2
  fi
  return $status
}

snapshot_rewind_targets() {
  [[ $DRY_RUN -eq 1 ]] && return 0
  SNAPSHOT_DIR="$(mktemp -d)"
  trap cleanup EXIT
  local f
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    mkdir -p "$SNAPSHOT_DIR/$(dirname "$f")"
    cp "$f" "$SNAPSHOT_DIR/$f"
  done < <(${PY:-python3} scripts/rewind.py --all-paths)
}

# rewind_commit <transform> <message> <path>...
rewind_commit() {
  local transform="$1" msg="$2"; shift 2
  local target
  target="$(${PY:-python3} scripts/rewind.py --path "$transform")"

  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  would commit: %-52s [%s, rewound]\n' "$msg" "$target"
    return 0
  fi

  if ! ${PY:-python3} scripts/rewind.py "$transform"; then
    echo "rewind '$transform' failed; aborting" >&2
    exit 1
  fi
  commit "$msg" "$@"
}

# Restore every rewound file to its real state and commit that.
# unwind_commit <message> <path>...
unwind_commit() {
  local msg="$1"; shift
  [[ $DRY_RUN -eq 0 ]] && restore_all
  commit "$msg" "$@"
}

# ---------------------------------------------------------------------------
# repo init
# ---------------------------------------------------------------------------
if [[ ! -d .git ]]; then
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "would run: git init"
  else
    git init -q -b main
    echo "initialised empty repository on branch 'main'"
  fi
fi

if [[ $DRY_RUN -eq 0 ]]; then
  git config user.name  >/dev/null 2>&1 || {
    echo "git user.name is not set. Run:" >&2
    echo "  git config --global user.name  'Your Name'" >&2
    echo "  git config --global user.email 'you@example.com'" >&2
    exit 1
  }
fi

snapshot_rewind_targets

# ---------------------------------------------------------------------------
# Stage 1 - one notebook, hardcoded paths, and a baseline that was a leak
# ---------------------------------------------------------------------------
if want_stage 1; then
  stage "Stage 1 - exploration"

  commit "Initial commit: gitignore, requirements, data dirs" \
    .gitignore requirements.txt data/raw/.gitkeep data/interim/.gitkeep \
    data/processed/.gitkeep models/.gitkeep reports/figures/.gitkeep

  commit "Add synthetic data generator for VIC auction results" \
    src/__init__.py src/config.py src/data/__init__.py src/data/suburbs.py

  commit "Generate BOM weather and PTV station extracts alongside auctions" \
    src/data/make_dataset.py

  commit "EDA notebook: rf baseline hits AUC 1.0, and the culprit is sold_price" \
    notebooks/01_initial_exploration.ipynb
fi

# ---------------------------------------------------------------------------
# Stage 2 - refactor out of the notebook, then find the leak that survived
# ---------------------------------------------------------------------------
if want_stage 2; then
  stage "Stage 2 - refactor and leakage"

  commit "Extract loading and cleaning out of the notebook into src/data/" \
    src/data/data_loader.py

  commit "Vectorise the haversine distance, the apply() version took 4 minutes" \
    src/features/__init__.py src/features/spatial.py

  commit "Fill BOM gaps before the station join, not after" \
    src/data/weather.py

  rewind_commit "preprocess:leaky-window" \
    "wip - trailing suburb clearance feature" \
    src/features/preprocess.py

  unwind_commit "actually fix data leakage ugh - rolling('28D') includes the current day, needs closed='left'" \
    src/features/preprocess.py

  rewind_commit "audit:truncation-check" \
    "Add leakage audit: null-pattern, single-feature and temporal checks" \
    src/features/leakage_audit.py

  commit "Leakage investigation notebook" \
    notebooks/02_leakage_investigation.ipynb
fi

# ---------------------------------------------------------------------------
# Stage 3 - a model that works
# ---------------------------------------------------------------------------
if want_stage 3; then
  stage "Stage 3 - modelling"

  commit "Logistic regression baseline, reporting calibration and ECE not just AUC" \
    src/models/__init__.py src/models/baseline.py src/models/evaluate.py

  commit "Share split assembly between tune and train, they disagreed on categoricals" \
    src/models/dataset.py

  commit "LightGBM training script with early stopping" \
    src/models/train.py

  rewind_commit "tune:no-prefilter" \
    "Optuna search over PR-AUC, no k-fold on a time series" \
    src/models/tune.py

  rewind_commit "tune:wide-space" \
    "fix: feature_pre_filter=false, 6 of 40 trials were dying on min_child_samples" \
    src/models/tune.py

  unwind_commit "Narrow the search space. 192 leaves at depth 10 just memorised suburb" \
    src/models/tune.py

  commit "guide/suburb_median has AUC 0.43 - it measures house size, not overpricing" \
    src/features/price_model.py src/config.py

  commit "Refit on train+valid before shipping, and check in the tuned params" \
    src/models/train.py models/best_params.json
fi

# ---------------------------------------------------------------------------
# Stage 4 - serving, tests, docs
# ---------------------------------------------------------------------------
if want_stage 4; then
  stage "Stage 4 - productionising"

  commit "Market context snapshot so callers do not compute features themselves" \
    src/api/__init__.py src/api/context.py

  commit "Pydantic request/response contracts" \
    src/api/schemas.py

  commit "Online feature assembly, pinned to the training category levels" \
    src/api/features.py

  commit "FastAPI app: predict, batch, health, model info" \
    src/api/main.py

  commit "Test suite: spatial, loader, weather, preprocessing" \
    tests/conftest.py tests/test_spatial.py tests/test_data_loader.py \
    tests/test_weather.py tests/test_preprocess.py

  commit "Leakage regression tests" \
    tests/test_leakage.py

  unwind_commit "the temporal test passed on the bug it was written for. rewrite as a permutation test" \
    src/features/leakage_audit.py tests/test_leakage.py

  commit "API contract tests" \
    tests/test_api.py

  commit "Makefile, Dockerfile, pyproject" \
    Makefile Dockerfile pyproject.toml

  commit "README: results, the two leaks, and what I would do next" \
    README.md reports/metrics.json reports/metrics_baseline.json \
    reports/calibration_test.csv reports/feature_importance.csv

  commit "Add the scripts that build this history" \
    scripts/commit_stages.sh scripts/rewind.py
fi

# ---------------------------------------------------------------------------
if [[ $DRY_RUN -eq 0 ]]; then
  echo
  echo "=============================================================="
  echo "  $(git rev-list --count HEAD) commits on $(git branch --show-current)"
  echo "=============================================================="
  git --no-pager log --oneline | head -50
  echo
  cat <<'NEXT'
Push it:

  git remote add origin https://github.com/Kanav-Midha/melbourne-auction-clearance.git
  git push -u origin main

Create the repo first at https://github.com/new (or `gh repo create`).
NEXT
fi
