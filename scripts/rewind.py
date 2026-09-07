"""Reconstruct an earlier state of a file, for the staged commit history.

``scripts/commit_stages.sh`` uses this to commit the genuinely-buggy version of
a file before committing its fix, so the history carries a real diff rather than
a commit message asserting that a bug once existed.

Every transform here undoes a change that was actually made during development.
Applying a transform and then restoring the file leaves the working tree
identical to where it started.

    python scripts/rewind.py <transform> <path>
"""
from __future__ import annotations

import sys
from pathlib import Path

# name -> (path, [(current_text, earlier_text), ...])
TRANSFORMS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    # The trailing-clearance window originally omitted closed="left", so the
    # 28-day window for a Saturday included that Saturday.
    "preprocess:leaky-window": (
        "src/features/preprocess.py",
        [
            ('roll = g[["sold", "held"]].rolling(window, closed="left").sum()',
             'roll = g[["sold", "held"]].rolling(window).sum()'),
            ('med = g["sold_price"].rolling(window, closed="left").median().rename("med")',
             'med = g["sold_price"].rolling(window).median().rename("med")'),
        ],
    ),
    # Optuna reused one lgb.Dataset across trials with different
    # min_child_samples; without feature_pre_filter=False LightGBM hard-errors.
    "tune:no-prefilter": (
        "src/models/tune.py",
        [
            ('''    "seed": config.RANDOM_SEED,
    # Required because the Dataset is constructed once and reused across trials
    # that vary min_child_samples. With the default (True), LightGBM pre-filters
    # features using the first trial's threshold and then hard-errors on any
    # later trial that lowers it -- which silently killed 6 of the first 40
    # trials before this was set.
    "feature_pre_filter": False,
}''',
             '''    "seed": config.RANDOM_SEED,
}'''),
        ],
    ),
    # The first search space was wide enough to memorise suburb.
    "tune:wide-space": (
        "src/models/tune.py",
        [
            ('"num_leaves": trial.suggest_int("num_leaves", 8, 64, log=True),',
             '"num_leaves": trial.suggest_int("num_leaves", 16, 192, log=True),'),
            ('"max_depth": trial.suggest_int("max_depth", 3, 8),',
             '"max_depth": trial.suggest_int("max_depth", 3, 10),'),
            ('"min_child_samples": trial.suggest_int("min_child_samples", 40, 600, log=True),',
             '"min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),'),
            ('''    The ranges are deliberately conservative. A wider first pass (up to 192
    leaves and depth 10) found a 510-round model that hit 0.76 train AUC and
    0.63 validation -- it had memorised ``suburb`` rather than learned anything.
    ``num_leaves``''',
             '''    ``num_leaves``'''),
        ],
    ),
    # The first temporal check rebuilt features on a truncated history. That
    # cannot detect a window which reaches only as far as the current row, so it
    # passed on the exact bug it was written for.
    "audit:truncation-check": (
        "src/features/leakage_audit.py",
        [
            ("__PLACEHOLDER__", "__PLACEHOLDER__"),  # replaced below
        ],
    ),
}

_AUDIT_V1 = '''def check_temporal_integrity(auctions: pd.DataFrame, stations: pd.DataFrame,
                             weather: pd.DataFrame, cutoff: str = "2022-01-01",
                             columns: tuple[str, ...] = (
                                 "suburb_clearance_l4w",
                                 "region_clearance_l4w",
                                 "suburb_median_price_l90d",
                             )) -> list[Finding]:
    """Rebuild features on a truncated history and require identical values.

    If a trailing feature only uses the past, then computing it over
    ``df[date < cutoff]`` must give exactly the same values for those rows as
    computing it over the full dataset. If a future-looking aggregate has crept
    back in, the two disagree.
    """
    from src.features.preprocess import build_features

    full = build_features(auctions, stations, weather)
    truncated = build_features(
        auctions[auctions["auction_date"] < cutoff].copy(), stations, weather
    )

    key = ["listing_id"]
    merged = full[key + list(columns)].merge(
        truncated[key + list(columns)], on=key, how="inner", suffixes=("_full", "_trunc")
    )

    out = []
    for col in columns:
        a = merged[f"{col}_full"].to_numpy(dtype=float)
        b = merged[f"{col}_trunc"].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        diff = ~(np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True) | both_nan)
        n_diff = int(diff.sum())
        if n_diff:
            worst = float(np.nanmax(np.abs(a[diff] - b[diff]))) if n_diff else 0.0
            out.append(Finding(
                "temporal", col,
                f"{n_diff}/{len(merged)} rows change when the future is removed "
                f"(max delta {worst:.4g})",
            ))
    return out
'''


def _current_audit_block(text: str) -> str:
    start = text.index("def check_temporal_integrity(")
    end = text.index("def check_forbidden_columns(")
    return text[start:end]


def apply(name: str) -> None:
    path_str, pairs = TRANSFORMS[name]
    path = Path(path_str)
    text = path.read_text()

    if name == "audit:truncation-check":
        pairs = [(_current_audit_block(text), _AUDIT_V1 + "\n\n")]

    for current, earlier in pairs:
        if current not in text:
            raise SystemExit(
                f"rewind {name}: anchor not found in {path_str}.\n"
                f"The file has changed; update scripts/rewind.py."
            )
        text = text.replace(current, earlier, 1)

    path.write_text(text)


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--all-paths":
        seen = []
        for path_str, _ in TRANSFORMS.values():
            if path_str not in seen:
                seen.append(path_str)
        print("\n".join(seen))
        return 0
    if len(argv) == 3 and argv[1] == "--path" and argv[2] in TRANSFORMS:
        print(TRANSFORMS[argv[2]][0])
        return 0
    if len(argv) == 2 and argv[1] in TRANSFORMS:
        apply(argv[1])
        return 0
    print(f"usage: {argv[0]} [--path|--all-paths] <{' | '.join(TRANSFORMS)}>",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
