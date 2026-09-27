"""Generate the figures embedded in the README.

    python -m src.models.figures

Each figure is rendered twice, for GitHub's light and dark themes, and the README
selects between them with a ``<picture>`` element. Colours come from a palette
validated for colour-vision deficiency separation and for contrast against each
surface, so the two series stay distinguishable for red/green colourblind readers
and in greyscale print.

Three deliberate choices worth noting:

* **The clearance rate and the cash rate get their own stacked panels**, not one
  chart with two y-axes. A dual-axis chart lets the author imply any correlation
  they like by rescaling one axis; shared-x panels show the same relationship
  without that freedom.
* **Every series is labelled directly as well as in the legend**, so identity
  never depends on colour alone.
* **Each figure has a table equivalent in the README**, which is what makes the
  numbers accessible to a screen reader.
"""
from __future__ import annotations

import logging

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from src import config  # noqa: E402

log = logging.getLogger(__name__)

#: Validated categorical slots 1 and 2, plus chrome, per surface.
THEME = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "s1": "#2a78d6", "s2": "#eb6834", "band": "#f0efec",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "s1": "#3987e5", "s2": "#d95926", "band": "#232321",
    },
}


def _new_fig(mode: str, figsize, nrows: int = 1, **kw):
    t = THEME[mode]
    fig, axes = plt.subplots(nrows, 1, figsize=figsize, facecolor=t["surface"], **kw)
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(t["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(t["axis"])
            ax.spines[side].set_linewidth(1.0)
        ax.tick_params(colors=t["muted"], labelsize=9, length=3, width=1.0)
        ax.grid(True, color=t["grid"], linewidth=0.8, alpha=1.0)
        ax.set_axisbelow(True)
    return fig, axes, t


def _save(fig, name: str, mode: str) -> None:
    path = config.FIGURES_DIR / f"{name}_{mode}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    log.info("wrote %s", path.relative_to(config.PROJECT_ROOT))


# --------------------------------------------------------------------------- #
def fig_calibration(cal_lr: pd.DataFrame, cal_lgbm: pd.DataFrame,
                    ece_lr: float, ece_lgbm: float, mode: str) -> None:
    """Predicted vs observed clearance, by probability decile.

    The headline claim of the project is calibration, so this is the figure that
    has to carry it: distance from the diagonal *is* the error.

    The axes are forced square and to identical limits. On a calibration plot an
    unequal aspect tilts the reference line away from 45 degrees, which makes
    over- and under-confidence look like different magnitudes when they are not.
    """
    fig, ax, t = _new_fig(mode, (6.8, 6.4))

    vals = pd.concat([cal_lr[["predicted", "observed"]],
                      cal_lgbm[["predicted", "observed"]]])
    lo = float(vals.min().min()) - 0.05
    hi = float(vals.max().max()) + 0.05

    ax.plot([lo, hi], [lo, hi], color=t["axis"], linewidth=1.4,
            linestyle=(0, (4, 3)), zorder=1)
    # Anchored low on the diagonal, where neither series reaches.
    ax.annotate("perfect calibration", xy=(lo + 0.055, lo + 0.055),
                xytext=(6, -4), textcoords="offset points",
                color=t["muted"], fontsize=8.5, ha="left", va="top",
                rotation=45, rotation_mode="anchor")

    for cal, colour, label, ece in (
        (cal_lr, t["s2"], "Logistic regression", ece_lr),
        (cal_lgbm, t["s1"], "LightGBM (tuned)", ece_lgbm),
    ):
        ax.plot(cal["predicted"], cal["observed"], color=colour, linewidth=2.0,
                marker="o", markersize=7, markeredgecolor=t["surface"],
                markeredgewidth=1.6, label=f"{label}  (ECE {ece:.3f})", zorder=3)

    # Direct labels at the decile where the two series separate most, so they
    # never collide with each other or with the reference line.
    n = min(len(cal_lr), len(cal_lgbm))
    gap = (cal_lr["observed"].to_numpy()[:n] - cal_lgbm["observed"].to_numpy()[:n])
    k = int(np.argmax(np.abs(gap)))
    # Upper-left of its point: the logistic series rises to the right, so any
    # label centred above it gets crossed by its own line.
    ax.annotate("Logistic", xy=(cal_lr["predicted"].iloc[k], cal_lr["observed"].iloc[k]),
                xytext=(-10, 10), textcoords="offset points", ha="right",
                color=t["s2"], fontsize=9.5, fontweight="medium")
    ax.annotate("LightGBM",
                xy=(cal_lgbm["predicted"].iloc[k], cal_lgbm["observed"].iloc[k]),
                xytext=(0, -20), textcoords="offset points", ha="center",
                color=t["s1"], fontsize=9.5, fontweight="medium")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted probability of clearing", color=t["ink2"], fontsize=10)
    ax.set_ylabel("Observed clearance rate", color=t["ink2"], fontsize=10)
    ax.set_title("Predicted vs observed clearance, 2024 held-out year",
                 color=t["ink"], fontsize=12.5, fontweight="semibold",
                 loc="left", pad=30)
    ax.text(0, 1.014, "each point is one decile of predictions \u00b7 closer to "
                      "the dashed line is better",
            transform=ax.transAxes, color=t["muted"], fontsize=9, va="bottom")

    leg = ax.legend(loc="lower right", frameon=False, fontsize=9.5,
                    handlelength=1.6, borderpad=0.6)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    _save(fig, "calibration", mode)


# --------------------------------------------------------------------------- #
def fig_timeline(df: pd.DataFrame, mode: str) -> None:
    """Monthly clearance rate and the RBA cash rate, as stacked panels.

    Deliberately not a dual-axis chart. Two panels sharing an x-axis show the
    same co-movement without letting the axis scaling manufacture it.
    """
    monthly = (
        df.assign(m=df["auction_date"].dt.to_period("M"))
        .groupby("m")
        .agg(clearance=(config.TARGET, "mean"),
             cash_rate=("rba_cash_rate", "first"),
             n=(config.TARGET, "size"))
    )
    # Months with almost no auctions get dropped rather than plotted: a
    # "clearance rate" over nine auctions is noise. Reindexing to a continuous
    # monthly range then leaves NaNs, so matplotlib *breaks* the line over the
    # 2020 lockdowns instead of interpolating a straight segment across a period
    # when on-site auctions were banned.
    monthly.loc[monthly["n"] < 40, "clearance"] = np.nan
    monthly = monthly.reindex(
        pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    )
    monthly["cash_rate"] = monthly["cash_rate"].ffill()
    x = monthly.index.to_timestamp()

    fig, axes, t = _new_fig(mode, (9.6, 5.8), nrows=2, sharex=True,
                            gridspec_kw={"height_ratios": [1.5, 1.0], "hspace": 0.18})
    ax_top, ax_bot = axes

    train_end = pd.Timestamp(config.TRAIN_END)
    valid_end = pd.Timestamp(config.VALID_END)
    for ax in axes:
        ax.axvspan(x.min(), train_end, color=t["band"], zorder=0)
        ax.axvline(train_end, color=t["axis"], linewidth=1.0,
                   linestyle=(0, (3, 3)), zorder=1)
        ax.axvline(valid_end, color=t["axis"], linewidth=1.0,
                   linestyle=(0, (3, 3)), zorder=1)

    ax_top.plot(x, 100 * monthly["clearance"], color=t["s1"], linewidth=2.0, zorder=3)
    ax_top.set_ylabel("Clearance rate (%)", color=t["ink2"], fontsize=10)
    ax_top.set_title("Melbourne auction clearance against the RBA cash rate",
                     color=t["ink"], fontsize=12.5, fontweight="semibold",
                     loc="left", pad=30)
    ax_top.text(0, 1.014, "monthly clearance, and the cash rate that drives it · "
                         "splits are chronological, never random",
                transform=ax_top.transAxes, color=t["muted"], fontsize=9, va="bottom")

    # Split labels sit in the top panel's headroom.
    span_mid = [
        (x.min() + (train_end - x.min()) / 2, "train  2019–2022"),
        (train_end + (valid_end - train_end) / 2, "valid  2023"),
        (valid_end + (x.max() - valid_end) / 2, "test  2024"),
    ]
    ytxt = ax_top.get_ylim()[1]
    for xm, label in span_mid:
        ax_top.annotate(label, xy=(xm, ytxt), xytext=(0, -12),
                        textcoords="offset points", ha="center",
                        color=t["muted"], fontsize=8.5)

    ax_bot.plot(x, monthly["cash_rate"], color=t["s2"], linewidth=2.0,
                drawstyle="steps-post", zorder=3)
    ax_bot.set_ylabel("Cash rate (%)", color=t["ink2"], fontsize=10)
    ax_bot.set_ylim(bottom=0)

    # One direct label per panel: identity without a legend, single series each.
    last = monthly["clearance"].last_valid_index()
    ax_top.annotate("clearance rate",
                    xy=(last.to_timestamp(), 100 * monthly["clearance"].loc[last]),
                    xytext=(-6, 14), textcoords="offset points", ha="right",
                    color=t["s1"], fontsize=9.5, fontweight="medium")
    ax_bot.annotate("cash rate", xy=(x[-1], monthly["cash_rate"].iloc[-1]),
                    xytext=(-6, -16), textcoords="offset points", ha="right",
                    color=t["s2"], fontsize=9.5, fontweight="medium")

    _save(fig, "clearance_timeline", mode)


# --------------------------------------------------------------------------- #
def fig_importance(imp: pd.DataFrame, mode: str, top_n: int = 12) -> None:
    """Top features by gain. Single series, so the title carries identity."""
    d = imp.head(top_n).iloc[::-1]
    fig, ax, t = _new_fig(mode, (8.0, 5.2))

    ypos = np.arange(len(d))
    height = 0.62
    radius = 0.045 * d["gain_pct"].max()

    # Rounded data-ends, anchored square to the baseline at x=0.
    for y, w in zip(ypos, d["gain_pct"], strict=True):
        r = min(radius, w / 2)
        ax.add_patch(FancyBboxPatch(
            (0, y - height / 2), max(w - r, 1e-6), height,
            boxstyle=f"round,pad=0,rounding_size={r}",
            linewidth=0, facecolor=t["s1"], mutation_aspect=1 / 18, zorder=3,
        ))
        ax.add_patch(plt.Rectangle((0, y - height / 2), r, height,
                                   linewidth=0, facecolor=t["s1"], zorder=3))

    for y, w in zip(ypos, d["gain_pct"], strict=True):
        ax.text(w + 0.3, y, f"{w:.1f}%", va="center", ha="left",
                color=t["ink2"], fontsize=9)

    ax.set_yticks(ypos)
    ax.set_yticklabels(d["feature"], color=t["ink2"], fontsize=9.5)
    ax.set_xlim(0, d["gain_pct"].max() * 1.16)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel("Share of total gain (%)", color=t["ink2"], fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"What the model uses: top {top_n} features by gain",
                 color=t["ink"], fontsize=12.5, fontweight="semibold",
                 loc="left", pad=30)
    ax.text(0, 1.014, "market momentum and rate direction dominate; "
                     "no property attribute competes",
            transform=ax.transAxes, color=t["muted"], fontsize=9, va="bottom")

    _save(fig, "feature_importance", mode)


# --------------------------------------------------------------------------- #
def main() -> None:
    import joblib

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.data.data_loader import load_auctions, load_stations
    from src.data.weather import build_weather_table
    from src.features.preprocess import build_features
    from src.models.baseline import _xy
    from src.models.dataset import load_splits
    from src.models.evaluate import calibration_table, expected_calibration_error

    if not (config.MODEL_PATH.exists() and config.BASELINE_PATH.exists()):
        raise SystemExit(
            "figures need both models. Run:\n"
            "  python -m src.models.baseline --save\n"
            "  python -m src.models.train --save"
        )

    (_, _, X_te), (_, _, y_te), (_, _, te_df) = load_splits()
    bundle = joblib.load(config.MODEL_PATH)
    p_lgbm = bundle["booster"].predict(X_te)

    baseline = joblib.load(config.BASELINE_PATH)["model"]
    X_lr, _ = _xy(te_df)
    p_lr = baseline.predict_proba(X_lr)[:, 1]

    cal_lgbm = calibration_table(y_te, p_lgbm)
    cal_lr = calibration_table(y_te, p_lr)
    ece_lgbm = expected_calibration_error(y_te, p_lgbm)
    ece_lr = expected_calibration_error(y_te, p_lr)

    imp = pd.DataFrame(bundle["feature_importance"])
    full = build_features(load_auctions(), load_stations(), build_weather_table())

    for mode in ("light", "dark"):
        fig_calibration(cal_lr, cal_lgbm, ece_lr, ece_lgbm, mode)
        fig_timeline(full, mode)
        fig_importance(imp, mode)

    log.info("ECE: logistic %.4f | lightgbm %.4f", ece_lr, ece_lgbm)


if __name__ == "__main__":
    main()
