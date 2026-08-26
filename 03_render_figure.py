"""
Step 3 of 3. Render Figure 1 as a vector PDF and a 300 dpi PNG.

Uncover This Tech Term: Model Calibration (Korean Journal of Radiology)

The cohort and the four models are generated exactly as in 02_simulate_metrics.py,
so the figure and the reported numbers cannot drift apart.

Panel (a), identical discrimination
    The three receiver operating characteristic curves are drawn on top of one
    another with decreasing line width, 3.2 then 1.9 then 0.9 points. Plotting
    them this way is what lets a reader see that the curves genuinely coincide
    rather than merely share a summary statistic.

Panel (b), different calibration
    Four series are shown. Model C* is drawn as a large open square and model A
    as a small filled circle placed on top of it, so that the recalibrated model
    is visibly sitting exactly on the well calibrated one. A horizontal arrow in
    the top decile marks the intercept update. The arrow is exactly horizontal
    because the observed proportion is identical across models, the ranking being
    preserved.

Colour
    Okabe-Ito palette, chosen for colour-vision deficiency. Worst pair
    separation is delta E 11.0 under deuteranopia and 18.7 under normal vision,
    both above the usual thresholds. Models C and C* share a hue because they are
    the same model before and after repair. Model A is placed in the most distant
    hue family because it overlaps C* across the whole plot and the two must stay
    distinguishable.

Print safety
    Series are separated by marker shape and line pattern as well as by colour,
    so the figure survives greyscale reproduction. Converting the PNG to
    greyscale is a quick way to confirm this.

Two details that are easy to get wrong
    The smoothing is computed on the logit scale with the robustifying iterations
    switched off. With a binary outcome the default robust iterations treat the
    ones as outliers and pull the curve down toward zero, which silently produces
    a flat and completely wrong calibration curve.

    The outer one per cent of each model's predictions is trimmed before
    plotting the smooth. The data are sparse there and the smoother produces
    artefacts that shoot toward the corners.

Usage
-----
    python 03_render_figure.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess

SEED, N = 4979, 5000
A_INT, B_SLP, SPREAD_K, SHIFT_C = -1.826, 1.341, 2.5, 1.2
OUTPUT_STEM = "Fig1_calibration"
SHOW_CONFIDENCE_INTERVALS = True

# Fonts are tried in order and the first one installed is used. Arial or
# Helvetica is what the journal expects; DejaVu Sans is matplotlib's built-in
# fallback and is there only so the script runs on a bare machine. The script
# prints which face it actually resolved, so a figure never goes out in the
# fallback face without anyone noticing.
FONT_CANDIDATES = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p / (1.0 - p))


def cal_intercept(y, p):
    lp = logit(np.clip(p, 1e-12, 1 - 1e-12))
    intercept = 0.0
    for _ in range(500):
        pr = expit(intercept + lp)
        step = (y - pr).sum() / -(pr * (1 - pr)).sum()
        intercept -= step
        if abs(step) < 1e-13:
            break
    return intercept


def cal_slope(y, p, iters=200, tol=1e-13):
    """Calibration slope by unpenalised Newton-Raphson. Written out rather than
    taken from a library because default regularisation would bias the slope
    toward zero, which is the very quantity being measured."""
    lp = logit(np.clip(p, 1e-12, 1 - 1e-12))
    design = np.column_stack([np.ones_like(lp), lp])
    beta = np.zeros(2)
    for _ in range(iters):
        pr = expit(design @ beta)
        weights = pr * (1 - pr)
        step = np.linalg.solve(-(design * weights[:, None]).T @ design,
                               design.T @ (y - pr))
        beta -= step
        if np.max(np.abs(step)) < tol:
            break
    return beta[1]


# --------------------------------------------------------------- cohort
rng = np.random.default_rng(SEED)
z = rng.normal(size=N)
lt = A_INT + B_SLP * z
p_true = expit(lt)
y = rng.binomial(1, p_true)

mean_lt = lt.mean()
shift = brentq(lambda c: expit(c + SPREAD_K * (lt - mean_lt)).mean() - p_true.mean(),
               -30, 30)
pA = p_true
pB = expit(shift + SPREAD_K * (lt - mean_lt))
pC = expit(lt + SHIFT_C)
pCs = expit(cal_intercept(y, pC) + logit(pC))

# --------------------------------------------------------------- style
OK_BLUE, OK_VERM, OK_GREEN = "#0072B2", "#D55E00", "#009E73"
C_A, C_B, C_C = OK_VERM, OK_GREEN, OK_BLUE   # A vermillion, B green, C and C* blue
GRID, INK, MUTED = "#d9d9d9", "#1a1a1a", "#666666"

# key, long label, probabilities, colour, line style, marker, marker face, line width, z order
SERIES = [
    ("A", "A  Well calibrated", pA, C_A, "-", "o", C_A, 1.3, 3),
    ("B", "B  Overconfident", pB, C_B, "-", "^", C_B, 1.3, 4),
    ("C", "C  Over-estimating", pC, C_C, "-", "s", C_C, 1.3, 5),
    ("Cs", "C*  C recalibrated", pCs, C_C, "--", "s", "white", 1.3, 6),
]

LEGEND_LABEL = {
    "A": "A  Well calibrated",
    "B": "B  Overconfident",
    "C": "C  Over-estimating",
    "Cs": "C*  C, recalibrated",
}

plt.rcParams.update({
    "font.family": FONT_CANDIDATES,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.edgecolor": "#4d4d4d",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": "#4d4d4d",
    "ytick.color": "#4d4d4d",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def decile(p, bins=10):
    groups = np.array_split(np.argsort(p), bins)
    xs = np.array([p[g].mean() for g in groups])
    ys = np.array([y[g].mean() for g in groups])
    ns = np.array([len(g) for g in groups])
    return xs, ys, ns


def wilson(successes, n, zc=1.96):
    """Wilson score interval. Preferred over the normal approximation because
    several deciles sit close to zero, where the normal interval misbehaves."""
    phat = successes / n
    denom = 1 + zc ** 2 / n
    centre = (phat + zc ** 2 / (2 * n)) / denom
    half = zc * np.sqrt(phat * (1 - phat) / n + zc ** 2 / (4 * n ** 2)) / denom
    return np.maximum(centre - half, 0), np.minimum(centre + half, 1)


def square(ax):
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_aspect("equal")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)


def build(with_ci, path_stem):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.09, 3.55))

    # ---------------------------------------------------- panel (a)
    square(ax1)
    ax1.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (1, 2)), zorder=1)
    roc_style = {"A": "-", "B": (0, (5, 2)), "C": (0, (1.2, 1.6))}
    for (key, label, p, colour, _ls, _mk, _fc, _lw, zo), width in zip(
            SERIES[:3], (3.2, 1.9, 0.9)):
        fpr, tpr, _ = roc_curve(y, p)
        ax1.plot(fpr, tpr, color=colour, lw=width, ls=roc_style[key],
                 solid_capstyle="round", zorder=zo, label=label.split("  ")[0])
    ax1.set_xlabel("1 − specificity")
    ax1.set_ylabel("Sensitivity")
    ax1.set_title("Identical discrimination", pad=6, loc="left", fontweight="bold")
    ax1.text(0.97, 0.06, f"AUC = {roc_auc_score(y, pA):.3f}\nfor all models",
             ha="right", va="bottom", fontsize=7.5, color=INK,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", lw=0.6))
    ax1.text(-0.20, 1.06, "(a)", transform=ax1.transAxes,
             fontsize=10, fontweight="bold", va="top")
    ax1.legend(loc="upper left", frameon=False, handlelength=2.4,
               borderaxespad=0.4, labelspacing=0.35,
               title="three curves superimposed", title_fontsize=6.5,
               alignment="left")

    # ---------------------------------------------------- panel (b)
    square(ax2)
    ax2.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (1, 2)), zorder=1)

    # Smoothed calibration curves. See the module docstring for why it=0 and why
    # the tails are trimmed.
    for key, _label, p, colour, ls, _mk, _fc, lw, zo in SERIES:
        lp = logit(np.clip(p, 1e-9, 1 - 1e-9))
        smooth = lowess(y, lp, frac=0.40, it=0, return_sorted=True)
        lo_x, hi_x = np.percentile(lp, [1, 99])
        keep = (smooth[:, 0] >= lo_x) & (smooth[:, 0] <= hi_x)
        ax2.plot(expit(smooth[keep, 0]), np.clip(smooth[keep, 1], 0, 1),
                 color=colour, lw=lw, ls=ls, zorder=zo, alpha=0.95,
                 solid_capstyle="round")

    # Markers are drawn C, then B, then C* as a large open square, then A as a
    # small filled circle on top, so that A appears to sit inside C*.
    marker_style = {"C": dict(ms=3.6, z=5), "B": dict(ms=3.8, z=6),
                    "Cs": dict(ms=6.2, z=7), "A": dict(ms=3.2, z=8)}
    handles = []
    for key, _label, p, colour, ls, mk, fc, lw, _zo in SERIES:
        xs, ys, ns = decile(p)
        st = marker_style[key]
        if with_ci:
            lo, hi = wilson(ys * ns, ns)
            ax2.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt="none",
                         ecolor=colour, elinewidth=0.7, capsize=1.5,
                         capthick=0.7, alpha=0.7, zorder=st["z"] - 0.5)
        ax2.plot(xs, ys, ls="none", marker=mk, ms=st["ms"], mfc=fc, mec=colour,
                 mew=1.0, zorder=st["z"])
        intercept, slope = cal_intercept(y, p), cal_slope(y, p)
        shown = 0.0 if abs(intercept) < 0.005 else intercept   # avoid printing -0.00
        handles.append(Line2D([], [], color=colour, ls=ls, lw=lw, marker=mk,
                              ms=min(st["ms"], 4.4), mfc=fc, mec=colour, mew=1.0,
                              label=f"{LEGEND_LABEL[key]}  ({shown:+.2f}, {slope:.2f})"))

    # Arrow marking the intercept update, drawn in the top decile. It is exactly
    # horizontal because the observed proportion is the same for C and C*.
    x_c, y_c = decile(pC)[0][9], decile(pC)[1][9]
    x_cs = decile(pCs)[0][9]
    ax2.annotate("", xy=(x_cs + 0.030, y_c), xytext=(x_c - 0.030, y_c),
                 arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.9,
                                 shrinkA=0, shrinkB=0, mutation_scale=8), zorder=9)
    ax2.text((x_c + x_cs) / 2 - 0.03, y_c + 0.032, "intercept update",
             ha="center", va="bottom", fontsize=6.6, color="#333333", style="italic")

    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Observed proportion")
    ax2.set_title("Different calibration", pad=6, loc="left", fontweight="bold")
    ax2.text(-0.20, 1.06, "(b)", transform=ax2.transAxes,
             fontsize=10, fontweight="bold", va="top")
    legend = ax2.legend(handles=handles, loc="upper left", frameon=False,
                        handlelength=2.4, borderaxespad=0.4, labelspacing=0.4,
                        title="model   (calibration intercept, slope)",
                        title_fontsize=6.5, alignment="left")
    legend.set_zorder(20)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.13, wspace=0.28)
    for ext, dpi in (("pdf", None), ("png", 300)):
        fig.savefig(f"{path_stem}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {path_stem}.pdf and {path_stem}.png")


def report_font():
    """Say which face was actually resolved, so a submission figure is never
    rendered in the fallback font unnoticed."""
    from matplotlib.font_manager import findfont, FontProperties
    resolved = findfont(FontProperties(family=FONT_CANDIDATES))
    name = resolved.rsplit("/", 1)[-1]
    print(f"font resolved to {name}")
    if "DejaVu" in name:
        print("  WARNING: falling back to DejaVu Sans. Install Arial or "
              "Helvetica before producing the submission figure.")


if __name__ == "__main__":
    report_font()
    build(with_ci=SHOW_CONFIDENCE_INTERVALS, path_stem=OUTPUT_STEM)
