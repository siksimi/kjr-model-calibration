"""
Step 3 of 3. Render Figure 1A and 1B as separate 300 dpi JPG/TIF files.

Each file has a 3.55 x 3.55 inch canvas (1065 x 1065 pixels).

Uncover This Tech Term: Model Calibration (Korean Journal of Radiology)

The cohort and the four models are generated exactly as in 02_simulate_metrics.py,
so the figure and the reported numbers cannot drift apart. In particular the
intercept update for C* is estimated in the separate recalibration cohort
(RECAL_SEED) and applied to the reported cohort, exactly as in that script.

Panel (A), identical discrimination
    The three receiver operating characteristic curves are drawn on top of one
    another with decreasing line width, 3.2 then 1.9 then 0.9 points. Plotting
    them this way is what lets a reader see that the curves genuinely coincide
    rather than merely share a summary statistic. A small box states the two
    transformations on the logit scale, so the figure is self-contained.

Panel (B), different calibration
    Four series are shown. Model C* is drawn as a large open square and model A
    as a small filled circle placed on top of it, so that the recalibrated model
    is visibly sitting on the well calibrated one. A horizontal arrow in the top
    decile marks the intercept update. The arrow is exactly horizontal because
    the observed proportion is identical across models, the ranking being
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
    so the figure survives greyscale reproduction. Converting the TIFF to
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
    python 03_render_figure.py --output-dir figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess

SEED, RECAL_SEED, N = 4979, 1, 5000
A_INT, B_SLP, SPREAD_K, SHIFT_C = -1.826, 1.341, 2.5, 1.2
OUTPUT_DPI = 300
PANEL_SIZE_INCHES = (3.55, 3.55)
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
def simulate(seed):
    """One cohort with models A, B and C, as in 02_simulate_metrics.py."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=N)
    lt = A_INT + B_SLP * z
    p_true = expit(lt)
    y = rng.binomial(1, p_true)
    mean_lt = lt.mean()
    shift = brentq(lambda c: expit(c + SPREAD_K * (lt - mean_lt)).mean() - p_true.mean(),
                   -30, 30)
    return y, p_true, expit(shift + SPREAD_K * (lt - mean_lt)), expit(lt + SHIFT_C)


y, pA, pB, pC = simulate(SEED)

# The intercept update is estimated in a separate cohort and applied here, so
# what the figure shows for C* is out-of-sample with respect to the update.
y_recal, _, _, pC_recal = simulate(RECAL_SEED)
pCs = expit(cal_intercept(y_recal, pC_recal) + logit(pC))

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


def draw_discrimination(ax1):
    """Draw the ROC panel."""
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
    box = dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", lw=0.6)
    ax1.text(0.97, 0.06, f"AUC = {roc_auc_score(y, pA):.3f}\nfor all models",
             ha="right", va="bottom", fontsize=7.5, color=INK, bbox=box)
    # The two transformations, stated on the logit scale so a reader can see
    # at a glance that both are strictly increasing. The constant c is chosen
    # so that model B keeps the same mean predicted probability as model A.
    ax1.text(0.97, 0.30,
             "$\\mathrm{B}:\\ \\mathrm{logit}\\,p \\rightarrow 2.5\\,\\mathrm{logit}\\,p + c$\n"
             "$\\mathrm{C}:\\ \\mathrm{logit}\\,p \\rightarrow \\mathrm{logit}\\,p + 1.2$",
             ha="right", va="bottom", multialignment="left", fontsize=7, color=INK,
             linespacing=1.5, bbox=box)
    ax1.legend(loc="upper left", frameon=False, handlelength=2.4,
               borderaxespad=0.4, labelspacing=0.35,
               title="three curves superimposed", title_fontsize=6.5,
               alignment="left")

def draw_calibration(ax2, with_ci):
    """Draw the calibration panel."""
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
    # The label sits below the arrow, on two lines, in the open region between
    # the A/C* curves and the C curve, and is drawn above every series so that no
    # curve runs across the letters.
    ax2.text((x_c + x_cs) / 2 - 0.02, y_c - 0.030, "intercept\nupdate",
             ha="center", va="top", fontsize=6.6, color="#333333", style="italic",
             linespacing=1.1, zorder=10)

    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Observed proportion")
    ax2.set_title("Different calibration", pad=6, loc="left", fontweight="bold")
    legend = ax2.legend(handles=handles, loc="upper left", frameon=False,
                        handlelength=2.4, borderaxespad=0.4, labelspacing=0.4,
                        title="model   (calibration intercept, slope)",
                        title_fontsize=6.5, alignment="left")
    legend.set_zorder(20)

def make_panel(panel, with_ci=SHOW_CONFIDENCE_INTERVALS):
    """Build a standalone panel at its final physical size and resolution."""
    if panel not in ("A", "B"):
        raise ValueError("panel must be 'A' or 'B'")
    fig, ax = plt.subplots(figsize=PANEL_SIZE_INCHES, dpi=OUTPUT_DPI)
    if panel == "A":
        draw_discrimination(ax)
    else:
        draw_calibration(ax, with_ci)
    # Identical canvases/margins align the axes when the two files are placed
    # side by side. Render directly at 300 dpi, without resizing raster output.
    fig.subplots_adjust(left=0.17, right=0.98, top=0.90, bottom=0.15)
    return fig, ax


def save_panel(fig, path_stem):
    """Encode one rendered canvas as RGB JPEG and lossless LZW TIFF."""
    fig.canvas.draw()
    raster = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    expected_size = tuple(round(inches * OUTPUT_DPI) for inches in PANEL_SIZE_INCHES)
    if raster.size != expected_size:
        raise RuntimeError(f"Unexpected canvas size: {raster.size}")
    raster.save(f"{path_stem}.jpg", quality=95, subsampling=0,
                dpi=(OUTPUT_DPI, OUTPUT_DPI))
    raster.save(f"{path_stem}.tif", compression="tiff_lzw",
                dpi=(OUTPUT_DPI, OUTPUT_DPI))
    print(f"saved {path_stem}.jpg and {path_stem}.tif "
          f"({raster.width} x {raster.height} pixels, {OUTPUT_DPI} dpi)")


def build(with_ci=SHOW_CONFIDENCE_INTERVALS, output_dir="."):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for panel in ("A", "B"):
        fig, _ax = make_panel(panel, with_ci=with_ci)
        try:
            save_panel(fig, output_dir / f"Fig_1{panel}")
        finally:
            plt.close(fig)


def report_font():
    """Say which face was actually resolved, so a submission figure is never
    rendered in the fallback font unnoticed."""
    from matplotlib.font_manager import findfont, FontProperties
    resolved = findfont(FontProperties(family=FONT_CANDIDATES))
    name = Path(resolved).name
    # Once selected, use the installed face directly instead of repeatedly
    # asking the renderer to resolve unavailable fallback families.
    plt.rcParams["font.family"] = [FontProperties(fname=resolved).get_name()]
    print(f"font resolved to {name}")
    if "DejaVu" in name:
        print("  WARNING: falling back to DejaVu Sans. Install Arial or "
              "Helvetica before producing the submission figure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(),
                        help="Directory for Fig_1A/1B JPG/TIF files (default: current directory)")
    args = parser.parse_args()
    report_font()
    build(with_ci=SHOW_CONFIDENCE_INTERVALS, output_dir=args.output_dir)
