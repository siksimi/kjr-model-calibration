"""
Step 2 of 3. Reproduce every number quoted in the article, the figure and the table.

Uncover This Tech Term: Model Calibration (Korean Journal of Radiology)

The idea being demonstrated
--------------------------
Applying a strictly increasing transformation to predicted probabilities leaves
the ranking of patients untouched. The receiver operating characteristic curve
is built entirely from that ranking, so the whole curve, and not merely the area
beneath it, is unchanged. The probability scale, on the other hand, can be
distorted freely. That is what makes it possible to place three models side by
side whose AUC agrees to four decimal places while their calibration differs
dramatically.

The same property also supplies the remedy. Updating the calibration intercept
adds a constant on the logit scale, which is again a rank-preserving
transformation, so mean calibration can be restored without giving up any
discrimination. Note the qualifier. An intercept update repairs mean calibration
only. It repairs calibration overall just when the slope is already correct,
which holds for model C by construction and does not hold for model B.

The four models
---------------
    A   well calibrated       predicted probability equals the true probability
    B   overconfident         probabilities spread toward 0 and 1, with the mean
                              predicted probability held equal to model A's, so
                              that the failure shows up in the slope alone
    C   over-estimating       every probability shifted upward on the logit
                              scale, so that the failure shows up in the
                              intercept alone
    C*  C recalibrated        model C after an intercept update

Model B is also passed through the same intercept update, reported as B*, to
show that the correction leaves a slope failure untouched. That contrast is the
point of the "Can it be fixed?" section of the article.

A note on the constants
-----------------------
The coefficients and the seed come from 01_tune_parameters.py. Seed 4979 gives a
draw with exactly 1,000 events out of 5,000 and a sample AUC of 0.8000, which
was chosen for clarity of exposition and is stated as such in the article.

Usage
-----
    python 02_simulate_metrics.py
"""

import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import brier_score_loss, roc_auc_score

SEED = 4979
N = 5000
A_INTERCEPT, B_SLOPE = -1.826, 1.341   # logit(p_true) = a + b * z
SPREAD_K = 2.5                          # model B, how far probabilities are spread
SHIFT_C = 1.2                           # model C, size of the upward logit shift
THRESHOLDS = (0.10, 0.20, 0.30)         # risk thresholds shown in Table 1


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p / (1.0 - p))


def _newton_logistic(design, y, iters=200, tol=1e-13):
    """Unpenalised logistic regression by Newton-Raphson.

    Written out rather than taken from a library for two reasons. The problem
    has at most two parameters and converges in a handful of steps, and any
    regularisation the library applies by default would bias the calibration
    slope toward zero, which is precisely the quantity being measured here.
    """
    beta = np.zeros(design.shape[1])
    for _ in range(iters):
        pr = expit(design @ beta)
        weights = pr * (1 - pr)
        gradient = design.T @ (y - pr)
        hessian = -(design * weights[:, None]).T @ design
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def calibration_intercept_slope(y, p):
    """Return the calibration intercept and the calibration slope.

    The slope is the coefficient of a logistic regression of the outcome on the
    logit of the predicted probability. It equals 1 when calibration is good and
    falls below 1 when predictions are too extreme.

    The intercept is a separate fit with the logit of the predicted probability
    held fixed as an offset and the slope therefore pinned at 1. That constraint
    is what makes the intercept mean "is the average right", so the two
    quantities are deliberately not read off a single regression.
    """
    lp = logit(np.clip(p, 1e-12, 1 - 1e-12))
    slope = _newton_logistic(np.column_stack([np.ones_like(lp), lp]), y)[1]
    intercept = 0.0
    for _ in range(500):
        pr = expit(intercept + lp)
        step = (y - pr).sum() / -(pr * (1 - pr)).sum()
        intercept -= step
        if abs(step) < 1e-13:
            break
    return intercept, slope


def recalibrate_intercept(y, p):
    """Logistic recalibration with the slope held at 1, that is, an intercept
    update. Because it adds a constant on the logit scale it preserves the
    ranking of patients and therefore leaves the AUC untouched.

    For illustration this is estimated in the same cohort. In practice an
    independent sample is required, and the article says so.
    """
    intercept, _ = calibration_intercept_slope(y, p)
    return expit(intercept + logit(np.clip(p, 1e-12, 1 - 1e-12)))


def make_cohort():
    rng = np.random.default_rng(SEED)
    z = rng.normal(size=N)
    lt = A_INTERCEPT + B_SLOPE * z
    p_true = expit(lt)
    y = rng.binomial(1, p_true)

    p_a = p_true

    # Model B. Spread the logits around their own mean so that the average
    # predicted probability is preserved. Without this recentring the model
    # would fail on both the intercept and the slope, and the two failure modes
    # would no longer be cleanly separated.
    mean_lt = lt.mean()
    shift = brentq(
        lambda c: expit(c + SPREAD_K * (lt - mean_lt)).mean() - p_a.mean(),
        -30, 30)
    p_b = expit(shift + SPREAD_K * (lt - mean_lt))

    # Model C. A pure shift on the logit scale, which leaves the slope at 1.
    p_c = expit(lt + SHIFT_C)

    models = {
        "A  well calibrated": p_a,
        "B  overconfident": p_b,
        "C  over-estimating": p_c,
        "C* C recalibrated": recalibrate_intercept(y, p_c),
        "B* B recalibrated": recalibrate_intercept(y, p_b),
    }
    return y, models


def expected_calibration_error(y, p, bins=10):
    """Mean absolute gap between predicted and observed probability, taken over
    equal-width bins and weighted by how many patients fall in each bin."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return sum((idx == k).sum() / len(p) * abs(p[idx == k].mean() - y[idx == k].mean())
               for k in range(bins) if (idx == k).sum())


def murphy_decomposition(y, p, bins=20):
    """Brier score decomposition, Brier = reliability - resolution + uncertainty,
    computed over bins of equal size.

    Two points about how to read the result.

    The bins are formed by rank, so a strictly increasing transformation puts
    exactly the same patients in the same bins. Resolution and uncertainty depend
    only on bin membership and on the outcomes, so they are identical across
    these models by construction rather than by coincidence, and that holds for
    any number of bins.

    The decomposition itself is exact only when the predicted probability is
    constant within each bin, which it is not here. The numbers below are
    therefore a binned approximation, and the claim that the spread in Brier
    score comes from calibration should be read in that light rather than as an
    algebraic identity.
    """
    groups = np.array_split(np.argsort(p), bins)
    ybar = y.mean()
    reliability = sum(len(g) / len(y) * (p[g].mean() - y[g].mean()) ** 2 for g in groups)
    resolution = sum(len(g) / len(y) * (y[g].mean() - ybar) ** 2 for g in groups)
    uncertainty = ybar * (1 - ybar)
    return reliability, resolution, uncertainty


def decile_points(y, p, bins=10):
    """Coordinates of the grouped calibration plot, that is, mean predicted
    probability against observed proportion within each decile of predicted
    risk. The observed proportions are identical across models, which is the
    visual evidence that the ranking is preserved."""
    groups = np.array_split(np.argsort(p), bins)
    return [p[g].mean() for g in groups], [y[g].mean() for g in groups]


def main():
    y, models = make_cohort()
    prevalence = y.mean()
    print(f"n = {N}   events = {y.sum()}   observed event rate = {prevalence:.4f}\n")

    print("=== Performance measures ===")
    print(f"{'model':22s}{'AUC':>9}{'Brier':>9}{'E/O':>7}"
          f"{'cal-icpt':>10}{'cal-slope':>11}{'ECE':>8}")
    for name, p in models.items():
        intercept, slope = calibration_intercept_slope(y, p)
        print(f"{name:22s}{roc_auc_score(y, p):9.4f}{brier_score_loss(y, p):9.4f}"
              f"{p.mean() / prevalence:7.2f}{intercept:10.3f}{slope:11.3f}"
              f"{expected_calibration_error(y, p):8.3f}")
    print("\nNote how B* differs from B only in the intercept. An intercept "
          "update cannot repair a slope failure.")

    print("\n=== Brier score decomposition (20 equal-size bins) ===")
    print(f"{'model':22s}{'reliability':>13}{'resolution':>12}{'uncertainty':>13}")
    for name, p in models.items():
        reliability, resolution, uncertainty = murphy_decomposition(y, p)
        print(f"{name:22s}{reliability:13.4f}{resolution:12.4f}{uncertainty:13.4f}")
    print("\nResolution and uncertainty are identical across models, so within "
          "this 20-bin grouped approximation the differences in Brier score "
          "come entirely from calibration.")

    print("\n=== Table 1, effect of applying a risk threshold ===")
    print(f"{'thr':>5} {'model':22s}{'flagged':>9}{'(%)':>7}{'TP':>6}{'FP':>6}"
          f"{'FN':>6}{'sens':>8}{'spec':>8}{'PPV':>8}")
    for threshold in THRESHOLDS:
        for name, p in models.items():
            if "*" in name:
                continue  # recalibrated models are not shown in the published table
            flagged = p >= threshold
            tp = int((flagged & (y == 1)).sum())
            fp = int((flagged & (y == 0)).sum())
            fn = int((~flagged & (y == 1)).sum())
            tn = int((~flagged & (y == 0)).sum())
            print(f"{threshold:5.0%} {name:22s}{int(flagged.sum()):9d}"
                  f"{100 * flagged.mean():7.1f}{tp:6d}{fp:6d}{fn:6d}"
                  f"{tp / (tp + fn):8.3f}{tn / (tn + fp):8.3f}"
                  f"{tp / max(tp + fp, 1):8.3f}")
        print()

    print("=== Figure 1b, decile coordinates (x = mean predicted, y = observed) ===")
    for name, p in models.items():
        xs, ys = decile_points(y, p)
        print(name)
        print("  x: " + " ".join(f"{v:.3f}" for v in xs))
        print("  y: " + " ".join(f"{v:.3f}" for v in ys))


if __name__ == "__main__":
    main()
