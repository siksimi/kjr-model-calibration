"""
Step 1 of 3. Choose the simulation constants used in the manuscript.

Uncover This Tech Term: Model Calibration (Korean Journal of Radiology)

Why this script exists
----------------------
The worked example in the article states three round numbers: a cohort of 5,000
patients, exactly 1,000 events (an event rate of exactly 20.0%), and an AUC of
exactly 0.800 shared by all three models. None of those are accidents. This
script documents how they were obtained, so that the choice is reproducible and
open to inspection rather than presented as a lucky draw.

Two things are tuned here.

1. The coefficients of the data-generating model, logit(p) = a + b * z with
   z ~ N(0, 1). The intercept a fixes the population event rate and the slope b
   fixes the population AUC. Both are solved numerically on a very large sample
   (2 x 10^6) so that they describe the population rather than one draw.

2. The random seed. With n = 5,000 and a 20% event rate the number of events in
   a given draw has a standard deviation of about 28, so a draw landing on
   exactly 1,000 events is uncommon but not rare. The script scans seeds and
   reports those that give exactly 1,000 events, ranked by how close the sample
   AUC is to 0.800. Seed 4979 was selected on that basis and is the seed used
   everywhere else in this project.

Selecting a seed for a clean exposition is legitimate for a purely illustrative
simulation, but it should be stated openly. The article does so, and so does
this file.

Runtime is roughly one to two minutes, most of it in the seed scan.

Usage
-----
    python 01_tune_parameters.py
"""

import numpy as np
from sklearn.metrics import roc_auc_score

# Targets for the worked example.
TARGET_PREVALENCE = 0.20
TARGET_AUC = 0.800
N_COHORT = 5000
N_EVENTS_WANTED = 1000

# Sample sizes used for the numerical searches.
N_POPULATION = 2_000_000   # for solving the intercept
N_AUC_PROBE = 400_000      # for evaluating the AUC of a candidate slope
SEED_SCAN_RANGE = 20_000   # how many seeds to try in the final scan


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def solve_intercept(slope, z_population):
    """Find the intercept a such that mean(expit(a + slope * z)) equals the
    target prevalence. The mean is monotone increasing in a, so a plain
    bisection converges reliably."""
    lo, hi = -6.0, 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if expit(mid + slope * z_population).mean() < TARGET_PREVALENCE:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def auc_of_slope(slope, z_population):
    """Population AUC for a candidate slope, with the intercept re-solved so the
    prevalence stays on target. Discrimination rises monotonically with the
    slope, which is what makes the outer bisection valid."""
    intercept = solve_intercept(slope, z_population)
    rng = np.random.default_rng(7)
    z = rng.normal(size=N_AUC_PROBE)
    p = expit(intercept + slope * z)
    y = rng.binomial(1, p)
    return roc_auc_score(y, p), intercept


def tune_coefficients():
    rng = np.random.default_rng(1)
    z_population = rng.normal(size=N_POPULATION)

    lo, hi = 0.5, 3.0
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        auc, _ = auc_of_slope(mid, z_population)
        if auc < TARGET_AUC:
            lo = mid
        else:
            hi = mid

    slope = 0.5 * (lo + hi)
    intercept = solve_intercept(slope, z_population)
    auc, _ = auc_of_slope(slope, z_population)
    prevalence = expit(intercept + slope * z_population).mean()
    return intercept, slope, auc, prevalence


def scan_seeds(intercept, slope):
    """Return every seed in the scan range that yields exactly the wanted number
    of events, sorted by distance between the sample AUC and the target."""
    hits = []
    for seed in range(1, SEED_SCAN_RANGE + 1):
        rng = np.random.default_rng(seed)
        z = rng.normal(size=N_COHORT)
        p = expit(intercept + slope * z)
        y = rng.binomial(1, p)
        if y.sum() == N_EVENTS_WANTED:
            auc = roc_auc_score(y, p)
            hits.append((abs(auc - TARGET_AUC), seed, auc))
    hits.sort()
    return hits


def main():
    print("Tuning the data-generating coefficients on a large sample ...")
    intercept, slope, auc, prevalence = tune_coefficients()
    print(f"  intercept a       = {intercept:.4f}")
    print(f"  slope b           = {slope:.4f}")
    print(f"  population AUC    = {auc:.4f}   (target {TARGET_AUC})")
    print(f"  population rate   = {prevalence:.4f}   (target {TARGET_PREVALENCE})")

    # The manuscript rounds these to three decimals. Use the rounded values from
    # here on so that the published constants are exactly what the other scripts
    # consume.
    a_pub, b_pub = round(intercept, 3), round(slope, 3)
    print(f"\nRounded constants used in the manuscript: a = {a_pub}, b = {b_pub}")

    print(f"\nScanning seeds 1 to {SEED_SCAN_RANGE} for a draw with exactly "
          f"{N_EVENTS_WANTED} events ...")
    hits = scan_seeds(a_pub, b_pub)
    print(f"  {len(hits)} seeds give exactly {N_EVENTS_WANTED} events")
    print("  best candidates by closeness of the sample AUC to the target:")
    for dist, seed, auc in hits[:8]:
        print(f"    seed {seed:6d}   sample AUC {auc:.4f}")

    if hits:
        print(f"\nSelected seed for the manuscript: {hits[0][1]}")
        print("(The project uses seed 4979, which appears among the candidates "
              "above when the scan range and rounded constants are unchanged.)")


if __name__ == "__main__":
    main()
