# Simulation code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22108983.svg)](https://doi.org/10.5281/zenodo.22108983)

Code for the simulated example accompanying **Uncover This Tech Term: Model
Calibration** (*Korean Journal of Radiology*), by Youho Myong, Soomin Jeon and
Yongsik Sim.

The example shows how models can have identical discrimination but different
calibration, and why an intercept update can correct systematic overestimation
without correcting overconfidence. All data are simulated; no external data
files are required.

## Quick start

Run these commands from the repository directory using Python and pip:

```bash
python -m pip install -r requirements.txt
python 02_simulate_metrics.py
python 03_render_figure.py --output-dir figures
```

Package versions, including Pillow for image export, are pinned in
`requirements.txt`. Use these versions to reproduce the reported results.

| Script | Purpose |
|---|---|
| `01_tune_parameters.py` | Reproduces the coefficient tuning and seed selection |
| `02_simulate_metrics.py` | Prints performance metrics, threshold results and calibration-decile coordinates |
| `03_render_figure.py` | Exports Figure 1A and 1B as separate JPG/TIF files |

Scripts 02 and 03 are self-contained and can run independently. To reproduce
the parameter search, run `python 01_tune_parameters.py`; it is not required
before running the other scripts.

## Figure output

The example command writes four files to `figures/`:

- `Fig_1A.jpg` and `Fig_1A.tif`: receiver operating characteristic curves.
- `Fig_1B.jpg` and `Fig_1B.tif`: calibration curves.

Each image is RGB, 1065 x 1065 pixels at 300 dpi (3.55 x 3.55 inches).
JPG files use quality 95 with no chroma subsampling; TIF files use lossless
LZW compression. Corner panel letters are omitted; model labels remain in
the legends. Without `--output-dir`, files are saved in the working directory.

Calibration markers represent deciles of predicted probability, with pointwise
Wilson 95% confidence intervals. LOWESS curves use `frac=0.40` and `it=0`,
are fitted on the logit scale using all observations, and are displayed over
each model's 1st–99th prediction percentiles. Arial is preferred; the script
prints the resolved font, which may vary with the fonts installed.

For binary outcomes, the default robustifying iterations can treat events
(`y=1`) as outliers and pull the curve toward zero without an error; `it=0`
disables these iterations.

## Simulation

The data-generating model is `logit(p) = -1.826 + 1.341 * z`, where
`z ~ N(0, 1)` and `y ~ Bernoulli(p)`. Both cohorts contain 5,000 patients
and use NumPy's `default_rng`.

| Cohort | Seed | Events | Purpose |
|---|---|---|---|
| Reported | 4979 | 1,000 | Metrics, table and figure |
| Recalibration | 1 | 1,016 | Estimate intercept updates |

Seed 4979 was selected from seeds 1–20,000 to obtain exactly 1,000 events
and a sample AUC closest to 0.800. This selection is for an illustrative
example and is reproduced by `01_tune_parameters.py`. The recalibration
seed was not tuned; its fitted updates are applied to the reported cohort.

| Model | Construction |
|---|---|
| A | True probability (reference) |
| B | `logit(p_B) = 2.5 * logit(p_A) + c`, with `c` chosen to preserve A's mean predicted probability |
| C | `logit(p_C) = logit(p_A) + 1.2` |
| C\* | C after an intercept update estimated in the recalibration cohort |
| B\* | B after an intercept update estimated in the recalibration cohort |

B illustrates overconfidence and C illustrates systematic overestimation.
B\* is an additional diagnostic output and is not included in the figure or
manuscript table.

## Expected output

Selected output from `02_simulate_metrics.py`:

```text
model                       AUC    Brier    E/O  cal-icpt  cal-slope     ECE
A  well calibrated       0.8000   0.1257   0.99     0.016      1.024   0.011
B  overconfident         0.8000   0.1404   0.99     0.027      0.410   0.092
C  over-estimating       0.8000   0.1665   1.92    -1.184      1.024   0.184
C* C recalibrated        0.8000   0.1258   1.02    -0.033      1.024   0.010
B* B recalibrated        0.8000   0.1413   1.02    -0.056      0.410   0.097
```

E/O is the expected-to-observed event ratio. The calibration intercept is
estimated with the prediction logit as an offset (slope fixed at 1); the
calibration slope is estimated with a freely fitted intercept. ECE uses
10 equal-width probability bins. The script also reports a Brier score
decomposition using 20 equal-size bins; this is a grouped approximation,
not an exact decomposition of the individual-level Brier score.

## Versions

- **v1.2.0** Figures 1A and 1B are exported as separate 300 dpi JPG/TIF files
  without panel letters. Output format only; the simulation is unchanged.
- **v1.1.0** The intercept update is estimated in a separate recalibration
  cohort, so the C\* and B\* rows are out-of-sample with respect to the update.
- **v1.0.0** Initial release.

## Citation and licence

Citation metadata and author ORCIDs are in [CITATION.cff](CITATION.cff).

```text
Myong Y, Jeon S, Sim Y. Simulation code for "Uncover This Tech Term: Model
Calibration". Zenodo; 2026. https://doi.org/10.5281/zenodo.22108983
```

The DOI above identifies all versions. Each release also has its own version
DOI, listed on the Zenodo record.

Released under the [MIT licence](LICENSE).
