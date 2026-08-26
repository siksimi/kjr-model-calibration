# Simulation code

Supplementary code for **Uncover This Tech Term: Model Calibration**
(*Korean Journal of Radiology*), by Youho Myong
([0000-0002-2469-839X](https://orcid.org/0000-0002-2469-839X)), Soomin Jeon
([0000-0003-1009-8227](https://orcid.org/0000-0003-1009-8227)) and Yongsik Sim
([0000-0003-2711-2793](https://orcid.org/0000-0003-2711-2793)).

Everything quoted in the article, in Figure 1 and in Table 1 is produced by these
three scripts. No data files are needed. The cohort is simulated from a fixed
seed and numpy's generator stream is stable across versions, so the cohort itself
is reproducible anywhere. The last decimal place of the fitted quantities and the
exact shape of the smoothed curves can still shift slightly with library
versions, which is why `requirements.txt` pins the versions the published numbers
were produced with.

## What the code demonstrates

A strictly increasing transformation of predicted probabilities leaves the
ranking of patients untouched. The receiver operating characteristic curve is
built from that ranking alone, so the whole curve, not merely the area beneath
it, is unchanged. The probability scale meanwhile can be distorted at will. This
is what makes it possible to put three models side by side whose AUC agrees to
four decimal places while their calibration differs dramatically.

The same property supplies the remedy. An intercept update adds a constant on the
logit scale and is therefore also rank-preserving, so mean calibration can be
restored without giving up any discrimination. The qualifier matters. An
intercept update repairs mean calibration only, and repairs calibration overall
just when the slope is already correct, which holds for model C by construction.
Applying that same update to a model whose failure lies in the slope changes
nothing, which is why the type of failure has to be diagnosed before a correction
is chosen.

## Files

| File | Purpose | Runtime |
|---|---|---|
| `01_tune_parameters.py` | Derives the simulation constants and documents how the seed was chosen | 1 to 2 min |
| `02_simulate_metrics.py` | Reproduces every number in the article, Figure 1 and Table 1 | a few seconds |
| `03_render_figure.py` | Renders Figure 1 as a vector PDF and a 300 dpi PNG | a few seconds |
| `requirements.txt` | Pinned package versions | |
| `LICENSE`, `CITATION.cff`, `.gitignore` | Repository housekeeping | |

Run them in order. Scripts 02 and 03 are independent of each other and both are
self-contained, so either can be run on its own once the constants in
`01_tune_parameters.py` are settled.

```
pip install -r requirements.txt
python 01_tune_parameters.py
python 02_simulate_metrics.py
python 03_render_figure.py
```

`03_render_figure.py` writes `Fig1_calibration.pdf` and `Fig1_calibration.png`
into the working directory.

## The simulated cohort

```
seed          4979  (numpy default_rng)
n             5000
predictor     z ~ N(0, 1), a single continuous variable
true risk     logit(p) = -1.826 + 1.341 * z
outcome       y ~ Bernoulli(p)
events        1000, an observed event rate of exactly 20.0%
sample AUC    0.8000
```

The intercept and slope were solved on a sample of 2 x 10^6 so that the
population event rate is 0.200 and the population AUC is 0.800.

**On the choice of seed.** With n = 5000 and a 20% event rate, the number of
events in a draw has a standard deviation of about 28, so landing on exactly 1000
is uncommon rather than rare. Seed 4979 was selected from 301 seeds in the scan
range that give exactly 1000 events, chosen because its sample AUC is also
closest to 0.800. Selecting a seed for a clean exposition is legitimate in a
purely illustrative simulation, but it should be stated openly rather than
presented as a lucky draw. `01_tune_parameters.py` reproduces the whole search,
and the article says the same thing in its own words.

## The models

| Model | Construction | Failure being illustrated |
|---|---|---|
| A | predicted probability equals the true probability | none, the reference |
| B | logits spread by a factor of 2.5 around their own mean, with the mean predicted probability held equal to A's | slope collapse, that is, overconfidence |
| C | every logit shifted upward by 1.2 | intercept shift, that is, systematic over-estimation |
| C\* | model C after an intercept update | repair of an intercept failure |
| B\* | model B after the same intercept update | the same repair failing on a slope problem |

Model B is recentred deliberately. Without it the model would fail on both the
intercept and the slope at once and the two failure modes would no longer be
cleanly separated, which would blunt the whole argument. B\* is computed and
printed but does not appear in the published table.

## Expected output

```
model                       AUC    Brier    E/O  cal-icpt  cal-slope     ECE
A  well calibrated       0.8000   0.1257   0.99     0.016      1.024   0.011
B  overconfident         0.8000   0.1404   0.99     0.027      0.410   0.092
C  over-estimating       0.8000   0.1665   1.92    -1.184      1.024   0.184
C* C recalibrated        0.8000   0.1257   1.00    -0.000      1.024   0.010
B* B recalibrated        0.8000   0.1407   1.00     0.000      0.410   0.094
```

Two things in this table carry the argument. The AUC column is constant, which is
the point of the whole construction. And comparing B with B\* shows an intercept
update leaving a slope failure exactly where it was, at 0.410, while the same
update takes C from an expected-to-observed ratio of 1.92 to 1.00.

The Brier score decomposition is worth reporting for a related reason. The bins
are formed by rank, so a rank-preserving transformation puts exactly the same
patients in the same bins. Resolution (0.0337) and uncertainty (0.1600) depend
only on bin membership and on the outcomes, and are therefore identical across
all five models by construction rather than by coincidence, for any number of
bins.

One caveat on how far to push that. The decomposition is exact only when the
predicted probability is constant within each bin, which it is not here. The
correct statement is therefore that the differences in Brier score come entirely
from calibration **within this 20-bin grouped approximation**. That is enough to
forestall the obvious question of why the Brier scores differ at all, but it is a
statement about this decomposition rather than an algebraic identity, and it
should not be quoted without the qualifier.

## Two pitfalls worth knowing about

**Smoothing a binary outcome.** The calibration curves are smoothed with lowess
on the logit scale, with the robustifying iterations switched off (`it=0`). Left
at the default, those iterations treat the ones as outliers and pull the curve
down toward zero, which silently produces a flat and completely wrong calibration
curve. This is easy to miss because the code raises no error.

**Smoothing artefacts in the tails.** The outer one per cent of each model's
predictions is trimmed before the smooth is plotted. The data are sparse there
and the smoother otherwise shoots toward the corners of the panel.

## Figure conventions

Colours are from the Okabe-Ito palette, chosen for colour-vision deficiency. The
worst pair separation is delta E 11.0 under deuteranopia and 18.7 under normal
vision. Models C and C\* share a hue because they are the same model before and
after repair, and model A sits in the most distant hue family because it overlaps
C\* across the whole plot. Series are separated by marker shape and line pattern
as well as by colour, so the figure survives greyscale reproduction. Converting
the PNG to greyscale is a quick way to check this.

`FONT_CANDIDATES` at the top of `03_render_figure.py` lists the faces tried in
order, Arial first. The script prints which face it actually resolved and warns
if it had to fall back to DejaVu Sans, so a submission figure cannot go out in
the wrong face unnoticed. Liberation Sans is metrically compatible with Arial and
is an acceptable stand-in on Linux.

## Note on the licence

MIT is a placeholder chosen as the usual default for research code. Change it if
the institutions involved require something else.
