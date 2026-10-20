# MNEE vs Path Tracing for Caustic Irradiance (2D Refractive Circle)

This experiment is motivated by "Implementation of Manifold Next Event Estimation for Caustics Rendering on Vulkan" (2023), which applies manifold next event estimation (MNEE) to caustic light paths. Caustics are hard for standard path tracing because light reaches the viewer only through specular refraction, so random directions rarely hit small light sources. MNEE instead solves the refraction constraint for each light sample, which should reduce noise at equal sample counts. This repository tests that claim in a small, fully controlled two-dimensional scene.

**Question:** Does manifold next event estimation (solving the Snell-law constraint per light sample) give lower caustic-irradiance error than plain path tracing at an equal sample budget in a small 2D refractive-circle scene?

**Dataset:** A procedural 2D scene generated in numpy: a glass circle, a finite area light segment, and a line of receiver bins where caustic irradiance is estimated. No download is needed. This is not the paper's own dataset.

**Baseline:** Plain unidirectional path tracing. Each receiver bin samples directions, refracts them through the circle twice, and counts hits on the light.

**Improved:** Manifold next event estimation. For each sampled light point, Newton iterations on the entry-angle parameter solve the Snell-law path constraint, and each root is weighted by its Jacobian.

Both methods use the same samples per bin, binning code, and error metric (RMSE against a high-sample reference on held-out bins). Render time and per-bin variance are reported as secondary metrics.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

## Results

Results are written to `results.json`. Figures are saved to `figures/`.

## Results (latest run)

On the procedural 2D refractive-circle caustic scene (16 held-out receiver bins, mean over seeds 42-44), plain unidirectional path tracing with area-light hits reached caustic-irradiance RMSE 0.0154 while manifold next event estimation reached 0.0122, a modest gain within seed standard deviations of 0.0030 and 0.0046.

| Metric | Value |
|---|---|
| baseline_score | 0.01538 |
| improved_score | 0.01219 |
| improvement | 0.003185 |
| baseline_score_std | 0.002974 |
| improved_score_std | 0.004595 |
| baseline_variance | 0.0001405 |
| improved_variance | 0.0001355 |
| baseline_time_s | 0.01977 |
| improved_time_s | 1.065 |
| selected_grid_K | 256 |
| n_train_bins | 48 |
| n_test_bins | 16 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: No training data: the split is over receiver bins (every 4th bin held out as test). The MNEE grid size is chosen on training bins with a pilot seed; the baseline has no tuned knob. The reference is MNEE with more samples, so it shares the estimator family and any bias of it. Ideal refraction only: no Fresnel terms, no radiance scaling. Folds (zero Jacobian) and missed or non-converged roots are dropped, which can bias both methods. Equal sample budget is not equal compute; times are reported separately. This is a stand-in for the paper's Vulkan implementation, and the paper's reported result is not stated, so no agreement with it is claimed.

## Reference

This project is motivated by the following paper:

> JW Jhang, CF Chang (2023). *Implementation of Manifold Next Event Estimation for Caustics Rendering on Vulkan*. 2023 9th International Conference on. https://ieeexplore.ieee.org/abstract/document/10179503/

Paper author: Prof. Chun-Fa Chang, National Taiwan Normal University (chunfa@ntnu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
