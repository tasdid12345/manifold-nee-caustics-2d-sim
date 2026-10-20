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

## Reference

This project is motivated by the following paper:

> JW Jhang, CF Chang (2023). *Implementation of Manifold Next Event Estimation for Caustics Rendering on Vulkan*. 2023 9th International Conference on. https://ieeexplore.ieee.org/abstract/document/10179503/

Paper author: Prof. Chun-Fa Chang, National Taiwan Normal University (chunfa@ntnu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
