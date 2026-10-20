"""MNEE vs plain path tracing for caustic irradiance in a 2D refractive-circle scene.

Procedural data only (no downloads). Receiver bins below a glass circle receive caustic
light from a finite area light above it. Both methods get N_SPP samples per bin; the error
is the RMSE of per-bin irradiance against a high-sample MNEE reference (lower is better).
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 42
RUN_SEEDS = [42, 43, 44]          # seeds for the final test-set runs
PILOT_SEED = 7                    # seed used only to choose the MNEE grid size on training bins
R, IOR = 1.0, 1.5                 # glass circle radius and refractive index
LIGHT_Y, LIGHT_HALF = 3.0, 0.5    # light segment height and half-width (radiance L = 1)
RECV_Y = -2.0                     # receiver line, below the circle
N_BINS, N_SPP = 64, 4096          # receiver bins and equal per-bin sample budget
N_REF, K_REF = 2**15, 256         # high-sample reference settings
K_CANDIDATES = [32, 64, 128, 256] # MNEE entry-angle grid sizes, selected on training bins
NEWTON_IT = 20
P = np.stack([np.linspace(-1.2, 1.2, N_BINS), np.full(N_BINS, RECV_Y)], 1)  # receiver points
TEST_MASK = np.arange(N_BINS) % 4 == 3   # every 4th bin is test (25%), others are training


def refract(d, nf, eta):
    """Vector Snell's law. d: unit incident dirs (...,2); nf: unit normals facing against d;
    eta = n_incident / n_transmitted. Returns refracted dirs and mask (False = total internal reflection)."""
    cosi = -np.sum(nf * d, -1, keepdims=True)
    k = 1 - eta**2 * (1 - cosi**2)
    t = eta * d + (eta * cosi - np.sqrt(np.maximum(k, 0))) * nf
    return t, k[..., 0] >= 0


def forward(p, phi):
    """Caustic path p -> entry e(phi) -> refract -> exit -> refract -> light plane y=LIGHT_Y.
    Returns the x-coordinate of the light-plane hit, NaN for invalid paths."""
    e = R * np.stack([np.cos(phi), np.sin(phi)], -1)
    w = e - p
    w = w / np.linalg.norm(w, axis=-1, keepdims=True)
    d1, ok1 = refract(w, e / R, 1 / IOR)              # enter the glass
    ok1 &= np.sum(w * e, -1) < 0                      # must be an entering ray
    t = -2 * np.sum(e * d1, -1, keepdims=True)        # second intersection of the chord
    x2 = e + t * d1                                   # exit point
    d2, ok2 = refract(d1, -x2 / R, IOR)               # leave the glass
    ok3 = d2[..., 1] > 0                              # must travel upward toward the light
    s = (LIGHT_Y - x2[..., 1]) / np.where(ok3, d2[..., 1], 1.0)
    xh = x2[..., 0] + s * d2[..., 0]
    return np.where(ok1 & ok2 & ok3, xh, np.nan)


def baseline_samples(p, rng, n):
    """Plain path tracing: uniform directions over the upper half-plane from receiver p.
    Each sample contributes pi*cos(theta) if its traced path hits the light segment."""
    th = rng.uniform(0, np.pi, n)
    om = np.stack([np.cos(th), np.sin(th)], -1)
    b = om @ p
    disc = b**2 - (p @ p - R**2)
    hit_circle = disc >= 0
    s = -b - np.sqrt(np.maximum(disc, 0))             # first intersection with the circle
    e = p + s[:, None] * om
    phi = np.arctan2(e[:, 1], e[:, 0])
    x = forward(p, np.where(hit_circle, phi, 0.0))
    hit = hit_circle & (np.abs(x) <= LIGHT_HALF)
    return np.pi * np.sin(th) * hit


def mnee_samples(p, rng, n, K):
    """MNEE: sample light points x_i uniformly on the segment. For each, find every entry angle phi
    whose path reaches x_i: bracket sign changes on a K-point grid, refine each bracket with Newton.
    Each root contributes G / |dX/dphi| (Jacobian weight), with G = dtheta/dphi at the receiver."""
    xs = rng.uniform(-LIGHT_HALF, LIGHT_HALF, n)
    grid = np.linspace(0, 2 * np.pi, K + 1)
    xg = forward(p, grid)
    f0, f1 = xg[:-1] - xs[:, None], xg[1:] - xs[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        i, k = np.nonzero(f0 * f1 < 0)                # NaN comparisons are False -> skipped
        lo, hi = grid[k], grid[k + 1]
        tgt, h = xs[i], 1e-6
        phi = 0.5 * (lo + hi)
        for _ in range(NEWTON_IT):                    # Newton on the surface parameter, kept in bracket
            f = forward(p, phi) - tgt
            df = (forward(p, phi + h) - forward(p, phi - h)) / (2 * h)
            phi = np.clip(phi - f / df, lo, hi)
        res = forward(p, phi) - tgt
        df = (forward(p, phi + h) - forward(p, phi - h)) / (2 * h)
        e = R * np.stack([np.cos(phi), np.sin(phi)], -1)
        r = np.linalg.norm(e - p, axis=-1)
        g = np.abs(np.sum(e * (e - p), -1)) / r**2    # dtheta/dphi
        cos_p = (e[:, 1] - p[1]) / r                  # receiver cosine (normal = +y)
        ok = (np.abs(res) < 1e-9) & (np.abs(df) > 1e-12) & (cos_p > 0)
        contrib = np.where(ok, cos_p * g / np.abs(np.where(ok, df, 1.0)), 0.0)
    return 2 * LIGHT_HALF * np.bincount(i, weights=contrib, minlength=n)


def run(method, seed, bins, n, K=None):
    """Per-bin irradiance estimates and per-bin estimator variances for the given receiver bins."""
    rng = np.random.default_rng(seed)
    est, var = [], []
    for j in bins:
        c = baseline_samples(P[j], rng, n) if method == "baseline" else mnee_samples(P[j], rng, n, K)
        est.append(c.mean())
        var.append(c.var(ddof=1) / n)
    return np.array(est), np.array(var)


def rmse(est, ref):
    return float(np.sqrt(np.mean((est - ref) ** 2)))


def main():
    t_start = time.time()
    print("1/4 Reference: high-sample MNEE on all receiver bins")
    ref, _ = run("improved", SEED, range(N_BINS), N_REF, K_REF)
    if not (np.all(np.isfinite(ref)) and ref.max() > 0):
        print("Reference invalid (no caustic energy found); aborting.")
        sys.exit(1)
    train, test = np.where(~TEST_MASK)[0], np.where(TEST_MASK)[0]

    print("2/4 Selecting MNEE grid size on TRAINING bins only (pilot seed)")
    val = {K: rmse(run("improved", PILOT_SEED, train, N_SPP, K)[0], ref[train]) for K in K_CANDIDATES}
    K_best = min(val, key=val.get)
    print("    training RMSE per K:", val, "-> selected K =", K_best)

    print("3/4 Final test-set runs with seeds", RUN_SEEDS)
    b_rmse, i_rmse, b_var, i_var, b_t, i_t = [], [], [], [], [], []
    for s in RUN_SEEDS:
        t = time.time(); eb, vb = run("baseline", s, test, N_SPP); b_t.append(time.time() - t)
        t = time.time(); ei, vi = run("improved", s, test, N_SPP, K_best); i_t.append(time.time() - t)
        b_rmse.append(rmse(eb, ref[test])); i_rmse.append(rmse(ei, ref[test]))
        b_var.append(vb.mean()); i_var.append(vi.mean())
        if s == SEED:
            profile = (eb, ei)
        print(f"    seed {s}: baseline RMSE {b_rmse[-1]:.6f}, improved RMSE {i_rmse[-1]:.6f}")

    print("4/4 Writing results.json and figures/")
    bs, im = float(np.mean(b_rmse)), float(np.mean(i_rmse))
    metrics = {
        "baseline_score": bs, "improved_score": im, "improvement": bs - im,
        "baseline_score_std": float(np.std(b_rmse)), "improved_score_std": float(np.std(i_rmse)),
        "baseline_variance": float(np.mean(b_var)), "improved_variance": float(np.mean(i_var)),
        "baseline_time_s": float(np.mean(b_t)), "improved_time_s": float(np.mean(i_t)),
        "selected_grid_K": float(K_best), "n_train_bins": float(len(train)), "n_test_bins": float(len(test)),
    }

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["baseline (path tracing)", "improved (MNEE)"], [bs, im],
           yerr=[np.std(b_rmse), np.std(i_rmse)], capsize=5)
    ax.set_ylabel("irradiance RMSE (test bins)")
    fig.tight_layout(); fig.savefig("figures/rmse_bar.png", dpi=120)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(P[test, 0], ref[test], "k-", label="reference")
    ax.plot(P[test, 0], profile[0], "o", ms=3, label="baseline")
    ax.plot(P[test, 0], profile[1], "s", ms=3, label="improved")
    ax.set_xlabel("receiver x"); ax.set_ylabel("irradiance"); ax.legend()
    fig.tight_layout(); fig.savefig("figures/profile_test_bins.png", dpi=120)

    results = {
        "question": "Does manifold next event estimation (solving the Snell-law constraint per light sample) "
                    "give lower caustic-irradiance error than plain path tracing at an equal sample budget "
                    "in a small 2D refractive-circle scene?",
        "method": "Procedural 2D caustic simulation; equal per-bin sample budget; RMSE vs high-sample MNEE reference "
                  "on held-out receiver bins; mean over seeds 42, 43, 44.",
        "dataset": {
            "name": "procedural 2D refractive-circle caustic scene (generated in numpy)",
            "source_url": "none (generated in code)",
            "n_rows": N_BINS, "n_train": int(len(train)), "n_test": int(len(test)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "RMSE of per-bin caustic irradiance vs high-sample reference (test bins, mean over seeds)",
        "higher_is_better": False,
        "baseline": "Plain unidirectional path tracing: uniform directions from each receiver bin, refracted "
                    "twice by the circle, counted when they hit the area light.",
        "improved": "Manifold next event estimation: for each sampled light point, Newton iterations on the entry-angle "
                    "parameter solve the Snell-law path constraint; roots are weighted by the Jacobian dX/dphi "
                    "and the receiver geometric term. Same samples, bins, and metric.",
        "metrics": metrics,
        "notes": "No training data: the split is over receiver bins (every 4th bin held out as test). "
                 "The MNEE grid size is chosen on training bins with a pilot seed; the baseline has no tuned knob. "
                 "The reference is MNEE with more samples, so it shares the estimator family and any bias of it. "
                 "Ideal refraction only: no Fresnel terms, no radiance scaling. Folds (zero Jacobian) and missed or "
                 "non-converged roots are dropped, which can bias both methods. Equal sample budget is not equal compute; "
                 "times are reported separately. This is a stand-in for the paper's Vulkan implementation, "
                 "and the paper's reported result is not stated, so no agreement with it is claimed.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Done in {time.time() - t_start:.1f}s; results.json written.")


if __name__ == "__main__":
    main()
