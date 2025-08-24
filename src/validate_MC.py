# scripts/validate_estimators.py
# Validate H_alpha Monte Carlo estimators against exact references.
# A) Two single Gaussians (I=J=1): exact via g_eps(d, eps)
# B) Axis-embedded mixtures (I,J>1, collinear means): exact via Gauss–Hermite

import math
import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss

from estimators import estimate_halpha_mc
from math_utils import g_eps_np  # exact Hα for two Gaussians via d=||μ-ν||

torch.set_default_dtype(torch.double)

# -------------------------- helpers --------------------------

def make_means_two_gauss(D: int, d: float, mode: str, seed: int = 0):
    torch.manual_seed(seed)
    if mode == 'axis':
        vec = torch.zeros(D); vec[0] = 1
    elif mode == 'ones':
        vec = torch.ones(D)
    elif mode == 'orth':
        vec = torch.zeros(D); vec[0] = vec[1] = 1
    elif mode == 'random':
        vec = torch.randn(D)
    else:
        raise ValueError(mode)
    vec = vec / vec.norm() * (d / 2)
    mu = vec.unsqueeze(0)   # (1, D)
    nu = -vec.unsqueeze(0)  # (1, D)
    return mu, nu

def random_orthogonal(D: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(D, D, generator=g)
    Q, R = torch.linalg.qr(A)
    s = torch.sign(torch.diag(R))
    return Q @ torch.diag(s)

def build_line_means(I: int, J: int, D: int):
    """μ_i = -i e1, ν_j = +j e1 in R^D."""
    mu = torch.zeros(I + 1, D)
    nu = torch.zeros(J + 1, D)
    mu[:, 0] = -torch.arange(I + 1, dtype=mu.dtype)
    nu[:, 0] =  torch.arange(J + 1, dtype=nu.dtype)
    return mu, nu

def random_simplex(k: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.uniform(size=k + 1)
    return w / w.sum()

# ------------------------ exact axis H_alpha ------------------------

def halpha_exact_axis(w, v, a, b, eps: float, n_gh: int = 240) -> float:
    """
    Exact H_alpha for axis-embedded mixtures (unit variance):
      P(x) = sum_i w_i N(x; a_i, 1),  Q(x) = sum_j v_j N(x; b_j, 1).
    Computes E_{X~N(0,1)}[(Σ w_i e^{a_i X - a_i^2/2} - α Σ v_j e^{b_j X - b_j^2/2})_+]
    with Gauss–Hermite quadrature (n_gh ~ 200–300 => ~1e-10 accuracy).
    """
    w = np.asarray(w, dtype=float)
    v = np.asarray(v, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    alpha = math.exp(eps)

    # Gauss–Hermite nodes/weights for ∫ e^{-x^2} f(x) dx
    x, gh_w = hermgauss(n_gh)          # (n,)
    X = np.sqrt(2.0) * x               # map to N(0,1)
    W = gh_w / np.sqrt(np.pi)          # weights for E_{N(0,1)}[·]

    Sa = np.exp(np.outer(X, a) - 0.5 * a**2) @ w
    Sb = np.exp(np.outer(X, b) - 0.5 * b**2) @ v
    S  = Sa - alpha * Sb
    return float(np.maximum(S, 0.0) @ W)

# --------------------------- sections ---------------------------

def section_two_gaussians_vs_geps():
    ds = [1.0, 2.0, 4.0]
    Ds = [3, 10, 20]
    dirs = ['axis', 'ones', 'orth', 'random']
    eps_list = [0.0, 0.3, math.log(2.0)]  # include α=1 TV case
    Nmc = 400_000

    print("== Two single Gaussians (I=J=1) vs g_eps(d, eps) ==")
    print(f"{'eps':>6} {'d':>5} {'D':>4} {'dir':>7} {'MC(P)':>12} {'MC(mix)':>12} {'g_eps':>12} {'rel.err':>10}")
    print("-" * 86)

    for eps in eps_list:
        alpha = math.exp(eps)
        for d in ds:
            for D in Ds:
                for mode in dirs:
                    mu, nu = make_means_two_gauss(D, d, mode, seed=123)
                    w = np.array([1.0]); v = np.array([1.0])

                    estP   = estimate_halpha_mc(mu, nu, w, v, alpha, Nmc=Nmc, mode="P")
                    estMix = estimate_halpha_mc(mu, nu, w, v, alpha, Nmc=Nmc, mode="mix")
                    exact  = float(g_eps_np(d, eps))
                    rel    = abs(estP - exact) / max(1e-12, abs(exact))

                    print(f"{eps:6.3f} {d:5.2f} {D:4d} {mode:>7} {estP:12.6f} {estMix:12.6f} {exact:12.6f} {rel:10.2e}")

def section_axis_mixtures_IJ_gt_1():
    rng = np.random.default_rng(123)
    eps = math.log(2.0)
    alpha = math.exp(eps)
    Nmc = 400_000

    print("\n== Axis-embedded mixtures (I,J>1): exact (Gauss–Hermite) vs MC ==")
    print(f"{'I':>3} {'J':>3} {'D':>3} {'weights':>10} {'exact':>12} {'MC(P)':>12} {'MC(mix)':>12} {'rel.err':>10}")
    print("-" * 88)

    cases = [
        (2, 2, 1,  "uniform"),
        (3, 3, 5,  "uniform"),
        (5, 4, 10, "random"),
        (8, 8, 20, "random"),
        # collapsed P (all μ_i=0), still axis-embedded
        (4, 6, 12, "collapsedP"),
    ]

    for I, J, D, mode in cases:
        mu, nu = build_line_means(I, J, D)
        if mode == "uniform":
            w = np.ones(I + 1) / (I + 1)
            v = np.ones(J + 1) / (J + 1)
            a = -np.arange(I + 1, dtype=float)
            b =  np.arange(J + 1, dtype=float)
            exact = halpha_exact_axis(w, v, a, b, eps, n_gh=240)
        elif mode == "random":
            w = random_simplex(I, rng); v = random_simplex(J, rng)
            a = -np.arange(I + 1, dtype=float)
            b =  np.arange(J + 1, dtype=float)
            exact = halpha_exact_axis(w, v, a, b, eps, n_gh=240)
        elif mode == "collapsedP":
            mu.zero_()
            w = random_simplex(I, rng); v = random_simplex(J, rng)
            a = np.zeros(I + 1, dtype=float)
            b = np.arange(J + 1, dtype=float)
            exact = halpha_exact_axis(w, v, a, b, eps, n_gh=240)
        else:
            raise ValueError(mode)

        estP  = estimate_halpha_mc(mu, nu, w, v, alpha, Nmc=Nmc, mode="P")
        estM  = estimate_halpha_mc(mu, nu, w, v, alpha, Nmc=Nmc, mode="mix")
        rel   = abs(estP - exact) / max(1e-12, abs(exact))
        print(f"{I:3d} {J:3d} {D:3d} {mode:>10} {exact:12.6f} {estP:12.6f} {estM:12.6f} {rel:10.2e}")

        # rotation invariance (collinear ⇒ value unchanged)
        Q = random_orthogonal(D, seed=7)
        mu_r, nu_r = mu @ Q, nu @ Q
        estP_rot = estimate_halpha_mc(mu_r, nu_r, w, v, alpha, Nmc=Nmc, mode="P")
        assert abs(estP_rot - estP) < 5e-3, "rotation invariance broken beyond MC noise"

def section_rotation_and_zero_cases():
    # rotation invariance (two Gaussians)
    d, D, eps = 3.0, 15, math.log(2.0)
    alpha = math.exp(eps)
    mu, nu = make_means_two_gauss(D, d, 'axis', seed=0)
    w = np.array([1.0]); v = np.array([1.0])
    base = estimate_halpha_mc(mu, nu, w, v, alpha, Nmc=300_000, mode="P")
    Q = random_orthogonal(D, seed=11)
    mu_r, nu_r = mu @ Q, nu @ Q
    rot = estimate_halpha_mc(mu_r, nu_r, w, v, alpha, Nmc=300_000, mode="P")

    print("\n== Rotation invariance (two Gaussians) ==")
    print(f"base={base:.6f}, rotated={rot:.6f}, |diff|={abs(base-rot):.3e}")

    # zero case P == Q
    mu0 = torch.zeros(1, D)
    est_zero = estimate_halpha_mc(mu0, mu0, w, w, alpha, Nmc=200_000, mode="P")
    print("\n== Zero case P==Q (should be 0) ==")
    print(f"estimate={est_zero:.6e}")

# ---------------------------- main ----------------------------

def main():
    section_two_gaussians_vs_geps()
    section_axis_mixtures_IJ_gt_1()
    section_rotation_and_zero_cases()

if __name__ == "__main__":
    main()
