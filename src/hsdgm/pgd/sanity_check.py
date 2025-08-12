import math
import numpy as np
import torch

from hsdgm.instances import default_instance
from hsdgm.constraints import check_constraints, project
from hsdgm.pgd.pairwise import pgd_pairwise, U_pairwise
from hsdgm.pgd.true import pgd_H_full_true, sample_H_mixture_true

torch.set_default_dtype(torch.double)

# ---------- helpers ----------
def collapsed_means(I: int, J: int, D: int):
    """μ_i ≡ 0, ν_j = j·e1."""
    mu = torch.zeros(I + 1, D, dtype=torch.get_default_dtype())
    nu = torch.zeros(J + 1, D, dtype=torch.get_default_dtype())
    nu[:, 0] = torch.arange(J + 1, dtype=nu.dtype)
    return mu, nu

def main():
    # same small-but-nontrivial setup we’ve been validating with
    INST_SEED = 0
    PGD_SEED  = 1
    I = J = D = 4

    inst  = default_instance(I=I, J=J, D=D, seed=INST_SEED)
    alpha = math.exp(inst["eps"])
    w, v  = inst["w"], inst["v"]

    # tolerances (MC noise)
    TOL_EQ = 8e-3      # proposal/reeval diff allowed
    TOL_HU = 8e-3      # H ≤ U slack
    TOL_COLL = 8e-3    # collapsed H ≤ PGD-H slack
    N_EVAL = 300_000   # big MC for re-eval/inequality checks

    print(f"dtype={torch.get_default_dtype()} | I=J=D={I} | eps={float(inst['eps']):.3f} | alpha={alpha:.3f}")

    # ── Pairwise PGD (upper bound surrogate) ─────────────────────────────────
    print("\n=== Pairwise PGD (upper bound surrogate U) ===")
    U_best, mu_U, nu_U = pgd_pairwise(inst, restarts=5, iters=400, lr=0.04,
                                      return_params=True, seed=PGD_SEED)
    check_constraints(mu_U, nu_U)
    print(f"U_best = {U_best:.6f}")

    # ── True H via NOTEBOOK-FAITHFUL 'true' PGD (x~P) ───────────────────────
    print("\n=== True Hα via 'true' (x~P) ===")
    H_best, mu_H, nu_H = pgd_H_full_true(inst, restarts=5, iters=600,
                                          Nmc=16_384, lr=0.02,
                                          seed=PGD_SEED, return_params=True)
    check_constraints(mu_H, nu_H)
    print(f"PGD true (running best): {H_best:.6f}")

    # high-precision re-eval (same estimator)
    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(23)
    H_re1 = float(sample_H_mixture_true(mu_H, nu_H, w, v, alpha, Nmc=N_EVAL, rng=g1))
    H_re2 = float(sample_H_mixture_true(mu_H, nu_H, w, v, alpha, Nmc=N_EVAL, rng=g2))
    H_re  = 0.5 * (H_re1 + H_re2)
    print(f"Re-eval true: {H_re1:.6f} (seed 11), {H_re2:.6f} (seed 23)  → mean={H_re:.6f}")

    # Inequality H ≤ U on the same parameters
    U_on_H = U_pairwise(mu_H, nu_H, w, v, inst["eps"]).item()
    print(f"U on true-PGD params: U={U_on_H:.6f} | U - H≈ {U_on_H - H_re:.6f}")
    assert U_on_H + TOL_HU >= H_re, "Inequality violated: H > U beyond tolerance."

    # ── Collapsed configuration (μ≡0, ν=j·e1) ────────────────────────────────
    print("\n=== Collapsed configuration (μ≡0, ν=j·e1) ===")
    mu_c, nu_c = collapsed_means(I, J, D)
    check_constraints(mu_c, nu_c)

    g3 = torch.Generator().manual_seed(101)
    H_coll = float(sample_H_mixture_true(mu_c, nu_c, w, v, alpha, Nmc=N_EVAL, rng=g3))
    U_coll = U_pairwise(mu_c, nu_c, w, v, inst["eps"]).item()
    print(f"Collapsed: H≈{H_coll:.6f}, U={U_coll:.6f}")

    # Collapsed should be ≤ PGD-H (since PGD maximizes H) and ≤ U (upper bound)
    print("\n=== Checks ===")
    print(f"1) Collapsed ≤ PGD-H?   {H_coll:.6f} ≤ {H_re:.6f}   (Δ={H_re - H_coll:.6f})")
    assert H_re + TOL_COLL >= H_coll, "Collapsed H exceeded PGD-H beyond tolerance."

    print(f"2) H ≤ U on collapsed?  {H_coll:.6f} ≤ {U_coll:.6f} (Δ={U_coll - H_coll:.6f})")
    assert U_coll + TOL_HU >= H_coll, "H(collapsed) > U(collapsed) beyond tolerance."

    print(f"3) H ≤ U on PGD params? {H_re:.6f} ≤ {U_on_H:.6f} (Δ={U_on_H - H_re:.6f})")
    assert U_on_H + TOL_HU >= H_re, "H(PGD) > U(PGD) beyond tolerance."

    print("\n✓ Sanity passed: constraints ok, H≤U ok, collapsed ≤ PGD-H ok.")

if __name__ == "__main__":
    main()
