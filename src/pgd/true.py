# src/hsdgm/pgd/true2.py
from __future__ import annotations
import math
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from constraints import project

# ──────────────────────────────────────────────────────────────────────────────
# Notebook-faithful Monte Carlo estimator: x ~ P, Hα = E_P[ max(1 - α q/p, 0) ]
# ──────────────────────────────────────────────────────────────────────────────
def sample_H_mixture_true(
    mu: torch.Tensor,
    nu: torch.Tensor,
    w_np,
    v_np,
    alpha: float,
    Nmc: int = 8192,
    rng: Optional[torch.Generator] = None,
    smooth_tau: float = 0.0,   # kept for signature parity; no smoothing used
) -> torch.Tensor:
    """
    Unbiased MC for H_α(P||Q) with P,Q Gaussian mixtures (identity covariances).
    Matches the estimator you shared from the notebook:
      - draw x ~ P
      - compute logP(x), logQ(x) via log-sum-exp
      - return mean(ReLU(1 - α * exp(logQ - logP)))

    Gradients flow through μ,ν via the reparameterized sampling x = μ[idx] + ε.
    """
    device, dtype = mu.device, mu.dtype
    D = mu.shape[1]
    logC = -0.5 * D * math.log(2 * math.pi)

    # mixture weights → tensors (normalize defensively)
    w = torch.as_tensor(w_np, dtype=dtype, device=device)
    v = torch.as_tensor(v_np, dtype=dtype, device=device)
    w = w / w.sum()
    v = v / v.sum()

    # 1) Sample x ~ P
    idx = torch.multinomial(w, Nmc, replacement=True, generator=rng)  # (Nmc,)
    x   = mu[idx] + torch.randn(Nmc, D, device=device, dtype=dtype, generator=rng)

    # 2) log p(x)
    d2P = ((x[:, None, :] - mu[None, :, :])**2).sum(dim=-1)    # (Nmc, I+1)
    logP_comp = -0.5 * d2P + logC
    logP = torch.logsumexp(torch.log(w)[None, :] + logP_comp, dim=1)  # (Nmc,)

    # 3) log q(x)
    d2Q = ((x[:, None, :] - nu[None, :, :])**2).sum(dim=-1)    # (Nmc, J+1)
    logQ_comp = -0.5 * d2Q + logC
    logQ = torch.logsumexp(torch.log(v)[None, :] + logQ_comp, dim=1)  # (Nmc,)

    # 4) integrand under P
    ratio     = torch.exp(logQ - logP)               # q/p
    integrand = F.relu(1.0 - alpha * ratio)          # max(1 - α·q/p, 0)

    # 5) average (no smoothing; matches notebook behavior)
    return integrand.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Notebook-style PGD using the above estimator, with autograd-safe projection
# ──────────────────────────────────────────────────────────────────────────────
def pgd_H_full_true(
    inst: dict,
    restarts: int = 20,
    iters: int = 600,
    lr: float = 0.02,
    Nmc: int = 8192,
    smooth_tau: float = 5.0,          # kept for API parity; not used by estimator
    initial_scale: float = 0.3,
    seed: int = 0,
    return_params: bool = False,
) -> Tuple[float, torch.Tensor, torch.Tensor] | float:
    """
    PGD to maximize Hα(P||Q) directly via MC (matches your notebook structure).
    Differences vs the raw notebook:
      • After each projection, we COPY the projected values back into the SAME
        leaf tensors (mu, nu) so autograd remains intact between iterations.
    """
    I, J, D, eps, w_np, v_np = (inst[k] for k in ("I", "J", "D", "eps", "w", "v"))
    alpha = math.exp(float(eps))

    best_val = -float("inf")
    best_mu = best_nu = None
    rng = torch.Generator().manual_seed(seed)

    for _ in range(restarts):
        # init
        mu = torch.randn(I + 1, D, generator=rng, dtype=torch.get_default_dtype()) * initial_scale
        nu = torch.randn(J + 1, D, generator=rng, dtype=torch.get_default_dtype()) * initial_scale
        mu[0].zero_(); nu[0].zero_()

        # project once, copy into the same leaf tensors
        with torch.no_grad():
            mu_p, nu_p = project(mu, nu)
            mu.copy_(mu_p); nu.copy_(nu_p)

        mu.requires_grad_(); nu.requires_grad_()
        opt = torch.optim.Adam([mu, nu], lr=lr)

        for t in range(iters):
            # (keep tau variable for parity; estimator ignores it)
            _ = smooth_tau * max(0.0, 1.0 - t / max(1, iters - 1))

            opt.zero_grad()
            H_est = sample_H_mixture_true(
                mu, nu, w_np, v_np, alpha,
                Nmc=Nmc, rng=rng, smooth_tau=0.0
            )
            loss = -H_est
            loss.backward()
            opt.step()

            # project back in-place (do NOT rebind mu,nu)
            with torch.no_grad():
                mu_p, nu_p = project(mu, nu)
                mu.copy_(mu_p); nu.copy_(nu_p)

        val = -float(loss.item())
        if val > best_val:
            best_val, best_mu, best_nu = val, mu.detach().clone(), nu.detach().clone()

    if return_params:
        return best_val, best_mu, best_nu
    return best_val
