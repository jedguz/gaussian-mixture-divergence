# src/hsdgm/estimators.py
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "logpdf_iso_gauss",
    "logpdf_mog",
    "sample_mog",
    "estimate_halpha_mc",         # unbiased MC for H_alpha
    "estimate_halpha_mc_batched", # memory-safe wrapper
]

# --------------------------- core log-densities --------------------------- #

def logpdf_iso_gauss(x: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """
    log N(x; mean, I_D) for x, mean with shape (..., D).
    Returns shape broadcasted over leading dims of x and mean → (...,).
    """
    D = x.shape[-1]
    diff2 = ((x - mean) ** 2).sum(dim=-1)
    return -0.5 * diff2 - 0.5 * D * math.log(2.0 * math.pi)


def logpdf_mog(x: torch.Tensor, means: torch.Tensor, weights) -> torch.Tensor:
    """
    Log-density of a Mixture of Gaussians with identity covariance.

    x      : (N, D)
    means  : (K, D)
    weights: array-like of length K, positive and sum to 1

    Returns: (N,) tensor with log-density log sum_k w_k N(x; mu_k, I)
    """
    w = torch.as_tensor(weights, dtype=x.dtype, device=x.device)
    if not torch.all(w > 0):
        raise ValueError("All mixture weights must be > 0.")
    if not torch.isclose(w.sum(), torch.tensor(1.0, dtype=w.dtype, device=w.device), rtol=1e-5, atol=1e-8):
        # normalise defensively
        w = w / w.sum()

    # (N, K) of component logpdfs
    # Using cdist is fine, but explicit form is often faster/stable here.
    # d2[n, k] = ||x_n - mu_k||^2
    d2 = ((x[:, None, :] - means[None, :, :]) ** 2).sum(dim=-1)
    logC = -0.5 * x.shape[1] * math.log(2.0 * math.pi)
    comp_logpdf = -0.5 * d2 + logC  # (N, K)

    # log-sum-exp across components with weights
    return torch.logsumexp(torch.log(w)[None, :] + comp_logpdf, dim=1)


# --------------------------- sampling utilities -------------------------- #

def sample_mog(means: torch.Tensor, weights, N: int, rng: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Draw N samples from MoG with identity covariance.
    means  : (K, D)
    weights: length-K (sum to 1)
    Returns: (N, D) samples
    """
    device, dtype = means.device, means.dtype
    w = torch.as_tensor(weights, dtype=dtype, device=device)
    w = w / w.sum()
    idx = torch.multinomial(w, N, replacement=True, generator=rng)  # (N,)
    eps = torch.randn(N, means.shape[1], dtype=dtype, device=device, generator=rng)
    return means[idx] + eps


# --------------------------- H_alpha estimators --------------------------- #

def _soft_hinge(u: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Softplus-based smoothing of positive part for gradient stability.
    tau=0 → exact ReLU. Otherwise, softplus(u * tau)/tau.
    """
    if tau <= 0:
        return F.relu(u)
    return F.softplus(u * tau) / tau


@torch.no_grad()
def estimate_halpha_mc(
    mu: torch.Tensor,
    nu: torch.Tensor,
    w,
    v,
    alpha: float,
    Nmc: int = 200_000,
    mode: str = "P",                # "P" (unbiased) or "mix" (unbiased, lower variance)
    rng: Optional[torch.Generator] = None,
    smooth_tau: float = 0.0,
) -> float:
    """
    Unbiased Monte Carlo estimator for H_alpha(P || Q) = ∫ (P - αQ)_+ dx,
    with P, Q Gaussian mixtures N(·; μ_i/ν_j, I).

    Modes:
      • "P"   : sample x ~ P, return E_P[(1 - α·Q/P)_+].  (UNBIASED)
      • "mix" : sample x ~ M = 0.5P + 0.5Q, return E_M[((P-αQ)_+) / M]. (UNBIASED)

    Arguments
    ---------
    mu : (I+1, D) tensor of P means
    nu : (J+1, D) tensor of Q means
    w, v : arrays of weights (sum to 1)
    alpha : > 0, with α = e^ε
    Nmc : number of MC samples
    mode : "P" or "mix"
    rng : optional torch.Generator
    smooth_tau : hinge smoothing (0 → exact)

    Returns
    -------
    float
        MC estimate of H_alpha.

    Notes
    -----
    • This function avoids the common pitfall that wrecked earlier results:
      if you sample from 0.5(P+Q) you *must* divide by M(x); if you sample
      from P you *must* use the ratio Q/P inside the hinge. Both here are correct.
    """
    device, dtype = mu.device, mu.dtype
    w_t = torch.as_tensor(w, dtype=dtype, device=device)
    v_t = torch.as_tensor(v, dtype=dtype, device=device)

    if mode == "P":
        # x ~ P
        x = sample_mog(mu, w_t, Nmc, rng=rng)                     # (N, D)
        logP = logpdf_mog(x, mu, w_t)                             # (N,)
        logQ = logpdf_mog(x, nu, v_t)                             # (N,)
        # integrand: max(1 - α Q/P, 0)
        integrand = _soft_hinge(1.0 - alpha * torch.exp(logQ - logP), smooth_tau)
        return float(integrand.mean().item())

    elif mode == "mix":
        # x ~ M = 0.5 P + 0.5 Q
        # sample half from P, half from Q (or as close as possible)
        N1 = Nmc // 2
        N2 = Nmc - N1
        xP = sample_mog(mu, w_t, N1, rng=rng)
        xQ = sample_mog(nu, v_t, N2, rng=rng)
        x = torch.cat([xP, xQ], dim=0)                             # (N, D)

        logP = logpdf_mog(x, mu, w_t)
        logQ = logpdf_mog(x, nu, v_t)

        P = torch.exp(logP)
        Q = torch.exp(logQ)
        M = 0.5 * (P + Q)
        integrand = _soft_hinge(P - alpha * Q, smooth_tau) / M
        return float(integrand.mean().item())

    else:
        raise ValueError("mode must be 'P' or 'mix'")


@torch.no_grad()
def estimate_halpha_mc_batched(
    mu: torch.Tensor,
    nu: torch.Tensor,
    w,
    v,
    alpha: float,
    Nmc: int = 1_000_000,
    batch: int = 100_000,
    mode: str = "P",
    rng: Optional[torch.Generator] = None,
    smooth_tau: float = 0.0,
) -> float:
    """
    Memory-safe wrapper: accumulates mean over batches.
    """
    remains = Nmc
    acc = 0.0
    n_seen = 0
    while remains > 0:
        b = min(batch, remains)
        val = estimate_halpha_mc(mu, nu, w, v, alpha, Nmc=b, mode=mode, rng=rng, smooth_tau=smooth_tau)
        acc += val * b
        n_seen += b
        remains -= b
    return acc / n_seen
