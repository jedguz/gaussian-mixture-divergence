# src/hsdgm/math_utils.py

import math
import numpy as np
import torch
from scipy.stats import norm

__all__ = [
    "phi_np",
    "torch_cdf",
    "g_eps_np",
    "gprime_eps_np",
    "g_eps_torch"
]

# -------------------------------------------------------------------
# NumPy helpers
# -------------------------------------------------------------------

def phi_np(x: float) -> float:
    """
    Standard Gaussian CDF Φ(x) using scipy.stats.norm.cdf.
    """
    return norm.cdf(x)


def g_eps_np(d, eps):
    """
    Hockey-stick divergence g_ε(d) for two N(·,I) separated by distance d.
    NumPy version, safe for d=0 (returns exactly 0).

    Parameters
    ----------
    d : float or np.ndarray
        Euclidean distance(s) between means.
    eps : float
        Privacy parameter ε.

    Returns
    -------
    np.ndarray or float
        Divergence value(s).
    """
    _SAFETY = 1e-12
    d = np.asarray(d, dtype=float)
    safe_mask = d > _SAFETY
    out = np.zeros_like(d, dtype=float)

    if np.any(safe_mask):
        ds = d[safe_mask]
        out[safe_mask] = norm.cdf(ds/2 - eps/ds) - np.exp(eps) * norm.cdf(-ds/2 - eps/ds)

    return out


def gprime_eps_np(d, eps):
    """
    Derivative of g_ε(d) w.r.t. d.
    NumPy version, with safety clamp for small d.
    """
    _SAFETY = 1e-12
    d = np.maximum(np.asarray(d, dtype=float), _SAFETY)
    a = d/2 - eps/d
    b = d/2 + eps/d
    return 0.5 * norm.pdf(a) * (1 + 2*eps/d**2) + \
           0.5 * np.exp(eps) * norm.pdf(b) * (1 - 2*eps/d**2)


# -------------------------------------------------------------------
# Torch helpers
# -------------------------------------------------------------------

def torch_cdf(x: torch.Tensor) -> torch.Tensor:
    """
    Standard Gaussian CDF Φ(x) using torch.erf.
    """
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def g_eps_torch(d: torch.Tensor, eps) -> torch.Tensor:
    """
    Hockey-stick divergence g_ε(d) for two N(·,I) separated by distance d.
    Torch version, safe for d=0 and works with eps as float or Tensor.

    Parameters
    ----------
    d : torch.Tensor
        Pairwise distances.
    eps : float or torch.Tensor
        Privacy parameter ε.

    Returns
    -------
    torch.Tensor
        Divergence values, same shape as d.
    """
    _SAFETY = 1e-12
    eps_t = torch.as_tensor(eps, device=d.device, dtype=d.dtype)
    d_safe = torch.clamp(d, min=_SAFETY)
    term1 = torch_cdf(d_safe/2 - eps_t/d_safe)
    term2 = torch.exp(eps_t) * torch_cdf(-d_safe/2 - eps_t/d_safe)
    return term1 - term2
