# src/hsdgm/pgd/pairwise.py
from __future__ import annotations
import math
from typing import Optional, Tuple, Literal, Callable

import torch

from math_utils import g_eps_torch
from constraints import project

__all__ = ["U_pairwise", "pgd_pairwise"]

def U_pairwise(mu: torch.Tensor,
               nu: torch.Tensor,
               w,
               v,
               eps: float) -> torch.Tensor:
    """
    Deterministic pairwise surrogate:
        U = sum_{i,j} w_i v_j g_eps(||mu_i - nu_j||_2).
    Shapes:
        mu: (I+1, D), nu: (J+1, D); w: len I+1; v: len J+1.
    Returns:
        Scalar tensor (same dtype/device as mu/nu).
    """
    d = torch.cdist(mu, nu, p=2)                       # (I+1, J+1)
    wt = torch.as_tensor(w, dtype=mu.dtype, device=mu.device)
    vt = torch.as_tensor(v, dtype=mu.dtype, device=mu.device)
    return torch.sum(wt[:, None] * vt[None, :] * g_eps_torch(d, eps))

def pgd_pairwise(inst: dict,
                 restarts: int = 10,
                 iters: int = 600,
                 lr: float = 0.04,
                 optimizer: Literal["adam","sgd"] = "adam",
                 initial_scale: float = 0.2,
                 seed: int = 0,
                 return_params: bool = False,
                 callback: Optional[Callable[[int, int, float, torch.Tensor, torch.Tensor], None]] = None
                ) -> float | Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Projected gradient ascent on the pairwise surrogate U.

    inst keys: I,J,D,eps,w,v (as in default_instance)
    Returns:
        best objective value, and optionally (mu,nu).
    """
    I, J, D = int(inst["I"]), int(inst["J"]), int(inst["D"])
    eps, w, v = float(inst["eps"]), inst["w"], inst["v"]

    best_val = -float("inf")
    best_mu = best_nu = None
    g = torch.Generator().manual_seed(seed)

    for r in range(restarts):
        mu = torch.randn(I+1, D, generator=g, dtype=torch.get_default_dtype()) * initial_scale
        nu = torch.randn(J+1, D, generator=g, dtype=torch.get_default_dtype()) * initial_scale
        mu[0].zero_(); nu[0].zero_()
        # ⬇ project once, but COPY INTO the same leaf tensors
        with torch.no_grad():
            mu0, nu0 = project(mu, nu)
            mu.copy_(mu0); nu.copy_(nu0)
        mu.requires_grad_(); nu.requires_grad_()

        params = [mu, nu]
        opt = torch.optim.Adam(params, lr=lr) if optimizer == "adam" else torch.optim.SGD(params, lr=lr, momentum=0.9)

        for t in range(iters):
            opt.zero_grad()
            obj = U_pairwise(mu, nu, w, v, eps)
            (-obj).backward()
            opt.step()

            # ⬇ KEEP SAME LEAF TENSORS; COPY PROJECTION INTO THEM
            with torch.no_grad():
                mu_p, nu_p = project(mu, nu)
                mu.copy_(mu_p); nu.copy_(nu_p)

            if callback:
                callback(r, t, obj.item(), mu, nu)

        with torch.no_grad():
            val = U_pairwise(mu, nu, w, v, eps).item()
        if val > best_val:
            best_val, best_mu, best_nu = val, mu.detach().clone(), nu.detach().clone()
    return (best_val, best_mu, best_nu) if return_params else best_val