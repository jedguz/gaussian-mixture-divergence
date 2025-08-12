#src/hsdgm/constraints.py

import torch

def check_constraints(mu: torch.Tensor, nu: torch.Tensor, tol: float = 1e-6) -> None:
    """
    Verify that (mu, nu) satisfy all problem constraints (P1), (Q1), (X).

    Note: The 'pairwise' intracluster constraints are equivalent to the original form, but are preferred because they simplify the optimization process.

    Parameters
    ----------
    mu : torch.Tensor, shape (I+1, D)
        Means for P.
    nu : torch.Tensor, shape (J+1, D)
        Means for Q.
    tol : float
        Numerical tolerance for constraint satisfaction.

    Raises
    ------
    AssertionError
        If any constraint is violated.
    """
    I = mu.shape[0] - 1
    J = nu.shape[0] - 1

    # (P1) intra–P chain step size <= 1
    for i in range(1, I + 1):
        d = (mu[i] - mu[i - 1]).norm().item()
        assert d <= 1.0 + tol, f"(P1) step {i} length {d:.6f} > 1"

    # (Q1) intra–Q chain step size <= 1
    for j in range(1, J + 1):
        d = (nu[j] - nu[j - 1]).norm().item()
        assert d <= 1.0 + tol, f"(Q1) step {j} length {d:.6f} > 1"

    # (X) cross-distance constraint <= max(i, j)
    for i in range(I + 1):
        for j in range(J + 1):
            R = max(i, j)
            d = (mu[i] - nu[j]).norm().item()
            assert d <= R + tol, f"(X) @({i},{j}) {d:.6f} > {R}"
