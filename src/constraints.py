import torch

__all__ = ["check_constraints", "project"]

def check_constraints(mu: torch.Tensor, nu: torch.Tensor, tol: float = 1e-6) -> None:
    """
    Verify (P1) ||μ_i−μ_{i-1}|| ≤ 1, (Q1) ||ν_j−ν_{j-1}|| ≤ 1, and (X) ||μ_i−ν_j|| ≤ max(i,j).
    Shapes: mu (I+1,D), nu (J+1,D).
    """
    I = mu.shape[0] - 1
    J = nu.shape[0] - 1

    # (P1)
    for i in range(1, I + 1):
        d = (mu[i] - mu[i - 1]).norm().item()
        assert d <= 1.0 + tol, f"(P1) step {i} length {d:.6f} > 1"

    # (Q1)
    for j in range(1, J + 1):
        d = (nu[j] - nu[j - 1]).norm().item()
        assert d <= 1.0 + tol, f"(Q1) step {j} length {d:.6f} > 1"

    # (X)
    for i in range(I + 1):
        for j in range(J + 1):
            R = max(i, j)
            d = (mu[i] - nu[j]).norm().item()
            assert d <= R + tol, f"(X) @({i},{j}) {d:.6f} > {R}"


def project(
    mu: torch.Tensor,
    nu: torch.Tensor,
    *,
    max_iters: int = 8,
    tol: float = 1e-6,
    fix_origins: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project (approximately) onto the feasible set:
      (P1) ||μ_i−μ_{i-1}|| ≤ 1, (Q1) ||ν_j−ν_{j-1}|| ≤ 1, (X) ||μ_i−ν_j|| ≤ max(i,j).
    If fix_origins=True, μ_0 and ν_0 are held at 0.

    Returns new (mu, nu). Operates in-place on clones (originals unchanged).
    """
    mu = mu.clone()
    nu = nu.clone()
    with torch.no_grad():
        if fix_origins:
            mu[0].zero_()
            nu[0].zero_()

        for _ in range(max_iters):
            max_delta = 0.0

            # --- (P1) clip μ chain steps to length ≤ 1
            for i in range(1, mu.shape[0]):
                d = mu[i] - mu[i - 1]
                n = d.norm()
                if n > 1.0:
                    new_i = mu[i - 1] + d / n
                    max_delta = max(max_delta, (mu[i] - new_i).norm().item())
                    mu[i].copy_(new_i)

            # --- (Q1) clip ν chain steps to length ≤ 1
            for j in range(1, nu.shape[0]):
                d = nu[j] - nu[j - 1]
                n = d.norm()
                if n > 1.0:
                    new_j = nu[j - 1] + d / n
                    max_delta = max(max_delta, (nu[j] - new_j).norm().item())
                    nu[j].copy_(new_j)

            # --- (X) enforce cross distances ≤ max(i,j)
            I = mu.shape[0] - 1
            J = nu.shape[0] - 1
            for i in range(I + 1):
                for j in range(J + 1):
                    R = max(i, j)
                    d = mu[i] - nu[j]
                    n = d.norm()
                    if n > R and n > 0:
                        delta = float(n - R)
                        u = d / n  # unit vector from ν_j to μ_i

                        # split the correction; keep origins fixed if requested
                        if fix_origins and i == 0 and j == 0:
                            pass  # already zero & R=0
                        elif fix_origins and i == 0:
                            # move only ν_j towards μ_0
                            nu[j].add_(delta * u)
                            max_delta = max(max_delta, delta)
                        elif fix_origins and j == 0:
                            # move only μ_i towards ν_0
                            mu[i].sub_(delta * u)
                            max_delta = max(max_delta, delta)
                        else:
                            # split correction equally
                            corr = 0.5 * delta
                            mu[i].sub_(corr * u)
                            nu[j].add_(corr * u)
                            max_delta = max(max_delta, corr)

            if fix_origins:
                mu[0].zero_()
                nu[0].zero_()

            if max_delta < tol:
                break

    return mu, nu
