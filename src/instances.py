# src/hsdgm/instances.py

import numpy as np

# Generate instances summing to 1 in a dictionary
def default_instance(I=2, J=2, D=2, eps=np.log(2.0), seed=0):
    """
    Generate a default Gaussian mixture instance with random weights.

    Parameters
    ----------
    I, J : int
        Number of steps/components for P and Q (exclusive of index 0).
    D : int
        Dimension of the means.
    eps : float
        Privacy parameter ε (alpha = exp(eps)).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys: I, J, D, eps, w, v.
        w and v are arrays of non-negative weights summing to 1.
    """
    rng = np.random.default_rng(seed)
    w = rng.uniform(size=I+1); w /= w.sum()
    v = rng.uniform(size=J+1); v /= v.sum()
    return dict(I=I, J=J, D=D, eps=eps, w=w, v=v)


if __name__ == "__main__":
    inst = default_instance()
    print("Generated default instance:")
    for k, val in inst.items():
        print(f"{k}: {val}")
