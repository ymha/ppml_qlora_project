"""
Calibrated correlated-noise generation for the null-space (pairwise) term.

Design (see conversation/design notes for the full derivation):
- Threat model: an adversary who colludes with up to t = threshold - 1 of a client's pairwise-noise neighbours (learning their shared z_ij values, 
  not just observing a noisy function of them) should still face the target privacy guarantee on n_i = sum_{j in neighbours} +-z_ij.
- Clipping is L_infinity, per element, to [-clipping_c, clipping_c] -- clipping_c is a hyperparameter, independent of SecAgg's own clipping_range (which was never calibrated for a DP purpose).
- A client's pairwise-noise neighbour count is (num_shares - 1), not num_shares: workflow.py's ring construction gives every node a neighbour SET of size num_shares that includes the node's own id
  (offset=0, used for the self-mask's Shamir share), and mod.py's _share_keys filters that self entry out before treating the rest as real edges -- see federated_correlated_noise/covariance.py's
  verify_covariance.py for a topology-level check of this. So of a client's (num_shares - 1) real edges, up to t = threshold - 1 may be fully compromised, 
  leaving (num_shares - 1 - t) = (num_shares - threshold) edges' worth of independent randomness to rely on.

Mechanism: Gaussian, (epsilon, delta)-DP with delta > 0. A sum of i.i.d. Gaussians is itself Gaussian, so the per-edge std can be divided by sqrt(num_shares - threshold) -- see gaussian_edge_sigma().
"""

import math

import numpy as np

def gaussian_edge_sigma(epsilon: float, delta: float, clipping_c: float, num_shares: int, threshold: int) -> float:
    """
    Per-edge std for the (epsilon, delta)-DP pairwise (correlated-noise) edge term (classic Gaussian mechanism, Dwork & Roth 
    -- valid only for epsilon < 1; L_infinity sensitivity 2*clipping_c per coordinate, reduced by sqrt(num_shares - threshold)).

    num_shares is the size of a client's Shamir share group including its own share (see module docstring), 
    so its real pairwise-edge degree is (num_shares - 1); with up to (threshold - 1) of those edges compromised, 
    (num_shares - 1) - (threshold - 1) = num_shares - threshold edges' worth of independent noise survive in the worst case.
    """
    if not 0 < delta < 1:
        raise ValueError(f"delta = {delta} must be in (0, 1) for the Gaussian mechanism.")
    remaining = num_shares - threshold
    if remaining <= 0:
        raise ValueError(
            f"num_shares - threshold = {remaining} <= 0: no edges remain "
            "uncompromised under this threat model -- increase num_shares or threshold."
        )
    sigma_total = (2 * clipping_c * math.sqrt(2 * math.log(1.25 / delta))) / epsilon
    return sigma_total / math.sqrt(remaining)


def _seed32_from_key(shared_key: bytes) -> int:
    """Same seed-folding convention as flwr's pseudo_rand_gen: XOR every 4-byte chunk of the shared key down to one 32-bit seed."""
    assert len(shared_key) & 0x3 == 0
    seed32 = 0
    for i in range(0, len(shared_key), 4):
        seed32 ^= int.from_bytes(shared_key[i : i + 4], "little")
    return seed32


def sample_gaussian_edge(shared_key: bytes, sigma: float, dimensions_list: list[tuple[int, ...]]) -> list[np.ndarray]:
    """
    Draw one edge's noise, i.i.d. per element, from a Gaussian(0, sigma) rounded to the nearest integer. 
    This project's target regime (epsilon close to 0) makes sigma large, where rounding a continuous draw is an approximation of 
    a true discrete Gaussian (Canonne-Kamath-Steinke) -- the correction from using the exact discrete pmf instead would be negligible at this scale.
    """
    gen = np.random.RandomState(_seed32_from_key(shared_key))
    output = []
    for dimension in dimensions_list:
        if len(dimension) == 0:
            arr = np.array(round(gen.normal(0, sigma)), dtype=np.int64)
        else:
            arr = np.round(gen.normal(0, sigma, dimension)).astype(np.int64)
        output.append(arr)
    return output
