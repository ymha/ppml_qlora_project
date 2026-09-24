"""
Covariance-matrix (graph-Laplacian) model of the null-space (pairwise) correlated noise, built to verify that the ad-hoc per-edge construction in
mod.py/workflow.py actually is what noise.py's calibration assumes it is.

Framing (see noise.py's module docstring for the full threat model): 
the K-client sum functional S(n) = 1^T n has a null space = {n in R^K : 1^T n = 0}, the (K-1)-dim subspace orthogonal to the all-ones vector. 
Any joint noise field n built as a weighted sum of antisymmetric edge terms

    n = sum_{edges (i,j)} z_ij * (e_j - e_i)

lands in that null space automatically (each (e_j - e_i) term already sums to zero), independent of what distribution the z_ij are drawn from 
-- which is what mod.py's per-node `if nid > node_id: += else: -=` does, one edge at a time. The joint covariance of n (per scalar coordinate, since
every coordinate is i.i.d.) is then

    Sigma = sum_{edges (i,j)} Var(z_ij) * (e_i - e_j)(e_i - e_j)^T

i.e. a weighted graph Laplacian: PSD by construction, and Sigma @ ones == 0 -- so the current implementation is already a (uniform-weight,
fixed-ring-topology) special case of a covariance-based design, it just never materializes Sigma or checks its properties explicitly. 
This module does that, and cross-checks noise.py's calibration formulas against the REAL topology instead of the informal "num_shares edges"
description in noise.py's docstring.

Deliberately NOT wired into mod.py/workflow.py (see conversation) -- this is a verification/analysis layer, topology and edge weights are still exactly
what the live protocol uses (ring, uniform weight). Optimizing either is out of scope here.
"""

from __future__ import annotations

import random

import numpy as np


def resolve_num_shares_and_threshold(
    num_samples: int, num_shares_cfg: int | float, threshold_cfg: int | float
) -> tuple[int, int]:
    """
    Mirrors CorrelatedNoiseWorkflow.setup_stage's num_shares/threshold resolution (federated_correlated_noise/workflow.py), both the
    float ("proportion of participants" -- what pyproject.toml's secagg-num-shares / secagg-reconstruction-threshold actually are) and
    int (used as-is) branches. Duplicated here rather than imported so this stays a light, dependency-free verification module 
    -- if workflow.py's resolution logic ever changes, this must be updated to match or the comparisons below become meaningless.
    """
    if isinstance(num_shares_cfg, float):
        num_shares = round(num_shares_cfg * num_samples)
        if num_shares < num_samples and num_shares & 1 == 0:
            num_shares += 1
        if num_shares <= 2:
            num_shares = num_samples
    else:
        num_shares = num_shares_cfg

    if isinstance(threshold_cfg, float):
        threshold = round(threshold_cfg * num_shares)
        threshold = max(threshold, 2)
    else:
        threshold = threshold_cfg

    if num_samples != num_shares and num_shares & 1 == 0:
        num_shares += 1

    return num_shares, threshold


def ring_topology(num_nodes: int, num_shares: int, seed: int = 0) -> dict[int, set[int]]:
    """
    Mirrors workflow.py's setup_stage neighbour-set construction (shuffle sampled node ids, then window of +-half_share around each
    node's shuffled position), including that a node's own id is a member of its own neighbour set (offset=0) -- see mod.py's _share_keys, 
    which is what actually filters self out before treating the rest as real pairwise edges. Node ids are 0..num_nodes-1 here; the real
    protocol shuffles Flower's node ids instead, but shuffling is just a relabelling and does not change the graph's structure, which is all
    this module checks.
    """
    node_ids = list(range(num_nodes))
    random.Random(seed).shuffle(node_ids)
    half_share = num_shares >> 1
    return {
        nid: {node_ids[(idx + offset) % num_nodes] for offset in range(-half_share, half_share + 1)}
        for idx, nid in enumerate(node_ids)
    }


def edges_from_topology(neighbours: dict[int, set[int]]) -> list[tuple[int, int]]:
    """Real pairwise (non-self) edges implied by a neighbour-set mapping --
    this is what mod.py actually exchanges ciphertexts and pairwise noise
    over (see _share_keys: `if nid == state.nid: (self, no ciphertext) else:
    (real edge)`). Asserts the topology is symmetric (j in neighbours[i] iff
    i in neighbours[j]), which workflow.py's construction guarantees for a
    circulant/ring window but isn't checked anywhere in the live code.
    """
    edges = set()
    for i, neighs in neighbours.items():
        for j in neighs:
            if j == i:
                continue
            assert i in neighbours[j], f"asymmetric topology: {j} lists {i} as a neighbour but not vice versa"
            edges.add(tuple(sorted((i, j))))
    return sorted(edges)


def real_edge_degree(edges: list[tuple[int, int]], num_nodes: int) -> np.ndarray:
    """
    Each node's actual number of pairwise-noise edges (self excluded) -- compare against `num_shares` (what noise.py's docstring/formula treats
    as the edge count) to check the off-by-one noted in the verification report below.
    """
    degree = np.zeros(num_nodes, dtype=int)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    return degree


def laplacian_covariance(edges: list[tuple[int, int]], num_nodes: int, edge_variance) -> np.ndarray:
    """
    Sigma = sum_{edges (i,j)} w_ij * (e_i - e_j)(e_i - e_j)^T.
    `edge_variance` is either a single scalar (uniform weight, what the live implementation uses) or a dict {(i, j): w_ij} for future
    weighted-topology work (not used yet -- see module docstring).
    """
    Sigma = np.zeros((num_nodes, num_nodes))
    for i, j in edges:
        w = edge_variance[(i, j)] if isinstance(edge_variance, dict) else edge_variance
        Sigma[i, i] += w
        Sigma[j, j] += w
        Sigma[i, j] -= w
        Sigma[j, i] -= w
    return Sigma


def verify_null_space_covariance(Sigma: np.ndarray, atol: float = 1e-8) -> dict:
    """
    Checks the two properties a valid null-space covariance must have:
    symmetric + PSD (a valid covariance matrix at all) and Sigma @ ones == 0
    (the joint noise field it generates sums to exactly zero across clients, i.e. never perturbs what the server's aggregate sees).
    """
    symmetric = bool(np.allclose(Sigma, Sigma.T, atol=atol))
    row_sums = Sigma @ np.ones(Sigma.shape[0])
    max_row_sum_abs = float(np.max(np.abs(row_sums)))
    eigenvalues = np.linalg.eigvalsh(Sigma)
    min_eigenvalue = float(np.min(eigenvalues))
    return {
        "symmetric": symmetric,
        "max_row_sum_abs": max_row_sum_abs,
        "min_eigenvalue": min_eigenvalue,
        "is_null_space": max_row_sum_abs <= atol,
        "is_psd": min_eigenvalue >= -atol,
    }


def residual_variance_after_worst_case_removal(Sigma: np.ndarray, node: int, num_compromised: int) -> float:
    """
    Variance of `node`'s pairwise noise that survives an adversary who compromises the `num_compromised` highest-variance edges incident to
    `node` (the worst case for the defender -- matches noise.py's threat model of up to threshold-1 colluding neighbours revealing their shared
    z_ij exactly). For the uniform-weight case this is just (degree - num_compromised) * edge_variance, but this also works for
    non-uniform weights (future topology/weight work).
    """
    row = -Sigma[node].copy()  # off-diagonal Laplacian entries are -w_ij
    row[node] = 0.0
    incident_weights = np.sort(row[row > 0])[::-1]
    removed = incident_weights[:num_compromised].sum()
    return float(incident_weights.sum() - removed)


def simulate_joint_field(
    edges: list[tuple[int, int]],
    num_nodes: int,
    dimensions_list: list[tuple[int, ...]],
    edge_sampler,
    rng: random.Random,
) -> dict[int, list[np.ndarray]]:
    """
    Draws the joint noise field using noise.py's real edge-sampling functions (bind sigma/scale via functools.partial before
    passing edge_sampler in), applying mod.py's exact sign convention (`if nid > node_id: += else: -=`, i.e. the higher-id endpoint of each
    edge adds, the lower-id one subtracts). 
    Returns each node's total pairwise-noise vector per parameter shape. Used to verify the cancellation is exact (integer arithmetic, 
    not just zero in expectation) and that empirical per-node variance matches Sigma's diagonal.
    """
    per_node = {n: [np.zeros(d, dtype=np.int64) for d in dimensions_list] for n in range(num_nodes)}
    for i, j in edges:  # i < j by construction (edges_from_topology sorts each pair)
        shared_key = bytes(rng.getrandbits(8) for _ in range(32))
        draw = edge_sampler(shared_key, dimensions_list=dimensions_list)
        for d_idx in range(len(dimensions_list)):
            per_node[j][d_idx] = per_node[j][d_idx] + draw[d_idx]
            per_node[i][d_idx] = per_node[i][d_idx] - draw[d_idx]
    return per_node
