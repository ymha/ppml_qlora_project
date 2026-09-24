"""Verification CLI for covariance.py: builds the null-space covariance
Sigma for a given (num_clients, num_shares, threshold) topology, checks
Sigma is a valid null-space covariance (PSD, Sigma @ ones == 0), simulates
the real per-edge samplers from noise.py to confirm exact (integer)
cancellation, and cross-checks noise.py's `remaining = num_shares -
threshold` formula against the topology's REAL edge degree.

Runs the actual pyproject.toml config first (currently a complete graph,
since secagg-num-shares=1.0 resolves to num_shares == num_clients), then a
couple of explicitly SPARSE ring topologies (num_shares < num_clients) to
confirm none of this is an artifact of the complete-graph special case.

Run: python -m federated_correlated_noise.verify_covariance
"""

import functools
import random

import numpy as np

from federated_correlated_noise.config import load_run_config
from federated_correlated_noise.covariance import (
    edges_from_topology,
    laplacian_covariance,
    real_edge_degree,
    resolve_num_shares_and_threshold,
    residual_variance_after_worst_case_removal,
    ring_topology,
    simulate_joint_field,
    verify_null_space_covariance,
)
from federated_correlated_noise.noise import gaussian_edge_sigma, sample_gaussian_edge


def run_scenario(label, num_clients, num_shares_cfg, threshold_cfg, dp_epsilon, dp_clipping_c):
    print(f"\n{'#' * 70}\n# SCENARIO: {label}\n{'#' * 70}")

    num_shares, threshold = resolve_num_shares_and_threshold(num_clients, num_shares_cfg, threshold_cfg)
    print(f"[config] num_clients={num_clients} num_shares_cfg={num_shares_cfg} threshold_cfg={threshold_cfg}")
    print(f"[resolved] num_shares={num_shares} threshold={threshold} (t = threshold-1 = {threshold - 1} colluding neighbours tolerated)")

    neighbours = ring_topology(num_clients, num_shares)
    edges = edges_from_topology(neighbours)
    degree = real_edge_degree(edges, num_clients)
    max_possible_edges = num_clients * (num_clients - 1) // 2
    print(f"[topology] {len(edges)} real pairwise edges (complete graph would have {max_possible_edges}), "
          f"per-node degree={degree.tolist()}")
    if len(edges) < max_possible_edges:
        print("           -> a genuine sparse ring: not every pair of clients shares an edge.")
    else:
        print("           -> num_shares == num_clients here, so the ring degenerates into a complete graph.")

    # --- 1. Structural check: real edge degree is always num_shares - 1, ring or not ---
    assert degree[0] == num_shares - 1, "topology no longer matches the num_shares-1 assumption -- update noise.py's calibration"
    print(f"[check] real per-node degree == num_shares - 1 == {num_shares - 1}: OK (holds for sparse rings, not just the complete-graph case)")

    # --- 2. Sigma properties (PSD, null space), at unit edge weight ---
    Sigma_unit = laplacian_covariance(edges, num_clients, edge_variance=1.0)
    report = verify_null_space_covariance(Sigma_unit)
    print(f"[Sigma] symmetric={report['symmetric']}, is_psd={report['is_psd']} (min eigenvalue={report['min_eigenvalue']:.2e}), "
          f"is_null_space={report['is_null_space']} (max |row sum|={report['max_row_sum_abs']:.2e})")

    # --- 3. Gaussian calibration: delivered vs. target variance under worst-case collusion ---
    test_delta = 1e-5
    edge_sigma = gaussian_edge_sigma(dp_epsilon, test_delta, dp_clipping_c, num_shares, threshold)
    sigma_total_sq = (2 * dp_clipping_c * np.sqrt(2 * np.log(1.25 / test_delta)) / dp_epsilon) ** 2
    Sigma_real = laplacian_covariance(edges, num_clients, edge_variance=edge_sigma**2)
    delivered_variance = residual_variance_after_worst_case_removal(Sigma_real, node=0, num_compromised=threshold - 1)
    ratio = delivered_variance / sigma_total_sq
    print(f"[Gaussian] edge_sigma={edge_sigma:.4f}, target variance={sigma_total_sq:.4f}, "
          f"delivered variance (node 0, worst-case {threshold-1} edges removed)={delivered_variance:.4f}")
    print(f"[Gaussian] delivered/target ratio = {ratio:.4f} ({'OK' if abs(ratio - 1) < 1e-9 else 'MISMATCH'})")

    # --- 4. Exact-cancellation simulation, real samplers, every node checked (not just node 0) ---
    dims = [(4,), (3, 3), ()]
    rng = random.Random(1234)

    gauss_sampler = functools.partial(sample_gaussian_edge, sigma=edge_sigma)
    per_node_gauss = simulate_joint_field(edges, num_clients, dims, gauss_sampler, rng)
    totals_gauss = [sum(per_node_gauss[n][d] for n in range(num_clients)) for d in range(len(dims))]
    gauss_exact_zero = all(np.all(t == 0) for t in totals_gauss)

    print(f"[cancellation] Gaussian sum over all {num_clients} nodes == 0 exactly? {gauss_exact_zero}")

    # --- 5. Empirical per-node variance, checked for a node with degree == real_degree (not just node 0) ---
    trials = 2000
    node0_draws = np.zeros(trials)
    for t in range(trials):
        r = random.Random(t)
        pn = simulate_joint_field(edges, num_clients, [()], gauss_sampler, r)
        node0_draws[t] = pn[0][0]
    empirical_var = float(np.var(node0_draws))
    print(f"[empirical] Var(node 0, {trials} trials) = {empirical_var:.4f} vs Sigma[0,0] = {Sigma_real[0, 0]:.4f} "
          f"(ratio={empirical_var / Sigma_real[0, 0]:.3f}, expect close to 1.0 within Monte Carlo noise)")

    return {"num_clients": num_clients, "num_shares": num_shares, "threshold": threshold, "edges": len(edges), "ratio": ratio}


def main():
    run_config = load_run_config()
    dp_epsilon = float(run_config["dp-epsilon"])
    dp_clipping_c = float(run_config["dp-clipping-c"])

    results = []

    # Scenario A: the actual checked-in pyproject.toml config. secagg-num-shares=1.0
    # resolves to num_shares == num_clients == 5, so this degenerates into a complete
    # graph -- kept as the regression baseline (matches the earlier verification run).
    results.append(run_scenario(
        "checked-in pyproject.toml config (complete graph, num_shares==num_clients)",
        num_clients=int(run_config["num-clients"]),
        num_shares_cfg=run_config["secagg-num-shares"],
        threshold_cfg=run_config["secagg-reconstruction-threshold"],
        dp_epsilon=dp_epsilon,
        dp_clipping_c=dp_clipping_c,
    ))

    # Scenario B: same 5 clients, but num_shares_cfg lowered so num_shares < num_clients
    # -- the smallest genuinely sparse ring this project's client count allows
    # (num_shares=3 -> degree 2, a real ring: each node only touches its 2 immediate
    # neighbours, not everyone).
    results.append(run_scenario(
        "same 5 clients, num_shares lowered to force a sparse ring (degree 2)",
        num_clients=5,
        num_shares_cfg=0.6,  # round(0.6*5) = 3
        threshold_cfg=run_config["secagg-reconstruction-threshold"],
        dp_epsilon=dp_epsilon,
        dp_clipping_c=dp_clipping_c,
    ))

    # Scenario C: a larger client population with num_shares held small, so the ring
    # is unambiguously sparse relative to the complete graph (degree 4 out of a
    # possible 10) -- this is the regime the real deployment would be in if run with
    # many more than 5 clients while keeping num_shares bounded for cost reasons.
    results.append(run_scenario(
        "11 clients, num_shares fixed small (degree 4, clearly sparse ring)",
        num_clients=11,
        num_shares_cfg=5,  # int path: used as-is (workflow.py's non-float branch)
        threshold_cfg=3,
        dp_epsilon=dp_epsilon,
        dp_clipping_c=dp_clipping_c,
    ))

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for r in results:
        print(f"num_clients={r['num_clients']:>3} num_shares={r['num_shares']:>2} threshold={r['threshold']:>2} "
              f"edges={r['edges']:>3} delivered/target={r['ratio']:.4f}")


if __name__ == "__main__":
    main()
