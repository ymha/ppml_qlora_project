"""Self-contained copy of federated/privacy_report.py's aggregation logic
(worst-case cumulative epsilon across clients) -- see that file for the full
rationale. Duplicated here (rather than imported) so federated_correlated_noise/ has zero
dependency on federated/; reads federated_correlated_noise/client_state/ instead.
"""

import glob
import json
import os

CLIENT_STATE_DIR = os.path.join(os.path.dirname(__file__), "client_state")
GLOBAL_ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "global-adapter")


def aggregate_privacy_reports(num_clients, num_rounds):
    """Reads every client_state/client_*_privacy_report.json (written fresh
    each round by client_app.py's fit()), and reports the system-level
    guarantee as the worst-case (max) cumulative epsilon across clients.

    This is a SEPARATE guarantee from the pairwise/correlated-noise
    mechanism's confidentiality property: this epsilon is the formal
    (epsilon, delta) record-level guarantee each client's own DP-SGD
    training gave its data, independent of whatever the aggregation
    protocol does with the resulting update.
    """
    latest_by_client = {}
    for path in glob.glob(os.path.join(CLIENT_STATE_DIR, "client_*_privacy_report.json")):
        with open(path) as f:
            report = json.load(f)
        client_id = report["client_id"]
        existing = latest_by_client.get(client_id)
        if existing is None or report["round"] > existing["round"]:
            latest_by_client[client_id] = report

    per_client_achieved_epsilon = {
        str(client_id): report["achieved_epsilon"] for client_id, report in latest_by_client.items()
    }
    per_client_delta = {str(client_id): report["delta"] for client_id, report in latest_by_client.items()}
    worst_case_epsilon = max(per_client_achieved_epsilon.values()) if per_client_achieved_epsilon else None
    worst_case_client_id = (
        max(per_client_achieved_epsilon, key=per_client_achieved_epsilon.get) if per_client_achieved_epsilon else None
    )
    any_report = next(iter(latest_by_client.values()), {})

    return {
        "dp_enabled": True,
        "secagg_enabled": False,
        "correlated_noise_enabled": True,
        "achieved_epsilon": worst_case_epsilon,
        "delta": per_client_delta.get(worst_case_client_id) if worst_case_client_id else None,
        "target_epsilon": any_report.get("target_epsilon"),
        "max_grad_norm": any_report.get("max_grad_norm"),
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "clients_reported": len(latest_by_client),
        "per_client_achieved_epsilon": per_client_achieved_epsilon,
        "per_client_delta": per_client_delta,
        "system_worst_case_epsilon": worst_case_epsilon,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--num-rounds", type=int, required=True)
    parser.add_argument("--output-dir", default=GLOBAL_ADAPTER_DIR)
    args = parser.parse_args()

    report = aggregate_privacy_reports(args.num_clients, args.num_rounds)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "privacy_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path}: system_worst_case_epsilon={report['system_worst_case_epsilon']}")


if __name__ == "__main__":
    main()
