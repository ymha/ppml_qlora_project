"""Entry point for the correlated-noise variant -- identical CLI/backend
wiring to federated/simulation.py, just launching federated_correlated_noise/'s
ClientApp/ServerApp instead of federated/'s.
"""

import argparse

from flwr.simulation import run_simulation

from federated_correlated_noise.client_app import app as client_app
from federated_correlated_noise.config import load_run_config
from federated_correlated_noise.server_app import app as server_app


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpus", type=int, default=4, help="CPUs reserved per client activation")
    parser.add_argument("--num-gpus", type=float, default=1.0, help="GPU share reserved per client activation")
    return parser.parse_args()


def main():
    args = parse_args()
    run_config = load_run_config()
    num_clients = int(run_config["num-clients"])

    backend_config = {
        "client_resources": {"num_cpus": args.num_cpus, "num_gpus": args.num_gpus},
    }

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
        backend_config=backend_config,
    )


if __name__ == "__main__":
    main()
