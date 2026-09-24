import os

import tomli

# Same rationale as federated/config.py: flwr's `flwr run` CLI needs a
# running SuperLink even for local simulations in this flwr version, and
# flwr.simulation.run_simulation() (what simulation.py actually uses) does
# NOT auto-populate Context.run_config from pyproject.toml -- so
# server_app.py / client_app.py / runtime.py's build_fake_args() read the
# `[tool.flwr.app.config]` table directly instead.
PYPROJECT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pyproject.toml")


def load_run_config():
    with open(PYPROJECT_PATH, "rb") as f:
        data = tomli.load(f)
    return data["tool"]["flwr"]["app"]["config"]
