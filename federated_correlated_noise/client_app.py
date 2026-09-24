"""Flower ClientApp for the correlated-noise variant. Self-contained copy of
federated/client_app.py's structure -- see federated/client_app.py for the
detailed rationale of each piece; the only substantive difference is
mods=[correlated_noise_mod] instead of mods=[secaggplus_mod].
"""

import json
import os
import time

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from common import build_tokenized_loader
from federated_correlated_noise.config import load_run_config
from federated_correlated_noise.mod import correlated_noise_mod
from federated_correlated_noise.runtime import (
    BaseModelCache,
    build_fake_args,
    get_lora_ndarrays,
    load_accountant_history,
    load_client_shard,
    save_accountant_history,
    save_client_privacy_report,
    set_lora_ndarrays,
    train_lock,
)
from train import train_dp_sgd

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "partitions", "manifest.json")


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


class FlowerLoRAClient(NumPyClient):
    """One simulated FL client: local QLoRA + Opacus DP-SGD on its own
    MIMIC-IV-Note shard, wrapped by correlated_noise_mod (see ClientApp
    below) so its parameter update is never sent to the server in the clear.
    """

    def __init__(self, client_id, peft_model, tokenizer, shard, run_config):
        self.client_id = client_id
        self.peft_model = peft_model
        self.tokenizer = tokenizer
        self.shard = shard
        self.run_config = run_config

    def fit(self, parameters, config):
        fake_args = build_fake_args({**self.run_config, **config})

        loader = build_tokenized_loader(
            self.shard,
            self.tokenizer,
            fake_args.text_column,
            fake_args.max_length,
            fake_args.logical_batch_size_dp_sgd,
            shuffle=True,
        )

        target_delta = 1.0 / (10 * len(self.shard))
        prior_history = load_accountant_history(self.client_id)

        print(f"[client {self.client_id}] waiting for train_lock at {time.time():.2f}")
        with train_lock():
            print(f"[client {self.client_id}] ACQUIRED train_lock at {time.time():.2f}")
            _, keys = get_lora_ndarrays(self.peft_model)
            set_lora_ndarrays(self.peft_model, parameters, keys)
            final_eps, history = train_dp_sgd(
                self.peft_model, loader, target_delta, fake_args, accountant_history=prior_history
            )
            new_ndarrays, _ = get_lora_ndarrays(self.peft_model)
            print(f"[client {self.client_id}] RELEASING train_lock at {time.time():.2f}")

        save_accountant_history(self.client_id, history)

        server_round = config.get("server_round", 0)
        save_client_privacy_report(
            self.client_id, server_round, final_eps, fake_args.target_epsilon, target_delta, fake_args.max_grad_norm
        )

        metrics = {"achieved_epsilon": final_eps, "client_id": self.client_id, "round": server_round}
        return new_ndarrays, len(self.shard), metrics


def client_fn(context: Context):
    client_id = int(context.node_config["partition-id"])
    run_config = load_run_config()

    manifest = load_manifest()
    client_entry = next(c for c in manifest["clients"] if c["client_id"] == client_id)

    fake_args = build_fake_args(run_config)
    peft_model, tokenizer = BaseModelCache.get_or_load(fake_args)

    shard = load_client_shard(
        manifest["dataset"],
        client_entry["subject_ids"],
        manifest["eval_fraction"],
        manifest["seed"],
        one_note_per_subject=bool(run_config.get("one-note-per-subject", True)),
    )

    return FlowerLoRAClient(client_id, peft_model, tokenizer, shard, run_config).to_client()


app = ClientApp(client_fn=client_fn, mods=[correlated_noise_mod])
