"""Self-contained copy of federated/runtime.py's plumbing (LoRA weight
(de)serialization, run_config->fake-args mapping, per-client shard loading,
the process-wide cached base model, cross-round DP accountant history I/O,
the train_lock() used to serialize client training on the shared model) --
see that file for the full rationale behind each piece. Duplicated here
(rather than imported) so federated_correlated_noise/ has zero dependency on federated/.
"""

import contextlib
import fcntl
import json
import os
from types import SimpleNamespace

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from common import build_model_and_tokenizer, load_split, select_one_note_per_subject

CLIENT_STATE_DIR = os.path.join(os.path.dirname(__file__), "client_state")
_TRAIN_LOCK_PATH = os.path.join(CLIENT_STATE_DIR, ".train.lock")


@contextlib.contextmanager
def train_lock():
    """Cross-task mutual exclusion around BaseModelCache's shared PeftModel
    -- see federated/runtime.py's train_lock() docstring for why a plain
    in-process lock isn't sufficient and an OS-level flock() is used
    instead. Uses its own lock file under federated_correlated_noise/client_state/, separate
    from federated/'s, so the two experiments never contend with each other
    (they're never run in the same process anyway).
    """
    os.makedirs(CLIENT_STATE_DIR, exist_ok=True)
    with open(_TRAIN_LOCK_PATH, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def get_lora_ndarrays(peft_model):
    state_dict = get_peft_model_state_dict(peft_model)
    keys = sorted(state_dict.keys())
    ndarrays = [state_dict[k].detach().cpu().float().numpy() for k in keys]
    return ndarrays, keys


def set_lora_ndarrays(peft_model, ndarrays, keys):
    state_dict = {k: torch.from_numpy(v) for k, v in zip(keys, ndarrays)}
    set_peft_model_state_dict(peft_model, state_dict)


def build_fake_args(run_config):
    max_steps = run_config.get("max-steps", 0)
    return SimpleNamespace(
        model_id=run_config["model-id"],
        dataset=run_config.get("dataset-path", ""),
        text_column=run_config.get("text-column", "text"),
        max_length=int(run_config.get("max-length", 128)),
        eval_fraction=float(run_config["eval-fraction"]),
        seed=int(run_config["seed"]),
        lora_r=int(run_config["lora-r"]),
        lora_alpha=int(run_config["lora-alpha"]),
        lora_dropout=float(run_config["lora-dropout"]),
        epochs=int(run_config.get("local-epochs", 1)),
        lr=float(run_config.get("lr", 2e-4)),
        batch_size=int(run_config["local-batch-size"]),
        logical_batch_size_dp_sgd=int(run_config["local-logical-batch-size-dp-sgd"]),
        target_epsilon=float(run_config["target-epsilon-per-round"]),
        max_grad_norm=float(run_config["max-grad-norm"]),
        max_steps=None if not max_steps else int(max_steps),
        log_every=int(run_config.get("log-every", 5)),
    )


def load_client_shard(dataset_path, subject_ids, eval_fraction, seed, one_note_per_subject=True):
    dataset = load_split(dataset_path, "train", eval_fraction, seed)
    subject_id_set = set(subject_ids)
    shard = dataset.filter(lambda row: row["subject_id"] in subject_id_set)
    if one_note_per_subject:
        shard = select_one_note_per_subject(shard, seed)
    return shard


class BaseModelCache:
    """Process-global cache for the loaded 4-bit base model + PEFT wrapper
    -- see federated/runtime.py's BaseModelCache docstring for the full
    rationale (Ray actor process reuse, round-to-round PeftModel reuse)."""

    _model = None
    _tokenizer = None
    _load_count = 0

    @classmethod
    def get_or_load(cls, args):
        with train_lock():
            if cls._model is None:
                cls._load_count += 1
                print(f"[BaseModelCache] loading base model (load #{cls._load_count}, pid={os.getpid()})")
                cls._model, cls._tokenizer = build_model_and_tokenizer(args)
        return cls._model, cls._tokenizer

    @classmethod
    def load_count(cls):
        return cls._load_count


def _history_path(client_id):
    return os.path.join(CLIENT_STATE_DIR, f"client_{client_id}_privacy.json")


def _report_path(client_id):
    return os.path.join(CLIENT_STATE_DIR, f"client_{client_id}_privacy_report.json")


def load_accountant_history(client_id):
    path = _history_path(client_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_accountant_history(client_id, history):
    os.makedirs(CLIENT_STATE_DIR, exist_ok=True)
    with open(_history_path(client_id), "w") as f:
        json.dump(history, f)


def save_client_privacy_report(client_id, server_round, achieved_epsilon, target_epsilon, delta, max_grad_norm):
    os.makedirs(CLIENT_STATE_DIR, exist_ok=True)
    report = {
        "dp_enabled": True,
        "client_id": client_id,
        "round": server_round,
        "target_epsilon": target_epsilon,
        "achieved_epsilon": achieved_epsilon,
        "delta": delta,
        "max_grad_norm": max_grad_norm,
    }
    with open(_report_path(client_id), "w") as f:
        json.dump(report, f, indent=2)
    return report


class LatestParamsHolder:
    """Mutable box the server's evaluate_fn stashes each round's aggregated
    LoRA ndarrays into, so the final global state is retrievable after the
    workflow finishes without digging into Strategy internals."""

    round = None
    ndarrays = None

    @classmethod
    def update(cls, server_round, ndarrays):
        cls.round = server_round
        cls.ndarrays = ndarrays
