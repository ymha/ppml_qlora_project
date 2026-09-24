"""Offline CLI: splits the training set into N client shards by subject_id,
writes federated_correlated_noise/partitions/manifest.json. Self-contained copy of
federated/partition_data.py -- see that file for the full rationale; this
one exists so federated_correlated_noise/ never imports anything from federated/. The two
manifests end up identical given the same pyproject.toml config (same seed,
same dataset, same partitioning algorithm), so results stay comparable
even though the two packages are built independently.
"""

import argparse
import json
import os
import random
from collections import Counter

from common import add_data_args, load_split
from federated_correlated_noise.config import load_run_config

DEFAULT_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "partitions", "manifest.json")


def parse_args():
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    parser.add_argument("--output", default=DEFAULT_MANIFEST_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    num_clients = load_run_config()["num-clients"]
    assert num_clients >= 2, (
        f"pyproject.toml's num-clients={num_clients}, but this protocol requires at "
        f"least 2 clients per round -- see server_app.py's matching check"
    )

    train_split = load_split(args.dataset, "train", args.eval_fraction, args.seed)
    eval_split = load_split(args.dataset, "test", args.eval_fraction, args.seed)

    train_subjects = set(train_split["subject_id"])
    eval_subjects = set(eval_split["subject_id"])
    subject_overlap = train_subjects & eval_subjects
    print(
        f"Note: the split is done at the note level, not the patient level. "
        f"So, a patient (subject_id) can appear in both train and test. "
        f"A patient usually has multiple notes. "
        f"({100 * len(subject_overlap) / len(eval_subjects):.1f}%) of test-split "
        f"patients also have at least one note in the train split."
    )

    note_counts = Counter(train_split["subject_id"])

    subject_ids = sorted(train_subjects)
    random.Random(args.seed).shuffle(subject_ids)
    shards = [subject_ids[i :: num_clients] for i in range(num_clients)]

    seen = set()
    for shard in shards:
        overlap = seen & set(shard)
        assert not overlap, f"{len(overlap)} subject_ids duplicated across shards"
        seen |= set(shard)

    clients = []
    for client_id, shard in enumerate(shards):
        num_rows = sum(note_counts[sid] for sid in shard)
        clients.append({"client_id": client_id, "subject_ids": shard, "num_rows": num_rows})

    manifest = {
        "num_clients": num_clients,
        "seed": args.seed,
        "eval_fraction": args.eval_fraction,
        "dataset": args.dataset,
        "clients": clients,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    row_counts = [c["num_rows"] for c in clients]
    print(f"Wrote manifest for {num_clients} clients to {args.output}")
    print(f"Row counts per client: {row_counts} (min={min(row_counts)}, max={max(row_counts)})")
    print(f"Total rows across shards: {sum(row_counts)} (train split has {len(train_split)})")
    print("Self-check passed: zero subject_id overlap between shards (no patient split across clients).")


if __name__ == "__main__":
    main()
