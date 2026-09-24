"""Flower ServerApp for the correlated-noise variant. Self-contained copy of
federated/server_app.py's structure (FedAvg strategy, centralized held-out
evaluation, saving the final aggregated adapter) -- the only substantive
difference is CorrelatedNoiseWorkflow instead of SecAggPlusWorkflow.
"""

import json
import os

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, SimpleClientManager
from flwr.server.compat import LegacyContext
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow

from common import build_eval_loader, build_model_and_tokenizer, perplexity
from federated_correlated_noise.client_app import load_manifest
from federated_correlated_noise.config import load_run_config
from federated_correlated_noise.privacy_report import aggregate_privacy_reports
from federated_correlated_noise.runtime import LatestParamsHolder, build_fake_args, get_lora_ndarrays, set_lora_ndarrays
from federated_correlated_noise.workflow import CorrelatedNoiseWorkflow

GLOBAL_ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "global-adapter")

app = ServerApp()


@app.main()
def main(grid, context: Context) -> None:
    run_config = load_run_config()
    num_clients = int(run_config["num-clients"])
    num_rounds = int(run_config["num-server-rounds"])

    manifest = load_manifest()
    assert len(manifest["clients"]) == num_clients, (
        f"pyproject.toml's num-clients={num_clients} does not match "
        f"manifest.json's {len(manifest['clients'])} client shards - rerun partition_data.py"
    )
    # CorrelatedNoiseWorkflow's pairwise term needs the same >=2 neighbour
    # structure SecAggPlusWorkflow does -- see federated_correlated_noise/workflow.py.
    assert num_clients >= 2, f"num-clients={num_clients}, but the pairwise term needs at least 2 clients per round"
    secagg_max_weight = float(run_config["secagg-max-weight"])
    max_shard_rows = max(c["num_rows"] for c in manifest["clients"])
    assert secagg_max_weight > max_shard_rows, (
        f"secagg-max-weight={secagg_max_weight} must exceed the largest shard's row count ({max_shard_rows})"
    )

    fake_args = build_fake_args(run_config)
    fake_args.dataset = manifest["dataset"]

    peft_model, tokenizer = build_model_and_tokenizer(fake_args)
    initial_ndarrays, keys = get_lora_ndarrays(peft_model)
    initial_parameters = ndarrays_to_parameters(initial_ndarrays)

    eval_loader = build_eval_loader(tokenizer, fake_args)
    eval_max_batches = run_config.get("eval-max-batches", 0) or None
    device = torch.device("cuda")

    def evaluate_fn(server_round, parameters_ndarrays, config):
        set_lora_ndarrays(peft_model, parameters_ndarrays, keys)
        loss, ppl = perplexity(peft_model, eval_loader, device, eval_max_batches)
        print(f"[federated_correlated_noise.server_app] round {server_round}: held-out loss={loss:.4f} perplexity={ppl:.2f}")
        LatestParamsHolder.update(server_round, parameters_ndarrays)
        return loss, {"perplexity": ppl}

    def on_fit_config_fn(server_round):
        return {"server_round": server_round}

    secagg_num_shares = run_config["secagg-num-shares"]
    secagg_threshold = run_config["secagg-reconstruction-threshold"]

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=on_fit_config_fn,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
    )

    legacy_context = LegacyContext(
        context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_manager=SimpleClientManager(),
    )

    secagg_timeout = run_config["secagg-timeout"]
    workflow = DefaultWorkflow(
        fit_workflow=CorrelatedNoiseWorkflow(
            num_shares=secagg_num_shares,
            reconstruction_threshold=secagg_threshold,
            max_weight=secagg_max_weight,
            clipping_range=float(run_config["secagg-clipping-range"]),
            quantization_range=int(run_config["secagg-quantization-range"]),
            timeout=None if not secagg_timeout else float(secagg_timeout),
            dp_epsilon=float(run_config["dp-epsilon"]),
            dp_delta=float(run_config["dp-delta"]),
            dp_clipping_c=float(run_config["dp-clipping-c"]),
        )
    )

    workflow(grid, legacy_context)

    assert LatestParamsHolder.ndarrays is not None, "no round ever completed evaluation"
    set_lora_ndarrays(peft_model, LatestParamsHolder.ndarrays, keys)
    peft_model.save_pretrained(GLOBAL_ADAPTER_DIR)
    tokenizer.save_pretrained(GLOBAL_ADAPTER_DIR)
    print(f"Saved correlated-noise global adapter to {GLOBAL_ADAPTER_DIR} (final round={LatestParamsHolder.round})")

    privacy_report = aggregate_privacy_reports(num_clients=num_clients, num_rounds=num_rounds)
    with open(os.path.join(GLOBAL_ADAPTER_DIR, "privacy_report.json"), "w") as f:
        json.dump(privacy_report, f, indent=2)
    print(f"Saved privacy_report.json: system_worst_case_epsilon={privacy_report['system_worst_case_epsilon']}")
