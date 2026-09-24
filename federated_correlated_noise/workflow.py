# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified by Youngmok Ha (with Claude Code), 2026: ported from flwr 1.30.0
# (flwr/server/workflow/secure_aggregation/secaggplus_workflow.py) and changed so the pairwise mask is DP-calibrated
# correlated noise (see noise.py) and the setup stage carries DP parameters.
"""
Server side of the correlated-noise variant of Flower's SecAgg+ protocol (pairs with federated_correlated_noise/mod.py on the client side).

Ported from flwr 1.30.0's SecAggPlusWorkflow (flwr/server/workflow/secure_aggregation/secaggplus_workflow.py) and modified: the stage logic
is copied into this file rather than inherited or imported -- only generic crypto/math utilities (Shamir, ECDH+HKDF, quantization, ndarray arithmetic) are reused. The four stages (setup ->
share_keys -> collect_masked_vectors -> unmask), ring neighbour topology, threshold checks, self-mask removal and dequantization are the same as
upstream. The only substantive differences:

- setup_stage additionally sends dp_epsilon / dp_delta / dp_clipping_c to every client (see mod.py's check_configs / _setup).
- unmask_stage regenerates a dropped client's pairwise terms with noise.py's DP-calibrated Gaussian per-edge sampler instead of
  flwr's uniform pseudo_rand_gen, using the same calibration the clients used -- so the pairwise (null-space) term still cancels exactly
  in the aggregate.

The self-mask is KEPT unchanged from upstream. Note it does NOT hide an active client's vector from the server: unmask_stage reconstructs every
active client's rd_seed, so a server that retains individual y_i recovers q_i + (pairwise term) per client. As in upstream SecAgg+, the self-mask only
protects a client that was treated as dropped (its sk1 reconstructed) but whose y_i still arrived. Per-client hiding from the server rests on the
pairwise term alone.
"""

import random
from dataclasses import dataclass, field
from logging import DEBUG, ERROR, INFO, WARN
from typing import cast

import flwr.common.recorddict_compat as compat
from flwr.app.message_type import MessageType
from flwr.common import (
    ConfigRecord,
    Context,
    FitRes,
    Message,
    NDArrays,
    RecordDict,
    bytes_to_ndarray,
    log,
    ndarrays_to_parameters,
)
from flwr.common.secure_aggregation.crypto.shamir import combine_shares
from flwr.common.secure_aggregation.crypto.symmetric_encryption import generate_shared_key
from flwr.common.secure_aggregation.ndarrays_arithmetic import (
    factor_extract,
    get_parameters_shape,
    parameters_addition,
    parameters_mod,
    parameters_subtraction,
)
from flwr.common.secure_aggregation.quantization import dequantize
from flwr.common.secure_aggregation.secaggplus_constants import RECORD_KEY_CONFIGS, Key, Stage
from flwr.common.secure_aggregation.secaggplus_utils import pseudo_rand_gen
from flwr.server.client_proxy import ClientProxy
from flwr.server.compat.legacy_context import LegacyContext
from flwr.server.grid import Grid
from flwr.server.workflow.constant import MAIN_CONFIGS_RECORD, MAIN_PARAMS_RECORD
from flwr.server.workflow.constant import Key as WorkflowKey
from flwr.supercore.primitives.asymmetric import bytes_to_private_key, bytes_to_public_key

from federated_correlated_noise.mod import DP_CLIPPING_C, DP_DELTA, DP_EPSILON
from federated_correlated_noise.noise import gaussian_edge_sigma, sample_gaussian_edge


@dataclass
class WorkflowState:  # pylint: disable=too-many-instance-attributes
    nid_to_proxies: dict[int, ClientProxy] = field(default_factory=dict)
    nid_to_fitins: dict[int, RecordDict] = field(default_factory=dict)
    sampled_node_ids: set[int] = field(default_factory=set)
    active_node_ids: set[int] = field(default_factory=set)
    num_shares: int = 0
    threshold: int = 0
    clipping_range: float = 0.0
    quantization_range: int = 0
    mod_range: int = 0
    max_weight: float = 0.0
    dp_epsilon: float = 0.0
    dp_delta: float = 0.0
    dp_clipping_c: float = 0.0
    nid_to_neighbours: dict[int, set[int]] = field(default_factory=dict)
    nid_to_publickeys: dict[int, list[bytes]] = field(default_factory=dict)
    forward_srcs: dict[int, list[int]] = field(default_factory=dict)
    forward_ciphertexts: dict[int, list[bytes]] = field(default_factory=dict)
    aggregate_ndarrays: NDArrays = field(default_factory=list)
    legacy_results: list[tuple[ClientProxy, FitRes]] = field(default_factory=list)
    failures: list[Exception] = field(default_factory=list)


class CorrelatedNoiseWorkflow:  # pylint: disable=too-many-instance-attributes
    """
    SecAggPlusWorkflow's protocol (self-mask + pairwise term, dropout-tolerant via Shamir), ported from flwr 1.30.0, with the pairwise
    term drawn from DP-calibrated noise instead of a uniform PRG mask.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_shares: int | float,
        reconstruction_threshold: int | float,
        *,
        max_weight: float = 1000.0,
        clipping_range: float = 8.0,
        quantization_range: int = 4194304,
        modulus_range: int = 4294967296,
        timeout: float | None = None,
        dp_epsilon: float = 8.0,
        dp_delta: float = 1e-5,
        dp_clipping_c: float = 8.0,
    ) -> None:
        self.num_shares = num_shares
        self.reconstruction_threshold = reconstruction_threshold
        self.max_weight = max_weight
        self.clipping_range = clipping_range
        self.quantization_range = quantization_range
        self.modulus_range = modulus_range
        self.timeout = timeout
        # Pairwise (correlated-noise) edge term's own (epsilon, delta)-DP Gaussian
        # calibration -- see noise.py. dp_delta must be in (0, 1).
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_clipping_c = dp_clipping_c
        self._check_init_params()

    def __call__(self, grid: Grid, context: Context) -> None:
        if not isinstance(context, LegacyContext):
            raise TypeError(f"Expect a LegacyContext, but get {type(context).__name__}.")
        state = WorkflowState()
        steps = (
            self.setup_stage,
            self.share_keys_stage,
            self.collect_masked_vectors_stage,
            self.unmask_stage,
        )
        log(INFO, "Correlated-noise aggregation commencing.")
        for step in steps:
            if not step(grid, context, state):
                log(INFO, "Correlated-noise aggregation halted.")
                return
        log(INFO, "Correlated-noise aggregation completed.")

    def _check_init_params(self) -> None:  # pylint: disable=too-many-branches
        if not isinstance(self.num_shares, (int, float)):
            raise TypeError("`num_shares` must be of type int or float.")
        if isinstance(self.num_shares, int):
            if self.num_shares == 1:
                self.num_shares = 1.0
            elif self.num_shares <= 2:
                raise ValueError("`num_shares` as an integer must be greater than 2.")
            elif self.num_shares > self.modulus_range / self.quantization_range:
                log(
                    WARN,
                    "A `num_shares` larger than `modulus_range / quantization_range` "
                    "will potentially cause overflow when computing the aggregated "
                    "model parameters.",
                )
        elif self.num_shares <= 0:
            raise ValueError("`num_shares` as a float must be greater than 0.")

        if not isinstance(self.reconstruction_threshold, (int, float)):
            raise TypeError("`reconstruction_threshold` must be of type int or float.")
        if isinstance(self.reconstruction_threshold, int):
            if self.reconstruction_threshold == 1:
                self.reconstruction_threshold = 1.0
            elif isinstance(self.num_shares, int):
                if self.reconstruction_threshold >= self.num_shares:
                    raise ValueError("`reconstruction_threshold` must be less than `num_shares`.")
        else:
            if not 0 < self.reconstruction_threshold <= 1:
                raise ValueError(
                    "If `reconstruction_threshold` is a float, it must be greater than 0 and less than or equal to 1."
                )

        if self.max_weight <= 0:
            raise ValueError("`max_weight` must be greater than 0.")
        if not isinstance(self.quantization_range, int) or self.quantization_range <= 0:
            raise ValueError("`quantization_range` must be an integer and greater than 0.")
        if not isinstance(self.modulus_range, int) or self.modulus_range <= self.quantization_range:
            raise ValueError("`modulus_range` must be an integer and greater than `quantization_range`.")
        if bin(self.modulus_range).count("1") != 1:
            raise ValueError("`modulus_range` must be a power of 2.")

    def _check_threshold(self, state: WorkflowState) -> bool:
        for node_id in state.sampled_node_ids:
            active_neighbors = state.nid_to_neighbours[node_id] & state.active_node_ids
            if len(active_neighbors) < state.threshold:
                log(ERROR, "Insufficient available nodes.")
                return False
        return True

    def setup_stage(  # pylint: disable=too-many-locals
        self, grid: Grid, context: LegacyContext, state: WorkflowState
    ) -> bool:
        cfg = context.state.config_records[MAIN_CONFIGS_RECORD]
        current_round = cast(int, cfg[WorkflowKey.CURRENT_ROUND])
        parameters = compat.arrayrecord_to_parameters(context.state.array_records[MAIN_PARAMS_RECORD], keep_input=True)
        proxy_fitins_lst = context.strategy.configure_fit(current_round, parameters, context.client_manager)
        if not proxy_fitins_lst:
            log(INFO, "configure_fit: no clients selected, cancel")
            return False

        state.nid_to_fitins = {
            proxy.node_id: compat.fitins_to_recorddict(fitins, True) for proxy, fitins in proxy_fitins_lst
        }
        state.nid_to_proxies = {proxy.node_id: proxy for proxy, _ in proxy_fitins_lst}

        sampled_node_ids = list(state.nid_to_fitins.keys())
        num_samples = len(sampled_node_ids)
        if num_samples < 2:
            log(ERROR, "The number of samples should be greater than 1.")
            return False
        if isinstance(self.num_shares, float):
            state.num_shares = round(self.num_shares * num_samples)
            if state.num_shares < num_samples and state.num_shares & 1 == 0:
                state.num_shares += 1
            if state.num_shares <= 2:
                state.num_shares = num_samples
        else:
            state.num_shares = self.num_shares
        if isinstance(self.reconstruction_threshold, float):
            state.threshold = round(self.reconstruction_threshold * state.num_shares)
            state.threshold = max(state.threshold, 2)
        else:
            state.threshold = self.reconstruction_threshold
        state.active_node_ids = set(sampled_node_ids)
        state.clipping_range = self.clipping_range
        state.quantization_range = self.quantization_range
        state.mod_range = self.modulus_range
        state.max_weight = self.max_weight
        state.dp_epsilon = self.dp_epsilon
        state.dp_delta = self.dp_delta
        state.dp_clipping_c = self.dp_clipping_c
        sa_params_dict = {
            Key.STAGE: Stage.SETUP,
            Key.SAMPLE_NUMBER: num_samples,
            Key.SHARE_NUMBER: state.num_shares,
            Key.THRESHOLD: state.threshold,
            Key.CLIPPING_RANGE: state.clipping_range,
            Key.TARGET_RANGE: state.quantization_range,
            Key.MOD_RANGE: state.mod_range,
            Key.MAX_WEIGHT: state.max_weight,
            DP_EPSILON: state.dp_epsilon,
            DP_DELTA: state.dp_delta,
            DP_CLIPPING_C: state.dp_clipping_c,
        }

        if num_samples != state.num_shares and state.num_shares & 1 == 0:
            log(WARN, "Number of shares in the correlated-noise protocol should be odd.")
            state.num_shares += 1

        random.shuffle(sampled_node_ids)
        half_share = state.num_shares >> 1
        state.nid_to_neighbours = {
            nid: {sampled_node_ids[(idx + offset) % num_samples] for offset in range(-half_share, half_share + 1)}
            for idx, nid in enumerate(sampled_node_ids)
        }
        state.sampled_node_ids = state.active_node_ids

        cfg_record = ConfigRecord(sa_params_dict)  # type: ignore
        content = RecordDict({RECORD_KEY_CONFIGS: cfg_record})

        def make(nid: int) -> Message:
            return Message(
                content=content,
                dst_node_id=nid,
                message_type=MessageType.TRAIN,
                group_id=str(cfg[WorkflowKey.CURRENT_ROUND]),
            )

        log(DEBUG, "[Stage 0] Sending configurations to %s clients.", len(state.active_node_ids))
        msgs = grid.send_and_receive([make(node_id) for node_id in state.active_node_ids], timeout=self.timeout)
        state.active_node_ids = {msg.metadata.src_node_id for msg in msgs if not msg.has_error()}

        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            key_dict = msg.content.config_records[RECORD_KEY_CONFIGS]
            node_id = msg.metadata.src_node_id
            pk1, pk2 = key_dict[Key.PUBLIC_KEY_1], key_dict[Key.PUBLIC_KEY_2]
            state.nid_to_publickeys[node_id] = [cast(bytes, pk1), cast(bytes, pk2)]

        return self._check_threshold(state)

    def share_keys_stage(  # pylint: disable=too-many-locals
        self, grid: Grid, context: LegacyContext, state: WorkflowState
    ) -> bool:
        cfg = context.state.config_records[MAIN_CONFIGS_RECORD]

        def make(nid: int) -> Message:
            neighbours = state.nid_to_neighbours[nid] & state.active_node_ids
            cfg_record = ConfigRecord({str(n): state.nid_to_publickeys[n] for n in neighbours})
            cfg_record[Key.STAGE] = Stage.SHARE_KEYS
            content = RecordDict({RECORD_KEY_CONFIGS: cfg_record})
            return Message(
                content=content,
                dst_node_id=nid,
                message_type=MessageType.TRAIN,
                group_id=str(cfg[WorkflowKey.CURRENT_ROUND]),
            )

        log(DEBUG, "[Stage 1] Forwarding public keys to %s clients.", len(state.active_node_ids))
        msgs = grid.send_and_receive([make(node_id) for node_id in state.active_node_ids], timeout=self.timeout)
        state.active_node_ids = {msg.metadata.src_node_id for msg in msgs if not msg.has_error()}

        srcs: list[int] = []
        dsts: list[int] = []
        ciphertexts: list[bytes] = []
        fwd_ciphertexts: dict[int, list[bytes]] = {nid: [] for nid in state.active_node_ids}
        fwd_srcs: dict[int, list[int]] = {nid: [] for nid in state.active_node_ids}
        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            node_id = msg.metadata.src_node_id
            res_dict = msg.content.config_records[RECORD_KEY_CONFIGS]
            dst_lst = cast(list[int], res_dict[Key.DESTINATION_LIST])
            ctxt_lst = cast(list[bytes], res_dict[Key.CIPHERTEXT_LIST])
            srcs += [node_id] * len(dst_lst)
            dsts += dst_lst
            ciphertexts += ctxt_lst

        for src, dst, ciphertext in zip(srcs, dsts, ciphertexts, strict=True):
            if dst in fwd_ciphertexts:
                fwd_ciphertexts[dst].append(ciphertext)
                fwd_srcs[dst].append(src)

        state.forward_srcs = fwd_srcs
        state.forward_ciphertexts = fwd_ciphertexts
        return self._check_threshold(state)

    def collect_masked_vectors_stage(self, grid: Grid, context: LegacyContext, state: WorkflowState) -> bool:
        cfg = context.state.config_records[MAIN_CONFIGS_RECORD]

        def make(nid: int) -> Message:
            cfg_dict = {
                Key.STAGE: Stage.COLLECT_MASKED_VECTORS,
                Key.CIPHERTEXT_LIST: state.forward_ciphertexts[nid],
                Key.SOURCE_LIST: state.forward_srcs[nid],
            }
            cfg_record = ConfigRecord(cfg_dict)  # type: ignore
            content = state.nid_to_fitins[nid]
            content.config_records[RECORD_KEY_CONFIGS] = cfg_record
            return Message(
                content=content,
                dst_node_id=nid,
                message_type=MessageType.TRAIN,
                group_id=str(cfg[WorkflowKey.CURRENT_ROUND]),
            )

        log(DEBUG, "[Stage 2] Forwarding encrypted key shares to %s clients.", len(state.active_node_ids))
        msgs = grid.send_and_receive([make(node_id) for node_id in state.active_node_ids], timeout=self.timeout)
        state.active_node_ids = {msg.metadata.src_node_id for msg in msgs if not msg.has_error()}

        del state.forward_ciphertexts, state.forward_srcs, state.nid_to_fitins

        masked_vector = None
        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            res_dict = msg.content.config_records[RECORD_KEY_CONFIGS]
            bytes_list = cast(list[bytes], res_dict[Key.MASKED_PARAMETERS])
            client_masked_vec = [bytes_to_ndarray(b) for b in bytes_list]
            masked_vector = (
                client_masked_vec if masked_vector is None else parameters_addition(masked_vector, client_masked_vec)
            )
        if masked_vector is not None:
            masked_vector = parameters_mod(masked_vector, state.mod_range)
            state.aggregate_ndarrays = masked_vector

        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            fitres = compat.recorddict_to_fitres(msg.content, True)
            proxy = state.nid_to_proxies[msg.metadata.src_node_id]
            state.legacy_results.append((proxy, fitres))

        return self._check_threshold(state)

    def unmask_stage(  # pylint: disable=too-many-locals,too-many-branches
        self, grid: Grid, context: LegacyContext, state: WorkflowState
    ) -> bool:
        cfg = context.state.config_records[MAIN_CONFIGS_RECORD]
        current_round = cast(int, cfg[WorkflowKey.CURRENT_ROUND])

        active_nids = state.active_node_ids
        dead_nids = state.sampled_node_ids - active_nids

        def make(nid: int) -> Message:
            neighbours = state.nid_to_neighbours[nid]
            cfg_dict = {
                Key.STAGE: Stage.UNMASK,
                Key.ACTIVE_NODE_ID_LIST: list(neighbours & active_nids),
                Key.DEAD_NODE_ID_LIST: list(neighbours & dead_nids),
            }
            cfg_record = ConfigRecord(cfg_dict)  # type: ignore
            content = RecordDict({RECORD_KEY_CONFIGS: cfg_record})
            return Message(content=content, dst_node_id=nid, message_type=MessageType.TRAIN, group_id=str(current_round))

        log(DEBUG, "[Stage 3] Requesting key shares from %s clients to remove masks.", len(state.active_node_ids))
        msgs = grid.send_and_receive([make(node_id) for node_id in state.active_node_ids], timeout=self.timeout)
        state.active_node_ids = {msg.metadata.src_node_id for msg in msgs if not msg.has_error()}

        collected_shares_dict: dict[int, list[bytes]] = {nid: [] for nid in state.sampled_node_ids}
        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            res_dict = msg.content.config_records[RECORD_KEY_CONFIGS]
            nids = cast(list[int], res_dict[Key.NODE_ID_LIST])
            shares = cast(list[bytes], res_dict[Key.SHARE_LIST])
            for owner_nid, share in zip(nids, shares, strict=True):
                collected_shares_dict[owner_nid].append(share)

        # Must match _collect_masked_vectors's calibration exactly
        # -- otherwise a dropped client's residual pairwise term won't be regenerated correctly and the aggregate comes out wrong.
        edge_sigma = gaussian_edge_sigma(state.dp_epsilon, state.dp_delta, state.dp_clipping_c, state.num_shares, state.threshold)

        masked_vector = state.aggregate_ndarrays
        del state.aggregate_ndarrays
        for nid, share_list in collected_shares_dict.items():
            if len(share_list) < state.threshold:
                log(ERROR, "Not enough shares to recover secret in unmask vectors stage")
                return False
            secret = combine_shares(share_list)
            if nid in active_nids:
                # The seed for PRG is the private mask seed of an active client.
                private_mask = pseudo_rand_gen(secret, state.mod_range, get_parameters_shape(masked_vector))
                masked_vector = parameters_subtraction(masked_vector, private_mask)
            else:
                # The seed for PRG is the secret key 1 of a dropped client.
                neighbours = state.nid_to_neighbours[nid] - {nid}
                for neighbor_nid in neighbours:
                    shared_key = generate_shared_key(
                        bytes_to_private_key(secret), bytes_to_public_key(state.nid_to_publickeys[neighbor_nid][0])
                    )
                    edge_noise = sample_gaussian_edge(shared_key, edge_sigma, get_parameters_shape(masked_vector))
                    if nid > neighbor_nid:
                        masked_vector = parameters_addition(masked_vector, edge_noise)
                    else:
                        masked_vector = parameters_subtraction(masked_vector, edge_noise)

        recon_parameters = parameters_mod(masked_vector, state.mod_range)
        q_total_ratio, recon_parameters = factor_extract(recon_parameters)
        inv_dq_total_ratio = state.quantization_range / q_total_ratio
        aggregated_vector = dequantize(recon_parameters, state.clipping_range, state.quantization_range)
        offset = -(len(active_nids) - 1) * state.clipping_range
        for vec in aggregated_vector:
            vec += offset
            vec *= inv_dq_total_ratio

        results = state.legacy_results
        parameters = ndarrays_to_parameters(aggregated_vector)
        for _, fitres in results:
            fitres.parameters = parameters

        log(INFO, "aggregate_fit: received %s results and %s failures", len(results), len(state.failures))
        aggregated_result = context.strategy.aggregate_fit(current_round, results, state.failures)  # type: ignore
        parameters_aggregated, metrics_aggregated = aggregated_result

        if parameters_aggregated:
            arr_record = compat.parameters_to_arrayrecord(parameters_aggregated, True)
            context.state.array_records[MAIN_PARAMS_RECORD] = arr_record
            context.history.add_metrics_distributed_fit(server_round=current_round, metrics=metrics_aggregated)
        return True
