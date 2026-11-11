"""The GPU tiers. Every GPU test is listed here once. A test outside the tensor-parallel
modules is in FAST_TIER, which runs in two shards, or in a single-GPU shard of FULL_TIER, which
the full tier adds; a test of those modules is in TENSOR_PARALLEL_TIER, which both tiers run in
the two tp shards, or in FULL_TIER["tp"], which the full tier adds as the full-tp shard on the
RTX PRO 6000 pair. Listed tests are marked
``gpu_fast_<shard>``, ``gpu_tp_<shard>``, or ``gpu_full`` and ``gpu_full_<shard>``."""

import gc
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from tests.support.checkpoints import checkpoints_of
from tests.support.gpu import QUALIFICATION_VARIABLE

_ROOT = Path(__file__).parent

# A module, test function or test ID under tests/gpu/ -> its tests' seconds on an RTX 5090 with
# kept JIT caches; fast_shards() splits these between the fast-1 and fast-2 shards.
FAST_TIER = {
    "inference/correctness/test_contract.py::test_qwen_fixed_snapshot_correctness_contract": 45,
    "inference/test_breakable_graph.py::test_qwen_breakable_graph_replays_match_the_padded_forward": 30,
    "inference/test_breakable_graph.py::test_qwen3_5_breakable_graph_replays_match_the_padded_forward": 17,
    "inference/test_cache_invalidation.py": 11,
    "inference/test_cache_reservation.py": 7,
    "inference/test_dflash_draft_graph.py": 60,
    "inference/test_dflash_fold.py::test_prompts_fold_into_rounds_and_follow_their_chains[captured-serial]": 43,
    "inference/test_dflash_lora.py::test_dflash_with_a_lora_target_follows_ordinary_decoding[Qwen/Qwen3.5-9B]": 159,
    "inference/test_dflash_parity.py::test_dflash_default_tier_matches_transformers[Qwen/Qwen3.5-9B]": 92,
    "inference/test_dflash_rounds.py::test_dflash_rounds_follow_the_scripted_acceptance[captured]": 37,
    "inference/test_dflash_rounds.py::test_dflash_rounds_follow_the_scripted_acceptance[gapped]": 37,
    "inference/test_dflash_sampling.py::test_a_check_fails_only_when_its_confirmation_fails_too": 1,
    "inference/test_dflash_sampling.py::test_pooled_statistics_keep_every_sample": 1,
    "inference/test_first_request.py::test_the_first_requests_compile_nothing[Qwen3-8B]": 84,
    "inference/test_first_request.py::test_the_first_requests_compile_nothing[Qwen3.5-9B-DFlash]": 188,
    "inference/test_flashinfer_workspace.py": 2,
    "inference/test_gated_delta_net_decode_graphs.py": 7,
    "inference/test_graph_runner_and_sampling_stack.py": 2,
    "inference/test_online_mixed_serving.py": 12,
    "inference/test_piecewise_cuda_graph.py": 28,
    "inference/test_radix_key_compare.py": 1,
    "inference/test_request_local_sampling.py": 1,
    "inference/test_scheduler_resource_retirement.py": 10,
    "integration/test_environment_stack.py": 13,
    "integration/test_grpo_cycle.py": 83,
    "integration/test_lora.py": 9,
    "integration/test_qwen3_5_lora.py": 114,
    "integration/test_runtime_lifecycle.py::test_qwen_flashinfer_graph_runtime_lifecycle": 26,
    "integration/test_runtime_lifecycle.py::test_qwen_invalid_target_is_rejected_at_submit": 8,
    "integration/test_runtime_lifecycle.py::test_qwen_online_request_is_serviced_at_idle_training_boundary": 8,
    "integration/test_yaml_config.py": 17,
    "layers/test_flashinfer_head_dim_256.py": 11,
    # Estimated: 53 timed, plus two child processes resolving and building prefills.
    "layers/test_gated_delta_net.py": 73,
    "layers/test_layer_operations.py": 1,
    "layers/test_mixed_query_attention.py": 23,
    "layers/test_vision_layers.py": 1,
    "models/test_qwen3_5_decode_graph.py": 225,
    "models/test_qwen3_5_prefix_memory.py::test_prefix_cache_real_memory_budget[naive]": 21,
    "models/test_qwen3_5_prefix_memory.py::test_prefix_cache_real_memory_budget[rows]": 37,
    "models/test_qwen3_5_prefix_reuse.py::test_cross_layer_copy_is_bitwise_and_bounds_temporary_memory": 1,
    "models/test_qwen3_5_prefix_reuse.py::test_decode_export_copies_the_replayed_rows_bitwise_and_nothing_else": 1,
    "models/test_qwen3_5_residency.py": 70,
    "switching/test_memory_residency.py": 19,
    "switching/test_native_backing.py": 1,
    "switching/test_qwen3_5_prefix_cache_switch.py": 53,
    "switching/test_recurrent_rows_resume.py": 4,
    "switching/test_qwen3_5_restore_rejection.py": 48,
    "switching/test_qwen3_5_training_residency.py::test_qwen3_5_training_switches_release_and_restore_the_inference_state[none-offload_optimizer]": 94,
    "switching/test_residency_restore.py": 2,
    "switching/test_training_memory_plan.py::test_qwen3_5_training_stays_within_the_logged_plan[none]": 52,
    "switching/test_training_memory_plan.py::test_qwen3_5_training_stays_within_the_logged_plan[packed]": 47,
    "switching/test_training_memory_pools.py": 3,
    "training/lora_backward_hf/test_contract.py": 13,
    "training/test_activation_checkpointing.py": 41,
    "training/test_grpo_layout_parity.py": 6,
    "training/test_lora_training_trajectory.py": 7,
    "training/test_mixed_precision_optimizer.py": 2,
    "training/test_packed_sft.py::test_packed_sft_matches_padded_qwen3": 8,
    "training/test_sft_staging.py": 1,
    "training/test_training_state_residency.py": 1,
}

# A tensor-parallel module, test function or test ID under tests/gpu/ -> its tests' seconds on
# the RTX PRO 6000 pair with cold JIT caches; tp_shards() splits these between the tp-1 and tp-2
# shards. Tests that share a module-scoped fixture are one entry, so no two shards build it.
TENSOR_PARALLEL_TIER = {
    "inference/test_dflash_prefix_tensor_parallel.py::test_tp2_dflash_radix_dflash_naive_and_radix_follow_the_fp32_truth[Qwen/Qwen3-8B]": 156,
    "inference/test_dflash_prefix_tensor_parallel.py::test_tp2_dflash_radix_dflash_naive_and_radix_follow_the_fp32_truth[Qwen/Qwen3.5-9B]": 338,
    "inference/test_dflash_tensor_parallel.py::test_tensor_parallel_rounds_follow_the_single_gpu[Qwen/Qwen3-8B]": 166,
    "inference/test_dflash_tensor_parallel.py::test_tensor_parallel_rounds_follow_the_single_gpu[Qwen/Qwen3.5-9B]": 233,
    # Timed on the RTX PRO 6000 pair; the TP4 cases skip on a pair.
    "inference/test_tensor_parallel.py::test_compact_bounded_tp_matches_across_ranks_in_eager_and_graph_modes[qwen3]": 0,
    "inference/test_tensor_parallel.py::test_compact_bounded_tp_matches_across_ranks_in_eager_and_graph_modes[qwen3_kv1]": 124,
    "inference/test_tensor_parallel.py::test_compact_bounded_tp_matches_across_ranks_in_eager_and_graph_modes[qwen3_5]": 43,
    "inference/test_tensor_parallel.py::test_compact_qwen3_5_lora_training_tp2_matches_one_rank_and_resumes_on_one_rank": 51,
    "inference/test_tensor_parallel.py::test_compact_online_tp_survives_idle_and_orders_concurrent_and_cancelled_requests[qwen3-2]": 81,
    "inference/test_tensor_parallel.py::test_compact_online_tp_survives_idle_and_orders_concurrent_and_cancelled_requests[qwen3-4]": 0,
    "inference/test_tensor_parallel.py::test_compact_online_tp_survives_idle_and_orders_concurrent_and_cancelled_requests[qwen3_5-2]": 83,
    "inference/test_tensor_parallel.py::test_compact_qwen3_5_tp2_cached_images_match_across_ranks": 13,
    "inference/test_tensor_parallel.py::test_compact_radix_reuses_prefixes_alike_on_one_and_two_ranks": 201,
    "inference/test_tensor_parallel.py::test_compact_tp2_auto_capacity_publishes_the_minimum_across_ranks": 24,
    "inference/test_tensor_parallel.py::test_compact_tp2_rank_death_fails_the_runtime_and_reclaims_every_rank": 10,
    "inference/test_tensor_parallel.py::test_compact_tp2_hung_rank_fails_the_runtime_within_the_timeout[rank-freeze]": 36,
    "inference/test_tensor_parallel.py::test_compact_tp2_hung_rank_fails_the_runtime_within_the_timeout[rank-stall]": 32,
    # Its prefix tests share each model's TP1 and TP2 runs.
    "inference/test_tensor_parallel_real_models.py::test_real_model_tp2_hits_reach_the_depths_of_one_rank[qwen3]": 322,
    "inference/test_tensor_parallel_real_models.py::test_real_model_hits_follow_the_fp32_truth_against_cold_runs[qwen3-1]": 14,
    "inference/test_tensor_parallel_real_models.py::test_real_model_hits_follow_the_fp32_truth_against_cold_runs[qwen3-2]": 230,
    "inference/test_tensor_parallel_real_models.py::test_real_model_tp2_hits_follow_the_near_tie_rule_against_one_rank[qwen3]": 1,
    "inference/test_tensor_parallel_real_models.py::test_real_model_tp_follows_the_near_tie_rule[qwen3-tp2]": 103,
    "inference/test_tensor_parallel_real_models.py::test_real_model_tp_follows_the_near_tie_rule[qwen3-tp4]": 0,
    "inference/test_tensor_parallel_real_models.py::test_real_model_tp_follows_the_near_tie_rule[qwen3_5-tp2]": 304,
    "inference/test_tensor_parallel_real_models.py::test_real_model_tp_follows_the_near_tie_rule[qwen3_5-images-tp2]": 152,
}

# Shard -> module, test function or test ID under tests/gpu/ -> why it waits for the full tier.
# Shards 1 and 2 run single-GPU tests, each within about 85% of a 35-minute budget; shard "tp"
# runs tensor-parallel tests on the pair, within about 85% of a 30-minute budget.
FULL_TIER: dict[int | str, dict[str, str]] = {
    1: {
        "inference/correctness/test_contract.py::test_qwen_fixed_snapshot_long_correctness_contract": "long context: 8K-24K-token prompts",
        "inference/test_dflash_fold.py::test_prompts_fold_into_rounds_and_follow_their_chains[captured-overlapped]": "the overlapped loop of the captured serial case, which stays fast",
        "inference/test_dflash_fold.py::test_prompts_fold_into_rounds_and_follow_their_chains[none-serial]": "a graph mode of the captured serial case, which stays fast",
        "inference/test_dflash_fold_truth.py::test_folded_prompts_follow_the_fp32_truth[Qwen/Qwen3-8B]": "forced folds against the FP32 truth",
        "inference/test_dflash_prefix.py::test_dflash_radix_dflash_naive_and_radix_follow_the_fp32_truth[Qwen/Qwen3-8B-1]": "DFlash with the radix cache against A/A-calibrated runs; the file runs about 2250 s on an RTX 5090, so its parameters split over the two shards (about 580 s each)",
        "inference/test_dflash_prefix.py::test_dflash_radix_dflash_naive_and_radix_follow_the_fp32_truth[Qwen/Qwen3.5-9B-16]": "DFlash with the radix cache against A/A-calibrated runs; the file runs about 2250 s on an RTX 5090, so its parameters split over the two shards (about 580 s each)",
        "inference/test_dflash_moe_combinations.py::test_dflash_moe_combination_follows_its_references[radix-prefix]": "35B on an RTX PRO 6000 only; a 5090 skips it at once",
        "inference/test_dflash_rounds.py::test_dflash_rounds_follow_the_scripted_acceptance[largest]": "a graph mode of the captured case, which stays fast",
        "inference/test_dflash_rounds.py::test_dflash_rounds_follow_the_scripted_acceptance[none]": "a graph mode of the captured case, which stays fast",
        "inference/test_first_request.py::test_the_first_requests_compile_nothing[Qwen3.5-9B]": "the second first-request check; Qwen3-8B stays fast as the cold start",
        "inference/test_first_request.py::test_the_first_requests_compile_nothing[Qwen3-8B-LoRA]": "the LoRA first-request check: an adapted projection's inference plan warms in prepare_inference",
        "switching/test_training_memory_plan.py::test_qwen3_5_training_stays_within_the_logged_plan[full]": "a checkpointing variant; none stays fast",
        "switching/test_qwen3_5_prefix_restore_after_switch.py": "prefix restores after a switch end to end, about 250 s; the row copies' resume test stays fast",
        "models/test_qwen3_5_default_options.py": "auto-sized running requests on a real budget; the CPU capacity tests stay fast",
        "models/test_qwen3_5_text_only.py": "a text-only load against the default one; the CPU text-only build test stays fast",
        "training/test_packed_sft.py::test_public_training_session_accepts_packed_sft": "the public session's native LoRA build",
        "training/test_padded_sft.py": "HF gradient parity in BF16 and FP32",
        "training/test_qwen3_5_activation_checkpointing.py": "Qwen3.5 checkpointing bit parity",
        "layers/test_moe_block.py": "MoE layers at Qwen3.5-35B-A3B's widths, not yet timed",
        "layers/test_fp8_block.py": "block-FP8 linear on its kernel; the quant gates judge models",
        "layers/test_fp8_experts.py": "block-FP8 routed experts on their kernel",
        "layers/test_modelopt_formats.py": "ModelOpt static FP8 and NVFP4 on their kernels",
        "layers/test_moe_expert_map.py": "MoE layers, not yet timed",
        "layers/test_moe_expert_lora.py": "per-expert LoRA layers, not yet timed",
        "layers/test_moe_training.py": "MoE training through the fused ops, not yet timed",
        "layers/test_moe_padded_rows.py": "MoE blocks over padded row buckets, not yet timed",
        "inference/test_breakable_graph.py::"
        "test_moe_prompt_steps_replay_their_sparse_blocks_inside_the_breakable_graph": (
            "MoE checkpoints; need an RTX PRO 6000"
        ),
        "integration/test_moe_lora_models.py": "MoE LoRA at 35B; needs an RTX PRO 6000",
        "models/test_qwen3_5_images.py::test_qwen3_5_image_prompts_follow_the_fp32_truth[Qwen/Qwen3.5-35B-A3B]": "image prompts on Qwen3.5-35B-A3B; needs an RTX PRO 6000",
        "models/test_qwen3_5_images.py::test_qwen3_5_cached_images_match_uncached_ones_under_eviction[Qwen/Qwen3.5-35B-A3B]": "cached image rows on Qwen3.5-35B-A3B; needs an RTX PRO 6000",
        "inference/test_dflash2_truth.py": "Qwen3.8-27B DFlash2 against the FP32 truth; needs an RTX PRO 6000",
        "inference/test_vision_graph.py": "opt-in vision graphs (EngineTuning.vision_graphs), not yet timed; their CPU packing tests stay fast",
    },
    2: {
        "inference/test_dflash_lora.py::test_dflash_with_a_lora_target_follows_ordinary_decoding[Qwen/Qwen3-8B]": "the second DFlash pair; Qwen3.5-9B stays fast",
        "inference/test_sampling_features.py::test_sampling_features_agree_with_and_without_dflash[Qwen/Qwen3-8B]": "the second DFlash pair; Qwen3.5-9B stays fast",
        "inference/test_sampling_features.py::test_sampling_features_agree_with_and_without_dflash[Qwen/Qwen3.5-9B]": "four engines per run; the fast tier holds DFlash x LoRA",
        "inference/test_dflash_prefix.py::test_dflash_radix_dflash_naive_and_radix_follow_the_fp32_truth[Qwen/Qwen3-8B-16]": "DFlash with the radix cache against A/A-calibrated runs; the file runs about 2250 s on an RTX 5090, so its parameters split over the two shards (about 580 s each)",
        "inference/test_dflash_prefix.py::test_dflash_radix_dflash_naive_and_radix_follow_the_fp32_truth[Qwen/Qwen3.5-9B-1]": "DFlash with the radix cache against A/A-calibrated runs; the file runs about 2250 s on an RTX 5090, so its parameters split over the two shards (about 580 s each)",
        "inference/test_dflash_async.py": "two rounds in flight; DFlash parity stays fast",
        "inference/test_dflash_draft_window.py": "long context: decoding past 2.5K-8K-token draft windows",
        "inference/test_dflash_fold_truth.py::test_folded_prompts_follow_the_fp32_truth[Qwen/Qwen3.5-9B]": "forced folds against the FP32 truth",
        "inference/test_dflash_parity.py::test_dflash_default_tier_matches_transformers[Qwen/Qwen3-8B]": "the second DFlash pair; Qwen3.5-9B stays fast",
        "inference/test_dflash_parity.py::test_greedy_dflash_matches_transformers_at_every_speculative_token_count[Qwen/Qwen3.5-9B]": "four engines, one per speculative-token count; the default count stays fast",
        "inference/test_dflash_sampling.py::test_sampled_dflash_is_lossless_and_seeded_rounds_repeat": "statistical: 1024 requests per setting",
        "inference/test_dflash_mixed_images.py": "image prompts (one, two, wide) mixed into a DFlash round; text prompts mix in the fast tier",
        "inference/test_dflash_moe_combinations.py::test_dflash_moe_combination_follows_its_references[expert-lora]": "35B on an RTX PRO 6000 only; a 5090 skips it at once",
        "inference/test_online_sampling_features.py": "the offline sampling features through a Qwen3-8B server",
        "inference/test_mixed_seed_min_p.py": "seeded and unseeded min-p rows in one step; the CPU sampler test stays fast",
        "inference/test_online_seeded_repeat.py": "a seeded call repeated through two Qwen3-8B servers",
        "inference/test_dflash_stream_chunks.py": "a Qwen3-8B DFlash server; the fast tier holds the stream contract on CPU",
        "integration/test_online_training_server.py": "two online engines and a latency comparison",
        "integration/test_runtime_lifecycle.py::test_qwen_online_training_policy_soak": "soak",
        "models/test_qwen3_5_images.py::test_qwen3_5_image_prompts_follow_the_fp32_truth[Qwen/Qwen3.5-9B]": "image prompts; Qwen3.5 text stays fast",
        "models/test_qwen3_5_images.py::test_qwen3_5_cached_images_match_uncached_ones_under_eviction[Qwen/Qwen3.5-9B]": "cached image rows; Qwen3.5 text stays fast",
        "models/test_qwen3_5_images.py::test_qwen3_5_an_image_split_across_chunks_matches_it_unsplit": "an image split across prefill chunks; Qwen3.5 text stays fast",
        "models/test_qwen3_5_images.py::test_qwen3_5_online_images_match_offline": "online image prompts; Qwen3.5 text stays fast",
        "models/test_qwen3_5_images.py::test_qwen3_5_dflash_image_prompts_follow_the_plain_engine": "image prompts under DFlash; Qwen3.5 text stays fast",
        "models/test_qwen3_5_prefix_memory.py::test_prefix_cache_real_memory_budget[ratio]": "the second sizing mode; rows stays fast",
        "models/test_qwen3_5_prefix_reuse.py::test_concurrent_prefix_capture_graph_and_eviction": "prefix pressure against A/A-calibrated runs",
        "models/test_qwen3_5_prefix_reuse.py::test_prefix_selected_logprobs_against_naive_and_aa": "prefix numerics against A/A-calibrated runs",
        "switching/test_qwen3_5_training_residency.py::test_qwen3_5_training_switches_release_and_restore_the_inference_state[full_offload-offload_optimizer]": "the offload variant; none stays fast",
        "switching/test_qwen3_5_training_residency.py::test_qwen3_5_training_switches_release_and_restore_the_inference_state[none-resident]": "the resident tier; offload_optimizer, the default, stays fast",
        "switching/test_qwen3_5_training_residency.py::test_qwen3_5_training_switches_release_and_restore_the_inference_state[none-offload_all]": "the offload_all tier; offload_optimizer, the default, stays fast",
        "switching/test_training_memory_plan.py::test_qwen3_5_training_stays_within_the_logged_plan[full_offload]": "a checkpointing variant; none stays fast",
        "integration/test_feature_matrix.py": "manual qualification only (CTI_MATRIX=1); too long for any shard",
        "integration/test_peft_adapter.py": "PEFT launch adapters on Qwen3-0.6B and Qwen3.5-9B against Transformers with PEFT; the fast shards are full",
        "training/test_training_memory_measured.py": "the training-memory plan against measured training memory, about 15 min on an RTX 5090",
    },
    "tp": {
        "inference/test_peft_adapter_tensor_parallel.py": "130 s on the pair (from TP1's 124 s): a PEFT launch adapter at TP2 against Transformers with PEFT; the tp shards are full",
        "inference/test_dflash_moe_expert_tp.py::test_dflash_at_etp2_follows_ep2": "610 s on the pair: 35B ETP2 against EP2 under DFlash; PRO-only",
        "inference/test_dflash_tensor_parallel.py::test_tensor_parallel_rounds_follow_the_single_gpu[Qwen/Qwen3.8-27B]": "488 s on the pair: DFlash2 TP2 at 27B, over the per-entry threshold",
        "inference/test_dflash_tensor_parallel.py::test_tensor_parallel_rounds_follow_the_single_gpu[Qwen/Qwen3.5-35B-A3B]": "418 s on the pair: DFlash TP2 at 35B-A3B; PRO-only, and the tp shards are full",
        "inference/test_tensor_parallel_real_models.py::test_real_model_tp2_hits_reach_the_depths_of_one_rank[qwen3_5]": "Qwen3.5 real-model prefix reuse, about 440 s on the pair (its runs are built once for the four)",
        "inference/test_tensor_parallel_real_models.py::test_real_model_hits_follow_the_fp32_truth_against_cold_runs[qwen3_5-1]": "Qwen3.5 real-model prefix reuse, about 440 s on the pair (its runs are built once for the four)",
        "inference/test_tensor_parallel_real_models.py::test_real_model_hits_follow_the_fp32_truth_against_cold_runs[qwen3_5-2]": "Qwen3.5 real-model prefix reuse, about 440 s on the pair (its runs are built once for the four)",
        "inference/test_tensor_parallel_real_models.py::test_real_model_tp2_hits_follow_the_near_tie_rule_against_one_rank[qwen3_5]": "Qwen3.5 real-model prefix reuse, about 440 s on the pair (its runs are built once for the four)",
    },
}


def lists(entry: str, test: str) -> bool:
    """Whether a tier entry (a module, a test function or one test ID) names ``test``."""
    return test == entry or test.startswith((f"{entry}::", f"{entry}["))


def _split(tier: dict[str, int]) -> dict[str, int]:
    """Each of ``tier``'s entries' shard, 1 or 2: longest first, each to the shard with less
    time so far (1 on a tie)."""
    loads = [0, 0]
    shards = {}
    for entry in sorted(tier, key=lambda entry: (-tier[entry], entry)):
        shard = loads.index(min(loads))
        loads[shard] += tier[entry]
        shards[entry] = shard + 1
    return shards


def fast_shards() -> dict[str, int]:
    """Each FAST_TIER entry's shard: fast-1 or fast-2."""
    return _split(FAST_TIER)


def tp_shards() -> dict[str, int]:
    """Each TENSOR_PARALLEL_TIER entry's shard: tp-1 or tp-2."""
    return _split(TENSOR_PARALLEL_TIER)


@pytest.fixture(scope="module", autouse=True)
def _release_cached_device_memory() -> Iterator[None]:
    """Hand a module's cached device memory back to the driver: an engine child a later
    module starts sizes its budget from the memory this process leaves free."""
    yield
    if torch.cuda.is_initialized():
        gc.collect()
        torch.cuda.empty_cache()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    timed = {"fast": fast_shards(), "tp": tp_shards()}
    for item in items:
        if not item.path.is_relative_to(_ROOT):
            continue
        test = f"{item.path.relative_to(_ROOT).as_posix()}::{item.nodeid.split('::', 1)[1]}"
        for tier, shards in timed.items():
            for entry, shard in shards.items():
                if lists(entry, test):
                    item.add_marker(getattr(pytest.mark, f"gpu_{tier}_{shard}"))
        for shard, entries in FULL_TIER.items():
            if any(lists(entry, test) for entry in entries):
                item.add_marker(pytest.mark.gpu_full)
                item.add_marker(getattr(pytest.mark, f"gpu_full_{shard}"))
    # A checkpoint the running qualification's devices do not hold is skipped, not failed.
    qualification = os.environ.get(QUALIFICATION_VARIABLE)
    if qualification:
        for item in items:
            misfits = sorted(
                f"{checkpoint.key} fits {', '.join(checkpoint.fits) or 'no qualification'}"
                for checkpoint in checkpoints_of(item)
                if qualification not in checkpoint.fits
            )
            if misfits:
                reason = f"not on {qualification}: {'; '.join(misfits)}"
                item.add_marker(pytest.mark.skip(reason=reason))
