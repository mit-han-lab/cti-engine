"""An image prompt that arrives while another request's DFlash rounds run mixes into the
next round on the pinned Qwen3.5-9B pair, and decodes as it does alone.

For each image chat (one image, two images, a wide image), one child serves an anchor text
request, then, once the anchor's rows run, the image chat prompt; the step that runs the image's placeholders must also run the anchor's round, through
the joint target forward. The same image prompt then runs alone on the same engine (no row to
mix with). Greedy, no prefix cache: the two runs may first differ only at a near tie of the
FP32 Transformers truth, the DFlash parity rule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.support.checkpoints import QWEN3_5_9B, QWEN3_5_9B_DFLASH, Checkpoint
from tests.support.dflash import PINNED_CHECKPOINTS, preflight
from tests.support.gpu import (
    assert_cuda_device_metadata,
    child_receipt,
    offline_child_env,
    run_isolated_process,
)
from tests.support.paths import REPO_ROOT
from tests.support.qwen3_5_gpu import receipts_dir

pytestmark = pytest.mark.checkpoints(QWEN3_5_9B, QWEN3_5_9B_DFLASH)

RECEIPT_PREFIX = "CTI_DFLASH_MIXED_IMAGES="
_ANCHOR, _MIXED, _ALONE = 0, 1, 2
_ANCHOR_TOKENS = 400
_IMAGE_TOKENS = 32


def _run_child(target: str, draft: str, chat: str) -> None:
    import torch
    from transformers import AutoTokenizer

    from cti.config import (
        CacheOptions,
        EngineOptions,
        EngineTuning,
        MemoryOptions,
        ModelOptions,
        SchedulingOptions,
        SpeculativeOptions,
    )
    from cti.inference._execution._speculative.dflash import SpeculativeRuntime
    from cti.protocol import SamplingParams, UserMsg
    from tests.support.dflash_designed import admit, engine_over
    from tests.support.gpu import start_rank
    from tests.support.qwen3_5_prompts import processed_image_prompts

    image = processed_image_prompts(target)[chat]
    tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
    anchor = tokenizer("A long story about a lighthouse keeper begins:")["input_ids"]
    worker = start_rank(
        EngineOptions(
            model=ModelOptions(path=target, max_seq_len=4096),
            memory=MemoryOptions(inference_ratio=0.8),
            scheduling=SchedulingOptions(max_running_requests=4, max_batch_tokens=2048),
            cache=CacheOptions(prefix="naive"),
            speculative=SpeculativeOptions(model_path=draft),
        ),
        EngineTuning(cuda_graph_bs=(1, 2, 3, 4)),
        checks=True,
    )
    engine, replies = engine_over(worker, overlap=True)
    scheduler = engine._scheduler

    settle_step = scheduler.settle_step
    steps: list[dict] = []

    def record(step, completion):
        if step.forward_rows:
            steps.append(
                {
                    "rows": [entry.request.uid for entry in step.round_rows],
                    "extensions": [
                        [entry.request.uid, entry.use.start, entry.use.end]
                        for entry in step.forward_rows
                    ],
                }
            )
        return settle_step(step, completion)

    scheduler.settle_step = record
    joint_forwards = 0
    extended_round = SpeculativeRuntime._extended_round

    def counted(self, *args):
        nonlocal joint_forwards
        joint_forwards += 1
        return extended_round(self, *args)

    SpeculativeRuntime._extended_round = counted

    def message(uid: int, input_ids, max_tokens: int, prompt=None) -> UserMsg:
        # The engine stamps seeds as it receives requests; this loop hands them over.
        params = SamplingParams(
            temperature=0, max_new_tokens=max_tokens, ignore_eos=True, seed=uid
        ).snapshot()
        if prompt is None:
            return UserMsg(uid, torch.tensor(input_ids, dtype=torch.int32), params)
        return UserMsg(
            uid, prompt.input_ids, params, images=prompt.images, rope_layout=prompt.rope_layout
        )

    def serve(first: UserMsg, later: UserMsg | None) -> None:
        admit(engine, replies, first)
        while engine.busy:
            engine.step()
            # Once the anchor's prompt is submitted its rows run: the next step's round.
            running = {slot.request.uid for slot in scheduler.slots.running}
            if later is not None and first.uid in running:
                admit(engine, replies, later)
                later = None
        scheduler.require_quiescent()

    serve(message(_ANCHOR, anchor, _ANCHOR_TOKENS), message(_MIXED, None, _IMAGE_TOKENS, image))
    mixed_joint_forwards = joint_forwards
    serve(message(_ALONE, None, _IMAGE_TOKENS, image), None)

    outputs: dict[int, list[int]] = {}
    for reply in replies:
        outputs.setdefault(reply.uid, []).extend(reply.token_ids)
    print(
        RECEIPT_PREFIX
        + json.dumps(
            {
                "outputs": {str(uid): tokens for uid, tokens in outputs.items()},
                "finished": sorted(reply.uid for reply in replies if reply.finished),
                "steps": steps,
                "image_spans": [[span.offset, span.end] for span in image.images.spans],
                "mixed_joint_forwards": mixed_joint_forwards,
                "alone_joint_forwards": joint_forwards - mixed_joint_forwards,
                "device": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
            }
        )
    )
    worker.close()


# two-images holds a known tie: step 6, tokens 22903 vs 1387, 0.226 nats apart in the FP32 truth.
@pytest.mark.parametrize("chat", ["one-image", "two-images", "wide-image"])
def test_an_image_prompt_mixed_into_a_round_decodes_as_alone(tmp_path: Path, chat: str) -> None:
    pair = PINNED_CHECKPOINTS[1]
    snapshots = preflight(pair)["snapshots"]
    env, log_root = offline_child_env(tmp_path, f"dflash-mixed-{chat}")
    output = run_isolated_process(
        (
            sys.executable,
            "-m",
            "tests.gpu.inference.test_dflash_mixed_images",
            snapshots["target"],
            snapshots["draft"],
            chat,
        ),
        log_root / "child.log",
        env=env,
        cwd=REPO_ROOT,
        timeout=1200,
    )

    receipt = child_receipt(output, RECEIPT_PREFIX)
    assert receipt["finished"] == [_ANCHOR, _MIXED, _ALONE]
    # The steps that ran the image's placeholders beside the anchor's round.
    mixed = [
        step
        for step in receipt["steps"]
        if _ANCHOR in step["rows"]
        and any(
            uid == _MIXED and start <= offset and end <= stop
            for uid, start, stop in step["extensions"]
            for offset, end in receipt["image_spans"]
        )
    ]
    assert mixed, receipt["steps"]
    assert receipt["mixed_joint_forwards"] >= len(mixed)
    # Alone, no row runs beside the image prompt: its prompt step is an ordinary forward.
    assert receipt["alone_joint_forwards"] == 0
    outputs = receipt["outputs"]
    assert len(outputs[str(_MIXED)]) == _IMAGE_TOKENS
    assert_cuda_device_metadata(receipt)
    mixed, alone = outputs[str(_MIXED)], outputs[str(_ALONE)]
    if mixed != alone:
        divergence = _judge(Path(snapshots["target"]), pair.target, chat, mixed, alone, tmp_path)
        assert divergence["allowed"], divergence


def _judge(
    target: Path,
    checkpoint: Checkpoint,
    chat: str,
    mixed: list[int],
    alone: list[int],
    tmp_path: Path,
):
    """Where the runs first differ, judged by the FP32 truth teacher-forced on the mixed run."""

    from tests.support.hf_truth import content_digest
    from tests.support.qwen3_5_hf_reference import prepare_image_inputs
    from tests.support.qwen3_5_prompts import MAX_IMAGE_TOKENS, image_chats, processed_image_prompts
    from tests.support.reference_truth import SequenceTruth, judge_divergence

    prompt = processed_image_prompts(target)[chat]
    inputs = prepare_image_inputs(
        target, image_chats()[chat], prompt, max_image_tokens=MAX_IMAGE_TOKENS
    )
    key = (tuple(prompt.input_ids.tolist()), content_digest(inputs["originals"]))
    job = (receipts_dir() or tmp_path) / f"qwen3_5-dflash-mixed-{chat}-truth-{checkpoint.slug}.pt"
    truth, _ = SequenceTruth(checkpoint, target, tmp_path).require(
        {(key, tuple(mixed))}, job, {key: inputs}
    )
    step = next(step for step, (x, y) in enumerate(zip(mixed, alone, strict=True)) if x != y)
    return judge_divergence(truth[key, tuple(mixed)], step, (mixed[step], alone[step]))


if __name__ == "__main__":
    _run_child(sys.argv[1], sys.argv[2], sys.argv[3])
