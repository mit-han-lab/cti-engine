# Transformers parity checks

Each test holds the engine on a pinned checkpoint of one model family against Transformers
through the public `LLM` API (`tests/support/hf_parity.py`): greedy agreement is report-only;
teacher-forced top-1 tokens and logprobs (one single-token request per prefix of the prompt and
the truth's continuation) gate parity against IEEE FP32 (`tests/support/hf_truth.py`),
calibrated by BF16 on the same terms with the bounds of `tests/support/hf_judge.py`. A MoE
checkpoint is judged cold and through prefix hits, each path excusing one prompt that fails only
on top-1 logprob error at a near tie of the truth.

The GPU tiers do not collect them: the FP32 truth of Qwen3.5-9B alone is 35 GiB of weights,
more than a 32 GB RTX 5090 holds, and streaming the rest from CPU RAM on every decode step
would outlast a GPU shard's 35 minutes. Run a family's test by hand on a GPU that holds the truth
whenever its model, its kernels or the parity prompts (`tests/gpu/models/parity_prompts.json`)
change. Each reads the checkpoint from the local Hugging Face cache and fails, rather than
skips, when the host has not provisioned it.

## Qwen3.5 and Qwen3.5-MoE: `test_qwen3_5_hf_parity.py`

```bash
# Qwen/Qwen3.5-9B (the default): about 6 minutes on an RTX PRO 6000. On a 32 GB RTX 5090
# its 35 GiB truth streams from CPU RAM, about 25 minutes.
CTI_QWEN35_RECEIPTS=/path/to/receipts python -m pytest -q tests/hf_parity/test_qwen3_5_hf_parity.py
# Qwen/Qwen3.5-0.8B, 2B or 4B on an RTX 5090: about 6 minutes each. They read Qwen3.5-9B's
# prompt ids, which their tokenizers and chat templates render too (checked first).
CTI_QWEN35_REPO=Qwen/Qwen3.5-4B CTI_QWEN35_RECEIPTS=/path/to/receipts \
    python -m pytest -q tests/hf_parity/test_qwen3_5_hf_parity.py
# Qwen/Qwen3.8-27B on one 96 GB GPU: hours, most of them the FP32 truth streaming its
# offloaded weights.
CTI_QWEN35_REPO=Qwen/Qwen3.8-27B CTI_QWEN35_MAX_RUNNING_REQ=128 \
    CTI_QWEN35_TRUTH_GPU_GIB=86 CTI_QWEN35_TRUTH_TIMEOUT=12600 \
    CTI_QWEN35_RECEIPTS=/path/to/receipts python -m pytest -q tests/hf_parity/test_qwen3_5_hf_parity.py
# On two: the FP32 truth's layers split between them, none offloaded; about 10 minutes on two
# RTX PRO 6000s. The BF16 calibration and the engine run on the first.
CUDA_VISIBLE_DEVICES=2,3 CTI_QWEN35_TRUTH_GPUS=2 \
    CTI_QWEN35_REPO=Qwen/Qwen3.8-27B CTI_QWEN35_MAX_RUNNING_REQ=128 \
    CTI_QWEN35_RECEIPTS=/path/to/receipts python -m pytest -q tests/hf_parity/test_qwen3_5_hf_parity.py
# Judge again against the truth a run kept (its checkpoint, prompts, greedy length and code):
# into new receipts, with the same settings as the run that computed it.
CTI_QWEN35_REUSE_TRUTH=/path/to/receipts/qwen3_5_hf_parity-Qwen--Qwen3.8-27B/truth \
    CTI_QWEN35_RECEIPTS=/path/to/new-receipts ... python -m pytest -q tests/hf_parity/test_qwen3_5_hf_parity.py
```

The `CTI_QWEN35_*` settings, read by `tests/support/qwen3_5_gpu.py` for this check and
the GPU model tests (`tests/gpu/models`):

| Variable | Unset | Meaning |
|---|---|---|
| `REPO`, `REVISION` | `Qwen/Qwen3.5-9B` at its pin | a pinned checkpoint (`PINNED_CHECKPOINTS`); another revision is refused |
| `MAX_RUNNING_REQ` | 16 | request slots, the recurrent-state pool's rows |
| `GREEDY_TOKENS` | 256 | tokens of each greedy continuation (no EOS stop) |
| `RESIDENCY_CYCLES` | 10 | suspend/resume cycles of the residency test |
| `RECEIPTS` | none | directory where each test writes its measurements before it asserts |
| `TRUTH_GPUS` | 1 | GPUs the FP32 truth's weights spread over |
| `TRUTH_GPU_GIB` | free memory less a headroom | GiB of truth weights each GPU keeps; the rest streams from CPU RAM |
| `TRUTH_TIMEOUT` | 7200 | seconds the FP32 truth child may run |
| `REUSE_TRUTH` | none | a kept `truth/` folder to judge against; one that ran anything else is refused |
| `TP_SIZE` | 1 | engine ranks, one per visible GPU from the first (a MoE model's experts split over them) |
| `EXPERT_PARALLEL` | 1 | `0`: a MoE model's routed experts tensor-parallel (every expert cut by width), not whole per rank |

Each child's output and the receipt stay under
`CTI_QWEN35_RECEIPTS`, pass or fail. The receipt is rewritten before each child, with
the finished children's manifests and the running child's name, so an engine failure after an
hours-long truth keeps the truth for `CTI_QWEN35_REUSE_TRUTH`. Each child's log
streams, while it runs, to `logs/qwen3_5_hf_parity/` under the test's `tmp_path` (pytest's
`--basetemp`).

## Qwen3: `test_qwen3_hf_parity.py`

`test_qwen3_matches_transformers` judges each pinned dense Qwen3 size cold, one case per size
and engine width: `0.6b-tp1`, `1.7b-tp1`, `4b-tp1` and `8b-tp1` on an RTX 5090; `8b-tp1` and
`8b-tp2` on RTX PRO 6000s, the truth on the first. A case skips below its GPU count. Each reads
Qwen3-30B-A3B's prompt ids, refused unless its tokenizer and chat template render every prompt to them.
About 4 minutes for 0.6B and 1.7B and 5 for 4B on an RTX 5090, 18 for 8B there
(its 30 GiB truth partly streams from CPU RAM); 5.5 for 8B at each width on RTX PRO 6000s.
Settings: `CTI_QWEN3_*`, as Qwen3-MoE's below but for `REPO`, `REVISION`, `TP_SIZE` and
`TRUTH_GPUS`, which each case sets.

```bash
python -m pytest -q 'tests/hf_parity/test_qwen3_hf_parity.py::test_qwen3_matches_transformers[8b-tp2]'
```

## Qwen3-MoE: `test_qwen3_moe_hf_parity.py`

`test_qwen3_moe_matches_transformers` judges `Qwen/Qwen3-30B-A3B` (the default, at its pin)
cold and through prefix hits.
Its FP32 truth is 114 GiB: two RTX PRO 6000s hold it whole, one streams part of it from CPU
RAM. `test_the_engine_renders_each_chat_template_as_transformers` needs only the tokenizer
files of the five published 30B-A3B checkpoints.

```bash
# On two RTX PRO 6000s: the truth spread over both; calibration and engine on the first.
CTI_QWEN3_MOE_TRUTH_GPUS=2 CTI_QWEN3_MOE_RECEIPTS=/path/to/receipts \
    python -m pytest -q tests/hf_parity/test_qwen3_moe_hf_parity.py
# Every engine path TP2/EP2 over both, judged against a kept truth.
CTI_QWEN3_MOE_TP_SIZE=2 \
    CTI_QWEN3_MOE_REUSE_TRUTH=/path/to/receipts/qwen3_moe_hf_parity-Qwen--Qwen3-30B-A3B/truth \
    CTI_QWEN3_MOE_RECEIPTS=/path/to/tp2-receipts python -m pytest -q tests/hf_parity/test_qwen3_moe_hf_parity.py
```

The `CTI_QWEN3_MOE_*` settings are the Qwen3.5 table's (`REPO`, `REVISION`,
`MAX_RUNNING_REQ`, `GREEDY_TOKENS`, `RECEIPTS`, `TRUTH_GPUS`,
`TRUTH_GPU_GIB`, `TRUTH_TIMEOUT`, `REUSE_TRUTH`, `TP_SIZE`, `EXPERT_PARALLEL`), read by `ParitySettings` in
`tests/support/hf_parity.py`.

## Quantized checkpoints: `test_quant_gates.py`

The GPU gates on four pinned quantized checkpoints, one test per gate and case, so that
`-k` selects them. A case is a checkpoint and the engine's topology:

| Case | Checkpoint | BF16 sibling | Devices |
|---|---|---|---|
| `qwen3-8b-fp8-tp1` | `Qwen/Qwen3-8B-FP8@220b46e3b2180893580a4454f21f22d3ebb187d3` | `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` | RTX 5090 |
| `qwen3.8-27b-nvfp4-tp1` | `nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80` | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | RTX 5090, RTX PRO 6000 |
| `qwen3.8-27b-fp8-tp1` | `Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` | `Qwen/Qwen3.8-27B@1d4bf0f2…` | RTX PRO 6000 |
| `qwen3.5-35b-a3b-fp8-ep1` | `Qwen/Qwen3.5-35B-A3B-FP8@9d1823d2dee688a6b25e77009dc727688c44936e` | `Qwen/Qwen3.5-35B-A3B@59d61f3ce65a6d9863b86d2e96597125219dc754` | RTX PRO 6000 |
| `qwen3.5-35b-a3b-fp8-ep2` | the same, experts whole over two ranks | the same | two RTX PRO 6000 |
| `qwen3.5-35b-a3b-fp8-tp2` | the same, every expert cut by width over two ranks | the same | two RTX PRO 6000 |

Every test is one segment: a GPU call of at most 30 minutes, selected by `-k` on its test name
and case. The gates and their segments:

1. `test_graph_equals_eager[<case>-b<B>]`: greedy decoding of the first B parity prompts with
   decode graphs and eagerly, B = 1..4. B = 1: the same tokens with bitwise-equal chosen-token
   logprobs. B = 2..4: within the accuracy tolerance (the same tokens up to a divergence at a
   near tie, `NEAR_TIE_NATS`; logprobs within `GRAPH_LOGPROB_TOLERANCE` of `hf_judge.py`), with
   bitwise equality recorded as pending a cti-kernels fix (attention dispatch differs between
   graph and eager at B>1). One child decodes all four B per case; the other B reuse its
   `child.json`.
2. Against the FP32 truths, teacher-forced through the public API (the engine child of
   `tests/support/hf_parity.py`), in four segments:
   - `test_dequantize_checkpoint[<checkpoint>]` (CPU): the checkpoint dequantized to FP32 by
     `cti/layers/quant/numerics.py`, with each quantized module's activation format.
   - `test_generate_truth[<checkpoint>-quantized]`: the FP32 quantized truth, Transformers in
     IEEE FP32 on that folder with every quantized module quantizing its input as the engine
     does (`tests/support/quant_truth.py`). `test_generate_truth[<checkpoint>-floor]`, after it:
     the same in BF16 on one GPU, forced along that truth (G-C's floor), kept beside it; with
     `QUANT_GATE_DROP_DEQUANT=1` it then deletes the dequantized folder (the truths' checks
     derive its config from the checkpoint). `test_generate_truth[<checkpoint>-bf16]`: the BF16 sibling's FP32
     truth (`tests/support/hf_truth.py`). Each is skipped when one is in place.
   - `test_matches_quantized_fp32_truth[<case>]` (G-C, gated): the engine within the bounds of
     `hf_judge.py` calibrated by the quantized truth in BF16 (the floor), under the dense
     parity rules or the MoE ones (cold and prefix hits, one near-tie prompt excused per path).
   - `test_reports_bf16_fp32_truth[<case>]` (G-D, recorded): top-1 agreement with the sibling's
     truth and the chosen-token logprob error where the top-1 agrees; its bounds are not yet frozen.
3. `test_latency[<case>-quantized|bf16]`: `benchmark.inference.offline`, 512 prompt and 128
   output tokens at B = 1..4, TTFT and TPOT medians; the second role recorded also records
   the speedup BF16 / quantized. Not gated. A sibling whose weights take more than 90% of the
   GPUs is recorded as not fitting. The benchmark places routed experts as the case does
   (`--expert-parallel` / `--no-expert-parallel`).
4. `test_switching_keeps_quantized_tensors[<case>]`: a LoRA training engine serves, enters and
   leaves three training sessions that discard their step, and serves again. On every rank,
   every quantized tensor keeps its `data_ptr` and bytes at each switch, and every serving phase
   repeats the first outputs bit for bit. The ranks run `tests/support/quant_gate_child.py` as
   their spawned main module, which probes `GPUWorker.release` / `acquire`.
5. `test_lora_on_quantized_base[<case>]`: a zero-initialized adapter serves the base engine's
   outputs bit for bit; one committed LoRA step has a finite loss and a positive gradient norm
   and changes the outputs. Gates 4 and 5 train, so they run only the cases whose recipe trains
   (block FP8); a ModelOpt checkpoint serves only.

Each test writes its result to `$QUANT_GATE_OUT/<gate>/<case>.json` before it asserts, and its
children's output under `$QUANT_GATE_OUT/<gate>/<case>/`; a segment refuses a folder an earlier
run left, so rerun one in a new `QUANT_GATE_OUT` or with its folder removed. Every child reads
the BF16 sibling's parity prompts: the frozen file where it holds the sibling's ids (Qwen3.8-27B,
Qwen3.5-35B-A3B), so a truth a parity check kept is reused as it is; otherwise (Qwen3-8B) the
frozen prompts tokenized under the sibling's key into `$QUANT_GATE_OUT/prompts`. The quantized
checkpoint's tokenizer must re-derive them.

The truths are read from `QUANT_GATE_REUSE_QUANT_TRUTH` / `QUANT_GATE_REUSE_BF16_TRUTH` when
set, else from `$QUANT_GATE_OUT/truths/<repo>/{quantized,bf16}`, where the generation segment
writes them; keep one `QUANT_GATE_OUT` (or name the kept truths) across a checkpoint's
segments. A kept truth must be the one this run would compute, checked as the parity checks'
`REUSE_TRUTH` is (`hf_truth.check_truth`: model key, config, prompts, greedy length, top-k,
FP32, code version, prompt names); the BF16 truth's identity is the sibling's own config, the
quantized truth's the dequantized config and its `quant_truth.json` (source checkpoint,
numerics contract, hooks). A Qwen3.5-35B-A3B truth from `test_qwen3_5_hf_parity.py`
(`<receipts>/qwen3_5_hf_parity-Qwen--Qwen3.5-35B-A3B/truth`) serves as its BF16 truth.

| `QUANT_GATE_*` | Unset | Meaning |
|---|---|---|
| `OUT` | required | results, children's output, generated truths |
| `DEQUANT_DIR` | `$QUANT_GATE_OUT/dequantized` | the dequantized FP32 checkpoints (8B: 33 GB, 27B: 108 GB, 35B: 140 GB), rebuilt only when the checkpoint, the numerics contract or the builder changes |
| `GREEDY_TOKENS` | 256 | tokens of each truth's greedy continuation |
| `MAX_RUNNING_REQ` | 16 | request slots of the teacher-forced engine |
| `TRUTH_GPUS`, `TRUTH_GPU_GIB` | 1, free less headroom | GPUs the FP32 truth spreads over, and GiB of weights each keeps |
| `TRUTH_TIMEOUT` | 1700 | seconds a truth child may run |
| `REUSE_QUANT_TRUTH`, `REUSE_BF16_TRUTH` | none | a kept truth folder (the floor is read from `floor/` beside the quantized one) |
| `DROP_DEQUANT` | 0 | `1`: the floor segment deletes the dequantized folder once the floor is written |
| `PERF_RUNS` | 3 | benchmark runs per role |
| `ALLOW_DIRTY` | 0 | `1`: benchmark an uncommitted tree (never a formal number) |

Order. RTX 5090, one gate of one case per call: `qwen3-8b-fp8-tp1` and
`qwen3.8-27b-nvfp4-tp1`, each through `graph_equals_eager`, `switching`, `lora_on_quantized`,
`latency` (the NVFP4 checkpoint serves only: the two training gates skip it). RTX PRO 6000, one
GPU: `qwen3.8-27b-nvfp4-tp1`, `qwen3.8-27b-fp8-tp1`, `qwen3.5-35b-a3b-fp8-ep1`, each through
those gates and then G-C / G-D once its truths exist. Truths, on an RTX PRO 6000 pair, per
checkpoint: `dequantize_checkpoint`, `generate_truth` `-quantized` (`QUANT_GATE_TRUTH_GPUS=2`;
FP32 27B takes 108 GB over the pair, 35B 140 GB), then `-floor` with `QUANT_GATE_DROP_DEQUANT=1`,
so the dequantized folder (33/108/140 GB) lives only between those segments; then `-bf16` where
the sibling's truth is not kept. The truth folders are small; copy them to the machine that runs
G-C / G-D and name them with `REUSE_*`. RTX PRO 6000 pair: `qwen3.5-35b-a3b-fp8-ep2` and `-tp2`
through every gate. No gate runs on an RTX 5090 pair.

```bash
# One GPU (RTX 5090 or RTX PRO 6000): one gate of one case per call.
CUDA_VISIBLE_DEVICES=0 QUANT_GATE_OUT=<out> python -m pytest -q \
    tests/hf_parity/test_quant_gates.py -k "graph_equals_eager and qwen3-8b-fp8-tp1"
CUDA_VISIBLE_DEVICES=0 QUANT_GATE_OUT=<out> python -m pytest -q \
    tests/hf_parity/test_quant_gates.py -k "latency and qwen3-8b-fp8-tp1-bf16"
CUDA_VISIBLE_DEVICES=0 QUANT_GATE_OUT=<out> python -m pytest -q \
    tests/hf_parity/test_quant_gates.py -k "switching and qwen3.8-27b-fp8-tp1"
# RTX PRO 6000 pair: truths, then the gates that compare against them.
CUDA_VISIBLE_DEVICES=0,1 QUANT_GATE_OUT=<out> QUANT_GATE_TRUTH_GPUS=2 python -m pytest -q \
    tests/hf_parity/test_quant_gates.py -k "generate_truth and qwen3.8-27b-fp8-bf16"
CUDA_VISIBLE_DEVICES=0 QUANT_GATE_OUT=<out> QUANT_GATE_DROP_DEQUANT=1 python -m pytest -q \
    tests/hf_parity/test_quant_gates.py -k "generate_truth and qwen3.8-27b-fp8-floor"
CUDA_VISIBLE_DEVICES=0 QUANT_GATE_OUT=<out> python -m pytest -q \
    tests/hf_parity/test_quant_gates.py -k "matches_quantized_fp32_truth and qwen3.8-27b-fp8-tp1"
CUDA_VISIBLE_DEVICES=0,1 QUANT_GATE_OUT=<out> \
    QUANT_GATE_REUSE_BF16_TRUTH=<receipts>/qwen3_5_hf_parity-Qwen--Qwen3.5-35B-A3B/truth \
    python -m pytest -q tests/hf_parity/test_quant_gates.py \
    -k "reports_bf16_fp32_truth and qwen3.5-35b-a3b-fp8-ep2"
```
