# Qwen3.5 image parity qualification

`tests/gpu/models/test_qwen3_5_images.py` judges greedy image chats on Qwen3.5-9B and
Qwen3.5-35B-A3B (BF16) by the FP32 Transformers truth, the rule the text checks use
(`tests/support/reference_truth.py`). The truth is teacher-forced on each engine arm's own
sequences, with HF's native FP32 pixels: the image chats and one image under several
questions (encoder-cache hits), each under the naive cache and under the radix cache serving
every batch twice (the second pass must hit), cached against uncached images under encoder
eviction, and a large image split across prefill chunks against it unsplit. Gates, as the text
prefix gate judges a cache hit: each cached, radix or split pass has at most two decided top-1
flips more than the plain pass (BF16 rounding), with finite teacher-forced top-1
log-probabilities (statistics in the receipt). The near-tie divergence rule stays for DFlash.

The 9B FP32 truth (about 36 GB) computes whole on an RTX PRO 6000; the 35B-A3B (about 140 GB)
offloaded on a host with enough CPU RAM. `CTI_REFERENCE_CACHE_DIR` keeps them, so
smaller GPUs read the cache.
