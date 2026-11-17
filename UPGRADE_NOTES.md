# SM12x NVFP4 upgrade — 2026-11-17

Original branch: `sm12x-nvfp4kv` at `63277ca386`.
Local rollback branch: `backup/sm12x-nvfp4kv-20261117`.
Upgrade base: upstream `main` at `935f87de84` (after v0.34.0).
The initial serving baseline carried no runtime changes relative to that
upstream commit. A separate local JIT-memory fix is described below.
AI assistance: Codex inspected the patches, rebased the archive, and performed
local validation. No upstream contribution is proposed.

## Patch disposition

| Local change | Current upstream replacement | Disposition |
| --- | --- | --- |
| 0001: SM12x FA2 NVFP4 routing, packed cache, overlay loader | `554340f3d3` / #46963 adds SM8x/SM12x support; `9bd074a996` / #60452 adds SM12x XQA decode | Drop local routing and overlay loader. |
| Extra pre-profile warmup in 0001 | `860da3dafd` / #56492 sizes extensible cache after warmup/capture | Drop duplicate profile call; verify cache capacity at runtime. |
| 0002: architecture-dependent linear V-scale writer | SM12x backend calls FlashInfer `nvfp4_quantize_append_paged_kv_cache_with_slot_mapping`; old CUDA file no longer exists upstream | Drop old CUDA writer and overlay requirement. |
| 0003: force HND layout in worker RPC | `_nvfp4_kv_views` detects HND/NHD strides and supplies the actual layout to both writer and reader | Drop forced HND; test all four supported layouts and long-context recall. |
| 0004: force prefill split-KV | FlashInfer 0.7.4 still disables NVFP4 split-KV on SM120/121 because of observed corruption | Not upstreamed. Retire override for baseline validation; do not claim equivalent performance. |
| 0005: materialize dense DFlash quantized QKV weights | Current DFlash runs packed projections through their quant method instead; post-load hooks rebuild derived buffers (#64357), with mixed-precision support in #63001 | Drop dense materialization and swallowed AttributeError. Current service uses MTP, so serving tests do not validate DFlash drafts. |
| `34f40991ac`: response_format with auto tool choice | `2b9b55c7f1` / #56086 | Already upstream; rebase skipped it. |

## Deployment baseline

Keep model `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` pinned to
`0cc27958cefbbe231782ec8511de8c4eb5233348`, the custom chat template, MTP
with three speculative tokens, 0.92 GPU utilization, eight sequences,
and 2048 batched tokens.

Remove obsolete `VLLM_SM12X_NVFP4KV_SO` and
`VLLM_SM12X_NVFP4_PREFILL_SPLIT_KV` environment variables from the upgraded
service. Keep `MAX_JOBS=2` and the existing memory controls.

## Memory sizing semantics

`gpu_memory_utilization` now defaults to None. With extensible KV cache it
resolves to 1.0; without extensible sizing it resolves to 0.92. An explicit
0.92 caps this instance at 92% of total device memory and does not disable
extensible sizing. The `--no-enable-extensible-kv-cache` flag restores the
old estimate-based sizing. The release note's suggestion of utilization 0.9
is an alternative smaller budget, not an equivalent feature toggle.

Measured sizing subtracts resident non-KV allocations, reserves the larger
of transient peak use and torch reserve growth, and leaves a physical-memory
margin of max(1 GiB, 0.75 * transient peak) on CUDA. Thus a 1.0 budget does
not mean allocating every last free byte. Cache physical pages are committed
after warmup; this is not unlimited growth during serving.

## Validation

Original service: 419,915 KV tokens; 1.60x concurrency at 262,144 tokens.
Before shutdown, deterministic arithmetic returned 391 and a 25,244-token
needle probe returned the exact secret. Full request/response artifacts are
outside the repository in `/home/anthony/projects/vllm-upgrade-validation/`.

At the original 0.92 utilization, unmodified upstream starts with 331,254
KV tokens (1.26x), using 139 blocks / 7.30 GiB of physical cache. The original
25,244-token recall case passes (2.77s vs 2.89s on the old service), as do
98,050/98,051-token beginning/middle retrieval probes. These small probes
are correctness smoke tests, not a broad model accuracy evaluation.

Commands completed:

- `.venv/bin/python -m pytest tests/kernels/attention/test_flashinfer.py -k nvfp4_kv_cache_update -v`: four passed.
- `.venv/bin/python -m pytest tests/v1/worker/test_extensible_kv_cache.py -k 'not engine_pads and not capture_gets and not sizing_warmup and not adaptive_verification and not drafter_capture' -v`: 31 passed, five deselected (separate model-start integration tests).
- Pre-commit on changed documentation: passed.

Longer-context validation and final service selection are in progress.

## Local fix for #46268

Based on the unmerged vLLM [#66864](https://github.com/vllm-project/vllm/pull/66864)
(commits `902215d5cb` and `e8c8cc046b`), with local corrections. No duplicate
upstream PR is proposed. Original author: tgghzyh.

The first selected CUTLASS NVFP4 kernel sets a conservative automatic
`MAX_JOBS` when there is no positive explicit override. It budgets half the
available memory at at least 8 GiB/job, scales the allowance with NVCC
threads, and caps jobs by CPU affinity. Local changes also walk inherited
cgroup v2 `memory.max` and `memory.high`, and replace zero/invalid values
rather than allowing unlimited Ninja jobs. Positive values are normalized
for FlashInfer's numeric environment parser. The chosen value persists in
the worker and applies per build, not as a global reservation across workers.
At least one job is allowed; this heuristic cannot guarantee that it fits.
The existing `MAX_JOBS=2` service override remains useful.

Other proposals reviewed and tested with deterministic synthetic memory
and environment inputs (not claims of full GPU validation of those PRs):

| Proposal | Finding |
| --- | --- |
| FlashInfer [#3636](https://github.com/flashinfer-ai/flashinfer/pull/3636) | Walks nested v2 limits and accounts for NVCC threads; accepts `MAX_JOBS=0` as unlimited. |
| FlashInfer [#3665](https://github.com/flashinfer-ai/flashinfer/pull/3665) | Walks nested v1/v2 limits and rejects zero; leaves Ninja's default when its CPU estimate fits, and ignores NVCC thread memory growth. |
| FlashInfer [#3899](https://github.com/flashinfer-ai/flashinfer/pull/3899) | Ignores cgroups; chose six jobs with only 12 GiB nested service headroom; preserves zero. |
| vLLM #66864 | Root-only memory helper misses normal nested systemd limits (four jobs with 12 GiB headroom); preserves zero. Local fixes address these cases. |
| vLLM [#57625](https://github.com/vllm-project/vllm/pull/57625) | Changes package-build parallelism in setup.py, not FlashInfer runtime JIT; not a fix for this startup path. |

`.venv/bin/python -m pytest tests/utils_/test_flashinfer_utils.py -q`:
17 passed, including nested service/ancestor limits, soft budgets, invalid
job counts, NVCC threading, and explicit override handling.
The selected-kernel startup logged the automatic choice of two compiler
jobs on this host. This startup reused the FP4 compilation cache; the actual
cold FP4 compilation was observed during the earlier upstream baseline,
with an explicit two-job limit.
