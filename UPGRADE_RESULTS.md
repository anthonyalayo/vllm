# RTX 5090 NVFP4 upgrade results — 2026-11-17

Pinned model: `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`, revision
`0cc27958cefbbe231782ec8511de8c4eb5233348`. GPU: RTX 5090, 32 GB VRAM.
The CUDA runtime reports 31.44 GiB of usable device memory on this host.

Before: branch `sm12x-nvfp4kv`, `63277ca386`, v0.30-based, Torch 2.13.0+cu132,
FlashInfer 0.6.18.post1. After: upstream main `935f87de84` (after v0.34.0),
Torch 2.14.0+cu132, FlashInfer 0.7.4, with the local host-JIT-memory fix
[documented here](UPGRADE_NOTES.md#local-fix-for-46268).

The model revision, custom template, NVFP4 KV format, three-token MTP,
eight maximum sequences, 2048 batched tokens, image limits, and prefix
caching are preserved. AI assistance was used for this migration and testing.
No duplicate upstream PR is proposed.

## Context, cache, and startup

**A request's context window and total KV capacity are different.** The
checkpoint's full 262,144-token request limit already fit before the upgrade.
A larger shared KV pool supports more concurrent or cached work; it does not
increase the checkpoint's positional limit.

| Measurement | Pinned branch | Validated upgrade, fresh autotune |
| --- | --- | --- |
| GPU utilization setting | 0.92 | Unset: measured sizing, 1.0 budget |
| Maximum request context | 262,144 tokens | 262,144 tokens |
| Shared KV capacity | 419,915 tokens | 424,196 tokens (+1.02%) |
| Full-context concurrency | 1.60185x | 1.61818x |
| Logical KV backing | 8.986 GiB | 9.350 GiB |
| Actual VMM backing | Not applicable | 9.363 GiB |
| Fresh autotune needs an initial 0.8 run | Existing operational practice | Not needed in the tested pinned configuration |
| Autotuner CUDA allocation failures | Reproduced at 0.92; falls back to default tactics | None in the fresh-autotune candidate |
| Host compiler job limit | Explicit MAX_JOBS=2 drop-in | MAX_JOBS unset; automatic guard cold-build tested |
| CUDA graph capture maximum | 64 | 32, covering eight requests x four verify tokens |
| Torch allocator | Default | expandable_segments:True |

### Why the first results got worse

This was measured and corrected; it was not assumed to be an unavoidable
tradeoff or addressed by lowering memory safety margins.

| Experiment | KV tokens | Full-context concurrency |
| --- | ---: | ---: |
| Original pinned branch, 0.92 | 419,915 | 1.60x |
| Main, still capped at 0.92 | 331,254 | 1.26x |
| Main, uncapped, existing autotune cache | 407,514 | 1.55x |
| Main + host-memory guard, uncapped, fresh autotune, default allocator | 367,001 | 1.40x |
| Fresh autotune, expandable allocator, capture maximum 64 | 419,430 | 1.60x |
| Fresh autotune, expandable allocator, capture maximum 32 | 424,196 | 1.62x |

1. **An explicit 0.92 still limits measured sizing.** Omitting utilization
   enables the full measured budget. It does not allocate every last byte:
   the candidate retains the upstream 1 GiB CUDA safety floor plus about
   0.85 GiB for measured transient/reservation growth. The flag
   `--no-enable-extensible-kv-cache`, rather than a utilization value,
   restores estimate-based allocation.
2. **Cold autotuning fragmented the default Torch allocator.** The cold
   snapshot contained 1.060 GiB of inactive splits, with 19.372 GiB allocated
   versus 20.512 GiB reserved. Final sizing already calls garbage collection
   and empty_cache; live allocations pin these split segments, and the
   separate VMM KV backing cannot use their holes. Expandable segments
   reduced measured inactive splits to zero and increased post-warmup free
   memory from 9.90 to 11.11 GiB. Thus the recovery was real free memory,
   not merely a changed statistic. A 1536-token batch experiment still
   produced 367,001 tokens: the cold allocation history, not that batch
   limit, explained this penalty. The exact tensors pinning the fragments
   were not identified; no particular buffer leak is claimed.
3. **Current GDN checkpoints and block alignment have real costs.** The old
   worker had 173 blocks, each 17 x 3,280,896 bytes. A full-length request
   required 93 attention blocks plus 3 x 5 GDN state blocks: 108 total.
   Current upstream uses 3,317,760-byte pages and 2880-token blocks, versus
   2848 before. Each GDN group now reserves a sixth block for mid-prefill
   checkpointing; attention requires 92 blocks, giving 110 total. This
   reproduces the earlier near-equal-byte result exactly:
   `floor(173 / 108 * 262144) = 419915` versus
   `floor(171 / 110 * 262144) = 407514`. The final candidate fits 178 blocks,
   so `floor(178 / 110 * 262144) = 424196`.
4. **The old branch already had a profiling workaround.** Its SM12x patch
   warmed the NVFP4 path before the measured profile. That is relevant
   context when comparing this fork with the generic release claims.
   Bigger gains on other models are not a guarantee for this hybrid model.

The safety floor, transient reserves, prefix caching, float32 GDN state,
maximum sequences, and batch limit were preserved. Disabling the NVFP4
split-KV corruption guard was not part of the upgrade.

## Matched serving benchmark

| Workload | Before output tok/s | After output tok/s | Change | Mean TTFT, ms before → after | Mean TPOT, ms before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2,048 input / 1 concurrent | 185.0 | 184.6 | -0.2% | 199 → 140 | 4.65 → 4.89 |
| 2,048 input / 4 concurrent | 532.1 | 496.8 | -6.6% | 394 → 317 | 5.84 → 6.07 |
| 32,768 input / 1 concurrent | 56.2 | 60.8 | +8.2% | 3282 → 2994 | 4.98 → 4.76 |
| 32,768 input / 4 concurrent | 70.3 | 77.9 | +10.7% | 7689 → 6592 | 25.04 → 24.00 |

All 32 requests in each arm completed; none failed. Throughput is aggregate
output tokens per second including prefill time, not pure decode speed.
The upgrade improves first-token latency in all four scenarios. Long-input
throughput improves 8–11%; short-input throughput is flat at concurrency one
and 6.6% lower at concurrency four. This is not an across-the-board speedup.
The synthetic greedy outputs and MTP acceptance also differ between versions:
for short concurrency four, mean accepted length is 3.55 before versus 3.26
after. These upstream/runtime changes are not isolated kernel timings.

Both arms use the upgraded benchmark client, the pinned snapshot tokenizer,
the same deterministic random workload for each scenario, 256 output
tokens, ignore_eos, explicit temperature=0, one warmup request, and the same
request counts/concurrency. Each scenario uses a distinct fixed dataset seed
to prevent an earlier scenario from warming all its prefixes. Prefix caching
is enabled in both arms. The client's initial/warmup request can warm one
request inside its scenario. These are small synthetic serving benchmarks,
not a broad accuracy evaluation or a guarantee for every production workload.
Earlier exploratory runs without explicit greedy sampling are retained as
artifacts but are not the final comparison.

## Validation and artifacts

- Upstream NVFP4 writer/layout suite: 4 passed, covering LBNHC, LBHNC,
  BLNHC, and BLHNC.
- Focused extensible-cache suite: 31 passed; five tests that start separate
  models were deselected to keep the target GPU available.
- Local JIT-memory unit suite: 17 passed.
- Unmodified upstream baseline: 10/10 serving probes passed.
- Final allocator/capture candidate: 10/10 probes passed, including exact
  retrieval from 25k, 98k, and 238,050-token prompts, multiple needle
  positions, and four concurrent 25k requests.
- Eight simultaneous requests: 8/8 returned their exact code and the
  complete ordered 1–50 sequence; each finished normally.
- Combined auto tool choice + JSON schema: both tool and JSON branches
  passed. A basic image/color check passed.
- Pre-commit passed for changed source, tests, and documentation.
- Candidate unit passed systemd-analyze verify.

This validates the currently served MTP model. Quantized DFlash drafts were
not GPU-tested; their local dense-materialization workaround has a different
upstream replacement, described in the patch disposition notes.

Raw logs, complete requests/responses, memory snapshots, benchmark JSON,
environment freeze, old unit backup, candidate unit, and reproducible scripts
are under `/home/anthony/projects/vllm-upgrade-validation/`. GPU server logs
also remain in `/tmp/vllm-*.log`. The original 14:34 systemd journal was saved
as `before-startup.log` in the validation directory.

## Deployment and rollback

The updated systemd service started successfully at 15:25:34 local time.
Its journal independently reports the same 424,196-token cache and 1.62x
concurrency as the cold candidate. Health, model discovery, both tool/schema
branches, and basic vision passed against the actual service. Its additional
238,050-token exact-recall probe passed in 104.4s, with zero restarts and zero
preemptions. The service
keeps the same pinned revision and 262,144-token request context.

The rebased feature branch is `sm12x-nvfp4kv` in
`/home/anthony/projects/vllm-upgrade`, which is also the service's working
directory. The original checkout remains at `63277ca386` on the local
`sm12x-nvfp4kv-pinned-20261117` branch, preserving its old editable environment
and CUDA overlay for rollback. Remote backups are
`backup/sm12x-nvfp4kv-20261117` and `backup/main-20261117`.

The original unit is also preserved as
`/etc/systemd/system/vllm.service.pre-upgrade-20261117`. To restore the old
service and environment, including the old explicit compiler job limit:

```bash
sudo systemctl stop vllm
sudo cp /etc/systemd/system/vllm.service.pre-upgrade-20261117 /etc/systemd/system/vllm.service
sudo cp /etc/systemd/system/vllm.service.d/10-memory.conf.pre-auto-jobs-20261117 /etc/systemd/system/vllm.service.d/10-memory.conf
sudo systemctl daemon-reload
sudo systemctl start vllm
```

The `10-memory.conf` drop-in retains MemoryHigh=32G and leaves MAX_JOBS unset.
The new unit enables expandable allocator segments, leaves
GPU utilization unset, uses capture maximum 32, and removes the custom
NVFP4 overlay and forced split-KV variables. Its historical 0.8 preparation
comment is replaced with the measured pinned-configuration behavior.

The tested autotune cache is preserved at
`/home/anthony/.cache/vllm/rtx5090-autotune-v2`.

## Follow-up: utilization cap and selective embedding UVA

These runs use the same pinned revision, allocator, capture maximum 32, model
arguments, and warmed autotune cache. Unless noted, utilization is unset. The
request window remains 262,144 tokens in every run.

| Configuration | KV tokens | Full-window concurrency | 32k x 8 pressure run |
| --- | ---: | ---: | --- |
| Rebased, expandable allocator, capture 32 | 424,196 | 1.62x | 0 preemptions; 63.15 aggregate tok/s |
| Same flags, `--gpu-memory-utilization 0.92` | 388,449 | 1.48x | 2 preemptions; 59.90 aggregate tok/s |
| Selective UVA offload of `embed_tokens` | 531,437 | 2.03x | 0 preemptions; 62.78 aggregate tok/s |
| UVA plus `--mamba-ssm-cache-dtype bfloat16` | 581,507 | 2.22x | Not measured separately |

The capped run is materially worse for cache capacity: it sizes to 388,449
tokens, 8.4% below the uncapped control. The old 0.92 setting is not part of
the recommended configuration. Unset utilization means extensible KV sizes
from measured free memory after warmup; it does not disable safety headroom or
allow one request beyond the configured model window.

The single shared embedding is BF16 `[248320, 5120]`, 2,542,796,800 bytes
(2.368 GiB). The main model and MTP already share one allocation on the
upstream control. A no-temporary-draft-table patch therefore reduces load-time
peak only and does not grow steady-state KV capacity. Selective UVA moves that
existing shared embedding to host memory and returns 107,241 KV tokens (+25.3%)
on this host. The embedding remains shared, and sampled row lookups match the
pinned checkpoint exactly (max absolute error 0.0).

| Serving case (256 generated tokens) | Upstream control | UVA | Change |
| --- | ---: | ---: | ---: |
| Short, concurrency 1 | 184.46 tok/s | 184.48 tok/s | +0.01% |
| Short, concurrency 4 | 495.89 tok/s | 502.76 tok/s | +1.4% |
| 32k input, concurrency 1 | 60.65 tok/s | 60.60 tok/s | -0.1% |
| 32k input, concurrency 4 | 77.70 tok/s | 77.61 tok/s | -0.1% |

Each row is one deterministic serving-benchmark run (one warmup, 256 output
tokens); treat sub-percent differences as measurement noise, not a speedup.
Eight concurrent requests at 8k and 32k completed without preemption with
UVA. The 238,053-token exact-recall prompt passed in 104.9s. The configured
maximum request context is still 262,144 tokens; cache pool capacity supports
more total concurrent/cached tokens, not a longer positional window. The
startup-only table-sharing candidate measured 424,196 tokens and control-like
throughput, as expected.

The tested UVA CLI additions are `--offload-backend uva --cpu-offload-gb 3
--cpu-offload-params embed_tokens`. Its effective config offloads 2.37 GiB.
The control source is `8d540b9144549aaf5ee6f26e47417ec055ceb3c1` and the
isolated UVA implementation is `3af5fd1d09d5793b8d3f78aa5837b2d6a1fb1113`.
The full tested argument vector is in
`/home/anthony/projects/vllm-upgrade-validation/uva-effective-launch.json`;
the UVA benchmark used the separate `followup-autotune` cache and
`PYTORCH_ALLOC_CONF=expandable_segments:True`, with `MAX_JOBS` unset. Each
serving-matrix cell is one measured run after one warmup; no confidence interval
is claimed. The four cells used short/32k randomized inputs and c1/c4 request
concurrency, each generated 256 tokens.
UVA therefore looks promising for capacity without the feared decode hit on
these measurements; long-context recall and these finite synthetic benchmarks
do not establish quality or performance for every workload.

The BF16 GDN-state follow-up adds 50,070 KV tokens over FP32 UVA (+9.4%),
but measured serving throughput was 180.16 / 581.31 / 60.27 / 74.22 tok/s
for short-c1 / short-c4 / 32k-c1 / 32k-c4. Relative to the FP32 UVA run,
long-c4 throughput was 4.4% lower and 238,054-token recall took 114.7s versus
104.9s. Both exact-recall checks passed. These are single runs and do not
establish a precision-quality result; BF16 remains an optional, unqualified
capacity mode, not the recommended setting.

Explicit checks against the FP32 UVA server passed function tool selection,
JSON schema output while a tool was available, and red-image recognition.
These complement the long recall and concurrent serving checks; they are
smoke checks, not exhaustive multimodal or tool-use evaluations.

## `--max-num-seqs` parallelism sweep

To distinguish the CLI limit from client-side queuing, the service was
restarted with `--max-num-seqs` set to 8, 12, and 16. Each configuration used
FP32 UVA, the same 531k-token cache (the 16-sequence run lost 2,383 tokens),
and a burst of unique 32k code prompts with 256 generated tokens per request.

| `--max-num-seqs` / simultaneous requests | Aggregate output | Batch elapsed | Mean TTFT | Peak running / waiting | Peak cache | Preemptions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 / 8 | 62.78 tok/s | 32.62s | Not captured in this run | 8 / 6 | 81.1% | 0 |
| 12 / 12 | 63.57 tok/s | 48.33s | 26.07s | 9 / 9 | 86.0% | 0 |
| 16 / 16 | 63.87 tok/s | 64.13s | 33.91s | 9 / 13 | 91.0% | 0 |

Throughput gains over 8 were only 1.3% at 12 and 1.7% at 16, while batch
completion time rose 48% and 97%. The 12/16 configurations never got all
requests running at once; higher configured limits mostly increased queuing
and latency. CUDA decode graphs remained captured only through 8 requests
even when `--max-num-seqs` was 12 or 16, which likely limits the benefit of
those larger settings. On this 5090, the best measured setting is to keep
`--max-num-seqs 8`; consider increasing it only alongside deliberate decode
graph capture tuning and a workload-specific concurrency test.

The sweep is a single run per setting with no confidence interval. The 8-case
measurement comes from the earlier UVA capacity stress script; 12 and 16 were
measured by `parallelism_sweep.py` with identical prompt/output dimensions.
See `systemd-uva-parallelism.json`, `max-seqs-12-launch.json`, and
`max-seqs-16-launch.json` under the validation directory. Production was
restored to FP32 UVA with max-num-seqs 8 after the experiment.

The systemd service now runs the FP32 UVA configuration. On 2026-11-17 it
started successfully with 531,437 KV tokens / 2.03x, logged 2.37 GiB offloaded,
and passed health, function-tool, JSON-with-tools, and image smoke checks.
The previous service file is preserved as
`/etc/systemd/system/vllm.service.pre-uva-20261117`; remove the three UVA CLI
arguments to return to the non-UVA configuration. The service continues to use
the same pinned model/revision and 262,144-token request window.

The `MAX_JOBS=2` workaround was removed from the service and drop-in. The local
guard selected one job with no override and completed a cold NVFP4 build in
487s under a 32 GiB test `MemoryMax`, peaking at 7.54 GiB with zero swap. This
tests the automatic path; it does not imply that each compiler process uses
less memory. Original service and drop-in copies are preserved as
`vllm.service.pre-auto-jobs-20261117` and
`10-memory.conf.pre-auto-jobs-20261117`. Restoring the pre-upgrade unit alone
is insufficient to restore the old explicit job cap: restore the corresponding
drop-in backup as well.
