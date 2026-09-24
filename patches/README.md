# SM12x NVFP4 KV + DFlash2 patches

Patch set for `--kv-cache-dtype nvfp4` on consumer Blackwell (sm_120,
RTX 5090) through the FlashInfer FA2 native paged path, plus the
DFlash2 quantized-draft loader. These are the five commits applied on
branch `sm12x-nvfp4kv` (based on `v0.30.0`). Each `.diff` is a
self-contained unified diff; apply in order at the fork root:

    for p in 0001 0002 0003 0004 0005; do
        git apply -p1 "patches/${p}-"*.diff
    done

Order matters: `0001`–`0003` are the NVFP4-KV core; `0004` builds on
`0001`'s route; `0005` is independent (DFlash2 drafter, no NVFP4-KV
dependency). If something regresses after a vLLM rebase, the "look for"
line under the matching patch is where to start.

## 0001 — FA2 native NVFP4 KV path
`0001-sm120-nvfp4kv-fa2-routing.diff`

Routes SM12x NVFP4 KV through FlashInfer's FA2 native paged reader
(prefill + decode) instead of trtllm-gen: backend gating, the uint8
paged pool of packed 4-bit KV with fp8 block scales, and reader
construction with the HND layout string to match the writer's
contiguous strides. The pool interior is head-major (HND).

- Without it: `--kv-cache-dtype nvfp4` on SM12x has no usable attention
  backend.
- Look for: no NVFP4 route selected on sm120, or a reader/writer
  layout mismatch.

## 0002 — V-scale swizzle, arch-conditional
`0002-nvfp4-writer-linear-vscale-sm12x.diff`

The FA2 sm12x paged reader reads V scales linearly, while the SM100
trtllm-gen reader wants a 4-token swizzle. Swizzle only on device
major < 12, and let the `block_size % 4` assert follow the same
condition. Both readers share the per-page
`[K_data | K_scale | V_data | V_scale]` layout. The serving cell loads
this kernel as a prebuilt `vllm_sm12x_nvfp4kv.so` overlay via
`VLLM_SM12X_NVFP4KV_SO`; applying the patch keeps the in-tree source in
sync with that artifact.

- Without it: SM12x reads V scales with the wrong swizzle.
- Look for: a V-scale read mismatch on sm120, or a `block_size % 4`
  assert on a non-multiple-of-4 block size.

## 0003 — HND layout-family fix
`0003-nvfp4kv-sm12x-layout-family-fix.diff`

The regression fix for upstream PR #51718 (KV-cache layout
standardization, commit `8bdc70ec`). Backend layout declarations moved
to the worker's layout RPC, which runs without the vllm_config context,
so the NVFP4 HND declaration is never reached and resolution falls back
to the LBNHC ("NHD") default. The FA2 sm12x NVFP4 paged reader is then
told NHD about a pool whose interior is physically HND, and every paged
decode reads uncorrelated K/V. This reports the HND family (LBHNC/BLHNC)
from the worker when the cache dtype is nvfp4 on SM12x, so resolution,
pool allocation, writer views and the reader string all agree.

- Without it: silent wrong outputs on NVFP4 KV paged decode.
- Look for: plausible-but-wrong text on long-context NVFP4 KV decode;
  live token loops / `finish_reason=length` in the live cell; failing
  needle probes.

## 0004 — Opt-in NVFP4 prefill split-KV
`0004-nvfp4-prefill-split-kv-sm12x.diff`

FlashInfer 0.6.18 forces `disable_split_kv=False` for packed NVFP4 KV,
reverting the 0.6.16 behavior, so the SM12x NVFP4 prefill/verify route
always ran with split-KV off. This adds a default-off lever,
`VLLM_SM12X_NVFP4_PREFILL_SPLIT_KV=1`, that restores the 0.6.16
split-KV behavior for that route. Nothing changes until the flag is set.

- Without it: long-context NVFP4 prefill/verify runs slower (split-KV
  off). The flag is opt-in, so leaving it unset is safe.
- Look for: a ~30K-context decode that should be ~144 tok/s collapsing
  to ~29 tok/s.

## 0005 — DFlash2 quantized-draft loader
`0005-dflash2-quantized-draft-loader.diff`

DFlash2's `_build_context_kv_buffers` assumes a dense `.weight` on each
QKV linear. A quantized draft (e.g. the syvai W4A16
compressed-tensors/Marlin checkpoint) stores `weight_packed` plus a
scale and exposes no `.weight`, so the fused context-KV build dies with
`AttributeError: 'QKVParallelLinear' object has no attribute 'weight'`.
This materializes the dense K/V projection once at load, through the
layer's own quant kernel (`eye(in) @ W^T`, transposed), and skips the
fused-KV build at load time when the quantized kernel cannot run yet;
the lazy context-KV rebuild runs after post-load repacking completes.

- Without it: any quantized DFlash2 draft (W4A16 / NVFP4) crashes at
  drafter load.
- Look for: a DFlash2 drafter load crash; bf16 drafts are unaffected.

The XQA decode patches (`0103`, `0112` in the serving repo) are a
separate feature and are not included; serving runs with XQA disabled
(`VLLM_SM12X_NVFP4_XQA=0`).
