# SM12x NVFP4 KV cache patches

Patch set for `--kv-cache-dtype nvfp4` on consumer Blackwell (sm_120,
RTX 5090) through the FlashInfer FA2 paged path. Mirrors the
`patches-v0290` series in the private serving repo. Apply in order on
branch `sm12x-nvfp4kv` (based on `v0.29.0rc2`) with `git apply`:

- `0001-sm120-nvfp4kv-fa2-routing.diff` — routes SM12x NVFP4 KV
  through the FlashInfer FA2 native path: backend gating, the uint8
  paged pool, and reader construction with the HND layout string
  (serving repo `0101`).
- `0002-nvfp4-writer-linear-vscale-sm12x.diff` — writer kernel:
  arch-conditional V-scale swizzle, so the FA2 sm12x reader can
  consume V scales linearly (serving repo `0102`). The serving cell
  loads this kernel as a prebuilt `vllm_sm12x_nvfp4kv.so` overlay via
  `VLLM_SM12X_NVFP4KV_SO`; applying this patch keeps the in-tree
  source in sync with that artifact.
- `0003-nvfp4kv-sm12x-layout-family-fix.diff` — the regression fix
  for upstream PR #51718: force the HND layout family in the
  worker's layout report when the cache dtype is nvfp4 on SM12x, so
  resolution, pool allocation, writer views and the reader string
  all agree. Without it the engine resolves LBNHC ("NHD") for a
  pool whose interior is physically HND, and every paged NVFP4
  decode reads uncorrelated K/V.

The XQA decode patches (`0103`, `0112` in the serving repo) are a
separate feature and are not included; serving runs with XQA
disabled (`VLLM_SM12X_NVFP4_XQA=0`).
