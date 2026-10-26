# CUDA Contrib Ops investigation (PR #137)

## Reproduction

- Builder baseline: VOICEVOX/onnxruntime-builder PR #137, commit `147bd8cfd16aac7b0eab494e8d2dcca2f8dc3df9`.
- ONNX Runtime: v1.31.0, pinned to `402979ac1bef89dcd55add41eaa7277c407599eb`.
- Windows: windows-2022, CUDA 12.8.1, cuDNN 9.8.0.87.
- Linux: ubuntu-22.04, GCC 13, CUDA 12.9.1, cuDNN 9.8.0.87.
- Same build options as PR #137; the two `contrib` entries remove ONLY `--disable_contrib_ops`.
- All four builds are cold builds. No releases or signing are performed.
- Normal 6-hour runner limit is retained; the compile step has a 320-minute timeout to leave time for diagnostics.

Run: https://github.com/Hiroshiba/onnxruntime-builder/actions/runs/40314439974

The PR's reference run https://github.com/sevenc-nanashi/onnxruntime-builder/actions/runs/40296059613 restored CUDA build caches and skipped compilation. It is not evidence of fresh build duration.

## Browser-free execution

The GitHub connector has no workflow-dispatch tool, and this local environment has neither gh nor a GitHub API token. A branch-scoped push workflow successfully invoked `gh workflow run build.yml` on a GitHub-hosted runner using its ephemeral GITHUB_TOKEN with actions:write. No personal access token or browser session was needed.

Dispatch proof: https://github.com/Hiroshiba/onnxruntime-builder/actions/runs/40314425400

The initial branch creation also triggered the original push workflow. That superseded run (40313919867) was cancelled through the same browser-free mechanism. The launcher currently contains that cancellation request; edit it deliberately for a subsequent dispatch. Do not rerun it to launch another build.

## Source findings (not build-result claims)

- In v1.31.0, disabling Contrib Ops no longer disables Flash Attention or memory-efficient attention: the standard ONNX Attention operator also uses that infrastructure. The v1.23.2 behavior differs.
- `onnxruntime_NVCC_THREADS=1` is meaningful: build.py defaults to 4 and appends cmake_extra_defines later, allowing this override.
- `onnxruntime_USE_FPA_INTB_GEMM=OFF` is redundant while Contrib Ops are disabled (CMake forces it OFF), but not after enabling them.
- `onnxruntime_QUICK_BUILD=ON` is not merely a Flash Attention build setting. It also changes MoE/CUTLASS instantiations and supported kernel paths when Contrib Ops are enabled. Remove it only with a separate build comparison.
- `onnxruntime_USE_FP8_KV_CACHE=OFF` changes kernel registration and FP8 paths, including XQA sources. It is not an unused option.
- `onnxruntime_CUDA_FATBIN_COMPRESS_SIZE=ON` controls binary compression and driver compatibility; it is not a Contrib Ops toggle.
- The PR's `75;80;90` architecture set is narrower than main's GPU architecture set. A successful PR-config build alone does not establish that the full main architecture set fits within the time limit.

## Validation and next decision

- actionlint, ShellCheck and Pinact passed on commit d5609f6a90008523d3702955dedfef0fc3cf65d7.
- The baseline options/toolchains were programmatically compared with PR #137. The enabled variants differ only by removal of --disable_contrib_ops.
- Build outcomes are PENDING. No claim that the timeout is fixed has been made.
- Compare Windows and Linux baseline/enabled durations and failures first. If enabling contrib succeeds, the disabled-contrib patch becomes inapplicable and can be removed from the final configuration. Next test QUICK_BUILD and the individual feature-disable options one at a time; retain resource/architecture constraints until measured.
- Diagnostics artifacts contain build output and CMakeCache.txt, including failure cases when the runner can still upload artifacts.
