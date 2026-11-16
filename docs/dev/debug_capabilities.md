# OpenVINO Debug Capabilities

OpenVINO components provides different debug capabilities, to get more information please read:

* [OpenVINO Model Debug Capabilities](https://docs.openvino.ai/2026/openvino-workflow/running-inference/model-representation.html#model-debugging-capabilities)
* [OpenVINO Pass Manager Debug Capabilities](#todo)

## GPU MoE Cache and Lookahead Statistics

For GPU expert offloading, set `MOE_OTD_LAYER_PERF_LOG=1` to emit
`[OTD_CACHE]` records on stderr at process exit. Each layer and `layer=ALL`
have separate `phase=prefill`, `phase=decode`, and `phase=unattributed` rows.
Hits are completed resident experts at their first demand lookup; pending
loads are misses and are also reported as `pending`. Percentages use only
that phase's `hits + misses`. Repeated experts within one layer invocation,
wave retries, and speculative probes do not add demand observations.
Single-token invocations are decode; multi-token invocations are prefill.

Enable `GPU_MOE_LOOKAHEAD_PREFETCH` for LES. With
`OV_GPU_LES_PREDICT_COMPARE=1`, `[LES_RECALL]` reports decode prediction
set-intersection counts (`matched`) and actual unique Top-K counts (`actual`)
for the same target layer and decode generation. Recall is `matched / actual`
over compared observations only. `missing`, `stale`, and `late` report excluded
observations separately; low comparison coverage must not be interpreted as
prediction accuracy. Statistics reuse the worker's existing host readback.

`[LES_PREFETCH]` distinguishes full deduplicated prediction candidates from
reserved, already resident, already loading, physically blocked, expired, and
canceled outcomes. All predicted Top-K candidates are attempted without a
Top-4 or capacity-derived candidate budget; pinned slots and in-flight writes
remain protected. A reservation is not a completed upload. Load completion,
cancellation, failure, unread task expiry, and demand conflict retries are
reported separately. Demand only drains the worker and retries when pending
speculation prevents a safe reservation; its original hit/miss observation is
retained across that wait.

Both environment gates require the exact value `1`. Counts include all
requests in the process, including warmup; timing tools may exclude warmup.
The legacy `MOE_OTD_PERF_LOG` aggregate is not a substitute for phase-separated
cache statistics.

## See also
 * [OpenVINO™ README](../../README.md)
 * [Developer documentation](../../docs/dev/index.md)
