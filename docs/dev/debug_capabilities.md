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

## GPU LES Joined Timeline

Set `OV_GPU_LES_TIMELINE=1` before starting the process to enable bounded native
split-LES decode diagnostics on stderr. Other values, including unset, disable
the trace. This gate does not enable or change the existing aggregate counters.
It adds no GPU readback, GPU event query, GPU wait or stream synchronization:
predictions reuse the worker's existing readback. CPU-side tracing has overhead.

Optionally set `OV_GPU_LES_TIMELINE_SAMPLE=measured:2:1,128,255` to retain
detailed Steps only for measured iteration 2, decode tokens 1, 128 and 255,
across every native MoE layer. For 40 layers and TopK 8 this selects 120 Steps
and 960 first-demand results. It does not shorten or otherwise change model
execution: ratio90, gen256, warmup1 and measured3 remain the runner's settings.
Unset preserves the full trace; the timeline gate unset/off still disables
tracing. The selector accepts `measured` or `warmup`, a canonical unsigned
iteration and unique positive canonical unsigned tokens. Empty, malformed,
overflowing and duplicate values throw an explicit configuration error when
tracing is enabled. Missing caller labels (`unlabeled`/`NA`) do not match a
sample; native callers without runner labels must provide labels to sample.
Directly constructed unit-test Steps explicitly bypass this registry filter.

Unselected Steps allocate no detailed Step/demand watchers or cache snapshots.
Lightweight per-load metadata remains owned by existing cache slots/jobs, not a
process-wide history. A selected snapshot (including non-TopK ready/loading
entries), candidate, or demand exposes its referenced load lifecycle exactly
once, even if the transfer completed in a previous token or iteration. Other
unselected loads produce no lifecycle output. Origins are immutable caller
keys; foreign `load_start`/`load_result` records retain their original key and
`obs_ns` and are **dependency evidence only**, never additional selected Steps.
Late exposure can place old observation timestamps after newer output sequence
numbers; `seq` orders output, not transfer time. Late demand subscribers receive
the original successful terminal time (with zero latency if already ready).

### Encoding and Joins

Each line is `LES_TIMELINE` followed by tab-separated `key=value` fields.
`v=1`, `seq` (process-local serialized output order), `obs_ns` (CPU steady-clock
nanoseconds), `network`, `request`, `token`, `epoch`, `layer`, `layer_id`, and
`stage` are common fields. All numbers are decimal. Strings percent-escape
bytes outside printable ASCII, whitespace, `%`, and `=` using uppercase hex.
Decode escapes *after* splitting fields. Empty lists/strings are `-`; unavailable
values are `NA`. IDs use commas; `slots` uses `expert:slot:load_gen` tuples.
No spaces or raw tabs/newlines occur inside a field. No JSON dependency is
available in this owner; the same encoding applies to every event.

Join a layer step by `(network, request, token, epoch, layer, layer_id)` within
one process log. `network` is the native network/stream identity (LES requires
one stream). `request` counts **first-layer prefill boundaries**, not benchmark
iterations or API infer calls; repeated/chunked prefill creates new segments.
Zero means no prefill boundary was observed. `token` is 1-based decode traversal
since that boundary. `epoch` is the existing scheduler generation, which also
advances per prefill layer; it must not be interpreted as a request counter.
Single-token prefill is indistinguishable from decode in this path. No benchmark
warmup/measured labels are inferred or read on the worker.
Optional `OV_GPU_LES_TIMELINE_PHASE` and `OV_GPU_LES_TIMELINE_ITER` labels are
copied on the caller thread when a layer-step context is created, and emitted
as `request_phase` and `iter`. They default to `unlabeled` and `NA`. A serialized
diagnostic runner may set them before each synchronous generation, restoring
the previous environment afterward. Workers retain the originating context
even if a subsequent generation changes the environment; they never reread it.

Join a load by `(cache, slot, load_gen)` within that process. `cache` is a native
cache address and must not be joined across processes. `load_gen` is assigned
under the cache mutex for each new reservation, including reusing the same
expert/slot. `expert` is the actual expert ID, never its slot index. Legacy
synchronously filled entries have generation zero. Initial snapshots include
the whole cache, not just Top-K. `resident` includes all mapped entries;
`ready` means fully uploaded and cache-published; `loading` means pending.
An unfilled legacy entry may be resident but neither ready nor loading.

### Events

| `stage` | Additional fields and meaning |
| --- | --- |
| `pre_prefetch` | `status,cache,resident,ready,loading,slots`; full cache snapshot on caller with `status=before_submit` before submitting this step's first prefetch task, hence before any speculative mutation by that task. Layer 0 uses `status=no_predecessor`; other layers without a submitted task use `status=not_submitted`. Both fallback cases use a demand-entry snapshot before any speculation for that step. |
| `prediction_submitted` | Router prediction was enqueued, not necessarily read back or prefetched. |
| `prediction_result` | `status=available,unbound,expired,canceled,failed`, `predicted`; unread/failed IDs are `NA`. Available IDs come only from the existing worker readback. |
| `prediction_at_demand` | `status=matched,late,missing,stale,no_predecessor`, `actual,predicted,prediction_epoch,recall=NA`; immutable first-demand availability status. `predicted=NA` if host IDs are not yet available. Stale IDs, if shown, are not compared. |
| `recall` | `predicted,actual,matched,actual_count`; same-step unique set intersection; recall is `matched/actual_count`, or NA for a zero denominator. May follow demand when data was late; never overwrites `prediction_at_demand`. No row means recall unavailable, not zero. |
| `candidate` | `expert,status=reserved,resident,loading,blocked,expired,canceled`; reserved, resident and loading entries carry `cache,slot,load_gen` observed atomically in the cache reservation decision, with that decision's `obs_ns`. Resident/loading observations do not reserve another job or transfer; generation zero identifies an original resident without a traced transfer. Blocked/expired/canceled entries have no load identity. One event per deduplicated candidate. Unread tasks report task-level expiry/cancellation instead. |
| `first_demand` | `wave=0,pass=0,actual,demand_ready,demand_pending,demand_missing,remaining` and full cache snapshot; one cache-locked observation of **all** actual routed IDs before the first wave. Pending is never retrospectively a hit. |
| `before_wave_partition` | Same set/snapshot fields, scoped to this wave's actual IDs; `wave` is zero-based, `pass=0`. |
| `after_retry_wait` | Same fields, only after the existing conflict-retry worker drain. This is not the first-demand classification. |
| `before_pass_submit` | Same fields plus `submit` (exact routed IDs in the mask), `shared=0/1`; `pass=1` means ready pass A, `pass=2` pending pass B. Its existing dependency barrier has been **enqueued**, not CPU-waited. Pending may still appear here. |
| `demand_load` | `expert,cache,slot,load_gen`; newly reserved demand load queued to the worker. |
| `promotion` | `expert,cache,slot,load_gen`; existing speculative queue entry moved to demand priority, with the original load identity preserved. An already-running load is linked by the first-demand cache snapshot, without a promotion event. |
| `load_start` | `expert,origin=speculative/demand,cache,slot,load_gen`; worker selected the job, immediately before disk/read/transpose/upload work. Queue delay precedes this event. |
| `load_result` | Same identity/origin fields, `status=ready,canceled,failed,superseded`; emitted once per load. `ready` is timestamped under the cache mutex only after the existing full upload routine succeeded and cache publication completed. Cancellation cannot turn into success. |
| `demand_result` | `expert,cache,slot,load_gen,status,demand_mark_ns,weights_ready_ns,latency_ns`; one result per distinct demanded expert. `ready` is success; canceled/failed/superseded/unavailable use `NA` for ready time and latency. Unresolved observations retire as unavailable. |

On observation rows, `demand_missing` means absent/unfilled, and `remaining`
means **not ready**, including both absent and pending IDs. Pass/wave rows are
CPU observations before submission, not GPU completion claims. Load publication
and snapshots are timestamped under the cache mutex; output happens outside it.
`seq` orders emitted lines, not observations: worker output can arrive after a
later observation, so join by identity and compare `obs_ns`, not line position.
Other stderr diagnostics can occur between complete timeline lines.

Demand-to-ready latency is monotonic CPU **wall** time from the first cache
observation of all actual IDs through successful full upload/cache publication.
It includes queueing, disk, transpose, copy and existing dependency waits, not
CPU utilization or GPU kernel time. Already-ready demand has exactly
`weights_ready_ns=demand_mark_ns` and `latency_ns=0`, even when its historical
load publication predates demand. Failed/canceled loads never receive a numeric
success latency. The original observation remains unchanged across retries,
promotion and later waves. Live per-layer-step records and cache/job-owned load
records retire on the next traversal, replacement, completion or cancellation;
there is no global event-history collection.

For example, the following parsed record means expert 7 first demanded at 100
became usable at 140, taking 40 ns wall time; slot 2 is not expert 2:

```text
{v:1, seq:12, obs_ns:140, network:3, request:1, token:2, epoch:8,
 layer:4, layer_id:"decoder\tmoe", stage:"demand_result", expert:7,
 cache:1234, slot:2, load_gen:9, status:"ready", demand_mark_ns:100,
 weights_ready_ns:140, latency_ns:40}
```

The encoded `layer_id` field for that record is `decoder%09moe`.

## See also
 * [OpenVINO™ README](../../README.md)
 * [Developer documentation](../../docs/dev/index.md)
