# LES Context And Chain Discovery

P1 introduced a default-off, network-local Host task service. P2 moved
expert reads, unchanged repacking and nonblocking uploads to its worker and
separate OCL transfer lane, initially waiting for all weights before the
original MoE kernels ran. P3 adds ready-first masked GEMV execution for the
existing Decode/small-batch GEMV path; larger Prefill retains P2 scheduling.

## Configuration

- `GPU_ENABLE_LES`: internal boolean, default `false`.
- `GPU_LES_QUEUE_CAPACITY`: internal queued-task limit, default 32, range 1-1024.

These are experimental internal properties, not a public performance promise.
They cannot be passed through the public `compile_model()` property map. Use
`OV_GPU_ENABLE_LES=1` and optionally `OV_GPU_LES_QUEUE_CAPACITY` for experiments;
internal GPU tests can set the typed properties directly.
The capacity excludes the running task and is not a staging-byte budget.

### Diagnostic Flags

LES and speculation are opt-in. The following switches are disabled unless set
to the exact value `1`:

| Environment Variable | Purpose |
| --- | --- |
| `OV_GPU_ENABLE_LES` | Enable the network-local LES service. |
| `OV_GPU_LES_PREDICT` | Compute lookahead expert predictions for supported Router links. |
| `OV_GPU_LES_PREFETCH` | Prefetch predicted experts; requires LES and prediction. |
| `OV_GPU_LES_PREDICT_COMPARE` | Compare predictions with actual expert routing. |
| `MOE_LES_TRACE` | Record scheduling and transfer diagnostics; adds synchronization overhead. |
| `MOE_OTD_PERF_LOG` | Enable cache, prediction and transfer performance counters. |
| `MOE_LES_DECODE_SINGLE_PASS` | Experimental single-pass Decode, not a logging switch. |

For diagnostic Demand + predict + prefetch, set all switches above to `1`
except `MOE_LES_DECODE_SINGLE_PASS=0` to retain dual-pass scheduling.
Use `OV_VERBOSE=3 OV_VERBOSE_COLOR=0 OV_GPU_LOG_TO_FILE=/absolute/path/verbose.txt`
with a debug-capable GPU build. Keep each process's file path distinct.
Capture stdout/stderr as well: runtime counter summaries and direct trace output
can appear there rather than in the GPU verbose stream. Trace/compare runs are
diagnostic runs, not production latency measurements.
Enable `PERF_COUNT=true` in the compile configuration when using trace, because
native event timestamps require profiling-enabled queues. The summary includes
`les_prediction_matches` (rank matches), `les_prediction_expert_matches`,
`les_prediction_pairs`, `les_prediction_rank_hit_at_k` and
`les_prediction_recall_at_k` (expert membership recall). Older logs without
`les_prediction_expert_matches` incorrectly called rank hit rate recall.

### Compressed Router Projection

The original LES proxy FC accepted only plain FP16 projection weights. The
`projection_compressed_weights` rejection was a limitation of that implementation,
not an algorithmic requirement of LES. Removing the rejection without decoding
the weights would incorrectly interpret packed integers as FP16 values.

The proxy now supports constant, contiguous i4/u4/i8/u8 weights with FP16 scalar,
per-expert or grouped scales and supported constant zero points. The group count
must divide the hidden width. On first prediction, OpenVINO constant conversion
unpacks the values and the proxy caches an FP16 weight copy. The actual Router
and expert weight paths are unchanged. MiniCPM-V4.7 uses grouped Router weights
with hidden width 2048 and group size 64. Unsupported layouts still report a
qualification rejection instead of running with an incorrect interpretation.

`[LES_PROXY_DECOMPRESS]` records initialization of the proxy weight cache.
Successful qualification alone is not activation: check prediction events and
nonzero prediction/prefetch counters, then inspect measured recall and cache
statistics. A correct generated answer does not prove prediction was active.

## Network Ownership

[network](../src/graph/network.cpp) creates a context only for a non-internal
network with LES enabled, a nonzero offload ratio and an executable compressed
MoE primitive with a nonzero OTD cache capacity. Disabled LES, resident-only
and non-MoE graphs create no context or worker.

An eligible network requires an offload ratio of 1-99, exactly one configured
stream and record/replay disabled. Unsupported combinations fail construction;
they do not silently fall back to a different scheduling mode. Each network
owns its context, including networks instantiated from the same program.

Each `execute()` starts a new epoch, drains Host tasks before returning and
cancels them if execution or the worker fails. Destruction stops and joins the
worker before releasing network resources. P2 callbacks drain their issued DMA
before returning, including failure paths. There is no new public infer-request
cancellation interface.

## Decode Metadata And Completion

LES Decode retains three network-local `usm_host` metadata buffers. Host writes
use `mem_lock<int32_t, mem_lock_type::write>` rather than blocking OpenCL memcpy.
The preceding wave consumer completes before these buffers are reused. This
does not change the memory budget, expert slots, kernel arithmetic or masks.

The last wave need not block the Host before reduction submission. Reduction
depends on its native consumer event, which the weight provider also retains
and waits on before slot release. Non-final waves still wait before metadata
reuse. Trace mode still waits to obtain native event timings; it is deliberately
not equivalent to production scheduling.

With `MOE_LES_TRACE=1` and `OV_VERBOSE=1`, Decode additionally emits
`route_readback_window`, `demand_reserve_window`, `decode_metadata_window` and
`consumer_wait_window`. They are Host windows in the existing device-clock
domain, not native GPU commands. These windows add no new device dependency;
their clock reads and logging are skipped when trace is off. The trace-on
consumer window does not measure a wait that exists on the final production
wave. `upload_join_window` remains the worker-future wait diagnostic.

## Experimental Single-Pass Decode Control

`MOE_LES_DECODE_SINGLE_PASS=1` is an opt-in control for the LES batched GEMV
path (Decode/small batches). Only the exact value `1` enables it; the process
samples it once. Default ready-first scheduling and Prefill are unchanged.
The control joins missing uploads before one gate/down pass over both ready
and missing pairs. Shared experts still execute once, and existing consumer
events, metadata reuse waits and reservation fences remain intact.

Trace records use `single_pass_gate` and `single_pass_down`. Fewer submissions
trade away ready-compute/DMA overlap; this is not a default performance policy.
Source Debug/OCL validation passed 96 directed and 572 full focused tests with
the control enabled. Unprofiled single/dual timing is blocked by concurrent
compilation, so no net speedup or Release acceptance is claimed.

## Worker Contract

[les_context](../include/intel_gpu/runtime/les_context.hpp) owns one worker and
a bounded FIFO. The service is independent of OCL/ZE handles and per-layer
expert-weight providers. It is not a process singleton.

- `begin_epoch()` cancels the old queue, waits for running Host work to finish,
  clears the previous epoch's error and returns a new epoch identifier.
- `submit(epoch, work)` blocks while the queue is full. It returns an epoch-tagged
  shared completion future. Empty work and stale epochs are rejected.
- `wait()` waits for Host work and its captured resources to finish, then
  rethrows the worker error, if any.
- A task exception fails its ticket and all queued tickets. Blocked producers
  wake and receive the same error. A new epoch permits recovery.
- `cancel()` invalidates the epoch, cancels queued work and waits for active
  work. It does not preempt a running read or a previously submitted GPU command.
- `stop()` additionally rejects new tasks and joins the worker. Repeated stop
  is harmless. Destruction calls stop.

Control operations and submit must not run inside a worker task: they can
otherwise wait for the worker itself. The service detects and rejects that
reentrancy. Control transitions are serialized; producers must retain the
context's owner until they have returned. Stop joins the worker, not external
producer threads.

Cancelled/failed queued closures are destroyed outside the task mutex. No GPU
wait or worker join is performed while holding that mutex.

## Host Completion Is Not Device Completion

A ticket means the Host callback has returned and released its captures. It
does not imply an upload event has completed. Epochs invalidate stale metadata;
they do not fence DMA or permit slot/staging reuse. P2 separately retains
staging and native events until uploads and previous consumers complete.

## Validation

The existing property fixture covers defaults, clone/roundtrip and invalid
capacity. [Lifecycle tests](../tests/unit/test_cases/moe_les_context_test.cpp)
use fake Host work, without a GPU, to cover single-worker FIFO, queue backpressure,
cancellation, error fan-out/recovery, epoch rotation, blocked producers,
reentrant control, repeated destruction, active-task join and captured-resource
release. The same source can be built with ThreadSanitizer.

The existing MoE numerical fixture also checks LES-on resident and OTD cases,
independent contexts, repeated execute epochs and rejected stream/ratio/recording
combinations. The property fixture checks that a non-MoE graph with LES enabled
and ratio 0 or 50 creates no context and preserves its output.

## Verified Milestones

| Stage / Step | Commit | Verification |
|---|---|---|
| Stage 1 / P1.1 | `580cb907ba` | Default-off configuration; 184 focused OCL tests passed |
| Stage 1 / P1.2 | `736dc82af2` | 12 worker tests, the same 12 under TSan, 17 integrated configuration/lifecycle tests passed |
| Stage 1 / P1.3 | `903156e707` | 26 P1-focused tests, 205 complete focused tests, 8 LES-on forced-fallback tests passed |
| Stage 1 / P2.1a | `a75f2d0b7e` | 12 cross-queue event tests and 215 complete focused OCL tests passed |
| Stage 1 / P2.1b | `be2210dcaa` | 12 CPU worker tests and 27 integrated LES/lane tests passed |
| Stage 1 / P2.2 | `e5a06299d5` | 31 reservation/LRU/provider tests passed |
| Stage 1 / P2.3 | `0b7014a295` | 14 upload/provider tests passed |
| Stage 1 / P2.4 | `9a8dcff850` | 303 complete focused tests, 80 LES-on forced-fallback tests, source Qwen ratio-80 LES-on smoke passed |
| Stage 1 / P3.1a | `94d1eaae6d` | Canonical pair snapshots and nonblocking demand foundation; 307 focused tests and 80 forced-fallback cases passed |
| Stage 1 / P3.1b-P3.3 | `000d056bf2` | Masked GEMV, ready-first demand overlap, bounded multi-wave execution; 331 focused tests, 104 forced-fallback cases, source model and actual overlap trace passed |
| Stage 2 / P4.1a | `02ec1d5e1c` | Read-only static MoE chain candidates; 16 targeted integration, 339 focused and 104 forced-fallback tests passed |
| Stage 2 / P4.1b | `0555339301` | Qualify constant single-token Router/FC descriptors; 345 focused and 104 forced-fallback tests passed |
| Stage 2 / P4.2-P4.3 | `64ed628a5a` | Opt-in independent proxy Top-k, recall diagnostics and worker prefetch; 354 focused and 112 forced-fallback cases passed |

P1.3 validation used the source Debug/OCL build on Intel Arc Lunar Lake with
assertions and existing numerical tolerances unchanged. All listed runs had
zero failures and skips; the subsets overlap and are not additive. The fallback
run disabled both `OV_GPU_MOE_USE_GROUPED_GEMM_PREFILL` and
`OV_GPU_MOE_USE_MICRO_GEMM_PREFILL`. The complete focused run is not the full GPU
unit suite, and no new ZE LES or real-model performance claim is made.

## P2.1a: Strict OCL External Dependencies

The opt-in `ocl_stream::enqueue_external_barrier(deps)` queues a device barrier
for individual native OpenCL events, independently of profiling and the stream's
normal synchronization mode. Unlike the legacy dependency conversion, it rejects
null dependencies, missing native handles, grouped events, foreign contexts and
events already known to have failed. All dependencies are validated before the
barrier is enqueued. An empty list is a no-op, not an all-queue wait.

Callers must explicitly flush producer queues so their events can progress.
Returning from this method means only that the barrier was enqueued. Subsequent
commands cannot consume the uploaded data before its dependencies complete;
execution errors arising later must still be observed through a real completion
event. There is no Host wait or compute-queue finish in this method.

Grouped events are rejected because their representative native handle does not
prove completion across independent producer queues. P2 upload aggregation must
provide the individual weight, scale and zero-point completion events instead.
The caller remains responsible for source-memory, staging and slot lifetimes.

P2.4 production LES demand uses this API with every returned upload event.
The default LES-off execution path remains unchanged. ZE adaptation is not
implemented.

Validation on the source Debug/OCL Lunar Lake build passed 12 cross-queue event
tests, including the two original native controls, and 215 complete focused
tests. Both runs had zero failures and skips; the subsets overlap. The external
visibility checks split the input between two independently gated producer
queues and use both in-order and out-of-order compute queues, with profiling
on and off. They also cover missing handles, grouped native events, context
mismatch, known errors and errors arising after barrier enqueue. No sleeps or
queue finish are used to establish visibility. This is not the full GPU suite,
ZE validation or a performance/overlap result.

## P2.1b: Transfer-Lane Ownership

Each eligible network creates a separate in-order transfer stream owned by its
LES context. The context remains backend-neutral; its CPU-only tests do not
require GPU headers or libraries. Destruction joins the worker before releasing
the stream. A GPU task must drain every upload it issues before returning or
unwinding, so stop/join also covers its staging and device work. The generic Host
task API itself cannot infer outstanding DMA from a callback.

The current asynchronous LES backend is explicitly OCL-only. LES-on eligible ZE
networks fail construction rather than using unverified event semantics; LES-off
and resident/non-MoE networks retain the existing path. A gated real-upload test
checks that context destruction waits for active device work, retains the stream
and source data, and releases the stream only after completion. The 27 integrated
tests also include the existing LES-on numerical checks. No dedicated copy-engine
selection or overlap performance is implied by creating a separate queue.

## P2.2: Slot Reservations

`ExpertSlotReservations` is separate from the legacy synchronous LRU. It checks
the entire deduplicated demand before changing state, pins every hit before
assigning misses and returns immediately when capacity or active consumers block
the request. Reservation does no I/O. Tickets carry epoch/id and internal slot
generations; late completion and duplicate release cannot affect a reused slot.
Only explicit completion marks reserved contents ready. Releasing an incomplete
ticket discards its pending entries, while ready entries remain cached.

Waiting for upload/consumer events must happen outside the reservation mutex;
the caller must drain old DMA before releasing its ticket. Five new tests plus
26 existing LRU/provider tests passed. P2.4 integrates this state machine with
upload completion and real consumer events; metadata alone is not a GPU fence.

## P2.3: Bounded Staging And Upload Completion

`fill_weights_memory_async` is a worker-side pipeline, not a nonblocking Host
return: it uses nonblocking copies while reading/repacking subsequent data, then
drains every issued upload before returning or propagating an error. Its returned
events cover every uploaded weight, scale and zero-point component. The worker
must open/use its positional reader itself; execution threads do not lazy-open it.

The entire read/slot plan is checked before the first copy. Two reusable 8 MiB
staging buffers chunk large weight tensors; a buffer is reused only after its
previous event completes. Transposed scale/ZP components must fit 8 MiB each per
expert. Including the existing packed-u4 transpose temporaries, transient staging
is bounded conservatively by 64 MiB per active worker. Oversized metadata is
rejected explicitly; the legacy synchronous path is unchanged.

Six upload tests and eight existing provider tests passed: all nine components
with profiling on/off, source preservation across staging reuse, preflight
validation, missing files, short reads and a later short read after an earlier
DMA has already been submitted. LES worker/chunk/byte counters are separate from
legacy synchronous tensor-load and timing counters. No device performance claim
is inferred from these counters.

## P2.4: Production Demand Integration

The offload provider reserves the complete demand before dispatching work. Its
worker owns positional file reads and staging; all nine W/S/ZP components and
every chunk must complete before the compute barrier and GPUReady transition.
A real compute-queue marker retains the last reservation until its consumer
finishes, independently of profiling. Consumer waits occur outside reservation
locks. Failed or cancelled queued demands release incomplete reservations;
active tasks drain uploads before cleanup and the next epoch can recover.

LES clones each provider's mutable reader/cache state per network and allocates
network-private, per-layer OTD buffers. The buffers use the existing partial
physical-slot/full-logical-layout convention. Original compiled constant
allocations still exist: private buffers add persistent memory beyond the
64 MiB transient staging bound. This is not a cross-layer pool or an unchanged
memory-footprint claim. Requests exceeding capacity use the original per-expert
fallback; no split arithmetic, changed tolerances or removed legacy waits are
introduced.

Final Debug/OCL validation on Lunar Lake passed 303 tests in 25 suites and
80 LES-on forced-fallback cases, with zero failures/skips. The overlapping
subsets include 18 upload/provider cases and 80 LES-on numerical cases spanning
U4/U8, 1/4/48 tokens, 1/2/4 slots, profiling on/off, both routing modes, changing
routes, independent networks and missing-file recovery. Gated native upload,
consumer and queued-cancellation tests exercise lifetime and retry behavior.
This is not the full GPU unit suite. A fresh standalone CPU worker TSan run
could not start (`unexpected memory mapping`); the earlier P1 TSan result is
historical, not a P2 provider/driver race-safety result.

The source Qwen3.6-35B VLM smoke passed with ratio 80, LES enabled, one stream,
21 input tokens and 8 generated tokens. Core, GPU/CPU plugins, IR frontend,
GenAI and tokenizers were checked against their source-build paths. Counters
recorded 2,930 worker tasks, 29,349 upload chunks, 5,329,465,344 upload bytes and
zero legacy synchronous tensor loads. Experiment evidence remains outside this
repository in the workspace `les/results/p2_source_les_on_ratio80_env/` directory.
The first attempt preserved in a separate directory failed because the
experimental entry point passed an internal option through the public API;
the successful run uses its supported environment variable. This single short
smoke is not model-accuracy, throughput, overlap or speedup evidence. Eligible
LES-on ZE execution remains explicitly unsupported.

## P3.1a: Canonical Pairs And Nonblocking Demand

Reservation tickets now snapshot `ready_pairs` and `missing_pairs` while holding
the metadata lock. Indices are original routed `token * top_k + rank` positions,
not compacted ranks, unique experts or shared-expert positions. Both lists keep
canonical order and cover the entire request, including repeated experts across
tokens. A pair belongs to the ready list only if its slot was already ready at
reservation time; later completion does not rewrite this historical snapshot.
Hits are pinned before any miss receives a slot, as in P2.

`begin_les_demand` reserves and dispatches missing uploads, then returns slot and
pair snapshots without waiting for uploads. It may still wait for previous
consumers or bounded worker queue capacity; it is not a generally wait-free API.
`finish_les_demand` observes worker errors, drains all uploads through the worker
contract, enqueues the strict native-event barrier and marks the ticket ready.
It must be called once for a successful demand. A second finish is rejected.
Capacity rejection returns no demand and dispatches no work. The P2 synchronous
acquisition is now a begin/finish compatibility wrapper.

Releasing an unfinished demand waits for its worker before releasing pins;
its pending cache entries are discarded rather than committed. Error cleanup
also releases resources after waiting and propagates the worker error. Provider
destruction waits for an outstanding callback before destroying reader/config
state captured through the provider. Explicit finish, context wait or release
remain responsible for reporting worker errors; destruction does not throw.
Consumers must still be fenced by the caller before a reservation is released.

Validation exhausts all 256 Top-8 cold/hot masks using two differently ordered
tokens with duplicate experts. Three native-gated provider tests check begin
returning before upload completion, preserved hit data and canonical mapping,
finish blocking, repeated-finish rejection, unfinished release/retry and active
upload destruction. Final Debug/OCL evidence: 24 reservation/LRU tests,
21 upload/provider tests, 307 complete focused tests and 80 LES-on forced-fallback
cases passed with zero failures/skips. Subsets overlap. A test-fixture file/layout
size mismatch was corrected without weakening production range validation.

At the P3.1a milestone, production kernels still used the P2 all-ready wrapper.
Its 307-test and P2 real-model evidence must not be interpreted as validation
of split GPU execution. The remaining P3 integration is described below.

## P3.1b-P3.3: Ready-First GEMV And Multi-Wave Execution

LES-off GEMV retains its original kernels and arguments. Eligible LES-on GEMV
uses separate masked gate/up and down variants. A zero mask makes the entire
workgroup return before reading weights or scratch; selected workgroups keep
the original arithmetic and write the original token/rank scratch positions.
The original reduction runs exactly once after all routed pairs finish. Shared
expert work is selected only in the first wave's Pass A, once per token.

After the real routing dependencies complete, the Host reads Top-k without the
old unconditional compute-queue finish. A wave selects at most the slot capacity
in unique experts and includes every still-pending pair of each selected expert.
Reservation pins hits before assigning misses; missing reads/repacking/uploads
start on the worker. Pass A runs the already-ready pairs while this work proceeds.
Finish observes upload completion and installs the strict barrier; Pass B runs
only missing pairs. Both passes write canonical positions, independent of order.

Each wave requests a real kernel completion event, flushes compute and waits for
that consumer before releasing temporary slot/mask buffers or reusing slots.
This is a narrow event wait, not a steady-state whole-queue finish. All-cold or
all-hot demands skip the empty routed pass. Capacity-one execution makes forward
progress through bounded waves rather than switching to a different accumulation
kernel. Exceptions drain compute and worker activity before releasing buffers;
cleanup retains the original error. The initial per-network ownership finish
and legacy larger-Prefill waits remain; this is not a claim that every finish
was removed from the plugin.

Final Debug/OCL verification on Lunar Lake passed 331 tests in 27 suites and
104 LES-on forced-fallback cases, with zero failures/skips. The eight Top-8 GPU
cases each exhaust all 256 masks with U4/U8, 1/4 tokens and both routing modes;
outputs are compared bitwise to the original unsplit GEMV. Existing LES matrix
cases also compare bitwise for GEMV-sized inputs, including 1/2/4-slot multi-wave
execution and profiling off/on. Sixteen shared-expert cases compare bitwise with
unsplit shared GEMV across U4/U8, 1/4 tokens, 1/4 slots and profiling off/on.
Mixed mask 0x55 additionally injects a short read after warm-up, then checks
next-epoch recovery bitwise. Subsets overlap; this is not the full GPU suite.
The shared fixture's router layout was corrected to [batch, seq, experts] after
its old four-dimensional layout produced only one token of Top-k. No production
range guard or numerical tolerance was weakened.

`MOE_LES_TRACE=1` enables opt-in JSON interval logging in debug-capable builds
with `OV_VERBOSE=1`. Both queues must have profiling enabled. Positional-read
intervals sample the OpenCL device timer before/after the read; Pass A/B kernels
use native profiling timestamps on that same device. Missing timer/profiling
support is an error, not a fabricated Host-enqueue interval. Trace is off by
default and perturbs timings; reads may be served from page cache. It demonstrates
read/compute concurrency, not cold-SSD DMA or dedicated-copy-engine throughput.

Diagnostic records now include network/layer IDs, LES epoch, token count,
chunk/wave, a process-local demand ID, expert ID, physical slot, component and
byte count. `hit`/`miss` records describe actual cache reservations. Worker
`read` and native-event `copy` records share those identifiers, so each demand
transfer can be joined to its preceding miss. `prefetch_hit`/`prefetch_miss`
describe speculative target-cache reservations separately; speculative copies
must not be counted as actual-demand misses.

`moe_span` uses queue markers and includes host dispatch gaps, compilation and
waits. Trace-only markers wait at the end of each MoE and perturb scheduling.
`upload_join_window` describes the host future wait, not GPU execution.
Legacy synchronous `read_sync` and `copy_sync_window` describe blocking host
calls in the device-clock domain; a blocking copy need not return a native
event. Async `copy` and observed kernels use actual native START/END timestamps.
Prefill `compute_tail` covers only the final observed wave kernel; oneDNN GEMMs
are not completely instrumented. No observed kernel records does not mean no
GPU execution. Phase labels are inferred: one token is Decode, more than 32 is
Prefill, and 2..32 is batched. Tracks represent logical roles, not OS thread IDs.

Workspace `les/analyze_les_timeline.py` exports CSV, separate Prefill/Decode
Markdown tables, summary JSON and a Perfetto-compatible trace. It validates
miss-before-transfer associations and unions intervals before measuring
observed overlap. These diagnostic samples are separate from trace/counters-off
performance runs. The historical branch-base runtime and current LES-off runtime
are distinct comparison rows; switching LES off is not a pre-LES baseline.

The trace GPU probe observed 148 read/Pass-A interval intersections. Actual source
Qwen ratio-80, 21-input/8-output-token generation passed and observed 437 such
intersections, with no synthetic delay. Its counters recorded 361 waves, 1,659
ready pairs, 7,301 missing pairs, 29,457 uploads and zero legacy tensor loads.
Path/hash provenance, trace and generation reports remain outside this repository
under workspace `les/results/p3_source_les_trace_ratio80/`.

Matched trace-off source runs used ratio 80, 8 output tokens, one warm-up and
three measured runs per LES setting. All output text matched. Mean wall generation
was 3.330858 s off versus 2.534071 s on; mean TTFT was 2.536180 s versus 1.427640 s,
but mean TPOT regressed from 0.113407 s to 0.157922 s. These short sequential Debug
runs have uncontrolled page cache and are not a general speedup guarantee. LES
remains default-off; Decode overhead needs performance tuning. ZE remains
unsupported, larger-Prefill split and cross-layer prediction/pooling are not P3,
and the previously documented TSan environment limitation is unchanged.

## P4.1a: Static Chain Candidates Only

`program::get_les_moe_links()` queries the program dependency graph and returns
value descriptors containing source/target MoE IDs, the source's hidden-input ID
(`proxy_hidden`), the target's hidden-input ID, and a status. It does not infer
causality from execution order or node names. LES-off and internal programs
return no descriptors. Callers must query the completed graph; results are
snapshots, not references to mutable graph nodes or live runtime bindings.

Discovery starts at each MoE's hidden input, not its routing or weight inputs.
It traverses ordinary dependencies, stops at MoEs and constant data, and rejects
loop/condition boundaries. It removes older MoE ancestors exposed by residual
bypasses before testing uniqueness. Multiple independent nearest predecessors
are ambiguous; a predecessor reaching multiple next MoEs is branched, including
when another successor is ambiguous. Neither case is a usable chain candidate.
Missing hidden dependencies are explicitly unsupported, and a chain root has
`no_predecessor`. IDs are returned in deterministic target-ID order, not layer
execution order. No graph edges, output tensors, memory plans or cache bytes
are changed. The query stores no graph pointers and runs only when requested.

`candidate` means topological adjacency only, not permission to launch prediction.
It does not validate Router weights/bias/routing semantics, OTD eligibility,
single-token shapes or hidden-buffer lifetime. No runtime consumer, proxy-router
kernel or speculative I/O is enabled by this step. Unsupported graphs remain on
the existing asynchronous-demand path. The current algorithm conservatively
rejects branches rather than guessing a chain, and may return no useful candidate
on more complex models.

Eight graph tests cover ordinary-layer/residual gaps, nearest-predecessor
selection, independent chains, merges, forks, Router-side dependencies, a loop
boundary, missing hidden inputs, LES-off and non-MoE graphs. Existing real MoE
fixtures also query their optimized program and retain all original numerical
and bitwise checks. Final Debug/OCL verification passed 339 tests in 28 suites
and the overlapping 104-case forced-fallback subset, zero failures/skips; the
16-case targeted run is included, not additive. The plugin was relinked. The
residual probe first failed as intended and passed after ancestor filtering.

P4 proxy execution is implemented for the documented plain-FP16, single-token
projection boundary. It is not a general predictor for arbitrary Router graphs:
compressed/quantized FC, fused FC operations, unsupported layouts and malformed
chains remain demand-only. Router descriptors are re-derived from the restored
program graph rather than serialized. A two-layer synthetic supported proxy
program was saved and loaded, then rerun with the restored proxy stage; descriptors,
prediction, actual route and bitwise output checks passed in all proxy variants.
OTD partial constants now serialize an explicit physical-byte marker with no
device readback; import reconstructs a resident-sized physical backing and full
logical view. Qwen cold export/warm import parity also passed in demand-only mode.
Formal performance benefit on a model with a supported proxy chain remains
unverified. Full-suite, ZE and TSan limits remain unchanged.

P4.1b adds `program::get_les_router_links()` as a stricter, read-only
qualification query. It accepts only a matching Router output-0/1 pair,
Softmax or SigmoidBias routing, a plain noncompressed/nonfused transposed FC
projection, constant data parameters, and static single-token proxy/target
hidden layouts. It returns IDs for the Router, FC, weights and optional
projection/routing bias and epsilon. It verifies descriptor input ports,
dimensions and element types. Unsupported shapes, variable parameters, and
malformed Router/projection graphs receive explicit statuses; no parameters are
copied, no kernel launched, and no existing route modified. Qualification is not
cached or serialized and must be queried from the final compiled program.

Fourteen targeted graph cases cover chain topology plus both routing modes,
optional FC bias, nonconstant weights, incorrect Router ports, unrelated Top-k
producers and multi-token rejection. Debug/OCL verification passed 345 focused
tests and 104 forced-fallback tests, zero failures/skips; the plugin linked.

## P4.2-P4.3: Independent Prediction And Optional Prefetch

`OV_GPU_LES_PREDICT=1` opts in to a plain-FP16 proxy FC and the existing Router
Top-k kernel. The source MoE copies its input activation to network-owned memory
with the upstream dependency event; the snapshot survives activation reuse and
is cleared at each LES epoch. The proxy uses dedicated logits, indices and
weights buffers. It never writes actual Router outputs or changes actual Top-k,
routing weights or MoE compute. Router descriptors are derived from the loaded
program graph; the proxy kernel stage is restored through the ordinary primitive
implementation cache path. Data cache records use the separate partial-OTD marker
documented above.

`OV_GPU_LES_PREDICT_COMPARE=1` waits for proxy and real Router IDs and records
Top-k rank and expert membership matches. This mode is diagnostic and synchronizes Host readback; it is
off by default. `OV_GPU_LES_PREFETCH=1` separately opts into worker-side loading
for predicted experts. It uses the bounded LES worker, reservation and transfer
lane, skips demands that cannot fit, releases speculative pins after upload, and
lets real demand reserve and validate the actual experts. A speculative read or
upload failure drains issued DMA, discards the incomplete reservation and does
not poison the request epoch; real demand retries via the normal path.
Unsupported links never enqueue prediction and remain demand-only. A proxy
kernel may still be compiled when prediction is requested, because final graph
qualification can differ from intermediate compilation-time qualification.

The diagnostic comparison also emits `[LES_PREDICTION]` JSON with `network`,
`layer`, `epoch`, `tokens`, `rank_matches`, `expert_matches`, `pairs`,
`predicted_ids` and `actual_ids`. The ID arrays are token-major, Top-k rank order.
`rank_matches / pairs` compares identical Top-k slots; `expert_matches / pairs`
compares membership in the actual Top-k of the same token, independent of rank.
Absent comparisons are unavailable, not zero recall. The extra computation and
logging occur only inside the existing default-off comparison branch, after
its already-required readbacks; no production fence or readback is added.
Enabled instrumentation has overhead and must be disabled for latency A/B.
Existing `[LES_TRACE]` hit/miss and native copy/pass records can be aggregated
offline by layer/epoch; asynchronous submission alone is not overlap proof.
Trace records additionally carry the emitting C++ `host_thread` ID. Host
`read`/`repack` records identify the copy worker; Host route/reservation/wait
windows identify the inference control thread. A thread emitting a GPU event
record is not the GPU execution thread or a hardware engine identifier.

For large logs, run the workspace analyzer with `--summary-only`:

```bash
python les/analyze_les_timeline.py /absolute/path/verbose.txt \
  --summary-only --output les/results/new_les_report
```

Use a new output directory. `cache_layers.csv` separates demand and speculative
lookup statistics by network/layer/phase. `predictions.csv` correlates each
network/layer/epoch with ranked prediction/actual arrays when available,
observed demand expert IDs and actual cache-miss IDs. `expert_cache_events.csv`
retains every lookup and its original file/line. `summary.json` and `summary.md`
include independent rank hit and expert recall, thread witnesses and positive
copy/kernel overlap examples with original timestamps and source lines.
Overlap examples use bounded samples, not exhaustive coverage percentages.
Missing ID arrays or thread IDs in old logs are reported as unavailable. Negative
intervals retain counters, are reported explicitly and are excluded from
overlap evidence. Dedicated hardware copy-engine use requires driver/native
engine profiling; two queues or a memcpy API alone cannot prove it.

The parser runs offline without modifying inference. New formatting/counters
remain inside existing opt-in comparison/trace branches; no additional production
readback, wait or kernel is introduced. Enabled comparison/trace still has real
overhead and must be disabled for performance comparisons.

Eight GPU variants cover Softmax/SigmoidBias, FC bias on/off, prediction-only and
prefetch. The prediction tests observe 0/8, 1/8, 3/8 and 8/8 Top-k matches. They
check valid IDs, preserve actual Router output and compare final target outputs
bitwise with LES-off. Repeated execution verifies snapshot/event/ID refresh.
Provider tests inject a speculative short read, check that the epoch remains
usable, then confirm demand loads the repaired file. Final Debug/OCL verification
passed 354 focused tests in 29 suites, 112 overlapping LES-on fallback cases and
all eight proxy variants, zero failures/skips. The plugin linked. In the prefetch
variant the complete program was serialized and loaded into a new network; restored
descriptors and proxy stage executed, and the final output remained bitwise equal
to the LES-off baseline.

Source Qwen ratio 80 generated successfully with prediction and prefetch options
set. All 39 adjacent chains were rejected because the Router FC uses compressed
weights; zero proxy comparisons ran. This confirms conservative demand-only
fallback, not prediction coverage on Qwen. No formal speedup for a supported
model is claimed; speculative prefetch remains opt-in and its latency benefit is
unestablished.

For compiled-model cache validation, `LES_CACHE_DIR` is optional in
`run_source_baseline.sh` and is forwarded as OpenVINO `CACHE_DIR`. Qwen ratio 80
with LES enabled, prediction/prefetch requested and a fresh cache completed a
cold export and a second warm import; both generated the same text and 21/2 input/
output token counts. The cache occupied 2.2 GiB. Since this Qwen graph has no
supported proxy chain, this validates OTD partial-constant serialization and
demand-only imported execution, not imported proxy-stage execution. The dedicated
`cache_serialization` suite passed 10/10, including the new zero-payload OTD
partial allocation roundtrip. Cache timing is not a performance result.

## P4.4: Eligible Real-Model Functional Acceptance

Qualification accepts an unfused reshape data-edge chain only when it resolves
to the exact target hidden node/output. Reshape cannot authorize a different
hidden source or an intervening computation. Proxy, target and FC hidden layouts
must retain a static feature width matching the FP16 constant matrix. Leading
dimensions may be dynamic only if they admit one token. Static multi-token and
dynamic-width inputs remain unsupported. At execution, snapshots are taken only
from contiguous FP16/bfyx single-token buffers; a multi-token Router does not
launch proxy kernels. Prefill continues through the original demand path.

The workspace experiment `les/prepare_p4_router_ir.py` materialized 40 Router
matrices of the existing Qwen IR as FP16, leaving compressed expert payloads
unchanged. It requires exact equality to the original represented weights; it
does not restore their pre-quantization values or implement compressed-FC proxy
execution. Product compressed/fused FC guards remain unchanged.

On the OCL Debug host, `les/run_p4_acceptance.py` passed six sequential modes:
LES-off, demand-only, prediction, prefetch, cold compiled cache and warm reuse.
Each used one warm-up plus two measured generations of eight tokens. All 18
generated texts and token counts agreed. Each enabled prediction mode qualified
39 chains, compared 6,552 expert slots and matched 2,133 slots. Prefetch modes
recorded 6,552 prefetched slots and zero failures. Counts correspond to decode
only; the 21-token prefill did not run predictions. Warm reuse retained the five
cache blobs unchanged and still executed predictions. Experiment evidence is in
workspace `les/results/p4_acceptance_20261102/acceptance.json`.

Focused GPU tests passed 357/357, and the overlapping forced-fallback subset
passed 112/112 with grouped/micro prefill explicitly disabled. New boundaries
cover same-source reshape acceptance, different-source rejection, dynamic tokens
with fixed width and unknown-width rejection. Eight proxy variants retain strict
bitwise LES-off final-output checks, including repeated execution and synthetic
program import. Real-model text/count parity is not logits-bitwise accuracy.

This closes the eligible-real-model P4 functional gate, not a performance gate.
Measured Debug generation means were 2.014619 s demand-only, 2.111490 s prediction
and 2.404974 s prefetch: speculation regressed, rather than demonstrating speedup.
Page cache was uncontrolled and diagnostic comparison readback was enabled.
Prefetch remains blocking at the target Router; no P5 early scheduling, P6/P7
support, general GPU suite, ZE LES or new TSan result is claimed.

## P4.5: OLMoE Export And Selected-Weight Sum

Softmax Router fusion permits additional consumers of the selected-weight
ReduceSum. It replaces normalized routing and indices but preserves the original
TopK/value-sum computation for other consumers. The replaced old indices tensor
names are cleared because the original TopK may remain live. A transformation
regression checks that the extra sum output survives; all 34 Router fusion tests
passed. This does not add a native non-normalizing fused Router mode.

The workspace-only OLMoE adapter exports tiled three-MatMul experts. For
`norm_topk_prob=false`, it uses normalized Top-k and multiplies the expert sum by
the original selected-weight sum. This is mathematically equivalent, not a
bitwise-preserving arithmetic rewrite. FP16 tracing must read the real stashed
Linear parameters, not the frontend's zero FP32 placeholders. Tiny FP32 logits
and two FP16 stateful prefill/decode/reset sequences passed PyTorch comparisons.
Final expert weights are INT4 with all 16 Router matrices kept as FP16 constants.

On OCL Debug, the complete OLMoE model passed six modes with one warm-up and two
measured eight-token generations each: off, demand, prediction, prefetch, cold
compiled cache and warm reuse. All 18 texts/counts agree. Fifteen chains executed
2,520 prediction slot comparisons per enabled mode, with 885 ordered slot matches;
prefetch modes recorded 2,520 slots and zero failures. Warm reuse left its single
blob unchanged and predictions active. Load time was 7.469 s cold / 1.179 s warm.
Evidence: workspace `les/results/olmoe_p4_acceptance2_20261102/acceptance.json`.

Measured generation means were 1.131499 s demand, 1.216670 s prediction and
1.231448 s prefetch. Prefetch uploaded 8,302,264,320 bytes versus demand's
7,890,419,712 bytes across three generations. These short diagnostic runs show
neither useful-byte efficiency nor performance acceptance. Full-model PyTorch
logit/quality validation, long contexts, the full GPU suite, ZE LES and fresh
TSan remain unverified. Default VLM/Qwen smoke also passed after runner changes.

## P5.1: Demand-Priority Worker Foundation

`les_context::submit` remains the required-demand API. Demand tasks retain FIFO
order within their class and run before queued speculative tasks. If the bounded
pending queue is full, demand may displace a not-yet-started speculative task;
its ticket completes with `les_task_cancelled`. Already active work is not
preempted, and its uploads still drain before buffers can be reused.

`try_submit_speculative` never waits for queue space. It returns an empty optional
when the queue is full or invalidating, while stale/stopped epochs remain errors.
Speculative failures complete their own ticket without poisoning required work.
Required-demand failures preserve the previous context-wide error propagation.
Epoch rotation, cancellation, shutdown, queue capacity and the single worker
contract are unchanged. Prefetch now uses the speculative API, releases its
reservation on admission refusal and treats policy cancellation as a skip rather
than an upload failure.

Focused OCL Debug tests passed 396/396 with zero failures/skips. New deterministic
tests cover demand-before-speculation, per-class FIFO, full-queue displacement,
nonblocking prefetch refusal, speculative error isolation and stale-epoch drops.
The provider refusal test checks unchanged GPU bytes and a successful subsequent
real demand. Evidence: workspace `les/p5_foundation_results.xml`.

OLMoE six-mode regression also passed with all 18 texts/counts equal, 15 qualified
chains, 2,520 compared slots per prediction mode, 2,520 prefetched slots per
prefetch mode, zero prefetch failures and unchanged warm cache blobs. Evidence:
workspace `les/results/p5_1_olmoe_acceptance_20261102/acceptance.json`.

At the P5.1 checkpoint this was only the scheduling foundation: prefetch still
blocked at the target Router and rejection did not restore replaced mappings.
The remaining P5 implementation and validation follow below.

## P5: Early, Bounded Speculation And Pollution Accounting

The source MoE snapshots its qualified single-token hidden input and invokes the
next Router's proxy stage before submitting source expert compute. The proxy uses
fixed single-token scratch, not the unexecuted target's dynamic input/output
buffers. Actual routing and expert arithmetic are unchanged. The Router reuses
the existing proxy result rather than predicting twice. The small prediction and
ID readback still have a Host wait; weight reads/repacking/uploads run on the
worker and no longer block the source until their completion. This enables
overlap but is not a new measured device-overlap or speedup claim.

`LesSpeculationPolicy` keeps `lookahead_layers=1`. The Router reads the complete
predicted top-k, and each provider keeps at most one speculative batch. There
is no four-expert, capacity-quarter or 16 MiB admission limit. Reservations
deduplicate prediction IDs and protect already-ready predicted hits from eviction;
only missing experts are uploaded, in prediction rank order. If capacity is
insufficient, the reservation accepts the highest-ranked misses that fit in
unpinned slots rather than rejecting the whole batch. In-flight experts are
not uploaded again and active demand slots are never evicted. Capacity one can
admit one missing expert; zero-byte metadata still disables uploads. This is
capacity-bounded speculation, not a guarantee of better throughput: larger
batches can increase DMA wait and cache pollution. No adaptive or trained
predictor is required.

Matching pending expert requests reuse the original task. A different request
does not create another batch. Real demand promotes a matching queued task to
required FIFO priority instead of uploading again, or cancels an unneeded queued
task. Active uploads cannot be preempted; demand may wait for that bounded batch.
Epoch cancellation drains active DMA, drops pending work and allows retry.

Speculation may replace only unpinned LRU slots; real active reservations are
never victims. Predicted hits do not refresh demand recency. A task rejected or
cancelled before any upload rolls back replaced metadata because device bytes
are still unchanged. Once any DMA has occurred, rollback is forbidden: cleanup
waits for all issued transfers, invalidates unused predictions and releases pins.
Useful predicted slots remain ready for actual demand. This is bounded soft
protection, not a guarantee of zero cache pollution. Independent demand-only
shadow reservations track the same successful demand sequence and capacity,
without prediction or any device allocations.

### MiniCPM Six-Mode Benchmark References

The workspace runner `les/run_minicpm_six_modes.py` defaults to the following
six modes. Reference versions are validated in the inference worker, not just
in archive filenames. OV23634 is not this comparison's reference: it has the
known dynamic-shape `get_dims()` failure for this model.

| Mode | Runtime/archive | LES | Predict | Prefetch |
| --- | --- | --- | --- | --- |
| LES base | `les/wheels/les-base/`, OV23745 | 0 | 0 | 0 |
| LES proxy base | `les/wheels/proxy-base-937/`, OV23659 / commit9379271c1e | 0 | 0 | 0 |
| Current off | Latest native build or archived current wheels | 0 | 0 | 0 |
| Current demand | Same current runtime | 1 | 0 | 0 |
| Current predict | Same current runtime | 1 | 1 | 0 |
| Current prefetch | Same current runtime | 1 | 1 | 1 |

Keep MiniCPM-V4.7, the original four-question81-token prompt, GPU/PA and
`MOE_LES_DECODE_SINGLE_PASS=0`. Use one common offload ratio, starting at60;
only an allocation/resource failure justifies a fresh full comparison at70.
Performance uses warmup plus three512-token rounds without trace/profiling.
Separate full-generation diagnostics collect phase/layer cache, prediction
and device-interval evidence; their timings must not replace clean performance
numbers. Missing intervals or inactive predictions are N/A, not zero.

Counters are cumulative and require `MOE_OTD_PERF_LOG=1`:

- `les_early_predictions`: proxy predictions created from the source hook.
- `les_prefetch_experts`: admitted missing experts, not proof of completed DMA.
- `les_prefetch_useful_bytes`: successfully uploaded missing experts used by the
  next actual demand batch.
- `les_prefetch_useless_bytes`: submitted speculative bytes not useful above,
  including partial failed transfers. Cancelled unstarted work contributes zero.
- `les_prefetch_late_bytes`: useful bytes whose upload future was not ready when
  actual demand arrived; this is a subset of useful bytes, not an extra total.
- `les_prefetch_extra_misses`: actual missing experts that hit in the demand-only
  shadow cache, measured on successful demand completion.
- `les_prefetch_deduplicated`, `les_prefetch_promoted`, `les_prefetch_skipped`:
  matching pending requests, queued-task promotions and policy/admission skips.

Final Lunar Lake OCL Debug validation passed 405/405 focused tests in 32 suites
and the overlapping 182/182 forced grouped/micro-Prefill-fallback subset, with
zero failures/skips. Deterministic gated tests cover priority/promotion,
duplicate versus different requests, cancellation, rollback, full-cache
speculation, pollution attribution and single-upload reuse. The queued-prefetch
epoch recovery check also passed 20 repeated runs. Evidence is in workspace
`les/p5_complete_final_results.xml`, `les/p5_complete_final_fallback_results.xml`
and `les/p5_epoch_final_results.xml`.

OLMoE LLM and Qwen VLM each passed all six modes (off/demand/predict/prefetch/
cache-cold/cache-warm), 18 repeated texts/token counts equal per model, with warm
cache blobs unchanged. The prefetch runs recorded respectively 315/819 early
predictions, 213/702 admitted experts, 676,601,856/1,026,342,912 useful bytes,
19,611,648/120,938,496 useless bytes, 22,880,256/34,320,384 late bytes and 26/34
extra misses; prefetch failures were zero. All 12 mode results passed the P5
early-execution and byte-budget/accounting gates. Evidence is in workspace
`les/results/p5_complete_olmoe_20261102/acceptance.json` and
`les/results/p5_complete_qwen_20261102/acceptance.json`; the existing acceptance
script now offers `--p5` and positive/negative parser self-tests.

P5 is complete within the existing plain-FP16, single-token, OCL qualification
boundary. LES remains default-off. These short sequential Debug samples do not
establish formal performance, model/logit accuracy, complete GPU-suite coverage,
ZE LES support or a new TSan result. P0 gaps remain deferred; larger-Prefill
overlap, cross-layer pools and the full compatibility matrix remain P6-P8.

## P6: Bounded Multi-Wave Prefill

LES-on OCL offloaded MoE execution now uses the original per-expert, grouped
oneDNN or micro-GEMM arithmetic path for inputs above the batched-GEMV threshold.
It partitions actual unique experts into waves, retaining original token/rank
pair indices. Grouped/micro wave outputs scatter into canonical pair scratch;
one final weighted reduction runs in original rank order. The per-expert path
keeps ascending-expert index-add order rather than accumulating wave partial
sums. Shared expert execution occurs once per token chunk, after routed output.

Grouped selection preserves the legacy OTD capacity rule: actual unique experts
across the entire Prefill, not configured expert count or per-wave count, must
fit the original resident capacity. Otherwise LES uses per-expert arithmetic
inside bounded waves, preserving fused routing weights and ascending-expert
accumulation. Sparse routing that fits still uses grouped waves. Micro-GEMM
selection is unchanged. No cross-path bitwise equivalence is assumed.

The per-expert path, including the grouped capacity fallback, keeps the full
token set for each expert. Splitting it into 128-token chunks changes oneDNN's
matrix M and can select different FP16 arithmetic than LES-off. Expert waves
retain ascending-expert accumulation and banked weight loading; shared expert
work runs after all routed waves. The last consumer event protects bank reuse,
and each routed expert uploads at most once per Prefill. Grouped/micro wave
computation and Decode scheduling are unchanged.

Per-expert scratch scales with the full token count. Micro scratch remains
limited to 128 tokens; grouped scratch is conservatively full-token because a
capacity miss can select the per-expert fallback. CPU routing metadata retains
the whole Prefill Top-k index array and wave pairs, proportional to tokens times
Top-k. Prefill tests check the selected path, pair/chunk counts, path-dependent
scratch limits and expert-upload bounds. Per-expert and grouped capacity
fallbacks require bitwise output agreement with same-capacity LES-off controls.

The two banks partition the existing resident slot count, including odd counts;
they do not each receive a full-capacity allocation. Capacity one advances
serially. Bank providers return absolute slots in the same weight allocation.
Current-wave completion permits next-bank worker upload while compute proceeds;
consumer events protect reuse. Exceptions drain issued compute and uploads
before cleanup. Switching to bank execution invalidates the main provider's
cache metadata, preventing later Decode from treating overwritten data as ready.

Grouped/micro wave computation processes tokens in chunks of at most 128 and
reuses gather, gate/up/down, routing metadata and canonical scratch across chunks.
Per-expert computation keeps one full-token chunk for numerical compatibility.
Full input/output tensors remain full-sized, and small layout-dependent metadata
still scales with the configured expert count. There is no universal token-bound
scratch contract, global fixed-byte ceiling or cross-layer pool. Empty inputs keep
the existing non-Prefill path; the wave planner accepts empty routes without
fabricating work. Large-prefill constants still have the existing P2/P5 private
allocation overhead, which P7 must address.

The original P6 milestone's focused tests passed 541/541 in 37 suites and its overlapping forced
grouped/micro fallback subset passed 258/258 in 14 suites, zero failures/skips.
Coverage includes U4/U8, both routing modes, profiling off/on, odd and one-slot
capacities, 33/65/257-token execution, a partial final chunk, changing routes,
shared experts, missing/truncated files after earlier uploads and next-epoch
recovery. Tests require actual per-path wave/chunk counters and bounded scratch.
Per-expert/grouped and shape-matched micro outputs are compared bitwise with
same-path full-capacity controls. Micro versus the old unchunked shape is also
checked with the original numerical tolerance: chunking can change micro-kernel
floating-point rounding by one ULP, so unchunked bitwise parity is not claimed.

After the MiniCPM accuracy repair, 138 focused GPU cases passed: 130 per-expert
and grouped cases, including H=2048/intermediate=512/group=128 geometry, plus
eight correct/wrong prediction and prefetch proxy cases. Eleven transformation
tests passed, including special-zero reshapes across dynamic token lengths.
The real 1031-input/512-output MiniCPM workload passed two Off and two Demand
runs with identical token IDs and text. Separate prefill/first-decode captures
have bitwise-identical first-layer MoE outputs and full-vocabulary logits.
This is not a rerun of the historical 541-case suite or activation of prediction
for MiniCPM compressed Router weights, which remain unqualified.

OLMoE LLM and Qwen VLM six-mode acceptance passed with respectively 267/272
input tokens, eight output tokens, 18 equal repeated outputs per model, active
Prefill counters and unchanged warm cache blobs. Evidence is in workspace
`les/p6_complete_results.xml`, `les/p6_complete_fallback_results.xml` and
`les/results/p6_complete_{olmoe,qwen}_20261102/acceptance.json`.
The native micro trace probe observed 51 read/kernel interval intersections
across 918 reads and 225 kernel events on one device timer. This is real
read/compute overlap, not cold-storage DMA throughput or end-to-end speedup.

P6 is functionally complete within the existing OCL fused-MoE boundary. LES stays
default-off. Performance requires a separate matched comparison; short Debug
acceptance timings do not establish a general benefit. ZE LES, other GPUs,
cross-layer physical sharing, the independent GroupedMatMul graph adapter and
the complete compatibility matrix remain outside this milestone.

### P6 Performance Follow-Up

A separate counterbalanced Qwen run on DUT4420LNL used Debug/OCL, ratio 80,
272 input tokens and 32 generated tokens. Each mode has six measured samples
after process-local warmup; trace and comparison were disabled. A later audit
found that setting `MOE_OTD_PERF_LOG=0` still enabled counters because the old
implementation tested variable presence. These historical timings are not
instrumentation-off measurements. The switch now requires exactly `1`, and the
benchmark runner removes the variable entirely for old/new binary comparisons.
All repeated and cross-mode texts/token counts matched. Mean wall generation,
TTFT and TPOT were respectively:

| Mode | Wall (s) | TTFT (s) | TPOT (s/token) |
|---|---:|---:|---:|
| LES off | 10.781 | 7.107 | 0.11847 |
| Demand | 12.260 | 7.991 | 0.13767 |
| Prefetch | 12.804 | 8.118 | 0.15113 |

Demand and prefetch were 13.7% and 18.8% slower in wall time. This establishes
no speedup for this workload; uncontrolled clocks, temperature and page cache
make it exploratory rather than a Release performance gate. Load time is
separate. Raw samples and standard deviations are in workspace
`les/results/p6_perf_qwen_20261102/performance.json`. The validated OLMoE IR was
missing from its external mount at that checkpoint, so no OLMoE performance
result was available then. It has since been rebuilt under workspace
`les/models/olmoe-p4-int4`: 48 FP16 expert tensors / 3,072 nonzero expert slices,
16 uncompressed Router projections, and long-input six-mode P5/P6 acceptance.
Workspace evidence is
`les/results/olmoe_rebuilt_20261102T143535Z_31992/acceptance.json`.

### Performance Investigation And Metadata Reuse

The instrumented Qwen and rebuilt OLMoE three-generation acceptance runs
recorded 25,014/38,940 and 4,804/8,637 off/demand expert misses respectively.
Token-chunk-outer Prefill traversal revisits experts with undersized banks;
these totals include Decode and do not assign an exact wall-time fraction to
Prefill. Per-component staging copies/events/flushes, Router readback and
ready/missing consumer waits also remain. Prediction adds snapshots and proxy
readback, and prefetch can add useless bytes and cache pollution. P7's memory
pool does not by itself remove these costs.

Decode now reuses three shape-matched USM Host metadata buffers per
implementation, cleared on clone. Existing blocking copies and consumer waits
still protect reuse. `les_decode_metadata_allocations` is available with
`MOE_OTD_PERF_LOG=1`. Changing-route, independent-network, multi-wave and recovery
tests require at most three initial allocations and zero later ones for fixed
shapes. Directed validation passed 337/337 and full focused regression passed
547/547, no failures/skips; the source GPU plugin rebuilt. Workspace evidence
is `les/perf_metadata_results.xml` and `les/perf_fix_full_results.xml`.
The diagnostic report is workspace `les/PERFORMANCE_RCA.md`; local allocation
reuse is not claimed to eliminate the Prefill reload/synchronization regression.

The corrected, genuinely counters-off Qwen four-mode comparison passed six
measured samples per mode: off/demand/predict/prefetch wall means were
11.363/13.387/13.715/14.029 s. Demand/Prefetch remained 17.8%/23.5% slower than
off. Against the old counters-off Demand baseline, metadata reuse changed wall
time only 0.19%, within sample variation. TPOT mean decreased 7.13%, but this
independent-process exploratory result does not establish a causal speedup.

Rebuilt OLMoE's default-path 32-token benchmark failed cross-mode text parity
at Demand, despite internally stable outputs and correct token counts. Legacy
OTD grouped execution falls back to per-expert when actual unique experts exceed
capacity; LES instead executes grouped waves. Forcing Demand to per-expert
restored text parity, localizing the observed divergence to path choice, not
proving rounding rather than a kernel/layout defect. The earlier eight-token
six-mode acceptance does not establish this longer-generation default gate.
That historical failure is addressed by the capacity-selection correction below;
output checks were not relaxed.

A separate explicit per-expert four-mode OLMoE control passed 32-token text/count
parity with wall means 4.938/6.786/7.133/6.842 s, Demand/Prefetch 37.4%/38.5%
slower than off. This is not a default-path PASS. Workspace reports are
`les/results/perf_rca_qwen_20261102/performance.json`,
`les/results/perf_rca_olmoe_per_expert_20261102/performance.json` and the failed
default driver `les/perf_rca_olmoe_driver.log`. All 16 successful processes used
one verified GPU binary, ratio 80, 32 output tokens and no counter variable.

### Default-Path Accuracy Follow-Up

A fixed-input 48-token MoE regression at the original resident capacity reproduced
16 bitwise failures against LES-off before the fix. LES had selected grouped
waves where legacy OTD selected per-expert arithmetic. Whole-Prefill capacity
selection now preserves that legacy rule. This fixes a path-compatibility
regression, not proof that all grouped differences are harmless rounding.
Cross-path intermediate/logit equivalence is not certified.

Grouped tests passed 56/56, including sparse-fit and full-capacity grouped
counters, overflow per-expert counters, U4/U8, both Routers, changing routes and
repeated bitwise legacy comparisons. Shared-expert tests passed 12/12 with
same-arithmetic full-capacity controls and unchanged original numerical checks.
Full focused regression passed 571/571 in 37 suites, zero failures/skips.
Evidence is workspace `les/accuracy_*_results.xml`.

The source plugin passed the previously failing default OLMoE workload: 267 input
tokens, 32 output tokens, eight counterbalanced processes and 32 identical
warmup/measured texts and counts. No grouped/micro override or counter variable
was set. GPU SHA-256 was
`9a29560e619e3af15dac3b2cae8a7fe4259411c6f61ffd30533add49cdd2180d`.
Evidence is `les/results/accuracy_olmoe_default_20261102/performance.json`.
Off/Demand/Predict/Prefetch wall means were 5.360/7.036/7.172/7.319 s;
Demand/Prefetch remain 31.3%/36.6% slower. This closes the tested accuracy gate,
not an acceleration claim or full-model quality certification. P7/P8 remain open.

## P7.1: Physical Pool Foundation (Not Enabled)

`ExpertWeightMemoryPool` adds an OCL-engine-bound, mutex-serialized allocator
with an explicit physical-payload byte ceiling. Exact layout classes include
all nine optional W/S/ZP layouts, expert count and slot count. Layouts must be
static, dense, byte-addressable bfyx. Same-sized different types/shapes and
different optional components cannot reuse each other's allocations.

Physical allocations retain slot-sized layouts; provider-facing views retain
the full logical expert dimension. Logical view `size()` is not physical
allocation accounting. Budget admission counts the retained physical payload,
not driver overhead, staging, activations or existing per-layer constants.
Exact completed layout classes are reused first. Under budget pressure, released
blocks with completed consumer fences can be reclaimed for a different layout;
the new layout is allocated separately, never reinterpreted from an incompatible
class. Active leases and incomplete consumers cannot be reclaimed. Admission
checks total eligible bytes before eviction, so an unsatisfiable request leaves
existing blocks intact and returns no lease without waiting for consumers.

Leases have pool ownership and unique IDs. Callers must explicitly release
every lease with a native consumer event covering all upload/compute uses from
the same OCL context. Dropping a lease does not release it. Stale or foreign
leases cannot release a newer owner. Released blocks cannot be reused until
their consumer completes; destruction drains released consumers. The engine
must outlive the pool, and all active leases/uses must be released before pool
destruction. Returned views must not be used after release. `fence_held_bytes`
conservatively counts retained retired fences until reuse, even if completed.

Five new native-OCL tests cover all-component accounting, exact-class isolation,
budget/owner/stale-ID rejection, foreign-context fences, destructor draining
and concurrent admission. Final directed validation passed 43/43; the complete
focused run passed 546/546 in 37 suites, zero failures/skips. The actual GPU
plugin also rebuilt successfully. Workspace evidence is
`les/p7_pool_final_results.xml` and `les/p7_foundation_full_results.xml`.

The subsequent heterogeneous-admission regression first reproduced the completed
incompatible-layout blockage, then passed with reclamation. All six pool tests
and 26 provider tests pass, and the source plugin rebuilt. Evidence is workspace
`les/p7_reclaim_{repro,pool,provider}_results.xml`. This admission extension is
still unenabled in production and is not network/budget integration.

This foundation is not connected to production network allocations and adds
no runtime switch. P7 remains incomplete: network-scoped ownership/budget,
provider binding and cache-generation invalidation, removal of private
per-layer allocations, cross-network/stream boundaries and the independent
GroupedMatMul adapter still require implementation and validation. No deployed
cross-layer sharing or model-level memory reduction is claimed.