# LES Context: Stage 1 / P1-P3

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
| Stage 1 / P3.1b-P3.3 | This commit | Masked GEMV, ready-first demand overlap, bounded multi-wave execution; 331 focused tests, 104 forced-fallback cases, source model and actual overlap trace passed |

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