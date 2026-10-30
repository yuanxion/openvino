# LES Context: Stage 1 / P1

P1 introduces a default-off, network-local Host task service. It does not move
expert reads or uploads off the execution lane, create a transfer queue, or
change the MoE numerical path. Those changes belong to P2 and later steps.

## Configuration

- `GPU_ENABLE_LES`: internal boolean, default `false`.
- `GPU_LES_QUEUE_CAPACITY`: internal queued-task limit, default 32, range 1-1024.

These are experimental internal properties, not a public performance promise.
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
worker before releasing network resources. This does not add GPU stream waits
or asynchronous upload behavior. P1 does not dispatch weight work or add a new
public infer-request cancellation interface.

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
they do not fence DMA or permit slot/staging reuse. P2 must separately retain
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
| Stage 1 / P2.2 | This commit | 31 reservation/LRU/provider tests passed |

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

This is the first P2.1 substep, not asynchronous expert loading. ZE adaptation
and the P2.2-P2.4 reservation/upload pipeline are pending.
Production weight-loading and the default execution path do not yet call this API.

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
26 existing LRU/provider tests passed. Production provider integration follows
with the upload ticket in P2.3-P2.4; this state machine alone is not a GPU fence.