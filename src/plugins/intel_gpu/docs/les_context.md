# LES Context: Stage 1 / P1

P1 introduces a default-off, network-local Host task service. It does not move
expert reads or uploads off the execution lane, create a transfer queue, or
change the MoE numerical path. Those changes belong to P2 and later steps.

## Configuration

- `GPU_ENABLE_LES`: internal boolean, default `false`.
- `GPU_LES_QUEUE_CAPACITY`: internal queued-task limit, default 32, range 1-1024.

These are experimental internal properties, not a public performance promise.
The capacity excludes the running task and is not a staging-byte budget.

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