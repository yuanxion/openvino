#include "intel_gpu/runtime/les_context.hpp"

#include <algorithm>
#include <utility>

namespace cldnn {

les_context::les_context(size_t queue_capacity, std::shared_ptr<stream> transfer_stream)
    : _capacity(queue_capacity), _transfer_stream(std::move(transfer_stream)) {
    if (_capacity == 0 || _capacity > 1024) {
        throw std::invalid_argument("LES queue capacity must be in [1, 1024]");
    }
    _worker = std::thread([this] { run(); });
    _worker_id = _worker.get_id();
}

les_context::~les_context() {
    stop();
}

void les_context::check_control_thread() const {
    if (std::this_thread::get_id() == _worker_id) {
        throw std::logic_error("LES control operations cannot run on the copy worker");
    }
}

void les_context::fail_tasks(std::deque<task>& tasks, std::exception_ptr error) {
    for (auto& pending : tasks) {
        pending.completion.set_exception(error);
    }
    tasks.clear();
}

void les_context::invalidate(bool stopping) {
    std::deque<task> cancelled;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        ++_epoch;
        _stopping = _stopping || stopping;
        _invalidating = true;
        cancelled.swap(_pending);
    }
    _changed.notify_all();
    fail_tasks(cancelled, std::make_exception_ptr(les_task_cancelled{}));
    std::unique_lock<std::mutex> lock(_mutex);
    _changed.wait(lock, [this] { return !_active; });
    _invalidating = false;
    _changed.notify_all();
}

uint64_t les_context::begin_epoch() {
    check_control_thread();
    std::lock_guard<std::mutex> control(_control_mutex);
    invalidate(false);
    std::lock_guard<std::mutex> lock(_mutex);
    if (_stopping) {
        throw les_task_cancelled{};
    }
    _error = nullptr;
    return _epoch;
}

les_context::ticket les_context::submit(uint64_t epoch, std::function<void()> work) {
    return *submit_task(epoch, std::move(work), false);
}

std::optional<les_context::ticket> les_context::try_submit_speculative(uint64_t epoch, std::function<void()> work) {
    return submit_task(epoch, std::move(work), true);
}

std::optional<les_context::ticket> les_context::submit_task(uint64_t epoch, std::function<void()> work, bool speculative) {
    check_control_thread();
    if (!work) {
        throw std::invalid_argument("LES requires a nonempty copy task");
    }
    std::unique_lock<std::mutex> lock(_mutex);
    if (!speculative) {
        _changed.wait(lock, [this, epoch] {
            return _stopping || epoch != _epoch || _error ||
                   (!_invalidating && (_pending.size() < _capacity ||
                    std::any_of(_pending.begin(), _pending.end(), [](const task& pending) { return pending.speculative; })));
        });
    }
    if (_stopping || epoch != _epoch) {
        throw les_task_cancelled{};
    }
    if (_error) {
        std::rethrow_exception(_error);
    }
    if (speculative && (_invalidating || _pending.size() == _capacity)) {
        return std::nullopt;
    }
    std::optional<task> discarded;
    if (_pending.size() == _capacity) {
        auto candidate = std::find_if(_pending.begin(), _pending.end(), [](const task& pending) { return pending.speculative; });
        discarded.emplace(std::move(*candidate));
        _pending.erase(candidate);
    }
    task pending{epoch, speculative, std::move(work), {}};
    ticket result{epoch, pending.completion.get_future().share()};
    auto position = speculative ? _pending.end() :
        std::find_if(_pending.begin(), _pending.end(), [](const task& queued) { return queued.speculative; });
    _pending.insert(position, std::move(pending));
    lock.unlock();
    if (discarded) {
        discarded->completion.set_exception(std::make_exception_ptr(les_task_cancelled{}));
    }
    _changed.notify_all();
    return result;
}

void les_context::wait() {
    check_control_thread();
    std::unique_lock<std::mutex> lock(_mutex);
    _changed.wait(lock, [this] { return _pending.empty() && !_active && !_invalidating; });
    if (_error) {
        std::rethrow_exception(_error);
    }
}

void les_context::cancel() {
    check_control_thread();
    std::lock_guard<std::mutex> control(_control_mutex);
    invalidate(false);
}

void les_context::stop() {
    check_control_thread();
    std::lock_guard<std::mutex> control(_control_mutex);
    if (_worker.joinable()) {
        invalidate(true);
        _worker.join();
    }
}

void les_context::run() {
    for (;;) {
        task current;
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _changed.wait(lock, [this] { return _stopping || (!_invalidating && !_pending.empty()); });
            if (_stopping) {
                return;
            }
            current = std::move(_pending.front());
            _pending.pop_front();
            _active = true;
        }
        _changed.notify_all();
        std::exception_ptr error;
        try {
            current.work();
        } catch (...) {
            error = std::current_exception();
        }
        current.work = {};
        std::deque<task> failed;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_stopping || current.epoch != _epoch) {
                error = std::make_exception_ptr(les_task_cancelled{});
            } else if (error && !current.speculative) {
                _error = error;
                failed.swap(_pending);
            }
        }
        if (error) {
            current.completion.set_exception(error);
            fail_tasks(failed, error);
        } else {
            current.completion.set_value();
        }
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _active = false;
        }
        _changed.notify_all();
    }
}

}  // namespace cldnn