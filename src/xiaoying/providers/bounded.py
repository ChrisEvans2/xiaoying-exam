from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading
from xiaoying.domain import AppError

_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="provider")
_slots = threading.BoundedSemaphore(4)


def bounded(call, timeout):
    """Bound running provider calls as well as waiting; timed-out threads cannot enqueue unbounded work."""
    if timeout <= 0:
        raise AppError("EXECUTION_LIMIT", "执行时间预算耗尽", 504)
    if not _slots.acquire(blocking=False):
        raise AppError("PROVIDER_BUSY", "模型或工具处理资源正忙", 503, True)

    def work():
        try:
            return call()
        finally:
            _slots.release()

    future = _pool.submit(work)
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        # The slot remains reserved until the underlying call actually exits.
        raise AppError("TIMEOUT", "模型或工具调用超时", 504, True) from None
