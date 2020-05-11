import functools
import time
import numpy as np


def get_stats(func, n=3, return_val=False):
    """Run func n times and show runtime stats.
    :return: tuple. In order, avg, std, min and max."""
    intervals = np.empty(n)
    stats = [np.mean, np.std, np.min, np.max]
    @functools.wraps(func)  # , intervals=intervals)
    def wrapper_timer(*args, **kwargs):
        for i in range(n):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            intervals[i] = run_time
        _stats = [f(intervals) for f in stats]
        if return_val:
            _stats.append(value)
        return(_stats)
    return(wrapper_timer)


def numba_get_stats(func, n=3, return_val=False):
    """Run func n times and show runtime stats.
    :return: tuple. In order, avg, std, min and max."""
    intervals = np.empty(n)
    stats = [np.mean, np.std, np.min, np.max]
    @functools.wraps(func)  # , intervals=intervals)
    def wrapper_timer(*args, **kwargs):
        value = func(*args, **kwargs)
        for i in range(n):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            intervals[i] = run_time
        _stats = [f(intervals) for f in stats]
        if return_val:
            _stats.append(value)
        return(_stats)
    return(wrapper_timer)


def timer(func):
    """Print runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer
