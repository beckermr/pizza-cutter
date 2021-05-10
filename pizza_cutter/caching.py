"""
Shared LRU cache across processes.

Based on the cpython lru_cache and the approach of using a Manager here
https://raw.githubusercontent.com/basnijholt/adaptive-scheduler/master/adaptive_scheduler/utils.py
"""
import functools
from multiprocessing import Manager

from collections import namedtuple

LRUCacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


class HashedFunctionCall(list):
    """A sequence which computes the hash of a function's args and kwargs.

    This object is based on how Python's lru_cache works, but combines the
    munging of the args and kwargs with the hash calue comp.

    Parameters
    ----------
    args : tuple
        The tuple of args for the function call.
    kargs : dict
        The dict of kwargs for the function call.
    """
    __slots__ = ['hash']

    def __init__(self, args, kwargs):
        tup = tuple()
        tup += args
        if kwargs:
            tup += tuple(kwargs.items())
        self[:] = tup
        self.hash = hash(tup)

    def __hash__(self):
        return self.hash


def _shared_lru_cache_wrapper(function, maxsize):
    manager = Manager()
    cache = manager.dict()
    queue = manager.list()
    lock = manager.Lock()
    hits = manager.Value(int, 0)
    misses = manager.Value(int, 0)

    cache_get = cache.get
    cache_len = cache.__len__

    if maxsize == 0:
        def wrapper(*args, **kwargs):
            nonlocal misses
            misses.value += 1
            return function(*args, **kwargs)
    else:
        def wrapper(*args, **kwargs):
            nonlocal hits, misses

            key = HashedFunctionCall(args, kwargs)
            found = False
            with lock:
                if key in queue:
                    hits.value += 1
                    found = True
                    val = cache_get(key)
                    queue.remove(key)
                    queue.append(key)
                else:
                    misses.value += 1
            if found:
                return val
            val = function(*args, **kwargs)
            with lock:
                if key in queue:
                    pass
                elif cache_len() == maxsize:
                    key_to_evict = queue.pop(0)
                    cache.pop(key_to_evict)
                    queue.append(key)
                    cache[key] = val
                else:
                    queue.append(key)
                    cache[key] = val
            return val

    def cache_info():
        with lock:
            return LRUCacheInfo(hits.value, misses.value, maxsize, cache_len())

    def cache_clear():
        nonlocal hits, misses
        with lock:
            cache.clear()
            queue[:] = []
            hits.value = 0
            misses.value = 0

    wrapper.cache_info = cache_info
    wrapper.cache_clear = cache_clear

    return wrapper


def shared_lru_cache(maxsize=128):
    """Create a cache similar to `functools.lru_cache`, but shared across
    python processes generated by the multiprocessing module.
    """

    def cache_decorator(function):
        wrapper = _shared_lru_cache_wrapper(function, maxsize)
        return functools.update_wrapper(wrapper, function)

    return cache_decorator
