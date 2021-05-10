import time
from concurrent.futures import ProcessPoolExecutor
from ..caching import shared_lru_cache


@shared_lru_cache(maxsize=12)
def compute_value_delayed(val):
    time.sleep(5)
    return val**2


def test_shared_lru_cache_smoke():
    compute_value_delayed.cache_clear()
    with ProcessPoolExecutor(max_workers=4) as exec:
        tinit = time.time()
        fut = exec.submit(compute_value_delayed, 10)
        res = fut.result()
        tinit = time.time() - tinit
        assert res == 100

        time.sleep(1)
        tcached = time.time()
        fut = exec.submit(compute_value_delayed, 10)
        res = fut.result()
        tcached = time.time() - tcached
        assert res == 100
        assert tcached < tinit

        assert compute_value_delayed.cache_info().misses == 1
        assert compute_value_delayed.cache_info().hits == 1


def test_shared_lru_cache_clear():
    compute_value_delayed.cache_clear()
    with ProcessPoolExecutor(max_workers=4) as exec:
        tinit = time.time()
        fut = exec.submit(compute_value_delayed, 10)
        res = fut.result()
        tinit = time.time() - tinit
        assert res == 100

        compute_value_delayed.cache_clear()

        tcached = time.time()
        fut = exec.submit(compute_value_delayed, 10)
        res = fut.result()
        tcached = time.time() - tcached
        assert res == 100

        assert tinit > 5
        assert tcached > 5

        tagain = time.time()
        fut = exec.submit(compute_value_delayed, 10)
        res = fut.result()
        tagain = time.time() - tagain
        assert res == 100
        assert tagain < 1


def test_shared_lru_cache_hammer():
    compute_value_delayed.cache_clear()
    with ProcessPoolExecutor(max_workers=4) as exec:
        futs = [
            exec.submit(compute_value_delayed, 10)
            for _ in range(100)
        ]

        ttot = time.time()
        for fut in futs:
            assert fut.result() == 100
        ttot = time.time() - ttot
        assert ttot < 500

        assert compute_value_delayed.cache_info().misses < 5
        assert compute_value_delayed.cache_info().hits > 1
        curr_misses = compute_value_delayed.cache_info().misses + 0

        fut = exec.submit(compute_value_delayed, 10)
        fut.result()
        assert compute_value_delayed.cache_info().misses == curr_misses

        print(compute_value_delayed.cache_info(), flush=True)
