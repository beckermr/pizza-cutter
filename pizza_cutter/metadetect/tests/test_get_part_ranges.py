import pytest

from ..run_metadetect import _get_part_ranges


@pytest.mark.parametrize("size, n_parts", [
    (1532, 10),
    (1, 1),
    (13, 1),
    (10, 10),
    (100, 1000),
    (4343, 4000)
])
def test_get_part_ranges(size, n_parts):
    max_num = -1
    min_num = size
    tot = 0
    for i in range(n_parts):
        start, num = _get_part_ranges(i+1, n_parts, size)

        max_num = max([max_num, num])
        if num > 0:
            min_num = min([min_num, num])

        if i > 0:
            assert prev_start + prev_num == start  # noqa

        if i == 0:
            assert start == 0

        if i == n_parts-1:
            assert start + num == size

        prev_start = start  # noqa
        prev_num = num  # noqa
        tot += num

    assert tot == size
    assert max_num / min_num <= 2.0
