import pytest
from ..locate import build_slice_locations


def test_build_slice_locations():
    row, col, start_row, start_col = build_slice_locations(
        central_size=20, buffer_size=10, image_width=100)

    for i in range(4):
        for j in range(4):
            ind = j + 4*i
            assert row[ind] == i*20 + 10 + 9.5
            assert col[ind] == j*20 + 10 + 9.5
            assert start_row[ind] == i * 20
            assert start_col[ind] == j * 20


def test_build_slice_locations_error():
    with pytest.raises(ValueError):
        build_slice_locations(
            central_size=15, buffer_size=10, image_width=100)
