import numpy as np

from ..symmetrize import (
    symmetrize_bmask,
    symmetrize_weight)


def test_symmetrize_weight():
    weight = np.ones((5, 5))
    weight[:, 0] = 0
    symmetrize_weight(weight=weight)

    assert np.all(weight[:, 0] == 0)
    assert np.all(weight[-1, :] == 0)


def test_symmetrize_weight_no_sym_mask():
    weight = np.ones((5, 5))
    weight[:, 0] = 0
    mask = np.ones_like(weight).astype(bool)
    mask[3:5, 0] = False
    mask = ~mask
    symmetrize_weight(weight=weight, no_sym_mask=mask)

    assert np.all(weight[:, 0] == 0)
    assert np.all(weight[-1, :3] == 0)
    assert np.all(weight[-1, 3:] == 1)


def test_symmetrize_weight_angle():
    weight = np.ones((64, 64))
    weight[:, 0] = 0
    weight_orig = weight.copy()
    weights = []
    for angle in [45, 90, 135, 180, 225, 270, 315]:
        weight_r = weight_orig.copy()
        symmetrize_weight(weight=weight_r, angle=angle)
        weights.append(weight_r)
        msk = weight_r == 0
        weight[msk] = 0

    assert np.array_equal(weight, np.rot90(weight))
    assert np.array_equal(weight, np.rot90(np.rot90(weight)))
    assert np.array_equal(weight, np.rot90(np.rot90(np.rot90(weight))))

    if False:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=3, ncols=3)
        axs = axs.ravel()
        weights.append(weight_orig)
        weights.append(weight)
        for i, bm in enumerate(weights):
            ax = axs[i]
            ax.pcolormesh(bm)
            ax.set_aspect(1)
        import pdb
        pdb.set_trace()


def test_symmetrize_weight_angle_no_sym_mask():
    weight = np.ones((64, 64))
    weight[:, 0] = 0
    weight_orig = weight.copy()

    mask = np.ones_like(weight).astype(bool)
    mask[55:, 0] = False
    mask = ~mask

    for angle in [90, 180, 270]:
        weight_r = weight_orig.copy()
        symmetrize_weight(weight=weight_r, angle=angle, no_sym_mask=mask)
        assert np.all(weight_r[:, 0] == 0)
        if angle == 90:
            assert np.all(weight_r[-1, :55] == 0)
            assert np.all(weight_r[-1, 55:] == 1)
        elif angle == 180:
            assert np.all(weight_r[-55:, -1] == 0)
            assert np.all(weight_r[:-55, -1] == 1)
        elif angle == 270:
            assert np.all(weight_r[0, -55:] == 0)
            # this test overalps with index [0, 0] that was weight zero start with
            assert np.all(weight_r[0, 1:-55] == 1)
        else:
            assert False, "missed an assert!"


def test_symmetrize_bmask():
    bmask = np.zeros((4, 4), dtype=np.int32)
    bmask[:, 0] = 1
    bmask[:, -2] = 2
    symmetrize_bmask(bmask=bmask)

    assert np.array_equal(
        bmask,
        [[1, 0, 2, 0],
         [3, 2, 2, 2],
         [1, 0, 2, 0],
         [1, 1, 3, 1]])


def test_symmetrize_bmask_sym_flags():
    bmask = np.zeros((4, 4), dtype=np.int32)
    bmask[:, 0] = 1
    bmask[:, -2] = 2
    bmask[0, -1] = 2**9
    symflags = (2**0) | (2**1)
    symmetrize_bmask(bmask=bmask, sym_flags=symflags)

    assert np.array_equal(
        bmask,
        [[1, 0, 2, 2**9],
         [3, 2, 2, 2],
         [1, 0, 2, 0],
         [1, 1, 3, 1]])

    bmask = np.zeros((4, 4), dtype=np.int32)
    bmask[:, 0] = 1
    bmask[:, -2] = 2
    bmask[0, -1] = 2**9
    symmetrize_bmask(bmask=bmask)

    assert np.array_equal(
        bmask,
        [[1 | 2**9, 0, 2, 2**9],
         [3, 2, 2, 2],
         [1, 0, 2, 0],
         [1, 1, 3, 1]])


def test_symmetrize_bmask_no_sym_mask():
    bmask = np.zeros((4, 4), dtype=np.int32)
    bmask[:, 0] = 1
    bmask[:, -2] = 2
    mask = np.ones_like(bmask).astype(bool)
    mask[2:4, 0] = False
    mask = ~mask

    symmetrize_bmask(bmask=bmask, no_sym_mask=mask)

    assert np.array_equal(
        bmask,
        [[1, 0, 2, 0],
         [3, 2, 2, 2],
         [1, 0, 2, 0],
         [1, 1, 2, 0]])


def test_symmetrize_bmask_no_sym_mask_sym_flags():
    bmask = np.zeros((4, 4), dtype=np.int32)
    bmask[:, 0] = 1
    bmask[:, -2] = 2
    bmask[0, -1] = 2**9
    symflags = (2**0) | (2**1)

    mask = np.ones_like(bmask).astype(bool)
    mask[2:4, 0] = False
    mask = ~mask

    symmetrize_bmask(bmask=bmask, no_sym_mask=mask, sym_flags=symflags)

    assert np.array_equal(
        bmask,
        [[1, 0, 2, 2**9],
         [3, 2, 2, 2],
         [1, 0, 2, 0],
         [1, 1, 2, 0]])


def test_symmetrize_bmask_angle():
    bmask = np.zeros((64, 64), dtype=np.int32)
    bmask[:, 0] = 1
    bmask[:, -2] = 2
    bmask_orig = bmask.copy()
    bmasks = []
    for angle in [45, 90, 135, 180, 225, 270, 315]:
        bmask_r = bmask_orig.copy()
        symmetrize_bmask(bmask=bmask_r, angle=angle)
        bmasks.append(bmask_r)
        bmask |= bmask_r

    assert np.array_equal(bmask, np.rot90(bmask))
    assert np.array_equal(bmask, np.rot90(np.rot90(bmask)))
    assert np.array_equal(bmask, np.rot90(np.rot90(np.rot90(bmask))))

    if False:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=3, ncols=3)
        axs = axs.ravel()
        bmasks.append(bmask_orig)
        bmasks.append(bmask)
        for i, bm in enumerate(bmasks):
            ax = axs[i]
            ax.pcolormesh(bm)
            ax.set_aspect(1)
        import pdb
        pdb.set_trace()


def test_symmetrize_bmask_angle_no_sym_mask_sym_flags():
    bmask = np.zeros((4, 4), dtype=np.int32)
    bmask[:, 0] = 1
    bmask[:, -2] = 2
    bmask[0, -1] = 2**9
    symflags = (2**0) | (2**1)

    mask = np.ones_like(bmask).astype(bool)
    mask[2:4, 0] = False
    mask = ~mask

    for angle in [90, 180, 270]:
        bmask_r = bmask.copy()
        symmetrize_bmask(
            bmask=bmask_r,
            angle=angle,
            no_sym_mask=mask,
            sym_flags=symflags,
        )
        if angle == 90:
            assert np.array_equal(
                bmask_r,
                [[1, 0, 2, 2**9],
                 [3, 2, 2, 2],
                 [1, 0, 2, 0],
                 [1, 1, 2, 0]])
        elif angle == 180:
            assert np.array_equal(
                bmask_r,
                [[1, 2, 2, 2**9],
                 [1, 2, 2, 0],
                 [1, 2, 2, 1],
                 [1, 2, 2, 1]])
        elif angle == 270:
            assert np.array_equal(
                bmask_r,
                [[1, 0, 3, 1 | 2**9],
                 [1, 0, 2, 0],
                 [3, 2, 2, 2],
                 [1, 0, 2, 0]])
        else:
            assert False, "missed an assert!"


def test_symmetrize_bmask_angle_no_sym_mask():
    bmask = np.zeros((4, 4), dtype=np.int32)
    bmask[:, 0] = 1
    bmask[:, -2] = 2

    mask = np.ones_like(bmask).astype(bool)
    mask[2:4, 0] = False
    mask = ~mask

    for angle in [90, 180, 270]:
        bmask_r = bmask.copy()
        symmetrize_bmask(bmask=bmask_r, angle=angle, no_sym_mask=mask)
        if angle == 90:
            assert np.array_equal(
                bmask_r,
                [[1, 0, 2, 0],
                 [3, 2, 2, 2],
                 [1, 0, 2, 0],
                 [1, 1, 2, 0]])
        elif angle == 180:
            assert np.array_equal(
                bmask_r,
                [[1, 2, 2, 0],
                 [1, 2, 2, 0],
                 [1, 2, 2, 1],
                 [1, 2, 2, 1]])
        elif angle == 270:
            assert np.array_equal(
                bmask_r,
                [[1, 0, 3, 1],
                 [1, 0, 2, 0],
                 [3, 2, 2, 2],
                 [1, 0, 2, 0]])
        else:
            assert False, "missed an assert!"


def test_symmetrize_bmask_angle_sym_flags():
    bmask = np.zeros((4, 4), dtype=np.int32)
    bmask[:, 0] = 1
    bmask[:, -2] = 2
    bmask[0, -1] = 2**9
    symflags = (2**0) | (2**1)

    for angle in [90, 180, 270]:
        bmask_r = bmask.copy()
        symmetrize_bmask(
            bmask=bmask_r,
            angle=angle,
            sym_flags=symflags,
        )
        if angle == 90:
            assert np.array_equal(
                bmask_r,
                [[1, 0, 2, 2**9],
                 [3, 2, 2, 2],
                 [1, 0, 2, 0],
                 [1, 1, 3, 1]])
        elif angle == 180:
            assert np.array_equal(
                bmask_r,
                [[1, 2, 2, 1 | 2**9],
                 [1, 2, 2, 1],
                 [1, 2, 2, 1],
                 [1, 2, 2, 1]])
        elif angle == 270:
            assert np.array_equal(
                bmask_r,
                [[1, 1, 3, 1 | 2**9],
                 [1, 0, 2, 0],
                 [3, 2, 2, 2],
                 [1, 0, 2, 0]])
        else:
            assert False, "missed an assert!"
