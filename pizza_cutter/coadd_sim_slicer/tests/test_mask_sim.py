import numpy as np
import pytest

from ..mask_sim import apply_bmask_symmetrize_and_interp


@pytest.mark.parametrize('symmetrize_masking', [True, False])
def test_apply_bmask_symmetrize_and_interp(symmetrize_masking):
    seed = 10
    rng = np.random.RandomState(seed=seed)
    y, x = np.mgrid[0:100, 0:100]
    image = (10 + x*5).astype(np.float32)
    weight = np.ones_like(image)
    noise = rng.normal(size=image.shape)
    noise_for_noise_interp = rng.normal(size=image.shape)

    se_interp_flags = 8
    noise_interp_flags = 16
    bmask = np.zeros_like(image, dtype=np.int32)
    bmask[60:70, 20:40] = 16
    bmask[20:25, 60:65] = 8

    weight[:, -10] = 0.0

    (interp_image, sym_weight,
     sym_bmask, interp_noise) = apply_bmask_symmetrize_and_interp(
        se_interp_flags=se_interp_flags,
        noise_interp_flags=noise_interp_flags,
        symmetrize_masking=symmetrize_masking,
        noise_for_noise_interp=noise_for_noise_interp,
        image=image,
        weight=weight.copy(),
        bmask=bmask.copy(),
        noise=noise)

    if symmetrize_masking:
        assert not np.array_equal(sym_weight, weight)
        assert np.all(sym_weight[:, -10] == 0.0)
        assert np.all(sym_weight[9, :] == 0.0)
        # make sure the rest are ones
        assert np.sum(sym_weight) == 100 * 100 - 199

        assert not np.array_equal(sym_bmask, bmask)
        assert np.all((sym_bmask[60:70, 20:40] & 16) != 0)
        assert np.all((sym_bmask[60:80, 60:70] & 16) != 0)

        assert np.all((sym_bmask[20:25, 60:65] & 8) != 0)
        assert np.all((sym_bmask[35:40, 20:25] & 8) != 0)

        msk = (sym_bmask & noise_interp_flags) != 0
        assert np.allclose(interp_image[msk], noise_for_noise_interp[msk])

        msk = (sym_bmask & se_interp_flags) != 0
        assert np.array_equal(interp_image[msk], image[msk])

        msk = (sym_bmask & se_interp_flags) != 0
        assert not np.array_equal(interp_noise[msk], noise[msk])
    else:
        assert np.array_equal(sym_bmask, bmask)
        assert np.array_equal(sym_weight, weight)

        msk = (sym_bmask & noise_interp_flags) != 0
        assert np.allclose(interp_image[msk], noise_for_noise_interp[msk])

        msk = (sym_bmask & se_interp_flags) != 0
        assert np.array_equal(interp_image[msk], image[msk])

        msk = (sym_bmask & se_interp_flags) != 0
        assert not np.array_equal(interp_noise[msk], noise[msk])
