import numpy as np
import pytest

from ..mask_sim import (
    apply_bmask_symmetrize_and_interp,
    interpolate_ngmix_multiband_obs)
from ngmix.observation import ObsList, MultiBandObsList, Observation


@pytest.fixture
def ngmix_data():
    n_bands = 3
    n_obs = 2
    seed = 42
    se_interp_flags = 8
    noise_interp_flags = 16

    rng = np.random.RandomState(seed=seed)

    mbobs = MultiBandObsList()
    for b in range(n_bands):
        obslist = ObsList()
        for o in range(n_obs):
            index = o + n_obs * b

            y, x = np.mgrid[0:100, 0:100]
            image = (index + x*5).astype(np.float32)
            weight = np.ones_like(image)
            noise = rng.normal(size=image.shape)
            bmask = np.zeros_like(image, dtype=np.int32)
            bmask[60+index:70+index, 20+index:40+index] = noise_interp_flags
            bmask[20+index:25+index, 60+index:65+index] = se_interp_flags
            weight[:, -10-index] = 0.0

            obs = Observation(
                image=image,
                weight=weight,
                bmask=bmask,
                noise=noise)
            obslist.append(obs)

        mbobs.append(obslist)

    return {
        'mbobs': mbobs,
        'se_interp_flags': se_interp_flags,
        'noise_interp_flags': noise_interp_flags,
        'n_bands': n_bands,
        'n_obs': n_obs}


@pytest.mark.parametrize('symmetrize_masking', [True, False])
def test_interpolate_ngmix_multiband_obs(symmetrize_masking, ngmix_data):
    seed = 42

    interp_mbobs = interpolate_ngmix_multiband_obs(
        mbobs=ngmix_data['mbobs'],
        se_interp_flags=ngmix_data['se_interp_flags'],
        noise_interp_flags=ngmix_data['noise_interp_flags'],
        symmetrize_masking=symmetrize_masking,
        rng=np.random.RandomState(seed=seed))

    assert len(interp_mbobs) == ngmix_data['n_bands']

    rng = np.random.RandomState(seed=seed)
    index = 0
    for interp_obslist, obslist in zip(interp_mbobs, ngmix_data['mbobs']):
        assert len(interp_obslist) == ngmix_data['n_obs']

        for interp_obs, obs in zip(interp_obslist, obslist):
            nse = rng.normal(size=obs.image.shape)

            if symmetrize_masking:
                assert not np.array_equal(interp_obs.weight, obs.weight)
                assert np.all(interp_obs.weight[:, -10-index] == 0.0)
                assert np.all(interp_obs.weight[9+index, :] == 0.0)
                # make sure the rest are ones
                assert np.sum(interp_obs.weight) == 100 * 100 - 199

                assert not np.array_equal(interp_obs.bmask, obs.bmask)
                assert np.all(
                    (interp_obs.bmask[60+index:70+index, 20+index:40+index]
                     & ngmix_data['noise_interp_flags']) != 0)
                assert np.all(
                    (interp_obs.bmask[60-index:80-index, 60+index:70+index] &
                     ngmix_data['noise_interp_flags']) != 0)

                assert np.all(
                    (interp_obs.bmask[20+index:25+index, 60+index:65+index] &
                     ngmix_data['se_interp_flags']) != 0)
                assert np.all(
                    (interp_obs.bmask[35-index:40-index, 20+index:25+index] &
                     ngmix_data['se_interp_flags']) != 0)

                msk = (
                    interp_obs.bmask & ngmix_data['noise_interp_flags']) != 0
                assert np.allclose(interp_obs.image[msk], nse[msk])

                msk = (interp_obs.bmask & ngmix_data['se_interp_flags']) != 0
                assert np.allclose(interp_obs.image[msk], obs.image[msk])

                msk = (interp_obs.bmask & ngmix_data['se_interp_flags']) != 0
                assert not np.allclose(
                    interp_obs.noise[msk], obs.noise[msk])

            else:
                assert np.array_equal(interp_obs.bmask, obs.bmask)
                assert np.array_equal(interp_obs.weight, obs.weight)

                msk = (
                    interp_obs.bmask & ngmix_data['noise_interp_flags']) != 0
                assert np.allclose(interp_obs.image[msk], nse[msk])

                msk = (interp_obs.bmask & ngmix_data['se_interp_flags']) != 0
                assert np.allclose(interp_obs.image[msk], obs.image[msk])

                msk = (interp_obs.bmask & ngmix_data['se_interp_flags']) != 0
                assert not np.allclose(
                    interp_obs.noise[msk], obs.noise[msk])

            index += 1


@pytest.mark.parametrize('symmetrize_masking', [True, False])
def test_apply_bmask_symmetrize_and_interp(symmetrize_masking):
    seed = 10
    rng = np.random.RandomState(seed=seed)
    y, x = np.mgrid[0:100, 0:100]
    image = (10 + x*5).astype(np.float32)
    weight = np.ones_like(image)
    noise = rng.normal(size=image.shape)
    nse = np.random.RandomState(seed=seed+1).normal(size=image.shape)

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
        rng=np.random.RandomState(seed=seed+1),
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
        assert np.allclose(interp_image[msk], nse[msk])

        msk = (sym_bmask & se_interp_flags) != 0
        assert np.array_equal(interp_image[msk], image[msk])

        msk = (sym_bmask & se_interp_flags) != 0
        assert not np.array_equal(interp_noise[msk], noise[msk])
    else:
        assert np.array_equal(sym_bmask, bmask)
        assert np.array_equal(sym_weight, weight)

        msk = (sym_bmask & noise_interp_flags) != 0
        assert np.allclose(interp_image[msk], nse[msk])

        msk = (sym_bmask & se_interp_flags) != 0
        assert np.array_equal(interp_image[msk], image[msk])

        msk = (sym_bmask & se_interp_flags) != 0
        assert not np.array_equal(interp_noise[msk], noise[msk])
