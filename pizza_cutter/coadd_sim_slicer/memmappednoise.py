import tempfile
import os
import numpy as np


class MemMappedNoiseImage(object):
    """A memory mapped image on disk of noise.

    Parameters
    ----------
    seed : int
        The seed for the RNG.
    weight : array-like
        The weight map for the noise field.
    sx : int, optional
        Size of patches to generate the noise in the x direction.
    sy : int, optional
        Size of patches to generate the noise in the y direction.
    """

    def __init__(self, *, seed, weight, sx=1000, sy=1000):
        rng = np.random.RandomState(seed=seed)

        # generate in chunks so that we don't use a ton of memory
        try:
            # if it is a FITS hdu, then this is right
            nx, ny = weight.get_dims()
        except Exception:
            # try a numpy array otherwise
            nx, ny = weight.shape
        self.shape = (nx, ny)

        n_sx = nx // sx
        n_sy = ny // sy
        assert n_sx * sx == nx
        assert n_sy * sy == ny

        # we use a memmapped array on disk to avoid memory usage
        self._temp = tempfile.TemporaryDirectory()
        self._fname = os.path.join(self._temp.name, 'arr.dat')
        self._noise = np.memmap(
            self._fname,
            dtype=np.float32,
            mode='w+',
            shape=(nx, ny),
            order='C')

        for isx in range(n_sx):
            for isy in range(n_sy):
                xl = isx * sx
                yl = isy * sy

                wgt = weight[xl:xl+sx, yl:yl+sy]
                self._noise[xl:xl+sx, yl:yl+sy] = rng.normal(
                    size=(sx, sy)) * np.sqrt(1.0 / wgt)
        self._noise.flush()

    def __getitem__(self, slices):
        return self._noise[slices]

    def get_dims(self):
        return self.shape
