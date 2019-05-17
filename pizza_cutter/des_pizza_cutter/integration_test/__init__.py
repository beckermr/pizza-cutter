"""Code to run integration tests for the pizza cutter.

We are going to coadd a Gaussian blob with various WCS models and
masking effects.

Things the code does:

    1. We will define a few noise bits to check various code features

        SIM_BMASK_BAD            2**0  # images that have a bad pixel
        SIM_BMASK_SPLINE_INTERP  2**1  # pixels get spline interpolated
        SIM_BMASK_NOISE_INTERP   2**2  # pixels get noise interpolated

    2. We add outlier pixels that should get flagged at some rate.

    3. We add images with large masked fractions.

    4. We add images with image_flags set in the input.

    5. We make sure some images do not fully overlap the coadd region.

    6. We add images that would overlap except for the edge buffer.

Note that this test is not meant to cover every possible combination of paths
in the code, but to verify it works once.
"""
