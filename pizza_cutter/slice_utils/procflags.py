# a full edge is masked, preventing interpolation
FULL_EDGE_MASKED = 2**0

# slice has flags that indicate it is a complete no-go
SLICE_HAS_FLAGS = 2**1

# high masked frac
HIGH_MASKED_FRAC = 2**2

# too much trail not masked by bright star mask
HIGH_UNMASKED_TRAIL_FRAC = 2**3

# second check of masking in interp
HIGH_INTERP_MASKED_FRAC = 2**4
