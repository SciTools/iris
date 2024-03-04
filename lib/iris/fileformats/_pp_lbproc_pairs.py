# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

import itertools

# LBPROC codes and their English equivalents
LBPROC_PAIRS = (
    (1, "Difference from another experiment"),
    (2, "Difference from zonal (or other spatial) mean"),
    (4, "Difference from time mean"),
    (8, "X-derivative (d/dx)"),
    (16, "Y-derivative (d/dy)"),
    (32, "Time derivative (d/dt)"),
    (64, "Zonal mean field"),
    (128, "Time mean field"),
    (256, "Product of two fields"),
    (512, "Square root of a field"),
    (1024, "Difference between fields at levels BLEV and BRLEV"),
    (2048, "Mean over layer between levels BLEV and BRLEV"),
    (4096, "Minimum value of field during time period"),
    (8192, "Maximum value of field during time period"),
    (16384, "Magnitude of a vector, not specifically wind speed"),
    (32768, "Log10 of a field"),
    (65536, "Variance of a field"),
    (131072, "Mean over an ensemble of parallel runs"),
)

# lbproc_map is dict mapping lbproc->English and English->lbproc
# essentially a one to one mapping
LBPROC_MAP = {
    x: y
    for x, y in itertools.chain(
        LBPROC_PAIRS, ((y, x) for x, y in LBPROC_PAIRS)
    )
}
