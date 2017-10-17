# (C) British Crown Copyright 2017, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

import itertools


# LBPROC codes and their English equivalents
LBPROC_PAIRS = ((1, "Difference from another experiment"),
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
                (131072, "Mean over an ensemble of parallel runs"))

# lbproc_map is dict mapping lbproc->English and English->lbproc
# essentially a one to one mapping
LBPROC_MAP = {x: y for x, y in
              itertools.chain(LBPROC_PAIRS, ((y, x) for x, y in LBPROC_PAIRS))}
