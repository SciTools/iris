# (C) British Crown Copyright 2014, Met Office
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
"""Unit tests for the :data:`iris.analysis.STD_DEV` aggregator."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import biggus
from iris.analysis import STD_DEV


class Test_lazy_aggregate(tests.IrisTest):
    def test_unsupported_mdtol(self):
        array = biggus.NumpyArrayAdapter(np.arange(8))
        msg = "unexpected keyword argument 'mdtol'"
        with self.assertRaisesRegexp(TypeError, msg):
            STD_DEV.lazy_aggregate(array, axis=0, mdtol=0.8)

    def test_ddof_one(self):
        array = biggus.NumpyArrayAdapter(np.arange(8))
        var = STD_DEV.lazy_aggregate(array, axis=0, ddof=1)
        self.assertArrayAlmostEqual(var.ndarray(), np.array(2.449489))

    def test_ddof_zero(self):
        array = biggus.NumpyArrayAdapter(np.arange(8))
        var = STD_DEV.lazy_aggregate(array, axis=0, ddof=0)
        self.assertArrayAlmostEqual(var.ndarray(), np.array(2.291287))


if __name__ == '__main__':
    tests.main()
