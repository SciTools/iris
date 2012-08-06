# (C) British Crown Copyright 2010 - 2012, Met Office
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
"""
Tests cube default palette.

"""

import iris.tests as tests

import matplotlib.cm as mcm

import iris.palette
import iris.plot as iplt
import iris.tests.stock


@iris.tests.skip_data
class TestCmapNorm(tests.IrisTest):
    def setUp(self):
        self.cube = iris.tests.stock.global_pp()

    def test_cmap_auto(self):
        iplt.contourf(self.cube)
        self.check_graphic()

    def test_cmap_default(self):
        iplt.contourf(self.cube, cmap=None)
        self.check_graphic()

    def test_cmap_override(self):
        # Diverging scheme.
        iplt.contourf(self.cube, cmap=mcm.get_cmap(name='BrBu_10'))
        self.check_graphic()
        # Other scheme.
        iplt.contourf(self.cube, cmap=mcm.get_cmap(name='StepSeq'))
        self.check_graphic()
        # Qualitative scheme.
        iplt.contourf(self.cube, cmap=mcm.get_cmap(name='PairedCat_12'))
        self.check_graphic()
        # Sequential scheme.
        iplt.contourf(self.cube, cmap=mcm.get_cmap(name='LBuDBu_10'))
        self.check_graphic()

    def test_norm_auto(self):
        self.cube.standard_name += '_anomaly'
        iplt.contourf(self.cube)
        self.check_graphic()

    def test_norm_default(self):
        self.cube.standard_name += '_anomaly'
        iplt.contourf(self.cube, norm=None)
        self.check_graphic()

    def test_norm_override(self):
        self.cube.standard_name += '_anomaly'
        norm = iris.palette.SymmetricNormalize(pivot=200)
        iplt.contourf(self.cube, norm=norm)
        self.check_graphic()


if __name__ == "__main__":
    tests.main()
    
