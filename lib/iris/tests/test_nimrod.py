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


# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import matplotlib.pyplot as plt

import iris
import iris.plot as iplt


@iris.tests.skip_data
class TestGribLoad(tests.GraphicsTest):
    
    def test_load(self):
        cube = iris.load(tests.get_data_path(('NIMROD', 'uk2km', 'WO0000000003452',
                        '201007020900_u1096_ng_ey00_visibility0180_screen_2km')))[0]
        self.assertCML(cube, ("nimrod", "load.cml"))
        
        plt.contourf(cube.data)
        self.check_graphic()
        
        # TODO: #84
#        iplt.contourf(cube)
#        iplt.gcm(cube).drawcoastlines()
#        self.check_graphic()
        
 

if __name__ == "__main__":
    tests.main()
