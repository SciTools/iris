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
import numpy as np
import cartopy.crs as ccrs

import iris
import iris.plot as iplt
import iris.quickplot as qplt


@iris.tests.skip_data
class TestLoad(tests.GraphicsTest):
    
    def test_load(self):
        cube = iris.load(tests.get_data_path(('NIMROD', 'uk2km', 'WO0000000003452',
                        '201007020900_u1096_ng_ey00_visibility0180_screen_2km')))[0]
        self.assertCML(cube, ("nimrod", "load.cml"))
        
        ax = plt.subplot(1,1,1, projection=ccrs.OSGB())
        c = qplt.contourf(cube, coords=["x", "y"], levels=np.linspace(-25000, 6000, 10))
        ax.coastlines()
        self.check_graphic()

    def test_orography(self):
        # Load visibility data and make it look like an orography field.
        # Don't bother with the coords, they're pretty much identical.
        viz_file = tests.get_data_path((
                        'NIMROD', 'uk2km', 'WO0000000003452',
                        '201007020900_u1096_ng_ey00_visibility0180_screen_2km'))
        
        import iris.fileformats.nimrod
        with open(viz_file, "rb") as infile:
            field = iris.fileformats.nimrod.NimrodField(infile)
            
        field.dt_year = field.dt_month = field.dt_day = field.int_mdi 
        field.dt_hour = field.dt_minute = field.int_mdi
        field.proj_biaxial_ellipsoid = 0
        field.tm_meridian_scaling = 0.999601
        field.field_code = 73
        field.vertical_coord_type = 1
        field.title = "(MOCK) 2km mean orography"
        field.units = "metres"
        field.source = "GLOBE DTM"
        
        cube = field.to_cube()
        self.assertCML(cube, ("nimrod", "orography.cml"))
        

if __name__ == "__main__":
    tests.main()
