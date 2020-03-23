# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Integration tests for the print strings of a UGRID-based cube.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.cube import CubeList
from iris import Constraint
from iris.fileformats.netcdf import load_cubes


@tests.skip_data
class TestUgrid(tests.IrisTest):
    def setUp(self):
        file_path = tests.get_data_path(
            ("NetCDF", "unstructured_grid", "theta_nodal_xios.nc")
        )

        # cube = iris.load_cube(file_path, "theta")
        # Note: cannot use iris.load, as merge does not yet preserve
        # the cube 'ugrid' properties.

        # Here's a thing that at least works.
        loaded_cubes = CubeList(load_cubes(file_path, temp_xios_fix=True))
        (self.cube,) = loaded_cubes.extract(Constraint("theta"))

    def test_str__short(self):
        text = self.cube.summary(shorten=True)
        expect = "Potential Temperature / (K)         (time: 1; levels: 6; *-- : 866)"
        self.assertEqual(text, expect)

    def test_str__long(self):
        self.cube.attributes.clear()  # Just remove some uninteresting content.
        text = str(self.cube)
        expect = """\
Potential Temperature / (K)         (time: 1; levels: 6; *-- : 866)
     Dimension coordinates:
          time                           x          -        -
          levels                         -          x        -
     Auxiliary coordinates:
          time                           x          -        -
     ugrid information:
          Mesh0.node                     -          -        x
          topology_dimension: 2
          node_coordinates: latitude longitude
     Cell methods:
          point: time\
"""
        self.assertEqual(text, expect)


if __name__ == "__main__":
    tests.main()
