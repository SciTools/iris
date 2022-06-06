# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Mirror of :mod:`iris.tests.unit.fileformats.netcdf.test_Saver`, but with lazy arrays."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from dask import array as da

from iris.coords import AuxCoord
from iris.fileformats.netcdf import Saver
from iris.tests import stock
from iris.tests.unit.fileformats.netcdf import test_Saver


class LazyMixin(tests.IrisTest):
    array_lib = da

    def result_path(self, basename=None, ext=""):
        # Precisely mirroring the tests in test_Saver, so use those CDL's.
        original = super().result_path(basename, ext)
        return original.replace("Saver__lazy", "Saver")


class Test_write(LazyMixin, test_Saver.Test_write):
    pass


class Test__create_cf_bounds(test_Saver.Test__create_cf_bounds):
    @staticmethod
    def climatology_3d():
        cube = stock.climatology_3d()
        aux_coord = AuxCoord.from_coord(cube.coord("time"))
        lazy_coord = aux_coord.copy(
            aux_coord.lazy_points(), aux_coord.lazy_bounds()
        )
        cube.replace_coord(lazy_coord)
        return cube


class Test_write__valid_x_cube_attributes(
    LazyMixin, test_Saver.Test_write__valid_x_cube_attributes
):
    pass


class Test_write__valid_x_coord_attributes(
    LazyMixin, test_Saver.Test_write__valid_x_coord_attributes
):
    pass


class Test_write_fill_value(LazyMixin, test_Saver.Test_write_fill_value):
    pass


class Test_check_attribute_compliance__valid_range(
    LazyMixin, test_Saver.Test_check_attribute_compliance__valid_range
):
    pass


class Test_check_attribute_compliance__valid_min(
    LazyMixin, test_Saver.Test_check_attribute_compliance__valid_min
):
    pass


class Test_check_attribute_compliance__valid_max(
    LazyMixin, test_Saver.Test_check_attribute_compliance__valid_max
):
    pass


class Test_check_attribute_compliance__exception_handling(
    LazyMixin, test_Saver.Test_check_attribute_compliance__exception_handling
):
    pass


class Test__create_cf_cell_measure_variable(
    LazyMixin, test_Saver.Test__create_cf_cell_measure_variable
):
    pass


class TestStreamed(tests.IrisTest):
    def setUp(self):
        self.cube = stock.simple_2d()
        self.store_watch = self.patch("dask.array.store")

    def save_common(self, cube_to_save):
        with self.temp_filename(".nc") as nc_path:
            with Saver(nc_path, "NETCDF4") as saver:
                saver.write(cube_to_save)

    def test_realised_not_streamed(self):
        self.save_common(self.cube)
        self.assertFalse(self.store_watch.called)

    def test_lazy_streamed_data(self):
        self.cube.data = self.cube.lazy_data()
        self.save_common(self.cube)
        self.assertTrue(self.store_watch.called)

    def test_lazy_streamed_coord(self):
        aux_coord = AuxCoord.from_coord(self.cube.coords()[0])
        lazy_coord = aux_coord.copy(
            aux_coord.lazy_points(), aux_coord.lazy_bounds()
        )
        self.cube.replace_coord(lazy_coord)
        self.save_common(self.cube)
        self.assertTrue(self.store_watch.called)

    def test_lazy_streamed_bounds(self):
        aux_coord = AuxCoord.from_coord(self.cube.coords()[0])
        lazy_coord = aux_coord.copy(aux_coord.points, aux_coord.lazy_bounds())
        self.cube.replace_coord(lazy_coord)
        self.save_common(self.cube)
        self.assertTrue(self.store_watch.called)


if __name__ == "__main__":
    tests.main()
