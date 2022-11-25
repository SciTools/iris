# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test pickling of Iris objects.

"""

# Import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests  # isort:skip

import io
import pickle

import cf_units
import numpy as np

import iris
from iris._lazy_data import as_concrete_data, as_lazy_data


class TestPickle(tests.IrisTest):
    def pickle_then_unpickle(self, obj):
        """
        Returns a generator of ("pickle protocol number", object) tuples.

        """
        for protocol in range(1 + pickle.HIGHEST_PROTOCOL):
            bio = io.BytesIO()
            pickle.dump(obj, bio, protocol)

            # move the bio back to the start and reconstruct
            bio.seek(0)
            reconstructed_obj = pickle.load(bio)

            yield protocol, reconstructed_obj

    @staticmethod
    def _real_data(cube):
        # Get the concrete data of the cube for performing data values
        # comparison checks.
        return as_concrete_data(cube.core_data())

    def assertCubeData(self, cube1, cube2):
        self.assertArrayEqual(self._real_data(cube1), self._real_data(cube2))

    @tests.skip_data
    def test_cube_pickle(self):
        cube = iris.load_cube(
            tests.get_data_path(("PP", "globClim1", "theta.pp"))
        )
        self.assertTrue(cube.has_lazy_data())
        self.assertCML(
            cube, ("cube_io", "pickling", "theta.cml"), checksum=False
        )

        for p, recon_cube in self.pickle_then_unpickle(cube):
            self.assertTrue(recon_cube.has_lazy_data())
            self.assertCML(
                recon_cube,
                ("cube_io", "pickling", "theta.cml"),
                checksum=False,
            )
            self.assertCubeData(cube, recon_cube)

    @tests.skip_data
    def test_cube_with_coord_points(self):
        filename = tests.get_data_path(
            ("NetCDF", "rotated", "xy", "rotPole_landAreaFraction.nc")
        )
        cube = iris.load_cube(filename)
        # Pickle and unpickle. Do not perform any CML tests
        # to avoid side effects.
        _, recon_cube = next(self.pickle_then_unpickle(cube))
        self.assertEqual(recon_cube, cube)

    def test_cube_with_deferred_unit_conversion(self):
        real_data = np.arange(12.0).reshape((3, 4))
        lazy_data = as_lazy_data(real_data)
        cube = iris.cube.Cube(lazy_data, units="m")
        cube.convert_units("ft")
        _, recon_cube = next(self.pickle_then_unpickle(cube))
        self.assertEqual(recon_cube, cube)

    @tests.skip_data
    def test_cubelist_pickle(self):
        cubelist = iris.load(
            tests.get_data_path(("PP", "COLPEX", "theta_and_orog_subset.pp"))
        )
        single_cube = cubelist[0]

        self.assertCML(cubelist, ("cube_io", "pickling", "cubelist.cml"))
        self.assertCML(single_cube, ("cube_io", "pickling", "single_cube.cml"))

        for _, reconstructed_cubelist in self.pickle_then_unpickle(cubelist):
            self.assertCML(
                reconstructed_cubelist, ("cube_io", "pickling", "cubelist.cml")
            )
            self.assertCML(
                reconstructed_cubelist[0],
                ("cube_io", "pickling", "single_cube.cml"),
            )

            for cube_orig, cube_reconstruct in zip(
                cubelist, reconstructed_cubelist
            ):
                self.assertArrayEqual(cube_orig.data, cube_reconstruct.data)
                self.assertEqual(cube_orig, cube_reconstruct)

    def test_picking_equality_misc(self):
        items_to_test = [
            cf_units.Unit(
                "hours since 2007-01-15 12:06:00",
                calendar=cf_units.CALENDAR_STANDARD,
            ),
            cf_units.as_unit("1"),
            cf_units.as_unit("meters"),
            cf_units.as_unit("no-unit"),
            cf_units.as_unit("unknown"),
        ]

        for orig_item in items_to_test:
            for protocol, reconst_item in self.pickle_then_unpickle(orig_item):
                fail_msg = (
                    "Items are different after pickling "
                    "at protocol {}.\nOrig item: {!r}\nNew item: {!r}"
                )
                fail_msg = fail_msg.format(protocol, orig_item, reconst_item)
                self.assertEqual(orig_item, reconst_item, fail_msg)


if __name__ == "__main__":
    tests.main()
