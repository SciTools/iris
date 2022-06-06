# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test the main loading API.

"""

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests  # isort:skip

import pathlib
from unittest import mock

import netCDF4

import iris
import iris.io


@tests.skip_data
class TestLoad(tests.IrisTest):
    def test_normal(self):
        paths = (tests.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        cubes = iris.load(paths)
        self.assertEqual(len(cubes), 1)

    def test_path_object(self):
        paths = (
            pathlib.Path(tests.get_data_path(["PP", "aPPglob1", "global.pp"])),
        )
        cubes = iris.load(paths)
        self.assertEqual(len(cubes), 1)

    def test_nonexist(self):
        paths = (
            tests.get_data_path(["PP", "aPPglob1", "global.pp"]),
            tests.get_data_path(["PP", "_guaranteed_non_exist.pp"]),
        )
        with self.assertRaises(IOError) as error_trap:
            _ = iris.load(paths)
        self.assertIn(
            "One or more of the files specified did not exist",
            str(error_trap.exception),
        )

    def test_nonexist_wild(self):
        paths = (
            tests.get_data_path(["PP", "aPPglob1", "global.pp"]),
            tests.get_data_path(["PP", "_guaranteed_non_exist_*.pp"]),
        )
        with self.assertRaises(IOError) as error_trap:
            _ = iris.load(paths)
        self.assertIn(
            "One or more of the files specified did not exist",
            str(error_trap.exception),
        )

    def test_bogus(self):
        paths = (tests.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        cubes = iris.load(paths, "wibble")
        self.assertEqual(len(cubes), 0)

    def test_real_and_bogus(self):
        paths = (tests.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        cubes = iris.load(paths, ("air_temperature", "wibble"))
        self.assertEqual(len(cubes), 1)

    def test_duplicate(self):
        paths = (
            tests.get_data_path(["PP", "aPPglob1", "global.pp"]),
            tests.get_data_path(["PP", "aPPglob1", "gl?bal.pp"]),
        )
        cubes = iris.load(paths)
        self.assertEqual(len(cubes), 2)


@tests.skip_data
class TestLoadCube(tests.IrisTest):
    def test_normal(self):
        paths = (tests.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        _ = iris.load_cube(paths)

    def test_path_object(self):
        paths = (
            pathlib.Path(tests.get_data_path(["PP", "aPPglob1", "global.pp"])),
        )
        _ = iris.load_cube(paths)

    def test_not_enough(self):
        paths = (tests.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        with self.assertRaises(iris.exceptions.ConstraintMismatchError):
            iris.load_cube(paths, "wibble")

    def test_too_many(self):
        paths = (
            tests.get_data_path(["PP", "aPPglob1", "global.pp"]),
            tests.get_data_path(["PP", "aPPglob1", "gl?bal.pp"]),
        )
        with self.assertRaises(iris.exceptions.ConstraintMismatchError):
            iris.load_cube(paths)


@tests.skip_data
class TestLoadCubes(tests.IrisTest):
    def test_normal(self):
        paths = (tests.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        cubes = iris.load_cubes(paths)
        self.assertEqual(len(cubes), 1)

    def test_path_object(self):
        paths = (
            pathlib.Path(tests.get_data_path(["PP", "aPPglob1", "global.pp"])),
        )
        cubes = iris.load_cubes(paths)
        self.assertEqual(len(cubes), 1)

    def test_not_enough(self):
        paths = (tests.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        with self.assertRaises(iris.exceptions.ConstraintMismatchError):
            iris.load_cubes(paths, "wibble")

    def test_not_enough_multi(self):
        paths = (tests.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        with self.assertRaises(iris.exceptions.ConstraintMismatchError):
            iris.load_cubes(paths, ("air_temperature", "wibble"))

    def test_too_many(self):
        paths = (
            tests.get_data_path(["PP", "aPPglob1", "global.pp"]),
            tests.get_data_path(["PP", "aPPglob1", "gl?bal.pp"]),
        )
        with self.assertRaises(iris.exceptions.ConstraintMismatchError):
            iris.load_cube(paths)


@tests.skip_data
class TestLoadRaw(tests.IrisTest):
    def test_normal(self):
        paths = (tests.get_data_path(["PP", "aPPglob1", "global.pp"]),)
        cubes = iris.load_raw(paths)
        self.assertEqual(len(cubes), 1)

    def test_path_object(self):
        paths = (
            pathlib.Path(tests.get_data_path(["PP", "aPPglob1", "global.pp"])),
        )
        cubes = iris.load_raw(paths)
        self.assertEqual(len(cubes), 1)


class TestOPeNDAP(tests.IrisTest):
    def setUp(self):
        self.url = "http://geoport.whoi.edu:80/thredds/dodsC/bathy/gom15"

    def test_load_http_called(self):
        # Check that calling iris.load_* with an http URI triggers a call to
        # ``iris.io.load_http``

        class LoadHTTPCalled(Exception):
            pass

        def new_load_http(passed_urls, *args, **kwargs):
            self.assertEqual(len(passed_urls), 1)
            self.assertEqual(self.url, passed_urls[0])
            raise LoadHTTPCalled()

        try:
            orig = iris.io.load_http
            iris.io.load_http = new_load_http

            for fn in [
                iris.load,
                iris.load_raw,
                iris.load_cube,
                iris.load_cubes,
            ]:
                with self.assertRaises(LoadHTTPCalled):
                    fn(self.url)

        finally:
            iris.io.load_http = orig

    def test_netCDF_Dataset_call(self):
        # Check that load_http calls netCDF4.Dataset and supplies the expected URL.

        # To avoid making a request to an OPeNDAP server in a test, instead
        # mock the call to netCDF.Dataset so that it returns a dataset for a
        # local file.
        filename = tests.get_data_path(
            ("NetCDF", "global", "xyt", "SMALL_total_column_co2.nc")
        )
        fake_dataset = netCDF4.Dataset(filename)

        with mock.patch(
            "netCDF4.Dataset", return_value=fake_dataset
        ) as dataset_loader:
            next(iris.io.load_http([self.url], callback=None))
        dataset_loader.assert_called_with(self.url, mode="r")


if __name__ == "__main__":
    tests.main()
