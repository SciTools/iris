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

import iris
import iris.io


@tests.skip_data
class TestLoad(tests.IrisTest):
    def test_normal(self):
        paths = (tests.get_data_path(["PP", "aPPglob1", "global.pp"]),)
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


class TestOpenDAP(tests.IrisTest):
    def test_load(self):
        # Check that calling iris.load_* with a http URI triggers a call to
        # ``iris.io.load_http``

        url = "http://geoport.whoi.edu:80/thredds/dodsC/bathy/gom15"

        class LoadHTTPCalled(Exception):
            pass

        def new_load_http(passed_urls, *args, **kwargs):
            self.assertEqual(len(passed_urls), 1)
            self.assertEqual(url, passed_urls[0])
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
                    fn(url)

        finally:
            iris.io.load_http = orig


if __name__ == "__main__":
    tests.main()
