# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.io.save` function."""

from pathlib import Path

import pytest

import iris
from iris.cube import Cube
import iris.fileformats.netcdf.saver


class TestSave:
    def test_pathlib_save(self, mocker):
        file_mock = mocker.Mock()
        # Have to configure after creation because "name" is special
        file_mock.configure_mock(name="string")

        find_saver_mock = mocker.patch(
            "iris.io.find_saver", return_value=(lambda *args, **kwargs: None)
        )

        def replace_expand(file_specs, files_expected=True):
            return file_specs

        # does not expand filepaths due to patch
        mocker.patch("iris.io.expand_filespecs", replace_expand)

        test_variants = [
            ("string", "string"),
            (Path("string/string"), "string/string"),
            (file_mock, "string"),
        ]

        for target, fs_val in test_variants:
            try:
                iris.save(Cube([]), target)
            except ValueError:
                print("ValueError")
                pass
            find_saver_mock.assert_called_with(fs_val)


class TestCubeListSavers:
    """Which savers take a `CubeList` whole, rather than one cube at a time.

    `iris.io._savers.cubelist_savers` holds saver functions, not extensions,
    because `save` is reached with savers it never looked up by extension:
    one passed by the caller, and the one chosen directly for an NCZarr
    target. Keyed by extension, both would fall to the `append=True` path,
    and the netCDF saver has no `append` parameter to fall to.
    """

    def test_the_netcdf_saver_is_registered(self):
        """Registered by identity, so no name has to be matched.

        Two earlier forms matched on the module name: a substring test, true
        of any module whose path happened to contain `iris.fileformats.netcdf`,
        and then an exact set of module strings. Neither could be checked by a
        reader against the function actually being called.
        """
        from iris.fileformats.netcdf.saver import save as netcdf_save

        iris.io._check_init_savers()
        assert netcdf_save in iris.io._savers.cubelist_savers

    def test_a_saver_passed_directly_is_recognised(self, tmp_path):
        """The path that never reaches `find_saver`.

        `netcdf.save` has no `append` parameter, so if the registry were
        keyed by extension and missed this saver, the fallback would raise
        rather than quietly saving one cube at a time.
        """
        from iris.fileformats.netcdf.saver import save as netcdf_save

        cubes = iris.cube.CubeList([Cube([1], long_name="a"), Cube([2], long_name="b")])
        target = tmp_path / "out.nc"
        iris.save(cubes, str(target), saver=netcdf_save)
        assert len(iris.load(str(target))) == 2

    def test_an_unregistered_saver_still_needs_append(self, tmp_path):
        def saver_without_append(cube, target):
            pass

        cubes = iris.cube.CubeList([Cube([1], long_name="a"), Cube([2], long_name="b")])
        with pytest.raises(ValueError, match="Cannot append cubes"):
            iris.save(cubes, str(tmp_path / "out.xyz"), saver=saver_without_append)
