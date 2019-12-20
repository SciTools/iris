# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.fileformats.netcdf._load_cube` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from unittest import mock

import numpy as np

from iris.coords import DimCoord
import iris.fileformats.cf
from iris.fileformats.netcdf import _load_cube


class TestCoordAttributes(tests.IrisTest):
    @staticmethod
    def _patcher(engine, cf, cf_group):
        coordinates = []
        for coord in cf_group:
            engine.cube.add_aux_coord(coord)
            coordinates.append((coord, coord.name()))
        engine.cube_parts["coordinates"] = coordinates

    def setUp(self):
        this = "iris.fileformats.netcdf._assert_case_specific_facts"
        patch = mock.patch(this, side_effect=self._patcher)
        patch.start()
        self.addCleanup(patch.stop)
        self.engine = mock.Mock()
        self.filename = "DUMMY"
        self.flag_masks = mock.sentinel.flag_masks
        self.flag_meanings = mock.sentinel.flag_meanings
        self.flag_values = mock.sentinel.flag_values
        self.valid_range = mock.sentinel.valid_range
        self.valid_min = mock.sentinel.valid_min
        self.valid_max = mock.sentinel.valid_max

    def _make(self, names, attrs):
        coords = [DimCoord(i, long_name=name) for i, name in enumerate(names)]
        shape = (1,)

        cf_group = {}
        for name, cf_attrs in zip(names, attrs):
            cf_attrs_unused = mock.Mock(return_value=cf_attrs)
            cf_group[name] = mock.Mock(cf_attrs_unused=cf_attrs_unused)
        cf = mock.Mock(cf_group=cf_group)

        cf_data = mock.Mock(_FillValue=None)
        cf_data.chunking = mock.MagicMock(return_value=shape)
        cf_var = mock.MagicMock(
            spec=iris.fileformats.cf.CFVariable,
            dtype=np.dtype("i4"),
            cf_data=cf_data,
            cf_name="DUMMY_VAR",
            cf_group=coords,
            shape=shape,
        )
        return cf, cf_var

    def test_flag_pass_thru(self):
        items = [
            ("masks", "flag_masks", self.flag_masks),
            ("meanings", "flag_meanings", self.flag_meanings),
            ("values", "flag_values", self.flag_values),
        ]
        for name, attr, value in items:
            names = [name]
            attrs = [[(attr, value)]]
            cf, cf_var = self._make(names, attrs)
            cube = _load_cube(self.engine, cf, cf_var, self.filename)
            self.assertEqual(len(cube.coords(name)), 1)
            coord = cube.coord(name)
            self.assertEqual(len(coord.attributes), 1)
            self.assertEqual(list(coord.attributes.keys()), [attr])
            self.assertEqual(list(coord.attributes.values()), [value])

    def test_flag_pass_thru_multi(self):
        names = ["masks", "meanings", "values"]
        attrs = [
            [("flag_masks", self.flag_masks), ("wibble", "wibble")],
            [
                ("flag_meanings", self.flag_meanings),
                ("add_offset", "add_offset"),
            ],
            [("flag_values", self.flag_values)],
            [("valid_range", self.valid_range)],
            [("valid_min", self.valid_min)],
            [("valid_max", self.valid_max)],
        ]
        cf, cf_var = self._make(names, attrs)
        cube = _load_cube(self.engine, cf, cf_var, self.filename)
        self.assertEqual(len(cube.coords()), 3)
        self.assertEqual(set([c.name() for c in cube.coords()]), set(names))
        expected = [
            attrs[0],
            [attrs[1][0]],
            attrs[2],
            attrs[3],
            attrs[4],
            attrs[5],
        ]
        for name, expect in zip(names, expected):
            attributes = cube.coord(name).attributes
            self.assertEqual(set(attributes.items()), set(expect))


class TestCubeAttributes(tests.IrisTest):
    def setUp(self):
        this = "iris.fileformats.netcdf._assert_case_specific_facts"
        patch = mock.patch(this)
        patch.start()
        self.addCleanup(patch.stop)
        self.engine = mock.Mock()
        self.cf = None
        self.filename = "DUMMY"
        self.flag_masks = mock.sentinel.flag_masks
        self.flag_meanings = mock.sentinel.flag_meanings
        self.flag_values = mock.sentinel.flag_values
        self.valid_range = mock.sentinel.valid_range
        self.valid_min = mock.sentinel.valid_min
        self.valid_max = mock.sentinel.valid_max

    def _make(self, attrs):
        shape = (1,)
        cf_attrs_unused = mock.Mock(return_value=attrs)
        cf_data = mock.Mock(_FillValue=None)
        cf_data.chunking = mock.MagicMock(return_value=shape)
        cf_var = mock.MagicMock(
            spec=iris.fileformats.cf.CFVariable,
            dtype=np.dtype("i4"),
            cf_data=cf_data,
            cf_name="DUMMY_VAR",
            cf_group=mock.Mock(),
            cf_attrs_unused=cf_attrs_unused,
            shape=shape,
        )
        return cf_var

    def test_flag_pass_thru(self):
        attrs = [
            ("flag_masks", self.flag_masks),
            ("flag_meanings", self.flag_meanings),
            ("flag_values", self.flag_values),
        ]
        for key, value in attrs:
            cf_var = self._make([(key, value)])
            cube = _load_cube(self.engine, self.cf, cf_var, self.filename)
            self.assertEqual(len(cube.attributes), 1)
            self.assertEqual(list(cube.attributes.keys()), [key])
            self.assertEqual(list(cube.attributes.values()), [value])

    def test_flag_pass_thru_multi(self):
        attrs = [
            ("flag_masks", self.flag_masks),
            ("wibble", "wobble"),
            ("flag_meanings", self.flag_meanings),
            ("add_offset", "add_offset"),
            ("flag_values", self.flag_values),
            ("standard_name", "air_temperature"),
            ("valid_range", self.valid_range),
            ("valid_min", self.valid_min),
            ("valid_max", self.valid_max),
        ]

        # Expect everything from above to be returned except those
        # corresponding to exclude_ind.
        expected = set([attrs[ind] for ind in [0, 1, 2, 4, 6, 7, 8]])
        cf_var = self._make(attrs)
        cube = _load_cube(self.engine, self.cf, cf_var, self.filename)
        self.assertEqual(len(cube.attributes), len(expected))
        self.assertEqual(set(cube.attributes.items()), expected)


if __name__ == "__main__":
    tests.main()
