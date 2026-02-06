# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.netcdf._load_cube` function."""

import numpy as np
import pytest

from iris.coords import DimCoord
import iris.fileformats.cf
from iris.fileformats.netcdf.loader import _load_cube
from iris.loading import LOAD_PROBLEMS
from iris.tests.unit.fileformats import MockerMixin


class NoStr:
    def __str__(self):
        raise RuntimeError("No string representation")


class TestCoordAttributes(MockerMixin):
    @staticmethod
    def _patcher(engine, cf, cf_group):
        coordinates = []
        for coord in cf_group:
            engine.cube.add_aux_coord(coord)
            coordinates.append((coord, coord.name()))
        engine.cube_parts["coordinates"] = coordinates

    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        this = "iris.fileformats.netcdf.loader._assert_case_specific_facts"
        _ = mocker.patch(this, side_effect=self._patcher)
        self.engine = mocker.Mock()
        self.filename = "DUMMY"
        self.flag_masks = mocker.sentinel.flag_masks
        self.flag_meanings = mocker.sentinel.flag_meanings
        self.flag_values = mocker.sentinel.flag_values
        self.valid_range = mocker.sentinel.valid_range
        self.valid_min = mocker.sentinel.valid_min
        self.valid_max = mocker.sentinel.valid_max

    def _make(self, names, attrs):
        coords = [DimCoord(i, long_name=name) for i, name in enumerate(names)]
        shape = (1,)

        cf_group = {}
        for name, cf_attrs in zip(names, attrs):
            cf_attrs_unused = self.mocker.Mock(return_value=cf_attrs)
            cf_group[name] = self.mocker.Mock(cf_attrs_unused=cf_attrs_unused)
        cf = self.mocker.Mock(cf_group=cf_group)

        cf_data = self.mocker.Mock(_FillValue=None)
        cf_data.chunking = self.mocker.MagicMock(return_value=shape)
        cf_var = self.mocker.MagicMock(
            spec=iris.fileformats.cf.CFVariable,
            dtype=np.dtype("i4"),
            cf_data=cf_data,
            cf_name="DUMMY_VAR",
            cf_group=coords,
            shape=shape,
            size=np.prod(shape),
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
            assert len(cube.coords(name)) == 1
            coord = cube.coord(name)
            assert len(coord.attributes) == 1
            assert list(coord.attributes.keys()) == [attr]
            assert list(coord.attributes.values()) == [value]

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
        assert len(cube.coords()) == 3
        assert set([c.name() for c in cube.coords()]) == set(names)
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
            assert set(attributes.items()) == set(expect)

    def test_load_problems(self):
        key_and_val = (NoStr(), "wibble")

        cf, cf_var = self._make(["foo"], [[key_and_val]])
        _ = _load_cube(self.engine, cf, cf_var, self.filename)
        load_problem = LOAD_PROBLEMS.problems[-1]
        assert "No string representation" in "".join(load_problem.stack_trace.format())
        destination = load_problem.destination
        assert destination.iris_class is DimCoord
        # Note: cannot test destination.identifier without large increase in
        #  complexity. Rely on TestCubeAttributes.test_load_problems for this.


class TestCubeAttributes(MockerMixin):
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        this = "iris.fileformats.netcdf.loader._assert_case_specific_facts"
        _ = mocker.patch(this)
        self.engine = mocker.Mock()
        self.cf = None
        self.filename = "DUMMY"
        self.flag_masks = mocker.sentinel.flag_masks
        self.flag_meanings = mocker.sentinel.flag_meanings
        self.flag_values = mocker.sentinel.flag_values
        self.valid_range = mocker.sentinel.valid_range
        self.valid_min = mocker.sentinel.valid_min
        self.valid_max = mocker.sentinel.valid_max

    def _make(self, attrs):
        shape = (1,)
        cf_attrs_unused = self.mocker.Mock(return_value=attrs)
        cf_data = self.mocker.Mock(_FillValue=None)
        cf_data.chunking = self.mocker.MagicMock(return_value=shape)
        cf_var = self.mocker.MagicMock(
            spec=iris.fileformats.cf.CFVariable,
            dtype=np.dtype("i4"),
            cf_data=cf_data,
            cf_name="DUMMY_VAR",
            filename="DUMMY",
            cf_group=self.mocker.Mock(),
            cf_attrs_unused=cf_attrs_unused,
            shape=shape,
            size=np.prod(shape),
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
            assert len(cube.attributes) == 1
            assert list(cube.attributes.keys()) == [key]
            assert list(cube.attributes.values()) == [value]

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
        assert len(cube.attributes) == len(expected)
        assert set(cube.attributes.items()) == expected

    def test_load_problems(self):
        key_and_val = (NoStr(), "wibble")

        cf_var = self._make([key_and_val])
        _ = _load_cube(self.engine, self.cf, cf_var, self.filename)
        load_problem = LOAD_PROBLEMS.problems[-1]
        assert "No string representation" in "".join(load_problem.stack_trace.format())
        destination = load_problem.destination
        assert destination.iris_class is self.engine.cube.__class__
        assert destination.identifier == cf_var.cf_name
