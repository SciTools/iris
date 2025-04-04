# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.util.equalise_cubes` function."""

import warnings

from cf_units import Unit
import numpy as np
import pytest

from iris.coords import DimCoord
from iris.cube import Cube
from iris.util import equalise_cubes
from iris.warnings import IrisUserWarning


def _scramble(inputs):
    # Reorder items (IN PLACE) to check that order does not affect operation.
    # Rather than anything more clever, we'll settle for just reversing the order.
    inputs[:] = inputs[::-1]


@pytest.fixture(params=["off", "on", "applyall", "scrambled"])
def usage(request):
    # Fixture to check different usage modes for a given operation control keyword
    return request.param


def _usage_common(usage, op_keyword_name, test_cubes):
    kwargs = {}
    if usage == "off":
        pass
    elif usage in ("on", "scrambled"):
        kwargs[op_keyword_name] = True
        if usage == "scrambled":
            # reorder the input cubes, but in-place
            _scramble(test_cubes)
    elif usage == "applyall":
        kwargs["apply_all"] = True
    else:
        raise ValueError(f"Unrecognised 'usage' option {usage!r}")
    default_expected_metadatas = [cube.metadata for cube in test_cubes]
    return kwargs, default_expected_metadatas


def _cube(
    stdname=None,
    longname=None,
    varname=None,
    units="unknown",
    cell_methods=(),
    **attributes,
):
    # Construct a simple test-cube with given metadata properties.
    cube = Cube(
        [1],
        standard_name=stdname,
        long_name=longname,
        var_name=varname,
        cell_methods=cell_methods,
        units=units,
        attributes=attributes,
    )
    return cube


_NO_OP_MESSAGE = "'equalise_cubes' call does nothing"


class TestNoOperation:
    def test(self):
        # A standalone test, that a call with no operations enabled raises a warning
        with pytest.warns(IrisUserWarning, match=_NO_OP_MESSAGE):
            equalise_cubes([])


class WarnChecked:
    @pytest.fixture(autouse=True)
    def nowarn(self, usage):
        if usage == "off":
            with pytest.warns(IrisUserWarning, match=_NO_OP_MESSAGE):
                yield
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                yield


class TestUnifyNames(WarnChecked):
    # Test the 'normalise_names' operation.
    def test_simple(self, usage):
        sn = "air_temperature"
        stdnames = [sn, sn, sn]
        longnames = [None, "long1", "long2"]
        varnames = ["var1", None, "var2"]
        test_cubes = [
            _cube(stdname=stdname, longname=longname, varname=varname)
            for stdname, longname, varname in zip(stdnames, longnames, varnames)
        ]
        kwargs, expected_metadatas = _usage_common(usage, "normalise_names", test_cubes)

        # Calculate expected results
        if usage != "off":
            # result cube metadata should all be the same, with no varname
            meta = _cube(stdname=sn).metadata
            expected_metadatas = [meta, meta, meta]

        # Apply operation
        results = equalise_cubes(test_cubes, **kwargs)

        # Assert result
        assert [cube.metadata for cube in results] == expected_metadatas

    def test_multi(self, usage):
        # Show that different cases are resolved independently
        sn1, sn2 = "air_temperature", "air_pressure"
        stdnames = [sn1, None, None, None, sn2, None]
        longnames = ["long1", "long2", None, None, "long3", None]
        varnames = ["var1", None, "var3", "var4", None, None]
        test_cubes = [
            _cube(stdname=stdname, longname=longname, varname=varname)
            for stdname, longname, varname in zip(stdnames, longnames, varnames)
        ]
        kwargs, expected_metadatas = _usage_common(usage, "normalise_names", test_cubes)

        # Calculate expected results
        if usage != "off":
            stdnames = [sn1, None, None, None, sn2, None]
            longnames = [None, "long2", None, None, None, None]
            varnames = [None, None, "var3", "var4", None, None]
            expected_metadatas = [
                _cube(stdname=stdname, longname=longname, varname=varname).metadata
                for stdname, longname, varname in zip(stdnames, longnames, varnames)
            ]
            if usage == "scrambled":
                _scramble(expected_metadatas)

        # Apply operation
        results = equalise_cubes(test_cubes, **kwargs)

        # Assert result
        assert [cube.metadata for cube in results] == expected_metadatas


class TestEqualiseAttributes(WarnChecked):
    # Test the 'equalise_attributes' operation.
    def test_calling(self, usage, mocker):
        patch = mocker.patch("iris.util.equalise_attributes")
        test_cubes = [_cube()]
        kwargs, expected_metadatas = _usage_common(
            usage, "equalise_attributes", test_cubes
        )

        # Apply operation
        equalise_cubes(test_cubes, **kwargs)

        expected_calls = 0 if usage == "off" else 1
        assert len(patch.call_args_list) == expected_calls

    def test_basic_function(self, usage):
        test_cubes = [_cube(att_a=10, att_b=1), _cube(att_a=10, att_b=2)]
        kwargs, expected_metadatas = _usage_common(
            usage, "equalise_attributes", test_cubes
        )

        # Calculate expected results
        if usage != "off":
            # result cube metadata should all be the same, with no varname
            meta = _cube(att_a=10).metadata
            expected_metadatas = [meta, meta]

        # Apply operation
        results = equalise_cubes(test_cubes, **kwargs)

        # Assert result
        assert [cube.metadata for cube in results] == expected_metadatas

    def test_operation_in_groups(self, usage):
        # Check that it acts independently within groups (as defined, here, by naming)
        test_cubes = [
            _cube(longname="a", att_a=10, att_b=1),
            _cube(longname="a", att_a=10, att_b=2),
            _cube(longname="b", att_a=10, att_b=1),
            _cube(longname="b", att_a=10, att_b=1),
        ]
        kwargs, expected_metadatas = _usage_common(
            usage, "equalise_attributes", test_cubes
        )

        # Calculate expected results
        if usage != "off":
            # result cube metadata should all be the same, with no varname
            expected_metadatas = [
                # the "a" cubes have lost att_b, but the "b" cubes retain it
                _cube(longname="a", att_a=10).metadata,
                _cube(longname="a", att_a=10).metadata,
                _cube(longname="b", att_a=10, att_b=1).metadata,
                _cube(longname="b", att_a=10, att_b=1).metadata,
            ]
            if usage == "scrambled":
                _scramble(expected_metadatas)

        # Apply operation
        results = equalise_cubes(test_cubes, **kwargs)

        # Assert result
        assert [cube.metadata for cube in results] == expected_metadatas

    def test_array_attributes(self, usage):
        # Array content is worth a special test because it breaks dictionary equality.
        a1 = np.array([4.1, 5.2, 6.3])
        a2 = np.array([1, 2])
        a3 = np.array([1, 3])
        test_cubes = [
            _cube(longname="a", v1=a1, v2=a2),
            _cube(longname="a", v1=a1, v2=a3),
            _cube(longname="b", v1=a1, v2=a2),
            _cube(longname="b", v1=a1, v2=a2),
        ]
        kwargs, expected_metadatas = _usage_common(
            usage, "equalise_attributes", test_cubes
        )

        # Calculate expected results
        if usage != "off":
            # result cube metadata should all be the same, with no varname
            result_cubes = [
                _cube(longname="a", v1=a1),
                _cube(longname="a", v1=a1),
                _cube(longname="b", v1=a1, v2=a2),
                _cube(longname="b", v1=a1, v2=a2),
            ]
            expected_metadatas = [cube.metadata for cube in result_cubes]
            if usage == "scrambled":
                _scramble(expected_metadatas)

        # Apply operation
        results = equalise_cubes(test_cubes, **kwargs)

        # Assert result
        assert [cube.metadata for cube in results] == expected_metadatas


class TestUnifyTimeUnits(WarnChecked):
    # Test the 'unify_time_units' operation.
    def test_calling(self, usage, mocker):
        patch = mocker.patch("iris.util.unify_time_units")
        test_cubes = [_cube()]
        kwargs, expected_metadatas = _usage_common(
            usage, "unify_time_units", test_cubes
        )

        # Apply operation
        equalise_cubes(test_cubes, **kwargs)

        expected_calls = 0 if usage == "off" else 1
        assert len(patch.call_args_list) == expected_calls

    def _cube_timeunits(self, unit, **kwargs):
        cube = _cube(**kwargs)
        cube.add_dim_coord(DimCoord([0.0], standard_name="time", units=unit), 0)
        return cube

    def test_basic_function(self, usage):
        if usage == "scrambled":
            pytest.skip("scrambled mode not supported")
        tu1, tu2 = [Unit(name) for name in ("days since 1970", "days since 1971")]
        cu1, cu2 = self._cube_timeunits(tu1), self._cube_timeunits(tu2)
        test_cubes = [cu1, cu2]
        kwargs, expected_metadatas = _usage_common(
            usage, "unify_time_units", test_cubes
        )

        expected_units = [tu1, tu2 if usage == "off" else tu1]

        # Apply operation
        results = equalise_cubes(test_cubes, **kwargs)

        # Assert result
        assert [cube.coord("time").units for cube in results] == expected_units

    def test_operation_in_groups(self, usage):
        # Check that it acts independently within groups (as defined, here, by naming)
        test_cubes = [
            _cube(longname="a", att_a=10, att_b=1),
            _cube(longname="a", att_a=10, att_b=2),
            _cube(longname="b", att_a=10, att_b=1),
            _cube(longname="b", att_a=10, att_b=1),
        ]
        kwargs, expected_metadatas = _usage_common(
            usage, "equalise_attributes", test_cubes
        )

        # Calculate expected results
        if usage != "off":
            # result cube metadata should all be the same, with no varname
            expected_metadatas = [
                # the "a" cubes have lost att_b, but the "b" cubes retain it
                _cube(longname="a", att_a=10).metadata,
                _cube(longname="a", att_a=10).metadata,
                _cube(longname="b", att_a=10, att_b=1).metadata,
                _cube(longname="b", att_a=10, att_b=1).metadata,
            ]
            if usage == "scrambled":
                _scramble(expected_metadatas)

        # Apply operation
        results = equalise_cubes(test_cubes, **kwargs)

        # Assert result
        assert [cube.metadata for cube in results] == expected_metadatas
