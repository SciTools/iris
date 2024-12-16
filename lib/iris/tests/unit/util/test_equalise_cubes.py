# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.util.equalise_cubes` function."""

import numpy as np
from numpy.random import Generator
import pytest

from iris.cube import Cube
from iris.util import equalise_cubes


@pytest.fixture(params=["off", "on", "applyall", "scrambled"])
def usage(request):
    return request.param


_RNG = 95297


def _scramble(inputs, rng=_RNG):
    # Make a simple check that the input order does not affect the result
    if not isinstance(rng, Generator):
        rng = np.random.default_rng(rng)
    n_inputs = len(inputs)
    # NOTE: make object array of explicit shape + fill it,
    # since np.array(inputs) *fails* specifically with a list of metadata objects
    inputs_array = np.empty((n_inputs,), dtype=object)
    inputs_array[:] = inputs
    n_inputs = inputs_array.shape[0]
    scramble_inds = rng.permutation(n_inputs)
    inputs_array = inputs_array[scramble_inds]
    # Modify input list **BUT N.B. IN PLACE**
    inputs[0:] = inputs_array
    return inputs


def _cube(
    stdname=None,
    varname=None,
    longname=None,
    units="unknown",
    cell_methods=(),
    **kwattributes,
):
    # Construct a simple test-cube with given metadata properties
    cube = Cube(
        [1],
        standard_name=stdname,
        long_name=longname,
        var_name=varname,
        cell_methods=cell_methods,
        units=units,
        attributes=kwattributes,
    )
    return cube


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


class TestUnifyNames:
    def test_stdnames_simple(self, usage):
        sn = "air_temperature"
        stdnames = [sn, sn, sn]
        longnames = [None, "long1", "long2"]
        varnames = ["var1", None, "var2"]
        test_cubes = [
            _cube(stdname=stdname, longname=longname, varname=varname)
            for stdname, longname, varname in zip(stdnames, longnames, varnames)
        ]
        kwargs, expected_metadatas = _usage_common(usage, "unify_names", test_cubes)

        # Calculate expected results
        if usage != "off":
            # result cube metadata should all be the same, with no varname
            meta = _cube(stdname=sn).metadata
            expected_metadatas = [meta] * len(test_cubes)

        # Apply operation
        results = equalise_cubes(test_cubes, **kwargs)

        # Assert result
        assert [cube.metadata for cube in results] == expected_metadatas

    def test_stdnames_multi(self, usage):
        # Show that two different standard-name groups are handled independently
        sn1, sn2 = "air_temperature", "air_pressure"
        stdnames = [sn1, sn1, sn1, sn2, sn2, sn2]
        varnames = ["v1", None, "v2", "v3", None, None]
        test_cubes = [
            _cube(stdname, varname) for stdname, varname in zip(stdnames, varnames)
        ]
        kwargs, expected_metadatas = _usage_common(usage, "unify_names", test_cubes)

        # Calculate expected results
        if usage != "off":
            # result cube metadata should be of only 2 types
            meta1 = _cube(stdname=sn1).metadata
            meta2 = _cube(stdname=sn2).metadata
            # the result cubes should still correspond to the original input order,
            # since all cube equalisation operations occur in-place
            expected_metadatas = [
                meta1 if cube.standard_name == sn1 else meta2 for cube in test_cubes
            ]

        # Apply operation
        results = equalise_cubes(test_cubes, **kwargs)

        # Assert result
        assert [cube.metadata for cube in results] == expected_metadatas

    def test_missing_names(self, usage):
        # Show that two different standard-name groups are handled independently
        sn = "air_temperature"
        stdnames = [sn, None, None, None]
        longnames = ["long1", "long2", None, None]
        varnames = ["var1", "var2", "var3", None]
        test_cubes = [
            _cube(stdname=stdname, longname=longname, varname=varname)
            for stdname, longname, varname in zip(stdnames, longnames, varnames)
        ]
        kwargs, expected_metadatas = _usage_common(usage, "unify_names", test_cubes)

        # Calculate expected results
        if usage != "off":
            stdnames = [sn, None, None, None]
            longnames = [None, "long2", None, None]
            varnames = [None, None, "var3", None]
            expected_metadatas = [
                _cube(stdname=stdname, longname=longname, varname=varname).metadata
                for stdname, longname, varname in zip(stdnames, longnames, varnames)
            ]
            if usage == "scrambled":
                expected_metadatas = _scramble(expected_metadatas)

        # Apply operation
        results = equalise_cubes(test_cubes, **kwargs)

        # Assert result
        assert [cube.metadata for cube in results] == expected_metadatas
