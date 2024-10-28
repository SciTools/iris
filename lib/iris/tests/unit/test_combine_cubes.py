# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.io.loading.combine_cubes` function.

Note: These tests are fairly extensive to cover functional uses within the loading
operations.
TODO: when function is public API, extend testing to the extended API options,
i.e. different types + defaulting of the 'options' arg, and **kwargs support.
"""

import pytest

from iris import LoadPolicy, _combine_cubes
from iris.tests.unit.fileformats.test_load_functions import cu


@pytest.fixture(params=list(LoadPolicy.SETTINGS.keys()))
def options(request):
    # N.B. "request" is a standard PyTest fixture
    return request.param  # Return the name of the attribute to test.


# Interface to convert settings-name / kwargs into an options dict,
# TODO: remove this wrapper when the API of "combine_cubes" is opened up.
def combine_cubes(cubes, settings_name="default", **kwargs):
    options = LoadPolicy.SETTINGS[settings_name]
    options.update(kwargs)
    return _combine_cubes(cubes, options, merge_require_unique=False)


class Test:
    def test_mergeable(self, options):
        c1, c2 = cu(t=1), cu(t=2)
        c12 = cu(t=(1, 2))
        input_cubes = [c1, c2]
        result = combine_cubes(input_cubes, options)
        expected = [c12]  # same in all cases
        assert result == expected

    def test_catable(self, options):
        c1, c2 = cu(t=(1, 2)), cu(t=(3, 4))
        c12 = cu(t=(1, 2, 3, 4))
        input_cubes = [c1, c2]
        result = combine_cubes(input_cubes, options)
        expected = {
            "legacy": [c1, c2],  # standard options can't do this ..
            "default": [c1, c2],
            "recommended": [c12],  # .. but it works if you enable concatenate
            "comprehensive": [c12],
        }[options]
        assert result == expected

    def test_cat_enables_merge(self, options):
        c1, c2 = cu(t=(1, 2), z=1), cu(t=(3, 4, 5), z=1)
        c3, c4 = cu(t=(1, 2, 3), z=2), cu(t=(4, 5), z=2)
        c1234 = cu(t=(1, 2, 3, 4, 5), z=(1, 2))
        c12 = cu(t=(1, 2, 3, 4, 5), z=1)
        c34 = cu(t=(1, 2, 3, 4, 5), z=2)
        input_cubes = [c1, c2, c3, c4]
        result = combine_cubes(input_cubes, options)
        expected = {
            "legacy": input_cubes,
            "default": input_cubes,
            "recommended": [c12, c34],  # standard "mc" sequence can't do this one..
            "comprehensive": [c1234],  # .. but works if you repeat
        }[options]
        assert result == expected

    def test_cat_enables_merge__custom(self):
        c1, c2 = cu(t=(1, 2), z=1), cu(t=(3, 4, 5), z=1)
        c3, c4 = cu(t=(1, 2, 3), z=2), cu(t=(4, 5), z=2)
        c1234 = cu(t=(1, 2, 3, 4, 5), z=(1, 2))
        input_cubes = [c1, c2, c3, c4]
        result = combine_cubes(input_cubes, merge_concat_sequence="cm")
        assert result == [c1234]

    def test_nocombine_overlapping(self, options):
        c1, c2 = cu(t=(1, 3)), cu(t=(2, 4))
        input_cubes = [c1, c2]
        result = combine_cubes(input_cubes, options)
        assert result == input_cubes  # same in all cases : can't do this

    def test_nocombine_dim_scalar(self, options):
        c1, c2 = cu(t=(1,)), cu(t=2)
        input_cubes = [c1, c2]
        result = combine_cubes(input_cubes, options)
        assert result == input_cubes  # can't do this at present
