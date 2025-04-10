# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.util.combine_cubes` function."""

import pytest

from iris import LoadPolicy
from iris.exceptions import DuplicateDataError
from iris.tests.unit.fileformats.test_load_functions import cu
from iris.util import combine_cubes


@pytest.fixture(params=list(LoadPolicy.SETTINGS.keys()))
def settings(request):
    # N.B. "request" is a standard PyTest fixture
    return request.param  # Return the name of the attribute to test.


class Test_settings:
    """These tests cover functional uses of the different "settings" choices, with specific
    sets of cubes sensitive to those choices.
    """

    def test_mergeable(self, settings):
        c1, c2 = cu(t=1), cu(t=2)
        c12 = cu(t=(1, 2))
        input_cubes = [c1, c2]
        result = combine_cubes(input_cubes, settings)
        expected = [c12]  # same in all cases
        assert result == expected

    def test_catable(self, settings):
        c1, c2 = cu(t=(1, 2)), cu(t=(3, 4))
        c12 = cu(t=(1, 2, 3, 4))
        input_cubes = [c1, c2]
        result = combine_cubes(input_cubes, settings)
        expected = {
            "legacy": [c1, c2],  # standard settings can't do this ..
            "default": [c1, c2],
            "recommended": [c12],  # .. but it works if you enable concatenate
            "comprehensive": [c12],
        }[settings]
        assert result == expected

    def test_cat_enables_merge(self, settings):
        c1, c2 = cu(t=(1, 2), z=1), cu(t=(3, 4, 5), z=1)
        c3, c4 = cu(t=(1, 2, 3), z=2), cu(t=(4, 5), z=2)
        c1234 = cu(t=(1, 2, 3, 4, 5), z=(1, 2))
        c12 = cu(t=(1, 2, 3, 4, 5), z=1)
        c34 = cu(t=(1, 2, 3, 4, 5), z=2)
        input_cubes = [c1, c2, c3, c4]
        result = combine_cubes(input_cubes, settings)
        expected = {
            "legacy": input_cubes,
            "default": input_cubes,
            "recommended": [c12, c34],  # standard "mc" sequence can't do this one..
            "comprehensive": [c1234],  # .. but works if you repeat
        }[settings]
        assert result == expected

    def test_nocombine_overlapping(self, settings):
        c1, c2 = cu(t=(1, 3)), cu(t=(2, 4))
        input_cubes = [c1, c2]
        result = combine_cubes(input_cubes, settings)
        assert result == input_cubes  # same in all cases : can't do this

    def test_nocombine_dim_scalar(self, settings):
        c1, c2 = cu(t=(1,)), cu(t=2)
        input_cubes = [c1, c2]
        result = combine_cubes(input_cubes, settings)
        assert result == input_cubes  # can't do this at present


def test_cat_enables_merge__custom():
    """A standalone testcase.

    This shows specifically how a 'cm' sequence can improve on the default 'm'.
    """
    c1, c2 = cu(t=(1, 2), z=1), cu(t=(3, 4, 5), z=1)
    c3, c4 = cu(t=(1, 2, 3), z=2), cu(t=(4, 5), z=2)
    c1234 = cu(t=(1, 2, 3, 4, 5), z=(1, 2))
    input_cubes = [c1, c2, c3, c4]
    result = combine_cubes(input_cubes, merge_concat_sequence="cm")
    assert result == [c1234]


class Test_options:
    """Test all the individual combine options keywords."""

    def test_equalise_cubes_kwargs(self):
        # two cubes will merge ..
        cubes = [cu(t=1), cu(t=2)]
        # .. but prevent by adding an attribute on one
        cubes[0].attributes["x"] = 3
        # won't combine..
        result = combine_cubes(cubes)
        assert len(result) == 2
        # ..but will if you enable attribute equalisation
        result = combine_cubes(
            cubes, equalise_cubes_kwargs={"equalise_attributes": True}
        )
        assert len(result) == 1

    def test_merge_concat_sequence(self):
        # cubes require concat, merge won't work
        cubes = [cu(t=[1, 2]), cu(t=[3, 4])]
        result = combine_cubes(cubes)
        assert len(result) == 2
        # .. but will if you put concat in the sequence
        result = combine_cubes(cubes, merge_concat_sequence="c")
        assert len(result) == 1

    def test_merge_unique(self):
        # two identical cubes
        cubes = [cu("myname"), cu("myname")]
        # the combine call (with merge) is OK if we *don't* insist on unique cubes
        combine_cubes(cubes)
        # .. but it errors if we *do* insist on uniqueness
        msg = "Duplicate 'myname' cube"
        with pytest.raises(DuplicateDataError, match=msg):
            combine_cubes(cubes, merge_unique=True)

    def test_repeat_until_unchanged(self):
        # construct a case that will only merge once it was previously concatenated
        cubes = [
            cu(t=[1, 2, 3], z=1),
            cu(t=[4, 5], z=1),
            cu(t=[1, 2], z=2),
            cu(t=[3, 4, 5], z=2),
        ]
        result = combine_cubes(cubes, merge_concat_sequence="mc")
        assert len(result) == 2
        result = combine_cubes(
            cubes, merge_concat_sequence="mc", repeat_until_unchanged=True
        )
        assert len(result) == 1

    # NOTE: "test_support_multiple_references" -- not tested here
    # this may be too hard
    # it is adequately tested in tests/integration/varying_references
