# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for :func:`iris.fileformats.rules.load_cubes`."""

import iris
from iris.fileformats import pp
from iris.fileformats.pp_load_rules import convert
from iris.fileformats.rules import load_cubes
from iris.tests import _shared_utils


class Test:
    @_shared_utils.skip_data
    def test_pp_with_stash_constraint(self):
        filenames = [_shared_utils.get_data_path(("PP", "globClim1", "dec_subset.pp"))]
        stcon = iris.AttributeConstraint(STASH="m01s00i004")
        pp_constraints = pp._convert_constraints(stcon)
        pp_loader = iris.fileformats.rules.Loader(pp.load, {}, convert)
        cubes = list(load_cubes(filenames, None, pp_loader, pp_constraints))
        assert len(cubes) == 38

    @_shared_utils.skip_data
    def test_pp_with_stash_constraints(self):
        filenames = [_shared_utils.get_data_path(("PP", "globClim1", "dec_subset.pp"))]
        stcon1 = iris.AttributeConstraint(STASH="m01s00i004")
        stcon2 = iris.AttributeConstraint(STASH="m01s00i010")
        pp_constraints = pp._convert_constraints([stcon1, stcon2])
        pp_loader = iris.fileformats.rules.Loader(pp.load, {}, convert)
        cubes = list(load_cubes(filenames, None, pp_loader, pp_constraints))
        assert len(cubes) == 76

    @_shared_utils.skip_data
    def test_pp_no_constraint(self):
        filenames = [_shared_utils.get_data_path(("PP", "globClim1", "dec_subset.pp"))]
        pp_constraints = pp._convert_constraints(None)
        pp_loader = iris.fileformats.rules.Loader(pp.load, {}, convert)
        cubes = list(load_cubes(filenames, None, pp_loader, pp_constraints))
        assert len(cubes) == 152
