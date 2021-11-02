# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for :func:`iris.fileformats.rules.load_cubes`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import iris
from iris.fileformats import pp
from iris.fileformats.pp_load_rules import convert
from iris.fileformats.rules import load_cubes


class Test(tests.IrisTest):
    @tests.skip_data
    def test_pp_with_stash_constraint(self):
        filenames = [tests.get_data_path(("PP", "globClim1", "dec_subset.pp"))]
        stcon = iris.AttributeConstraint(STASH="m01s00i004")
        pp_constraints = pp._convert_constraints(stcon)
        pp_loader = iris.fileformats.rules.Loader(pp.load, {}, convert)
        cubes = list(load_cubes(filenames, None, pp_loader, pp_constraints))
        self.assertEqual(len(cubes), 38)

    @tests.skip_data
    def test_pp_with_stash_constraints(self):
        filenames = [tests.get_data_path(("PP", "globClim1", "dec_subset.pp"))]
        stcon1 = iris.AttributeConstraint(STASH="m01s00i004")
        stcon2 = iris.AttributeConstraint(STASH="m01s00i010")
        pp_constraints = pp._convert_constraints([stcon1, stcon2])
        pp_loader = iris.fileformats.rules.Loader(pp.load, {}, convert)
        cubes = list(load_cubes(filenames, None, pp_loader, pp_constraints))
        self.assertEqual(len(cubes), 76)

    @tests.skip_data
    def test_pp_no_constraint(self):
        filenames = [tests.get_data_path(("PP", "globClim1", "dec_subset.pp"))]
        pp_constraints = pp._convert_constraints(None)
        pp_loader = iris.fileformats.rules.Loader(pp.load, {}, convert)
        cubes = list(load_cubes(filenames, None, pp_loader, pp_constraints))
        self.assertEqual(len(cubes), 152)


if __name__ == "__main__":
    tests.main()
