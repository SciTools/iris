# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :mod:`iris._representation.cube_summary` module."""

import iris.tests as tests

import numpy as np
import iris
from iris.coords import AuxCoord
from iris._representation import cube_summary as icr
import iris.tests.stock as istk


from iris._representation.cube_printout import CubePrinter


def test_cube():
    cube = istk.realistic_3d()
    # print(cube)
    rotlats_1d, rotlons_1d = (
        cube.coord("grid_latitude").points,
        cube.coord("grid_longitude").points,
    )
    rotlons_2d, rotlats_2d = np.meshgrid(rotlons_1d, rotlats_1d)

    cs = cube.coord_system()
    trulons, trulats = iris.analysis.cartography.unrotate_pole(
        rotlons_2d,
        rotlats_2d,
        cs.grid_north_pole_longitude,
        cs.grid_north_pole_latitude,
    )
    cube.add_aux_coord(
        AuxCoord(trulons, standard_name="longitude", units="degrees"), (1, 2)
    )
    cube.add_aux_coord(
        AuxCoord(trulats, standard_name="latitude", units="degrees"), (1, 2)
    )

    cube.attributes[
        "history"
    ] = "Exceedingly and annoying long message with many sentences.  And more and more.  And more and more."

    return cube


class TestCubePrintout(tests.IrisTest):
    def test_basic(self):
        cube = test_cube()
        summ = icr.CubeSummary(cube)
        printer = CubePrinter(summ)
        print("full:")
        print(cube)
        print(printer.to_string())
        print("")
        print("oneline:")
        print(repr(cube))
        print(printer.to_string(oneline=True))
