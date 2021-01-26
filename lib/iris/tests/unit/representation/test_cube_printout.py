# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :mod:`iris._representation.cube_summary` module."""

import iris.tests as tests

import numpy as np
import iris
from iris.cube import Cube
from iris.coords import AuxCoord, DimCoord
from iris._representation import cube_summary as icr
import iris.tests.stock as istk


from iris._representation.cube_printout import CubePrinter


def test_cube(n_extra_dims=0):
    cube = istk.realistic_3d()

    # Add multiple extra dimensions to test the width controls
    if n_extra_dims > 0:
        new_dims = cube.shape + (1,) * n_extra_dims
        new_data = cube.data.reshape(new_dims)
        new_cube = Cube(new_data)
        for i_dim in range(new_cube.ndim):
            if i_dim < cube.ndim:
                dimco = cube.coord(dimensions=i_dim, dim_coords=True)
            else:
                dim_name = "long_name_dim_{}".format(i_dim)
                dimco = DimCoord([0], long_name=dim_name)
            new_cube.add_dim_coord(dimco, i_dim)

        # Copy all aux coords too
        for co, dims in cube._aux_coords_and_dims:
            new_cube.add_aux_coord(co, dims)

        # Copy attributes
        new_cube.attributes = cube.attributes

        # Nothing else to copy ?
        assert not cube.ancillary_variables()
        assert not cube.cell_measures()  # Includes scalar ones
        assert not cube.aux_factories

        cube = new_cube

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
        cube = test_cube(n_extra_dims=4)
        summ = icr.CubeSummary(cube)
        printer = CubePrinter(summ, max_width=110)
        print("EXISTING full :")
        print(cube)
        print("---full--- :")
        print(printer.to_string(max_width=80))
        print("")
        print("EXISTING oneline :")
        print(repr(cube))
        print("---oneline--- :")
        print(printer.to_string(oneline=True))
        print("")
        print("original table form:")
        print(printer.table)
