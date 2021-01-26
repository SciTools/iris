# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :mod:`iris._representation.cube_summary` module."""

import iris.tests as tests

import numpy as np
import iris
from iris.coords import AuxCoord, DimCoord
from iris._representation import cube_summary as icr
import iris.tests.stock as istk
from iris.util import new_axis

from iris._representation.cube_printout import CubePrinter


def test_cube(n_extra_dims=0):
    cube = istk.realistic_4d()  # this one has a derived coord

    # Optionally : add multiple extra dimensions to test the width controls
    if n_extra_dims > 0:

        new_cube = cube.copy()
        # Add n extra scalar *1 coords
        for i_dim in range(n_extra_dims):
            dim_name = "long_name_dim_{}".format(i_dim + cube.ndim)
            dimco = DimCoord([0], long_name=dim_name)
            new_cube.add_aux_coord(dimco)
            # Promote to dim coord
            new_cube = new_axis(new_cube, dim_name)

        # Put them all at the back
        dim_order = list(range(new_cube.ndim))
        dim_order = dim_order[n_extra_dims:] + dim_order[:n_extra_dims]
        new_cube.transpose(dim_order)  # dontcha hate this inplace way ??

        # Replace the original test cube
        cube = new_cube

    # Add extra things to test all aspects
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
    co_lat, co_lon = cube.coord(axis="y"), cube.coord(axis="x")
    latlon_dims = cube.coord_dims(co_lat) + cube.coord_dims(co_lon)
    cube.add_aux_coord(
        AuxCoord(trulons, standard_name="longitude", units="degrees"),
        latlon_dims,
    )
    cube.add_aux_coord(
        AuxCoord(trulats, standard_name="latitude", units="degrees"),
        latlon_dims,
    )

    cube.attributes[
        "history"
    ] = "Exceedingly and annoying long message with many sentences.  And more and more.  And more and more."

    cube.add_cell_method(iris.coords.CellMethod("mean", ["time"]))
    cube.add_cell_method(
        iris.coords.CellMethod(
            "max", ["latitude"], intervals="3 hour", comments="remark"
        )
    )
    latlons_shape = [cube.shape[i_dim] for i_dim in latlon_dims]
    cube.add_cell_measure(
        iris.coords.CellMeasure(
            np.zeros(latlons_shape), long_name="cell-timings", units="s"
        ),
        latlon_dims,
    )
    cube.add_cell_measure(
        iris.coords.CellMeasure(
            [4.3], long_name="whole_cell_factor", units="m^2"
        ),
        (),
    )  # a SCALAR cell-measure

    time_dim = cube.coord_dims(cube.coord(axis="t"))
    cube.add_ancillary_variable(
        iris.coords.AncillaryVariable(
            np.zeros(cube.shape[0]), long_name="time_scalings", units="ppm"
        ),
        time_dim,
    )
    cube.add_ancillary_variable(
        iris.coords.AncillaryVariable(
            [3.2], long_name="whole_cube_area_factor", units="m^2"
        ),
        (),
    )  # a SCALAR ancillary

    # Add some duplicate-named coords (not possible for dim-coords)
    vector_duplicate_name = "level_height"
    co_orig = cube.coord(vector_duplicate_name)
    dim_orig = cube.coord_dims(co_orig)
    co_new = co_orig.copy()
    co_new.attributes.update(dict(a=1, b=2))
    cube.add_aux_coord(co_new, dim_orig)

    vector_different_name = "sigma"
    co_orig = cube.coord(vector_different_name)
    co_orig.attributes["setting"] = "a"
    dim_orig = cube.coord_dims(co_orig)
    co_new = co_orig.copy()
    co_new.attributes["setting"] = "B"
    cube.add_aux_coord(co_new, dim_orig)

    # Also need to test this with a SCALAR coord
    scalar_duplicate_name = "forecast_period"
    co_orig = cube.coord(scalar_duplicate_name)
    co_new = co_orig.copy()
    co_new.points = co_new.points + 2.3
    co_new.attributes["different"] = "True"
    cube.add_aux_coord(co_new)

    # Add a scalar coord with a *really* long name, to challenge the column width formatting
    long_name = "long_long_long_long_long_long_long_long_long_long_long_name"
    cube.add_aux_coord(DimCoord([0], long_name=long_name))
    return cube


class TestCubePrintout(tests.IrisTest):
    def _exercise_methods(self, cube):
        summ = icr.CubeSummary(cube)
        printer = CubePrinter(summ, max_width=110)
        has_scalar_ancils = any(
            len(anc.cube_dims(cube)) == 0 for anc in cube.ancillary_variables()
        )
        unprintable = has_scalar_ancils and cube.ndim == 0
        print("EXISTING full :")
        if unprintable:
            print("  ( would fail, due to scalar-cube with scalar-ancils )")
        else:
            print(cube)
        print("---full--- :")
        print(printer.to_string(max_width=120))
        print("")
        print("EXISTING oneline :")
        print(repr(cube))
        print("---oneline--- :")
        print(printer.to_string(oneline=True))
        print("")
        print("original table form:")
        tb = printer.table
        tb.maxwidth = 140
        print(tb)
        print("")
        print("")

    def test_basic(self):
        cube = test_cube(
            n_extra_dims=4
        )  # NB does not yet work with factories.
        self._exercise_methods(cube)

    def test_scalar_cube(self):
        cube = test_cube()[0, 0, 0, 0]
        self._exercise_methods(cube)
