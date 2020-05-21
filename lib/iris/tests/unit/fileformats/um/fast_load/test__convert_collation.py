# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for :func:`iris.fileformats.um._fast_load._convert_collation`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from unittest import mock

import cf_units
import cftime
import numpy as np

from iris.fileformats.um._fast_load import (
    _convert_collation as convert_collation,
)
import iris.aux_factory
import iris.coord_systems
import iris.coords
import iris.fileformats.pp
import iris.fileformats.rules


COORD_SYSTEM = iris.coord_systems.GeogCS(6371229.0)
LATITUDE = iris.coords.DimCoord(
    [15, 0, -15], "latitude", units="degrees", coord_system=COORD_SYSTEM
)
LONGITUDE = iris.coords.DimCoord(
    [0, 20, 40, 60],
    "longitude",
    units="degrees",
    coord_system=COORD_SYSTEM,
    circular=True,
)


class Test(tests.IrisTest):
    def _field(self):
        # Create PP field for X wind on a regular lat-lon grid.
        header = [0] * 64
        # Define the regular lat-lon grid.
        header[15] = 1  # LBCODE
        header[17] = 3  # LBROW
        header[18] = 4  # LBNPT
        header[55] = 90  # BPLAT
        header[58] = 30  # BZY
        header[59] = -15  # BDY
        header[60] = -20  # BZX
        header[61] = 20  # BDX
        # Define the STASH code m01s00i002.
        header[41] = 2  # LBUSER(4)
        header[44] = 1  # LBUSER(7)
        field = iris.fileformats.pp.PPField3(header)
        return field

    def _check_phenomenon(self, metadata, factory=None):
        if factory is None:
            self.assertEqual(metadata.factories, [])
        else:
            self.assertEqual(metadata.factories, [factory])
        self.assertEqual(metadata.references, [])
        self.assertEqual(metadata.standard_name, "x_wind")
        self.assertIsNone(metadata.long_name)
        self.assertEqual(metadata.units, cf_units.Unit("m s-1"))
        self.assertEqual(metadata.attributes, {"STASH": (1, 0, 2)})
        self.assertEqual(metadata.cell_methods, [])

    def test_all_scalar(self):
        field = self._field()
        field.lbtim = 11
        field.t1 = cftime.datetime(1970, 1, 1, 18)
        field.t2 = cftime.datetime(1970, 1, 1, 12)
        collation = mock.Mock(
            fields=[field], vector_dims_shape=(), element_arrays_and_dims={}
        )
        metadata = convert_collation(collation)
        self._check_phenomenon(metadata)
        coords_and_dims = [(LONGITUDE, 1), (LATITUDE, 0)]
        self.assertEqual(metadata.dim_coords_and_dims, coords_and_dims)
        coords_and_dims = [
            (
                iris.coords.DimCoord(18, "time", units="hours since epoch"),
                None,
            ),
            (
                iris.coords.DimCoord(
                    12, "forecast_reference_time", units="hours since epoch"
                ),
                None,
            ),
            (iris.coords.DimCoord(6, "forecast_period", units="hours"), None),
        ]
        self.assertEqual(metadata.aux_coords_and_dims, coords_and_dims)

    def test_vector_t1(self):
        field = self._field()
        field.lbtim = 11
        field.t2 = cftime.datetime(1970, 1, 1, 12)
        t1 = (
            [
                cftime.datetime(1970, 1, 1, 18),
                cftime.datetime(1970, 1, 2, 0),
                cftime.datetime(1970, 1, 2, 6),
            ],
            [0],
        )
        collation = mock.Mock(
            fields=[field],
            vector_dims_shape=(3,),
            element_arrays_and_dims={"t1": t1},
        )
        metadata = convert_collation(collation)
        self._check_phenomenon(metadata)
        coords_and_dims = [
            (LONGITUDE, 2),
            (LATITUDE, 1),
            (
                iris.coords.DimCoord(
                    [18, 24, 30], "time", units="hours since epoch"
                ),
                (0,),
            ),
        ]
        self.assertEqual(metadata.dim_coords_and_dims, coords_and_dims)
        coords_and_dims = [
            (
                iris.coords.DimCoord(
                    12, "forecast_reference_time", units="hours since epoch"
                ),
                None,
            ),
            (
                iris.coords.DimCoord(
                    [6, 12, 18], "forecast_period", units="hours"
                ),
                (0,),
            ),
        ]
        self.assertEqual(metadata.aux_coords_and_dims, coords_and_dims)

    def test_vector_t2(self):
        field = self._field()
        field.lbtim = 11
        field.t1 = cftime.datetime(1970, 1, 1, 18)
        t2 = (
            [
                cftime.datetime(1970, 1, 1, 12),
                cftime.datetime(1970, 1, 1, 15),
                cftime.datetime(1970, 1, 1, 18),
            ],
            [0],
        )
        collation = mock.Mock(
            fields=[field],
            vector_dims_shape=(3,),
            element_arrays_and_dims={"t2": t2},
        )
        metadata = convert_collation(collation)
        self._check_phenomenon(metadata)
        coords_and_dims = [
            (LONGITUDE, 2),
            (LATITUDE, 1),
            (
                iris.coords.DimCoord(
                    [12, 15, 18],
                    "forecast_reference_time",
                    units="hours since epoch",
                ),
                (0,),
            ),
        ]
        self.assertEqual(metadata.dim_coords_and_dims, coords_and_dims)
        coords_and_dims = [
            (
                iris.coords.DimCoord(18, "time", units="hours since epoch"),
                None,
            ),
            (
                iris.coords.DimCoord(
                    [6, 3, 0.0], "forecast_period", units="hours"
                ),
                (0,),
            ),
        ]
        self.assertEqual(metadata.aux_coords_and_dims, coords_and_dims)

    def test_vector_lbft(self):
        field = self._field()
        field.lbtim = 21
        field.t1 = cftime.datetime(1970, 1, 1, 12)
        field.t2 = cftime.datetime(1970, 1, 1, 18)
        lbft = ([18, 15, 12], [0])
        collation = mock.Mock(
            fields=[field],
            vector_dims_shape=(3,),
            element_arrays_and_dims={"lbft": lbft},
        )
        metadata = convert_collation(collation)
        self._check_phenomenon(metadata)
        coords_and_dims = [
            (LONGITUDE, 2),
            (LATITUDE, 1),
            (
                iris.coords.DimCoord(
                    [0, 3, 6],
                    "forecast_reference_time",
                    units="hours since epoch",
                ),
                (0,),
            ),
        ]
        coords_and_dims = [
            (
                iris.coords.DimCoord(
                    15, "time", units="hours since epoch", bounds=[[12, 18]]
                ),
                None,
            ),
            (
                iris.coords.DimCoord(
                    [15, 12, 9],
                    "forecast_period",
                    units="hours",
                    bounds=[[12, 18], [9, 15], [6, 12]],
                ),
                (0,),
            ),
        ]
        self.assertEqual(metadata.aux_coords_and_dims, coords_and_dims)

    def test_vector_t1_and_t2(self):
        field = self._field()
        field.lbtim = 11
        t1 = (
            [
                cftime.datetime(1970, 1, 2, 6),
                cftime.datetime(1970, 1, 2, 9),
                cftime.datetime(1970, 1, 2, 12),
            ],
            [1],
        )
        t2 = (
            [cftime.datetime(1970, 1, 1, 12), cftime.datetime(1970, 1, 2, 0)],
            [0],
        )
        collation = mock.Mock(
            fields=[field],
            vector_dims_shape=(2, 3),
            element_arrays_and_dims={"t1": t1, "t2": t2},
        )
        metadata = convert_collation(collation)
        self._check_phenomenon(metadata)
        coords_and_dims = [
            (LONGITUDE, 3),
            (LATITUDE, 2),
            (
                iris.coords.DimCoord(
                    [30, 33, 36], "time", units="hours since epoch"
                ),
                (1,),
            ),
            (
                iris.coords.DimCoord(
                    [12, 24],
                    "forecast_reference_time",
                    units="hours since epoch",
                ),
                (0,),
            ),
        ]
        self.assertEqual(metadata.dim_coords_and_dims, coords_and_dims)
        coords_and_dims = [
            (
                iris.coords.AuxCoord(
                    [[18, 21, 24], [6, 9, 12]],
                    "forecast_period",
                    units="hours",
                ),
                (0, 1),
            )
        ]
        self.assertEqual(metadata.aux_coords_and_dims, coords_and_dims)

    def test_vertical_pressure(self):
        field = self._field()
        field.lbvc = 8
        blev = ([1000, 850, 700], (0,))
        lblev = ([1000, 850, 700], (0,))
        collation = mock.Mock(
            fields=[field],
            vector_dims_shape=(3,),
            element_arrays_and_dims={"blev": blev, "lblev": lblev},
        )
        metadata = convert_collation(collation)
        self._check_phenomenon(metadata)
        coords_and_dims = [
            (LONGITUDE, 2),
            (LATITUDE, 1),
            (
                iris.coords.DimCoord(
                    [1000, 850, 700], long_name="pressure", units="hPa"
                ),
                (0,),
            ),
        ]
        self.assertEqual(metadata.dim_coords_and_dims, coords_and_dims)
        coords_and_dims = []
        self.assertEqual(metadata.aux_coords_and_dims, coords_and_dims)

    def test_soil_level(self):
        field = self._field()
        field.lbvc = 6
        points = [10, 20, 30]
        lower = [0] * 3
        upper = [0] * 3
        lblev = (points, (0,))
        brsvd1 = (lower, (0,))
        brlev = (upper, (0,))
        collation = mock.Mock(
            fields=[field],
            vector_dims_shape=(3,),
            element_arrays_and_dims={
                "lblev": lblev,
                "brsvd1": brsvd1,
                "brlev": brlev,
            },
        )
        metadata = convert_collation(collation)
        self._check_phenomenon(metadata)
        level = iris.coords.DimCoord(
            points,
            long_name="soil_model_level_number",
            attributes={"positive": "down"},
            units="1",
        )
        coords_and_dims = [(LONGITUDE, 2), (LATITUDE, 1), (level, (0,))]
        self.assertEqual(metadata.dim_coords_and_dims, coords_and_dims)
        coords_and_dims = []
        self.assertEqual(metadata.aux_coords_and_dims, coords_and_dims)

    def test_soil_depth(self):
        field = self._field()
        field.lbvc = 6
        points = [10, 20, 30]
        lower = [0, 15, 25]
        upper = [15, 25, 35]
        blev = (points, (0,))
        brsvd1 = (lower, (0,))
        brlev = (upper, (0,))
        collation = mock.Mock(
            fields=[field],
            vector_dims_shape=(3,),
            element_arrays_and_dims={
                "blev": blev,
                "brsvd1": brsvd1,
                "brlev": brlev,
            },
        )
        metadata = convert_collation(collation)
        self._check_phenomenon(metadata)
        depth = iris.coords.DimCoord(
            points,
            standard_name="depth",
            bounds=np.vstack((lower, upper)).T,
            units="m",
            attributes={"positive": "down"},
        )
        coords_and_dims = [(LONGITUDE, 2), (LATITUDE, 1), (depth, (0,))]
        self.assertEqual(metadata.dim_coords_and_dims, coords_and_dims)
        coords_and_dims = []
        self.assertEqual(metadata.aux_coords_and_dims, coords_and_dims)

    def test_vertical_hybrid_height(self):
        field = self._field()
        field.lbvc = 65
        blev = ([5, 18, 38], (0,))
        lblev = ([1000, 850, 700], (0,))
        brsvd1 = ([10, 26, 50], (0,))
        brsvd2 = ([0.9989, 0.9970, 0.9944], (0,))
        brlev = ([0, 10, 26], (0,))
        bhrlev = ([1, 0.9989, 0.9970], (0,))
        lblev = ([1, 2, 3], (0,))
        bhlev = ([0.9994, 0.9979, 0.9957], (0,))
        collation = mock.Mock(
            fields=[field],
            vector_dims_shape=(3,),
            element_arrays_and_dims={
                "blev": blev,
                "lblev": lblev,
                "brsvd1": brsvd1,
                "brsvd2": brsvd2,
                "brlev": brlev,
                "bhrlev": bhrlev,
                "lblev": lblev,
                "bhlev": bhlev,
            },
        )
        metadata = convert_collation(collation)
        factory = iris.fileformats.rules.Factory(
            iris.aux_factory.HybridHeightFactory,
            [
                {"long_name": "level_height"},
                {"long_name": "sigma"},
                iris.fileformats.rules.Reference("orography"),
            ],
        )
        self._check_phenomenon(metadata, factory)
        coords_and_dims = [
            (LONGITUDE, 2),
            (LATITUDE, 1),
            (
                iris.coords.DimCoord(
                    [1, 2, 3],
                    "model_level_number",
                    attributes={"positive": "up"},
                    units="1",
                ),
                (0,),
            ),
        ]
        self.assertEqual(metadata.dim_coords_and_dims, coords_and_dims)
        coords_and_dims = [
            (
                iris.coords.DimCoord(
                    [5, 18, 38],
                    long_name="level_height",
                    units="m",
                    bounds=[[0, 10], [10, 26], [26, 50]],
                    attributes={"positive": "up"},
                ),
                (0,),
            ),
            (
                iris.coords.AuxCoord(
                    [0.9994, 0.9979, 0.9957],
                    long_name="sigma",
                    bounds=[[1, 0.9989], [0.9989, 0.9970], [0.9970, 0.9944]],
                    units="1",
                ),
                (0,),
            ),
        ]
        self.assertEqual(metadata.aux_coords_and_dims, coords_and_dims)


if __name__ == "__main__":
    tests.main()
