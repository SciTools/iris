# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.coords._DimensionalMetadata` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip


from cf_units import Unit
import numpy as np

import iris._lazy_data as lazy
from iris.coord_systems import GeogCS
from iris.coords import (
    AncillaryVariable,
    AuxCoord,
    CellMeasure,
    DimCoord,
    _DimensionalMetadata,
)
from iris.experimental.ugrid.mesh import Connectivity
from iris.tests.stock import climatology_3d as cube_with_climatology
from iris.tests.stock.mesh import sample_meshcoord


class Test___init____abstractmethod(tests.IrisTest):
    def test(self):
        emsg = "Can't instantiate abstract class _DimensionalMetadata"
        with self.assertRaisesRegex(TypeError, emsg):
            _ = _DimensionalMetadata(0)


class Mixin__string_representations:
    """Common testcode for generic `__str__`, `__repr__` and `summary` methods.

    Effectively, __str__ and __repr__ are thin wrappers around `summary`.
    These are used by all the subclasses : notably Coord/DimCoord/AuxCoord,
    but also AncillaryVariable, CellMeasure and MeshCoord.

    There are a lot of different aspects to consider:

    * different object classes with different class-specific properties
    * changing with array sizes + dimensionalities
    * masked data
    * data types : int, float, string and (special) dates
    * for Coords, handling of bounds
    * "summary" controls (also can be affected by numpy printoptions).

    NOTE: since the details of formatting are important to us here, the basic
    test method is to check printout results against an exact 'snapshot'
    embedded (visibly) in the test itself.

    """

    def repr_str_strings(self, dm, linewidth=55):
        """Return a simple combination of repr and str printouts.

        N.B. we control linewidth to make the outputs easier to compare.
        """
        with np.printoptions(linewidth=linewidth):
            result = repr(dm) + "\n" + str(dm)
        return result

    def sample_data(self, datatype=float, units="m", shape=(5,), masked=False):
        """Make a sample data array for a test _DimensionalMetadata object."""
        # Get an actual Unit
        units = Unit(units)
        if units.calendar:
            # fix string datatypes for date-based units
            datatype = float

        # Get a dtype
        dtype = np.dtype(datatype)

        # Make suitable test values for type/shape/masked
        length = int(np.prod(shape))
        if dtype.kind == "U":
            # String content.
            digit_strs = [str(i) * (i + 1) for i in range(0, 10)]
            if length < 10:
                # ['0', '11', '222, '3333', ..]
                values = np.array(digit_strs[:length])
            else:
                # [... '9999999999', '0', '11' ....]
                indices = [(i % 10) for i in range(length)]
                values = np.array(digit_strs)[indices]
        else:
            # numeric content : a simple [0, 1, 2 ...]
            values = np.arange(length).astype(dtype)

        if masked:
            if np.prod(shape) >= 3:
                # Mask 1 in 3 points : [x -- x x -- x ...]
                i_firstmasked = 1
            else:
                # Few points, mask 1 in 3 starting at 0 [-- x x -- x x -- ...]
                i_firstmasked = 0
            masked_points = [(i % 3) == i_firstmasked for i in range(length)]
            values = np.ma.masked_array(values, mask=masked_points)

        values = values.reshape(shape)
        return values

    # Make a sample Coord, as _DimensionalMetadata is abstract and this is the
    # obvious concrete subclass to use for testing
    def sample_coord(
        self,
        datatype=float,
        dates=False,
        units="m",
        long_name="x",
        shape=(5,),
        masked=False,
        bounded=False,
        dimcoord=False,
        lazy_points=False,
        lazy_bounds=False,
        *coord_args,
        **coord_kwargs,
    ):
        if masked:
            dimcoord = False
        if dates:
            # Use a pre-programmed date unit.
            units = Unit("days since 1970-03-5")
        if not isinstance(units, Unit):
            # This operation is *not* a no-op, it will wipe calendars !
            units = Unit(units)
        values = self.sample_data(
            datatype=datatype, units=units, shape=shape, masked=masked
        )
        cls = DimCoord if dimcoord else AuxCoord
        coord = cls(
            points=values,
            units=units,
            long_name=long_name,
            *coord_args,
            **coord_kwargs,
        )
        if bounded or lazy_bounds:
            if shape == (1,):
                # Guess-bounds doesn't work !
                val = coord.points[0]
                bounds = [val - 10, val + 10]
                # NB preserve masked/unmasked : avoid converting masks to NaNs
                if np.ma.isMaskedArray(coord.points):
                    array = np.ma.array
                else:
                    array = np.array
                coord.bounds = array(bounds)
            else:
                coord.guess_bounds()
        if lazy_points:
            coord.points = lazy.as_lazy_data(coord.points)
        if lazy_bounds:
            coord.bounds = lazy.as_lazy_data(coord.bounds)
        return coord

    def coord_representations(self, *args, **kwargs):
        """Create a test coord and return its string representations.

        Pass args+kwargs to 'sample_coord' and return the 'repr_str_strings'.

        """
        coord = self.sample_coord(*args, **kwargs)
        return self.repr_str_strings(coord)

    def assertLines(self, list_of_expected_lines, string_result):
        r"""Assert equality between a result and expected output lines.

        For convenience, the 'expected lines' are joined with a '\\n',
        because a list of strings is nicer to construct in code.
        They should then match the actual result, which is a simple string.

        """
        self.assertEqual(list_of_expected_lines, string_result.split("\n"))


class Test__print_common(Mixin__string_representations, tests.IrisTest):
    """Test aspects of __str__ and __repr__ output common to all
    _DimensionalMetadata instances.
    I.E. those from CFVariableMixin, plus values array (data-manager).

    Aspects :
    * standard_name:
    * long_name:
    * var_name:
    * attributes
    * units
    * shape
    * dtype

    """

    def test_simple(self):
        result = self.coord_representations()
        expected = [
            "<AuxCoord: x / (m)  [0., 1., 2., 3., 4.]  shape(5,)>",
            "AuxCoord :  x / (m)",
            "    points: [0., 1., 2., 3., 4.]",
            "    shape: (5,)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_minimal(self):
        result = self.coord_representations(long_name=None, units=None, shape=(1,))
        expected = [
            "<AuxCoord: unknown / (unknown)  [0.]>",
            "AuxCoord :  unknown / (unknown)",
            "    points: [0.]",
            "    shape: (1,)",
            "    dtype: float64",
        ]
        self.assertLines(expected, result)

    def test_names(self):
        result = self.coord_representations(
            standard_name="height", long_name="this", var_name="x_var"
        )
        expected = [
            "<AuxCoord: height / (m)  [0., 1., 2., 3., 4.]  shape(5,)>",
            "AuxCoord :  height / (m)",
            "    points: [0., 1., 2., 3., 4.]",
            "    shape: (5,)",
            "    dtype: float64",
            "    standard_name: 'height'",
            "    long_name: 'this'",
            "    var_name: 'x_var'",
        ]
        self.assertLines(expected, result)

    def test_bounded(self):
        result = self.coord_representations(shape=(3,), bounded=True)
        expected = [
            "<AuxCoord: x / (m)  [0., 1., 2.]+bounds  shape(3,)>",
            "AuxCoord :  x / (m)",
            "    points: [0., 1., 2.]",
            "    bounds: [",
            "        [-0.5,  0.5],",
            "        [ 0.5,  1.5],",
            "        [ 1.5,  2.5]]",
            "    shape: (3,)  bounds(3, 2)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_masked(self):
        result = self.coord_representations(masked=True)
        expected = [
            "<AuxCoord: x / (m)  [0.0, -- , 2.0, 3.0, -- ]  shape(5,)>",
            "AuxCoord :  x / (m)",
            "    points: [0.0, -- , 2.0, 3.0, -- ]",
            "    shape: (5,)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_dtype_int(self):
        result = self.coord_representations(units="1", datatype=np.int16)
        expected = [
            "<AuxCoord: x / (1)  [0, 1, 2, 3, 4]  shape(5,)>",
            "AuxCoord :  x / (1)",
            "    points: [0, 1, 2, 3, 4]",
            "    shape: (5,)",
            "    dtype: int16",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_dtype_date(self):
        # Note: test with a date 'longer' than the built-in one in
        # 'sample_coord(dates=True)', because it includes a time-of-day
        full_date_unit = Unit("days since 1892-05-17 03:00:25", calendar="360_day")
        result = self.coord_representations(units=full_date_unit)
        expected = [
            ("<AuxCoord: x / (days since 1892-05-17 03:00:25)  [...]  shape(5,)>"),
            ("AuxCoord :  x / (days since 1892-05-17 03:00:25, 360_day calendar)"),
            "    points: [",
            "        1892-05-17 03:00:25, 1892-05-18 03:00:25,",
            "        1892-05-19 03:00:25, 1892-05-20 03:00:25,",
            "        1892-05-21 03:00:25]",
            "    shape: (5,)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_attributes(self):
        # NOTE: scheduled for future change, to put each attribute on a line
        coord = self.sample_coord(
            attributes={
                "array": np.arange(7.0),
                "list": [1, 2, 3],
                "empty": [],
                "None": None,
                "string": "this",
                "long_long_long_long_long_name": 3,
                "other": (
                    "long_long_long_long_long_long_long_long_"
                    "long_long_long_long_long_long_long_long_value"
                ),
                "float": 4.3,
            }
        )
        result = self.repr_str_strings(coord)
        expected = [
            "<AuxCoord: x / (m)  [0., 1., 2., 3., 4.]  shape(5,)>",
            "AuxCoord :  x / (m)",
            "    points: [0., 1., 2., 3., 4.]",
            "    shape: (5,)",
            "    dtype: float64",
            "    long_name: 'x'",
            "    attributes:",
            "        array                          [0. 1. 2. 3. 4. 5. 6.]",
            "        list                           [1, 2, 3]",
            "        empty                          []",
            "        None                           None",
            "        string                         'this'",
            "        long_long_long_long_long_name  3",
            (
                "        other                          "
                "'long_long_long_long_long_long_long_long_"
                "long_long_long_long_long_long..."
            ),
            "        float                          4.3",
        ]
        self.assertLines(expected, result)

    def test_lazy_points(self):
        result = self.coord_representations(lazy_points=True)
        expected = [
            "<AuxCoord: x / (m)  <lazy>  shape(5,)>",
            "AuxCoord :  x / (m)",
            "    points: <lazy>",
            "    shape: (5,)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_lazy_bounds(self):
        result = self.coord_representations(lazy_bounds=True)
        expected = [
            "<AuxCoord: x / (m)  [0., 1., 2., 3., 4.]+bounds  shape(5,)>",
            "AuxCoord :  x / (m)",
            "    points: [0., 1., 2., 3., 4.]",
            "    bounds: <lazy>",
            "    shape: (5,)  bounds(5, 2)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_lazy_points_and_bounds(self):
        result = self.coord_representations(lazy_points=True, lazy_bounds=True)
        expected = [
            "<AuxCoord: x / (m)  <lazy>+bounds  shape(5,)>",
            "AuxCoord :  x / (m)",
            "    points: <lazy>",
            "    bounds: <lazy>",
            "    shape: (5,)  bounds(5, 2)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_scalar(self):
        result = self.coord_representations(shape=(1,), bounded=True)
        expected = [
            "<AuxCoord: x / (m)  [0.]+bounds>",
            "AuxCoord :  x / (m)",
            "    points: [0.]",
            "    bounds: [[-10.,  10.]]",
            "    shape: (1,)  bounds(1, 2)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_scalar_masked(self):
        result = self.coord_representations(shape=(1,), bounded=True, masked=True)
        expected = [
            "<AuxCoord: x / (m)  [--]+bounds>",
            "AuxCoord :  x / (m)",
            "    points: [--]",
            "    bounds: [[--, --]]",
            "    shape: (1,)  bounds(1, 2)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_length_short(self):
        result = self.coord_representations(shape=(2,), bounded=True)
        expected = [
            "<AuxCoord: x / (m)  [0., 1.]+bounds  shape(2,)>",
            "AuxCoord :  x / (m)",
            "    points: [0., 1.]",
            "    bounds: [",
            "        [-0.5,  0.5],",
            "        [ 0.5,  1.5]]",
            "    shape: (2,)  bounds(2, 2)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_length_medium(self):
        # Where bounds are truncated, but points not.
        result = self.coord_representations(shape=(14,), bounded=True)
        expected = [
            "<AuxCoord: x / (m)  [ 0., 1., ..., 12., 13.]+bounds  shape(14,)>",
            "AuxCoord :  x / (m)",
            "    points: [",
            "         0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,",
            "         9., 10., 11., 12., 13.]",
            "    bounds: [",
            "        [-0.5,  0.5],",
            "        [ 0.5,  1.5],",
            "        ...,",
            "        [11.5, 12.5],",
            "        [12.5, 13.5]]",
            "    shape: (14,)  bounds(14, 2)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_length_long(self):
        # Completely truncated representations
        result = self.coord_representations(shape=(150,), bounded=True)
        expected = [
            ("<AuxCoord: x / (m)  [ 0., 1., ..., 148., 149.]+bounds  shape(150,)>"),
            "AuxCoord :  x / (m)",
            "    points: [  0.,   1., ..., 148., 149.]",
            "    bounds: [",
            "        [ -0.5,   0.5],",
            "        [  0.5,   1.5],",
            "        ...,",
            "        [147.5, 148.5],",
            "        [148.5, 149.5]]",
            "    shape: (150,)  bounds(150, 2)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_strings(self):
        result = self.coord_representations(datatype=str)
        expected = [
            "<AuxCoord: x / (m)  [0 , 11 , 222 , 3333 , 44444]  shape(5,)>",
            "AuxCoord :  x / (m)",
            "    points: [0    , 11   , 222  , 3333 , 44444]",
            "    shape: (5,)",
            "    dtype: <U5",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_strings_long_aligned(self):
        result = self.coord_representations(datatype=str, shape=(15,))
        expected = [
            "<AuxCoord: x / (m)  [0 , 11 , ..., 3333 , 44444 ]  shape(15,)>",
            "AuxCoord :  x / (m)",
            "    points: [",
            "        0         , 11        , 222       ,",
            "        3333      , 44444     , 555555    ,",
            "        6666666   , 77777777  , 888888888 ,",
            "        9999999999, 0         , 11        ,",
            "        222       , 3333      , 44444     ]",
            "    shape: (15,)",
            "    dtype: <U10",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_onepoint_toolong_placeholder(self):
        # When even one point won't display, get "[...]".
        # This applies to dates here, but only because we reduced linewidth
        result = self.coord_representations(shape=(2,), dates=True)
        expected = [
            "<AuxCoord: x / (days since 1970-03-5)  [...]  shape(2,)>",
            "AuxCoord :  x / (days since 1970-03-5, standard calendar)",
            "    points: [1970-03-05 00:00:00, 1970-03-06 00:00:00]",
            "    shape: (2,)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_dates_scalar(self):
        # Printouts for a scalar date coord.
        # Demonstrate that a "typical" datetime coord can print with the date
        # value visible in the repr.
        long_time_unit = Unit("hours since 2025-03-23 01:00:00")
        coord = self.sample_coord(
            standard_name="time",
            long_name=None,
            shape=(1,),
            units=long_time_unit,
        )
        # Do this one with a default linewidth, not our default reduced one, so
        # that we can get the date value in the repr output.
        result = self.repr_str_strings(coord, linewidth=None)
        expected = [
            (
                "<AuxCoord: time / (hours since 2025-03-23 01:00:00)  "
                "[2025-03-23 01:00:00]>"
            ),
            (
                "AuxCoord :  time / (hours since 2025-03-23 01:00:00, "
                "standard calendar)"
            ),
            "    points: [2025-03-23 01:00:00]",
            "    shape: (1,)",
            "    dtype: float64",
            "    standard_name: 'time'",
        ]
        self.assertLines(expected, result)

    def test_dates_bounds(self):
        result = self.coord_representations(dates=True, bounded=True)
        expected = [
            "<AuxCoord: x / (days since 1970-03-5)  [...]+bounds  shape(5,)>",
            "AuxCoord :  x / (days since 1970-03-5, standard calendar)",
            "    points: [",
            "        1970-03-05 00:00:00, 1970-03-06 00:00:00,",
            "        1970-03-07 00:00:00, 1970-03-08 00:00:00,",
            "        1970-03-09 00:00:00]",
            "    bounds: [",
            "        [1970-03-04 12:00:00, 1970-03-05 12:00:00],",
            "        [1970-03-05 12:00:00, 1970-03-06 12:00:00],",
            "        [1970-03-06 12:00:00, 1970-03-07 12:00:00],",
            "        [1970-03-07 12:00:00, 1970-03-08 12:00:00],",
            "        [1970-03-08 12:00:00, 1970-03-09 12:00:00]]",
            "    shape: (5,)  bounds(5, 2)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_dates_masked(self):
        result = self.coord_representations(dates=True, masked=True)
        expected = [
            "<AuxCoord: x / (days since 1970-03-5)  [...]  shape(5,)>",
            "AuxCoord :  x / (days since 1970-03-5, standard calendar)",
            "    points: [",
            "        1970-03-05 00:00:00, --                 ,",
            "        1970-03-07 00:00:00, 1970-03-08 00:00:00,",
            "        --                 ]",
            "    shape: (5,)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_untypical_bounds(self):
        # Check printing when n-bounds > 2
        coord = self.sample_coord()
        bounds = coord.points.reshape((5, 1)) + np.array([[-3.0, -2, 2, 3]])
        coord.bounds = bounds
        result = self.repr_str_strings(coord)
        expected = [
            "<AuxCoord: x / (m)  [0., 1., 2., 3., 4.]+bounds  shape(5,)>",
            "AuxCoord :  x / (m)",
            "    points: [0., 1., 2., 3., 4.]",
            "    bounds: [",
            "        [-3., -2.,  2.,  3.],",
            "        [-2., -1.,  3.,  4.],",
            "        ...,",
            "        [ 0.,  1.,  5.,  6.],",
            "        [ 1.,  2.,  6.,  7.]]",
            "    shape: (5,)  bounds(5, 4)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_multidimensional(self):
        # Demonstrate formatting of multdimensional arrays
        result = self.coord_representations(shape=(7, 5, 3))
        # This one is a bit unavoidably long ..
        expected = [
            "<AuxCoord: x / (m)  [...]  shape(7, 5, 3)>",
            "AuxCoord :  x / (m)",
            "    points: [",
            "        [[  0.,   1.,   2.],",
            "         [  3.,   4.,   5.],",
            "         ...,",
            "         [  9.,  10.,  11.],",
            "         [ 12.,  13.,  14.]],",
            "       ",
            "        [[ 15.,  16.,  17.],",
            "         [ 18.,  19.,  20.],",
            "         ...,",
            "         [ 24.,  25.,  26.],",
            "         [ 27.,  28.,  29.]],",
            "       ",
            "        ...,",
            "       ",
            "        [[ 75.,  76.,  77.],",
            "         [ 78.,  79.,  80.],",
            "         ...,",
            "         [ 84.,  85.,  86.],",
            "         [ 87.,  88.,  89.]],",
            "       ",
            "        [[ 90.,  91.,  92.],",
            "         [ 93.,  94.,  95.],",
            "         ...,",
            "         [ 99., 100., 101.],",
            "         [102., 103., 104.]]]",
            "    shape: (7, 5, 3)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_multidimensional_small(self):
        # Demonstrate that a small-enough multidim will print in the repr.
        result = self.coord_representations(shape=(2, 2), datatype=int)
        expected = [
            "<AuxCoord: x / (m)  [[0, 1], [2, 3]]  shape(2, 2)>",
            "AuxCoord :  x / (m)",
            "    points: [",
            "        [0, 1],",
            "        [2, 3]]",
            "    shape: (2, 2)",
            "    dtype: int64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_integers_short(self):
        result = self.coord_representations(datatype=np.int16)
        expected = [
            "<AuxCoord: x / (m)  [0, 1, 2, 3, 4]  shape(5,)>",
            "AuxCoord :  x / (m)",
            "    points: [0, 1, 2, 3, 4]",
            "    shape: (5,)",
            "    dtype: int16",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_integers_masked(self):
        result = self.coord_representations(datatype=int, masked=True)
        expected = [
            "<AuxCoord: x / (m)  [0 , --, 2 , 3 , --]  shape(5,)>",
            "AuxCoord :  x / (m)",
            "    points: [0 , --, 2 , 3 , --]",
            "    shape: (5,)",
            "    dtype: int64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_integers_masked_long(self):
        result = self.coord_representations(shape=(20,), datatype=int, masked=True)
        expected = [
            "<AuxCoord: x / (m)  [0 , --, ..., 18, --]  shape(20,)>",
            "AuxCoord :  x / (m)",
            "    points: [0 , --, ..., 18, --]",
            "    shape: (20,)",
            "    dtype: int64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)


class Test__print_Coord(Mixin__string_representations, tests.IrisTest):
    """Test Coord-specific aspects of __str__ and __repr__ output.

    Aspects :
    * DimCoord / AuxCoord
    * coord_system
    * climatological
    * circular

    """

    def test_dimcoord(self):
        result = self.coord_representations(dimcoord=True)
        expected = [
            "<DimCoord: x / (m)  [0., 1., 2., 3., 4.]  shape(5,)>",
            "DimCoord :  x / (m)",
            "    points: [0., 1., 2., 3., 4.]",
            "    shape: (5,)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_coord_system(self):
        result = self.coord_representations(coord_system=GeogCS(1000.0))
        expected = [
            "<AuxCoord: x / (m)  [0., 1., 2., 3., 4.]  shape(5,)>",
            "AuxCoord :  x / (m)",
            "    points: [0., 1., 2., 3., 4.]",
            "    shape: (5,)",
            "    dtype: float64",
            "    long_name: 'x'",
            "    coord_system: GeogCS(1000.0)",
        ]
        self.assertLines(expected, result)

    def test_climatological(self):
        cube = cube_with_climatology()
        coord = cube.coord("time")
        coord = coord[:1]  # Just to make it a bit shorter
        result = self.repr_str_strings(coord)
        expected = [
            ("<DimCoord: time / (days since 1970-01-01 00:00:00-00)  [...]+bounds>"),
            (
                "DimCoord :  time / (days since 1970-01-01 00:00:00-00, "
                "standard calendar)"
            ),
            "    points: [2001-01-10 00:00:00]",
            "    bounds: [[2001-01-10 00:00:00, 2011-01-10 00:00:00]]",
            "    shape: (1,)  bounds(1, 2)",
            "    dtype: float64",
            "    standard_name: 'time'",
            "    climatological: True",
        ]
        self.assertLines(expected, result)

    def test_circular(self):
        coord = self.sample_coord(shape=(2,), dimcoord=True)
        coord.circular = True
        result = self.repr_str_strings(coord)
        expected = [
            "<DimCoord: x / (m)  [0., 1.]  shape(2,)>",
            "DimCoord :  x / (m)",
            "    points: [0., 1.]",
            "    shape: (2,)",
            "    dtype: float64",
            "    long_name: 'x'",
            "    circular: True",
        ]
        self.assertLines(expected, result)


class Test__print_noncoord(Mixin__string_representations, tests.IrisTest):
    """Limited testing of other _DimensionalMetadata subclasses.

    * AncillaryVariable
    * CellMeasure
    * Connectivity
    * MeshCoord

    """

    def test_ancillary(self):
        # Check we can print an AncillaryVariable
        # Practically, ~identical to an AuxCoord, but without bounds, and the
        # array is called 'data'.
        data = self.sample_data()
        ancil = AncillaryVariable(data, long_name="v_aux", units="m s-1")
        result = self.repr_str_strings(ancil)
        expected = [
            "<AncillaryVariable: v_aux / (m s-1)  [0., ...]  shape(5,)>",
            "AncillaryVariable :  v_aux / (m s-1)",
            "    data: [0., 1., 2., 3., 4.]",
            "    shape: (5,)",
            "    dtype: float64",
            "    long_name: 'v_aux'",
        ]
        self.assertLines(expected, result)

    def test_cellmeasure(self):
        # Check we can print an AncillaryVariable
        # N.B. practically, identical to an AuxCoord (without bounds)
        # Check we can print an AncillaryVariable
        # Practically, ~identical to an AuxCoord, but without bounds, and the
        # array is called 'data'.
        data = self.sample_data()
        cell_measure = CellMeasure(
            data, measure="area", long_name="cell_area", units="m^2"
        )
        result = self.repr_str_strings(cell_measure)
        expected = [
            "<CellMeasure: cell_area / (m^2)  [0., 1., 2., 3., 4.]  shape(5,)>",
            "CellMeasure :  cell_area / (m^2)",
            "    data: [0., 1., 2., 3., 4.]",
            "    shape: (5,)",
            "    dtype: float64",
            "    long_name: 'cell_area'",
            "    measure: 'area'",
        ]
        self.assertLines(expected, result)

    def test_connectivity(self):
        # Check we can print a Connectivity
        # Like a Coord, but always print : cf_role, location_axis, start_index
        data = self.sample_data(shape=(3, 2), datatype=int)
        conn = Connectivity(
            data, cf_role="edge_node_connectivity", long_name="enc", units="1"
        )
        result = self.repr_str_strings(conn)
        expected = [
            "<Connectivity: enc / (1)  [[0, 1], [2, 3], [4, 5]]  shape(3, 2)>",
            "Connectivity :  enc / (1)",
            "    data: [",
            "        [0, 1],",
            "        [2, 3],",
            "        [4, 5]]",
            "    shape: (3, 2)",
            "    dtype: int64",
            "    long_name: 'enc'",
            "    cf_role: 'edge_node_connectivity'",
            "    start_index: 0",
            "    location_axis: 0",
        ]
        self.assertLines(expected, result)

    def test_connectivity__start_index(self):
        # Check we can print a Connectivity
        # Like a Coord, but always print : cf_role, location_axis, start_index
        data = self.sample_data(shape=(3, 2), datatype=int)
        conn = Connectivity(
            data + 1,
            start_index=1,
            cf_role="edge_node_connectivity",
            long_name="enc",
            units="1",
        )
        result = self.repr_str_strings(conn)
        expected = [
            "<Connectivity: enc / (1)  [[1, 2], [3, 4], [5, 6]]  shape(3, 2)>",
            "Connectivity :  enc / (1)",
            "    data: [",
            "        [1, 2],",
            "        [3, 4],",
            "        [5, 6]]",
            "    shape: (3, 2)",
            "    dtype: int64",
            "    long_name: 'enc'",
            "    cf_role: 'edge_node_connectivity'",
            "    start_index: 1",
            "    location_axis: 0",
        ]
        self.assertLines(expected, result)

    def test_connectivity__location_axis(self):
        # Check we can print a Connectivity
        # Like a Coord, but always print : cf_role, location_axis, start_index
        data = self.sample_data(shape=(3, 2), datatype=int)
        conn = Connectivity(
            data.transpose(),
            location_axis=1,
            cf_role="edge_node_connectivity",
            long_name="enc",
            units="1",
        )
        result = self.repr_str_strings(conn)
        expected = [
            "<Connectivity: enc / (1)  [[0, 2, 4], [1, 3, 5]]  shape(2, 3)>",
            "Connectivity :  enc / (1)",
            "    data: [",
            "        [0, 2, 4],",
            "        [1, 3, 5]]",
            "    shape: (2, 3)",
            "    dtype: int64",
            "    long_name: 'enc'",
            "    cf_role: 'edge_node_connectivity'",
            "    start_index: 0",
            "    location_axis: 1",
        ]
        self.assertLines(expected, result)

    def test_meshcoord(self):
        meshco = sample_meshcoord()
        meshco.mesh.long_name = "test_mesh"  # For stable printout of the Mesh
        result = self.repr_str_strings(meshco)
        expected = [
            (
                "<MeshCoord: longitude / (unknown)  "
                "mesh(test_mesh) location(face)  "
                "[...]+bounds  shape(3,)>"
            ),
            "MeshCoord :  longitude / (unknown)",
            "    mesh: <MeshXY: 'test_mesh'>",
            "    location: 'face'",
            "    points: [3100, 3101, 3102]",
            "    bounds: [",
            "        [1100, 1101, 1102, 1103],",
            "        [1104, 1105, 1106, 1107],",
            "        [1108, 1109, 1110, 1111]]",
            "    shape: (3,)  bounds(3, 4)",
            "    dtype: int64",
            "    standard_name: 'longitude'",
            "    axis: 'x'",
        ]
        self.assertLines(expected, result)


class Test_summary(Mixin__string_representations, tests.IrisTest):
    """Test the controls of the 'summary' method."""

    def test_shorten(self):
        coord = self.sample_coord()
        expected = self.repr_str_strings(coord)
        result = coord.summary(shorten=True) + "\n" + coord.summary()
        self.assertEqual(expected, result)

    def test_max_values__default(self):
        coord = self.sample_coord()
        result = coord.summary()
        expected = [
            "AuxCoord :  x / (m)",
            "    points: [0., 1., 2., 3., 4.]",
            "    shape: (5,)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_max_values__2(self):
        coord = self.sample_coord()
        result = coord.summary(max_values=2)
        expected = [
            "AuxCoord :  x / (m)",
            "    points: [0., 1., ..., 3., 4.]",
            "    shape: (5,)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_max_values__bounded__2(self):
        coord = self.sample_coord(bounded=True)
        result = coord.summary(max_values=2)
        expected = [
            "AuxCoord :  x / (m)",
            "    points: [0., 1., ..., 3., 4.]",
            "    bounds: [",
            "        [-0.5,  0.5],",
            "        [ 0.5,  1.5],",
            "        ...,",
            "        [ 2.5,  3.5],",
            "        [ 3.5,  4.5]]",
            "    shape: (5,)  bounds(5, 2)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_max_values__0(self):
        coord = self.sample_coord(bounded=True)
        result = coord.summary(max_values=0)
        expected = [
            "AuxCoord :  x / (m)",
            "    points: [...]",
            "    bounds: [...]",
            "    shape: (5,)  bounds(5, 2)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_linewidth__default(self):
        coord = self.sample_coord()
        coord.points = coord.points + 1000.003  # Make the output numbers wider
        result = coord.summary()
        expected = [
            "AuxCoord :  x / (m)",
            "    points: [1000.003, 1001.003, 1002.003, 1003.003, 1004.003]",
            "    shape: (5,)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

        # Show that, when unset, it follows the numpy setting
        with np.printoptions(linewidth=35):
            result = coord.summary()
        expected = [
            "AuxCoord :  x / (m)",
            "    points: [",
            "        1000.003, 1001.003,",
            "        1002.003, 1003.003,",
            "        1004.003]",
            "    shape: (5,)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

    def test_linewidth__set(self):
        coord = self.sample_coord()
        coord.points = coord.points + 1000.003  # Make the output numbers wider
        expected = [
            "AuxCoord :  x / (m)",
            "    points: [",
            "        1000.003, 1001.003,",
            "        1002.003, 1003.003,",
            "        1004.003]",
            "    shape: (5,)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        result = coord.summary(linewidth=35)
        self.assertLines(expected, result)

        with np.printoptions(linewidth=999):
            # Show that, when set, it ignores the numpy setting
            result = coord.summary(linewidth=35)
        self.assertLines(expected, result)

    def test_convert_dates(self):
        coord = self.sample_coord(dates=True)
        result = coord.summary()
        expected = [
            "AuxCoord :  x / (days since 1970-03-5, standard calendar)",
            "    points: [",
            (
                "        1970-03-05 00:00:00, 1970-03-06 00:00:00, "
                "1970-03-07 00:00:00,"
            ),
            "        1970-03-08 00:00:00, 1970-03-09 00:00:00]",
            "    shape: (5,)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)

        result = coord.summary(convert_dates=False)
        expected = [
            "AuxCoord :  x / (days since 1970-03-5, standard calendar)",
            "    points: [0., 1., 2., 3., 4.]",
            "    shape: (5,)",
            "    dtype: float64",
            "    long_name: 'x'",
        ]
        self.assertLines(expected, result)


if __name__ == "__main__":
    tests.main()
