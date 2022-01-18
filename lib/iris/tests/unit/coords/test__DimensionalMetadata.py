# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.coords._DimensionalMetadata` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip


from cf_units import Unit
import numpy as np

from iris.coords import AuxCoord, DimCoord, _DimensionalMetadata


class Test___init____abstractmethod(tests.IrisTest):
    def test(self):
        emsg = (
            "Can't instantiate abstract class _DimensionalMetadata with "
            "abstract methods __init__"
        )
        with self.assertRaisesRegex(TypeError, emsg):
            _ = _DimensionalMetadata(0)


class Mixin__string_representations:
    """
    Common testcode for generic `__str__`, `__repr__` and `summary` methods.

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

    """

    def repr_str_strings(self, dm):
        """Return a simple combination of repr and str printouts."""
        return repr(dm) + "\n" + str(dm)

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
                values = digit_strs[:length]
            else:
                # [... '9999999999', '0', '11' ....]
                indices = [(i % 10) for i in range(length)]
                values = np.array(digit_strs)[indices]
        else:
            # numeric content : a simple [0, 1, 2 ...]
            values = np.arange(length).astype(dtype)

        if masked:
            # Mask 1 in 3 points : [x -- x x -- x ...]
            masked_points = [(i % 3) == 1 for i in range(length)]
            values = np.ma.masked_array(values, mask=masked_points)

        values = values.reshape(shape)
        return values

    # Make a sample Coord, as _DimensionalMetadata is abstract and this is the
    # obvious concrete subclass to use for testing
    def sample_coord(
        self,
        datatype=float,
        units="m",
        long_name="x",
        shape=(5,),
        masked=False,
        bounded=False,
        dimcoord=False,
        *coord_args,
        **coord_kwargs,
    ):
        if masked:
            dimcoord = False
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
        if bounded:
            coord.guess_bounds()
        return coord

    def assertLines(self, list_of_expected_lines, string_result):
        """
        Assert equality between a result and expected output lines.

        For convenience, the 'expected lines' are joined with a '\\n',
        because a list of strings is nicer to construct in code.
        They should then match the actual result, which is a simple string.

        """
        self.assertEqual("\n".join(list_of_expected_lines), string_result)


class Mixin__cfvariable_common(Mixin__string_representations, tests.IrisTest):
    """
    Testcode common to all _DimensionalMetadata instances,
    that is the CFVariableMixin inheritance, plus values array (data-manager).

    Aspects :
    * standard_name:
    * long_name:
    * var_name:
    * attributes
    * units
    * shape
    * dtype

    """

    def coord_representations(self, *args, **kwargs):
        """
        Create a test coord and return its string representations.

        Pass args+kwargs to 'sample_coord' and return the 'repr_str_strings'.

        """
        coord = self.sample_coord(*args, **kwargs)
        return self.repr_str_strings(coord)


class Test__cfvariable_common(Mixin__cfvariable_common, tests.IrisTest):
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
        result = self.coord_representations(
            long_name=None, units=None, shape=(1,)
        )
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
            "<AuxCoord: x / (m)  [0., 1., 2.]  shape(3,)>",
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
        result = self.coord_representations(
            units=Unit("days since 1892-05-17 03:00:25", calendar="360_day"),
        )
        expected = [
            (
                "<AuxCoord: x / (days since 1892-05-17 03:00:25)  "
                "[1892-05-17 03:00:25, ...]  shape(5,)>"
            ),
            (
                "AuxCoord :  x / (days since 1892-05-17 03:00:25, "
                "360_day calendar)"
            ),
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
        coord = self.sample_coord(
            attributes={
                "array": np.arange(7.0),
                "list": [1, 2, 3],
                "empty": [],
                "None": None,
                "string": "this",
                "long_long_long_long_long_long_name": 3,
                "other": "long_long_long_long_long_long_value",
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
            "    attributes: {'array': array([0., 1., 2., 3., 4., 5., 6.]), 'list': [1, 2, 3], 'empty': [], "
            "'None': None, 'string': 'this', 'long_long_long_long_long_long_name': 3, "
            "'other': 'long_long_long_long_long_long_value', 'float': 4.3}",
        ]
        self.assertLines(expected, result)


if __name__ == "__main__":
    tests.main()
