# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Definitions of coordinates and other dimensional metadata."""

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from collections.abc import Container
import copy
from functools import lru_cache
from itertools import zip_longest
import operator
import warnings
import zlib

import dask.array as da
import numpy as np
import numpy.ma as ma

from iris._data_manager import DataManager
import iris._lazy_data as _lazy
from iris.common import (
    AncillaryVariableMetadata,
    BaseMetadata,
    CellMeasureMetadata,
    CFVariableMixin,
    CoordMetadata,
    DimCoordMetadata,
    metadata_manager_factory,
)
import iris.exceptions
import iris.time
import iris.util
import iris.warnings

#: The default value for ignore_axis which controls guess_coord_axis' behaviour
DEFAULT_IGNORE_AXIS = False


class _DimensionalMetadata(CFVariableMixin, metaclass=ABCMeta):
    """Superclass for dimensional metadata."""

    _MODE_ADD = 1
    _MODE_SUB = 2
    _MODE_MUL = 3
    _MODE_DIV = 4
    _MODE_RDIV = 5
    _MODE_SYMBOL = {
        _MODE_ADD: "+",
        _MODE_SUB: "-",
        _MODE_MUL: "*",
        _MODE_DIV: "/",
        _MODE_RDIV: "/",
    }

    # Used by printout methods : __str__ and __repr__
    # Overridden in subclasses : Coord->'points', Connectivity->'indices'
    _values_array_name = "data"

    @abstractmethod
    def __init__(
        self,
        values,
        standard_name=None,
        long_name=None,
        var_name=None,
        units=None,
        attributes=None,
    ):
        """Construct a single dimensional metadata object.

        Parameters
        ----------
        values :
            The values of the dimensional metadata.
        standard_name : optional
            CF standard name of the dimensional metadata.
        long_name : optional
            Descriptive name of the dimensional metadata.
        var_name : optional
            The netCDF variable name for the dimensional metadata.
        units : optional
            The :class:`~cf_units.Unit` of the dimensional metadata's values.
            Can be a string, which will be converted to a Unit object.
        attributes : optional
            A dictionary containing other cf and user-defined attributes.

        """
        # Note: this class includes bounds handling code for convenience, but
        # this can only run within instances which are also Coords, because
        # only they may actually have bounds.  This parent class has no
        # bounds-related getter/setter properties, and no bounds keywords in
        # its __init__ or __copy__ methods.  The only bounds-related behaviour
        # it provides is a 'has_bounds()' method, which always returns False.

        # Configure the metadata manager.
        if not hasattr(self, "_metadata_manager"):
            self._metadata_manager = metadata_manager_factory(BaseMetadata)

        #: CF standard name of the quantity that the metadata represents.
        self.standard_name = standard_name

        #: Descriptive name of the metadata.
        self.long_name = long_name

        #: The netCDF variable name for the metadata.
        self.var_name = var_name

        #: Unit of the quantity that the metadata represents.
        self.units = units

        #: Other attributes, including user specified attributes that
        #: have no meaning to Iris.
        self.attributes = attributes

        # Set up DataManager attributes and values.
        self._values_dm = None
        self._values = values
        self._bounds_dm = None  # Only ever set on Coord-derived instances.

    def __getitem__(self, keys):
        """Return a new dimensional metadata whose values are obtained by conventional array indexing.

        .. note::

            Indexing of a circular coordinate results in a non-circular
            coordinate if the overall shape of the coordinate changes after
            indexing.

        """
        # Note: this method includes bounds handling code, but it only runs
        # within Coord type instances, as only these allow bounds to be set.

        # Fetch the values.
        values = self._values_dm.core_data()

        # Index values with the keys.
        _, values = iris.util._slice_data_with_keys(values, keys)

        # Copy values after indexing to avoid making metadata that is a
        # view on another metadata. This will not realise lazy data.
        values = values.copy()

        # If the metadata is a coordinate and it has bounds, repeat the above
        # with the bounds.
        copy_args = {}
        if self.has_bounds():
            bounds = self._bounds_dm.core_data()
            _, bounds = iris.util._slice_data_with_keys(bounds, keys)
            # Pass into the copy method : for Coords, it has a 'bounds' key.
            copy_args["bounds"] = bounds.copy()

        # The new metadata is a copy of the old one with replaced content.
        new_metadata = self.copy(values, **copy_args)

        return new_metadata

    def copy(self, values=None):
        """Return a copy of this dimensional metadata object.

        Parameters
        ----------
        values : optional
            An array of values for the new dimensional metadata object.
            This may be a different shape to the original values array being
            copied.

        """
        # Note: this is overridden in Coord subclasses, to add bounds handling
        # and a 'bounds' keyword.
        new_metadata = copy.deepcopy(self)
        if values is not None:
            new_metadata._values_dm = None
            new_metadata._values = values

        return new_metadata

    @abstractmethod
    def cube_dims(self, cube):
        """Identify the cube dims of any _DimensionalMetadata object.

        Return the dimensions in the cube of a matching _DimensionalMetadata
        object, if any.

        Equivalent to cube.coord_dims(self) for a Coord,
        or cube.cell_measure_dims for a CellMeasure, and so on.
        Simplifies generic code to handle any _DimensionalMetadata objects.

        """
        # Only makes sense for specific subclasses.
        raise NotImplementedError()

    def _sanitise_array(self, src, ndmin):
        if _lazy.is_lazy_data(src):
            # Lazy data : just ensure ndmin requirement.
            ndims_missing = ndmin - src.ndim
            if ndims_missing <= 0:
                result = src
            else:
                extended_shape = tuple([1] * ndims_missing + list(src.shape))
                result = src.reshape(extended_shape)
        else:
            # Real data : a few more things to do in this case.
            # Ensure the array is writeable.
            # NB. Returns the *same object* if src is already writeable.
            result = np.require(src, requirements="W")
            # Ensure the array has enough dimensions.
            # NB. Returns the *same object* if result.ndim >= ndmin
            func = ma.array if ma.isMaskedArray(result) else np.array
            result = func(result, ndmin=ndmin, copy=False)
            # We don't need to copy the data, but we do need to have our
            # own view so we can control the shape, etc.
            result = result.view()
        return result

    @property
    def _values(self):
        """The _DimensionalMetadata values as a NumPy array."""
        return self._values_dm.data.view()

    @_values.setter
    def _values(self, values):
        # Set the values to a new array - as long as it's the same shape.

        # Ensure values has an ndmin of 1 and is either a numpy or lazy array.
        # This will avoid Scalar _DimensionalMetadata with values of shape ()
        # rather than the desired (1,).
        values = self._sanitise_array(values, 1)

        # Set or update DataManager.
        if self._values_dm is None:
            self._values_dm = DataManager(values)
        else:
            self._values_dm.data = values

    def _lazy_values(self):
        """Return a lazy array representing the dimensional metadata values."""
        return self._values_dm.lazy_data()

    def _core_values(self):
        """Value array of this dimensional metadata which may be a NumPy array or a dask array."""
        result = self._values_dm.core_data()
        if not _lazy.is_lazy_data(result):
            result = result.view()

        return result

    def _has_lazy_values(self):
        """Indicate whether the metadata's values array is a lazy dask array or not."""
        return self._values_dm.has_lazy_data()

    def summary(
        self,
        shorten=False,
        max_values=None,
        edgeitems=2,
        linewidth=None,
        precision=None,
        convert_dates=True,
        _section_indices=None,
    ):
        r"""Make a printable text summary.

        Parameters
        ----------
        shorten : bool, default=False
            If True, produce an abbreviated one-line summary.
            If False, produce a multi-line summary, with embedded newlines.
        max_values : int or None
            If more than this many data values, print truncated data arrays
            instead of full contents.
            If 0, print only the shape.
            The default is 5 if :attr:`shorten`\ =True, or 15 otherwise.
            This overrides ``numpy.get_printoptions['threshold']``\ .
        linewidth : int or None
            Character-width controlling line splitting of array outputs.
            If unset, defaults to ``numpy.get_printoptions['linewidth']``\ .
        edgeitems : int, default=2
            Controls truncated array output.
            Overrides ``numpy.getprintoptions['edgeitems']``\ .
        precision : int or None
            Controls number decimal formatting.
            When :attr:`shorten`\ =True this is defaults to 3, in which case it
            overrides ``numpy.get_printoptions()['precision']``\ .
        convert_dates : bool, default=True
            If the units has a calendar, then print array values as date
            strings instead of the actual numbers.

        Returns
        -------
        str
            Output text, with embedded newlines when :attr:`shorten`\ =False.

        Notes
        -----
        .. note::
            Arrays are formatted using :meth:`numpy.array2string`. Some aspects
            of the array formatting are controllable in the usual way, via
            :meth:`numpy.printoptions`, but others are overridden as detailed
            above.
            Control of those aspects is still available, but only via the call
            arguments.

        """
        # NOTE: the *private* key "_section_indices" can be set to a dict, to
        # return details of which (line, character) each particular section of
        # the output text begins at.
        # Currently only used by MeshCoord.summary(), which needs this info to
        # modify the result string, for idiosyncratic reasons.

        def array_summary(data, n_max, n_edge, linewidth, precision):
            # Return a text summary of an array.
            # Take account of strings, dates and masked data.
            result = ""
            formatter = None
            if convert_dates and self.units.is_time_reference():
                # Account for dates, if enabled.
                # N.B. a time unit with a long time interval ("months"
                # or "years") cannot be converted to a date using
                # `num2date`, so gracefully fall back to printing
                # values as numbers.
                if not self.units.is_long_time_interval():
                    # Otherwise ... replace all with strings.
                    if ma.is_masked(data):
                        mask = data.mask
                    else:
                        mask = None
                    data = np.array(self.units.num2date(data))
                    data = data.astype(str)
                    # Masked datapoints do not survive num2date.
                    if mask is not None:
                        data = np.ma.masked_array(data, mask)

            if ma.is_masked(data):
                # Masks are not handled by np.array2string, whereas
                # MaskedArray.__str__ is using a private method to convert to
                # objects.
                # Our preferred solution is to convert to strings *and* fill
                # with '--'.   This is not ideal because numbers will not align
                # with a common numeric format, but there is no *public* logic
                # in numpy to arrange that, so let's not overcomplicate.
                # It happens that array2string *also* does not use a common
                # format (width) for strings, but we fix that below...
                data = data.astype(str).filled("--")

            if data.dtype.kind == "U":
                # Strings : N.B. includes all missing data
                # find the longest.
                length = max(len(str(x)) for x in data.flatten())
                # Pre-apply a common formatting width.
                formatter = {"all": lambda x: str(x).ljust(length)}

            result = np.array2string(
                data,
                separator=", ",
                edgeitems=n_edge,
                threshold=n_max,
                max_line_width=linewidth,
                formatter=formatter,
                precision=precision,
            )

            return result

        units_str = str(self.units)
        if self.units.calendar and not shorten:
            units_str += f", {self.units.calendar} calendar"
        title_str = f"{self.name()} / ({units_str})"
        cls_str = type(self).__name__
        shape_str = str(self.shape)

        # Implement conditional defaults for control args.
        if max_values is None:
            max_values = 5 if shorten else 15
        precision = 3 if shorten else None
        n_indent = 4
        indent = " " * n_indent
        newline_indent = "\n" + indent
        if linewidth is not None:
            given_array_width = linewidth
        else:
            given_array_width = np.get_printoptions()["linewidth"]
        using_array_width = given_array_width - n_indent * 2
        # Make a printout of the main data array (or maybe not, if lazy).
        if self._has_lazy_values():
            data_str = "<lazy>"
        elif max_values == 0:
            data_str = "[...]"
        else:
            data_str = array_summary(
                self._values,
                n_max=max_values,
                n_edge=edgeitems,
                linewidth=using_array_width,
                precision=precision,
            )

        # The output under construction, divided into lines for convenience.
        output_lines = [""]

        def add_output(text, section=None):
            # Append output text and record locations of named 'sections'
            if section and _section_indices is not None:
                # defined a named 'section', recording the current line number
                # and character position as its start position
                i_line = len(output_lines) - 1
                i_char = len(output_lines[-1])
                _section_indices[section] = (i_line, i_char)
            # Split the text-to-add into lines
            lines = text.split("\n")
            # Add initial text (before first '\n') to the current line
            output_lines[-1] += lines[0]
            # Add subsequent lines as additional output lines
            for line in lines[1:]:
                output_lines.append(line)  # Add new lines

        if shorten:
            add_output(f"<{cls_str}: ")
            add_output(f"{title_str}  ", section="title")

            if data_str != "<lazy>":
                # Flatten to a single line, reducing repeated spaces.
                def flatten_array_str(array_str):
                    array_str = array_str.replace("\n", " ")
                    array_str = array_str.replace("\t", " ")
                    while "  " in array_str:
                        array_str = array_str.replace("  ", " ")
                    return array_str

                data_str = flatten_array_str(data_str)
                # Adjust maximum-width to allow for the title width in the
                # repr form.
                current_line_len = len(output_lines[-1])
                using_array_width = given_array_width - current_line_len
                # Work out whether to include a summary of the data values
                if len(data_str) > using_array_width:
                    # Make one more attempt, printing just the *first* point,
                    # as this is useful for dates.
                    data_str = data_str = array_summary(
                        self._values[:1],
                        n_max=max_values,
                        n_edge=edgeitems,
                        linewidth=using_array_width,
                        precision=precision,
                    )
                    data_str = flatten_array_str(data_str)
                    data_str = data_str[:-1] + ", ...]"
                    if len(data_str) > using_array_width:
                        # Data summary is still too long : replace with array
                        # "placeholder" representation.
                        data_str = "[...]"

            if self.has_bounds():
                data_str += "+bounds"

            if self.shape != (1,):
                # Anything non-scalar : show shape as well.
                data_str += f"  shape{shape_str}"

            # single-line output in 'shorten' mode
            add_output(f"{data_str}>", section="data")

        else:
            # Long (multi-line) output format.
            add_output(f"{cls_str} :  ")
            add_output(f"{title_str}", section="title")

            def reindent_data_string(text, n_indent):
                lines = [line for line in text.split("\n")]
                indent = " " * (n_indent - 1)  # allow 1 for the initial '['
                # Indent all but the *first* line.
                line_1, rest_lines = lines[0], lines[1:]
                rest_lines = ["\n" + indent + line for line in rest_lines]
                result = line_1 + "".join(rest_lines)
                return result

            data_array_str = reindent_data_string(data_str, 2 * n_indent)

            # NOTE: actual section name is variable here : data/points/indices
            data_text = f"{self._values_array_name}: "
            if "\n" in data_array_str:
                # Put initial '[' here, and the rest on subsequent lines
                data_text += "[" + newline_indent + indent + data_array_str[1:]
            else:
                # All on one line
                data_text += data_array_str

            # N.B. indent section and record section start after that
            add_output(newline_indent)
            add_output(data_text, section="data")

            if self.has_bounds():
                # Add a bounds section : basically just like the 'data'.
                if self._bounds_dm.has_lazy_data():
                    bounds_array_str = "<lazy>"
                elif max_values == 0:
                    bounds_array_str = "[...]"
                else:
                    bounds_array_str = array_summary(
                        self._bounds_dm.data,
                        n_max=max_values,
                        n_edge=edgeitems,
                        linewidth=using_array_width,
                        precision=precision,
                    )
                    bounds_array_str = reindent_data_string(
                        bounds_array_str, 2 * n_indent
                    )

                bounds_text = "bounds: "
                if "\n" in bounds_array_str:
                    # Put initial '[' here, and the rest on subsequent lines
                    bounds_text += "[" + newline_indent + indent + bounds_array_str[1:]
                else:
                    # All on one line
                    bounds_text += bounds_array_str

                # N.B. indent section and record section start after that
                add_output(newline_indent)
                add_output(bounds_text, section="bounds")

            if self.has_bounds():
                shape_str += f"  bounds{self._bounds_dm.shape}"

            # Add shape section (always)
            add_output(newline_indent)
            add_output(f"shape: {shape_str}", section="shape")

            # Add dtype section (always)
            add_output(newline_indent)
            add_output(f"dtype: {self.dtype}", section="dtype")

            for name in self._metadata_manager._fields:
                if name == "units":
                    # This was already included in the header line
                    continue
                val = getattr(self, name, None)
                if isinstance(val, Container):
                    # Don't print empty containers, like attributes={}
                    show = bool(val)
                else:
                    # Don't print properties when not present, or set to None,
                    # or False.
                    # This works OK as long as we are happy to treat all
                    # boolean properties as 'off' when False :  Which happens to
                    # work for all those defined so far.
                    show = val is not None and val is not False
                if show:
                    if name == "attributes":
                        # Use a multi-line form for this.
                        add_output(newline_indent)
                        add_output("attributes:", section="attributes")
                        max_attname_len = max(len(attr) for attr in val.keys())
                        for attrname, attrval in val.items():
                            attrname = attrname.ljust(max_attname_len)
                            if isinstance(attrval, str):
                                # quote strings
                                attrval = repr(attrval)
                                # and abbreviate really long ones
                                attrval = iris.util.clip_string(attrval)
                            attr_string = f"{attrname}  {attrval}"
                            add_output(newline_indent + indent + attr_string)
                    else:
                        # add a one-line section for this property
                        # (aka metadata field)
                        add_output(newline_indent)
                        add_output(f"{name}: {val!r}", section=name)

        return "\n".join(output_lines)

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return self.summary(shorten=True)

    def __eq__(self, other):
        if other is self:
            return True

        # Note: this method includes bounds handling code, but it only runs
        #  within Coord type instances, as only these allow bounds to be set.

        eq = NotImplemented
        # If the other object has a means of getting its definition, then do
        #  the comparison, otherwise return a NotImplemented to let Python try
        #  to resolve the operator elsewhere.
        if hasattr(other, "metadata"):
            # metadata comparison
            eq = self.metadata == other.metadata
            # data values comparison
            if eq and eq is not NotImplemented:
                eq = iris.util.array_equal(
                    self._core_values(), other._core_values(), withnans=True
                )

            # Also consider bounds, if we have them.
            # (N.B. though only Coords can ever actually *have* bounds).
            if eq and eq is not NotImplemented:
                if self.has_bounds() and other.has_bounds():
                    eq = iris.util.array_equal(
                        self.core_bounds(), other.core_bounds(), withnans=True
                    )
                else:
                    eq = not self.has_bounds() and not other.has_bounds()

        return eq

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result

    # Must supply __hash__ as Python 3 does not enable it if __eq__ is defined.
    # NOTE: Violates "objects which compare equal must have the same hash".
    # We ought to remove this, as equality of two dimensional metadata can
    # *change*, so they really should not be hashable.
    # However, current code needs it, e.g. so we can put them in sets.
    # Fixing it will require changing those uses.  See #962 and #1772.
    def __hash__(self):
        return hash(id(self))

    def __binary_operator__(self, other, mode_constant):
        """Perform common code which is called by add, sub, mul and div.

        Mode constant is one of ADD, SUB, MUL, DIV, RDIV

        .. note::

            The unit is *not* changed when doing scalar operations on a
            metadata object. This means that a metadata object which represents
            "10 meters" when multiplied by a scalar i.e. "1000" would result in
            a metadata object of "10000 meters". An alternative approach could
            be taken to multiply the *unit* by 1000 and the resultant metadata
            object would represent "10 kilometers".

        """
        # Note: this method includes bounds handling code, but it only runs
        # within Coord type instances, as only these allow bounds to be set.

        if isinstance(other, _DimensionalMetadata):
            emsg = (
                f"{self.__class__.__name__} "
                f"{self._MODE_SYMBOL[mode_constant]} "
                f"{other.__class__.__name__}"
            )
            raise iris.exceptions.NotYetImplementedError(emsg)

        if isinstance(other, (int, float, np.number)):

            def op(values):
                if mode_constant == self._MODE_ADD:
                    new_values = values + other
                elif mode_constant == self._MODE_SUB:
                    new_values = values - other
                elif mode_constant == self._MODE_MUL:
                    new_values = values * other
                elif mode_constant == self._MODE_DIV:
                    new_values = values / other
                elif mode_constant == self._MODE_RDIV:
                    new_values = other / values
                return new_values

            new_values = op(self._values_dm.core_data())
            result = self.copy(new_values)

            if self.has_bounds():
                result.bounds = op(self._bounds_dm.core_data())
        else:
            # must return NotImplemented to ensure invocation of any
            # associated reflected operator on the "other" operand
            # see https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
            result = NotImplemented

        return result

    def __add__(self, other):
        return self.__binary_operator__(other, self._MODE_ADD)

    def __sub__(self, other):
        return self.__binary_operator__(other, self._MODE_SUB)

    def __mul__(self, other):
        return self.__binary_operator__(other, self._MODE_MUL)

    def __div__(self, other):
        return self.__binary_operator__(other, self._MODE_DIV)

    def __truediv__(self, other):
        return self.__binary_operator__(other, self._MODE_DIV)

    __radd__ = __add__

    def __rsub__(self, other):
        return (-self) + other

    def __rdiv__(self, other):
        return self.__binary_operator__(other, self._MODE_RDIV)

    def __rtruediv__(self, other):
        return self.__binary_operator__(other, self._MODE_RDIV)

    __rmul__ = __mul__

    def __neg__(self):
        values = -self._core_values()
        copy_args = {}
        if self.has_bounds():
            copy_args["bounds"] = -self.core_bounds()
        return self.copy(values, **copy_args)

    def convert_units(self, unit):
        """Change the units, converting the values of the metadata."""
        # If the coord has units convert the values in points (and bounds if
        # present).
        # Note: this method includes bounds handling code, but it only runs
        # within Coord type instances, as only these allow bounds to be set.
        if self.units.is_unknown():
            raise iris.exceptions.UnitConversionError(
                "Cannot convert from unknown units. "
                'The "units" attribute may be set directly.'
            )

        # Set up a delayed conversion for use if either values or bounds (if
        # present) are lazy.
        # Make fixed copies of old + new units for a delayed conversion.
        old_unit = self.units
        new_unit = unit

        # Define a delayed conversion operation (i.e. a callback).
        def pointwise_convert(values):
            return old_unit.convert(values, new_unit)

        if self._has_lazy_values():
            new_values = _lazy.lazy_elementwise(self._lazy_values(), pointwise_convert)
        else:
            new_values = self.units.convert(self._values, unit)
        self._values = new_values
        if self.has_bounds():
            if self.has_lazy_bounds():
                new_bounds = _lazy.lazy_elementwise(
                    self.lazy_bounds(), pointwise_convert
                )
            else:
                new_bounds = self.units.convert(self.bounds, unit)
            self.bounds = new_bounds
        self.units = unit

    def is_compatible(self, other, ignore=None):
        """Return whether the current dimensional metadata object is compatible with another."""
        compatible = self.name() == other.name() and self.units == other.units

        if compatible:
            common_keys = set(self.attributes).intersection(other.attributes)
            if ignore is not None:
                if isinstance(ignore, str):
                    ignore = (ignore,)
                common_keys = common_keys.difference(ignore)
            for key in common_keys:
                if np.any(self.attributes[key] != other.attributes[key]):
                    compatible = False
                    break

        return compatible

    @property
    def dtype(self):
        """The NumPy dtype of the current dimensional metadata object, as specified by its values."""
        return self._values_dm.dtype

    @property
    def ndim(self):
        """Return the number of dimensions of the current dimensional metadata object."""
        return self._values_dm.ndim

    def has_bounds(self):
        """Indicate whether the current dimensional metadata object has a bounds array."""
        # Allows for code to handle unbounded dimensional metadata agnostic of
        # whether the metadata is a coordinate or not.
        return False

    @property
    def shape(self):
        """The fundamental shape of the metadata, expressed as a tuple."""
        return self._values_dm.shape

    def xml_element(self, doc):
        """Create XML element.

        Create the :class:`xml.dom.minidom.Element` that describes this
        :class:`_DimensionalMetadata`.

        Parameters
        ----------
        doc :
            The parent :class:`xml.dom.minidom.Document`.

        Returns
        -------
        :class:`xml.dom.minidom.Element`
            :class:`xml.dom.minidom.Element` that will describe this
            :class:`_DimensionalMetadata`.

        """
        # deferred import to avoid possible circularity
        from iris.mesh import Connectivity

        # Create the XML element as the camelCaseEquivalent of the
        # class name.
        element_name = type(self).__name__
        element_name = element_name[0].lower() + element_name[1:]
        element = doc.createElement(element_name)

        element.setAttribute("id", self._xml_id())

        if self.standard_name:
            element.setAttribute("standard_name", str(self.standard_name))
        if self.long_name:
            element.setAttribute("long_name", str(self.long_name))
        if self.var_name:
            element.setAttribute("var_name", str(self.var_name))
        element.setAttribute("units", repr(self.units))
        if isinstance(self, Coord):
            if self.climatological:
                element.setAttribute("climatological", str(self.climatological))
        if self.attributes:
            attributes_element = doc.createElement("attributes")
            for name in sorted(self.attributes.keys()):
                attribute_element = doc.createElement("attribute")
                attribute_element.setAttribute("name", name)
                attribute_element.setAttribute("value", str(self.attributes[name]))
                attributes_element.appendChild(attribute_element)
            element.appendChild(attributes_element)

        if isinstance(self, Coord):
            if self.coord_system:
                element.appendChild(self.coord_system.xml_element(doc))

        # Add the values
        element.setAttribute("value_type", str(self._value_type_name()))
        element.setAttribute("shape", str(self.shape))

        # The values are referred to "points" of a coordinate and "data"
        # otherwise.
        if isinstance(self, Coord):
            values_term = "points"
        elif isinstance(self, Connectivity):
            values_term = "indices"
        else:
            values_term = "data"
        element.setAttribute(values_term, self._xml_array_repr(self._values))

        return element

    def _xml_id_extra(self, unique_value):
        return unique_value

    def _xml_id(self):
        # Returns a consistent, unique string identifier for this coordinate.
        unique_value = b""
        if self.standard_name:
            unique_value += self.standard_name.encode("utf-8")
        unique_value += b"\0"
        if self.long_name:
            unique_value += self.long_name.encode("utf-8")
        unique_value += b"\0"
        unique_value += str(self.units).encode("utf-8") + b"\0"
        for k, v in sorted(self.attributes.items()):
            unique_value += (str(k) + ":" + str(v)).encode("utf-8") + b"\0"
        # Extra modifications to unique_value that are specialised in child
        # classes
        unique_value = self._xml_id_extra(unique_value)
        # Mask to ensure consistency across Python versions & platforms.
        crc = zlib.crc32(unique_value) & 0xFFFFFFFF
        return "%08x" % (crc,)

    @staticmethod
    def _xml_array_repr(data):
        if hasattr(data, "to_xml_attr"):
            result = data._values.to_xml_attr()
        else:
            result = iris.util.format_array(data)
        return result

    def _value_type_name(self):
        """Provide a simple name for the data type of the dimensional metadata values."""
        dtype = self._core_values().dtype
        kind = dtype.kind
        if kind in "SU":
            # Establish the basic type name for 'string' type data.
            if kind == "S":
                value_type_name = "bytes"
            else:
                value_type_name = "string"
        else:
            value_type_name = dtype.name

        return value_type_name


class AncillaryVariable(_DimensionalMetadata):
    def __init__(
        self,
        data,
        standard_name=None,
        long_name=None,
        var_name=None,
        units=None,
        attributes=None,
    ):
        """Construct a single ancillary variable.

        Parameters
        ----------
        data :
            The values of the ancillary variable.
        standard_name : optional
            CF standard name of the ancillary variable.
        long_name : optional
            Descriptive name of the ancillary variable.
        var_name : optional
            The netCDF variable name for the ancillary variable.
        units : optional
            The :class:`~cf_units.Unit` of the ancillary variable's values.
            Can be a string, which will be converted to a Unit object.
        attributes : optional
            A dictionary containing other cf and user-defined attributes.

        """
        # Configure the metadata manager.
        if not hasattr(self, "_metadata_manager"):
            self._metadata_manager = metadata_manager_factory(AncillaryVariableMetadata)

        super().__init__(
            values=data,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            units=units,
            attributes=attributes,
        )

    @property
    def data(self):
        return self._values

    @data.setter
    def data(self, data):
        self._values = data

    def lazy_data(self):
        """Return a lazy array representing the ancillary variable's data.

        Accessing this method will never cause the data values to be loaded.
        Similarly, calling methods on, or indexing, the returned Array
        will not cause the ancillary variable to have loaded data.

        If the data have already been loaded for the ancillary variable, the
        returned Array will be a new lazy array wrapper.

        Returns
        -------
        A lazy array, representing the ancillary variable data array.

        """
        return super()._lazy_values()

    def core_data(self):
        """Return data array at the core of this ancillary variable.

        The data array at the core of this ancillary variable, which may be a
        NumPy array or a dask array.

        """
        return super()._core_values()

    def has_lazy_data(self):
        """Indicate whether the ancillary variable's data array is a lazy dask array or not."""
        return super()._has_lazy_values()

    def cube_dims(self, cube):
        """Return the cube dimensions of this AncillaryVariable.

        Equivalent to "cube.ancillary_variable_dims(self)".

        """
        return cube.ancillary_variable_dims(self)


class CellMeasure(AncillaryVariable):
    """A CF Cell Measure, providing area or volume properties of a cell.

    A CF Cell Measure, providing area or volume properties of a cell
    where these cannot be inferred from the Coordinates and
    Coordinate Reference System.

    """

    def __init__(
        self,
        data,
        standard_name=None,
        long_name=None,
        var_name=None,
        units=None,
        attributes=None,
        measure=None,
    ):
        """Construct a single cell measure.

        Parameters
        ----------
        data :
            The values of the measure for each cell.
            Either a 'real' array (:class:`numpy.ndarray`) or a 'lazy' array
            (:class:`dask.array.Array`).
        standard_name : optional
            CF standard name of the coordinate.
        long_name : optional
            Descriptive name of the coordinate.
        var_name : optional
            The netCDF variable name for the coordinate.
        units : optional
            The :class:`~cf_units.Unit` of the coordinate's values.
            Can be a string, which will be converted to a Unit object.
        attributes : optional
            A dictionary containing other CF and user-defined attributes.
        measure : optional
            A string describing the type of measure. Supported values are
            'area' and 'volume'. The default is 'area'.

        """
        # Configure the metadata manager.
        self._metadata_manager = metadata_manager_factory(CellMeasureMetadata)

        super().__init__(
            data=data,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            units=units,
            attributes=attributes,
        )

        if measure is None:
            measure = "area"

        #: String naming the measure type.
        self.measure = measure

    @property
    def measure(self):
        return self._metadata_manager.measure

    @measure.setter
    def measure(self, measure):
        if measure not in ["area", "volume"]:
            emsg = f"measure must be 'area' or 'volume', got {measure!r}"
            raise ValueError(emsg)
        self._metadata_manager.measure = measure

    def cube_dims(self, cube):
        """Return the cube dimensions of this CellMeasure.

        Equivalent to "cube.cell_measure_dims(self)".

        """
        return cube.cell_measure_dims(self)

    def xml_element(self, doc):
        """Create the :class:`xml.dom.minidom.Element` that describes this :class:`CellMeasure`.

        Parameters
        ----------
        doc :
            The parent :class:`xml.dom.minidom.Document`.

        Returns
        -------
        :class:`xml.dom.minidom.Element`
            The :class:`xml.dom.minidom.Element` that describes this
            :class:`CellMeasure`.

        """
        # Create the XML element as the camelCaseEquivalent of the
        # class name
        element = super().xml_element(doc=doc)

        # Add the 'measure' property
        element.setAttribute("measure", self.measure)

        return element


class CoordExtent(
    namedtuple(
        "_CoordExtent",
        [
            "name_or_coord",
            "minimum",
            "maximum",
            "min_inclusive",
            "max_inclusive",
        ],
    )
):
    """Defines a range of values for a coordinate."""

    def __new__(
        cls,
        name_or_coord,
        minimum,
        maximum,
        min_inclusive=True,
        max_inclusive=True,
    ):
        """Create a CoordExtent for the specified coordinate and range of values.

        Parameters
        ----------
        name_or_coord :
            Either a coordinate name or a coordinate, as defined in
            :meth:`iris.cube.Cube.coords()`.
        minimum :
            The minimum value of the range to select.
        maximum :
            The maximum value of the range to select.
        min_inclusive : bool, default=True
            If True, coordinate values equal to `minimum` will be included
            in the selection. Default is True.
        max_inclusive : bool, default=True
            If True, coordinate values equal to `maximum` will be included
            in the selection. Default is True.

        """
        return super().__new__(
            cls, name_or_coord, minimum, maximum, min_inclusive, max_inclusive
        )

    __slots__ = ()


# Coordinate cell styles. Used in plot and cartography.
POINT_MODE = 0
BOUND_MODE = 1

BOUND_POSITION_START = 0
BOUND_POSITION_MIDDLE = 0.5
BOUND_POSITION_END = 1


def _get_2d_coord_bound_grid(bounds):
    """Create a grid using the bounds of a 2D coordinate with 4 sided cells.

    Assumes that the four vertices of the cells are in an anti-clockwise order
    (bottom-left, bottom-right, top-right, top-left).

    Selects the zeroth vertex of each cell. A final column is added, which
    contains the first vertex of the cells in the final column. A final row
    is added, which contains the third vertex of all the cells in the final
    row, except for in the final column where it uses the second vertex.
    e.g.
    # 0-0-0-0-1
    # 0-0-0-0-1
    # 3-3-3-3-2

    Parameters
    ----------
    bounds : array
        Coordinate bounds array of shape (Y, X, 4).

    Returns
    -------
    array
        Grid of shape (Y+1, X+1).

    """
    # Check bds has the shape (ny, nx, 4)
    if not (bounds.ndim == 3 and bounds.shape[-1] == 4):
        raise ValueError(
            "Bounds for 2D coordinates must be 3-dimensional and "
            "have 4 bounds per point."
        )

    bounds_shape = bounds.shape
    result = np.zeros((bounds_shape[0] + 1, bounds_shape[1] + 1))

    result[:-1, :-1] = bounds[:, :, 0]
    result[:-1, -1] = bounds[:, -1, 1]
    result[-1, :-1] = bounds[-1, :, 3]
    result[-1, -1] = bounds[-1, -1, 2]

    return result


class Cell(namedtuple("Cell", ["point", "bound"])):
    """A coordinate cell containing a single point, or point and bounds.

    An immutable representation of a single cell of a coordinate, including the
    sample point and/or boundary position.

    Notes on cell comparison:

    Cells are compared in two ways, depending on whether they are
    compared to another Cell, or to a number/string.

    Cell-Cell comparison is defined to produce a strict ordering. If
    two cells are not exactly equal (i.e. including whether they both
    define bounds or not) then they will have a consistent relative
    order.

    Cell-number and Cell-string comparison is defined to support
    Constraint matching. The number/string will equal the Cell if, and
    only if, it is within the Cell (including on the boundary). The
    relative comparisons (lt, le, ..) are defined to be consistent with
    this interpretation. So for a given value `n` and Cell `cell`, only
    one of the following can be true:

    |    n < cell
    |    n == cell
    |    n > cell

    Similarly, `n <= cell` implies either `n < cell` or `n == cell`.
    And `n >= cell` implies either `n > cell` or `n == cell`.

    """

    # This subclass adds no attributes.
    __slots__ = ()

    # Make this class's comparison operators override those of numpy
    __array_priority__ = 100

    def __new__(cls, point=None, bound=None):
        """Construct a Cell from point or point-and-bound information."""
        if point is None:
            raise ValueError("Point must be defined.")

        if bound is not None:
            bound = tuple(bound)

        if isinstance(point, np.ndarray):
            point = tuple(point.flatten())

        if isinstance(point, (tuple, list)):
            if len(point) != 1:
                raise ValueError(
                    "Point may only be a list or tuple if it has length 1."
                )
            point = point[0]

        return super().__new__(cls, point, bound)

    def __mod__(self, mod):
        point = self.point
        bound = self.bound
        if point is not None:
            point = point % mod
        if bound is not None:
            bound = tuple([val % mod for val in bound])
        return Cell(point, bound)

    def __add__(self, mod):
        point = self.point
        bound = self.bound
        if point is not None:
            point = point + mod
        if bound is not None:
            bound = tuple([val + mod for val in bound])
        return Cell(point, bound)

    def __hash__(self):
        # See __eq__ for the definition of when two cells are equal.
        if self.bound is None:
            return hash(self.point)
        bound = self.bound
        rbound = bound[::-1]
        if rbound < bound:
            bound = rbound
        return hash((self.point, bound))

    def __eq__(self, other):
        """Compare Cell equality depending on the type of the object to be compared."""
        if isinstance(other, (int, float, np.number)) or hasattr(other, "timetuple"):
            if self.bound is not None:
                return self.contains_point(other)
            else:
                return self.point == other
        elif isinstance(other, Cell):
            return (self.point == other.point) and (
                self.bound == other.bound or self.bound == other.bound[::-1]
            )
        elif (
            isinstance(other, str)
            and self.bound is None
            and isinstance(self.point, str)
        ):
            return self.point == other
        else:
            return NotImplemented

    # Must supply __ne__, Python does not defer to __eq__ for negative equality
    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            result = not result
        return result

    def __common_cmp__(self, other, operator_method):
        """Common equality comparison.

        Common method called by the rich comparison operators. The method of
        checking equality depends on the type of the object to be compared.

        Cell vs Cell comparison is used to define a strict order.
        Non-Cell vs Cell comparison is used to define Constraint matching.

        """  # noqa: D401
        if (isinstance(other, list) and len(other) == 1) or (
            isinstance(other, np.ndarray) and other.shape == (1,)
        ):
            other = other[0]
        if isinstance(other, np.ndarray) and other.shape == ():
            other = float(other)
        if not (
            isinstance(other, (int, float, np.number, Cell))
            or hasattr(other, "timetuple")
        ):
            raise TypeError("Unexpected type of other {}.".format(type(other)))
        if operator_method not in (
            operator.gt,
            operator.lt,
            operator.ge,
            operator.le,
        ):
            raise ValueError("Unexpected operator_method")

        if isinstance(other, Cell):
            # Cell vs Cell comparison for providing a strict sort order
            if self.bound is None:
                if other.bound is None:
                    # Point vs point
                    # - Simple ordering
                    result = operator_method(self.point, other.point)
                else:
                    # Point vs point-and-bound
                    # - Simple ordering of point values, but if the two
                    #   points are equal, we make the arbitrary choice
                    #   that the point-only Cell is defined as less than
                    #   the point-and-bound Cell.
                    if self.point == other.point:
                        result = operator_method in (operator.lt, operator.le)
                    else:
                        result = operator_method(self.point, other.point)
            else:
                if other.bound is None:
                    # Point-and-bound vs point
                    # - Simple ordering of point values, but if the two
                    #   points are equal, we make the arbitrary choice
                    #   that the point-only Cell is defined as less than
                    #   the point-and-bound Cell.
                    if self.point == other.point:
                        result = operator_method in (operator.gt, operator.ge)
                    else:
                        result = operator_method(self.point, other.point)
                else:
                    # Point-and-bound vs point-and-bound
                    # - Primarily ordered on minimum-bound. If the
                    #   minimum-bounds are equal, then ordered on
                    #   maximum-bound. If the maximum-bounds are also
                    #   equal, then ordered on point values.
                    if self.bound[0] == other.bound[0]:
                        if self.bound[1] == other.bound[1]:
                            result = operator_method(self.point, other.point)
                        else:
                            result = operator_method(self.bound[1], other.bound[1])
                    else:
                        result = operator_method(self.bound[0], other.bound[0])
        else:
            # Cell vs number (or string, or datetime-like) for providing
            # Constraint behaviour.
            if self.bound is None:
                # Point vs number
                # - Simple matching
                me = self.point
            else:
                # Point-and-bound vs number
                # - Match if "within" the Cell
                if operator_method in [operator.gt, operator.le]:
                    me = min(self.bound)
                else:
                    me = max(self.bound)

            result = operator_method(me, other)

        return result

    def __ge__(self, other):
        return self.__common_cmp__(other, operator.ge)

    def __le__(self, other):
        return self.__common_cmp__(other, operator.le)

    def __gt__(self, other):
        return self.__common_cmp__(other, operator.gt)

    def __lt__(self, other):
        return self.__common_cmp__(other, operator.lt)

    def __str__(self):
        if self.bound is not None:
            return repr(self)
        else:
            return str(self.point)

    def contains_point(self, point):
        """For a bounded cell, returns whether the given point lies within the bounds.

        .. note:: The test carried out is equivalent to min(bound)
                  <= point <= max(bound).

        """
        if self.bound is None:
            raise ValueError("Point cannot exist inside an unbounded cell.")
        return np.min(self.bound) <= point <= np.max(self.bound)


class Coord(_DimensionalMetadata):
    """Abstract base class for coordinates."""

    _values_array_name = "points"

    @abstractmethod
    def __init__(
        self,
        points,
        standard_name=None,
        long_name=None,
        var_name=None,
        units=None,
        bounds=None,
        attributes=None,
        coord_system=None,
        climatological=False,
    ):
        """Coordinate abstract base class.

        As of ``v3.0.0`` you **cannot** create an instance of :class:`Coord`.

        Parameters
        ----------
        points :
            The values (or value in the case of a scalar coordinate) for each
            cell of the coordinate.
        standard_name : optional
            CF standard name of the coordinate.
        long_name : optional
            Descriptive name of the coordinate.
        var_name : optional
            The netCDF variable name for the coordinate.
        units : optional
            The :class:`~cf_units.Unit` of the coordinate's values.
            Can be a string, which will be converted to a Unit object.
        bounds : optional
            An array of values describing the bounds of each cell. Given n
            bounds for each cell, the shape of the bounds array should be
            points.shape + (n,). For example, a 1D coordinate with 100 points
            and two bounds per cell would have a bounds array of shape
            (100, 2)
            Note if the data is a climatology, `climatological`
            should be set.
        attributes : optional
            A dictionary containing other CF and user-defined attributes.
        coord_system : optional
            A :class:`~iris.coord_systems.CoordSystem` representing the
            coordinate system of the coordinate,
            e.g., a :class:`~iris.coord_systems.GeogCS` for a longitude coordinate.
        climatological : bool, default=False
            When True: the coordinate is a NetCDF climatological time axis.
            When True: saving in NetCDF will give the coordinate variable a
            'climatology' attribute and will create a boundary variable called
            '<coordinate-name>_climatology' in place of a standard bounds
            attribute and bounds variable.
            Will set to True when a climatological time axis is loaded
            from NetCDF.
            Always False if no bounds exist.

        """
        # Configure the metadata manager.
        if not hasattr(self, "_metadata_manager"):
            self._metadata_manager = metadata_manager_factory(CoordMetadata)

        super().__init__(
            values=points,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            units=units,
            attributes=attributes,
        )

        #: Relevant coordinate system (if any).
        self.coord_system = coord_system

        # Set up bounds DataManager attributes and the bounds values.
        self._bounds_dm = None
        self.bounds = bounds
        self.climatological = climatological

        self._ignore_axis = DEFAULT_IGNORE_AXIS

    def copy(self, points=None, bounds=None):
        """Return a copy of this coordinate.

        Parameters
        ----------
        points : optional
            A points array for the new coordinate.
            This may be a different shape to the points of the coordinate
            being copied.
        bounds : optional
            A bounds array for the new coordinate.
            Given n bounds for each cell, the shape of the bounds array
            should be points.shape + (n,). For example, a 1d coordinate
            with 100 points and two bounds per cell would have a bounds
            array of shape (100, 2).

        Notes
        -----
        .. note:: If the points argument is specified and bounds are not, the
                  resulting coordinate will have no bounds.

        """
        if points is None and bounds is not None:
            raise ValueError("If bounds are specified, points must also be specified")

        new_coord = super().copy(values=points)
        if points is not None:
            # Regardless of whether bounds are provided as an argument, new
            # points will result in new bounds, discarding those copied from
            # self.
            new_coord.bounds = bounds

        # The state of ignore_axis is controlled by the coordinate rather than
        # the metadata manager
        new_coord.ignore_axis = self.ignore_axis

        return new_coord

    @classmethod
    def from_coord(cls, coord):
        """Create a new Coord of this type, from the given coordinate."""
        kwargs = {
            "points": coord.core_points(),
            "bounds": coord.core_bounds(),
            "standard_name": coord.standard_name,
            "long_name": coord.long_name,
            "var_name": coord.var_name,
            "units": coord.units,
            "attributes": coord.attributes,
            "coord_system": copy.deepcopy(coord.coord_system),
            "climatological": coord.climatological,
        }
        if issubclass(cls, DimCoord):
            # DimCoord introduces an extra constructor keyword.
            kwargs["circular"] = getattr(coord, "circular", False)

        new_coord = cls(**kwargs)

        # The state of ignore_axis is controlled by the coordinate rather than
        # the metadata manager
        new_coord.ignore_axis = coord.ignore_axis

        return new_coord

    @property
    def points(self):
        """The coordinate points values as a NumPy array."""
        return self._values

    @points.setter
    def points(self, points):
        self._values = points

    @property
    def bounds(self):
        """Coordinate bounds values.

        The coordinate bounds values, as a NumPy array,
        or None if no bound values are defined.

        .. note:: The shape of the bound array should be: ``points.shape +
            (n_bounds, )``.

        """
        bounds = None
        if self.has_bounds():
            bounds = self._bounds_dm.data.view()
        return bounds

    @bounds.setter
    def bounds(self, bounds):
        # Ensure the bounds are a compatible shape.
        if bounds is None:
            self._bounds_dm = None
            self.climatological = False
        else:
            bounds = self._sanitise_array(bounds, 2)
            if self.shape != bounds.shape[:-1]:
                raise ValueError("Bounds shape must be compatible with points shape.")
            if not self.has_bounds() or self.core_bounds().shape != bounds.shape:
                # Construct a new bounds DataManager.
                self._bounds_dm = DataManager(bounds)
            else:
                self._bounds_dm.data = bounds

    @property
    def coord_system(self):
        """The coordinate-system of the coordinate."""
        return self._metadata_manager.coord_system

    @coord_system.setter
    def coord_system(self, value):
        self._metadata_manager.coord_system = value

    @property
    def climatological(self):
        """Flag for representing a climatological time axis.

        A boolean that controls whether the coordinate is a climatological
        time axis, in which case the bounds represent a climatological period
        rather than a normal period.

        Always reads as False if there are no bounds.
        On set, the input value is cast to a boolean, exceptions raised
        if units are not time units or if there are no bounds.

        """
        if not self.has_bounds():
            self._metadata_manager.climatological = False
        if not self.units.is_time_reference():
            self._metadata_manager.climatological = False
        return self._metadata_manager.climatological

    @climatological.setter
    def climatological(self, value):
        # Ensure the bounds are a compatible shape.
        value = bool(value)
        if value:
            if not self.units.is_time_reference():
                emsg = (
                    "Cannot set climatological coordinate, does not have"
                    " valid time reference units, got {!r}."
                )
                raise TypeError(emsg.format(self.units))

            if not self.has_bounds():
                emsg = "Cannot set climatological coordinate, no bounds exist."
                raise ValueError(emsg)

        self._metadata_manager.climatological = value

    @property
    def ignore_axis(self):
        """A boolean controlling if iris.util.guess_coord_axis acts on this coordinate.

        Defaults to ``False``, and when set to ``True`` it will be skipped by
        :func:`iris.util.guess_coord_axis`.
        """
        return self._ignore_axis

    @ignore_axis.setter
    def ignore_axis(self, value):
        if not isinstance(value, bool):
            emsg = "'ignore_axis' can only be set to 'True' or 'False'"
            raise ValueError(emsg)
        self._ignore_axis = value

    def lazy_points(self):
        """Return a lazy array representing the coord points.

        Accessing this method will never cause the points values to be loaded.
        Similarly, calling methods on, or indexing, the returned Array
        will not cause the coord to have loaded points.

        If the data have already been loaded for the coord, the returned
        Array will be a new lazy array wrapper.

        Returns
        -------
        A lazy array, representing the coord points array.

        """
        return super()._lazy_values()

    def lazy_bounds(self):
        """Return a lazy array representing the coord bounds.

        Accessing this method will never cause the bounds values to be loaded.
        Similarly, calling methods on, or indexing, the returned Array
        will not cause the coord to have loaded bounds.

        If the data have already been loaded for the coord, the returned
        Array will be a new lazy array wrapper.

        Returns
        -------
        lazy array
            A lazy array representing the coord bounds array or `None` if the
            coord does not have bounds.

        """
        lazy_bounds = None
        if self.has_bounds():
            lazy_bounds = self._bounds_dm.lazy_data()
        return lazy_bounds

    def core_points(self):
        """Core points array at the core of this coord, which may be a NumPy array or a dask array."""
        return super()._core_values()

    def core_bounds(self):
        """Core bounds. The points array at the core of this coord, which may be a NumPy array or a dask array."""
        result = None
        if self.has_bounds():
            result = self._bounds_dm.core_data()
            if not _lazy.is_lazy_data(result):
                result = result.view()
        return result

    def has_lazy_points(self):
        """Return a boolean whether the coord's points array is a lazy dask array or not."""
        return super()._has_lazy_values()

    def has_lazy_bounds(self):
        """Whether coordinate bounds are lazy.

        Return a boolean indicating whether the coord's bounds array is a
        lazy dask array or not.

        """
        result = False
        if self.has_bounds():
            result = self._bounds_dm.has_lazy_data()
        return result

    # Must supply __hash__ as Python 3 does not enable it if __eq__ is defined.
    # NOTE: Violates "objects which compare equal must have the same hash".
    # We ought to remove this, as equality of two coords can *change*, so they
    # really should not be hashable.
    # However, current code needs it, e.g. so we can put them in sets.
    # Fixing it will require changing those uses.  See #962 and #1772.
    def __hash__(self):
        return hash(id(self))

    def cube_dims(self, cube):
        """Return the cube dimensions of this Coord.

        Equivalent to "cube.coord_dims(self)".

        """
        return cube.coord_dims(self)

    def convert_units(self, unit):
        r"""Change the coordinate's units, converting the values in its points and bounds arrays.

        For example, if a coordinate's :attr:`~iris.coords.Coord.units`
        attribute is set to radians then::

            coord.convert_units('degrees')

        will change the coordinate's
        :attr:`~iris.coords.Coord.units` attribute to degrees and
        multiply each value in :attr:`~iris.coords.Coord.points` and
        :attr:`~iris.coords.Coord.bounds` by 180.0/:math:`\pi`.

        Full list of supported units can be found in the UDUNITS-2 documentation
        https://docs.unidata.ucar.edu/udunits/current/#Database
        """
        super().convert_units(unit=unit)

    def cells(self):
        """Return an iterable of Cell instances for this Coord.

        For example::

           for cell in self.cells():
              ...

        """
        if self.ndim != 1:
            raise iris.exceptions.CoordinateMultiDimError(self)

        points = self.points
        bounds = self.bounds
        if self.units.is_time_reference():
            points = self.units.num2date(points)
            if self.has_bounds():
                bounds = self.units.num2date(bounds)

        if self.has_bounds():
            for point, bound in zip(points, bounds):
                yield Cell(point, bound)
        else:
            for point in points:
                yield Cell(point)

    def _sanity_check_bounds(self):
        if self.ndim == 1:
            if self.nbounds != 2:
                raise ValueError(
                    "Invalid operation for {!r}, with {} "
                    "bound(s). Contiguous bounds are only "
                    "defined for 1D coordinates with 2 "
                    "bounds.".format(self.name(), self.nbounds)
                )
        elif self.ndim == 2:
            if self.nbounds != 4:
                raise ValueError(
                    "Invalid operation for {!r}, with {} "
                    "bound(s). Contiguous bounds are only "
                    "defined for 2D coordinates with 4 "
                    "bounds.".format(self.name(), self.nbounds)
                )
        else:
            raise ValueError(
                "Invalid operation for {!r}. Contiguous bounds "
                "are not defined for coordinates with more than "
                "2 dimensions.".format(self.name())
            )

    def _discontiguity_in_bounds(self, rtol=1e-5, atol=1e-8):
        """Check that the bounds of the coordinate are contiguous.

        rtol : float, default=1e-5
            Relative tolerance that is used when checking contiguity. Defaults
            to 1e-5.
        atol : float, default=1e-8
            Absolute tolerance that is used when checking contiguity. Defaults
            to 1e-8.

        Returns
        -------
        contiguous : bool
            True if there are no discontiguities.
        diffs : array or tuple of arrays
            A boolean array or tuple of boolean arrays which are true where
            there are discontiguities between neighbouring bounds. If self is
            a 2D coord of shape (Y, X), a pair of arrays is returned, where
            the first is an array of differences along the x-axis, of the
            shape (Y, X-1) and the second is an array of differences along
            the y-axis, of the shape (Y-1, X).

        """
        self._sanity_check_bounds()

        if self.ndim == 1:
            contiguous = np.allclose(
                self.bounds[1:, 0], self.bounds[:-1, 1], rtol=rtol, atol=atol
            )
            diffs = ~np.isclose(
                self.bounds[1:, 0], self.bounds[:-1, 1], rtol=rtol, atol=atol
            )

        elif self.ndim == 2:

            def mod360_adjust(compare_axis):
                bounds = self.bounds.copy()

                if compare_axis == "x":
                    # Extract the pairs of upper bounds and lower bounds which
                    # connect along the "x" axis. These connect along indices
                    # as shown by the following diagram:
                    #
                    # 3---2 + 3---2
                    # |   |   |   |
                    # 0---1 + 0---1
                    upper_bounds = np.stack((bounds[:, :-1, 1], bounds[:, :-1, 2]))
                    lower_bounds = np.stack((bounds[:, 1:, 0], bounds[:, 1:, 3]))
                elif compare_axis == "y":
                    # Extract the pairs of upper bounds and lower bounds which
                    # connect along the "y" axis. These connect along indices
                    # as shown by the following diagram:
                    #
                    # 3---2
                    # |   |
                    # 0---1
                    # +   +
                    # 3---2
                    # |   |
                    # 0---1
                    upper_bounds = np.stack((bounds[:-1, :, 3], bounds[:-1, :, 2]))
                    lower_bounds = np.stack((bounds[1:, :, 0], bounds[1:, :, 1]))

                if self.name() in ["longitude", "grid_longitude"]:
                    # If longitude, adjust for longitude wrapping
                    diffs = upper_bounds - lower_bounds
                    index = np.abs(diffs) > 180
                    if index.any():
                        sign = np.sign(diffs)
                        modification = (index.astype(int) * 360) * sign
                        upper_bounds -= modification

                diffs_along_bounds = ~np.isclose(
                    upper_bounds, lower_bounds, rtol=rtol, atol=atol
                )
                diffs_along_axis = np.logical_or(
                    diffs_along_bounds[0], diffs_along_bounds[1]
                )

                contiguous_along_axis = ~np.any(diffs_along_axis)
                return diffs_along_axis, contiguous_along_axis

            diffs_along_x, match_cell_x1 = mod360_adjust(compare_axis="x")
            diffs_along_y, match_cell_y1 = mod360_adjust(compare_axis="y")

            contiguous = match_cell_x1 and match_cell_y1
            diffs = (diffs_along_x, diffs_along_y)

        return contiguous, diffs

    def is_contiguous(self, rtol=1e-05, atol=1e-08):
        """Whether coordinate has contiguous bounds.

        Return True if, and only if, this Coord is bounded with contiguous
        bounds to within the specified relative and absolute tolerances.

        1D coords are contiguous if the upper bound of a cell aligns,
        within a tolerance, to the lower bound of the next cell along.

        2D coords, with 4 bounds, are contiguous if the lower right corner of
        each cell aligns with the lower left corner of the cell to the right of
        it, and the upper left corner of each cell aligns with the lower left
        corner of the cell above it.

        Parameters
        ----------
        rtol : float, default=1e-05
            The relative tolerance parameter (default is 1e-05).
        atol : float, default=1e-08
            The absolute tolerance parameter (default is 1e-08).

        Returns
        -------
        bool

        """
        if self.has_bounds():
            contiguous, _ = self._discontiguity_in_bounds(rtol=rtol, atol=atol)
        else:
            contiguous = False
        return contiguous

    def contiguous_bounds(self):  # numpydoc ignore=SS05
        """Contiguous bounds of 1D coordinate.

        Return the N+1 bound values for a contiguous bounded 1D coordinate
        of length N, or the (N+1, M+1) bound values for a contiguous bounded 2D
        coordinate of shape (N, M).

        Only 1D or 2D coordinates are supported.

        .. note::

            If the coordinate has bounds, this method assumes they are
            contiguous.

            If the coordinate is 1D and does not have bounds, this method will
            return bounds positioned halfway between the coordinate's points.

            If the coordinate is 2D and does not have bounds, an error will be
            raised.

        """
        if not self.has_bounds():
            if self.ndim == 1:
                warnings.warn(
                    "Coordinate {!r} is not bounded, guessing "
                    "contiguous bounds.".format(self.name()),
                    category=iris.warnings.IrisGuessBoundsWarning,
                )
                bounds = self._guess_bounds()
            elif self.ndim == 2:
                raise ValueError(
                    "2D coordinate {!r} is not bounded. Guessing "
                    "bounds of 2D coords is not currently "
                    "supported.".format(self.name())
                )
        else:
            self._sanity_check_bounds()
            bounds = self.bounds

        if self.ndim == 1:
            c_bounds = np.resize(bounds[:, 0], bounds.shape[0] + 1)
            c_bounds[-1] = bounds[-1, 1]
        elif self.ndim == 2:
            c_bounds = _get_2d_coord_bound_grid(bounds)
        return c_bounds

    def is_monotonic(self):
        """Return True if, and only if, this Coord is monotonic."""
        if self.ndim != 1:
            raise iris.exceptions.CoordinateMultiDimError(self)

        if self.shape == (1,):
            return True

        if self.points is not None:
            if not iris.util.monotonic(self.points, strict=True):
                return False

        if self.has_bounds():
            for b_index in range(self.nbounds):
                if not iris.util.monotonic(self.bounds[..., b_index], strict=True):
                    return False

        return True

    def is_compatible(self, other, ignore=None):
        """Return whether the coordinate is compatible with another.

        Compatibility is determined by comparing
        :meth:`iris.coords.Coord.name()`, :attr:`iris.coords.Coord.units`,
        :attr:`iris.coords.Coord.coord_system` and
        :attr:`iris.coords.Coord.attributes` that are present in both objects.

        Parameters
        ----------
        other :
            An instance of :class:`iris.coords.Coord`,
            :class:`iris.common.CoordMetadata` or
            :class:`iris.common.DimCoordMetadata`.
        ignore : optional
           A single attribute key or iterable of attribute keys to ignore when
           comparing the coordinates. Default is None. To ignore all
           attributes, set this to other.attributes.

        Returns
        -------
        bool

        """
        compatible = False
        if self.coord_system == other.coord_system:
            compatible = super().is_compatible(other=other, ignore=ignore)

        return compatible

    @property
    def bounds_dtype(self):
        """The NumPy dtype of the coordinates bounds.

        The NumPy dtype of the coord's bounds. Will be `None` if the coord
        does not have bounds.

        """
        result = None
        if self.has_bounds():
            result = self._bounds_dm.dtype
        return result

    @property
    def nbounds(self):
        """Return the number of bounds that this coordinate has (0 for no bounds)."""
        nbounds = 0
        if self.has_bounds():
            nbounds = self._bounds_dm.shape[-1]
        return nbounds

    def has_bounds(self):
        """Return a boolean indicating whether the coord has a bounds array."""
        return self._bounds_dm is not None

    def cell(self, index):
        """Point/bound cell at the given coordinate index.

        Return the single :class:`Cell` instance which results from slicing the
        points/bounds with the given index.

        """
        index = iris.util._build_full_slice_given_keys(index, self.ndim)

        point = tuple(np.array(self.core_points()[index], ndmin=1).flatten())
        if len(point) != 1:
            raise IndexError(
                "The index %s did not uniquely identify a single "
                "point to create a cell with." % (index,)
            )

        bound = None
        if self.has_bounds():
            bound = tuple(np.array(self.core_bounds()[index], ndmin=1).flatten())

        if self.units.is_time_reference():
            point = self.units.num2date(point)
            if bound is not None:
                bound = self.units.num2date(bound)

        return Cell(point, bound)

    def collapsed(self, dims_to_collapse=None):
        """Return a copy of this coordinate, which has been collapsed along the specified dimensions.

        Replaces the points & bounds with a simple bounded region.
        """
        # Ensure dims_to_collapse is a tuple to be able to pass
        # through to numpy
        if isinstance(dims_to_collapse, (int, np.integer)):
            dims_to_collapse = (dims_to_collapse,)
        if isinstance(dims_to_collapse, list):
            dims_to_collapse = tuple(dims_to_collapse)

        if np.issubdtype(self.dtype, np.str_):
            # Collapse the coordinate by serializing the points and
            # bounds as strings.
            def serialize(x, axis):
                if axis is None:
                    return "|".join(str(i) for i in x.flatten())

                # np.apply_along_axis combined with str.join will truncate strings in
                # some cases (https://github.com/numpy/numpy/issues/8352), so we need to
                # loop through the array directly. First move (possibly multiple) axis
                # of interest to trailing dim(s), then make a 2D array we can loop
                # through.
                work_array = np.moveaxis(x, axis, range(-len(axis), 0))
                out_shape = work_array.shape[: -len(axis)]
                work_array = work_array.reshape(np.prod(out_shape, dtype=int), -1)

                joined = []
                for arr_slice in work_array:
                    joined.append(serialize(arr_slice, None))

                return np.array(joined).reshape(out_shape)

            bounds = None
            if self.has_bounds():
                # Express dims_to_collapse as non-negative integers.
                if dims_to_collapse is None:
                    dims_to_collapse = range(self.ndim)
                else:
                    dims_to_collapse = tuple(
                        dim % self.ndim for dim in dims_to_collapse
                    )
                bounds = serialize(self.bounds, dims_to_collapse)

            points = serialize(self.points, dims_to_collapse)
            # Create the new collapsed coordinate.
            coord = self.copy(points=np.array(points), bounds=bounds)
        else:
            # Collapse the coordinate by calculating the bounded extremes.
            if self.ndim > 1:
                msg = (
                    "Collapsing a multi-dimensional coordinate. "
                    "Metadata may not be fully descriptive for {!r}."
                )
                warnings.warn(
                    msg.format(self.name()),
                    category=iris.warnings.IrisVagueMetadataWarning,
                )
            else:
                try:
                    self._sanity_check_bounds()
                except ValueError as exc:
                    msg = (
                        "Cannot check if coordinate is contiguous: {} "
                        "Metadata may not be fully descriptive for {!r}. "
                        "Ignoring bounds."
                    )
                    warnings.warn(
                        msg.format(str(exc), self.name()),
                        category=iris.warnings.IrisVagueMetadataWarning,
                    )
                    self.bounds = None
                else:
                    if not self.is_contiguous():
                        msg = (
                            "Collapsing a non-contiguous coordinate. "
                            "Metadata may not be fully descriptive for {!r}."
                        )
                        warnings.warn(
                            msg.format(self.name()),
                            category=iris.warnings.IrisVagueMetadataWarning,
                        )

            if self.has_bounds():
                item = self.core_bounds()
                if dims_to_collapse is not None:
                    # Express main dims_to_collapse as non-negative integers
                    # and add the last (bounds specific) dimension.
                    dims_to_collapse = tuple(
                        dim % self.ndim for dim in dims_to_collapse
                    ) + (-1,)
            else:
                item = self.core_points()

            # Determine the array library for stacking
            al = da if _lazy.is_lazy_data(item) else np

            # Calculate the bounds and points along the right dims
            bounds = al.stack(
                [
                    item.min(axis=dims_to_collapse),
                    item.max(axis=dims_to_collapse),
                ],
                axis=-1,
            )
            points = al.array(bounds.sum(axis=-1) * 0.5, dtype=self.dtype)

            # Create the new collapsed coordinate.
            coord = self.copy(points=points, bounds=bounds)
        return coord

    def _guess_bounds(self, bound_position=0.5, monthly=False, yearly=False):
        """Return bounds for this coordinate based on its points.

        Parameters
        ----------
        bound_position : float, default=0.5
            The desired position of the bounds relative to the position
            of the points.
        monthly : bool, default=False
            If True, the coordinate must be monthly and bounds are set to the
            start and ends of each month.
        yearly : bool, default=False
            If True, the coordinate must be yearly and bounds are set to the
            start and ends of each year.

        Returns
        -------
        A numpy array of shape (len(self.points), 2).

        Notes
        -----
        .. note::

            This method only works for coordinates with ``coord.ndim == 1``.

        """
        # XXX Consider moving into DimCoord
        # ensure we have monotonic points
        if not self.is_monotonic():
            raise ValueError(
                "Need monotonic points to generate bounds for %s" % self.name()
            )

        if self.ndim != 1:
            raise iris.exceptions.CoordinateMultiDimError(self)

        if not monthly and self.shape[0] < 2:
            raise ValueError("Cannot guess bounds for a coordinate of length 1.")

        if self.has_bounds():
            raise ValueError(
                "Coord already has bounds. Remove the bounds before guessing new ones."
            )

        if monthly or yearly:
            if monthly and yearly:
                raise ValueError(
                    "Cannot guess monthly and yearly bounds simultaneously."
                )
            dates = self.units.num2date(self.points)
            lower_bounds = []
            upper_bounds = []
            months_and_years = []
            if monthly:
                for date in dates:
                    if date.month == 12:
                        lyear = date.year
                        uyear = date.year + 1
                        lmonth = 12
                        umonth = 1
                    else:
                        lyear = uyear = date.year
                        lmonth = date.month
                        umonth = date.month + 1
                    date_pair = (date.year, date.month)
                    if date_pair not in months_and_years:
                        months_and_years.append(date_pair)
                    else:
                        raise ValueError(
                            "Cannot guess monthly bounds for a coordinate with multiple "
                            "points in a month."
                        )
                    lower_bounds.append(date.__class__(lyear, lmonth, 1, 0, 0))
                    upper_bounds.append(date.__class__(uyear, umonth, 1, 0, 0))
            elif yearly:
                for date in dates:
                    year = date.year
                    if year not in months_and_years:
                        months_and_years.append(year)
                    else:
                        raise ValueError(
                            "Cannot guess yearly bounds for a coordinate with multiple "
                            "points in a year."
                        )
                    lower_bounds.append(date.__class__(date.year, 1, 1, 0, 0))
                    upper_bounds.append(date.__class__(date.year + 1, 1, 1, 0, 0))
            bounds = self.units.date2num(np.array([lower_bounds, upper_bounds]).T)
            contiguous = np.ma.allclose(bounds[1:, 0], bounds[:-1, 1])
            if not contiguous:
                raise ValueError("Cannot guess bounds for a non-contiguous coordinate.")

        # if not monthly or yearly
        else:
            if getattr(self, "circular", False):
                points = np.empty(self.shape[0] + 2)
                points[1:-1] = self.points
                direction = 1 if self.points[-1] > self.points[0] else -1
                points[0] = self.points[-1] - (self.units.modulus * direction)
                points[-1] = self.points[0] + (self.units.modulus * direction)
                diffs = np.diff(points)
            else:
                diffs = np.diff(self.points)
                diffs = np.insert(diffs, 0, diffs[0])
                diffs = np.append(diffs, diffs[-1])

            min_bounds = self.points - diffs[:-1] * bound_position
            max_bounds = self.points + diffs[1:] * (1 - bound_position)

            bounds = np.array([min_bounds, max_bounds]).transpose()

            if self.name() in ("latitude", "grid_latitude") and self.units == "degree":
                points = self.points
                if (points >= -90).all() and (points <= 90).all():
                    np.clip(bounds, -90, 90, out=bounds)

        return bounds

    def guess_bounds(self, bound_position=0.5, monthly=False, yearly=False):
        """Add contiguous bounds to a coordinate, calculated from its points.

        Puts a cell boundary at the specified fraction between each point and
        the next, plus extrapolated lowermost and uppermost bound points, so
        that each point lies within a cell.

        With regularly spaced points, the resulting bounds will also be
        regular, and all points lie at the same position within their cell.
        With irregular points, the first and last cells are given the same
        widths as the ones next to them.

        Parameters
        ----------
        bound_position : float, default=0.5
            The desired position of the bounds relative to the position
            of the points.
        monthly : bool, default=False
            If True, the coordinate must be monthly and bounds are set to the
            start and ends of each month.
        yearly : bool, default=False
            If True, the coordinate must be yearly and bounds are set to the
            start and ends of each year.


        Notes
        -----
        .. note::

            An error is raised if the coordinate already has bounds, is not
            one-dimensional, or is not monotonic.

        .. note::

            Unevenly spaced values, such from a wrapped longitude range, can
            produce unexpected results :  In such cases you should assign
            suitable values directly to the bounds property, instead.

        .. note::

            Monthly and Yearly work differently from the standard case. They
            can work for single points but cannot be used together.


        """
        self.bounds = self._guess_bounds(bound_position, monthly, yearly)

    def intersect(self, other, return_indices=False):
        """Return a new coordinate from the intersection of two coordinates.

        Both coordinates must be compatible as defined by
        :meth:`~iris.coords.Coord.is_compatible`.

        Parameters
        ----------
        return_indices : bool, default=False
            If True, changes the return behaviour to return the intersection
            indices for the "self" coordinate.

        """
        if not self.is_compatible(other):
            msg = (
                "The coordinates cannot be intersected. They are not "
                "compatible because of differing metadata."
            )
            raise ValueError(msg)

        # Cache self.cells for speed. We can also use the dict for fast index
        # lookup.
        self_cells = {cell: idx for idx, cell in enumerate(self.cells())}

        # Maintain a list of indices on self for which cells exist in both self
        # and other.
        self_intersect_indices = []
        for cell in other.cells():
            if cell in self_cells:
                self_intersect_indices.append(self_cells[cell])

        if return_indices is False and self_intersect_indices == []:
            raise ValueError(
                "No intersection between %s coords possible." % self.name()
            )

        self_intersect_indices = np.array(self_intersect_indices)

        # Return either the indices, or a Coordinate instance of the
        # intersection.
        if return_indices:
            return self_intersect_indices
        else:
            return self[self_intersect_indices]

    def nearest_neighbour_index(self, point):
        """Return the index of the cell nearest to the given point.

        Only works for one-dimensional coordinates.

        For example:

        >>> cube = iris.load_cube(iris.sample_data_path('ostia_monthly.nc'))
        >>> cube.coord('latitude').nearest_neighbour_index(0)
        np.int64(9)
        >>> cube.coord('longitude').nearest_neighbour_index(10)
        np.int64(12)

        .. note:: If the coordinate contains bounds, these will be used to
            determine the nearest neighbour instead of the point values.

        .. note:: For circular coordinates, the 'nearest' point can wrap around
            to the other end of the values.

        """
        points = self.points
        bounds = self.bounds if self.has_bounds() else np.array([])
        if self.ndim != 1:
            raise ValueError(
                "Nearest-neighbour is currently limited to one-dimensional coordinates."
            )
        do_circular = getattr(self, "circular", False)
        if do_circular:
            wrap_modulus = self.units.modulus
            # wrap 'point' to a range based on lowest points or bounds value.
            wrap_origin = np.min(np.hstack((points, bounds.flatten())))
            point = wrap_origin + (point - wrap_origin) % wrap_modulus

        # Calculate the nearest neighbour.
        # The algorithm:  given a single value (V),
        #   if coord has bounds,
        #     make bounds cells complete and non-overlapping
        #     return first cell containing V
        #   else (no bounds),
        #     find the point which is closest to V
        #     or if two are equally close, return the lowest index
        if self.has_bounds():
            # make bounds ranges complete+separate, so point is in at least one
            increasing = self.bounds[0, 1] > self.bounds[0, 0]
            # identify data type that bounds and point can safely cast to
            dtype = np.result_type(bounds, point)
            bounds = bounds.astype(dtype)
            # sort the bounds cells by their centre values
            sort_inds = np.argsort(np.mean(bounds, axis=1))
            bounds = bounds[sort_inds]
            # replace all adjacent bounds with their averages
            if increasing:
                mid_bounds = 0.5 * (bounds[:-1, 1] + bounds[1:, 0])
                bounds[:-1, 1] = mid_bounds
                bounds[1:, 0] = mid_bounds
            else:
                mid_bounds = 0.5 * (bounds[:-1, 0] + bounds[1:, 1])
                bounds[:-1, 0] = mid_bounds
                bounds[1:, 1] = mid_bounds

            # if point lies beyond either end, fix the end cell to include it
            bounds[0, 0] = min(point, bounds[0, 0])
            bounds[-1, 1] = max(point, bounds[-1, 1])
            # get index of first-occurring cell that contains the point
            inside_cells = np.logical_and(
                point >= np.min(bounds, axis=1),
                point <= np.max(bounds, axis=1),
            )
            result_index = np.where(inside_cells)[0][0]
            # return the original index of the cell (before the bounds sort)
            result_index = sort_inds[result_index]

        # Or, if no bounds, we always have points ...
        else:
            if do_circular:
                # add an extra, wrapped max point (simpler than bounds case)
                # NOTE: circular implies a DimCoord, so *must* be monotonic
                if points[-1] >= points[0]:
                    # ascending value order : add wrapped lowest value to end
                    index_offset = 0
                    points = np.hstack((points, points[0] + wrap_modulus))
                else:
                    # descending order : add wrapped lowest value at start
                    index_offset = 1
                    points = np.hstack((points[-1] + wrap_modulus, points))
            # return index of first-occurring nearest point
            distances = np.abs(points - point)
            result_index = np.where(distances == np.min(distances))[0][0]
            if do_circular:
                # convert index back from circular-adjusted points
                result_index = (result_index - index_offset) % self.shape[0]

        return result_index

    def xml_element(self, doc):
        """Create the :class:`xml.dom.minidom.Element` that describes this :class:`Coord`.

        Parameters
        ----------
        doc :
            The parent :class:`xml.dom.minidom.Document`.

        Returns
        -------
        :class:`xml.dom.minidom.Element`
            The :class:`xml.dom.minidom.Element` that will describe this
            :class:`DimCoord`.

        """
        # Create the XML element as the camelCaseEquivalent of the
        # class name
        element = super().xml_element(doc=doc)

        # Add bounds, points are handled by the parent class.
        if self.has_bounds():
            element.setAttribute("bounds", self._xml_array_repr(self.bounds))

        return element

    def _xml_id_extra(self, unique_value):
        """Coord specific stuff for the xml id."""
        unique_value += str(self.coord_system).encode("utf-8") + b"\0"
        return unique_value


_regular_points = lru_cache(iris.util.regular_points)
"""Caching version of iris.util.regular_points"""


class DimCoord(Coord):
    """A coordinate that is 1D, and numeric.

    With values that have a strict monotonic ordering. Missing values are not
    permitted in a :class:`DimCoord`.

    """

    @classmethod
    def from_regular(
        cls,
        zeroth,
        step,
        count,
        standard_name=None,
        long_name=None,
        var_name=None,
        units=None,
        attributes=None,
        coord_system=None,
        circular=False,
        climatological=False,
        with_bounds=False,
    ):
        """Create a :class:`DimCoord` with regularly spaced points, and optionally bounds.

        The majority of the arguments are defined as for
        :class:`Coord`, but those which differ are defined below.

        Parameters
        ----------
        zeroth :
            The value *prior* to the first point value.
        step :
            The numeric difference between successive point values.
        count :
            The number of point values.
        with_bounds : bool, default=False
            If True, the resulting DimCoord will possess bound values
            which are equally spaced around the points. Otherwise no
            bounds values will be defined. Defaults to False.

        """
        # Use lru_cache because this is done repeatedly with the same arguments
        # (particularly in field-based file loading).
        points = _regular_points(zeroth, step, count).copy()
        points.flags.writeable = False

        if with_bounds:
            delta = 0.5 * step
            bounds = np.concatenate([[points - delta], [points + delta]]).T
            bounds.flags.writeable = False
        else:
            bounds = None

        return cls(
            points,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            units=units,
            bounds=bounds,
            attributes=attributes,
            coord_system=coord_system,
            circular=circular,
            climatological=climatological,
        )

    def __init__(
        self,
        points,
        standard_name=None,
        long_name=None,
        var_name=None,
        units=None,
        bounds=None,
        attributes=None,
        coord_system=None,
        circular=False,
        climatological=False,
    ):
        """Create a 1D, numeric, and strictly monotonic coordinate with **immutable** points and bounds.

        Missing values are not permitted.

        Parameters
        ----------
        points :
            1D numpy array-like of values (or single value in the case of a
            scalar coordinate) for each cell of the coordinate.  The values
            must be strictly monotonic and masked values are not allowed.
        standard_name : optional
            CF standard name of the coordinate.
        long_name : optional
            Descriptive name of the coordinate.
        var_name : optional
            The netCDF variable name for the coordinate.
        units : :class:`~cf_units.Unit`, optional
            The :class:`~cf_units.Unit` of the coordinate's values.
            Can be a string, which will be converted to a Unit object.
        bounds : optional
            An array of values describing the bounds of each cell. Given n
            bounds and m cells, the shape of the bounds array should be
            (m, n). For each bound, the values must be strictly monotonic along
            the cells, and the direction of monotonicity must be consistent
            across the bounds.  For example, a DimCoord with 100 points and two
            bounds per cell would have a bounds array of shape (100, 2), and
            the slices ``bounds[:, 0]`` and ``bounds[:, 1]`` would be monotonic
            in the same direction.  Masked values are not allowed.
            Note if the data is a climatology, `climatological`
            should be set.
        attributes : optional
            A dictionary containing other CF and user-defined attributes.
        coord_system : :class:`~iris.coord_systems.CoordSystem`, optional
            A :class:`~iris.coord_systems.CoordSystem` representing the
            coordinate system of the coordinate,
            e.g., a :class:`~iris.coord_systems.GeogCS` for a longitude coordinate.
        circular : bool, default=False
            Whether the coordinate wraps by the :attr:`~iris.coords.DimCoord.units.modulus`
            i.e., the longitude coordinate wraps around the full great circle.
        climatological : bool, default=False
            When True: the coordinate is a NetCDF climatological time axis.
            When True: saving in NetCDF will give the coordinate variable a
            'climatology' attribute and will create a boundary variable called
            '<coordinate-name>_climatology' in place of a standard bounds
            attribute and bounds variable.
            Will set to True when a climatological time axis is loaded
            from NetCDF.
            Always False if no bounds exist.
        """
        # Configure the metadata manager.
        self._metadata_manager = metadata_manager_factory(DimCoordMetadata)

        super().__init__(
            points,
            standard_name=standard_name,
            long_name=long_name,
            var_name=var_name,
            units=units,
            bounds=bounds,
            attributes=attributes,
            coord_system=coord_system,
            climatological=climatological,
        )

        #: Whether the coordinate wraps by ``coord.units.modulus``.
        self.circular = circular

    def __deepcopy__(self, memo):  # numpydoc ignore=SS02
        """coord.__deepcopy__() -> Deep copy of coordinate.

        Used if copy.deepcopy is called on a coordinate.

        """
        new_coord = copy.deepcopy(super(), memo)
        # Ensure points and bounds arrays are read-only.
        new_coord._values_dm.data.flags.writeable = False
        if new_coord._bounds_dm is not None:
            new_coord._bounds_dm.data.flags.writeable = False
        return new_coord

    @property
    def circular(self):
        return self._metadata_manager.circular

    @circular.setter
    def circular(self, circular):
        self._metadata_manager.circular = bool(circular)

    def copy(self, points=None, bounds=None):
        new_coord = super().copy(points=points, bounds=bounds)
        # Make the arrays read-only.
        new_coord._values_dm.data.flags.writeable = False
        if bounds is not None:
            new_coord._bounds_dm.data.flags.writeable = False
        return new_coord

    def __eq__(self, other):
        result = NotImplemented
        if isinstance(other, DimCoord):
            # The "circular" member participates in DimCoord to DimCoord
            # equivalence. We require to do this explicitly here
            # as the "circular" member does NOT participate in
            # DimCoordMetadata to DimCoordMetadata equivalence.
            result = self.circular == other.circular and super().__eq__(other)
        return result

    # The __ne__ operator from Coord implements the not __eq__ method.

    # For Python 3, we must explicitly re-implement the '__hash__' method, as
    # defining an '__eq__' has blocked its inheritance.  See ...
    # https://docs.python.org/3.1/reference/datamodel.html#object.__hash__
    # "If a class that overrides __eq__() needs to retain the
    # implementation of __hash__() from a parent class, the interpreter
    # must be told this explicitly".
    __hash__ = Coord.__hash__

    def __getitem__(self, key):
        coord = super().__getitem__(key)
        coord.circular = self.circular and coord.shape == self.shape
        return coord

    def collapsed(self, dims_to_collapse=None):
        coord = Coord.collapsed(self, dims_to_collapse=dims_to_collapse)
        if self.circular and self.units.modulus is not None:
            bnds = coord.bounds.copy()
            bnds[0, 1] = coord.bounds[0, 0] + self.units.modulus
            coord.bounds = bnds
            coord.points = np.array(np.sum(coord.bounds) * 0.5, dtype=self.points.dtype)
        # XXX This isn't actually correct, but is ported from the old world.
        coord.circular = False
        return coord

    def _new_points_requirements(self, points):
        """Confirm that a new set of coord points adheres to the requirements.

        Confirm that a new set of coord points adheres to the requirements for
        :class:`~iris.coords.DimCoord` points, being:

        * points are scalar or 1D,
        * points are numeric,
        * points are not masked, and
        * points are monotonic.

        """
        if points.ndim not in (0, 1):
            emsg = "The {!r} {} points array must be scalar or 1-dimensional."
            raise ValueError(emsg.format(self.name(), self.__class__.__name__))
        if not np.issubdtype(points.dtype, np.number):
            emsg = "The {!r} {} points array must be numeric."
            raise ValueError(emsg.format(self.name(), self.__class__.__name__))
        if ma.is_masked(points):
            emsg = "A {!r} {} points array must not be masked."
            raise TypeError(emsg.format(self.name(), self.__class__.__name__))
        if points.size > 1 and not iris.util.monotonic(points, strict=True):
            emsg = "The {!r} {} points array must be strictly monotonic."
            raise ValueError(emsg.format(self.name(), self.__class__.__name__))

    @property
    def _values(self):
        # Overridden just to allow .setter override.
        return super()._values

    @_values.setter
    def _values(self, points):
        # DimCoord always realises the points, to allow monotonicity checks.
        # Ensure it is an actual array, and also make our own copy so that we
        # can make it read-only.
        points = _lazy.as_concrete_data(points)
        # Make sure that we have an array (any type of array).
        points = np.asanyarray(points)

        # Check validity requirements for dimension-coordinate points.
        self._new_points_requirements(points)
        # Cast to a numpy array for masked arrays with no mask.
        points = np.array(points)

        super(DimCoord, self.__class__)._values.fset(self, points)

        if self._values_dm is not None:
            # Re-fetch the core array, as the super call may replace it.
            points = self._values_dm.core_data()
            # N.B. always a *real* array, as we realised 'points' at the start.

            # Make the array read-only.
            points.flags.writeable = False

    def _new_bounds_requirements(self, bounds):
        """Confirm that a new set of coord bounds adheres to the requirements.

        Confirm that a new set of coord bounds adheres to the requirements for
        :class:`~iris.coords.DimCoord` bounds, being:

        * bounds are compatible in shape with the points
        * bounds are numeric,
        * bounds are not masked, and
        * bounds are monotonic in the first dimension.

        Also reverse the order of the second dimension if necessary to match the
        first dimension's direction.  I.e. both should increase or both should
        decrease.

        """
        # Ensure the bounds are a compatible shape.
        if self.shape != bounds.shape[:-1] and not (
            self.shape == (1,) and bounds.ndim == 1
        ):
            emsg = (
                "The shape of the {!r} {} bounds array should be "
                "points.shape + (n_bounds)"
            )
            raise ValueError(emsg.format(self.name(), self.__class__.__name__))
        # Checks for numeric.
        if not np.issubdtype(bounds.dtype, np.number):
            emsg = "The {!r} {} bounds array must be numeric."
            raise ValueError(emsg.format(self.name(), self.__class__.__name__))
        # Check not masked.
        if ma.is_masked(bounds):
            emsg = "A {!r} {} bounds array must not be masked."
            raise TypeError(emsg.format(self.name(), self.__class__.__name__))

        # Check bounds are monotonic.
        if bounds.ndim > 1:
            n_bounds = bounds.shape[-1]
            n_points = bounds.shape[0]
            if n_points > 1:
                directions = set()
                for b_index in range(n_bounds):
                    monotonic, direction = iris.util.monotonic(
                        bounds[:, b_index], strict=True, return_direction=True
                    )
                    if not monotonic:
                        emsg = "The {!r} {} bounds array must be strictly monotonic."
                        raise ValueError(
                            emsg.format(self.name(), self.__class__.__name__)
                        )
                    directions.add(direction)

                if len(directions) != 1:
                    emsg = (
                        "The direction of monotonicity for {!r} {} must "
                        "be consistent across all bounds."
                    )
                    raise ValueError(emsg.format(self.name(), self.__class__.__name__))

                if n_bounds == 2:
                    # Make ordering of bounds consistent with coord's direction
                    # if possible.
                    (direction,) = directions
                    diffs = bounds[:, 0] - bounds[:, 1]
                    if np.all(np.sign(diffs) == direction):
                        bounds = np.flip(bounds, axis=1)

        return bounds

    @property
    def bounds(self):
        # Overridden just to allow .setter override.
        return super().bounds

    @bounds.setter
    def bounds(self, bounds):
        if bounds is not None:
            # Ensure we have a realised array of new bounds values.
            bounds = _lazy.as_concrete_data(bounds)
            # Make sure we have an array (any type of array).
            bounds = np.asanyarray(bounds)

            # Check validity requirements for dimension-coordinate bounds and reverse
            # trailing dimension if necessary.
            bounds = self._new_bounds_requirements(bounds)
            # Cast to a numpy array for masked arrays with no mask.
            bounds = np.array(bounds)

        # Call the parent bounds setter.
        super(DimCoord, self.__class__).bounds.fset(self, bounds)

        if self._bounds_dm is not None:
            # Re-fetch the core array, as the super call may replace it.
            bounds = self._bounds_dm.core_data()
            # N.B. always a *real* array, as we realised 'bounds' at the start.

            # Ensure the array is read-only.
            bounds.flags.writeable = False

    def is_monotonic(self):
        return True

    def xml_element(self, doc):
        """Create the :class:`xml.dom.minidom.Element` that describes this :class:`DimCoord`.

        Parameters
        ----------
        doc :
            The parent :class:`xml.dom.minidom.Document`.

        Returns
        -------
        :class:`xml.dom.minidom.Element`
            The :class:`xml.dom.minidom.Element` that describes this
            :class:`DimCoord`.

        """
        element = super().xml_element(doc)
        if self.circular:
            element.setAttribute("circular", str(self.circular))
        return element


class AuxCoord(Coord):
    """A CF auxiliary coordinate."""

    def __init__(self, *args, **kwargs):
        """Create a coordinate with **mutable** points and bounds.

        Parameters
        ----------
        points :
            The values (or value in the case of a scalar coordinate) for each
            cell of the coordinate.
        standard_name : optional
            CF standard name of the coordinate.
        long_name : optional
            Descriptive name of the coordinate.
        var_name : optional
            The netCDF variable name for the coordinate.
        unit : :class:`~cf_units.Unit`, optional
            The :class:`~cf_units.Unit` of the coordinate's values.
            Can be a string, which will be converted to a Unit object.
        bounds : optional
            An array of values describing the bounds of each cell. Given n
            bounds for each cell, the shape of the bounds array should be
            points.shape + (n,). For example, a 1D coordinate with 100 points
            and two bounds per cell would have a bounds array of shape
            (100, 2)
            Note if the data is a climatology, `climatological`
            should be set.
        attributes : optional
            A dictionary containing other CF and user-defined attributes.
        coord_system : :class:`~iris.coord_systems.CoordSystem`, optional
            A :class:`~iris.coord_systems.CoordSystem` representing the
            coordinate system of the coordinate,
            e.g., a :class:`~iris.coord_systems.GeogCS` for a longitude coordinate.
        climatological bool, optional
            When True: the coordinate is a NetCDF climatological time axis.
            When True: saving in NetCDF will give the coordinate variable a
            'climatology' attribute and will create a boundary variable called
            '<coordinate-name>_climatology' in place of a standard bounds
            attribute and bounds variable.
            Will set to True when a climatological time axis is loaded
            from NetCDF.
            Always False if no bounds exist.

        """
        super().__init__(*args, **kwargs)

    # Logically, :class:`Coord` is an abstract class and all actual coords must
    # be members of some concrete subclass, i.e. an :class:`AuxCoord` or
    # a :class:`DimCoord`.
    # So we retain :class:`AuxCoord` as a distinct concrete subclass.
    # This provides clarity, backwards compatibility, and so we can add
    # AuxCoord-specific code if needed in future.


class CellMethod(iris.util._OrderedHashable):
    """Represents a sub-cell pre-processing operation."""

    # Declare the attribute names relevant to the _OrderedHashable behaviour.
    _names = ("method", "coord_names", "intervals", "comments")

    #: The name of the operation that was applied. e.g. "mean", "max", etc.
    method = None

    #: The tuple of coordinate names over which the operation was applied.
    coord_names = None

    #: A description of the original intervals over which the operation
    #: was applied.
    intervals = None

    #: Additional comments.
    comments = None

    def __init__(self, method, coords=None, intervals=None, comments=None):
        """Call Method initialise.

        Parameters
        ----------
        method :
            The name of the operation.
        coords : :class:`.Coord` instances, optional
            A single instance or sequence of :class:`.Coord` instances or
            coordinate names.
        intervals : optional
            A single string, or a sequence strings, describing the intervals
            within the cell method.
        comments : optional
            A single string, or a sequence strings, containing any additional
            comments.

        """
        if not isinstance(method, str):
            raise TypeError("'method' must be a string - got a '%s'" % type(method))

        default_name = BaseMetadata.DEFAULT_NAME
        _coords = []

        if coords is None:
            pass
        elif isinstance(coords, Coord):
            _coords.append(coords.name(token=True))
        elif isinstance(coords, str):
            _coords.append(BaseMetadata.token(coords) or default_name)
        else:
            normalise = (
                lambda coord: coord.name(token=True)
                if isinstance(coord, Coord)
                else BaseMetadata.token(coord) or default_name
            )
            _coords.extend([normalise(coord) for coord in coords])

        _intervals = []
        if intervals is None:
            pass
        elif isinstance(intervals, str):
            _intervals = [intervals]
        else:
            _intervals.extend(intervals)

        _comments = []
        if comments is None:
            pass
        elif isinstance(comments, str):
            _comments = [comments]
        else:
            _comments.extend(comments)

        self._init(method, tuple(_coords), tuple(_intervals), tuple(_comments))

    def __str__(self):
        """Return a custom string representation of CellMethod."""
        # Group related coord names intervals and comments together
        coord_string = " ".join([f"{coord}:" for coord in self.coord_names])
        method_string = str(self.method)
        interval_string = " ".join(
            [f"interval: {interval}" for interval in self.intervals]
        )
        comment_string = " ".join([comment for comment in self.comments])

        if interval_string and comment_string:
            comment_string = "".join(
                [f" comment: {comment}" for comment in self.comments]
            )
        cm_summary = f"{coord_string} {method_string}"

        if interval_string or comment_string:
            cm_summary += f" ({interval_string}{comment_string})"

        return cm_summary

    def __add__(self, other):
        # Disable the default tuple behaviour of tuple concatenation
        return NotImplemented

    def xml_element(self, doc):
        """Create the :class:`xml.dom.minidom.Element` that describes this :class:`CellMethod`.

        Parameters
        ----------
        doc :
            The parent :class:`xml.dom.minidom.Document`.

        Returns
        -------
        :class:`xml.dom.minidom.Element`
            The :class:`xml.dom.minidom.Element` that describes this
            :class:`CellMethod`.

        """
        cellMethod_xml_element = doc.createElement("cellMethod")
        cellMethod_xml_element.setAttribute("method", self.method)

        for coord_name, interval, comment in zip_longest(
            self.coord_names, self.intervals, self.comments
        ):
            coord_xml_element = doc.createElement("coord")
            if coord_name is not None:
                coord_xml_element.setAttribute("name", coord_name)
                if interval is not None:
                    coord_xml_element.setAttribute("interval", interval)
                if comment is not None:
                    coord_xml_element.setAttribute("comment", comment)
                cellMethod_xml_element.appendChild(coord_xml_element)

        return cellMethod_xml_element
