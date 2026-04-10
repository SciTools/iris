# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Iris pint-based units, based on the 'cfpint' package."""

from enum import StrEnum
from typing import TypeAlias

import cf_units
import cfpint
import cftime
import numpy as np
import pint


class CfpintUnit(cfpint.Unit):
    """Specialise cfpint.Unit with extensions for Iris.

    Notably, add support for "no-unit" + "unknown", as separate 'categories'.

    This class contains all our *permanent* Iris-specific extensions.

    Meanwhile ..

    *  the 'CfulikeUnit' class specialises this + adds *temporary* feaures
       for backward compatibility with cf_units -- which are all deprecated.

    * the 'PintUnit' class then specialises 'CfulikeUnit', but adds only a docstring.
      - this is the 'public' face of the module.

    """

    @classmethod
    def from_unit(cls, unit):
        """Cast anything into the standard Unit class for use within Iris.

        Unit may be a string,
        """
        if isinstance(unit, cls):
            result = unit
        elif isinstance(unit, cf_units.Unit):
            # We need a special case for cf_unit conversion.
            # Although we fallback to str() for native Pint units ('else' below),
            # we can't do that for cf-units **because the str() omits calendars**.
            result = cls(str(unit), calendar=unit.calendar)
        elif unit is None:
            # A special case, so we can support "None" -> "unknown" for object
            # creation with no given units.
            result = cls("unknown")
        else:
            # E.G. probably a string, or a native Pint unit: take the str()
            result = cls(str(unit))
        return result

    class UnitCategory(StrEnum):
        """The possible categories of units.

        These are basically employed to define 'no-unit' and 'unknown' as distinct
        "categories" of a PintUnit, while "regular" implies that the unit is equivalent
        to an 'ordinary' cfpint.Pint object.

        Units of a category other than "regular" are only of meaning *within Iris*.
        For example, they can't be saved to a netcdf-CF dataset.
        """

        regular = "regular"
        no_unit = "no_unit"
        unknown = "unknown"

    _IRIS_CATEGORY_ALIASES = {
        "unknown": ["unknown", "?", ""],
        "no_unit": ["no-unit", "no_unit", "-"],
    }

    @property
    def category(self):
        return self._category

    @category.setter
    def category(self, value):
        self._category = CfpintUnit.UnitCategory(value)

    def __init__(self, *args, **kwargs):
        """Create an Iris pint-based unit."""
        self._category: CfpintUnit.UnitCategory = "regular"
        if args and (arg := args[0]) is None or isinstance(arg, str):
            # Catch + transform "extra" special-category cases.
            if arg is None:
                arg = ""
            arg = arg.lower()
            for name, matches in self._IRIS_CATEGORY_ALIASES.items():
                if arg in matches:
                    self.category = name
                    arg = "1"  # this is how we do it...
            if self.category != "regular":
                # Replace args[0]
                args = tuple([arg] + list(args[1:]))
        super().__init__(*args, **kwargs)
        self.calendar = self.calendar_string

    def __str__(self):
        """Correct the str() to support the additional categories."""
        # N.B. cfpint.Unit.__repr__ is based on __str__, so we only overload this.
        if self.category != "regular":
            result = str(self.category)
        else:
            result = super().__str__()
        return result

    def __repr__(self):
        """Correct the repr() to support the additional categories."""
        if self.category != "regular":
            result = f"<Unit('{self.category}')>"
        elif self.dimensionless:
            # Cfpint fixes this for "str" but not "repr"
            result = f"<Unit('1')>"
        else:
            result = super().__repr__()
        return result

    def is_valid_cf(self):
        """Determine whether this is a valid CF unit."""
        # TODO: this may require an active runtime check on whether the content is
        # valid or not -- effectively == "should we write this to netcdf?"
        ok = self.category == "regular"

    def convert(self, arraylike, other):
        """Scale arraylike data from this unit to another.

        Since Iris objects have a separate 'units' property, and therefore don't
        support 'cfpint.Quantity' in data, this is a useful extension to the cfpint
        API.
        """
        # NOTE: as-at Pint v0.25.3, according to docs, *should* be able to create
        #  masked quantity by multiplying by a unit in the usual way.
        # Currently *cannot*, due to https://github.com/numpy/numpy/issues/15200
        # TODO: fix when possible -- remove mask-specific behaviour.
        is_masked = np.ma.isMaskedArray(arraylike)
        data = arraylike.data if is_masked else arraylike
        quantity = data * self
        quantity = quantity.to(str(other))
        result = quantity.m
        if is_masked:
            result = np.ma.masked_array(result, arraylike.mask)
        return result


class CfuLikeUnit(CfpintUnit):
    """Add cf_units backward compatibility to CfpintUnit.

    This adds specific behaviours to mimic the cf_units API.

    All of this functionality is intended to be temporary.

    We will progressively replace the Iris units code, to stop using these cf_units-like
    operations on pint-type units, which will eventually enable us to stop supporting
    cf_units.

    """

    def __str__(self):
        """Adjust iris pint units __str__() to be more like cf_units."""
        result = super().__str__()
        if self.is_datelike() or self.is_time():
            # Recognise short time units + replace with long forms
            # This seems odd, as Pint units generally do use the longer "name" forms,
            # but we are *reversing* the action of the cfpint "short_formatter"
            # (which mirrors the one in cfxarray.units)
            # : see "cfpint._cfarray_units_like.short_formatter".
            result = self._make_unitstr_cftimelike(result)
        return result

    def __repr__(self):
        """Adjust iris pint units __repr__() to be more like cf_units.

        Note: mostly needed, because assert_CML (i.e. the xml methods) use the reprs.
        """
        result = super().__repr__()
        # Strip off the "<>" wrapping, to give a more cf_units-like repr.
        if len(result) and result[0] == "<":
            result = result[1:]
        if len(result) and result[-1] == ">":
            result = result[:-1]

        if self.is_datelike() or self.is_time():
            # TODO: this should probably be fixed **in cfpint** ?
            result = self._make_unitstr_cftimelike(result)
        return result

    def _make_unitstr_cftimelike(self, units: str) -> str:
        """Replace time-period symbols with names, to be more like cf_units."""
        reps = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
        for char, name in reps.items():
            if units == char:
                units = name
            elif units.startswith(char + " "):
                units = units.replace(char + " ", name + " ", 1)
        return units

    def num2date(
        self,
        time_value,
        only_use_cftime_datetimes=True,
        only_use_python_datetimes=False,
    ) -> np.ndarray:
        """Convert numeric time value(s) to datetimes (1 second resolution).

        The units of the numeric time value are described by the unit and its
        ``.calendar`` property. The returned datetime object(s) represent UTC with
        no time-zone offset, even if the specified unit contain a time-zone
        offset.

        The current unit must be of the form:
        '<time-unit> since <time-origin>'
        e.g. 'hours since 1970-01-01 00:00:00'

        By default, the datetime instances returned are ``cftime.datetime`` objects,
        regardless of calendar.  If both ``only_use_cftime_datetimes`` and
        ``only_use_python_datetimes`` keywords are set ``False``, then the return
        type depends on date : they are datetime.datetime objects if the date falls
        in the Gregorian calendar (i.e. calendar is
        'proleptic_gregorian', 'standard' or 'gregorian' **and** the date is after
        1582-10-15); **otherwise** they are ``cftime.datetime`` objects.
        The datetime instances do not contain a time-zone offset, even if the
        specified unit contains one.

        Works for scalars, sequences and numpy arrays. Returns a scalar
        if input is a scalar, else returns a numpy array.

        Args:

        * time_value (float or arraylike):
            Numeric time value/s. Maximum resolution is 1 second.

        Kwargs:

        * only_use_cftime_datetimes (bool):
            If True, will always return cftime datetime objects, regardless of
            calendar.  If False, returns datetime.datetime instances where
            possible.  Defaults to True.

        * only_use_python_datetimes (bool):
            If True, will always return datetime.datetime instances where
            possible, and raise an exception if not.  Ignored if
            only_use_cftime_datetimes is True.  Defaults to False.

        Returns
        -------
            datetime, or numpy.ndarray of datetime object.
            Either Python ``datetime.datetime`` or ``cftime.datetime`` are possible.

        Notes
        -----
        This mimics :meth:`cf_units.Unit.num2date`.
        But here, it is explicitly re-implemented using only cftime.

        .. deprecated:: 3.15.0
            This method is for interim backwards compatibility with cf_units, and will
            be removed in a future release.  You should replace uses with explicit use
            of :mod:`cftime`.

        """
        if not self.is_datelike():
            raise ValueError(f"Called 'num2date' on a non-datelike unit: {self!r}.")
        units_str = str(self)
        # TODO: this should probably be fixed **in cfpint** ?
        units_str = self._make_unitstr_cftimelike(units_str)
        calendar = self.calendar
        if calendar is None:
            calendar = "standard"
        result = cftime.num2date(
            time_value,
            units=units_str,
            calendar=calendar,
            only_use_cftime_datetimes=only_use_cftime_datetimes,
            only_use_python_datetimes=only_use_python_datetimes,
        )
        return result

    def date2num(self, date):
        """Convert datetime(s) to the numeric time (offset) values.

        Calculated from the datetime objects using the current unit, including its
        ``calendar_string`` property.

        The current unit must be of the form:
        '<time-unit> since <time-origin>'
        e.g. 'hours since 1970-01-01 00:00:00'

        Works for scalars, sequences and numpy arrays. Returns a scalar
        if input is a scalar, else returns a numpy array.

        Return type will be of type `integer` if (all) the times can be
        encoded exactly as an integer with the specified units,
        otherwise a float type will be returned.

        Args:

        * date (datetime):
            A datetime object or a sequence of datetime objects.
            The datetime objects should not include a time-zone offset.
            Both Python ``datetime.datetime`` and ``cftime.datetime`` are supported.

        Returns
        -------
            float/integer or numpy.ndarray of floats/integers

        Notes
        -----
        This mimics :meth:`cf_units.Unit.date2num`.
        But here, it is explicitly re-implemented using only cftime.

        .. deprecated:: 3.15.0
            This method is for interim backwards compatibility with cf_units, and will
            be removed in a future release.  You should replace uses with explicit use
            of :mod:`cftime`.

        """
        if not self.is_datelike():
            raise ValueError(f"Called 'date2num' on a non-datelike unit: {self!r}.")
        units_str = str(self)
        # TODO: this should probably be fixed **in cfpint** ?
        units_str = self._make_unitstr_cftimelike(units_str)
        result = cftime.date2num(date, units_str, self.calendar)
        return result

    #
    # cf_units-like unit-type test methods.
    #

    def is_udunits(self):
        """Whether this is a valid CF unit.

        Tells whether this would be a valid netcdf-CF 'units' attribute.

        Notes
        -----
        This mimics :meth:`cf_units.Unit.is_udunits`.

        "udunits" is actually a misnomer, since we also allow units like 'levels' and
        date units in CF styles, which are not strictly UDUNITS2.
        However, the name is required for compatibility with cf_units.

        .. deprecated:: 3.15.0
            This method is for interim backwards compatibility with cf_units, and will
            be removed in a future release.  You should replace uses with
            :meth:`iris.common.units.PintUnit.is_valid_cf`.

        """
        return self.is_valid_cf()

    def is_unknown(self):
        """Whether this unit is "unknown".

        Notes
        -----
        This mimics :meth:`cf_units.Unit.is_unknown`.

        .. deprecated:: 3.15.0
            This method is for interim backwards compatibility with cf_units, and will
            be removed in a future release.  You should replace uses by testing with
            equality, or test ``unit._category``.

        """
        return self.category == PintUnit.UnitCategory.unknown

    def is_no_unit(self):
        """Whether this unit is "no-unit".

        Notes
        -----
        This mimics :meth:`cf_units.Unit.is_no_unit`.

        .. deprecated:: 3.15.0
            This method is for interim backwards compatibility with cf_units, and will
            be removed in a future release.  You should replace uses by testing with
            equality, or test ``unit._category``.

        """
        return self._category == PintUnit.UnitCategory.no_unit

    def is_time_reference(self):
        """Whether this unit is a date, or time-reference type.

        Notes
        -----
        This mimics :meth:`cf_units.Unit.is_time_reference`.

        .. deprecated:: 3.15.0
            This method is for interim backwards compatibility with cf_units, and will
            be removed in a future release.  You should replace uses with
            :meth:`~iris.common.units.PintUnit.is_datelike`.

        """
        return self.is_datelike()

    def is_long_time_interval(self):
        """Whether the unit period is valid for cftime.

        Notes
        -----
        This mimics :meth:`cf_units.Unit.is_long_time_interval`.
        That method is itself now obsolete, and deprecated.

        .. deprecated:: 3.15.0
            This method is for interim backwards compatibility with cf_units, and will
            be removed in a future release.  Code should no longer have any reason to
            use this.

        """
        return False

    def is_convertible(self, other):
        """Whether the unit period is valid for cftime.

        Notes
        -----
        This mimics :meth:`cf_units.Unit.is_convertible`.

        .. deprecated:: 3.15.0
            This method is for interim backwards compatibility with cf_units, and will
            be removed in a future release.  You should replace uses with
            :meth:`pint.Unit.is_compatible_with`.

        """
        if not isinstance(other, cfpint.Unit):
            other = CfpintUnit(other)
        return self.dimensionality == other.dimensionality

    def is_dimensionless(self):
        """Whether the unit is a pure number.

        Notes
        -----
        This mimics :meth:`cf_units.Unit.is_dimensionless`.

        .. deprecated:: 3.15.0
            This method is for interim backwards compatibility with cf_units, and will
            be removed in a future release.  You should replace uses with the
            :attr:`pint.Unit.dimensionless` property.

        """
        return self.dimensionless

    def is_time(self):
        """Whether the unit is a time period.

        Notes
        -----
        This mimics :meth:`cf_units.Unit.is_time`.

        .. deprecated:: 3.15.0
            This method is for interim backwards compatibility with cf_units, and will
            be removed in a future release.  You should replace uses with a test of the
            :attr:`pint.Unit.dimensionality` property.

        """
        return self.dimensionality == {"[time]": 1}

    def is_vertical(self):
        """Whether the unit is a time period.

        Notes
        -----
        This mimics :meth:`cf_units.Unit.is_time`.

        .. deprecated:: 3.15.0
            This method is for interim backwards compatibility with cf_units, and will
            be removed in a future release.  You should replace uses with a test of the
            :attr:`pint.Unit.dimensionality` property.

        """
        pressure_dims = {"[length]": -1, "[mass]": 1, "[time]": -2}
        height_dims = {"[length]": 1}
        return self.dimensionality in (pressure_dims, height_dims)


class PintUnit(CfuLikeUnit):
    """The Iris class for pint-based units.

    This is the "standard" class used to create pint-based unit properties in Iris.

    Provided pint units will be converted to this, and non-unit content such as strings
    and number also, when enabled by :data:`iris.experimental.units.USE_CFPINT`.

    In a future release, this will become the standard type of units used in Iris.
    At present, Iris object units may be *either* :class:`PintUnit` or :class:`CfUnit`.

    """

    pass


# See: https://pint.readthedocs.io/en/stable/advanced/custom-registry-class.html#custom-quantity-and-unit-class
class IrispintRegistry(pint.registry.UnitRegistry):
    Quantity: TypeAlias = pint.Quantity
    Unit: TypeAlias = PintUnit


# Create our own registry, based on our own UnitRegistry subclass
from cfpint._cfarray_units_like import make_registry

IRIS_PINT_REGISTRY: IrispintRegistry = make_registry(
    IrispintRegistry
)  # include all 'normal' features
pint.set_application_registry(IRIS_PINT_REGISTRY)
pint.application_registry.default_system = "SI"
pint.application_registry.default_format = "cfu"
