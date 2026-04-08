# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Control for unit types."""

from contextlib import contextmanager
from datetime import timedelta
import numpy as np
import threading
from typing import TypeAlias
import warnings

import cftime

try:
    import cf_units
except ImportError:
    cf_units = None

try:
    import cfpint
    import pint
except ImportError:
    cfpint = None


if cfpint is None and cf_units is None:
    raise ImportError("Either 'cfpint' and or 'cf_units' must be installed.")

if cf_units is not None:

    class CfUnit(cf_units.Unit):
        # TODO: remove this subclass once FUTURE.date_microseconds is removed.

        @classmethod
        def from_unit(cls, unit: cf_units.Unit):
            """Cast a :class:`cf_units.Unit` to an :class:`Unit`."""
            unit = cf_units.as_unit(unit)
            if isinstance(unit, CfUnit):
                result = unit
            elif isinstance(unit, cf_units.Unit):
                result = cls.__new__(cls)
                result.__dict__.update(unit.__dict__)
            else:
                message = f"Expected a cf_units.Unit, got {type(unit)}"
                raise TypeError(message)
            return result

        def num2date(
            self,
            time_value,
            only_use_cftime_datetimes=True,
            only_use_python_datetimes=False,
        ):
            # Used to patch the cf_units.Unit.num2date method to round to the
            #  nearest second, which was the legacy behaviour. This is under a FUTURE
            #  flag - users will need to adapt to microsecond precision eventually,
            #  which may involve floating point issues.
            from iris import FUTURE

            def _round(date):
                if date.microsecond == 0:
                    return date
                elif date.microsecond < 500000:
                    return date - timedelta(microseconds=date.microsecond)
                else:
                    return (
                        date
                        + timedelta(seconds=1)
                        - timedelta(microseconds=date.microsecond)
                    )

            result = super().num2date(
                time_value, only_use_cftime_datetimes, only_use_python_datetimes
            )
            if FUTURE.date_microseconds is False:
                message = (
                    "You are using legacy date precision for Iris units - max "
                    "precision is seconds. In future, Iris will use microsecond "
                    "precision - available since cf-units version 3.3 - which may "
                    "affect core behaviour. To opt-in to the "
                    "new behaviour, set `iris.FUTURE.date_microseconds = True`."
                )
                warnings.warn(message, category=FutureWarning)

                if hasattr(result, "shape"):
                    vfunc = np.frompyfunc(_round, 1, 1)
                    result = vfunc(result)
                else:
                    result = _round(result)

            return result

        def __repr__(self):
            # Adjust repr to look like the parent class, to avoid many CML errors.
            string = super().__repr__()
            string = string.replace(self.__class__.__name__ + "(", "Unit(", 1)
            return string


if cfpint is not None:

    class CfpintUnit(cfpint.Unit):
        """Specialisation of cfpint.Unit adding methods for Iris convenience.

        We subclass the basic cfpint.Unit and add convenience operations.
        Most of this is, effectively, for backwards compatibility with cf_units.
        But we probably want to avoid publicising that !

        In the future, it would be great to drop most of this, but we probably will
        continue to need to support the  Iris-internal unit categories "unknown" and
        "no_unit".
        Ideally, the other methods can progressively be dropped by making Iris code
        pint-aware.  But this is awkward whilst we are still supporting both.
        """

        # TODO: ideally we would get rid of this class altogether.
        @classmethod
        def from_unit(cls, unit):
            """Cast anything into the standard Unit class for use within Iris.

            Unit may be a string,
            """
            if isinstance(unit, CfpintUnit):
                result = unit
            elif isinstance(unit, cf_units.Unit):
                # We need a special case for cf_unit conversion.
                # Although we fallback to str() for native Pint units ('else' below),
                # we can't do that for cf-units **because the str() omits calendars**.
                result = CfpintUnit(str(unit), calendar=unit.calendar)
            elif unit is None:
                # A special case, so we can support "None" -> "unknown" for object
                # creation with no given units.
                result = CfpintUnit("unknown")
            else:
                # E.G. probably a string, or a native Pint unit: take the str()
                result = CfpintUnit(str(unit))
            return result

        _IRIS_EXTRA_CATEGORIES = {
            "unknown": ["unknown", "?", ""],
            "no_unit": ["no-unit", "no_unit", "-"],
        }

        def __init__(self, *args, **kwargs):
            self.category = "regular"
            if args and (arg := args[0]) is None or isinstance(arg, str):
                # Catch + transform "extra" special-category cases.
                if arg is None:
                    arg = ""
                arg = arg.lower()
                for name, matches in self._IRIS_EXTRA_CATEGORIES.items():
                    if arg in matches:
                        self.category = name
                        arg = "1"  # this is how we do it...
                if self.category != "regular":
                    # Replace args[0]
                    args = tuple([arg] + list(args[1:]))
            super().__init__(*args, **kwargs)
            self.calendar = self.calendar_string

        def __str__(self):
            # N.B. cfpint.Unit.__repr__ is based on __str__, so we only overload this.
            if self.category != "regular":
                result = self.category
            else:
                result = super().__str__()
                if self.is_datelike() or self.is_time():
                    # Recognise short time units + replace with long forms
                    #  -- weird, as most Pint units work "the other way",
                    #       e.g. "m" -> "metre" !
                    #  -- but probably due to the cfxarray-like "short_formatter"
                    #       - see "cfpint._cfarray_units_like.short_formatter"
                    # TODO: this should probably be fixed **in cfpint** ?
                    result = self._make_unitstr_cftimelike(result)
            return result

        # remove <> from reprs, since CDL seems to use this
        #  (? so calendars are recorded, due to not appearing in str(date-unit) ?)
        # TODO: remove this
        _REPR_NO_LTGT = True

        def __repr__(self):
            """Correct the repr.

            For fuller backwards-compatibility with cf_units,
            mostly because assert_CML (i.e. the xml methods) need it.
            TODO: remove this
            """
            if self.category != "regular":
                result = f"<Unit('{self.category}')>"
            elif self.dimensionless:
                # Cfpint fixes this for "str" but not "repr"
                result = f"<Unit('1')>"
            else:
                result = super().__repr__()

            if self._REPR_NO_LTGT:
                # Just strip off the "<>" wrapping.  Result should then be equivalent
                if len(result) and result[0] == "<":
                    result = result[1:]
                if len(result) and result[-1] == ">":
                    result = result[:-1]

            if self.is_datelike() or self.is_time():
                # TODO: this should probably be fixed **in cfpint** ?
                result = self._make_unitstr_cftimelike(result)
            return result

        def convert(self, arraylike, other):
            is_masked = np.ma.isMaskedArray(arraylike)
            if is_masked:
                arraylike = arraylike.data
            quantity = arraylike * self
            quantity = quantity.to(str(other))
            # TODO: I *think* this is the appropriate way to strip the units.
            result = quantity.m
            if is_masked:
                result = np.ma.masked_array(result, arraylike.mask)
            return result

        def _make_unitstr_cftimelike(self, units: str) -> str:
            """Make a unit string cftime-compatible."""
            # Some kludges needed for now!
            # TODO: to aid use of cftime, this fix **should be in cfpint***
            #   - ideally, fix how cfpint units represent h/m/s/d
            #   - if *not* fixed in basic units registry, at least fix str(), as here
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
            This mimic what we already have in Iris, derived from cf_units behaviour.
            But here, it is explicitly re-implemented using only cftime.
            Ultimately, we will lose this, and users should use cftime explicitly.
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
            This mimics what we already have in Iris, derived from cf_units behaviour.
            But here, it is explicitly re-implemented using only cftime.
            Ultimately, we will lose this, and users should use cftime explicitly.
            """
            if not self.is_datelike():
                raise ValueError(f"Called 'date2num' on a non-datelike unit: {self!r}.")
            units_str = str(self)
            # TODO: this should probably be fixed **in cfpint** ?
            units_str = self._make_unitstr_cftimelike(units_str)
            result = cftime.date2num(date, units_str, self.calendar)
            return result

        def is_udunits(self):
            # TODO: For now!
            return self.category == "regular"

        def is_unknown(self):
            return self.category == "unknown"

        def is_no_unit(self):
            return self.category == "no_unit"

        def is_time_reference(self):
            return self.is_datelike()

        def is_long_time_interval(self):
            return False

        def is_convertible(self, other):
            if not isinstance(other, cfpint.Unit):
                other = CfpintUnit(other)
            return self.dimensionality == other.dimensionality

        def is_dimensionless(self):
            return self.dimensionless

        def is_time(self):
            return self.dimensionality == {"[time]": 1}

        def is_vertical(self):
            pressure_dims = {"[length]": -1, "[mass]": 1, "[time]": -2}
            height_dims = {"[length]": 1}
            return self.dimensionality in (pressure_dims, height_dims)


# And force pint too.
# TODO: since we may have seen problems with doing this dynamically, this could affect
#  the whole attempt to divide functions between Iris and Cfpint functionality

if cfpint:
    # See: https://pint.readthedocs.io/en/stable/advanced/custom-registry-class.html#custom-quantity-and-unit-class
    class IrispintRegistry(pint.registry.UnitRegistry):
        Quantity: TypeAlias = pint.Quantity
        Unit: TypeAlias = CfpintUnit

    # Create our own registry, based on our own UnitRegistry subclass
    from cfpint._cfarray_units_like import make_registry

    IRIS_PINT_REGISTRY: IrispintRegistry = make_registry(
        IrispintRegistry
    )  # include all 'normal' features
    pint.set_application_registry(IRIS_PINT_REGISTRY)
    pint.application_registry.default_system = "SI"
    pint.application_registry.default_format = "cfu"


class UseCfpint(threading.local):
    def __init__(self):
        """Thead-safe state to enable experimental cfpint based unit creation.

        A flag for dictating whether to use the experimental cfpint based units
        :class:`~iris.common.mixin.CfpintUnit` when interpreting unit strings.
        When True, units attributes will be created as :class:`~iris.common.mixin.CfpintUnit`
        (based on the cfpint class :class:`cfpint.Unit`) by default. At present
        you can still assign class:`cf_units.Unit` objects explicitly, and either
        may be used. However, support for cf_units will eventually be retired.
        Object is thread-safe.
        """
        self._state = False

    def __bool__(self):
        return self._state

    @contextmanager
    def context(self, pint_units=True):
        """Temporarily activate experimental cfpint based unit creation.

        Create cfpint based units :class:`~iris.common.mixin.CfpintUnit` when
        interpreting unit strings while within the context manager.

        Use via the run-time switch :const:`~iris.experimental.units.USE_CFPINT`.
        """
        old_state = self._state
        try:
            self._state = pint_units
            yield
        finally:
            self._state = old_state


USE_CFPINT = UseCfpint()


def _default_units_class():
    from iris.experimental.units import USE_CFPINT

    if USE_CFPINT:
        result = CfpintUnit
    else:
        result = CfUnit
    return result


def make_unit(
        arg: None | str | cf_units.Unit | pint.Unit
) -> CfUnit | CfpintUnit:
    """Convert input into an Iris unit.

    Converts strings to units, and pint/cf_units Units to the Iris specialised
    derived unit types .
    """
    unit_class = _default_units_class()
    return unit_class.from_unit(arg)
