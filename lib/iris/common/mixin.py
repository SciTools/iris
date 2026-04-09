# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Provides common metadata mixin behaviour."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from functools import wraps
from typing import Any, TypeAlias
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
    pint = None

import numpy as np

import iris.std_names

from .metadata import BaseMetadata

__all__ = ["CFVariableMixin", "LimitedAttributeDict"]


if cfpint is None and cf_units is None:
    raise ImportError("Either 'cfpint' and or 'cf_units' must be installed.")


def _get_valid_standard_name(name):
    # Standard names are optionally followed by a standard name
    # modifier, separated by one or more blank spaces

    if name is not None:
        # Supported standard name modifiers. Ref: [CF] Appendix C.
        valid_std_name_modifiers = [
            "detection_minimum",
            "number_of_observations",
            "standard_error",
            "status_flag",
        ]

        name_groups = name.split(maxsplit=1)
        if name_groups:
            std_name = name_groups[0]
            name_is_valid = std_name in iris.std_names.STD_NAMES
            try:
                std_name_modifier = name_groups[1]
            except IndexError:
                pass  # No modifier
            else:
                name_is_valid &= std_name_modifier in valid_std_name_modifiers

            if not name_is_valid:
                raise ValueError("{!r} is not a valid standard_name".format(name))

    return name


class LimitedAttributeDict(dict):
    """A specialised 'dict' subclass, which forbids (errors) certain attribute names.

    Used for the attribute dictionaries of all Iris data objects (that is,
    :class:`CFVariableMixin` and its subclasses).

    The "excluded" attributes are those which either :mod:`netCDF4` or Iris intpret and
    control with special meaning, which therefore should *not* be defined as custom
    'user' attributes on Iris data objects such as cubes.

    For example : "coordinates", "grid_mapping", "scale_factor".

    The 'forbidden' attributes are those listed in
    :data:`iris.common.mixin.LimitedAttributeDict.CF_ATTRS_FORBIDDEN` .

    All the forbidden attributes are amongst those listed in
    `Appendix A of the CF Conventions: <https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#attribute-appendix>`_
    -- however, not *all* of them, since not all are interpreted by Iris.

    """

    CF_ATTRS_FORBIDDEN = (
        "standard_name",
        "long_name",
        "units",
        "bounds",
        "axis",
        "calendar",
        "leap_month",
        "leap_year",
        "month_lengths",
        "coordinates",
        "grid_mapping",
        "climatology",
        "cell_methods",
        "formula_terms",
        "compress",
        "add_offset",
        "scale_factor",
        "_FillValue",
    )
    """Attributes with special CF meaning, forbidden in Iris attribute dictionaries."""

    IRIS_RAW = "IRIS_RAW"
    """Key used by Iris to store ALL attributes when problems are encountered during loading.

    See Also
    --------
    iris.loading.LOAD_PROBLEMS: The destination for captured loading problems.
    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        # Check validity of keys
        for key in self.keys():
            if key in self.CF_ATTRS_FORBIDDEN:
                raise ValueError(f"{key!r} is not a permitted attribute")

    def __eq__(self, other):
        # Extend equality to allow for NumPy arrays.
        match = set(self.keys()) == set(other.keys())
        if match:
            for key, value in self.items():
                # TODO: should this use the iris.common.metadata approach of
                #  using hexdigest? Might be a breaking change for some corner
                #  cases, so would need a major release.
                match = np.array_equal(
                    np.array(value, ndmin=1), np.array(other[key], ndmin=1)
                )
                if not match:
                    break
        return match

    def __ne__(self, other):
        return not self == other

    def __setitem__(self, key, value):
        if key in self.CF_ATTRS_FORBIDDEN:
            raise ValueError(f"{key!r} is not a permitted attribute")
        dict.__setitem__(self, key, value)

    def update(self, other, **kwargs):
        """Perform standard ``dict.update()`` operation."""
        # Gather incoming keys
        keys = []
        if hasattr(other, "keys"):
            keys += list(other.keys())
        else:
            keys += [k for k, v in other]

        keys += list(kwargs.keys())

        # Check validity of keys
        for key in keys:
            if key in self.CF_ATTRS_FORBIDDEN:
                raise ValueError(f"{key!r} is not a permitted attribute")

        dict.update(self, other, **kwargs)


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
        """Specialisation of cfpint.Unit with extensions for Iris.

        We subclass the basic cfpint.Unit and add Iris-specific behaviour.

        The overrides and extensions here implement necessary behaviour extensions
        for Iris use.

        Further extensions are implemented in :class:`IrisCfulikeCfpintUnit`, which is
        the class *actually used in Iris* for the foreseeable future :  However, those
        further extensions are intended to be temporary -- see there.
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
            """Correct the str() to support the additional categories."""
            # N.B. cfpint.Unit.__repr__ is based on __str__, so we only overload this.
            if self.category != "regular":
                result = self.category
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

        def is_valid_cf_unit(self):
            """Determine whether this is a valid CF unit.

            Notes
            -----
            "udunits" is actually a misnomer, since it allows forms like 'levels' and
            date units in CF styles, which are not strictly UDUNITS2.
            """
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

    class IrisCfulikePintUnit(CfpintUnit):
        """Specialisation of IrisCfpintUnit for backward compatibility.

        This makes a class with specific behaviours to mimic the cf_units API.

        All of this functionality is intended to be temporary :  We will progressively
        replace the Iris units code to stop using these cf_units-like operations on
        pint-type units, which will eventually enable us to stop supporting cf_units.

        We subclass the basic IrisCfpintUnit and add convenience operations for
        backwards compatibility with cf_units.
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

        #
        # cf_units-like unit category test methods.
        #

        def is_udunits(self):
            return self.is_valid_cf_unit()

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

    class PintUnit(IrisCfulikePintUnit):
        """The Iris class for pint-based units.

        This is the standard class used to create pint-based unit properties in Iris.

        Eventually, you will use this as standard to create units,
        i.e. ``PintUnit(arg)`` replaces ``cf_units.Unit(arg)``.

        """

        pass


if cfpint:
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


def _default_units_class():
    from iris.experimental.units import USE_CFPINT

    if USE_CFPINT:
        result = PintUnit
    else:
        result = CfUnit
    return result


def make_unit(arg: cf_units.Unit | pint.Unit | Any) -> CfUnit | CfpintUnit:
    """Convert input into an Iris unit.

    Converts strings to units, and pint/cf_units Units to the Iris specialised
    derived unit types .
    """
    if cf_units is not None and isinstance(arg, cf_units.Unit):
        unit_class = CfUnit
    elif pint is not None and isinstance(arg, pint.Unit):
        unit_class = PintUnit
    else:
        unit_class = _default_units_class()
    return unit_class.from_unit(arg)


class CFVariableMixin:
    _metadata_manager: Any

    @wraps(BaseMetadata.name)
    def name(
        self,
        default: str | None = None,
        token: bool | None = None,
    ) -> str:
        return self._metadata_manager.name(default=default, token=token)

    def rename(self, name: str | None) -> None:
        """Change the human-readable name.

        If 'name' is a valid standard name it will assign it to
        :attr:`standard_name`, otherwise it will assign it to
        :attr:`long_name`.

        """
        try:
            self.standard_name = name
            self.long_name = None
        except ValueError:
            self.standard_name = None
            self.long_name = str(name)

        # Always clear var_name when renaming.
        self.var_name = None

    @property
    def standard_name(self) -> str | None:
        """The CF Metadata standard name for the object."""
        return self._metadata_manager.standard_name

    @standard_name.setter
    def standard_name(self, name: str | None) -> None:
        self._metadata_manager.standard_name = _get_valid_standard_name(name)

    @property
    def long_name(self) -> str | None:
        """The CF Metadata long name for the object."""
        return self._metadata_manager.long_name

    @long_name.setter
    def long_name(self, name: str | None) -> None:
        self._metadata_manager.long_name = name

    @property
    def var_name(self) -> str | None:
        """The NetCDF variable name for the object."""
        return self._metadata_manager.var_name

    @var_name.setter
    def var_name(self, name: str | None) -> None:
        if name is not None:
            result = self._metadata_manager.token(name)
            if result is None or not name:
                emsg = "{!r} is not a valid NetCDF variable name."
                raise ValueError(emsg.format(name))
        self._metadata_manager.var_name = name

    @property
    def units(self) -> cf_units.Unit | cfpint.Unit:
        """The S.I. unit of the object."""
        return self._metadata_manager.units

    @units.setter
    def units(self, unit: cf_units.Unit | cfpint.Unit | str | None) -> None:
        unit = make_unit(unit)
        self._metadata_manager.units = unit

    @property
    def attributes(self) -> LimitedAttributeDict:
        return self._metadata_manager.attributes

    @attributes.setter
    def attributes(self, attributes: Mapping) -> None:
        self._metadata_manager.attributes = LimitedAttributeDict(attributes or {})

    @property
    def metadata(self):
        return self._metadata_manager.values

    @metadata.setter
    def metadata(self, metadata):
        cls = self._metadata_manager.cls
        fields = self._metadata_manager.fields
        arg = metadata

        try:
            # Try dict-like initialisation...
            metadata = cls(**metadata)
        except TypeError:
            try:
                # Try iterator/namedtuple-like initialisation...
                metadata = cls(*metadata)
            except TypeError:
                if hasattr(metadata, "_asdict"):
                    metadata = metadata._asdict()

                if isinstance(metadata, Mapping):
                    fields = [field for field in fields if field in metadata]
                else:
                    # Generic iterable/container with no associated keys.
                    missing = [
                        field for field in fields if not hasattr(metadata, field)
                    ]

                    if missing:
                        missing = ", ".join(map(lambda i: "{!r}".format(i), missing))
                        emsg = "Invalid {!r} metadata, require {} to be specified."
                        raise TypeError(emsg.format(type(arg), missing))

        for field in fields:
            if hasattr(metadata, field):
                value = getattr(metadata, field)
            else:
                value = metadata[field]

            # Ensure to always set state through the individual mixin/container
            # setter functions.
            setattr(self, field, value)
