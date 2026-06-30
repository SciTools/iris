# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Iris cf_units-based units."""

from datetime import timedelta
import warnings

import cf_units
import numpy as np


class CfUnit(cf_units.Unit):
    """Specialised class for Iris units based on cf_units."""

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
