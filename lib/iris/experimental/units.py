# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Control for unit types."""

from contextlib import contextmanager
import threading


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
