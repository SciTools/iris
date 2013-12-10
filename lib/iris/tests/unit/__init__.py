# (C) British Crown Copyright 2013, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for the :mod:`iris` package."""

# In python 2.x when in __main__ __builtins__ is the built-in module, but
# __builtin__ when in any other module.
# NOTE: __builtin__ has been renamed to builtins in python 3.x
import __builtin__
from contextlib import contextmanager

import mock


@contextmanager
def patched_isinstance(*args, **kwargs):
    """
    Provides a context manager to patch isinstance, which cannot be patched
    in the usual way due to mock's own internal use.

    In order to patch isinstance, the isinstance patched function dances
    around the problem by temporarily reverting to the original isinstance
    during the mock call stage.

    """
    # Take a copy of the original isinstance.
    isinstance_orig = isinstance

    with mock.patch('__builtin__.isinstance',
                    *args, **kwargs) as isinstance_patched:
        # Take a copy of the "callable" function of the patched isinstance.
        mocked_callable = isinstance_patched._mock_call

        # Define a function which reverts isinstance to the original before
        # actually running the mocked callable.
        def revert_isinstance_then_call(*args, **kwargs):
            try:
                __builtin__.isinstance = isinstance_orig
                return mocked_callable(*args, **kwargs)
            finally:
                __builtin__.isinstance = isinstance_patched

        # Update the patched isinstance to use the newly defined function.
        isinstance_patched._mock_call = revert_isinstance_then_call
        yield isinstance_patched
