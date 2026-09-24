# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the module
:mod:`iris.fileformats.netcdf._nc_load_rules.helpers` .

"""

import pytest
from pytest_mock import MockerFixture

from iris.fileformats.cf import CFDataVariable
from iris.fileformats.cf.dataset import TrackedAttributes


class MockerMixin:
    mocker: MockerFixture

    @pytest.fixture(autouse=True)
    def _mocker_mixin_setup(self, mocker):
        self.mocker = mocker


class CFVariableDouble:
    """A minimal stand-in for :class:`iris.fileformats.cf.CFVariable`.

    The loading rules read a CF variable's file attributes through exactly
    two things: a real :class:`~iris.fileformats.cf.dataset.TrackedAttributes`
    mapping at ``.attributes``, and ``CFVariable.__getattr__`` routing unknown
    names to it. This double models both, so a test can read
    ``double.some_name`` to build its own expected value and the production
    code's ``double.attributes.get("some_name")`` resolves the identical
    value from the same underlying mapping - unlike a flat
    ``Mock(some_name=...)``, which has nothing behind ``.attributes`` at all.

    Every keyword given becomes a CF attribute: present in ``.attributes``
    and returned via plain attribute access (through ``__getattr__``). A name
    not given raises ``AttributeError`` on plain access and, through
    ``.attributes.get``, returns the caller's default - exactly as a real
    missing file attribute would. Structural members that production
    ``CFVariable`` exposes as plain instance attributes (``cf_name``,
    ``cf_group``, ``cf_data``, ``filename``, ...) are not modelled here;
    set them directly on the instance after construction, as real
    ``CFVariable`` does outside ``.attributes``.
    """

    def __init__(self, **attributes):
        self.attributes = TrackedAttributes(dict(attributes))

    def __getattr__(self, name):
        # Mirrors production CFVariable.__getattr__'s recursion guard
        # (_variables.py:86,275): "attributes" is what this method reads to
        # resolve anything else, so it must never be resolved by this method
        # itself. Without this, an instance with no "attributes" yet in its
        # __dict__ - e.g. `cls.__new__(cls)` during copy/deepcopy/unpickling,
        # then a `__setstate__` probe - recurses infinitely instead of
        # raising AttributeError.
        if name.startswith("__") or name == "attributes":
            raise AttributeError(name)
        try:
            return self.attributes[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, key):
        """Index the double's data, exactly as ``CFVariable.__getitem__`` does.

        Real ``CFVariable.__getitem__`` returns ``self.cf_data[key]``; a test
        that needs indexing sets ``.cf_data`` to an indexable data array, the
        same way it sets any other structural member.
        """
        return self.cf_data[key]

    def cf_attrs(self):
        """Return all attribute name/value pairs, exactly as ``CFVariable`` does.

        Used by the "last resort" raw-cube fallback (``build_raw_cube``), so a
        double that hits that path still works without further setup.
        """
        attributes = self.attributes.untracked
        return tuple((name, attributes[name]) for name in sorted(attributes))


# A handful of call sites assert `isinstance(cf_var, CFDataVariable)` as a
# sanity check on their own caller (e.g. `helpers.get_attr_units`, invoked
# with `capture_invalid=True` only when building a Cube's own units). Register
# the double as a virtual subclass so that check passes without inheriting
# CFDataVariable's construction or behaviour - a double is not a real
# CFDataVariable, but the loading rules only ever probe it with `isinstance`.
CFDataVariable.register(CFVariableDouble)
