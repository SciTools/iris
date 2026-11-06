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
from iris.fileformats.cf._variables import _CF_ATTRS_IGNORE
from iris.fileformats.cf.dataset import TrackedAttributes


class MockerMixin:
    mocker: MockerFixture

    @pytest.fixture(autouse=True)
    def _mocker_mixin_setup(self, mocker):
        self.mocker = mocker


class CFVariableStandIn:
    """A minimal stand-in for :class:`iris.fileformats.cf.CFVariable`.

    The loading rules read a CF variable's file attributes through exactly
    two things: a real :class:`~iris.fileformats.cf.dataset.TrackedAttributes`
    mapping at ``.attributes``, and ``CFVariable.__getattr__`` routing unknown
    names to it. This class models both, so a test can read
    ``stand_in.some_name`` to build its own expected value and the production
    code's ``stand_in.attributes.get("some_name")`` resolves the identical
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
        # Seed with the same ignored names production CFVariable.__init__
        # does: scale_factor, add_offset and friends must start already-read
        # here too, or a test that puts one into a stand-in sees it as unread
        # (and so, e.g., surviving onto a built object's attributes) when a
        # real load never would.
        self.attributes = TrackedAttributes(dict(attributes), ignored=_CF_ATTRS_IGNORE)

    def __getattr__(self, name):
        # Mirrors production CFVariable.__getattr__'s recursion guard, keyed
        # on _GETATTR_RECURSION_GUARD: "attributes" is what this method reads
        # to resolve anything else, so it must never be resolved by this method
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
        """Index the stand-in's data, exactly as ``CFVariable.__getitem__`` does.

        Real ``CFVariable.__getitem__`` returns ``self.cf_data[key]``; a test
        that needs indexing sets ``.cf_data`` to an indexable data array, the
        same way it sets any other structural member.
        """
        return self.cf_data[key]

    def cf_attrs(self):
        """Return all attribute name/value pairs, exactly as ``CFVariable`` does.

        Used by the "last resort" raw-cube fallback (``build_raw_cube``), so a
        stand-in that hits that path still works without further setup.
        """
        attributes = self.attributes.untracked
        return tuple((name, attributes[name]) for name in sorted(attributes))


class RealArrayCfData:
    """Wrap a real (non-lazy) array as a :class:`CFVariableStandIn`'s ``cf_data``.

    ``read_data`` reads ``is_emulated`` and ``is_variable_length`` before it
    ever looks at size, so a stand-in backed directly by a plain array - as
    most of these tests are, since the array given is the coordinate's real
    data rather than something read from a file - needs this much of the
    storage interface even though the array is always far too small to reach
    ``.chunking``/``.variable``, the two members only the lazy-loading branch
    reads.
    """

    is_emulated = False
    is_variable_length = False

    def __init__(self, array):
        self._array = array

    def __getitem__(self, key):
        return self._array[key]

    def read_data(self, chunking_policy):
        """Return the real array, exactly as the small-variable path would.

        Always far too small to reach the lazy branch, so ``chunking_policy``
        is never consulted.
        """
        return self._array[:]


class _MinimalStorage:
    """The least a :class:`~iris.fileformats.cf.CFVariable` can be built over.

    ``CFVariable.__init__`` reads ``attributes`` (copied into a fresh
    :class:`~iris.fileformats.cf.dataset.TrackedAttributes`) and ``location``
    (which becomes ``filename``); ``dtype`` is read later, by the read-only
    ``CFVariable.dtype`` property. Nothing else here is needed to build one.
    """

    def __init__(self, attributes, location, dtype):
        self.attributes = attributes
        self.location = location
        self.dtype = dtype


def real_cf_data_variable(name="wibble", location="DUMMY", dtype=float, **attributes):
    """Build a genuine :class:`~iris.fileformats.cf.CFDataVariable`.

    ``get_attr_units`` asserts ``isinstance(cf_var, cf.CFDataVariable)`` on the
    ``capture_invalid=True`` branch it takes when building a Cube's own units.
    :class:`CFVariableStandIn` cannot satisfy that assert - it deliberately does
    not inherit from ``CFDataVariable`` - so a test reaching that branch builds
    a real one here instead, over storage minimal enough to construct inline.

    Prefer :class:`CFVariableStandIn` everywhere else. The real class takes its
    ``dtype``, ``shape``, ``ndim`` and ``size`` from storage through read-only
    properties, so a test cannot simply assign them.
    """
    return CFDataVariable(name, _MinimalStorage(dict(attributes), location, dtype))
