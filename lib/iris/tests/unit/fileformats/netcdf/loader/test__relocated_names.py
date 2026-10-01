# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the deprecated aliases left behind in the netCDF loader.

``CHUNK_CONTROL`` and ``ChunkControl`` moved to
:mod:`iris.fileformats.cf.loader`; the old names still answer, with a warning.
"""

import importlib
import warnings

import pytest

from iris._deprecation import IrisDeprecation
from iris.fileformats.cf import loader as cf_loader
from iris.fileformats.netcdf import loader as netcdf_loader

RELOCATED = ["CHUNK_CONTROL", "ChunkControl"]


@pytest.mark.parametrize("name", RELOCATED)
def test_the_old_name_warns(name):
    with pytest.warns(IrisDeprecation, match=f"netcdf.loader.{name}"):
        getattr(netcdf_loader, name)


@pytest.mark.parametrize("name", RELOCATED)
def test_the_message_names_the_new_location(name):
    with pytest.warns(IrisDeprecation, match=r"iris\.fileformats\.cf\.loader"):
        getattr(netcdf_loader, name)


@pytest.mark.parametrize("name", RELOCATED)
def test_the_alias_is_the_same_object(name):
    # Spec 4.8: "The alias is the same object, not a copy, so existing code
    # keeps working including inside a with CHUNK_CONTROL.set(...)". A copy
    # would give a with-block that silently chunked nothing.
    with pytest.warns(IrisDeprecation):
        assert getattr(netcdf_loader, name) is getattr(cf_loader, name)


def test_importing_the_module_does_not_warn():
    # Spec 4.8: "Warnings are emitted on use, not on import, so simply
    # importing iris.fileformats.netcdf stays quiet."
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        importlib.reload(importlib.import_module("iris.fileformats.netcdf"))


def test_an_unrelated_name_raises_without_warning():
    # hasattr() probes land in __getattr__ too, and a probe that comes back
    # False is not a use of anything.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(AttributeError, match="no_such_name"):
            netcdf_loader.no_such_name


def test_debug_is_not_relocated():
    # Spec 4.8 lists netcdf.DEBUG among the names that do not move. It is a
    # flag users assign to, and a module __getattr__ cannot intercept
    # assignment - so forwarding it would make `netcdf.loader.DEBUG = True`
    # silently stop working. Pin that it is a real attribute here.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert netcdf_loader.DEBUG is False
    assert "DEBUG" not in netcdf_loader._RELOCATED_NAMES


def test_assignment_to_a_relocated_name_does_not_reach_the_new_home():
    # The known limitation of the PEP 562 forward, written down so that it is
    # a documented edge and not a surprise. Assigning here shadows the
    # forward; it does not change iris.fileformats.cf.loader. Nothing in Iris
    # assigns to CHUNK_CONTROL - it is used through its context managers - but
    # if that ever changes, this test says where to look.
    original = cf_loader.CHUNK_CONTROL
    netcdf_loader.CHUNK_CONTROL = "shadow"
    try:
        assert netcdf_loader.CHUNK_CONTROL == "shadow"
        assert cf_loader.CHUNK_CONTROL is original
    finally:
        del netcdf_loader.CHUNK_CONTROL
