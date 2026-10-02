# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the deprecated aliases left behind in the netCDF loader.

``CHUNK_CONTROL`` and ``ChunkControl`` moved to
:mod:`iris.fileformats.cf.loader`; the old names still answer, with a warning.
"""

import importlib
from unittest import mock
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


def test_the_warning_points_at_the_caller():
    # warn_deprecated's default stacklevel=2 would name this module's own
    # __getattr__ - an extra frame PEP 562 inserts between the caller and
    # warn_deprecated - rather than the line below. Pin the fix: the warning
    # must be attributed to *this* file, not to netcdf/loader.py.
    with pytest.warns(IrisDeprecation) as record:
        netcdf_loader.CHUNK_CONTROL
    assert record[0].filename == __file__


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
    #
    # Reloads this module, not the whole iris.fileformats.netcdf package: the
    # package's own body only *fetches* DEBUG/NetCDFDataProxy/load_cubes as
    # already-bound names from this (unreloaded) submodule, so reloading the
    # package would never re-execute the code this test means to exercise.
    # Confirmed: reloading just this module leaves the shared
    # "iris.fileformats.netcdf" logger's handler list untouched (the
    # StreamHandler-per-call side effect lives in the package's __init__, in
    # iris.config.get_logger, not here), so there is no handler-leak risk to
    # guard against on this path.
    #
    # Reloading *does* rebuild `load_cubes` as a brand new function object,
    # though, and that is consequential: iris.fileformats.netcdf.load_cubes -
    # bound once, at the package's own first import, and the handler
    # iris.fileformats.FORMAT_AGENT registered at Iris startup - keeps
    # pointing at the *old* one, while iris.fileformats.netcdf.ugrid_load's
    # load_meshes recognises netCDF sources by `==` identity against this
    # module's (new, post-reload) one. Left unrestored, that silently breaks
    # mesh loading for the rest of the process - confirmed by reproducing the
    # failure directly before adding this restore.
    original_load_cubes = netcdf_loader.load_cubes
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            importlib.reload(netcdf_loader)
    finally:
        netcdf_loader.load_cubes = original_load_cubes


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


@pytest.mark.parametrize("name", RELOCATED)
def test_dir_includes_the_relocated_name(name):
    # A module __getattr__ forwards reads but is invisible to dir(), so
    # __dir__ has to union the relocated names in by hand.
    assert name in dir(netcdf_loader)


def test_dir_still_includes_a_name_that_genuinely_lives_here():
    # Pin that __dir__ adds the relocated names rather than replacing the
    # real module listing.
    assert "load_cubes" in dir(netcdf_loader)
    assert "DEBUG" in dir(netcdf_loader)


def test_mock_patch_on_the_old_path_is_a_silent_no_op():
    # mock.patch("iris.fileformats.netcdf.loader.CHUNK_CONTROL", ...)
    # succeeds, because patch simply sets an attribute here - the same
    # mechanism test_assignment_to_a_relocated_name_does_not_reach_the_new_home
    # pins for a plain assignment. Production code reads
    # iris.fileformats.cf.loader.CHUNK_CONTROL, so the patch never reaches it.
    # A downstream test suite patching the old path would go green while
    # testing nothing - this test states that limitation rather than leaving
    # a future reader to discover it the hard way.
    original = cf_loader.CHUNK_CONTROL
    with mock.patch("iris.fileformats.netcdf.loader.CHUNK_CONTROL", "patched"):
        assert netcdf_loader.CHUNK_CONTROL == "patched"
        assert cf_loader.CHUNK_CONTROL is original


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
