# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFVariable`."""

from iris.fileformats import cf as cf


class CFVariableSub(cf.CFVariable):
    """A subclass of CFVariable for testing purposes."""

    def identify(self, variables, ignore=None, target=None, warn=True):
        pass


def test_cached(mocker):
    # Make sure attribute access to the underlying netCDF4.Variable
    # is cached.
    name = "foo"
    nc_var = mocker.MagicMock()
    cf_var = CFVariableSub(name, nc_var)
    assert nc_var.ncattrs.call_count == 1

    # Accessing a netCDF attribute should result in no further calls
    # to nc_var.ncattrs() and the creation of an attribute on the
    # cf_var.
    # NB. Can't use hasattr() because that triggers the attribute
    # to be created!
    assert "coordinates" not in cf_var.__dict__
    _ = cf_var.coordinates
    assert nc_var.ncattrs.call_count == 1
    assert "coordinates" in cf_var.__dict__

    # Trying again results in no change.
    _ = cf_var.coordinates
    assert nc_var.ncattrs.call_count == 1
    assert "coordinates" in cf_var.__dict__

    # Trying another attribute results in just a new attribute.
    assert "standard_name" not in cf_var.__dict__
    _ = cf_var.standard_name
    assert nc_var.ncattrs.call_count == 1
    assert "standard_name" in cf_var.__dict__
