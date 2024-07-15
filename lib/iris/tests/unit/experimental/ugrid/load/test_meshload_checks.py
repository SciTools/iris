# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for mesh handling within iris netcdf loads."""

import pytest

import iris
from iris.experimental.ugrid.load import PARSE_UGRID_ON_LOAD

from .test_load_meshes import (
    _TEST_CDL_HEAD,
    _TEST_CDL_TAIL,
    cdl_to_nc,
)


class TestMeshLoad:
    def _create_testnc(self, location="node", meshdim="node"):
        # Add an extra (possibly mal-formed) mesh data to the testfile.
        if location is None:
            location_cdl = ""
        else:
            location_cdl = f'extra_data:location = "{location}" ;'

        extra_cdl = f"""
        float extra_data(levels, {meshdim}) ;
            extra_data:coordinates = "node_x node_y" ;
            {location_cdl}
            extra_data:mesh = "mesh" ;
        """
        # Insert this into the definitions part of the 'standard' testfile CDL
        extended_cdl = _TEST_CDL_HEAD + extra_cdl + _TEST_CDL_TAIL
        testfile_path = cdl_to_nc(extended_cdl, tmpdir=self.tmpdir)
        return testfile_path

    @pytest.fixture(params=["nolocation", "badlocation", "baddim"])
    def failnc(self, request, tmp_path_factory):
        self.param = request.param
        kwargs = {}
        if self.param == "nolocation":
            kwargs["location"] = None
        elif self.param == "badlocation":
            kwargs["location"] = "invalid_location"
        elif self.param == "baddim":
            kwargs["meshdim"] = "vertex"
        else:
            raise ValueError(f"unexpected param: {self.param}")

        self.tmpdir = tmp_path_factory.mktemp("meshload")
        return self._create_testnc(**kwargs)

    def test_extrameshvar__ok(self, tmp_path_factory):
        # Check that the default cdl construction loads OK
        self.tmpdir = tmp_path_factory.mktemp("meshload")
        testnc = self._create_testnc()
        with PARSE_UGRID_ON_LOAD.context():
            iris.load(testnc)

    def test_extrameshvar__fail(self, failnc):
        # Check that the expected errors are raised in various cases.
        param = self.param
        if param == "nolocation":
            match_msg = (
                "mesh data variable 'extra_data' has an " "invalid location='<empty>'."
            )
        elif param == "badlocation":
            match_msg = (
                "mesh data variable 'extra_data' has an "
                "invalid location='invalid_location'."
            )
        elif param == "baddim":
            match_msg = (
                "mesh data variable 'extra_data' does not have the node mesh "
                "dimension 'node', in its dimensions."
            )
        else:
            raise ValueError(f"unexpected param: {param}")

        with PARSE_UGRID_ON_LOAD.context():
            with pytest.raises(ValueError, match=match_msg):
                iris.load(failnc)
