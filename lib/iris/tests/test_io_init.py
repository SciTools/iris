# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test the io/__init__.py module."""

from io import BytesIO
from pathlib import Path

import pytest

import iris.fileformats as iff
import iris.io
from iris.tests import _shared_utils


class TestDecodeUri:
    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            (
                "/data/local/someDir/PP/COLPEX/COLPEX_16a_pj001.pp",
                ("file", "/data/local/someDir/PP/COLPEX/COLPEX_16a_pj001.pp", None),
            ),
            (
                r"C:\data\local\someDir\PP\COLPEX\COLPEX_16a_pj001.pp",
                ("file", r"C:\data\local\someDir\PP\COLPEX\COLPEX_16a_pj001.pp", None),
            ),
            (
                "file:///data/local/someDir/PP/COLPEX/COLPEX_16a_pj001.pp",
                ("file", "///data/local/someDir/PP/COLPEX/COLPEX_16a_pj001.pp", None),
            ),
            (
                "https://www.somehost.com:8080/resource/thing.grib",
                ("https", "//www.somehost.com:8080/resource/thing.grib", None),
            ),
            (
                "file:////data/users/joe.bloggs/air_pressure#mode=nczarr,file",
                (
                    "file",
                    "////data/users/joe.bloggs/air_pressure",
                    "mode=nczarr,file",
                ),
            ),
            (
                "file:////data/users/joe.bloggs/air_pressure#mode=file",
                (
                    "file",
                    "////data/users/joe.bloggs/air_pressure",
                    "mode=file",
                ),
            ),
            (
                "/data/local/someDir/2013-11-25T13:49:17.632797",
                ("file", "/data/local/someDir/2013-11-25T13:49:17.632797", None),
            ),
            (
                "/data/local/someDir/air_pressure.zarr",
                ("file", "/data/local/someDir/air_pressure.zarr", None),
            ),
        ],
    )
    def test_decode_uri__str(self, uri, expected):
        assert iris.io.decode_uri(uri) == expected

    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            (
                "/data/local/someDir/PP/COLPEX/COLPEX_16a_pj001.pp",
                ("file", "/data/local/someDir/PP/COLPEX/COLPEX_16a_pj001.pp", None),
            ),
            (
                r"C:\data\local\someDir\PP\COLPEX\COLPEX_16a_pj001.pp",
                ("file", r"C:\data\local\someDir\PP\COLPEX\COLPEX_16a_pj001.pp", None),
            ),
            (
                "/data/local/someDir/2013-11-25T13:49:17.632797",
                ("file", "/data/local/someDir/2013-11-25T13:49:17.632797", None),
            ),
        ],
    )
    def test_decode_uri__path(self, uri, expected):
        assert iris.io.decode_uri(Path(uri)) == expected


class TestFileFormatPicker:
    def test_known_formats(self, request):
        _shared_utils.assert_string(
            request,
            str(iff.FORMAT_AGENT),
            _shared_utils.get_result_path(("file_load", "known_loaders.txt")),
        )

    @_shared_utils.skip_data
    @pytest.mark.parametrize(
        # ways to test the format picker = list of (format-name, file-spec)
        ("expected_format_name", "file_spec"),
        [
            (
                "NetCDF",
                ["NetCDF", "global", "xyt", "SMALL_total_column_co2.nc"],
            ),
            (
                "NetCDF 64 bit offset format",
                ["NetCDF", "global", "xyt", "SMALL_total_column_co2.nc.k2"],
            ),
            (
                "NetCDF_v4",
                ["NetCDF", "global", "xyt", "SMALL_total_column_co2.nc4.k3"],
            ),
            (
                "NetCDF_v4",
                ["NetCDF", "global", "xyt", "SMALL_total_column_co2.nc4.k4"],
            ),
            ("UM Fieldsfile (FF) post v5.2", ["FF", "n48_multi_field"]),
            (
                "GRIB",
                ["GRIB", "grib1_second_order_packing", "GRIB_00008_FRANX01"],
            ),
            ("GRIB", ["GRIB", "jpeg2000", "file.grib2"]),
            ("UM Post Processing file (PP)", ["PP", "simple_pp", "global.pp"]),
            (
                "UM Post Processing file (PP) little-endian",
                ["PP", "little_endian", "qrparm.orog.pp"],
            ),
            (
                "UM Fieldsfile (FF) ancillary",
                ["FF", "ancillary_fixed_length_header"],
            ),
            #            ('BUFR',
            #                ['BUFR', 'mss', 'BUFR_Samples',
            #                 'JUPV78_EGRR_121200_00002501']),
            (
                "NIMROD",
                [
                    "NIMROD",
                    "uk2km",
                    "WO0000000003452",
                    "201007020900_u1096_ng_ey00_visibility0180_screen_2km",
                ],
            ),
        ],
        #            ('NAME',
        #                ['NAME', '20100509_18Z_variablesource_12Z_VAAC',
        #                 'Fields_grid1_201005110000.txt']),
    )
    def test_format_picker(self, expected_format_name, file_spec):
        # test that each filespec is identified as the expected format
        test_path = _shared_utils.get_data_path(file_spec)
        with open(test_path, "rb") as test_file:
            a = iff.FORMAT_AGENT.get_spec(test_path, test_file)
            assert a.name == expected_format_name

    @pytest.mark.parametrize("header_length", [21, 80, 41, 42])
    def test_format_picker_nodata(self, header_length):
        # The following is to replace the above at some point as no real files
        # are required.
        # (Used binascii.unhexlify() to convert from hex to binary)

        # Packaged grib, magic number offset by set length, this length is
        # specific to WMO bulletin headers
        binary_string = header_length * b"\x00" + b"GRIB" + b"\x00" * 100
        with BytesIO(b"rw") as bh:
            bh.write(binary_string)
            bh.name = "fake_file_handle"
            a = iff.FORMAT_AGENT.get_spec(bh.name, bh)
        assert a.name == "GRIB"

    def test_open_dap(self):
        # tests that *ANY* http or https URL is seen as an OPeNDAP service.
        # This may need to change in the future if other protocols are
        # supported.
        DAP_URI = "https://geoport.whoi.edu/thredds/dodsC/bathy/gom15"
        a = iff.FORMAT_AGENT.get_spec(DAP_URI, None)
        assert a.name == "NetCDF OPeNDAP"

    def test_nczarr_url_without_zarr_suffix(self):
        uri = "file:////data/users/joe.bloggs/air_pressure#mode=nczarr,file"
        a = iff.FORMAT_AGENT.get_spec(uri, None)
        assert a.name == "NcZarr"
