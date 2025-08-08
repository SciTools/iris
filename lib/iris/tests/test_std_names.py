# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

from iris.std_names import CF_STANDARD_NAMES_TABLE_VERSION, STD_NAMES


class TestStandardNames:
    """standard_names.py is a machine generated file which contains a single dictionary
    called STD_NAMES.
    """

    longMessage = True

    def test_standard_names_table(self):
        # Check we have a dict
        assert isinstance(STD_NAMES, dict)

        keyset = set(STD_NAMES)

        # Check for some known standard names
        valid_nameset = set(["air_density", "northward_wind", "wind_speed"])
        assert valid_nameset.issubset(keyset), (
            "Known standard name missing from STD_NAMES"
        )

        # Check for some invalid standard names
        invalid_nameset = set(
            [
                "invalid_air_density",
                "invalid_northward_wind",
                "invalid_wind_speed",
                "stratiform_snowfall_rate",
            ]
        )
        assert invalid_nameset - keyset == invalid_nameset, (
            "\nInvalid standard name(s) present in STD_NAMES"
        )

    def test_standard_names_version(self):
        # Check we have a dict
        assert isinstance(CF_STANDARD_NAMES_TABLE_VERSION, int)
        # Check the value is roughly sensible.
        assert 70 < CF_STANDARD_NAMES_TABLE_VERSION < 999
