# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests  # isort:skip

from iris.std_names import STD_NAMES


class TestStandardNames(tests.IrisTest):
    """
    standard_names.py is a machine generated file which contains a single dictionary
    called STD_NAMES
    """

    longMessage = True

    def test_standard_names(self):
        # Check we have a dict
        self.assertIsInstance(STD_NAMES, dict)

        keyset = set(STD_NAMES)

        # Check for some known standard names
        valid_nameset = set(["air_density", "northward_wind", "wind_speed"])
        self.assertTrue(
            valid_nameset.issubset(keyset),
            "Known standard name missing from STD_NAMES",
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
        self.assertSetEqual(
            invalid_nameset - keyset,
            invalid_nameset,
            "\nInvalid standard name(s) present in STD_NAMES",
        )


if __name__ == "__main__":
    tests.main()
