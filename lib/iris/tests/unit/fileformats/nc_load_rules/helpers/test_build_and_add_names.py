# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

from iris.fileformats._nc_load_rules.helpers import build_and_add_names
from iris.loading import LOAD_PROBLEMS

from .test_build_cube_metadata import _make_engine


class TestCubeName(tests.IrisTest):
    def setUp(self):
        LOAD_PROBLEMS.clear()

    def check_cube_names(self, inputs, expected):
        # Inputs - attributes on the fake CF Variable.
        standard_name, long_name = inputs
        # Expected - The expected cube attributes.
        exp_standard_name, exp_long_name = expected

        engine = _make_engine(standard_name=standard_name, long_name=long_name)
        build_and_add_names(engine)

        # Check the cube's standard name and long name are as expected.
        self.assertEqual(engine.cube.standard_name, exp_standard_name)
        self.assertEqual(engine.cube.long_name, exp_long_name)

    def check_load_problems(self, invalid_standard_name=None):
        if invalid_standard_name is None:
            self.assertEqual(LOAD_PROBLEMS, {})
        else:
            load_problems = list(LOAD_PROBLEMS.values())[-1]
            load_problem = load_problems[-1]
            self.assertEqual(
                load_problem.loaded, {"standard_name": invalid_standard_name}
            )

    def test_standard_name_none_long_name_none(self):
        inputs = (None, None)
        expected = (None, None)
        self.check_cube_names(inputs, expected)
        self.check_load_problems()

    def test_standard_name_none_long_name_set(self):
        inputs = (None, "ice_thickness_long_name")
        expected = (None, "ice_thickness_long_name")
        self.check_cube_names(inputs, expected)
        self.check_load_problems()

    def test_standard_name_valid_long_name_none(self):
        inputs = ("sea_ice_thickness", None)
        expected = ("sea_ice_thickness", None)
        self.check_cube_names(inputs, expected)
        self.check_load_problems()

    def test_standard_name_valid_long_name_set(self):
        inputs = ("sea_ice_thickness", "ice_thickness_long_name")
        expected = ("sea_ice_thickness", "ice_thickness_long_name")
        self.check_cube_names(inputs, expected)
        self.check_load_problems()

    def test_standard_name_invalid_long_name_none(self):
        inputs = ("not_a_standard_name", None)
        expected = (
            None,
            "not_a_standard_name",
        )
        self.check_cube_names(inputs, expected)
        self.check_load_problems("not_a_standard_name")

    def test_standard_name_invalid_long_name_set(self):
        inputs = ("not_a_standard_name", "ice_thickness_long_name")
        expected = (None, "ice_thickness_long_name")
        self.check_cube_names(inputs, expected)
        self.check_load_problems("not_a_standard_name")


if __name__ == "__main__":
    tests.main()
