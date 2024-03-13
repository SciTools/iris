from pathlib import Path

import pytest

from iris import tests
from iris.tests._shared_utils import result_path

under_test = tests.IrisTest.result_path


class TestResultPath:
    """Tests to fully exercise iris.tests.IrisTest.result_path"""

    def test_basic(self):
        from iris.tests.test_cf import TestLoad

        result = under_test(TestLoad())
        assert result == (
            f"/tmp/persistent/repos/iris/lib/iris/tests/results/cf/TestLoad/basic"
        )

    def test_basename(self):
        from iris.tests.test_cf import TestLoad

        result = under_test(TestLoad(), basename="foo")
        assert result == (
            f"/tmp/persistent/repos/iris/lib/iris/tests/results/cf/TestLoad/foo"
        )

    def test_ext(self):
        result = under_test(tests.IrisTest(), ext=".nc")
        assert Path(result).suffix == ".nc"

    def test_ext_no_dot(self):
        result = under_test(tests.IrisTest(), ext="nc")
        assert Path(result).suffix == ".nc"

    def test_ext_none(self):
        result = under_test(tests.IrisTest())
        assert Path(result).suffix == ""


class TestTmp(tests.IrisTest):
    def test_tmp(self):
        print(result_path("test_tmp", "foo"))
        print(self.result_path())


def test_tmp():
    print(result_path())


class TestMy:
    @pytest.fixture(autouse=True)
    def graphics_setup(self, check_graphic_caller):
        self.graphics_tester = check_graphic_caller

    def test_my(self):
        self.graphics_tester()
        self.graphics_tester()

    def test_my2(self):
        self.graphics_tester()
