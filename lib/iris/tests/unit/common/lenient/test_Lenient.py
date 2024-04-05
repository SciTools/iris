# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.common.lenient.Lenient`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest.mock import sentinel

from iris.common.lenient import _LENIENT, Lenient


class Test___init__(tests.IrisTest):
    def test_default(self):
        lenient = Lenient()
        expected = dict(maths=True)
        self.assertEqual(expected, lenient.__dict__)

    def test_kwargs(self):
        lenient = Lenient(maths=False)
        expected = dict(maths=False)
        self.assertEqual(expected, lenient.__dict__)

    def test_kwargs_invalid(self):
        emsg = "Invalid .* option, got 'merge'."
        with self.assertRaisesRegex(KeyError, emsg):
            _ = Lenient(merge=True)


class Test___contains__(tests.IrisTest):
    def setUp(self):
        self.lenient = Lenient()

    def test_in(self):
        self.assertIn("maths", self.lenient)

    def test_not_in(self):
        self.assertNotIn("concatenate", self.lenient)


class Test___getitem__(tests.IrisTest):
    def setUp(self):
        self.lenient = Lenient()

    def test_in(self):
        self.assertTrue(self.lenient["maths"])

    def test_not_in(self):
        emsg = "Invalid .* option, got 'MATHS'."
        with self.assertRaisesRegex(KeyError, emsg):
            _ = self.lenient["MATHS"]


class Test___repr__(tests.IrisTest):
    def setUp(self):
        self.lenient = Lenient()

    def test(self):
        expected = "Lenient(maths=True)"
        self.assertEqual(expected, repr(self.lenient))


class Test___setitem__(tests.IrisTest):
    def setUp(self):
        self.lenient = Lenient()

    def test_key_invalid(self):
        emsg = "Invalid .* option, got 'MATHS."
        with self.assertRaisesRegex(KeyError, emsg):
            self.lenient["MATHS"] = False

    def test_maths_value_invalid(self):
        value = sentinel.value
        emsg = f"Invalid .* option 'maths' value, got {value!r}."
        with self.assertRaisesRegex(ValueError, emsg):
            self.lenient["maths"] = value

    def test_maths_disable__lenient_enable_true(self):
        self.assertTrue(_LENIENT.enable)
        self.lenient["maths"] = False
        self.assertFalse(self.lenient.__dict__["maths"])
        self.assertFalse(_LENIENT.enable)

    def test_maths_disable__lenient_enable_false(self):
        _LENIENT.__dict__["enable"] = False
        self.assertFalse(_LENIENT.enable)
        self.lenient["maths"] = False
        self.assertFalse(self.lenient.__dict__["maths"])
        self.assertFalse(_LENIENT.enable)

    def test_maths_enable__lenient_enable_true(self):
        self.assertTrue(_LENIENT.enable)
        self.lenient["maths"] = True
        self.assertTrue(self.lenient.__dict__["maths"])
        self.assertTrue(_LENIENT.enable)

    def test_maths_enable__lenient_enable_false(self):
        _LENIENT.__dict__["enable"] = False
        self.assertFalse(_LENIENT.enable)
        self.lenient["maths"] = True
        self.assertTrue(self.lenient.__dict__["maths"])
        self.assertTrue(_LENIENT.enable)


class Test_context(tests.IrisTest):
    def setUp(self):
        self.lenient = Lenient()

    def test_nop(self):
        self.assertTrue(self.lenient["maths"])

        with self.lenient.context():
            self.assertTrue(self.lenient["maths"])

        self.assertTrue(self.lenient["maths"])

    def test_maths_disable__lenient_true(self):
        # synchronised
        self.assertTrue(_LENIENT.enable)
        self.assertTrue(self.lenient["maths"])

        with self.lenient.context(maths=False):
            # still synchronised
            self.assertFalse(_LENIENT.enable)
            self.assertFalse(self.lenient["maths"])

        # still synchronised
        self.assertTrue(_LENIENT.enable)
        self.assertTrue(self.lenient["maths"])

    def test_maths_disable__lenient_false(self):
        # not synchronised
        _LENIENT.__dict__["enable"] = False
        self.assertFalse(_LENIENT.enable)
        self.assertTrue(self.lenient["maths"])

        with self.lenient.context(maths=False):
            # now synchronised
            self.assertFalse(_LENIENT.enable)
            self.assertFalse(self.lenient["maths"])

        # still synchronised
        self.assertTrue(_LENIENT.enable)
        self.assertTrue(self.lenient["maths"])

    def test_maths_enable__lenient_true(self):
        # not synchronised
        self.assertTrue(_LENIENT.enable)
        self.lenient.__dict__["maths"] = False
        self.assertFalse(self.lenient["maths"])

        with self.lenient.context(maths=True):
            # now synchronised
            self.assertTrue(_LENIENT.enable)
            self.assertTrue(self.lenient["maths"])

        # still synchronised
        self.assertFalse(_LENIENT.enable)
        self.assertFalse(self.lenient["maths"])

    def test_maths_enable__lenient_false(self):
        # synchronised
        _LENIENT.__dict__["enable"] = False
        self.assertFalse(_LENIENT.enable)
        self.lenient.__dict__["maths"] = False
        self.assertFalse(self.lenient["maths"])

        with self.lenient.context(maths=True):
            # still synchronised
            self.assertTrue(_LENIENT.enable)
            self.assertTrue(self.lenient["maths"])

        # still synchronised
        self.assertFalse(_LENIENT.enable)
        self.assertFalse(self.lenient["maths"])


if __name__ == "__main__":
    tests.main()
