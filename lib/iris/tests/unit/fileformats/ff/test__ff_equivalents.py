# (C) British Crown Copyright 2013 - 2016, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Unit tests for :class:`iris.fileformat.ff`.

The real functional tests now all target :class:`iris.fileformat._ff`.
This is just here to check that the one is a clean interface to the other.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import warnings

with warnings.catch_warnings():
    # Also suppress invalid units warnings from the Iris loader code.
    warnings.filterwarnings("ignore")
    import iris.fileformats.ff as ff


# A global used to catch call information in our mock constructor calls.
# Used because, as patches, those calls don't receive the test context
# (i.e. the call gets a different 'self').
constructor_calls_data = []


class Mixin_ConstructorTest(object):
    # A test mixin for the 'ff' wrapper subclasses creation calls.
    # For each one, patch out the __init__ of the target class in '_ff', and
    # check that the ff class constructor correctly calls this with both
    # unnamed args and keywords.

    # Name of the target class (in 'ff'): **inheritor defines**.
    target_class_name = 'name_of_a_class_in_ff'

    # Constructor function replacement: **inheritor defines**.
    #    @staticmethod
    #    def dummy_constructor_call(self):
    #        pass
    # N.B. do *not* actually define one here, as oddly this affects the call
    # properties of the overriding methods ?

    def constructor_setup(self):
        """
        A 'setUp' helper method.

        Only defined as a separate method because tests can't inherit an
        *actual* 'setUp' method, for some reason.

        """
        # Reset the constructor calls data log.
        global constructor_calls_data
        constructor_calls_data = []

        # Define an import string for the __init__ of the corresponding class
        # in '_ff', where the real implementation is.
        tgt_fmt = 'iris.fileformats._ff.{}.__init__'
        patch_target_name = tgt_fmt.format(self.target_class_name)

        # Patch the implementation class with the replacement 'dummy'
        # constructor call, to record usage.
        self.patch(patch_target_name, self.dummy_constructor_call)

    def check_call(self, target_args, target_keys, expected_result):
        """
        Test instantiation of the target class.

        Call with given args and kwargs, and check that the parent class
        (in _ff) got the expected call.

        """
        # Invoke the main target while blocking warnings.
        # The parent class is already patched, by the setUp operation.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            test_class = getattr(ff, self.target_class_name)
            result = test_class(*target_args, **target_keys)

        # Check we returned the right type of thing.
        self.assertIsInstance(result, test_class)

        # Check the call args that the parent constructor call received.
        # Note: as it is a list, this also ensures we only got *one* call.
        global constructor_calls_data
        self.assertEqual(constructor_calls_data, [expected_result])


class Mixin_Grid_Tests(object):
    # A mixin with tests for the 'Grid' derived classes.

    @staticmethod
    def dummy_constructor_call(self, column_dependent_constants,
                               row_dependent_constants,
                               real_constants, horiz_grid_type):
        # A replacement for the 'real' constructor call in the parent class.
        # Used to check for correct args and kwargs in the call.
        # Just record the call arguments in a global, for testing.
        global constructor_calls_data
        # It's global because in use, we do not have the right 'self' here.
        # Note: as we append, it contains a full call history.
        constructor_calls_data += [(column_dependent_constants,
                                    row_dependent_constants,
                                    real_constants,
                                    horiz_grid_type)]
        return self

    def test__basic(self):
        # Call with four unnamed args.
        self.check_call(['cdc', 'rdc', 'rc', 'ht'], {},
                        expected_result=('cdc', 'rdc', 'rc', 'ht'))

    def test__all_named_keys(self):
        # Call with four named keys.
        kwargs = {'column_dependent_constants': 1,
                  'row_dependent_constants': 2,
                  'real_constants': 3,
                  'horiz_grid_type': 4}
        self.check_call([], kwargs, expected_result=(1, 2, 3, 4))

    def test__badargs(self):
        # Make sure we can catch a bad number of arguments.
        with self.assertRaises(TypeError):
            self.check_call([], {})

    def test__badkey(self):
        # Make sure we can catch an unrecognised keyword.
        with self.assertRaises(TypeError):
            self.check_call([], {'_bad_key': 1})


class Test_Grid(Mixin_ConstructorTest, Mixin_Grid_Tests, tests.IrisTest):
    target_class_name = 'Grid'

    def setUp(self):
        self.constructor_setup()


class Test_ArakawaC(Mixin_ConstructorTest, Mixin_Grid_Tests, tests.IrisTest):
    target_class_name = 'ArakawaC'

    def setUp(self):
        self.constructor_setup()


class Test_ENDGame(Mixin_ConstructorTest, Mixin_Grid_Tests, tests.IrisTest):
    target_class_name = 'ENDGame'

    def setUp(self):
        self.constructor_setup()


class Test_NewDynamics(Mixin_ConstructorTest, Mixin_Grid_Tests,
                       tests.IrisTest):
    target_class_name = 'NewDynamics'

    def setUp(self):
        self.constructor_setup()


class Test_FFHeader(Mixin_ConstructorTest, tests.IrisTest):
    target_class_name = 'FFHeader'

    @staticmethod
    def dummy_constructor_call(self, filename, word_depth=16):
        # A replacement for the 'real' constructor call in the parent class.
        # Used to check for correct args and kwargs in the call.
        # Just record the call arguments in a global, for testing.
        global constructor_calls_data
        # It's global because in use, we do not have the right 'self' here.
        # Note: as we append, it contains a full call history.
        constructor_calls_data += [(filename, word_depth)]
        return self

    def setUp(self):
        self.constructor_setup()

    def test__basic(self):
        # Call with just a filename.
        self.check_call(['filename'], {},
                        expected_result=('filename', 16))

    def test__word_depth(self):
        # Call with a word-depth.
        self.check_call(['filename'], {'word_depth': 4},
                        expected_result=('filename', 4))

    def test__badargs(self):
        # Make sure we can catch a bad number of arguments.
        with self.assertRaises(TypeError):
            self.check_call([], {})

    def test__badkey(self):
        # Make sure we can catch an unrecognised keyword.
        with self.assertRaises(TypeError):
            self.check_call([], {'_bad_key': 1})


class Test_FF2PP(Mixin_ConstructorTest, tests.IrisTest):
    target_class_name = 'FF2PP'

    @staticmethod
    def dummy_constructor_call(self, filename, read_data=False,
                               word_depth=16):
        # A replacement for the 'real' constructor call in the parent class.
        # Used to check for correct args and kwargs in the call.
        # Just record the call arguments in a global, for testing.
        global constructor_calls_data
        # It's global because in use, we do not have the right 'self' here.
        # Note: as we append, it contains a full call history.
        constructor_calls_data += [(filename, read_data, word_depth)]
        return self

    def setUp(self):
        self.constructor_setup()

    def test__basic(self):
        # Call with just a filename.
        self.check_call(['filename'], {},
                        expected_result=('filename', False, 16))

    def test__read_data(self):
        # Call with a word-depth.
        self.check_call(['filename', True], {},
                        expected_result=('filename', True, 16))

    def test__word_depth(self):
        # Call with a word-depth keyword.
        self.check_call(['filename'], {'word_depth': 4},
                        expected_result=('filename', False, 4))

    def test__badargs(self):
        # Make sure we can catch a bad number of arguments.
        with self.assertRaises(TypeError):
            self.check_call([], {})

    def test__badkey(self):
        # Make sure we can catch an unrecognised keyword.
        with self.assertRaises(TypeError):
            self.check_call([], {'_bad_key': 1})


if __name__ == "__main__":
    tests.main()
