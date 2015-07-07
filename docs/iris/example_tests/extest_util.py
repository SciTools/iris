# (C) British Crown Copyright 2010 - 2015, Met Office
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
Provides context managers which are fundamental to the ability
to run the example tests.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import contextlib
import os.path
import sys

import matplotlib.pyplot as plt

import iris.plot as iplt
import iris.quickplot as qplt
from iris.tests import _DEFAULT_IMAGE_TOLERANCE


EXAMPLE_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 'example_code')
EXAMPLE_DIRECTORIES = [os.path.join(EXAMPLE_DIRECTORY, the_dir)
                       for the_dir in os.listdir(EXAMPLE_DIRECTORY)]


@contextlib.contextmanager
def add_examples_to_path():
    """
    Creates a context manager which can be used to add the iris examples
    to the PYTHONPATH. The examples are only importable throughout the lifetime
    of this context manager.

    """
    orig_sys_path = sys.path
    sys.path = sys.path[:]
    sys.path += EXAMPLE_DIRECTORIES
    yield
    sys.path = orig_sys_path


@contextlib.contextmanager
def show_replaced_by_check_graphic(test_case, tol=_DEFAULT_IMAGE_TOLERANCE):
    """
    Creates a context manager which can be used to replace the functionality
    of matplotlib.pyplot.show with a function which calls the check_graphic
    method on the given test_case (iris.tests.IrisTest.check_graphic).

    """
    def replacement_show():
        # form a closure on test_case and tolerance
        test_case.check_graphic(tol=tol)

    orig_show = plt.show
    plt.show = iplt.show = qplt.show = replacement_show
    yield
    plt.show = iplt.show = qplt.show = orig_show
