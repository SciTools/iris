# (C) British Crown Copyright 2010 - 2012, Met Office
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


# Import Iris tests first so that some things can be initialised before importing anything else.
import iris.tests as tests

import glob
import os

import matplotlib.pyplot as plt

import iris.examples
import iris.plot as iplt


class TestIrisExamples(tests.GraphicsTest):
    """
    Test the iris example code.

    Tests are attached to the class at import time.
    
    """
    @staticmethod
    def create_test_fn(module):
        """Returns a function which can be attached to this Class."""
        def do_check(self):
            # monkey patch the plt.show function to call self.check_graphic instead
            orig_show = plt.show
            try:
                plt.show = iplt.show = self.check_graphic
                module.main()
            finally:
                plt.show = iplt.show = orig_show
        return do_check


# list all the .py files in iris.examples (__init__ ignored later on)
examples = glob.glob(os.path.join(os.path.dirname(iris.examples.__file__), '*.py'))


for example in examples:
    name = os.path.basename(example)[:-3]
    if name == '__init__':
        continue
    
    iris_mod = __import__('iris.examples.%s' % name)
    mod = getattr(iris_mod.examples, name)
    fn = TestIrisExamples.create_test_fn(mod)
    fn.__name__ = 'test_%s' % name
    setattr(TestIrisExamples, fn.__name__, fn)
    

if __name__ == '__main__':
    tests.main()
