# (C) British Crown Copyright 2010 - 2016, Met Office
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
Check the the latest "whatsnew" contributions files have usable names.

The files in "./docs...whatsnew/contributions_<xx.xx>/" should have filenames
with a particular structure, which encodes a summary name.
These names are interpreted by "./docs...whatsnew/aggregate_directory.py".
This test just ensures that all those files have names which that process
can accept.

.. note:
    This only works within a developer installation: In a 'normal' install the
    location of the docs sources is not known.
    In a Travis installation, this test silently passes and the .travis.yml
    invokes the checking command directly.

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first.
import iris.tests as tests

import os
import os.path
import subprocess
import sys

import iris


class TestWhatsnewContribs(tests.IrisTest):
    def test_check_contributions(self):
        # Get dirpath of overall iris installation.
        # Note: assume iris at "<install>/lib/iris".
        iris_module_dirpath = os.path.dirname(iris.__file__)
        iris_dirs = iris_module_dirpath.split(os.sep)
        install_dirpath = os.sep.join(iris_dirs[:-2])

        # Construct path to docs 'whatsnew' directory.
        # Note: assume docs at "<install>/docs".
        whatsnew_dirpath = os.path.join(
            install_dirpath, 'docs', 'iris', 'src', 'whatsnew')

        # Quietly ignore if the directory does not exist: It is only there in
        # in a developer installation, not a normal install.
        # Travis bypasses this problem by running the test directly.
        if os.path.exists(whatsnew_dirpath):
            # Run a 'check contributions' command in that directory.
            cmd = [sys.executable, 'aggregate_directory.py', '--checkonly',
                   '--quiet']
            subprocess.check_call(cmd, cwd=whatsnew_dirpath)


if __name__ == "__main__":
    tests.main()
