# (C) British Crown Copyright 2010 - 2014, Met Office
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
Provides testing capabilities for installed copies of Iris.

"""

from __future__ import (absolute_import, division, print_function)

import argparse

from ._runner import TestRunner


parser = argparse.ArgumentParser('iris.tests',
                                 description=TestRunner.description)
for long_opt, short_opt, help_text in TestRunner.user_options:
    long_opt = long_opt.strip('=')
    if long_opt in TestRunner.boolean_options:
        parser.add_argument('--' + long_opt, '-' + short_opt,
                            action='store_true', help=help_text)
    else:
        parser.add_argument('--' + long_opt, '-' + short_opt,
                            help=help_text)
args = parser.parse_args()

runner = TestRunner()

runner.initialize_options()
for long_opt, short_opt, help_text in TestRunner.user_options:
    arg = long_opt.replace('-', '_').strip('=')
    setattr(runner, arg, getattr(args, arg))
runner.finalize_options()

runner.run()
