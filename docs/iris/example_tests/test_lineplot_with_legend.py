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

import unittest

# Import updated sys.path for example_code.
import example_code_path
import lineplot_with_legend 
import override_mpl_show


class TestLineplotWithLegend(tests.GraphicsTest):
    """Test the lineplot_with_legend example code."""
    def setUp(self):
        override_mpl_show.init(self)

    def test_lineplot_with_legend(self):
        lineplot_with_legend.main() 


if __name__ == '__main__':
    tests.main()
