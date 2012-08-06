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


# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import os

import iris
import iris.fileformats.pp
import iris.config as config
import iris.fileformats.rules as rules


@iris.tests.skip_data
class TestVerboseLogging(tests.IrisTest):
    def test_verbose_logging(self):
        # check that verbose logging no longer breaks in pp.save()
        # load some data, enable logging, and save a cube to PP.
        data_path = tests.get_data_path(('PP', 'simple_pp', 'global.pp'))
        cube = iris.load_strict(data_path)
        OLD_RULE_LOG_DIR = config.RULE_LOG_DIR
        config.RULE_LOG_DIR = '/var/tmp'
        old_log = rules.log
        rules.log = rules._prepare_rule_logger(verbose=True)
        
        temp_filename1 = iris.util.create_temp_filename(suffix='.pp')
  
        # Test writing to a file handle to test that the logger uses the handle name      
        with open(temp_filename1, "wb") as mysavefile:
            try:
                iris.save(cube, mysavefile)
            finally:
                # Restore old logging config
                config.RULE_LOG_DIR = OLD_RULE_LOG_DIR
                rules.log = old_log
                os.unlink(temp_filename1) 

if __name__ == "__main__":
    tests.main()
