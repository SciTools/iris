# (C) British Crown Copyright 2010 - 2013, Met Office
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


import contextlib
import os.path

import iris
import iris.tests as tests


class PPTest(object):
    """
    A mixin class to provide PP-specific utilities to subclasses of tests.IrisTest.

    """
    @contextlib.contextmanager
    def cube_save_test(self, reference_txt_path, reference_cubes=None, reference_pp_path=None, **kwargs):
        """
        A context manager for testing the saving of Cubes to PP files.

        Args:
        
        * reference_txt_path:
            The path of the file containing the textual PP reference data.

        Kwargs:

        * reference_cubes:
            The cube(s) from which the textual PP reference can be re-built if necessary.
        * reference_pp_path:
            The location of a PP file from which the textual PP reference can be re-built if necessary.
            NB. The "reference_cubes" argument takes precedence over this argument.

        The return value from the context manager is the name of a temporary file
        into which the PP data to be tested should be saved.

        Example::
            with self.cube_save_test(reference_txt_path, reference_cubes=cubes) as temp_pp_path:
                iris.save(cubes, temp_pp_path)

        """
        # Watch out for a missing reference text file
        if not os.path.isfile(reference_txt_path):
            tests.logger.warning('Creating result file: %s', reference_txt_path)
            if reference_cubes:
                temp_pp_path = iris.util.create_temp_filename(".pp")
                try:
                    iris.save(reference_cubes, temp_pp_path, **kwargs)
                    self._create_reference_txt(reference_txt_path, temp_pp_path)
                finally:
                    os.remove(temp_pp_path)
            elif reference_pp_path:
                self._create_reference_txt(reference_txt_path, reference_pp_path)
            else:
                raise ValueError('Missing all of reference txt file, cubes, and PP path.')

        temp_pp_path = iris.util.create_temp_filename(".pp")
        try:
            # This value is returned to the target of the "with" statement's "as" clause.
            yield temp_pp_path

            # Load deferred data for all of the fields (but don't do anything with it)
            pp_fields = list(iris.fileformats.pp.load(temp_pp_path))
            for pp_field in pp_fields:
                pp_field.data
                
            reference = ''.join(open(reference_txt_path, 'r'))
            self._assert_str_same(reference + '\n', str(pp_fields) + '\n',
                                    reference_txt_path, type_comparison_name='PP files')
        finally:
            os.remove(temp_pp_path)

    def _create_reference_txt(self, txt_path, pp_path):
        # Load the reference data
        pp_fields = list(iris.fileformats.pp.load(pp_path))
        for pp_field in pp_fields:
            pp_field.data

        # Clear any header words we don't use
        unused = ('lbexp', 'lbegin', 'lbnrec', 'lbproj', 'lbtyp')
        for pp_field in pp_fields:
            for word_name in unused:
                setattr(pp_field, word_name, 0)
            lbuser = list(pp_field.lbuser)
            lbuser[1] = 0
            pp_field.lbuser = tuple(lbuser)

        # Save the textual representation of the PP fields
        with open(txt_path, 'w') as txt_file:
            txt_file.writelines(str(pp_fields))
