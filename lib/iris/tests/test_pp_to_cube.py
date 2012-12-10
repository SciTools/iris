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

import warnings

import os

import iris
import iris.fileformats.pp
import iris.io
import iris.util
import iris.tests.stock


@iris.tests.skip_data
class TestPPLoadCustom(tests.IrisTest):
    def setUp(self):
        self.filename = tests.get_data_path(('PP', 'aPPglob1', 'global.pp'))
        iris.fileformats.pp._ensure_load_rules_loaded()
        self.load_rules = iris.fileformats.pp._load_rules

    def test_lbtim_2(self):
        subcubes = iris.cube.CubeList()
        template = iris.fileformats.pp.load(self.filename).next()
        for delta in range(10):
            field = template.copy()
            field.lbtim = 2
            field.lbdat += delta
            rules_result = self.load_rules.result(field)
            subcubes.append(rules_result.cube)
        cube = subcubes.merge()[0]
        self.assertCML(cube, ('pp_rules', 'lbtim_2.cml'))


class TestReferences(tests.IrisTest):
    def setUp(self):
        target = iris.tests.stock.simple_2d()
        target.data = target.data.astype('f4')
        self.target = target
        self.ref = target.copy()

    def test_regrid_missing_coord(self):
        # If the target cube is missing one of the source dimension
        # coords, ensure the re-grid fails nicely - i.e. returns None.
        self.target.remove_coord('bar')
        new_ref = iris.fileformats.rules._ensure_aligned({}, self.ref,
                                                         self.target)
        self.assertIsNone(new_ref)

    def test_regrid_codimension(self):
        # If the target cube has two of the source dimension coords
        # sharing the same dimension (e.g. a trajectory) then ensure
        # the re-grid fails nicely - i.e. returns None.
        self.target.remove_coord('foo')
        new_foo = self.target.coord('bar').copy()
        new_foo.rename('foo')
        self.target.add_aux_coord(new_foo, 0)
        new_ref = iris.fileformats.rules._ensure_aligned({}, self.ref,
                                                         self.target)
        self.assertIsNone(new_ref)

    def test_regrid_identity(self):
        new_ref = iris.fileformats.rules._ensure_aligned({}, self.ref,
                                                         self.target)
        # Bounds don't make it through the re-grid process
        self.ref.coord('bar').bounds = None
        self.ref.coord('foo').bounds = None
        self.assertEqual(new_ref, self.ref)


@iris.tests.skip_data
class TestPPLoading(tests.IrisTest):
    def test_simple(self):
        cube = iris.tests.stock.simple_pp()
        self.assertCML(cube, ('cube_io', 'pp', 'load', 'global.cml'))


@iris.tests.skip_data
class TestPPLoadRules(tests.IrisTest):
    def test_pp_load_rules(self):
        # Test PP loading and rule evaluation.

        cube = iris.tests.stock.simple_pp()
        self.assertCML(cube, ('pp_rules', 'global.cml'))

        data_path = tests.get_data_path(('PP', 'rotated_uk', 'rotated_uk.pp'))
        cube = iris.load(data_path)[0]
        self.assertCML(cube, ('pp_rules', 'rotated_uk.cml'))

    def test_lbproc(self):
        data_path = tests.get_data_path(('PP', 'meanMaxMin', '200806081200__qwpb.T24.pp'))
        # Set up standard name and T+24 constraint
        constraint = iris.Constraint('air_temperature', forecast_period=24)
        cubes = iris.load(data_path, constraint)
        cubes = iris.cube.CubeList([cubes[0], cubes[3], cubes[1], cubes[2], cubes[4]]) 
        self.assertCML(cubes, ('pp_rules', 'lbproc_mean_max_min.cml'))

    def test_custom_rules(self):
        # Test custom rule evaluation.
        # Default behaviour
        data_path = tests.get_data_path(('PP', 'aPPglob1', 'global.pp'))
        cube = iris.load_cube(data_path)
        self.assertEqual(cube.standard_name, 'air_temperature')

        # Custom behaviour
        temp_path = iris.util.create_temp_filename()
        f = open(temp_path, 'w')
        f.write('\n'.join((
            'IF',
            'f.lbuser[3] == 16203',
            'THEN',
            'CMAttribute("standard_name", None)', 
            'CMAttribute("long_name", "customised")'))) 
        f.close()
        iris.fileformats.pp.add_load_rules(temp_path)
        cube = iris.load_cube(data_path)
        self.assertEqual(cube.name(), 'customised')
        os.remove(temp_path)
        
        # Back to default
        iris.fileformats.pp.reset_load_rules()
        cube = iris.load_cube(data_path)
        self.assertEqual(cube.standard_name, 'air_temperature')

    def test_cell_methods(self):
        # Test cell methods are created for correct values of lbproc
        orig_file = tests.get_data_path(('PP', 'aPPglob1', 'global.pp'))
        
        # Values that result in cell methods being created
        cell_method_values = {128 : "mean", 4096 : "minimum", 8192 : "maximum"}
        
        # Make test values as list of single bit values and some multiple bit values
        single_bit_values = list(iris.fileformats.pp.LBPROC_PAIRS)
        multiple_bit_values = [(128 + 64, ""), (4096 + 2096, ""), (8192 + 1024, "")]
        test_values = list(single_bit_values) + multiple_bit_values
        
        for value, _ in test_values:
            f = iris.fileformats.pp.load(orig_file).next()
            f.lbproc = value # set value

            # Write out pp file
            temp_filename = iris.util.create_temp_filename(".pp")
            f.save(open(temp_filename, 'wb'))
        
            # Load pp file
            cube = iris.load_cube(temp_filename)
        
            if value in cell_method_values:
                # Check for cell method on cube
                self.assertEqual(cube.cell_methods[0].method, cell_method_values[value])
            else:
                # Check no cell method was created for values other than 128, 4096, 8192
                self.assertEqual(len(cube.cell_methods), 0)
        
            os.remove(temp_filename)   


    def test_process_flags(self):
        # Test that process flags are created for correct values of lbproc
        orig_file = tests.get_data_path(('PP', 'aPPglob1', 'global.pp'))
   
        # Values that result in process flags attribute NOT being created
        omit_process_flags_values = (128, 4096, 8192)
        
        # Test single flag values
        for value, _ in iris.fileformats.pp.LBPROC_PAIRS:
            f = iris.fileformats.pp.load(orig_file).next()
            f.lbproc = value # set value

            # Write out pp file
            temp_filename = iris.util.create_temp_filename(".pp")
            f.save(open(temp_filename, 'wb'))
        
            # Load pp file
            cube = iris.load_cube(temp_filename)

            if value in omit_process_flags_values:
                # Check ukmo__process_flags attribute not created
                self.assertEqual(cube.attributes.get("ukmo__process_flags", None), None)
            else:
                # Check ukmo__process_flags attribute contains correct values
                self.assertIn(iris.fileformats.pp.lbproc_map[value], cube.attributes["ukmo__process_flags"])
        
            os.remove(temp_filename) 

        # Test multiple flag values
        multiple_bit_values = ((128, 64), (4096, 1024), (8192, 1024))
        
        # Maps lbproc value to the process flags that should be created
        multiple_map = {sum(x) : [iris.fileformats.pp.lbproc_map[y] for y in x] for x in multiple_bit_values}
        
        for bit_values in multiple_bit_values:
            f = iris.fileformats.pp.load(orig_file).next()
            f.lbproc = sum(bit_values) # set value

            # Write out pp file
            temp_filename = iris.util.create_temp_filename(".pp")
            f.save(open(temp_filename, 'wb'))
        
            # Load pp file
            cube = iris.load_cube(temp_filename)

            # Check the process flags created
            self.assertEquals(set(cube.attributes["ukmo__process_flags"]), set(multiple_map[sum(bit_values)]), "Mismatch between expected and actual process flags.")

            os.remove(temp_filename)


@iris.tests.skip_data
class TestStdName(tests.IrisTest):
    def test_no_std_name(self):
        fname = tests.get_data_path(['PP', 'simple_pp', 'bad_global.pp'])
        cube = iris.load_cube(fname)
        self.assertCML([cube], ['cube_io', 'pp', 'no_std_name.cml'])
        
        
if __name__ == "__main__":
    tests.main()
