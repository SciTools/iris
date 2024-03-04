# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests  # isort:skip

import os

import iris
import iris.fileformats.pp
import iris.fileformats.pp_load_rules
import iris.fileformats.rules
import iris.io
import iris.tests.stock
import iris.util


@tests.skip_data
class TestPPLoadCustom(tests.IrisTest):
    def setUp(self):
        self.subcubes = iris.cube.CubeList()
        filename = tests.get_data_path(("PP", "aPPglob1", "global.pp"))
        self.template = next(iris.fileformats.pp.load(filename))

    def _field_to_cube(self, field):
        cube, _, _ = iris.fileformats.rules._make_cube(
            field, iris.fileformats.pp_load_rules.convert
        )
        return cube

    def test_lbtim_2(self):
        for delta in range(10):
            field = self.template.copy()
            field.lbtim = 2
            field.lbdat += delta
            cube = self._field_to_cube(field)
            self.subcubes.append(cube)
        cube = self.subcubes.merge()[0]
        self.assertCML(cube, ("pp_load_rules", "lbtim_2.cml"))

    def _ocean_depth(self, bounded=False):
        lbuser = list(self.template.lbuser)
        lbuser[6] = 2
        lbuser[3] = 101
        lbuser = tuple(lbuser)
        for level_and_depth in enumerate([5.0, 15.0, 25.0, 35.0, 45.0]):
            field = self.template.copy()
            field.lbuser = lbuser
            field.lbvc = 2
            field.lbfc = 601
            field.lblev, field.blev = level_and_depth
            if bounded:
                brsvd = list(field.brsvd)
                brsvd[0] = field.blev - 1
                field.brsvd = tuple(brsvd)
                field.brlev = field.blev + 1
            cube = self._field_to_cube(field)
            self.subcubes.append(cube)

    def test_ocean_depth(self):
        self._ocean_depth()
        cube = self.subcubes.merge()[0]
        self.assertCML(cube, ("pp_load_rules", "ocean_depth.cml"))

    def test_ocean_depth_bounded(self):
        self._ocean_depth(bounded=True)
        cube = self.subcubes.merge()[0]
        self.assertCML(cube, ("pp_load_rules", "ocean_depth_bounded.cml"))


class TestReferences(tests.IrisTest):
    def setUp(self):
        target = iris.tests.stock.simple_2d()
        target.data = target.data.astype("f4")
        self.target = target
        self.ref = target.copy()

    def test_regrid_missing_coord(self):
        # If the target cube is missing one of the source dimension
        # coords, ensure the re-grid fails nicely - i.e. returns None.
        self.target.remove_coord("bar")
        new_ref = iris.fileformats.rules._ensure_aligned(
            {}, self.ref, self.target
        )
        self.assertIsNone(new_ref)

    def test_regrid_codimension(self):
        # If the target cube has two of the source dimension coords
        # sharing the same dimension (e.g. a trajectory) then ensure
        # the re-grid fails nicely - i.e. returns None.
        self.target.remove_coord("foo")
        new_foo = self.target.coord("bar").copy()
        new_foo.rename("foo")
        self.target.add_aux_coord(new_foo, 0)
        new_ref = iris.fileformats.rules._ensure_aligned(
            {}, self.ref, self.target
        )
        self.assertIsNone(new_ref)

    def test_regrid_identity(self):
        new_ref = iris.fileformats.rules._ensure_aligned(
            {}, self.ref, self.target
        )
        # Bounds don't make it through the re-grid process
        self.ref.coord("bar").bounds = None
        self.ref.coord("foo").bounds = None
        self.assertEqual(new_ref, self.ref)


@tests.skip_data
class TestPPLoading(tests.IrisTest):
    def test_simple(self):
        cube = iris.tests.stock.simple_pp()
        self.assertCML(cube, ("cube_io", "pp", "load", "global.cml"))


@tests.skip_data
class TestPPLoadRules(tests.IrisTest):
    def test_pp_load_rules(self):
        # Test PP loading and rule evaluation.

        cube = iris.tests.stock.simple_pp()
        self.assertCML(cube, ("pp_load_rules", "global.cml"))

        data_path = tests.get_data_path(("PP", "rotated_uk", "rotated_uk.pp"))
        cube = iris.load(data_path)[0]
        self.assertCML(cube, ("pp_load_rules", "rotated_uk.cml"))

    def test_lbproc(self):
        data_path = tests.get_data_path(
            ("PP", "meanMaxMin", "200806081200__qwpb.T24.pp")
        )
        # Set up standard name and T+24 constraint
        constraint = iris.Constraint("air_temperature", forecast_period=24)
        cubes = iris.load(data_path, constraint)
        cubes = iris.cube.CubeList(
            [cubes[0], cubes[3], cubes[1], cubes[2], cubes[4]]
        )
        self.assertCML(cubes, ("pp_load_rules", "lbproc_mean_max_min.cml"))

    def test_cell_methods(self):
        # Test cell methods are created for correct values of lbproc
        orig_file = tests.get_data_path(("PP", "aPPglob1", "global.pp"))

        # Values that result in cell methods being created
        cell_method_values = {
            64: "mean",
            128: "mean within years",
            4096: "minimum",
            8192: "maximum",
        }

        # Make test values as list of single bit values and some multiple bit values
        single_bit_values = list(iris.fileformats.pp.LBPROC_PAIRS)
        multiple_bit_values = [
            (128 + 32, ""),
            (4096 + 2096, ""),
            (8192 + 1024, ""),
        ]
        test_values = list(single_bit_values) + multiple_bit_values

        for value, _ in test_values:
            f = next(iris.fileformats.pp.load(orig_file))
            f.lbproc = value  # set value

            # Write out pp file
            temp_filename = iris.util.create_temp_filename(".pp")
            with open(temp_filename, "wb") as temp_fh:
                f.save(temp_fh)

            # Load pp file
            cube = iris.load_cube(temp_filename)

            if value in cell_method_values:
                # Check for cell method on cube
                self.assertEqual(
                    cube.cell_methods[0].method, cell_method_values[value]
                )
            else:
                # Check no cell method was created for values other than 128, 4096, 8192
                self.assertEqual(len(cube.cell_methods), 0)

            os.remove(temp_filename)

    def test_process_flags(self):
        # Test that process flags are created for correct values of lbproc
        orig_file = tests.get_data_path(("PP", "aPPglob1", "global.pp"))

        # Values that result in process flags attribute NOT being created
        omit_process_flags_values = (64, 128, 4096, 8192)

        # Test single flag values
        for value, _ in iris.fileformats.pp.LBPROC_PAIRS:
            f = next(iris.fileformats.pp.load(orig_file))
            f.lbproc = value  # set value

            # Write out pp file
            temp_filename = iris.util.create_temp_filename(".pp")
            with open(temp_filename, "wb") as temp_fh:
                f.save(temp_fh)

            # Load pp file
            cube = iris.load_cube(temp_filename)

            if value in omit_process_flags_values:
                # Check ukmo__process_flags attribute not created
                self.assertEqual(
                    cube.attributes.get("ukmo__process_flags", None), None
                )
            else:
                # Check ukmo__process_flags attribute contains correct values
                self.assertIn(
                    iris.fileformats.pp.lbproc_map[value],
                    cube.attributes["ukmo__process_flags"],
                )

            os.remove(temp_filename)

        # Test multiple flag values
        multiple_bit_values = ((128, 32), (4096, 1024), (8192, 1024))

        # Maps lbproc value to the process flags that should be created
        multiple_map = {
            sum(x): [iris.fileformats.pp.lbproc_map[y] for y in x]
            for x in multiple_bit_values
        }

        for bit_values in multiple_bit_values:
            f = next(iris.fileformats.pp.load(orig_file))
            f.lbproc = sum(bit_values)  # set value

            # Write out pp file
            temp_filename = iris.util.create_temp_filename(".pp")
            with open(temp_filename, "wb") as temp_fh:
                f.save(temp_fh)

            # Load pp file
            cube = iris.load_cube(temp_filename)

            # Check the process flags created
            self.assertEqual(
                set(cube.attributes["ukmo__process_flags"]),
                set(multiple_map[sum(bit_values)]),
                "Mismatch between expected and actual process " "flags.",
            )

            os.remove(temp_filename)


if __name__ == "__main__":
    tests.main()
