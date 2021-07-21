# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.coords.DimCoord` class.

Note: a lot of these methods are actually defined by the :class:`Coord` class,
but can only be tested on concrete instances (DimCoord or AuxCoord).

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np
import numpy.ma as ma

from iris.coords import DimCoord
from iris.tests.unit.coords import (
    CoordTestMixin,
    coords_all_dtypes_and_lazynesses,
    lazyness_string,
)


class DimCoordTestMixin(CoordTestMixin):
    # Define a 1-D default array shape.
    def setupTestArrays(self, shape=(3,), masked=False):
        super().setupTestArrays(shape, masked=masked)


class Test__init__(tests.IrisTest, DimCoordTestMixin):
    # Test for DimCoord creation, with various combinations of points and
    # bounds = real / lazy / None.
    def setUp(self):
        self.setupTestArrays(masked=True)

    def test_lazyness_and_dtype_combinations(self):
        for (
            coord,
            points_type_name,
            bounds_type_name,
        ) in coords_all_dtypes_and_lazynesses(self, DimCoord):
            pts = coord.core_points()
            bds = coord.core_bounds()
            # Check properties of points.
            # Points array should not be identical to the reference one.
            self.assertArraysDoNotShareData(
                pts,
                self.pts_real,
                "Points are the same data as the provided array.",
            )
            # the original points array was cast to a test dtype.
            check_pts = self.pts_real.astype(coord.dtype)
            self.assertEqualRealArraysAndDtypes(pts, check_pts)

            # Check properties of bounds.
            if bounds_type_name != "no":
                # Bounds array should not be the reference data.
                self.assertArraysDoNotShareData(
                    bds,
                    self.bds_real,
                    "Bounds are the same data as the provided array.",
                )
                # the original bounds array was cast to a test dtype.
                check_bds = self.bds_real.astype(coord.bounds_dtype)
                self.assertEqualRealArraysAndDtypes(bds, check_bds)

    def test_fail_bounds_shape_mismatch(self):
        bds_shape = list(self.bds_real.shape)
        bds_shape[0] += 1
        bds_wrong = np.zeros(bds_shape)
        msg = "The shape of the 'unknown' DimCoord bounds array should be"
        with self.assertRaisesRegex(ValueError, msg):
            DimCoord(self.pts_real, bounds=bds_wrong)

    def test_fail_nonmonotonic(self):
        msg = "must be strictly monotonic"
        with self.assertRaisesRegex(ValueError, msg):
            DimCoord([1, 2, 0, 3])

    def test_no_masked_pts_real(self):
        data = self.no_masked_pts_real
        self.assertTrue(ma.isMaskedArray(data))
        self.assertEqual(ma.count_masked(data), 0)
        coord = DimCoord(data)
        self.assertFalse(coord.has_lazy_points())
        self.assertFalse(ma.isMaskedArray(coord.points))
        self.assertEqual(ma.count_masked(coord.points), 0)

    def test_no_masked_pts_lazy(self):
        data = self.no_masked_pts_lazy
        computed = data.compute()
        self.assertTrue(ma.isMaskedArray(computed))
        self.assertEqual(ma.count_masked(computed), 0)
        coord = DimCoord(data)
        # DimCoord always realises its points.
        self.assertFalse(coord.has_lazy_points())
        self.assertFalse(ma.isMaskedArray(coord.points))

    def test_masked_pts_real(self):
        data = self.masked_pts_real
        self.assertTrue(ma.isMaskedArray(data))
        self.assertTrue(ma.count_masked(data))
        emsg = "points array must not be masked"
        with self.assertRaisesRegex(TypeError, emsg):
            DimCoord(data)

    def test_masked_pts_lazy(self):
        data = self.masked_pts_lazy
        computed = data.compute()
        self.assertTrue(ma.isMaskedArray(computed))
        self.assertTrue(ma.count_masked(computed))
        emsg = "points array must not be masked"
        with self.assertRaisesRegex(TypeError, emsg):
            DimCoord(data)

    def test_no_masked_bds_real(self):
        data = self.no_masked_bds_real
        self.assertTrue(ma.isMaskedArray(data))
        self.assertEqual(ma.count_masked(data), 0)
        coord = DimCoord(self.pts_real, bounds=data)
        self.assertFalse(coord.has_lazy_bounds())
        self.assertFalse(ma.isMaskedArray(coord.bounds))
        self.assertEqual(ma.count_masked(coord.bounds), 0)

    def test_no_masked_bds_lazy(self):
        data = self.no_masked_bds_lazy
        computed = data.compute()
        self.assertTrue(ma.isMaskedArray(computed))
        self.assertEqual(ma.count_masked(computed), 0)
        coord = DimCoord(self.pts_real, bounds=data)
        # DimCoord always realises its bounds.
        self.assertFalse(coord.has_lazy_bounds())
        self.assertFalse(ma.isMaskedArray(coord.bounds))

    def test_masked_bds_real(self):
        data = self.masked_bds_real
        self.assertTrue(ma.isMaskedArray(data))
        self.assertTrue(ma.count_masked(data))
        emsg = "bounds array must not be masked"
        with self.assertRaisesRegex(TypeError, emsg):
            DimCoord(self.pts_real, bounds=data)

    def test_masked_bds_lazy(self):
        data = self.masked_bds_lazy
        computed = data.compute()
        self.assertTrue(ma.isMaskedArray(computed))
        self.assertTrue(ma.count_masked(computed))
        emsg = "bounds array must not be masked"
        with self.assertRaisesRegex(TypeError, emsg):
            DimCoord(self.pts_real, bounds=data)


class Test_core_points(tests.IrisTest, DimCoordTestMixin):
    # Test for DimCoord.core_points() with various types of points and bounds.
    def setUp(self):
        self.setupTestArrays()

    def test_real_points(self):
        data = self.pts_real
        coord = DimCoord(data)
        result = coord.core_points()
        self.assertArraysDoNotShareData(
            result,
            self.pts_real,
            "core_points() are the same data as the internal array.",
        )

    def test_lazy_points(self):
        lazy_data = self.pts_lazy
        coord = DimCoord(lazy_data)
        result = coord.core_points()
        self.assertEqualRealArraysAndDtypes(result, self.pts_real)


class Test_core_bounds(tests.IrisTest, DimCoordTestMixin):
    # Test for DimCoord.core_bounds() with various types of points and bounds.
    def setUp(self):
        self.setupTestArrays()

    def test_no_bounds(self):
        coord = DimCoord(self.pts_real)
        result = coord.core_bounds()
        self.assertIsNone(result)

    def test_real_bounds(self):
        coord = DimCoord(self.pts_real, bounds=self.bds_real)
        result = coord.core_bounds()
        self.assertArraysDoNotShareData(
            result,
            self.bds_real,
            "core_bounds() are the same data as the internal array.",
        )

    def test_lazy_bounds(self):
        coord = DimCoord(self.pts_real, bounds=self.bds_lazy)
        result = coord.core_bounds()
        self.assertEqualRealArraysAndDtypes(result, self.bds_real)


class Test_lazy_points(tests.IrisTest, DimCoordTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_real_core(self):
        coord = DimCoord(self.pts_real)
        result = coord.lazy_points()
        self.assertEqualLazyArraysAndDtypes(result, self.pts_lazy)

    def test_lazy_core(self):
        coord = DimCoord(self.pts_lazy)
        result = coord.lazy_points()
        self.assertEqualLazyArraysAndDtypes(result, self.pts_lazy)
        # NOTE: identity, as in "result is self.pts_lazy" does *NOT* work.


class Test_lazy_bounds(tests.IrisTest, DimCoordTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_no_bounds(self):
        coord = DimCoord(self.pts_real)
        result = coord.lazy_bounds()
        self.assertIsNone(result)

    def test_real_core(self):
        coord = DimCoord(self.pts_real, bounds=self.bds_real)
        result = coord.lazy_bounds()
        self.assertEqualLazyArraysAndDtypes(result, self.bds_lazy)

    def test_lazy_core(self):
        coord = DimCoord(self.pts_real, bounds=self.bds_lazy)
        result = coord.lazy_bounds()
        self.assertEqualLazyArraysAndDtypes(result, self.bds_lazy)
        # NOTE: identity, as in "result is self.bds_lazy" does *NOT* work.


class Test_has_lazy_points(tests.IrisTest, DimCoordTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_real_core(self):
        coord = DimCoord(self.pts_real)
        result = coord.has_lazy_points()
        self.assertFalse(result)

    def test_lazy_core(self):
        coord = DimCoord(self.pts_lazy)
        result = coord.has_lazy_points()
        self.assertFalse(result)


class Test_has_lazy_bounds(tests.IrisTest, DimCoordTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_real_core(self):
        coord = DimCoord(self.pts_real, bounds=self.bds_real)
        result = coord.has_lazy_bounds()
        self.assertFalse(result)

    def test_lazy_core(self):
        coord = DimCoord(self.pts_real, bounds=self.bds_lazy)
        result = coord.has_lazy_bounds()
        self.assertFalse(result)


class Test_bounds_dtype(tests.IrisTest):
    def test_i16(self):
        test_dtype = np.int16
        coord = DimCoord([1], bounds=np.array([[0, 4]], dtype=test_dtype))
        result = coord.bounds_dtype
        self.assertEqual(result, test_dtype)

    def test_u16(self):
        test_dtype = np.uint16
        coord = DimCoord([1], bounds=np.array([[0, 4]], dtype=test_dtype))
        result = coord.bounds_dtype
        self.assertEqual(result, test_dtype)

    def test_f16(self):
        test_dtype = np.float16
        coord = DimCoord([1], bounds=np.array([[0, 4]], dtype=test_dtype))
        result = coord.bounds_dtype
        self.assertEqual(result, test_dtype)


class Test__getitem__(tests.IrisTest, DimCoordTestMixin):
    # Test for DimCoord indexing with various types of points and bounds.
    def setUp(self):
        self.setupTestArrays()

    def test_dtypes(self):
        # Index coords with all combinations of real+lazy points+bounds, and
        # either an int or floating dtype.
        # Check that dtypes remain the same in all cases, taking the dtypes
        # directly from the core points and bounds (as we have no masking).
        for (
            main_coord,
            points_type_name,
            bounds_type_name,
        ) in coords_all_dtypes_and_lazynesses(self, DimCoord):

            sub_coord = main_coord[:2]

            coord_dtype = main_coord.dtype
            msg = (
                "Indexing main_coord of dtype {} "
                "with {} points and {} bounds "
                "changed dtype of {} to {}."
            )

            sub_points = sub_coord.core_points()
            self.assertEqual(
                sub_points.dtype,
                coord_dtype,
                msg.format(
                    coord_dtype,
                    points_type_name,
                    bounds_type_name,
                    "points",
                    sub_points.dtype,
                ),
            )

            if bounds_type_name != "no":
                sub_bounds = sub_coord.core_bounds()
                main_bounds_dtype = main_coord.bounds_dtype
                self.assertEqual(
                    sub_bounds.dtype,
                    main_bounds_dtype,
                    msg.format(
                        main_bounds_dtype,
                        points_type_name,
                        bounds_type_name,
                        "bounds",
                        sub_bounds.dtype,
                    ),
                )

    def test_lazyness(self):
        # Index coords with all combinations of real+lazy points+bounds, and
        # either an int or floating dtype.
        # Check that lazy data stays lazy and real stays real, in all cases.
        for (
            main_coord,
            points_type_name,
            bounds_type_name,
        ) in coords_all_dtypes_and_lazynesses(self, DimCoord):
            # N.B. 'points_type_name' and 'bounds_type_name' in the iteration
            # are the original types (lazy/real/none) of the points+bounds,
            # but the DimCoord itself only ever has real data.
            if points_type_name == "lazy":
                points_type_name = "real"
            if bounds_type_name == "lazy":
                bounds_type_name = "real"

            sub_coord = main_coord[:2]

            msg = (
                "Indexing coord of dtype {} "
                "with {} points and {} bounds "
                'changed "lazyness" of {} from {!r} to {!r}.'
            )
            coord_dtype = main_coord.dtype
            sub_points_lazyness = lazyness_string(sub_coord.core_points())
            self.assertEqual(
                sub_points_lazyness,
                points_type_name,
                msg.format(
                    coord_dtype,
                    points_type_name,
                    bounds_type_name,
                    "points",
                    points_type_name,
                    sub_points_lazyness,
                ),
            )

            if bounds_type_name != "no":
                sub_bounds_lazy = lazyness_string(sub_coord.core_bounds())
                self.assertEqual(
                    sub_bounds_lazy,
                    bounds_type_name,
                    msg.format(
                        coord_dtype,
                        points_type_name,
                        bounds_type_name,
                        "bounds",
                        bounds_type_name,
                        sub_bounds_lazy,
                    ),
                )

    def test_real_data_copies(self):
        # Index coords with all combinations of real+lazy points+bounds.
        # In all cases, check that any real arrays are copied by the indexing.
        for (
            main_coord,
            points_lazyness,
            bounds_lazyness,
        ) in coords_all_dtypes_and_lazynesses(self, DimCoord):

            sub_coord = main_coord[:2]

            msg = (
                "Indexed coord with {} points and {} bounds "
                "does not have its own separate {} array."
            )
            if points_lazyness == "real":
                main_points = main_coord.core_points()
                sub_points = sub_coord.core_points()
                sub_main_points = main_points[:2]
                self.assertEqualRealArraysAndDtypes(
                    sub_points, sub_main_points
                )
                self.assertArraysDoNotShareData(
                    sub_points,
                    sub_main_points,
                    msg.format(points_lazyness, bounds_lazyness, "points"),
                )

            if bounds_lazyness == "real":
                main_bounds = main_coord.core_bounds()
                sub_bounds = sub_coord.core_bounds()
                sub_main_bounds = main_bounds[:2]
                self.assertEqualRealArraysAndDtypes(
                    sub_bounds, sub_main_bounds
                )
                self.assertArraysDoNotShareData(
                    sub_bounds,
                    sub_main_bounds,
                    msg.format(points_lazyness, bounds_lazyness, "bounds"),
                )


class Test_copy(tests.IrisTest, DimCoordTestMixin):
    # Test for DimCoord.copy() with various types of points and bounds.
    def setUp(self):
        self.setupTestArrays()

    def test_writable_points(self):
        coord1 = DimCoord(
            np.arange(5), bounds=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
        )
        coord2 = coord1.copy()
        msg = "destination is read-only"

        with self.assertRaisesRegex(ValueError, msg):
            coord1.points[:] = 0

        with self.assertRaisesRegex(ValueError, msg):
            coord2.points[:] = 0

        with self.assertRaisesRegex(ValueError, msg):
            coord1.bounds[:] = 0

        with self.assertRaisesRegex(ValueError, msg):
            coord2.bounds[:] = 0

    def test_realdata_readonly(self):
        # Copy coords with all combinations of real+lazy points+bounds.
        # In all cases, check that data arrays are read-only.
        for (
            main_coord,
            points_type_name,
            bounds_type_name,
        ) in coords_all_dtypes_and_lazynesses(self, DimCoord):

            copied_coord = main_coord.copy()

            copied_points = copied_coord.core_points()
            expected_error_msg = "output array is read-only"
            with self.assertRaisesRegex(ValueError, expected_error_msg):
                copied_points[:1] += 33

            if bounds_type_name != "no":
                copied_bounds = copied_coord.core_bounds()
                with self.assertRaisesRegex(ValueError, expected_error_msg):
                    copied_bounds[:1] += 33


class Test_points__getter(tests.IrisTest, DimCoordTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_real_points(self):
        # Getting real points returns a copy
        coord = DimCoord(self.pts_real)
        result = coord.core_points()
        self.assertArraysDoNotShareData(
            result,
            self.pts_real,
            "Points are the same array as the provided data.",
        )


class Test_points__setter(tests.IrisTest, DimCoordTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_set_real(self):
        # Setting points copies the data
        coord = DimCoord(self.pts_real)
        new_pts = self.pts_real + 102.3
        coord.points = new_pts
        result = coord.core_points()
        self.assertArraysDoNotShareData(
            result, new_pts, "Points are the same data as the assigned array."
        )

    def test_fail_bad_shape(self):
        # Setting real points requires matching shape.
        points = [1.0, 2.0]
        coord = DimCoord(points)
        msg = r"Require data with shape \(2,\), got \(3,\)"
        with self.assertRaisesRegex(ValueError, msg):
            coord.points = np.array([1.0, 2.0, 3.0])
        self.assertArrayEqual(coord.points, points)

    def test_fail_not_monotonic(self):
        # Setting real points requires that they are monotonic.
        coord = DimCoord(self.pts_real, bounds=self.bds_real)
        msg = "strictly monotonic"
        with self.assertRaisesRegex(ValueError, msg):
            coord.points = np.array([3.0, 1.0, 2.0])
        self.assertArrayEqual(coord.points, self.pts_real)

    def test_set_lazy(self):
        # Setting new lazy points realises them.
        coord = DimCoord(self.pts_real)
        new_pts = self.pts_lazy + 102.3
        coord.points = new_pts
        result = coord.core_points()
        self.assertEqualRealArraysAndDtypes(result, new_pts.compute())

    def test_copy_array(self):
        # Assigning points creates a copy
        pts = np.array([1, 2, 3])
        coord = DimCoord(pts)
        pts[1] = 5
        self.assertEqual(coord.points[1], 2)


class Test_bounds__getter(tests.IrisTest, DimCoordTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_real_bounds(self):
        # Getting real bounds does not change or copy them.
        coord = DimCoord(self.pts_real, bounds=self.bds_real)
        result = coord.bounds
        self.assertArraysDoNotShareData(
            result,
            self.bds_real,
            "Bounds are the same array as the provided data.",
        )


class Test_bounds__setter(tests.IrisTest, DimCoordTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_set_real(self):
        # Setting bounds does not copy, but makes a readonly view.
        coord = DimCoord(self.pts_real, bounds=self.bds_real)
        new_bounds = self.bds_real + 102.3
        coord.bounds = new_bounds
        result = coord.core_bounds()
        self.assertArraysDoNotShareData(
            result,
            new_bounds,
            "Bounds are the same data as the assigned array.",
        )

    def test_fail_bad_shape(self):
        # Setting real points requires matching shape.
        coord = DimCoord(self.pts_real, bounds=self.bds_real)
        msg = "The shape of the 'unknown' DimCoord bounds array should be"
        with self.assertRaisesRegex(ValueError, msg):
            coord.bounds = np.array([1.0, 2.0, 3.0])
        self.assertArrayEqual(coord.bounds, self.bds_real)

    def test_fail_not_monotonic(self):
        # Setting real bounds requires that they are monotonic.
        coord = DimCoord(self.pts_real, bounds=self.bds_real)
        msg = "strictly monotonic"
        with self.assertRaisesRegex(ValueError, msg):
            coord.bounds = np.array([[3.0, 2.0], [1.0, 0.0], [2.0, 1.0]])
        self.assertArrayEqual(coord.bounds, self.bds_real)

    def test_set_lazy(self):
        # Setting new lazy bounds realises them.
        coord = DimCoord(self.pts_real, bounds=self.bds_lazy)
        new_bounds = self.bds_lazy + 102.3
        coord.bounds = new_bounds
        result = coord.core_bounds()
        self.assertEqualRealArraysAndDtypes(result, new_bounds.compute())

    def test_copy_array(self):
        # Assigning bounds creates a copy
        pts = np.array([2, 4, 6])
        bnds = np.array([[1, 3], [3, 5], [5, 7]])
        coord = DimCoord(pts, bounds=bnds)
        bnds[1, 1] = 10
        self.assertEqual(coord.bounds[1, 1], 5)


if __name__ == "__main__":
    tests.main()
