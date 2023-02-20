# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.coords.AuxCoord` class.

Note: a lot of these methods are actually defined by the :class:`Coord` class,
but can only be tested on concrete instances (DimCoord or AuxCoord).

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from cf_units import Unit
import numpy as np
import numpy.ma as ma

from iris._lazy_data import as_lazy_data
from iris.coords import AuxCoord
from iris.tests.unit.coords import (
    CoordTestMixin,
    coords_all_dtypes_and_lazynesses,
    lazyness_string,
)


class AuxCoordTestMixin(CoordTestMixin):
    # Define a 2-D default array shape.
    def setupTestArrays(self, shape=(2, 3), masked=False):
        super().setupTestArrays(shape, masked=masked)


class Test__init__(tests.IrisTest, AuxCoordTestMixin):
    # Test for AuxCoord creation, with various combinations of points and
    # bounds = real / lazy / None.
    def setUp(self):
        self.setupTestArrays(masked=True)

    def test_lazyness_and_dtype_combinations(self):
        for (
            coord,
            points_type_name,
            bounds_type_name,
        ) in coords_all_dtypes_and_lazynesses(self, AuxCoord):
            pts = coord.core_points()
            bds = coord.core_bounds()
            # Check properties of points.
            if points_type_name == "real":
                # Real points.
                if coord.dtype == self.pts_real.dtype:
                    self.assertArraysShareData(
                        pts,
                        self.pts_real,
                        "Points are not the same data as the provided array.",
                    )
                    self.assertIsNot(
                        pts,
                        self.pts_real,
                        "Points array is the same instance as the provided "
                        "array.",
                    )
                else:
                    # the original points were cast to a test dtype.
                    check_pts = self.pts_real.astype(coord.dtype)
                    self.assertEqualRealArraysAndDtypes(pts, check_pts)
            else:
                # Lazy points : the core data may be promoted to float.
                check_pts = self.pts_lazy.astype(pts.dtype)
                self.assertEqualLazyArraysAndDtypes(pts, check_pts)
                # The realisation type should be correct, though.
                target_dtype = coord.dtype
                self.assertEqual(coord.points.dtype, target_dtype)

            # Check properties of bounds.
            if bounds_type_name == "real":
                # Real bounds.
                if coord.bounds_dtype == self.bds_real.dtype:
                    self.assertArraysShareData(
                        bds,
                        self.bds_real,
                        "Bounds are not the same data as the provided array.",
                    )
                    self.assertIsNot(
                        pts,
                        self.pts_real,
                        "Bounds array is the same instance as the provided "
                        "array.",
                    )
                else:
                    # the original bounds were cast to a test dtype.
                    check_bds = self.bds_real.astype(coord.bounds_dtype)
                    self.assertEqualRealArraysAndDtypes(bds, check_bds)
            elif bounds_type_name == "lazy":
                # Lazy points : the core data may be promoted to float.
                check_bds = self.bds_lazy.astype(bds.dtype)
                self.assertEqualLazyArraysAndDtypes(bds, check_bds)
                # The realisation type should be correct, though.
                target_dtype = coord.bounds_dtype
                self.assertEqual(coord.bounds.dtype, target_dtype)

    def test_fail_bounds_shape_mismatch(self):
        bds_shape = list(self.bds_real.shape)
        bds_shape[0] += 1
        bds_wrong = np.zeros(bds_shape)
        msg = "Bounds shape must be compatible with points shape"
        with self.assertRaisesRegex(ValueError, msg):
            AuxCoord(self.pts_real, bounds=bds_wrong)

    def test_no_masked_pts_real(self):
        data = self.no_masked_pts_real
        self.assertTrue(ma.isMaskedArray(data))
        self.assertEqual(ma.count_masked(data), 0)
        coord = AuxCoord(data)
        self.assertFalse(coord.has_lazy_points())
        self.assertTrue(ma.isMaskedArray(coord.points))
        self.assertEqual(ma.count_masked(coord.points), 0)

    def test_no_masked_pts_lazy(self):
        data = self.no_masked_pts_lazy
        computed = data.compute()
        self.assertTrue(ma.isMaskedArray(computed))
        self.assertEqual(ma.count_masked(computed), 0)
        coord = AuxCoord(data)
        self.assertTrue(coord.has_lazy_points())
        self.assertTrue(ma.isMaskedArray(coord.points))
        self.assertEqual(ma.count_masked(coord.points), 0)

    def test_masked_pts_real(self):
        data = self.masked_pts_real
        self.assertTrue(ma.isMaskedArray(data))
        self.assertTrue(ma.count_masked(data))
        coord = AuxCoord(data)
        self.assertFalse(coord.has_lazy_points())
        self.assertTrue(ma.isMaskedArray(coord.points))
        self.assertTrue(ma.count_masked(coord.points))

    def test_masked_pts_lazy(self):
        data = self.masked_pts_lazy
        computed = data.compute()
        self.assertTrue(ma.isMaskedArray(computed))
        self.assertTrue(ma.count_masked(computed))
        coord = AuxCoord(data)
        self.assertTrue(coord.has_lazy_points())
        self.assertTrue(ma.isMaskedArray(coord.points))
        self.assertTrue(ma.count_masked(coord.points))

    def test_no_masked_bds_real(self):
        data = self.no_masked_bds_real
        self.assertTrue(ma.isMaskedArray(data))
        self.assertEqual(ma.count_masked(data), 0)
        coord = AuxCoord(self.pts_real, bounds=data)
        self.assertFalse(coord.has_lazy_bounds())
        self.assertTrue(ma.isMaskedArray(coord.bounds))
        self.assertEqual(ma.count_masked(coord.bounds), 0)

    def test_no_masked_bds_lazy(self):
        data = self.no_masked_bds_lazy
        computed = data.compute()
        self.assertTrue(ma.isMaskedArray(computed))
        self.assertEqual(ma.count_masked(computed), 0)
        coord = AuxCoord(self.pts_real, bounds=data)
        self.assertTrue(coord.has_lazy_bounds())
        self.assertTrue(ma.isMaskedArray(coord.bounds))
        self.assertEqual(ma.count_masked(coord.bounds), 0)

    def test_masked_bds_real(self):
        data = self.masked_bds_real
        self.assertTrue(ma.isMaskedArray(data))
        self.assertTrue(ma.count_masked(data))
        coord = AuxCoord(self.pts_real, bounds=data)
        self.assertFalse(coord.has_lazy_bounds())
        self.assertTrue(ma.isMaskedArray(coord.bounds))
        self.assertTrue(ma.count_masked(coord.bounds))

    def test_masked_bds_lazy(self):
        data = self.masked_bds_lazy
        computed = data.compute()
        self.assertTrue(ma.isMaskedArray(computed))
        self.assertTrue(ma.count_masked(computed))
        coord = AuxCoord(self.pts_real, bounds=data)
        self.assertTrue(coord.has_lazy_bounds())
        self.assertTrue(ma.isMaskedArray(coord.bounds))
        self.assertTrue(ma.count_masked(coord.bounds))


class Test_core_points(tests.IrisTest, AuxCoordTestMixin):
    # Test for AuxCoord.core_points() with various types of points and bounds.
    def setUp(self):
        self.setupTestArrays()

    def test_real_points(self):
        coord = AuxCoord(self.pts_real)
        result = coord.core_points()
        self.assertArraysShareData(
            result,
            self.pts_real,
            "core_points() do not share data with the internal array.",
        )

    def test_lazy_points(self):
        coord = AuxCoord(self.pts_lazy)
        result = coord.core_points()
        self.assertEqualLazyArraysAndDtypes(result, self.pts_lazy)

    def test_lazy_points_realise(self):
        coord = AuxCoord(self.pts_lazy)
        real_points = coord.points
        result = coord.core_points()
        self.assertEqualRealArraysAndDtypes(result, real_points)


class Test_core_bounds(tests.IrisTest, AuxCoordTestMixin):
    # Test for AuxCoord.core_bounds() with various types of points and bounds.
    def setUp(self):
        self.setupTestArrays()

    def test_no_bounds(self):
        coord = AuxCoord(self.pts_real)
        result = coord.core_bounds()
        self.assertIsNone(result)

    def test_real_bounds(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        result = coord.core_bounds()
        self.assertArraysShareData(
            result,
            self.bds_real,
            "core_bounds() do not share data with the internal array.",
        )

    def test_lazy_bounds(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)
        result = coord.core_bounds()
        self.assertEqualLazyArraysAndDtypes(result, self.bds_lazy)

    def test_lazy_bounds_realise(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)
        real_bounds = coord.bounds
        result = coord.core_bounds()
        self.assertEqualRealArraysAndDtypes(result, real_bounds)


class Test_lazy_points(tests.IrisTest, AuxCoordTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_real_core(self):
        coord = AuxCoord(self.pts_real)
        result = coord.lazy_points()
        self.assertEqualLazyArraysAndDtypes(result, self.pts_lazy)

    def test_lazy_core(self):
        coord = AuxCoord(self.pts_lazy)
        result = coord.lazy_points()
        self.assertIs(result, self.pts_lazy)


class Test_lazy_bounds(tests.IrisTest, AuxCoordTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_no_bounds(self):
        coord = AuxCoord(self.pts_real)
        result = coord.lazy_bounds()
        self.assertIsNone(result)

    def test_real_core(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        result = coord.lazy_bounds()
        self.assertEqualLazyArraysAndDtypes(result, self.bds_lazy)

    def test_lazy_core(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)
        result = coord.lazy_bounds()
        self.assertIs(result, self.bds_lazy)


class Test_has_lazy_points(tests.IrisTest, AuxCoordTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_real_core(self):
        coord = AuxCoord(self.pts_real)
        result = coord.has_lazy_points()
        self.assertFalse(result)

    def test_lazy_core(self):
        coord = AuxCoord(self.pts_lazy)
        result = coord.has_lazy_points()
        self.assertTrue(result)

    def test_lazy_core_realise(self):
        coord = AuxCoord(self.pts_lazy)
        coord.points
        result = coord.has_lazy_points()
        self.assertFalse(result)


class Test_has_lazy_bounds(tests.IrisTest, AuxCoordTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_real_core(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        result = coord.has_lazy_bounds()
        self.assertFalse(result)

    def test_lazy_core(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)
        result = coord.has_lazy_bounds()
        self.assertTrue(result)

    def test_lazy_core_realise(self):
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)
        coord.bounds
        result = coord.has_lazy_bounds()
        self.assertFalse(result)


class Test_bounds_dtype(tests.IrisTest, AuxCoordTestMixin):
    def test_i16(self):
        test_dtype = np.int16
        coord = AuxCoord([1], bounds=np.array([[0, 4]], dtype=test_dtype))
        result = coord.bounds_dtype
        self.assertEqual(result, test_dtype)

    def test_u16(self):
        test_dtype = np.uint16
        coord = AuxCoord([1], bounds=np.array([[0, 4]], dtype=test_dtype))
        result = coord.bounds_dtype
        self.assertEqual(result, test_dtype)

    def test_f16(self):
        test_dtype = np.float16
        coord = AuxCoord([1], bounds=np.array([[0, 4]], dtype=test_dtype))
        result = coord.bounds_dtype
        self.assertEqual(result, test_dtype)


class Test__getitem__(tests.IrisTest, AuxCoordTestMixin):
    # Test for AuxCoord indexing with various types of points and bounds.
    def setUp(self):
        self.setupTestArrays()

    def test_partial_slice_data_copy(self):
        parent_coord = AuxCoord([1.0, 2.0, 3.0])
        sub_coord = parent_coord[:1]
        values_before_change = sub_coord.points.copy()
        parent_coord.points[:] = -999.9
        self.assertArrayEqual(sub_coord.points, values_before_change)

    def test_full_slice_data_copy(self):
        parent_coord = AuxCoord([1.0, 2.0, 3.0])
        sub_coord = parent_coord[:]
        values_before_change = sub_coord.points.copy()
        parent_coord.points[:] = -999.9
        self.assertArrayEqual(sub_coord.points, values_before_change)

    def test_dtypes(self):
        # Index coords with all combinations of real+lazy points+bounds, and
        # either an int or floating dtype.
        # Check that dtypes remain the same in all cases, taking the dtypes
        # directly from the core points and bounds (as we have no masking).
        for (
            main_coord,
            points_type_name,
            bounds_type_name,
        ) in coords_all_dtypes_and_lazynesses(self, AuxCoord):
            sub_coord = main_coord[:2, 1]

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
        ) in coords_all_dtypes_and_lazynesses(self, AuxCoord):
            sub_coord = main_coord[:2, 1]

            msg = (
                "Indexing coord of dtype {} "
                "with {} points and {} bounds "
                "changed laziness of {} from {!r} to {!r}."
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
        ) in coords_all_dtypes_and_lazynesses(self, AuxCoord):
            sub_coord = main_coord[:2, 1]

            msg = (
                "Indexed coord with {} points and {} bounds "
                "does not have its own separate {} array."
            )
            if points_lazyness == "real":
                main_points = main_coord.core_points()
                sub_points = sub_coord.core_points()
                sub_main_points = main_points[:2, 1]
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
                sub_main_bounds = main_bounds[:2, 1]
                self.assertEqualRealArraysAndDtypes(
                    sub_bounds, sub_main_bounds
                )
                self.assertArraysDoNotShareData(
                    sub_bounds,
                    sub_main_bounds,
                    msg.format(points_lazyness, bounds_lazyness, "bounds"),
                )


class Test_copy(tests.IrisTest, AuxCoordTestMixin):
    # Test for AuxCoord.copy() with various types of points and bounds.
    def setUp(self):
        self.setupTestArrays()

    def test_lazyness(self):
        # Copy coords with all combinations of real+lazy points+bounds, and
        # either an int or floating dtype.
        # Check that lazy data stays lazy and real stays real, in all cases.
        for (
            main_coord,
            points_lazyness,
            bounds_lazyness,
        ) in coords_all_dtypes_and_lazynesses(self, AuxCoord):
            coord_dtype = main_coord.dtype
            copied_coord = main_coord.copy()

            msg = (
                "Copying main_coord of dtype {} "
                "with {} points and {} bounds "
                "changed lazyness of {} from {!r} to {!r}."
            )

            copied_pts_lazyness = lazyness_string(copied_coord.core_points())
            self.assertEqual(
                copied_pts_lazyness,
                points_lazyness,
                msg.format(
                    coord_dtype,
                    points_lazyness,
                    bounds_lazyness,
                    "points",
                    points_lazyness,
                    copied_pts_lazyness,
                ),
            )

            if bounds_lazyness != "no":
                copied_bds_lazy = lazyness_string(copied_coord.core_bounds())
                self.assertEqual(
                    copied_bds_lazy,
                    bounds_lazyness,
                    msg.format(
                        coord_dtype,
                        points_lazyness,
                        bounds_lazyness,
                        "bounds",
                        bounds_lazyness,
                        copied_bds_lazy,
                    ),
                )

    def test_realdata_copies(self):
        # Copy coords with all combinations of real+lazy points+bounds.
        # In all cases, check that any real arrays are copies, not views.
        for (
            main_coord,
            points_lazyness,
            bounds_lazyness,
        ) in coords_all_dtypes_and_lazynesses(self, AuxCoord):
            copied_coord = main_coord.copy()

            msg = (
                "Copied coord with {} points and {} bounds "
                "does not have its own separate {} array."
            )

            if points_lazyness == "real":
                main_points = main_coord.core_points()
                copied_points = copied_coord.core_points()
                self.assertEqualRealArraysAndDtypes(main_points, copied_points)
                self.assertArraysDoNotShareData(
                    main_points,
                    copied_points,
                    msg.format(points_lazyness, bounds_lazyness, "points"),
                )

            if bounds_lazyness == "real":
                main_bounds = main_coord.core_bounds()
                copied_bounds = copied_coord.core_bounds()
                self.assertEqualRealArraysAndDtypes(main_bounds, copied_bounds)
                self.assertArraysDoNotShareData(
                    main_bounds,
                    copied_bounds,
                    msg.format(points_lazyness, bounds_lazyness, "bounds"),
                )


class Test_points__getter(tests.IrisTest, AuxCoordTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_mutable_real_points(self):
        # Check that coord.points returns a modifiable array, and changes to it
        # are reflected to the coord.
        data = np.array([1.0, 2.0, 3.0, 4.0])
        coord = AuxCoord(data)
        initial_values = data.copy()
        coord.points[1:2] += 33.1
        result = coord.points
        self.assertFalse(np.all(result == initial_values))

    def test_real_points(self):
        # Getting real points does not change or copy them.
        coord = AuxCoord(self.pts_real)
        result = coord.points
        self.assertArraysShareData(
            result,
            self.pts_real,
            "Points do not share data with the provided array.",
        )

    def test_lazy_points(self):
        # Getting lazy points realises them.
        coord = AuxCoord(self.pts_lazy)
        self.assertTrue(coord.has_lazy_points())
        result = coord.points
        self.assertFalse(coord.has_lazy_points())
        self.assertEqualRealArraysAndDtypes(result, self.pts_real)

    def test_real_points_with_real_bounds(self):
        # Getting real points does not change real bounds.
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        coord.points
        result = coord.core_bounds()
        self.assertArraysShareData(
            result,
            self.bds_real,
            "Bounds do not share data with the provided array.",
        )

    def test_real_points_with_lazy_bounds(self):
        # Getting real points does not touch lazy bounds.
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)
        coord.points
        self.assertTrue(coord.has_lazy_bounds())

    def test_lazy_points_with_real_bounds(self):
        # Getting lazy points does not affect real bounds.
        coord = AuxCoord(self.pts_lazy, bounds=self.bds_real)
        coord.points
        result = coord.core_bounds()
        self.assertEqualRealArraysAndDtypes(result, self.bds_real)

    def test_lazy_points_with_lazy_bounds(self):
        # Getting lazy points does not touch lazy bounds.
        coord = AuxCoord(self.pts_lazy, bounds=self.bds_lazy)
        coord.points
        self.assertTrue(coord.has_lazy_bounds())


class Test_points__setter(tests.IrisTest, AuxCoordTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_real_set_real(self):
        # Setting new real points does not make a copy.
        coord = AuxCoord(self.pts_real)
        new_pts = self.pts_real + 102.3
        coord.points = new_pts
        result = coord.core_points()
        self.assertArraysShareData(
            result,
            new_pts,
            "Points do not share data with the assigned array.",
        )

    def test_fail_bad_shape(self):
        # Setting real points requires matching shape.
        coord = AuxCoord([1.0, 2.0])
        msg = r"Require data with shape \(2,\), got \(3,\)"
        with self.assertRaisesRegex(ValueError, msg):
            coord.points = np.array([1.0, 2.0, 3.0])

    def test_real_set_lazy(self):
        # Setting new lazy points does not make a copy.
        coord = AuxCoord(self.pts_real)
        new_pts = self.pts_lazy + 102.3
        coord.points = new_pts
        result = coord.core_points()
        self.assertEqualLazyArraysAndDtypes(result, new_pts)

    def test_set_points_with_lazy_bounds(self):
        # Setting points does not touch lazy bounds.
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)
        new_pts = self.pts_real + 102.3
        coord.points = new_pts
        result = coord.core_bounds()
        self.assertEqualLazyArraysAndDtypes(result, self.bds_lazy)


class Test_bounds__getter(tests.IrisTest, AuxCoordTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_mutable_real_bounds(self):
        # Check that coord.bounds returns a modifiable array, and changes to it
        # are reflected to the coord.
        pts_data = np.array([1.5, 2.5])
        bds_data = np.array([[1.4, 1.6], [2.4, 2.6]])
        coord = AuxCoord(pts_data, bounds=bds_data)
        initial_values = bds_data.copy()
        coord.bounds[1:2] += 33.1
        result = coord.bounds
        self.assertFalse(np.all(result == initial_values))

    def test_real_bounds(self):
        # Getting real bounds does not change or copy them.
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        result = coord.bounds
        self.assertArraysShareData(
            result,
            self.bds_real,
            "Bounds do not share data with the provided array.",
        )

    def test_lazy_bounds(self):
        # Getting lazy bounds realises them.
        coord = AuxCoord(self.pts_real, bounds=self.bds_lazy)
        self.assertTrue(coord.has_lazy_bounds())
        result = coord.bounds
        self.assertFalse(coord.has_lazy_bounds())
        self.assertEqualRealArraysAndDtypes(result, self.bds_real)

    def test_lazy_bounds_with_lazy_points(self):
        # Getting lazy bounds does not fetch the points.
        coord = AuxCoord(self.pts_lazy, bounds=self.bds_lazy)
        coord.bounds
        self.assertTrue(coord.has_lazy_points())


class Test_bounds__setter(tests.IrisTest, AuxCoordTestMixin):
    def setUp(self):
        self.setupTestArrays()

    def test_set_real_bounds(self):
        # Setting new real bounds does not make a copy.
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        new_bounds = self.bds_real + 102.3
        coord.bounds = new_bounds
        result = coord.core_bounds()
        self.assertArraysShareData(
            result,
            new_bounds,
            "Bounds do not share data with the assigned array.",
        )

    def test_fail_bad_shape(self):
        # Setting real points requires matching shape.
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        msg = "must be compatible with points shape"
        with self.assertRaisesRegex(ValueError, msg):
            coord.bounds = np.array([1.0, 2.0, 3.0])

    def test_set_lazy_bounds(self):
        # Setting new lazy bounds.
        coord = AuxCoord(self.pts_real, bounds=self.bds_real)
        new_bounds = self.bds_lazy + 102.3
        coord.bounds = new_bounds
        result = coord.core_bounds()
        self.assertEqualLazyArraysAndDtypes(result, new_bounds)

    def test_set_bounds_with_lazy_points(self):
        # Setting bounds does not change lazy points.
        coord = AuxCoord(self.pts_lazy, bounds=self.bds_real)
        new_bounds = self.bds_real + 102.3
        coord.bounds = new_bounds
        self.assertTrue(coord.has_lazy_points())


class Test_convert_units(tests.IrisTest):
    def test_preserves_lazy(self):
        test_bounds = np.array(
            [
                [[11.0, 12.0], [12.0, 13.0], [13.0, 14.0]],
                [[21.0, 22.0], [22.0, 23.0], [23.0, 24.0]],
            ]
        )
        test_points = np.array([[11.1, 12.2, 13.3], [21.4, 22.5, 23.6]])
        lazy_points = as_lazy_data(test_points)
        lazy_bounds = as_lazy_data(test_bounds)
        coord = AuxCoord(points=lazy_points, bounds=lazy_bounds, units="m")
        coord.convert_units("ft")
        self.assertTrue(coord.has_lazy_points())
        self.assertTrue(coord.has_lazy_bounds())
        test_points_ft = Unit("m").convert(test_points, "ft")
        test_bounds_ft = Unit("m").convert(test_bounds, "ft")
        self.assertArrayAllClose(coord.points, test_points_ft)
        self.assertArrayAllClose(coord.bounds, test_bounds_ft)


class TestEquality(tests.IrisTest):
    def test_nanpoints_eq_self(self):
        co1 = AuxCoord([1.0, np.nan, 2.0])
        self.assertEqual(co1, co1)

    def test_nanpoints_eq_copy(self):
        co1 = AuxCoord([1.0, np.nan, 2.0])
        co2 = co1.copy()
        self.assertEqual(co1, co2)

    def test_nanbounds_eq_self(self):
        co1 = AuxCoord([15.0, 25.0], bounds=[[14.0, 16.0], [24.0, np.nan]])
        self.assertEqual(co1, co1)


if __name__ == "__main__":
    tests.main()
