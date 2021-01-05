# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.common.resolve.Resolve`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import unittest.mock as mock
from unittest.mock import sentinel

from iris.common.resolve import Resolve
from iris.cube import Cube


class Test___init__(tests.IrisTest):
    def setUp(self):
        target = "iris.common.resolve.Resolve.__call__"
        self.mcall = mock.MagicMock(return_value=sentinel.return_value)
        _ = self.patch(target, new=self.mcall)

    def _assert_members_none(self, resolve):
        self.assertIsNone(resolve.lhs_cube_resolved)
        self.assertIsNone(resolve.rhs_cube_resolved)
        self.assertIsNone(resolve.lhs_cube_category)
        self.assertIsNone(resolve.rhs_cube_category)
        self.assertIsNone(resolve.lhs_cube_category_local)
        self.assertIsNone(resolve.rhs_cube_category_local)
        self.assertIsNone(resolve.category_common)
        self.assertIsNone(resolve.lhs_cube_dim_coverage)
        self.assertIsNone(resolve.lhs_cube_aux_coverage)
        self.assertIsNone(resolve.rhs_cube_dim_coverage)
        self.assertIsNone(resolve.rhs_cube_aux_coverage)
        self.assertIsNone(resolve.map_rhs_to_lhs)
        self.assertIsNone(resolve.mapping)
        self.assertIsNone(resolve.prepared_category)
        self.assertIsNone(resolve.prepared_factories)
        self.assertIsNone(resolve._broadcast_shape)

    def test_lhs_rhs_default(self):
        resolve = Resolve()
        self.assertIsNone(resolve.lhs_cube)
        self.assertIsNone(resolve.rhs_cube)
        self._assert_members_none(resolve)
        self.assertEqual(self.mcall.call_count, 0)

    def test_lhs_rhs_provided(self):
        lhs = sentinel.lhs
        rhs = sentinel.rhs
        resolve = Resolve(lhs=lhs, rhs=rhs)
        self.assertIsNone(resolve.lhs_cube)
        self.assertIsNone(resolve.rhs_cube)
        self._assert_members_none(resolve)
        self.assertEqual(self.mcall.call_count, 1)
        call_args = mock.call(lhs, rhs)
        self.assertEqual(self.mcall.call_args, call_args)


class Test___call__(tests.IrisTest):
    def setUp(self):
        self.lhs = mock.MagicMock(spec=Cube)
        self.rhs = mock.MagicMock(spec=Cube)
        target = "iris.common.resolve.Resolve.{method}"
        method = target.format(method="_metadata_resolve")
        self.metadata_resolve = self.patch(method)
        method = target.format(method="_metadata_coverage")
        self.metadata_coverage = self.patch(method)
        method = target.format(method="_metadata_mapping")
        self.metadata_mapping = self.patch(method)
        method = target.format(method="_metadata_prepare")
        self.metadata_prepare = self.patch(method)

    def test_lhs_not_cube(self):
        emsg = "'LHS' argument to be a 'Cube'"
        with self.assertRaisesRegex(TypeError, emsg):
            _ = Resolve(rhs=self.rhs)

    def test_rhs_not_cube(self):
        emsg = "'RHS' argument to be a 'Cube'"
        with self.assertRaisesRegex(TypeError, emsg):
            _ = Resolve(lhs=self.lhs)

    def _assert_called_metadata_methods(self):
        call_args = mock.call()
        self.assertEqual(self.metadata_resolve.call_count, 1)
        self.assertEqual(self.metadata_resolve.call_args, call_args)
        self.assertEqual(self.metadata_coverage.call_count, 1)
        self.assertEqual(self.metadata_coverage.call_args, call_args)
        self.assertEqual(self.metadata_mapping.call_count, 1)
        self.assertEqual(self.metadata_mapping.call_args, call_args)
        self.assertEqual(self.metadata_prepare.call_count, 1)
        self.assertEqual(self.metadata_prepare.call_args, call_args)

    def test_map_rhs_to_lhs__less_than(self):
        self.lhs.ndim = 2
        self.rhs.ndim = 1
        resolve = Resolve(lhs=self.lhs, rhs=self.rhs)
        self.assertEqual(resolve.lhs_cube, self.lhs)
        self.assertEqual(resolve.rhs_cube, self.rhs)
        self.assertTrue(resolve.map_rhs_to_lhs)
        self._assert_called_metadata_methods()

    def test_map_rhs_to_lhs__equal(self):
        self.lhs.ndim = 2
        self.rhs.ndim = 2
        resolve = Resolve(lhs=self.lhs, rhs=self.rhs)
        self.assertEqual(resolve.lhs_cube, self.lhs)
        self.assertEqual(resolve.rhs_cube, self.rhs)
        self.assertTrue(resolve.map_rhs_to_lhs)
        self._assert_called_metadata_methods()

    def test_map_lhs_to_rhs__equal(self):
        self.lhs.ndim = 2
        self.rhs.ndim = 3
        resolve = Resolve(lhs=self.lhs, rhs=self.rhs)
        self.assertEqual(resolve.lhs_cube, self.lhs)
        self.assertEqual(resolve.rhs_cube, self.rhs)
        self.assertFalse(resolve.map_rhs_to_lhs)
        self._assert_called_metadata_methods()


if __name__ == "__main__":
    tests.main()
