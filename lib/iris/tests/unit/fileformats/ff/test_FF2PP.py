# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.fileformat.ff.FF2PP` class."""

import collections
import contextlib

import numpy as np
import pytest

from iris.exceptions import NotYetImplementedError
import iris.fileformats._ff as ff
from iris.fileformats._ff import FF2PP
import iris.fileformats.pp as pp
from iris.tests import _shared_utils
from iris.tests.unit.fileformats import MockerMixin
from iris.warnings import IrisLoadWarning

# PP-field: LBPACK N1 values.
_UNPACKED = 0
_WGDOS = 1
_CRAY = 2

# PP-field: LBUSER(1) values.
_REAL = 1
_INTEGER = 2


_DummyField = collections.namedtuple(
    "_DummyField", "lbext lblrec lbnrec raw_lbpack lbuser boundary_packing"
)
_DummyFieldWithSize = collections.namedtuple(
    "_DummyFieldWithSize",
    "lbext lblrec lbnrec raw_lbpack lbuser boundary_packing lbnpt lbrow",
)
_DummyBoundaryPacking = collections.namedtuple(
    "_DummyBoundaryPacking", "x_halo y_halo rim_width"
)


class Test____iter__(MockerMixin):
    def test_call_structure(self, mocker):
        # Check that the iter method calls the two necessary utility
        # functions
        _FFHeader = mocker.patch("iris.fileformats._ff.FFHeader")
        extract_result = mocker.Mock()
        interpret_patch = mocker.patch(
            "iris.fileformats.pp._interpret_fields",
            autospec=True,
            return_value=iter([]),
        )
        extract_patch = mocker.patch(
            "iris.fileformats._ff.FF2PP._extract_field",
            autospec=True,
            return_value=extract_result,
        )

        FF2PP_instance = ff.FF2PP("mock")
        list(iter(FF2PP_instance))

        interpret_patch.assert_called_once_with(extract_result)
        extract_patch.assert_called_once_with(FF2PP_instance)


class Test__extract_field__LBC_format(MockerMixin):
    @contextlib.contextmanager
    def mock_for_extract_field(self, fields, x=None, y=None):
        """A context manager to ensure FF2PP._extract_field gets a field
        instance looking like the next one in the "fields" iterable from
        the "make_pp_field" call.

        """
        with self.mocker.patch("iris.fileformats._ff.FFHeader"):
            ff2pp = ff.FF2PP("mock")
        ff2pp._ff_header.lookup_table = [0, 0, len(fields)]
        # Fake level constants, with shape specifying just one model-level.
        ff2pp._ff_header.level_dependent_constants = np.zeros(1)
        grid = self.mocker.Mock()
        grid.vectors = self.mocker.Mock(return_value=(x, y))
        ff2pp._ff_header.grid = self.mocker.Mock(return_value=grid)

        open_func = "builtins.open"
        with (
            self.mocker.patch(
                "iris.fileformats._ff._parse_binary_stream", return_value=[0]
            ),
            self.mocker.patch(open_func),
            self.mocker.patch("struct.unpack_from", return_value=[4]),
            self.mocker.patch("iris.fileformats.pp.make_pp_field", side_effect=fields),
            self.mocker.patch(
                "iris.fileformats._ff.FF2PP._payload", return_value=(0, 0)
            ),
        ):
            yield ff2pp

    def _mock_lbc(self, **kwargs):
        """Return a Mock object representing an LBC field."""
        # Default kwargs for a valid LBC field mapping just 1 model-level.
        field_kwargs = dict(lbtim=0, lblev=7777, lbvc=0, lbhem=101)
        # Apply provided args (replacing any defaults if specified).
        field_kwargs.update(kwargs)
        # Return a mock with just those properties pre-defined.
        return self.mocker.Mock(**field_kwargs)

    def test_LBC_header(self):
        bzx, bzy = -10, 15
        # stash m01s00i001
        lbuser = [None, None, 121416, 1, None, None, 1]
        field = self._mock_lbc(
            lbegin=0,
            lbrow=10,
            lbnpt=12,
            bdx=1,
            bdy=1,
            bzx=bzx,
            bzy=bzy,
            lbuser=lbuser,
        )
        with self.mock_for_extract_field([field]) as ff2pp:
            ff2pp._ff_header.dataset_type = 5
            result = list(ff2pp._extract_field())

        assert [field] == result
        assert 10 + 14 * 2 == field.lbrow
        assert 12 + 16 * 2 == field.lbnpt

        name_mapping_dict = dict(
            rim_width=slice(4, 6), y_halo=slice(2, 4), x_halo=slice(0, 2)
        )
        boundary_packing = pp.SplittableInt(121416, name_mapping_dict)

        assert field.boundary_packing == boundary_packing
        assert field.bzy == bzy - boundary_packing.y_halo * field.bdy
        assert field.bzx == bzx - boundary_packing.x_halo * field.bdx

    def check_non_trivial_coordinate_warning(self, field):
        field.lbegin = 0
        field.lbrow = 10
        field.lbnpt = 12
        # stash m01s31i020
        field.lbuser = [None, None, 121416, 20, None, None, 1]
        orig_bdx, orig_bdy = field.bdx, field.bdy

        x = np.array([1, 2, 6])
        y = np.array([1, 2, 6])
        with self.mock_for_extract_field([field], x, y) as ff2pp:
            ff2pp._ff_header.dataset_type = 5
            msg = (
                "The x or y coordinates of your boundary condition field may "
                "be incorrect, not having taken into account the boundary "
                "size."
            )
            with pytest.warns(IrisLoadWarning, match=msg):
                list(ff2pp._extract_field())

        # Check the values are unchanged.
        assert field.bdy == orig_bdy
        assert field.bdx == orig_bdx

    def test_LBC_header_non_trivial_coords_both(self):
        # Check a warning is raised when both bdx and bdy are bad.
        field = self._mock_lbc(bdx=0, bdy=0, bzx=10, bzy=10)
        self.check_non_trivial_coordinate_warning(field)

        field.bdy = field.bdx = field.bmdi
        self.check_non_trivial_coordinate_warning(field)

    def test_LBC_header_non_trivial_coords_x(self):
        # Check a warning is raised when bdx is bad.
        field = self._mock_lbc(bdx=0, bdy=10, bzx=10, bzy=10)
        self.check_non_trivial_coordinate_warning(field)

        field.bdx = field.bmdi
        self.check_non_trivial_coordinate_warning(field)

    def test_LBC_header_non_trivial_coords_y(self):
        # Check a warning is raised when bdy is bad.
        field = self._mock_lbc(bdx=10, bdy=0, bzx=10, bzy=10)
        self.check_non_trivial_coordinate_warning(field)

        field.bdy = field.bmdi
        self.check_non_trivial_coordinate_warning(field)

    def test_negative_bdy(self):
        # Check a warning is raised when bdy is negative,
        # we don't yet know what "north" means in this case.
        field = self._mock_lbc(
            bdx=10,
            bdy=-10,
            bzx=10,
            bzy=10,
            lbegin=0,
            lbuser=[0, 0, 121416, 0, None, None, 0],
            lbrow=10,
            lbnpt=12,
        )
        with self.mock_for_extract_field([field]) as ff2pp:
            ff2pp._ff_header.dataset_type = 5
            msg = "The LBC has a bdy less than 0."
            with pytest.warns(IrisLoadWarning, match=msg):
                list(ff2pp._extract_field())


class Test__payload(MockerMixin):
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        # Create a mock LBC type PPField.
        self.mock_field = mocker.Mock()
        field = self.mock_field
        field.raw_lbpack = _UNPACKED
        field.lbuser = [_REAL]
        field.lblrec = 777
        field.lbext = 222
        field.lbnrec = 50
        field.boundary_packing = None

    def _test(self, mock_field, expected_depth, expected_dtype, word_depth=None):
        with self.mocker.patch("iris.fileformats._ff.FFHeader", return_value=None):
            kwargs = {}
            if word_depth is not None:
                kwargs["word_depth"] = word_depth
            ff2pp = FF2PP("dummy_filename", **kwargs)
            data_depth, data_dtype = ff2pp._payload(mock_field)
            assert data_depth == expected_depth
            assert data_dtype == expected_dtype

    def test_unpacked_real(self):
        mock_field = _DummyField(
            lbext=0,
            lblrec=100,
            lbnrec=-1,
            raw_lbpack=_UNPACKED,
            lbuser=[_REAL],
            boundary_packing=None,
        )
        self._test(mock_field, 800, ">f8")

    def test_unpacked_real_ext(self):
        mock_field = _DummyField(
            lbext=5,
            lblrec=100,
            lbnrec=-1,
            raw_lbpack=_UNPACKED,
            lbuser=[_REAL],
            boundary_packing=None,
        )
        self._test(mock_field, 760, ">f8")

    def test_unpacked_integer(self):
        mock_field = _DummyField(
            lbext=0,
            lblrec=200,
            lbnrec=-1,
            raw_lbpack=_UNPACKED,
            lbuser=[_INTEGER],
            boundary_packing=None,
        )
        self._test(mock_field, 1600, ">i8")

    def test_unpacked_integer_ext(self):
        mock_field = _DummyField(
            lbext=10,
            lblrec=200,
            lbnrec=-1,
            raw_lbpack=_UNPACKED,
            lbuser=[_INTEGER],
            boundary_packing=None,
        )
        self._test(mock_field, 1520, ">i8")

    def test_unpacked_real_ext_different_word_depth(self):
        mock_field = _DummyField(
            lbext=5,
            lblrec=100,
            lbnrec=-1,
            raw_lbpack=_UNPACKED,
            lbuser=[_REAL],
            boundary_packing=None,
        )
        self._test(mock_field, 380, ">f4", word_depth=4)

    def test_wgdos_real(self):
        mock_field = _DummyField(
            lbext=0,
            lblrec=-1,
            lbnrec=100,
            raw_lbpack=_WGDOS,
            lbuser=[_REAL],
            boundary_packing=None,
        )
        self._test(mock_field, 800, ">f4")

    def test_wgdos_real_ext(self):
        mock_field = _DummyField(
            lbext=5,
            lblrec=-1,
            lbnrec=100,
            raw_lbpack=_WGDOS,
            lbuser=[_REAL],
            boundary_packing=None,
        )
        self._test(mock_field, 800, ">f4")

    def test_wgdos_integer(self):
        mock_field = _DummyField(
            lbext=0,
            lblrec=-1,
            lbnrec=200,
            raw_lbpack=_WGDOS,
            lbuser=[_INTEGER],
            boundary_packing=None,
        )
        self._test(mock_field, 1600, ">i4")

    def test_wgdos_integer_ext(self):
        mock_field = _DummyField(
            lbext=10,
            lblrec=-1,
            lbnrec=200,
            raw_lbpack=_WGDOS,
            lbuser=[_INTEGER],
            boundary_packing=None,
        )
        self._test(mock_field, 1600, ">i4")

    def test_cray_real(self):
        mock_field = _DummyField(
            lbext=0,
            lblrec=100,
            lbnrec=-1,
            raw_lbpack=_CRAY,
            lbuser=[_REAL],
            boundary_packing=None,
        )
        self._test(mock_field, 400, ">f4")

    def test_cray_real_ext(self):
        mock_field = _DummyField(
            lbext=5,
            lblrec=100,
            lbnrec=-1,
            raw_lbpack=_CRAY,
            lbuser=[_REAL],
            boundary_packing=None,
        )
        self._test(mock_field, 380, ">f4")

    def test_cray_integer(self):
        mock_field = _DummyField(
            lbext=0,
            lblrec=200,
            lbnrec=-1,
            raw_lbpack=_CRAY,
            lbuser=[_INTEGER],
            boundary_packing=None,
        )
        self._test(mock_field, 800, ">i4")

    def test_cray_integer_ext(self):
        mock_field = _DummyField(
            lbext=10,
            lblrec=200,
            lbnrec=-1,
            raw_lbpack=_CRAY,
            lbuser=[_INTEGER],
            boundary_packing=None,
        )
        self._test(mock_field, 760, ">i4")

    def test_lbpack_unsupported(self):
        mock_field = _DummyField(
            lbext=10,
            lblrec=200,
            lbnrec=-1,
            raw_lbpack=1239,
            lbuser=[_INTEGER],
            boundary_packing=None,
        )
        with pytest.raises(
            NotYetImplementedError,
            match="PP fields with LBPACK of 1239 are not supported.",
        ):
            self._test(mock_field, None, None)

    def test_lbc_unpacked(self):
        boundary_packing = _DummyBoundaryPacking(x_halo=11, y_halo=7, rim_width=3)
        mock_field = _DummyFieldWithSize(
            lbext=10,
            lblrec=200,
            lbnrec=-1,
            raw_lbpack=_UNPACKED,
            lbuser=[_REAL],
            boundary_packing=boundary_packing,
            lbnpt=47,
            lbrow=34,
        )
        self._test(mock_field, ((47 * 34) - (19 * 14)) * 8, ">f8")

    def test_lbc_wgdos_unsupported(self):
        mock_field = _DummyField(
            lbext=5,
            lblrec=-1,
            lbnrec=100,
            raw_lbpack=_WGDOS,
            lbuser=[_REAL],
            # Anything not None will do here.
            boundary_packing=0,
        )
        with pytest.raises(ValueError, match="packed LBC data is not supported"):
            self._test(mock_field, None, None)

    def test_lbc_cray(self):
        boundary_packing = _DummyBoundaryPacking(x_halo=11, y_halo=7, rim_width=3)
        mock_field = _DummyFieldWithSize(
            lbext=10,
            lblrec=200,
            lbnrec=-1,
            raw_lbpack=_CRAY,
            lbuser=[_REAL],
            boundary_packing=boundary_packing,
            lbnpt=47,
            lbrow=34,
        )
        self._test(mock_field, ((47 * 34) - (19 * 14)) * 4, ">f4")


class Test__det_border:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        _FFH_patch = mocker.patch("iris.fileformats._ff.FFHeader")
        _FFH_patch.start()

    def test_unequal_spacing_eitherside(self, mocker):
        # Ensure that we do not interpret the case where there is not the same
        # spacing on the lower edge as the upper edge.
        ff2pp = FF2PP("dummy")
        field_x = np.array([1, 2, 10])

        msg = (
            "The x or y coordinates of your boundary condition field may "
            "be incorrect, not having taken into account the boundary "
            "size."
        )

        with pytest.warns(IrisLoadWarning, match=msg):
            result = ff2pp._det_border(field_x, None)
        assert result is field_x

    def test_increasing_field_values(self):
        # Field where its values a increasing.
        ff2pp = FF2PP("dummy")
        field_x = np.array([1, 2, 3])
        com = np.array([0, 1, 2, 3, 4])
        result = ff2pp._det_border(field_x, 1)
        _shared_utils.assert_array_equal(result, com)

    def test_decreasing_field_values(self):
        # Field where its values a decreasing.
        ff2pp = FF2PP("dummy")
        field_x = np.array([3, 2, 1])
        com = np.array([4, 3, 2, 1, 0])
        result = ff2pp._det_border(field_x, 1)
        _shared_utils.assert_array_equal(result, com)


class Test__adjust_field_for_lbc:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        # Patch FFHeader to produce a mock header instead of opening a file.
        self.mock_ff_header = mocker.Mock()
        self.mock_ff_header.dataset_type = 5
        self.mock_ff = mocker.patch(
            "iris.fileformats._ff.FFHeader", return_value=self.mock_ff_header
        )

        # Create a mock LBC type PPField.
        self.mock_field = mocker.Mock()
        field = self.mock_field
        field.lbtim = 0
        field.lblev = 7777
        field.lbvc = 0
        field.lbnpt = 1001
        field.lbrow = 2001
        field.lbuser = (None, None, 80504)
        field.lbpack = pp.SplittableInt(0)
        field.boundary_packing = None
        field.bdx = 1.0
        field.bzx = 0.0
        field.bdy = 1.0
        field.bzy = 0.0

    def test__basic(self):
        ff2pp = FF2PP("dummy_filename")
        field = self.mock_field
        ff2pp._adjust_field_for_lbc(field)
        assert field.lbtim == 11
        assert field.lbvc == 65
        assert field.boundary_packing.rim_width == 8
        assert field.boundary_packing.y_halo == 5
        assert field.boundary_packing.x_halo == 4
        assert field.lbnpt == 1009
        assert field.lbrow == 2011

    def test__bad_lbtim(self):
        self.mock_field.lbtim = 717
        ff2pp = FF2PP("dummy_filename")
        with pytest.raises(ValueError, match="LBTIM of 717, expected only 0 or 11"):
            ff2pp._adjust_field_for_lbc(self.mock_field)

    def test__bad_lbvc(self):
        self.mock_field.lbvc = 312
        ff2pp = FF2PP("dummy_filename")
        with pytest.raises(ValueError, match="LBVC of 312, expected only 0 or 65"):
            ff2pp._adjust_field_for_lbc(self.mock_field)


class Test__fields_over_all_levels:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        # Patch FFHeader to produce a mock header instead of opening a file.
        self.mock_ff_header = mocker.Mock()
        self.mock_ff_header.dataset_type = 5

        # Fake the level constants to look like 3 model levels.
        self.n_all_levels = 3
        self.mock_ff_header.level_dependent_constants = np.zeros((self.n_all_levels))
        self.mock_ff = mocker.patch(
            "iris.fileformats._ff.FFHeader", return_value=self.mock_ff_header
        )

        # Create a simple mock for a test field.
        self.mock_field = mocker.Mock()
        field = self.mock_field
        field.lbhem = 103
        self.original_lblev = mocker.sentinel.untouched_lbev
        field.lblev = self.original_lblev

    def _check_expected_levels(self, results, n_levels):
        if n_levels == 0:
            assert len(results) == 1
            assert results[0].lblev == self.original_lblev
        else:
            assert len(results) == n_levels
            assert [fld.lblev for fld in results] == list(range(n_levels))

    def test__is_lbc(self):
        ff2pp = FF2PP("dummy_filename")
        field = self.mock_field
        results = list(ff2pp._fields_over_all_levels(field))
        self._check_expected_levels(results, 3)

    def test__lbhem_too_small(self):
        ff2pp = FF2PP("dummy_filename")
        field = self.mock_field
        field.lbhem = 100
        with pytest.raises(ValueError, match="hence >= 101"):
            _ = list(ff2pp._fields_over_all_levels(field))

    def test__lbhem_too_large(self):
        ff2pp = FF2PP("dummy_filename")
        field = self.mock_field
        field.lbhem = 105
        with pytest.raises(
            ValueError, match="more than the total number of levels in the file = 3"
        ):
            _ = list(ff2pp._fields_over_all_levels(field))
