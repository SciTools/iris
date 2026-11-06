# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.pp.PPField` class."""

import numpy as np
import pytest

import iris.fileformats.pp as pp
from iris.fileformats.pp import PPField, SplittableInt
from iris.tests import _shared_utils
from iris.tests.unit.fileformats import MockerMixin
from iris.warnings import IrisDefaultingWarning, IrisMaskValueMatchWarning

# The PPField class is abstract, so to test we define a minimal,
# concrete subclass with the `t1` and `t2` properties.
#
# NB. We define dummy header items to allow us to zero the unused header
# items when written to disk and get consistent results.


DUMMY_HEADER = [
    ("dummy1", (0, 11)),
    ("lbtim", (12,)),
    ("dummy2", (13,)),
    ("lblrec", (14,)),
    ("dummy3", (15, 16)),
    ("lbrow", (17,)),
    ("dummy4", (18,)),
    ("lbext", (19,)),
    ("lbpack", (20,)),
    ("dummy5", (21, 37)),
    ("lbuser", (38, 39, 40, 41, 42, 43, 44)),
    ("brsvd", (45, 46, 47, 48)),
    ("bdatum", (49,)),
    ("dummy6", (50, 61)),
    ("bmdi", (62,)),
    ("dummy7", (63,)),
]


class DummyPPField(PPField):
    HEADER_DEFN = DUMMY_HEADER
    HEADER_DICT = dict(DUMMY_HEADER)

    def _ready_for_save(self):
        self.dummy1 = 0
        self.dummy2 = 0
        self.dummy3 = 0
        self.dummy4 = 0
        self.dummy5 = 0
        self.dummy6 = 0
        self.dummy7 = 0
        self.lbtim = 0
        self.lblrec = 0
        self.lbrow = 0
        self.lbext = 0
        self.lbpack = 0
        self.lbuser = 0
        self.brsvd = 0
        self.bdatum = 0
        self.bmdi = -1e30
        return self

    @property
    def t1(self):
        return None

    @property
    def t2(self):
        return None


class Test_save:
    def test_float64(self, tmp_path):
        # Tests down-casting of >f8 data to >f4.

        def field_checksum(data):
            field = DummyPPField()._ready_for_save()
            field.data = data
            temp_filename = tmp_path / "temp.pp"
            with open(temp_filename, "wb") as pp_file:
                field.save(pp_file)
            checksum = _shared_utils.file_checksum(temp_filename)
            return checksum

        data_64 = np.linspace(0, 1, num=10, endpoint=False).reshape(2, 5)
        checksum_32 = field_checksum(data_64.astype(">f4"))
        msg = "Downcasting array precision from float64 to float32 for save."
        with pytest.warns(IrisDefaultingWarning, match=msg):
            checksum_64 = field_checksum(data_64.astype(">f8"))
        assert checksum_32 == checksum_64

    def test_masked_mdi_value_warning(self, tmp_path):
        # Check that an unmasked MDI value raises a warning.
        field = DummyPPField()._ready_for_save()
        # Make float32 data, as float64 default produces an extra warning.
        field.bmdi = np.float32(-123.4)
        field.data = np.ma.masked_array([1.0, field.bmdi, 3.0], dtype=np.float32)
        msg = "PPField data contains unmasked points"
        temp_filename = tmp_path / "temp.pp"
        with pytest.warns(IrisMaskValueMatchWarning, match=msg):
            with open(temp_filename, "wb") as pp_file:
                field.save(pp_file)

    def test_unmasked_mdi_value_warning(self, tmp_path):
        # Check that MDI in *unmasked* data raises a warning.
        field = DummyPPField()._ready_for_save()
        field.bmdi = -123.4
        # Make float32 data, as float64 default produces an extra warning.
        field.data = np.array([1.0, field.bmdi, 3.0], dtype=np.float32)
        msg = "PPField data contains unmasked points"
        temp_filename = tmp_path / "temp.pp"
        with pytest.warns(IrisMaskValueMatchWarning, match=msg):
            with open(temp_filename, "wb") as pp_file:
                field.save(pp_file)

    def test_mdi_masked_value_nowarning(self, tmp_path):
        # Check that a *masked* MDI value does not raise a warning.
        field = DummyPPField()._ready_for_save()
        field.bmdi = -123.4
        # Make float32 data, as float64 default produces an extra warning.
        field.data = np.ma.masked_array(
            [1.0, 2.0, 3.0], mask=[0, 1, 0], dtype=np.float32
        )
        # Set underlying data value at masked point to BMDI value.
        field.data.data[1] = field.bmdi
        _shared_utils.assert_array_all_close(field.data.data[1], field.bmdi)
        with _shared_utils.assert_no_warnings_regexp(r"\(mask\|fill\)"):
            temp_filename = tmp_path / "temp.pp"
            with open(temp_filename, "wb") as pp_file:
                field.save(pp_file)


class Test_calendar:
    def test_greg(self):
        field = DummyPPField()
        field.lbtim = SplittableInt(1, {"ia": 2, "ib": 1, "ic": 0})
        assert field.calendar == "standard"

    def test_360(self):
        field = DummyPPField()
        field.lbtim = SplittableInt(2, {"ia": 2, "ib": 1, "ic": 0})
        assert field.calendar == "360_day"

    def test_365(self):
        field = DummyPPField()
        field.lbtim = SplittableInt(4, {"ia": 2, "ib": 1, "ic": 0})
        assert field.calendar == "365_day"


class Test_coord_system(MockerMixin):
    def _check_cs(self, bplat, bplon, rotated):
        field = DummyPPField()
        field.bplat = bplat
        field.bplon = bplon
        mock_cs_mod = self.mocker.patch("iris.fileformats.pp.iris.coord_systems")
        result = field.coord_system()
        if not rotated:
            # It should return a standard unrotated CS.
            assert mock_cs_mod.GeogCS.call_count == 1
            assert result == mock_cs_mod.GeogCS()
        else:
            # It should return a rotated CS with the correct makeup.
            assert mock_cs_mod.GeogCS.call_count == 1
            assert mock_cs_mod.RotatedGeogCS.call_count == 1
            assert result == mock_cs_mod.RotatedGeogCS()
            assert mock_cs_mod.RotatedGeogCS.call_args_list[0] == self.mocker.call(
                bplat, bplon, ellipsoid=mock_cs_mod.GeogCS()
            )

    def test_normal_unrotated(self):
        # Check that 'normal' BPLAT,BPLON=90,0 produces an unrotated system.
        self._check_cs(bplat=90, bplon=0, rotated=False)

    def test_bplon_180_unrotated(self):
        # Check that BPLAT,BPLON=90,180 behaves the same as 90,0.
        self._check_cs(bplat=90, bplon=180, rotated=False)

    def test_odd_bplat_rotated(self):
        # Show that BPLAT != 90 produces a rotated field.
        self._check_cs(bplat=75, bplon=180, rotated=True)

    def test_odd_bplon_rotated(self):
        # Show that BPLON != 0 or 180 produces a rotated field.
        self._check_cs(bplat=90, bplon=123.45, rotated=True)


class Test__init__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        header_longs = np.zeros(pp.NUM_LONG_HEADERS, dtype=np.int_)
        header_floats = np.zeros(pp.NUM_FLOAT_HEADERS, dtype=np.float64)
        self.header = list(header_longs) + list(header_floats)

    def test_no_headers(self):
        field = DummyPPField()
        assert field._raw_header is None
        assert field.raw_lbtim is None
        assert field.raw_lbpack is None

    def test_lbtim_lookup(self):
        assert DummyPPField.HEADER_DICT["lbtim"] == (12,)

    def test_lbpack_lookup(self):
        assert DummyPPField.HEADER_DICT["lbpack"] == (20,)

    def test_raw_lbtim(self):
        raw_lbtim = 4321
        (loc,) = DummyPPField.HEADER_DICT["lbtim"]
        self.header[loc] = raw_lbtim
        field = DummyPPField(header=self.header)
        assert field.raw_lbtim == raw_lbtim

    def test_raw_lbpack(self):
        raw_lbpack = 4321
        (loc,) = DummyPPField.HEADER_DICT["lbpack"]
        self.header[loc] = raw_lbpack
        field = DummyPPField(header=self.header)
        assert field.raw_lbpack == raw_lbpack


class Test__getattr__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        header_longs = np.zeros(pp.NUM_LONG_HEADERS, dtype=np.int_)
        header_floats = np.zeros(pp.NUM_FLOAT_HEADERS, dtype=np.float64)
        self.header = list(header_longs) + list(header_floats)

    def test_attr_singular_long(self):
        lbrow = 1234
        (loc,) = DummyPPField.HEADER_DICT["lbrow"]
        self.header[loc] = lbrow
        field = DummyPPField(header=self.header)
        assert field.lbrow == lbrow

    def test_attr_multi_long(self):
        lbuser = (100, 101, 102, 103, 104, 105, 106)
        loc = DummyPPField.HEADER_DICT["lbuser"]
        self.header[loc[0] : loc[-1] + 1] = lbuser
        field = DummyPPField(header=self.header)
        assert field.lbuser == lbuser

    def test_attr_singular_float(self):
        bdatum = 1234
        (loc,) = DummyPPField.HEADER_DICT["bdatum"]
        self.header[loc] = bdatum
        field = DummyPPField(header=self.header)
        assert field.bdatum == bdatum

    def test_attr_multi_float(self):
        brsvd = (100, 101, 102, 103)
        loc = DummyPPField.HEADER_DICT["brsvd"]
        start = loc[0]
        stop = loc[-1] + 1
        self.header[start:stop] = brsvd
        field = DummyPPField(header=self.header)
        assert field.brsvd == brsvd

    def test_attr_lbtim(self):
        raw_lbtim = 4321
        (loc,) = DummyPPField.HEADER_DICT["lbtim"]
        self.header[loc] = raw_lbtim
        field = DummyPPField(header=self.header)
        result = field.lbtim
        assert result == raw_lbtim
        assert isinstance(result, SplittableInt)
        result = field._lbtim
        assert result == raw_lbtim
        assert isinstance(result, SplittableInt)

    def test_attr_lbpack(self):
        raw_lbpack = 4321
        (loc,) = DummyPPField.HEADER_DICT["lbpack"]
        self.header[loc] = raw_lbpack
        field = DummyPPField(header=self.header)
        result = field.lbpack
        assert result == raw_lbpack
        assert isinstance(result, SplittableInt)
        result = field._lbpack
        assert result == raw_lbpack
        assert isinstance(result, SplittableInt)

    def test_attr_raw_lbtim_assign(self):
        field = DummyPPField(header=self.header)
        assert field.raw_lbpack == 0
        assert field.lbtim == 0
        raw_lbtim = 4321
        field.lbtim = raw_lbtim
        assert field.raw_lbtim == raw_lbtim
        assert not isinstance(field.raw_lbtim, SplittableInt)

    def test_attr_raw_lbpack_assign(self):
        field = DummyPPField(header=self.header)
        assert field.raw_lbpack == 0
        assert field.lbpack == 0
        raw_lbpack = 4321
        field.lbpack = raw_lbpack
        assert field.raw_lbpack == raw_lbpack
        assert not isinstance(field.raw_lbpack, SplittableInt)

    def test_attr_unknown(self):
        with pytest.raises(
            AttributeError, match="'DummyPPField' object has no attribute 'x'"
        ):
            DummyPPField().x


class Test_lbtim:
    def test_get_splittable(self):
        headers = [0] * 64
        headers[12] = 12345
        field = DummyPPField(headers)
        assert isinstance(field.lbtim, SplittableInt)
        assert field.lbtim.ia == 123
        assert field.lbtim.ib == 4
        assert field.lbtim.ic == 5

    def test_set_int(self):
        headers = [0] * 64
        headers[12] = 12345
        field = DummyPPField(headers)
        field.lbtim = 34567
        assert isinstance(field.lbtim, SplittableInt)
        assert field.lbtim.ia == 345
        assert field.lbtim.ib == 6
        assert field.lbtim.ic == 7
        assert field.raw_lbtim == 34567

    def test_set_splittable(self):
        # Check that assigning a SplittableInt to lbtim uses the integer
        # value. In other words, check that you can't assign an
        # arbitrary SplittableInt with crazy named attributes.
        headers = [0] * 64
        headers[12] = 12345
        field = DummyPPField(headers)
        si = SplittableInt(34567, {"foo": 0})
        field.lbtim = si
        assert isinstance(field.lbtim, SplittableInt)
        with pytest.raises(
            AttributeError, match="'SplittableInt' object has no attribute 'foo'"
        ):
            field.lbtim.foo
        assert field.lbtim.ia == 345
        assert field.lbtim.ib == 6
        assert field.lbtim.ic == 7
        assert field.raw_lbtim == 34567
