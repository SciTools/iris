# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.fileformats.pp.PPField` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np

import iris.fileformats.pp as pp
from iris.fileformats.pp import PPField, SplittableInt

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


class Test_save(tests.IrisTest):
    def test_float64(self):
        # Tests down-casting of >f8 data to >f4.

        def field_checksum(data):
            field = DummyPPField()._ready_for_save()
            field.data = data
            with self.temp_filename(".pp") as temp_filename:
                with open(temp_filename, "wb") as pp_file:
                    field.save(pp_file)
                checksum = self.file_checksum(temp_filename)
            return checksum

        data_64 = np.linspace(0, 1, num=10, endpoint=False).reshape(2, 5)
        checksum_32 = field_checksum(data_64.astype(">f4"))
        msg = "Downcasting array precision from float64 to float32 for save."
        with self.assertWarnsRegex(UserWarning, msg):
            checksum_64 = field_checksum(data_64.astype(">f8"))
        self.assertEqual(checksum_32, checksum_64)

    def test_masked_mdi_value_warning(self):
        # Check that an unmasked MDI value raises a warning.
        field = DummyPPField()._ready_for_save()
        field.bmdi = -123.4
        # Make float32 data, as float64 default produces an extra warning.
        field.data = np.ma.masked_array(
            [1.0, field.bmdi, 3.0], dtype=np.float32
        )
        msg = "PPField data contains unmasked points"
        with self.assertWarnsRegex(UserWarning, msg):
            with self.temp_filename(".pp") as temp_filename:
                with open(temp_filename, "wb") as pp_file:
                    field.save(pp_file)

    def test_unmasked_mdi_value_warning(self):
        # Check that MDI in *unmasked* data raises a warning.
        field = DummyPPField()._ready_for_save()
        field.bmdi = -123.4
        # Make float32 data, as float64 default produces an extra warning.
        field.data = np.array([1.0, field.bmdi, 3.0], dtype=np.float32)
        msg = "PPField data contains unmasked points"
        with self.assertWarnsRegex(UserWarning, msg):
            with self.temp_filename(".pp") as temp_filename:
                with open(temp_filename, "wb") as pp_file:
                    field.save(pp_file)

    def test_mdi_masked_value_nowarning(self):
        # Check that a *masked* MDI value does not raise a warning.
        field = DummyPPField()._ready_for_save()
        field.bmdi = -123.4
        # Make float32 data, as float64 default produces an extra warning.
        field.data = np.ma.masked_array(
            [1.0, 2.0, 3.0], mask=[0, 1, 0], dtype=np.float32
        )
        # Set underlying data value at masked point to BMDI value.
        field.data.data[1] = field.bmdi
        self.assertArrayAllClose(field.data.data[1], field.bmdi)
        with self.assertNoWarningsRegexp(r"\(mask\|fill\)"):
            with self.temp_filename(".pp") as temp_filename:
                with open(temp_filename, "wb") as pp_file:
                    field.save(pp_file)


class Test_calendar(tests.IrisTest):
    def test_greg(self):
        field = DummyPPField()
        field.lbtim = SplittableInt(1, {"ia": 2, "ib": 1, "ic": 0})
        self.assertEqual(field.calendar, "standard")

    def test_360(self):
        field = DummyPPField()
        field.lbtim = SplittableInt(2, {"ia": 2, "ib": 1, "ic": 0})
        self.assertEqual(field.calendar, "360_day")

    def test_365(self):
        field = DummyPPField()
        field.lbtim = SplittableInt(4, {"ia": 2, "ib": 1, "ic": 0})
        self.assertEqual(field.calendar, "365_day")


class Test_coord_system(tests.IrisTest):
    def _check_cs(self, bplat, bplon, rotated):
        field = DummyPPField()
        field.bplat = bplat
        field.bplon = bplon
        with mock.patch(
            "iris.fileformats.pp.iris.coord_systems"
        ) as mock_cs_mod:
            result = field.coord_system()
        if not rotated:
            # It should return a standard unrotated CS.
            self.assertTrue(mock_cs_mod.GeogCS.call_count == 1)
            self.assertEqual(result, mock_cs_mod.GeogCS())
        else:
            # It should return a rotated CS with the correct makeup.
            self.assertTrue(mock_cs_mod.GeogCS.call_count == 1)
            self.assertTrue(mock_cs_mod.RotatedGeogCS.call_count == 1)
            self.assertEqual(result, mock_cs_mod.RotatedGeogCS())
            self.assertEqual(
                mock_cs_mod.RotatedGeogCS.call_args_list[0],
                mock.call(bplat, bplon, ellipsoid=mock_cs_mod.GeogCS()),
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


class Test__init__(tests.IrisTest):
    def setUp(self):
        header_longs = np.zeros(pp.NUM_LONG_HEADERS, dtype=np.int_)
        header_floats = np.zeros(pp.NUM_FLOAT_HEADERS, dtype=np.float64)
        self.header = list(header_longs) + list(header_floats)

    def test_no_headers(self):
        field = DummyPPField()
        self.assertIsNone(field._raw_header)
        self.assertIsNone(field.raw_lbtim)
        self.assertIsNone(field.raw_lbpack)

    def test_lbtim_lookup(self):
        self.assertEqual(DummyPPField.HEADER_DICT["lbtim"], (12,))

    def test_lbpack_lookup(self):
        self.assertEqual(DummyPPField.HEADER_DICT["lbpack"], (20,))

    def test_raw_lbtim(self):
        raw_lbtim = 4321
        (loc,) = DummyPPField.HEADER_DICT["lbtim"]
        self.header[loc] = raw_lbtim
        field = DummyPPField(header=self.header)
        self.assertEqual(field.raw_lbtim, raw_lbtim)

    def test_raw_lbpack(self):
        raw_lbpack = 4321
        (loc,) = DummyPPField.HEADER_DICT["lbpack"]
        self.header[loc] = raw_lbpack
        field = DummyPPField(header=self.header)
        self.assertEqual(field.raw_lbpack, raw_lbpack)


class Test__getattr__(tests.IrisTest):
    def setUp(self):
        header_longs = np.zeros(pp.NUM_LONG_HEADERS, dtype=np.int_)
        header_floats = np.zeros(pp.NUM_FLOAT_HEADERS, dtype=np.float64)
        self.header = list(header_longs) + list(header_floats)

    def test_attr_singular_long(self):
        lbrow = 1234
        (loc,) = DummyPPField.HEADER_DICT["lbrow"]
        self.header[loc] = lbrow
        field = DummyPPField(header=self.header)
        self.assertEqual(field.lbrow, lbrow)

    def test_attr_multi_long(self):
        lbuser = (100, 101, 102, 103, 104, 105, 106)
        loc = DummyPPField.HEADER_DICT["lbuser"]
        self.header[loc[0] : loc[-1] + 1] = lbuser
        field = DummyPPField(header=self.header)
        self.assertEqual(field.lbuser, lbuser)

    def test_attr_singular_float(self):
        bdatum = 1234
        (loc,) = DummyPPField.HEADER_DICT["bdatum"]
        self.header[loc] = bdatum
        field = DummyPPField(header=self.header)
        self.assertEqual(field.bdatum, bdatum)

    def test_attr_multi_float(self):
        brsvd = (100, 101, 102, 103)
        loc = DummyPPField.HEADER_DICT["brsvd"]
        start = loc[0]
        stop = loc[-1] + 1
        self.header[start:stop] = brsvd
        field = DummyPPField(header=self.header)
        self.assertEqual(field.brsvd, brsvd)

    def test_attr_lbtim(self):
        raw_lbtim = 4321
        (loc,) = DummyPPField.HEADER_DICT["lbtim"]
        self.header[loc] = raw_lbtim
        field = DummyPPField(header=self.header)
        result = field.lbtim
        self.assertEqual(result, raw_lbtim)
        self.assertIsInstance(result, SplittableInt)
        result = field._lbtim
        self.assertEqual(result, raw_lbtim)
        self.assertIsInstance(result, SplittableInt)

    def test_attr_lbpack(self):
        raw_lbpack = 4321
        (loc,) = DummyPPField.HEADER_DICT["lbpack"]
        self.header[loc] = raw_lbpack
        field = DummyPPField(header=self.header)
        result = field.lbpack
        self.assertEqual(result, raw_lbpack)
        self.assertIsInstance(result, SplittableInt)
        result = field._lbpack
        self.assertEqual(result, raw_lbpack)
        self.assertIsInstance(result, SplittableInt)

    def test_attr_raw_lbtim_assign(self):
        field = DummyPPField(header=self.header)
        self.assertEqual(field.raw_lbpack, 0)
        self.assertEqual(field.lbtim, 0)
        raw_lbtim = 4321
        field.lbtim = raw_lbtim
        self.assertEqual(field.raw_lbtim, raw_lbtim)
        self.assertNotIsInstance(field.raw_lbtim, SplittableInt)

    def test_attr_raw_lbpack_assign(self):
        field = DummyPPField(header=self.header)
        self.assertEqual(field.raw_lbpack, 0)
        self.assertEqual(field.lbpack, 0)
        raw_lbpack = 4321
        field.lbpack = raw_lbpack
        self.assertEqual(field.raw_lbpack, raw_lbpack)
        self.assertNotIsInstance(field.raw_lbpack, SplittableInt)

    def test_attr_unknown(self):
        with self.assertRaises(AttributeError):
            DummyPPField().x


class Test_lbtim(tests.IrisTest):
    def test_get_splittable(self):
        headers = [0] * 64
        headers[12] = 12345
        field = DummyPPField(headers)
        self.assertIsInstance(field.lbtim, SplittableInt)
        self.assertEqual(field.lbtim.ia, 123)
        self.assertEqual(field.lbtim.ib, 4)
        self.assertEqual(field.lbtim.ic, 5)

    def test_set_int(self):
        headers = [0] * 64
        headers[12] = 12345
        field = DummyPPField(headers)
        field.lbtim = 34567
        self.assertIsInstance(field.lbtim, SplittableInt)
        self.assertEqual(field.lbtim.ia, 345)
        self.assertEqual(field.lbtim.ib, 6)
        self.assertEqual(field.lbtim.ic, 7)
        self.assertEqual(field.raw_lbtim, 34567)

    def test_set_splittable(self):
        # Check that assigning a SplittableInt to lbtim uses the integer
        # value. In other words, check that you can't assign an
        # arbitrary SplittableInt with crazy named attributes.
        headers = [0] * 64
        headers[12] = 12345
        field = DummyPPField(headers)
        si = SplittableInt(34567, {"foo": 0})
        field.lbtim = si
        self.assertIsInstance(field.lbtim, SplittableInt)
        with self.assertRaises(AttributeError):
            field.lbtim.foo
        self.assertEqual(field.lbtim.ia, 345)
        self.assertEqual(field.lbtim.ib, 6)
        self.assertEqual(field.lbtim.ic, 7)
        self.assertEqual(field.raw_lbtim, 34567)


if __name__ == "__main__":
    tests.main()
