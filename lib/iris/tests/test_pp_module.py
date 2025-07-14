# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

from copy import deepcopy
import os
from types import GeneratorType

import cftime
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from iris.cube import NP_PRINTOPTIONS_LEGACY
import iris.fileformats.pp as pp
from iris.tests import _shared_utils


@_shared_utils.skip_data
class TestPPCopy:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.filename = _shared_utils.get_data_path(("PP", "aPPglob1", "global.pp"))

    def test_copy_field_deferred(self):
        field = next(pp.load(self.filename))
        clone = field.copy()
        assert field == clone
        clone.lbyr = 666
        assert field != clone

    def test_deepcopy_field_deferred(self):
        field = next(pp.load(self.filename))
        clone = deepcopy(field)
        assert field == clone
        clone.lbyr = 666
        assert field != clone

    def test_copy_field_non_deferred(self):
        field = next(pp.load(self.filename, True))
        clone = field.copy()
        assert field == clone
        clone.data[0][0] = 666
        assert field != clone

    def test_deepcopy_field_non_deferred(self):
        field = next(pp.load(self.filename, True))
        clone = deepcopy(field)
        assert field == clone
        clone.data[0][0] = 666
        assert field != clone


class IrisPPTest:
    def check_pp(self, pp_fields, reference_filename):
        """Checks the given iterable of PPField objects matches the reference file, or creates the
        reference file if it doesn't exist.

        """
        # turn the generator into a list
        pp_fields = list(pp_fields)

        # Load deferred data for all of the fields (but don't do anything with it)
        for pp_field in pp_fields:
            pp_field.data

        with np.printoptions(legacy=NP_PRINTOPTIONS_LEGACY):
            test_string = str(pp_fields)
        reference_path = _shared_utils.get_result_path(reference_filename)
        if os.path.isfile(reference_path):
            with open(reference_path, "r") as reference_fh:
                reference = "".join(reference_fh.readlines())
            _shared_utils._assert_str_same(
                reference + "\n",
                test_string + "\n",
                reference_filename,
                type_comparison_name="PP files",
            )
        else:
            with open(reference_path, "w") as reference_fh:
                reference_fh.writelines(test_string)


class TestPPHeaderDerived:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pp = pp.PPField2()
        self.pp.lbuser = (0, 1, 2, 3, 4, 5, 6)
        self.pp.lbtim = 11
        self.pp.lbproc = 65539

    def test_standard_access(self):
        assert self.pp.lbtim == 11

    def test_lbtim_access(self):
        assert self.pp.lbtim[0] == 1
        assert self.pp.lbtim.ic == 1

    def test_lbtim_setter(self):
        self.pp.lbtim[4] = 4
        self.pp.lbtim[0] = 4
        assert self.pp.lbtim[0] == 4
        assert self.pp.lbtim.ic == 4

        self.pp.lbtim.ib = 9
        assert self.pp.lbtim.ib == 9
        assert self.pp.lbtim[1] == 9

    def test_set_lbuser(self):
        self.pp.stash = "m02s12i003"
        assert self.pp.stash == pp.STASH(2, 12, 3)
        self.pp.lbuser[6] = 5
        assert self.pp.stash == pp.STASH(5, 12, 3)
        self.pp.lbuser[3] = 4321
        assert self.pp.stash == pp.STASH(5, 4, 321)

    def test_set_stash(self):
        self.pp.stash = "m02s12i003"
        assert self.pp.stash == pp.STASH(2, 12, 3)

        self.pp.stash = pp.STASH(3, 13, 4)
        assert self.pp.stash == pp.STASH(3, 13, 4)
        assert self.pp.lbuser[3] == self.pp.stash.lbuser3()
        assert self.pp.lbuser[6] == self.pp.stash.lbuser6()

        with pytest.raises(ValueError):
            self.pp.stash = (4, 15, 5)

    def test_lbproc_bad_access(self):
        try:
            print(self.pp.lbproc.flag65537)
        except AttributeError:
            pass
        except Exception as err:
            pytest.fail("Should return a better error: " + str(err))


@_shared_utils.skip_data
class TestPPField_GlobalTemperature(IrisPPTest):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.original_pp_filepath = _shared_utils.get_data_path(
            ("PP", "aPPglob1", "global.pp")
        )
        self.r = list(pp.load(self.original_pp_filepath))

    def test_full_file(self):
        self.check_pp(self.r[0:10], ("PP", "global_test.pp.txt"))

    def test_lbtim_access(self):
        assert self.r[0].lbtim[0] == 2
        assert self.r[0].lbtim.ic == 2

    def test_t1_t2_access(self):
        field = self.r[0]
        calendar = "360_day"
        assert (
            field.t1.timetuple()
            == cftime.datetime(1994, 12, 1, 0, 0, calendar=calendar).timetuple()
        )

    def test_save_single(self, tmp_path):
        temp_filename = tmp_path / "foo.pp"
        with open(temp_filename, "wb") as temp_fh:
            self.r[0].save(temp_fh)
        assert _shared_utils.file_checksum(
            temp_filename
        ) == _shared_utils.file_checksum(self.original_pp_filepath)

    def test_save_api(self, tmp_path):
        filepath = self.original_pp_filepath

        f = next(pp.load(filepath))

        temp_filename = tmp_path / "foo.pp"

        with open(temp_filename, "wb") as temp_fh:
            f.save(temp_fh)
        assert _shared_utils.file_checksum(
            temp_filename
        ) == _shared_utils.file_checksum(filepath)


@_shared_utils.skip_data
class TestPackedPP(IrisPPTest):
    def test_wgdos(self, mocker, tmp_path):
        filepath = _shared_utils.get_data_path(
            ("PP", "wgdos_packed", "nae.20100104-06_0001.pp")
        )
        r = pp.load(filepath)

        # Check that the result is a generator and convert to a list so that we
        # can index and get the first one
        assert isinstance(r, GeneratorType)
        r = list(r)

        self.check_pp(r, ("PP", "nae_unpacked.pp.txt"))

        # check that trying to save this field again raises an error
        # (we cannot currently write WGDOS packed fields without mo_pack)
        temp_filename = tmp_path / "foo.pp"
        mocker.patch("iris.fileformats.pp.mo_pack", None)
        with pytest.raises(NotImplementedError):
            with open(temp_filename, "wb") as temp_fh:
                r[0].save(temp_fh)

    @pytest.mark.skipif(pp.mo_pack is None, reason="Requires mo_pack.")
    def test_wgdos_mo_pack(self, tmp_path):
        filepath = _shared_utils.get_data_path(
            ("PP", "wgdos_packed", "nae.20100104-06_0001.pp")
        )
        orig_fields = pp.load(filepath)
        temp_filename = tmp_path / "foo.pp"
        with open(temp_filename, "wb") as fh:
            for field in orig_fields:
                field.save(fh)
        saved_fields = pp.load(temp_filename)
        for orig_field, saved_field in zip(orig_fields, saved_fields):
            assert_array_equal(orig_field.data, saved_field.data)

    def test_rle(self, tmp_path):
        r = pp.load(_shared_utils.get_data_path(("PP", "ocean_rle", "ocean_rle.pp")))

        # Check that the result is a generator and convert to a list so that we
        # can index and get the first one
        assert isinstance(r, GeneratorType)
        r = list(r)

        self.check_pp(r, ("PP", "rle_unpacked.pp.txt"))

        # check that trying to save this field again raises an error
        # (we cannot currently write RLE packed fields)
        temp_filename = tmp_path / "foo.pp"
        with pytest.raises(NotImplementedError):
            with open(temp_filename, "wb") as temp_fh:
                r[0].save(temp_fh)


@_shared_utils.skip_data
class TestPPFile(IrisPPTest):
    def test_lots_of_extra_data(self):
        r = pp.load(
            _shared_utils.get_data_path(
                ("PP", "cf_processing", "HadCM2_ts_SAT_ann_18602100.b.pp")
            )
        )
        r = list(r)
        assert r[0].lbcode.ix == 13
        assert r[0].lbcode.iy == 23
        assert len(r[0].lbcode) == 5
        self.check_pp(r, ("PP", "extra_data_time_series.pp.txt"))


@_shared_utils.skip_data
class TestPPFileExtraXData(IrisPPTest):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.original_pp_filepath = _shared_utils.get_data_path(
            ("PP", "ukV1", "ukVpmslont.pp")
        )
        self.r = list(pp.load(self.original_pp_filepath))[0:5]

    def test_full_file(self):
        self.check_pp(self.r, ("PP", "extra_x_data.pp.txt"))

    def test_save_single(self, tmp_path):
        filepath = _shared_utils.get_data_path(
            ("PP", "ukV1", "ukVpmslont_first_field.pp")
        )
        f = next(pp.load(filepath))

        temp_filename = tmp_path / "foo.pp"
        with open(temp_filename, "wb") as temp_fh:
            f.save(temp_fh)

        s = next(pp.load(temp_filename))

        # force the data to be loaded (this was done for f when save was run)
        s.data
        _shared_utils._assert_str_same(
            str(s) + "\n", str(f) + "\n", "", type_comparison_name="PP files"
        )

        assert _shared_utils.file_checksum(
            temp_filename
        ) == _shared_utils.file_checksum(filepath)


@_shared_utils.skip_data
class TestPPFileWithExtraCharacterData(IrisPPTest):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.original_pp_filepath = _shared_utils.get_data_path(
            ("PP", "globClim1", "dec_subset.pp")
        )
        self.r = pp.load(self.original_pp_filepath)
        self.r_loaded_data = pp.load(self.original_pp_filepath, read_data=True)

        # Check that the result is a generator and convert to a list so that we can index and get the first one
        assert isinstance(self.r, GeneratorType)
        self.r = list(self.r)

        assert isinstance(self.r_loaded_data, GeneratorType)
        self.r_loaded_data = list(self.r_loaded_data)

    def test_extra_field_title(self):
        assert (
            self.r[0].field_title
            == "AJHQA Time mean  !C Atmos u compnt of wind after timestep at 9.998 metres !C 01/12/2007 00:00 -> 01/01/2008 00:00"
        )

    def test_full_file(self):
        self.check_pp(self.r[0:10], ("PP", "extra_char_data.pp.txt"))
        self.check_pp(
            self.r_loaded_data[0:10],
            ("PP", "extra_char_data.w_data_loaded.pp.txt"),
        )

    def test_save_single(self, tmp_path):
        filepath = _shared_utils.get_data_path(
            ("PP", "model_comp", "dec_first_field.pp")
        )
        f = next(pp.load(filepath))

        temp_filename = tmp_path / "foo.pp"
        with open(temp_filename, "wb") as temp_fh:
            f.save(temp_fh)

        s = next(pp.load(temp_filename))

        # force the data to be loaded (this was done for f when save was run)
        s.data
        _shared_utils._assert_str_same(
            str(s) + "\n", str(f) + "\n", "", type_comparison_name="PP files"
        )

        assert _shared_utils.file_checksum(
            temp_filename
        ) == _shared_utils.file_checksum(filepath)


class TestSplittableInt:
    def test_3(self):
        t = pp.SplittableInt(3)
        assert t[0] == 3

    def test_grow_str_list(self):
        t = pp.SplittableInt(3)
        t[1] = 3
        assert t[1] == 3

        t[5] = 4

        assert t[5] == 4

        assert int(t) == 400033

        assert t == 400033
        assert t != 33

        assert t >= 400033
        assert not t >= 400034

        assert t <= 400033
        assert not t <= 400032

        assert t > 400032
        assert not t > 400034

        assert t < 400034
        assert not t < 400032

    def test_name_mapping(self):
        t = pp.SplittableInt(33214, {"ones": 0, "tens": 1, "hundreds": 2})
        assert t.ones == 4
        assert t.tens == 1
        assert t.hundreds == 2

        t.ones = 9
        t.tens = 4
        t.hundreds = 0

        assert t.ones == 9
        assert t.tens == 4
        assert t.hundreds == 0

    def test_name_mapping_multi_index(self):
        t = pp.SplittableInt(
            33214,
            {
                "weird_number": slice(None, None, 2),
                "last_few": slice(-2, -5, -2),
                "backwards": slice(None, None, -1),
            },
        )
        assert t.weird_number == 324
        assert t.last_few == 13
        pytest.raises(ValueError, setattr, t, "backwards", 1)
        pytest.raises(ValueError, setattr, t, "last_few", 1)
        assert t.backwards == 41233
        assert t == 33214

        t.weird_number = 99
        # notice that this will zero the 5th number

        assert t == 3919
        t.weird_number = 7899
        assert t == 7083919
        t.foo = 1

        t = pp.SplittableInt(33214, {"ix": slice(None, 2), "iy": slice(2, 4)})
        assert t.ix == 14
        assert t.iy == 32

        t.ix = 21
        assert t == 33221

        t = pp.SplittableInt(33214, {"ix": slice(-1, 2)})
        assert t.ix == 0

        t = pp.SplittableInt(4, {"ix": slice(None, 2), "iy": slice(2, 4)})
        assert t.ix == 4
        assert t.iy == 0

    def test_33214(self):
        t = pp.SplittableInt(33214)
        assert t[4] == 3
        assert t[3] == 3
        assert t[2] == 2
        assert t[1] == 1
        assert t[0] == 4

        # The rest should be zero
        for i in range(5, 100):
            assert t[i] == 0

    def test_negative_number(self):
        with pytest.raises(
            ValueError,
            match="Negative numbers not supported with splittable integers object",
        ):
            _ = pp.SplittableInt(-5)


class TestSplittableIntEquality:
    def test_not_implemented(self):
        class Terry:
            pass

        sin = pp.SplittableInt(0)
        assert sin.__eq__(Terry()) is NotImplemented
        assert sin.__ne__(Terry()) is NotImplemented


class TestPPDataProxyEquality:
    def test_not_implemented(self):
        class Terry:
            pass

        pox = pp.PPDataProxy(
            "john",
            "michael",
            "eric",
            "graham",
            "brian",
            "spam",
            "beans",
            "eggs",
        )
        assert pox.__eq__(Terry()) is NotImplemented
        assert pox.__ne__(Terry()) is NotImplemented


class TestPPFieldEquality:
    def test_not_implemented(self):
        class Terry:
            pass

        pox = pp.PPField3()
        assert pox.__eq__(Terry()) is NotImplemented
        assert pox.__ne__(Terry()) is NotImplemented
