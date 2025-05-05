# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.pp._interpret_field` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import contextlib
from copy import deepcopy
from unittest import mock

import numpy as np

import iris
import iris.fileformats.pp as pp


class Test__interpret_fields__land_packed_fields(tests.IrisTest):
    def setUp(self):
        return_value = ("dummy", 0, 0, np.dtype("f4"))
        core_data = mock.MagicMock(return_value=return_value)
        # A field packed using a land/sea mask.
        self.pp_field = mock.Mock(
            lblrec=1,
            lbext=0,
            lbuser=[0] * 7,
            lbrow=0,
            lbnpt=0,
            raw_lbpack=21,
            lbpack=mock.Mock(n1=0, n2=2, n3=1),
            core_data=core_data,
        )
        # The field specifying the land/seamask.
        lbuser = [None, None, None, 30, None, None, 1]  # m01s00i030
        self.land_mask_field = mock.Mock(
            lblrec=1,
            lbext=0,
            lbuser=lbuser,
            lbrow=3,
            lbnpt=4,
            raw_lbpack=0,
            core_data=core_data,
        )

    def test_non_deferred_fix_lbrow_lbnpt(self):
        # Checks the fix_lbrow_lbnpt is applied to fields which are not
        # deferred.
        f1, mask = self.pp_field, self.land_mask_field
        self.assertEqual(f1.lbrow, 0)
        self.assertEqual(f1.lbnpt, 0)
        list(pp._interpret_fields([mask, f1]))
        self.assertEqual(f1.lbrow, 3)
        self.assertEqual(f1.lbnpt, 4)
        # Check the data's shape has been updated too.
        self.assertEqual(f1.data.shape, (3, 4))

    def test_fix_lbrow_lbnpt_no_mask_available(self):
        # Check a warning is issued when loading a land masked field
        # without a land mask.
        with mock.patch("warnings.warn") as warn:
            list(pp._interpret_fields([self.pp_field]))
        self.assertEqual(warn.call_count, 1)
        warn_msg = warn.call_args[0][0]
        self.assertTrue(
            warn_msg.startswith(
                "Landmask compressed fields existed without a landmask"
            ),
            "Unexpected warning message: {!r}".format(warn_msg),
        )

    def test_deferred_mask_field(self):
        # Check that the order of the load is yielded last if the mask
        # hasn't yet been seen.
        result = list(pp._interpret_fields([self.pp_field, self.land_mask_field]))
        self.assertEqual(result, [self.land_mask_field, self.pp_field])

    def test_not_deferred_mask_field(self):
        # Check that the order of the load is unchanged if a land mask
        # has already been seen.
        f1, mask = self.pp_field, self.land_mask_field
        mask2 = deepcopy(mask)
        result = list(pp._interpret_fields([mask, f1, mask2]))
        self.assertEqual(result, [mask, f1, mask2])

    def test_deferred_fix_lbrow_lbnpt(self):
        # Check the fix is also applied to fields which are deferred.
        f1, mask = self.pp_field, self.land_mask_field
        list(pp._interpret_fields([f1, mask]))
        self.assertEqual(f1.lbrow, 3)
        self.assertEqual(f1.lbnpt, 4)

    def test_landsea_unpacking_uses_dask(self):
        # Ensure that the graph of the (lazy) landsea-masked data contains an
        # explicit reference to a (lazy) landsea-mask field.
        # Otherwise its compute() will need to invoke another compute().
        # See https://github.com/SciTools/iris/issues/3237

        # This is too complex to explore in a mock-ist way, so let's load a
        # tiny bit of real data ...
        testfile_path = tests.get_data_path(
            ["FF", "landsea_masked", "testdata_mini_lsm.ff"]
        )
        landsea_mask, soil_temp = iris.load_cubes(
            testfile_path, ("land_binary_mask", "soil_temperature")
        )

        # Now check that the soil-temp dask graph correctly references the
        # landsea mask, in its dask graph.
        lazy_mask_array = landsea_mask.core_data()
        lazy_soildata_array = soil_temp.core_data()

        def is_pp_layer(obj):
            result = False
            if hasattr(obj, "values"):
                if len(obj) == 1:
                    (result,) = [hasattr(v, "_lbpack") for v in obj.values()]
            return result

        # Find the single reference to the PPDataProxy within the dask graph.
        (layer,) = [
            lay for lay in lazy_mask_array.dask.layers.values() if is_pp_layer(lay)
        ]
        ((layer_key, proxy),) = layer.items()
        # Replace the PPDataProxy with a mock.
        #  By replacing directly within the dask graph, we can prove whether an
        #   operation is relying on the graph in the expected way.
        mock_proxy = mock.MagicMock(spec=pp.PPDataProxy, wraps=proxy)
        layer.mapping[layer_key] = mock_proxy

        self.assertFalse(mock_proxy.__getitem__.called)
        with contextlib.suppress(TypeError):
            # Expect this to crash - data realisation is complex, and it is not
            #  worth mocking this fully.
            _ = lazy_soildata_array.compute()
        # Proves that the soil data graph includes the mask graph - computing
        #  the soil data has accessed the mask graph object which we mocked.
        self.assertTrue(mock_proxy.__getitem__.called)


if __name__ == "__main__":
    tests.main()
