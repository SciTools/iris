# (C) British Crown Copyright 2014, Met Office
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
"""
Tests for function
:func:`iris.fileformats.grib._load_convert.translate_phenomenon`.

"""
# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from copy import deepcopy

from iris.coords import DimCoord
from iris.unit import Unit
from iris.fileformats.grib._load_convert import Probability
from iris.fileformats.grib.grib_phenom_translation import _GribToCfDataClass

from iris.fileformats.grib._load_convert import translate_phenomenon


class Test_probability(tests.IrisTest):
    def setUp(self):
        # Patch inner call to return a given phenomenon type.
        target_module = 'iris.fileformats.grib._load_convert'
        self.phenom_lookup_patch = self.patch(
            target_module + '.itranslation.grib2_phenom_to_cf_info',
            return_value=_GribToCfDataClass('air_temperature', '', 'K', None))
        # Construct dummy call arguments
        self.probability = Probability('<prob_type>', 22.0)
        self.metadata = {'aux_coords_and_dims': []}

    def test_basic(self):
        result = translate_phenomenon(self.metadata, None, None, None,
                                      probability=self.probability)
        # Check metadata.
        thresh_coord = DimCoord([22.0],
                                standard_name='air_temperature',
                                long_name='', units='K')
        self.assertEqual(self.metadata, {
            'standard_name': None,
            'long_name': 'probability_of_air_temperature_<prob_type>',
            'units': Unit(1),
            'aux_coords_and_dims': [(thresh_coord, None)]})

    def test_no_phenomenon(self):
        original_metadata = deepcopy(self.metadata)
        self.phenom_lookup_patch.return_value = None
        result = translate_phenomenon(self.metadata, None, None, None,
                                      probability=self.probability)
        self.assertEqual(self.metadata, original_metadata)


if __name__ == '__main__':
    tests.main()
