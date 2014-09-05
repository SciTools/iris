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
Defines a lightweight wrapper class to wrap a single grib message.

"""

from collections import OrderedDict
from distutils.version import StrictVersion
import re

import gribapi


_GRIBAPI_VERSION = gribapi.grib_get_api_version()

_FILE_SPECIFIC_CODED_KEYS = [
    'identifier', 'discipline', 'editionNumber', 'totalLength',
    'section1Length', 'numberOfSection', 'centre', 'subCentre',
    'tablesVersion', 'localTablesVersion', 'significanceOfReferenceTime',
    'year', 'month', 'day', 'hour', 'minute', 'second',
    'productionStatusOfProcessedData', 'typeOfProcessedData', 'section2Length',
    'numberOfSection', 'grib2LocalSectionNumber', 'marsClass', 'marsType',
    'marsStream', 'experimentVersionNumber', 'section3Length',
    'numberOfSection', 'sourceOfGridDefinition', 'numberOfDataPoints',
    'numberOfOctectsForNumberOfPoints', 'interpretationOfNumberOfPoints',
    'gridDefinitionTemplateNumber', 'shapeOfTheEarth',
    'scaleFactorOfRadiusOfSphericalEarth',
    'scaledValueOfRadiusOfSphericalEarth', 'scaleFactorOfEarthMajorAxis',
    'scaledValueOfEarthMajorAxis', 'scaleFactorOfEarthMinorAxis',
    'scaledValueOfEarthMinorAxis', 'Ni', 'Nj',
    'basicAngleOfTheInitialProductionDomain', 'subdivisionsOfBasicAngle',
    'latitudeOfFirstGridPoint', 'longitudeOfFirstGridPoint',
    'resolutionAndComponentFlags', 'latitudeOfLastGridPoint',
    'longitudeOfLastGridPoint', 'iDirectionIncrement', 'jDirectionIncrement',
    'scanningMode', 'section4Length', 'numberOfSection', 'NV',
    'productDefinitionTemplateNumber', 'parameterCategory', 'parameterNumber',
    'typeOfGeneratingProcess', 'backgroundProcess',
    'generatingProcessIdentifier', 'hoursAfterDataCutoff',
    'minutesAfterDataCutoff', 'indicatorOfUnitOfTimeRange', 'forecastTime',
    'typeOfFirstFixedSurface', 'scaleFactorOfFirstFixedSurface',
    'scaledValueOfFirstFixedSurface', 'typeOfSecondFixedSurface',
    'scaleFactorOfSecondFixedSurface', 'scaledValueOfSecondFixedSurface', 'pv',
    'section5Length', 'numberOfSection', 'numberOfValues',
    'dataRepresentationTemplateNumber', 'referenceValue', 'binaryScaleFactor',
    'decimalScaleFactor', 'bitsPerValue', 'typeOfOriginalFieldValues',
    'section6Length', 'numberOfSection', 'bitMapIndicator', 'section7Length',
    'numberOfSection', 'codedValues', '7777']


class GribMessage(object):
    """
    Lightweight grib message wrapper, containing **only** the coded keys and
    data attribute of the input grib message.

    """

    def __init__(self, grib_id):
        """
        A GribData object contains the **coded** keys and data attribute from
        a grib message that is identified by the input message id.

        Args:

        * grib_id:
            An integer referencing a grib message within an open grib file.

        """
        self._grib_id = grib_id
        self._sections = None
        self._new_section_key_matcher = re.compile(r'section([0-9]{1})Length')

    @property
    def data(self):
        return self._get_message_data()

    @property
    def sections(self):
        if self._sections is None:
            self._sections = self._get_message_sections()
        return self._sections

    @property
    def keys(self):
        """Returns a list of all coded keys, not sorted by section."""
        return self._get_message_keys()

    def get_containing_section(self, key):
        """
        Returns the section number that contains the queried key or raises
        an error if the key is not found in the message.

        """
        res = None
        for section_number, section in self.sections.iteritems():
            if key in section.keys():
                res = section_number
                break
        if res is None:
            msg = 'Requested key {} was not found in message.'
            raise KeyError(msg.format(key))
        return res

    def _get_message_keys(self):
        """
        Returns a list of all the **coded** keys in the message.

        This currently does not work, but a fix is coming in gribapi v1.13; see
        AVD-317.

        """
        if StrictVersion(_GRIBAPI_VERSION) < StrictVersion('1.13.0'):
            keys = _FILE_SPECIFIC_CODED_KEYS
        else:
            keys = []
            keys_itr = gribapi.grib_keys_iterator_new(self._grib_id)
            gribapi.grib_skip_computed(keys_itr)
            while gribapi.grib_keys_iterator_next(keys_itr):
                key_name = gribapi.grib_keys_iterator_get_name(keys_itr)
                keys.append(key_name)

            gribapi.grib_keys_iterator_delete(keys_itr)
        return keys

    def _get_message_sections(self):
        """
        Sorts keys in the grib message by containing section.

        Returns a list of :class:`collections.OrderedDict` objects. One such
        object is made for each section in the message such that the message
        index equals the section number. Each object contains key:value pairs
        for all of the **coded** message keys in each given section.

        """
        keys = self._get_message_keys()
        sections = OrderedDict()
        # The first keys in a message are for the whole message and are
        # contained in section 0.
        section = new_section = 0
        # Use a `collections.OrderedDict` to retain key ordering.
        section_keys = OrderedDict()

        for key in keys:
            key_match = re.match(self._new_section_key_matcher, key)
            if key_match is not None:
                new_section = int(key_match.group(1))
            # This key only shows up in section 8, which doesn't have a
            # `section8Length` coded key...
            elif key == '7777':
                new_section = 8

            if section != new_section:
                sections[section] = section_keys
                section_keys = OrderedDict()
            # This key is repeated in each section meaning that the last value
            # is always returned, so override the api-retrieved value.
            if key != 'numberOfSection':
                section_keys[key] = self._get_key_value(key)
            else:
                section_keys[key] = section
            section = new_section
        # Write the last section's dictionary to sections so it's not lost.
        sections[section] = section_keys
        return sections

    def _get_message_data(self):
        """Get the data array from the grib message."""
        return gribapi.grib_get_values(self._grib_id)

    def _get_key_value(self, key):
        """
        Get the value associated with the given key in the grib message.

        Args:

        * key:
            The grib key to retrieve the value of.

        Returns the value associated with the requested key in the grib
        message.

        """
        res = None
        # See http://nullege.com/codes/search/gribapi.grib_get_values.
        try:
            if key in ['codedValues', 'pv']:
                res = gribapi.grib_get_array(self._grib_id, key)
            else:
                res = gribapi.grib_get(self._grib_id, key)
        # Deal with gribapi not differentiating between exception types.
        except gribapi.GribInternalError as e:
            if e.msg == "Passed array is too small":
                res = gribapi.grib_get_array(self._grib_id, key)
            else:
                raise e
        return res
