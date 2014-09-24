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
Processing of metarelate metOcean content to provide Iris encodings of
metOcean mapping translations.

"""

from datetime import datetime
import os.path

from metarelate.fuseki import FusekiServer

# from translator import *

from translator import FORMAT_URIS, FieldcodeCFMapping, StashCFMapping
from translator import CFFieldcodeMapping, GRIB1LocalParamCFConstrainedMapping
from translator import GRIB1LocalParamCFMapping, GRIB2ParamCFMapping
from translator import CFConstrainedGRIB1LocalParamMapping
from translator import CFGRIB2ParamMapping, CFGRIB1LocalParamMapping

HEADER = """# (C) British Crown Copyright 2013 - {year}, Met Office
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
#
# DO NOT EDIT: AUTO-GENERATED
{doc_string}
from collections import namedtuple


CFName = namedtuple('CFName', 'standard_name long_name units')
"""

HEADER_GRIB = """
DimensionCoordinate = namedtuple('DimensionCoordinate',
                                 'standard_name units points')

G1LocalParam = namedtuple('G1LocalParam', 'edition t2version centre iParam')
G2Param = namedtuple('G2Param', 'edition discipline category number')
"""

DOC_STRING_GRIB = r'''"""
Provides GRIB/CF phenomenon translations.

"""'''

DOC_STRING_UM = r'''"""
Provides UM/CF phenomenon translations.

"""'''

DIR_BASE = '../lib/iris/fileformats'
FILE_UM_CF = os.path.join(DIR_BASE, 'um_cf_map.py')
FILE_GRIB_CF = os.path.join(DIR_BASE, 'grib', '_grib_cf_map.py')
YEAR = datetime.utcnow().year


def _retrieve_mappings(fuseki, source, target):
    """
    Interrogate the metarelate triple store for all
    phenomenon translation mappings from the source
    scheme to the target scheme.

    Args:
    * fuseki:
        The :class:`metrelate.fuseki.FusekiServer` instance.
    * source:
        The source metarelate metadata type for the mapping.
    * target:
        The target metarelate metadata type for the mapping.

    Return:
        The sequence of :class:`metarelate.Mapping`
        instances.

    """
    msg = 'Retrieving {!r} to {!r} mappings ...'
    print msg.format(source, target)
    return fuseki.retrieve_mappings(source, target)


def build_um_cf_map(fuseki, filename):
    """
    Encode the UM/CF phenomenon translation mappings
    within the specified file.

    Args:
    * fuseki:
        The :class:`metarelate.fuseki.FusekiServer` instance.
    * filename:
        The name of the file to contain the translations.

    """
    # Create the base directory.
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    # Create the file to contain UM/CF translations.
    with open(filename, 'w') as fh:
        fh.write(HEADER.format(year=YEAR, doc_string=DOC_STRING_UM))
        fh.write('\n')

        # Encode the relevant UM to CF translations.
        maps = _retrieve_mappings(fuseki, FORMAT_URIS['umf'],
                                  FORMAT_URIS['cff'])
        fh.writelines(FieldcodeCFMapping(maps).lines(fuseki))
        fh.writelines(StashCFMapping(maps).lines(fuseki))

        # Encode the relevant CF to UM translations.
        maps = _retrieve_mappings(fuseki, FORMAT_URIS['cff'],
                                  FORMAT_URIS['umf'])
        fh.writelines(CFFieldcodeMapping(maps).lines(fuseki))


def build_grib_cf_map(fuseki, filename):
    """
    Encode the GRIB/CF phenomenon translation mappings
    within the specified file.

    Args:
    * fuseki:
        The :class:`metarelate.fuseki.FusekiServer` instance.
    * filename:
        The name of the file to contain the translations.

    """
    # Create the base directory.
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    # Create the file to contain GRIB/CF translations.
    with open(FILE_GRIB_CF, 'w') as fh:
        fh.write(HEADER.format(year=YEAR, doc_string=DOC_STRING_GRIB))
        fh.write(HEADER_GRIB)
        fh.write('\n')

        # Encode the relevant GRIB to CF translations.
        maps = _retrieve_mappings(fuseki, FORMAT_URIS['gribm'],
                                  FORMAT_URIS['cff'])
        fh.writelines(GRIB1LocalParamCFConstrainedMapping(maps).lines(fuseki))
        fh.writelines(GRIB1LocalParamCFMapping(maps).lines(fuseki))
        fh.writelines(GRIB2ParamCFMapping(maps).lines(fuseki))

        # Encode the relevant CF to GRIB translations.
        maps = _retrieve_mappings(fuseki, FORMAT_URIS['cff'],
                                  FORMAT_URIS['gribm'])
        fh.writelines(CFConstrainedGRIB1LocalParamMapping(maps).lines(fuseki))
        fh.writelines(CFGRIB1LocalParamMapping(maps).lines(fuseki))
        fh.writelines(CFGRIB2ParamMapping(maps).lines(fuseki))


if __name__ == '__main__':
    with FusekiServer() as fuseki:
        fuseki.load()
        build_um_cf_map(fuseki, FILE_UM_CF)
        build_grib_cf_map(fuseki, FILE_GRIB_CF)
