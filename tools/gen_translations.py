# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Processing of metarelate metOcean content to provide Iris encodings of
metOcean mapping translations.

"""

from datetime import datetime
import os.path
import requests
import sys

import metarelate
from metarelate.fuseki import FusekiServer

from translator import (FORMAT_URIS, FieldcodeCFMappings, StashCFNameMappings,
                        StashCFHeightConstraintMappings,
                        CFFieldcodeMappings,
                        GRIB1LocalParamCFConstrainedMappings,
                        GRIB1LocalParamCFMappings, GRIB2ParamCFMappings,
                        CFConstrainedGRIB1LocalParamMappings,
                        CFGRIB2ParamMappings, CFGRIB1LocalParamMappings)

HEADER = """# Copyright {name} contributors
#
# This file is part of {name} and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
#
# DO NOT EDIT: AUTO-GENERATED
# Created on {datestamp} from 
# http://www.metarelate.net/metOcean
# at commit {git_sha}
# https://github.com/metarelate/metOcean/commit/{git_sha}
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
    suri = 'http://www.metarelate.net/sparql/metOcean'
    msg = 'Retrieving {!r} to {!r} mappings ...'
    print(msg.format(source, target))
    return fuseki.retrieve_mappings(source, target, service=suri)


def build_um_cf_map(fuseki, now, git_sha, base_dir):
    """
    Encode the UM/CF phenomenon translation mappings
    within the specified file.

    Args:
    * fuseki:
        The :class:`metarelate.fuseki.FusekiServer` instance.
    * now:
        Time stamp to write into the file
    * git_sha:
        The git SHA1 of the metarelate commit
    * base_dir:
        The root directory of the Iris source.

    """
    filename = os.path.join(base_dir, 'lib', 'iris', 'fileformats',
                            'um_cf_map.py')

    # Create the base directory.
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    # Create the file to contain UM/CF translations.
    with open(filename, 'w') as fh:
        fh.write(HEADER.format(year=YEAR, doc_string=DOC_STRING_UM,
                               datestamp=now, git_sha=git_sha, name='Iris'))
        fh.write('\n')

        # Encode the relevant UM to CF translations.
        maps = _retrieve_mappings(fuseki, FORMAT_URIS['umf'],
                                  FORMAT_URIS['cff'])
        # create the collections, then call lines on each one
        # for thread safety during lines and encode
        fccf = FieldcodeCFMappings(maps)
        stcf = StashCFNameMappings(maps)
        stcfhcon = StashCFHeightConstraintMappings(maps)
        fh.writelines(fccf.lines(fuseki))
        fh.writelines(stcf.lines(fuseki))
        fh.writelines(stcfhcon.lines(fuseki))

        # Encode the relevant CF to UM translations.
        maps = _retrieve_mappings(fuseki, FORMAT_URIS['cff'],
                                  FORMAT_URIS['umf'])
        # create the collections, then call lines on each one
        # for thread safety during lines and encode
        cffc = CFFieldcodeMappings(maps)
        fh.writelines(cffc.lines(fuseki))


def build_grib_cf_map(fuseki, now, git_sha, base_dir):
    """
    Encode the GRIB/CF phenomenon translation mappings
    within the specified file.

    Args:
    * fuseki:
        The :class:`metarelate.fuseki.FusekiServer` instance.
    * now:
        Time stamp to write into the file
    * git_sha:
        The git SHA1 of the metarelate commit
    * base_dir:
        The root directory of the Iris source.

    """
    filename = os.path.join(base_dir, 'lib', 'iris', 'fileformats',
                            'grib', '_grib_cf_map.py')
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    # Create the file to contain GRIB/CF translations.
    with open(filename, 'w') as fh:
        fh.write(HEADER.format(year=YEAR, doc_string=DOC_STRING_GRIB,
                               datestamp=now, git_sha=git_sha,
                               name='iris-grib'))
        fh.write(HEADER_GRIB)
        fh.write('\n')

        # Encode the relevant GRIB to CF translations.
        maps = _retrieve_mappings(fuseki, FORMAT_URIS['gribm'],
                                  FORMAT_URIS['cff'])
        # create the collections, then call lines on each one
        # for thread safety during lines and encode
        g1cfc = GRIB1LocalParamCFConstrainedMappings(maps)
        g1c = GRIB1LocalParamCFMappings(maps)
        g2c = GRIB2ParamCFMappings(maps)
        fh.writelines(g1cfc.lines(fuseki))
        fh.writelines(g1c.lines(fuseki))
        fh.writelines(g2c.lines(fuseki))

        # Encode the relevant CF to GRIB translations.
        maps = _retrieve_mappings(fuseki, FORMAT_URIS['cff'],
                                  FORMAT_URIS['gribm'])
        # create the collections, then call lines on each one
        # for thread safety during lines and encode
        cfcg1 = CFConstrainedGRIB1LocalParamMappings(maps)
        cg1 = CFGRIB1LocalParamMappings(maps)
        cg2 = CFGRIB2ParamMappings(maps)
        fh.writelines(cfcg1.lines(fuseki))
        fh.writelines(cg1.lines(fuseki))
        fh.writelines(cg2.lines(fuseki))


def main():
    # Protect metarelate resource from 1.0 emergent bug
    if not float(metarelate.__version__) >= 1.1:
        raise ValueError("Please ensure that Metarelate Version is >= 1.1")
    now = datetime.utcnow().strftime('%d %B %Y %H:%m')
    git_sha = requests.get('http://www.metarelate.net/metOcean/latest_sha').text
    gen_path = os.path.abspath(sys.modules['__main__'].__file__)
    iris_path = os.path.dirname(os.path.dirname(gen_path))
    with FusekiServer() as fuseki:
        build_um_cf_map(fuseki, now, git_sha, iris_path)
        build_grib_cf_map(fuseki, now, git_sha, iris_path)
        
    if (git_sha !=
        requests.get('http://www.metarelate.net/metOcean/latest_sha').text):
        raise ValueError('The metarelate translation store has altered during'
                         'your retrieval, the results may not be stable.\n'
                         'Please rerun your retrieval.')

if __name__ == '__main__':
    main()
