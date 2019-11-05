# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

import json
import urllib
import urllib2

from iris.fileformats.pp import STASH

import gen_helpers


HEADER = '''
"""
Auto-generated from iris/tools/gen_stash_refs.py
Relates grid code and field code to the stash code.

"""
'''

CODE_PREAMBLE = ("from collections import namedtuple\n\n\n"
                 "Stash = namedtuple('Stash', "
                 "'grid_code field_code pseudo_level_type')\n\n\n")


def _value_from_xref(xref, name):
    """Return the value for the key name from xref.

    Will return 0 if the key does not look like an integer.
    """

    result = xref.get(name)
    try:
        int(result)
    except (ValueError, TypeError):
        result = 0
    return result


def write_cross_reference_module(module_path, xrefs):
    gen_helpers.prep_module_file(module_path)
    with open(module_path, 'a') as module_file:
        module_file.write(HEADER)
        module_file.write(CODE_PREAMBLE)
        module_file.write('STASH_TRANS = {\n')
        for xref in xrefs:
            stash = xref.get('stash')
            try:
                STASH.from_msi(stash.replace('"', ''))
            except ValueError:
                msg = ('stash code is not of a recognised'
                       '"m??s??i???" form: {}'.format(stash))
                print(msg)
            grid = xref.get('grid')
            if grid is not None:
                try:
                    int(grid)
                except ValueError:
                    msg = ('grid code retrieved from STASH lookup'
                           'is not an integer: {}'.format(grid))
                    print(msg)
            else:
                grid = 0

            lbfc = _value_from_xref(xref, 'lbfcn')
            pseudT = _value_from_xref(xref, 'pseudT')

            module_file.write(
                '    "{}": Stash({}, {}, {}),\n'.format(stash,
                                                        grid,
                                                        lbfc,
                                                        pseudT))
        module_file.write('}\n')


def stash_grid_retrieve():
    """return a dictionary of stash codes and rel;ated information from
    the Met Office Reference Registry
    """
    baseurl = 'http://reference.metoffice.gov.uk/system/query?query='
    query = '''prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
prefix skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?stash ?grid ?lbfcn ?pseudT
WHERE {
  ?stashcode rdf:type <http://reference.metoffice.gov.uk/um/c4/stash/Stash> ;
  skos:notation ?stash ;
  <http://reference.metoffice.gov.uk/um/c4/stash/grid> ?gridcode .
OPTIONAL { ?gridcode skos:notation ?grid .}
OPTIONAL {?stashcode <http://reference.metoffice.gov.uk/um/c4/stash/ppfc> ?lbfc .
         ?lbfc skos:notation ?lbfcn .}
OPTIONAL {?stashcode <http://reference.metoffice.gov.uk/um/c4/stash/pseudT> ?pseudT_id .
          ?pseudT_id skos:notation ?pseudT . }
}
order by ?stash'''

    encquery = urllib.quote_plus(query)
    out_format = '&output=json'
    url = baseurl + encquery + out_format

    response = urllib2.urlopen(url)
    stash = json.loads(response.read())

    ## heads will be of the form [u'stash', u'grid', u'lbfcn', u'pseudT']
    ## as defined in the query string
    heads = stash['head']['vars']

    stashcodes = []

    for result in stash['results']['bindings']:
        res = {}
        for head in heads:
            if head in result:
                res[head] = result[head]['value']
        stashcodes.append(res)
    return stashcodes


if __name__ == '__main__':
    xrefs = stash_grid_retrieve()
    outfile = '../lib/iris/fileformats/_ff_cross_references.py'
    write_cross_reference_module(outfile, xrefs)
