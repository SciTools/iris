# (C) British Crown Copyright 2013 - 2014, Met Office
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
                 "Stash = namedtuple('Stash', 'grid_code field_code')\n\n\n")


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
                print msg
            grid = xref.get('grid')
            if grid is not None:
                try:
                    int(grid)
                except ValueError:
                    msg = ('grid code retrieved from STASH lookup'
                           'is not an interger: {}'.format(grid))
                    print msg
            else:
                grid = 0
            lbfc = xref.get('lbfcn')
            try:
                int(lbfc)
            except (ValueError, TypeError):
                lbfc = 0
            module_file.write(
                '    "{}": Stash({}, {}),\n'.format(stash, grid, lbfc))
        module_file.write('}\n')


def stash_grid_retrieve():
    """return a dictionary of stash codes and rel;ated information from
    the Met Office Reference Registry
    """
    baseurl = 'http://reference.metoffice.gov.uk/system/query?query='
    query = '''prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
prefix skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?stash ?grid ?lbfcn
WHERE {
  ?stashcode rdf:type <http://reference.metoffice.gov.uk/um/c4/stash/Stash> ;
  skos:notation ?stash ;
  <http://reference.metoffice.gov.uk/um/c4/stash/grid> ?gridcode .
OPTIONAL { ?gridcode skos:notation ?grid .}
OPTIONAL {?stashcode <http://reference.metoffice.gov.uk/um/c4/stash/ppfc> ?lbfc .
         ?lbfc skos:notation ?lbfcn .}
}
order by ?stash'''
    
    encquery = urllib.quote_plus(query)
    out_format = '&output=json'
    url = baseurl + encquery + out_format

    response = urllib2.urlopen(url)
    stash = json.loads(response.read())

    ## heads will be of the form [u'stash', u'grid', u'lbfcn']
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
