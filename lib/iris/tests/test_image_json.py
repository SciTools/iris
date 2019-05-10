# (C) British Crown Copyright 2016 - 2019, Met Office
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


from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import codecs
import itertools
import json
import os
import requests


@tests.skip_inet
class TestImageFile(tests.IrisTest):
    def test_resolve(self):
        listingfile_uri = (
            'https://raw.githubusercontent.com/SciTools/test-iris-imagehash'
            '/gh-pages/v4_files_listing.txt')
        req = requests.get(listingfile_uri)
        if req.status_code != 200:
            raise ValueError('GET failed on image listings file: {}'.format(
                listingfile_uri))

        listings_text = req.content.decode('utf-8')
        reference_image_filenames = [line.strip()
                                     for line in listings_text.split('\n')]
        base = 'https://scitools.github.io/test-iris-imagehash/images/v4'
        reference_image_uris = set('{}/{}'.format(base, name)
                                   for name in reference_image_filenames)

        imagerepo_json_filepath = os.path.join(
            os.path.dirname(__file__), 'results', 'imagerepo.json')
        with open(imagerepo_json_filepath, 'rb') as fi:
            imagerepo = json.load(codecs.getreader('utf-8')(fi))

        # "imagerepo" maps key: list-of-uris. Put all the uris in one big set.
        tests_uris = set(itertools.chain.from_iterable(
            six.itervalues(imagerepo)))

        missing_refs = list(tests_uris - reference_image_uris)
        n_missing_refs = len(missing_refs)
        if n_missing_refs > 0:
            amsg = ('Missing images: These {} image uris are referenced in '
                    'imagerepo.json, but not listed in {} : ')
            amsg = amsg.format(n_missing_refs, listingfile_uri)
            amsg += ''.join('\n        {}'.format(uri)
                            for uri in missing_refs)
            # Always fails when we get here: report the problem.
            self.assertEqual(n_missing_refs, 0, msg=amsg)


if __name__ == "__main__":
    tests.main()
