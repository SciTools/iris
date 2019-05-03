# (C) British Crown Copyright 2016 - 2018, Met Office
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
import unittest
import time


@tests.skip_inet
@tests.skip_data
class TestImageFile(tests.IrisTest):
    def test_resolve(self):
        listingfile_uri = (
            'https://raw.githubusercontent.com/pp-mo/test-iris-imagehash'
            '/image_listing/v4_files_listing.txt')
        req = requests.get(listingfile_uri)
        if req.status_code != 200:
            raise ValueError('Github API get failed: {}'.format(
                listingfile_uri))

        reference_image_names = [line.strip()
                                 for line in req.content.split('\n')]
        base = 'https://scitools.github.io/test-iris-imagehash/images/v4'
        reference_image_uris = set('{}/{}'.format(base, name)
                                   for name in reference_image_names)

        imagerepo_json_filepath = os.path.join(
            os.path.dirname(__file__), 'results', 'imagerepo.json')
        with open(imagerepo_json_filepath, 'rb') as fi:
            imagerepo = json.load(codecs.getreader('utf-8')(fi))

        # "imagerepo" is {key: list_of_uris}. Put all uris in one big set.
        tests_uris = set(itertools.chain.from_iterable(
            six.itervalues(imagerepo)))

        missing_refs = list(tests_uris - reference_image_uris)
        if missing_refs:
            amsg = ('Images are referenced in imagerepo.json '
                    'but not published in {}:\n    {}')
            amsg = amsg.format(base, '    \n'.join(missing_refs))
            # Already seen the problThis should always fail
            self.assertFalse(bool(missing_refs), msg=amsg)


if __name__ == "__main__":
    tests.main()
