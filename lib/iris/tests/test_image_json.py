# (C) British Crown Copyright 2016, Met Office
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
import logging
import os
import requests


@tests.skip_inet
class TestImageFile(tests.IrisTest):
    def test_resolve(self):

        iuri = ('https://api.github.com/repos/scitools/test-iris-imagehash/'
                'contents/images')
        r = requests.get(iuri)
        if r.status_code != 200:
            raise ValueError('Github API get failed: {}'.format(iuri))
        rj = r.json()
        prefix = 'https://scitools.github.io/test-iris-imagehash/images/'

        known_image_uris = set([prefix + rji['name'] for rji in rj])

        repo_fname = os.path.join(os.path.dirname(__file__), 'results',
                                  'imagerepo.json')
        with open(repo_fname, 'rb') as fi:
            repo = json.load(codecs.getreader('utf-8')(fi))
        uris = set(itertools.chain.from_iterable(six.itervalues(repo)))

        amsg = 'Images are referenced in imagerepo.json but not published:\n{}'
        amsg = amsg.format(uris.difference(known_image_uris))

        self.assertTrue(uris.issubset(known_image_uris), msg=amsg)


if __name__ == "__main__":
    tests.main()
