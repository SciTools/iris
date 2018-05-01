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
        # https://developer.github.com/v3/#user-agent-required
        headers = {'User-Agent': 'scitools-bot'}
        rate_limit_uri = 'https://api.github.com/rate_limit'
        rl = requests.get(rate_limit_uri, headers=headers)
        some_left = False
        if rl.status_code == 200:
            rates = rl.json()
            remaining = rates.get('rate', {})
            ghapi_remaining = remaining.get('remaining')
        else:
            ghapi_remaining = 0

        # Only run this test if there are IP based rate limited calls left.
        # 3 is an engineering tolerance, in case of race conditions.
        amin = 3
        if ghapi_remaining < amin:
            return unittest.skip("Less than {} anonymous calls to "
                                 "GH API left!".format(amin))
        iuri = ('https://api.github.com/repos/scitools/'
                'test-iris-imagehash/contents/images/v4')
        r = requests.get(iuri, headers=headers)
        if r.status_code != 200:
            raise ValueError('Github API get failed: {}'.format(iuri,
                                                                r.text))
        rj = r.json()
        base = 'https://scitools.github.io/test-iris-imagehash/images/v4'

        known_image_uris = set([os.path.join(base, rji['name']) for rji in rj])

        repo_fname = os.path.join(os.path.dirname(__file__), 'results',
                                  'imagerepo.json')
        with open(repo_fname, 'rb') as fi:
            repo = json.load(codecs.getreader('utf-8')(fi))
        uris = set(itertools.chain.from_iterable(six.itervalues(repo)))

        amsg = ('Images are referenced in imagerepo.json but not published '
                'in {}:\n{}')
        diffs = list(uris.difference(known_image_uris))
        amsg = amsg.format(base, '\n'.join(diffs))

        self.assertTrue(uris.issubset(known_image_uris), msg=amsg)


if __name__ == "__main__":
    tests.main()
