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
import json
import logging
import os
import requests
from collections import deque
from itertools import chain
from six.moves.queue import Queue
from threading import Thread


# Maximum number of threads for multi-threading code.
MAXTHREADS = 8

# Turn down requests logging.
logging.getLogger("requests").setLevel(logging.CRITICAL)


class _ResolveWorkerThread(Thread):
    """
    A :class:threading.Thread which moves objects from an input queue to an
    output deque using a 'dowork' method, as defined by a subclass.

    """
    def __init__(self, aqueue, adeque, exceptions):
        self.queue = aqueue
        self.deque = adeque
        self.exceptions = exceptions
        Thread.__init__(self)
        self.daemon = True

    def run(self):
        while not self.queue.empty():
            resource = self.queue.get()
            try:
                result = requests.head(resource)
                if result.status_code == 200:
                    self.deque.append(resource)
                else:
                    msg = '{} is not resolving correctly.'.format(resource)
                    self.exceptions.append(ValueError(msg))
            except Exception as e:
                self.exceptions.append(e)
            self.queue.task_done()


@tests.skip_inet
class TestImageFile(tests.IrisTest):
    def test_resolve(self):
        repo_fname = os.path.join(os.path.dirname(__file__), 'results',
                                  'imagerepo.json')
        with open(repo_fname, 'rb') as fi:
            repo = json.load(codecs.getreader('utf-8')(fi))
        uris = list(chain.from_iterable(six.itervalues(repo)))
        uri_list = deque()
        exceptions = deque()
        uri_queue = Queue()
        prefix = 'https://scitools.github.io/test-iris-imagehash'
        for uri in uris:
            if uri.startswith(prefix):
                uri_queue.put(uri)
            else:
                msg = '{} is not a valid resource.'.format(uri)
                exceptions.append(ValueError(msg))

        for i in range(MAXTHREADS):
            _ResolveWorkerThread(uri_queue, uri_list, exceptions).start()
        uri_queue.join()
        self.assertEqual(deque(), exceptions)


if __name__ == "__main__":
    tests.main()
