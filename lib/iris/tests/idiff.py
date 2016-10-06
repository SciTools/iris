#!/usr/bin/env python
# (C) British Crown Copyright 2010 - 2016, Met Office
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
Provides "diff-like" comparison of images.

Currently relies on matplotlib for image processing so limited to PNG format.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import argparse
import codecs
import contextlib
import hashlib
import json
import os.path
import requests
import shutil
import sys
import warnings

import filelock
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import matplotlib.testing.compare as mcompare
import matplotlib.widgets as mwidget

# Force iris.tests to use the ```tkagg``` backend by using the '-d'
# command-line argument as idiff is an interactive tool that requires a
# gui interface.
sys.argv.append('-d')
import iris.tests
import iris.util as iutil


_HASH_DIR = os.path.join('results', 'visual_tests')
_POSTFIX_LOCK = os.path.join('results', 'imagerepo.lock')
_POSTFIX_JSON = os.path.join('results', 'imagerepo.json')
_PREFIX_URI = 'https://scitools.github.io/test-images-scitools/image_files'
_TIMEOUT = 30
_TOL = 0


@contextlib.contextmanager
def temp_png(suffix=''):
    if suffix:
        suffix = '-{}'.format(suffix)
    fname = iutil.create_temp_filename(suffix+'.png')
    try:
        yield fname
    finally:
        os.remove(fname)


def diff_viewer(repo, key, dname, expected_fname, result_fname, diff_fname):
    plt.figure(figsize=(14, 12))
    plt.suptitle(os.path.basename(expected_fname))
    ax = plt.subplot(221)
    ax.imshow(mimg.imread(expected_fname))
    ax = plt.subplot(222, sharex=ax, sharey=ax)
    ax.imshow(mimg.imread(result_fname))
    ax = plt.subplot(223, sharex=ax, sharey=ax)
    ax.imshow(mimg.imread(diff_fname))

    # Determine the new image hash.png name.
    with open(result_fname, 'rb') as fi:
        sha1 = hashlib.sha1(fi.read())
    hash_fname = os.path.join(dname, _HASH_DIR, sha1.hexdigest()+'.png')
    uri = os.path.join(_PREFIX_URI, os.path.basename(hash_fname))

    def accept(event):
        if uri not in repo[key]:
            # Ensure to maintain strict time order where the first uri
            # associated with the repo key is the oldest, and the last
            # uri is the youngest
            repo[key].append(uri)
            with open(os.path.join(dname, _POSTFIX_JSON), 'wb') as fo:
                json.dump(repo, fo, indent=4, sort_keys=True)
            shutil.copy2(result_fname, hash_fname)
            msg = 'ACCEPTED:  {} -> {}'
            print(msg.format(os.path.basename(result_fname),
                             os.path.basename(hash_fname)))
        else:
            msg = 'DUPLICATE: {} -> {} (ignored)'
            print(msg.format(os.path.basename(result_fname),
                             os.path.basename(hash_fname)))
        os.remove(result_fname)
        os.remove(diff_fname)
        plt.close()

    def reject(event):
        if uri not in repo[key]:
            print('REJECTED:  {}'.format(os.path.basename(result_fname)))
        else:
            msg = 'DUPLICATE: {} -> {} (ignored)'
            print(msg.format(os.path.basename(result_fname),
                             os.path.basename(hash_fname)))
            os.remove(result_fname)
        os.remove(diff_fname)
        plt.close()

    ax_accept = plt.axes([0.7, 0.05, 0.1, 0.075])
    ax_reject = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = mwidget.Button(ax_accept, 'Accept')
    bnext.on_clicked(accept)
    bprev = mwidget.Button(ax_reject, 'Reject')
    bprev.on_clicked(reject)
    plt.show()


def step_over_diffs(result_dir):
    dname = os.path.dirname(iris.tests.__file__)
    lock = filelock.FileLock(os.path.join(dname, _POSTFIX_LOCK))

    with lock.acquire(timeout=_TIMEOUT):
        fname = os.path.join(dname, _POSTFIX_JSON)
        with open(fname, 'rb') as fi:
            repo = json.load(codecs.getreader('utf-8')(fi))

        result_dir = os.path.join(dname, 'result_image_comparison')
        for fname in sorted(os.listdir(result_dir)):
            result_fname = os.path.join(result_dir, fname)
            ext = os.path.splitext(result_fname)[1]
            if not (os.path.isfile(result_fname) and ext == '.png'):
                continue
            key = os.path.splitext('-'.join(fname.split('-')[1:]))[0]
            try:
                uri = repo[key][0]
            except KeyError:
                wmsg = 'Ignoring unregistered test result {!r}.'
                warnings.warn(wmsg.format(key))
                continue
            with temp_png(key) as expected_fname:
                resource = requests.get(uri)
                with open(expected_fname, 'wb') as fo:
                    fo.write(resource.content)
                mcompare.compare_images(expected_fname, result_fname, tol=_TOL)
                diff_fname = result_fname[:-4] + '-failed-diff.png'
                diff_viewer(repo, key, dname, expected_fname, result_fname,
                            diff_fname)


if __name__ == '__main__':
    default = os.path.join(os.path.dirname(iris.tests.__file__),
                           'result_image_comparison')
    description = 'Iris graphic test difference tool.'
    parser = argparse.ArgumentParser(description=description)
    help = 'Path to iris tests result image directory (default: %(default)s)'
    parser.add_argument('--resultdir', '-r',
                        default=default,
                        help=help)
    help = 'Force "iris.tests" to use the tkagg backend (default: %(default)s)'
    parser.add_argument('-d',
                        action='store_true',
                        default=True,
                        help=help)
    args = parser.parse_args()
    result_dir = args.resultdir
    if not os.path.isdir(result_dir):
        emsg = 'Invalid results directory: {}'
        raise ValueError(emsg.format(result_dir))
    step_over_diffs(result_dir)
