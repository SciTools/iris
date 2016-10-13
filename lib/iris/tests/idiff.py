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
from glob import glob
import hashlib
import json
import os.path
import shutil
import sys
import warnings

from PIL import Image
import filelock
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import matplotlib.testing.compare as mcompare
import matplotlib.widgets as mwidget
import requests

# Force iris.tests to use the ```tkagg``` backend by using the '-d'
# command-line argument as idiff is an interactive tool that requires a
# gui interface.
sys.argv.append('-d')
import iris.tests
import iris.util as iutil


_POSTFIX_DIFF = '-failed-diff.png'
_POSTFIX_JSON = os.path.join('results', 'imagerepo.json')
_POSTFIX_LOCK = os.path.join('results', 'imagerepo.lock')
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


def diff_viewer(repo, key, repo_fname,
                expected_fname, result_fname, diff_fname):
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
    result_dir = os.path.dirname(result_fname)
    fname = sha1.hexdigest() + '.png'
    base_uri = 'https://scitools.github.io/test-images-scitools/image_files/{}'
    uri = base_uri.format(fname)
    hash_fname = os.path.join(result_dir, fname)

    def accept(event):
        if uri not in repo[key]:
            # Ensure to maintain strict time order where the first uri
            # associated with the repo key is the oldest, and the last
            # uri is the youngest
            repo[key].append(uri)
            # Update the image repo.
            with open(repo_fname, 'wb') as fo:
                json.dump(repo, codecs.getwriter('utf-8')(fo),
                          indent=4, sort_keys=True)
            os.rename(result_fname, hash_fname)
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

    def skip(event):
        # Let's keep both the result and the diff files.
        print('SKIPPED:   {}'.format(os.path.basename(result_fname)))
        plt.close()

    ax_accept = plt.axes([0.59, 0.05, 0.1, 0.075])
    ax_reject = plt.axes([0.7, 0.05, 0.1, 0.075])
    ax_skip = plt.axes([0.81, 0.05, 0.1, 0.075])
    baccept = mwidget.Button(ax_accept, 'Accept')
    baccept.on_clicked(accept)
    breject = mwidget.Button(ax_reject, 'Reject')
    breject.on_clicked(reject)
    bskip = mwidget.Button(ax_skip, 'Skip')
    bskip.on_clicked(skip)
    plt.show()


def step_over_diffs(result_dir, index):
    processed = False
    dname = os.path.dirname(iris.tests.__file__)
    lock = filelock.FileLock(os.path.join(dname, _POSTFIX_LOCK))
    prog = os.path.basename(os.path.splitext(sys.argv[0])[0])
    msg = '\n{}: Comparing result image with {} expected test image.'
    print(msg.format(prog, 'oldest' if index == 0 else 'youngest'))

    # Remove old image diff results.
    target = os.path.join(result_dir, '*{}'.format(_POSTFIX_DIFF))
    for fname in glob(target):
        os.remove(fname)

    with lock.acquire(timeout=_TIMEOUT):
        # Load the imagerepo.
        repo_fname = os.path.join(dname, _POSTFIX_JSON)
        with open(repo_fname, 'rb') as fi:
            repo = json.load(codecs.getreader('utf-8')(fi))

        target = os.path.join(result_dir, 'result-*.png')
        for result_fname in sorted(glob(target)):
            # We only care about PNG images.
            try:
                im = Image.open(result_fname)
                if im.format != 'PNG':
                    # Ignore - it's not a png image.
                    continue
            except IOError:
                # Ignore - it's not an image.
                continue
            key = os.path.splitext('-'.join(result_fname.split('-')[1:]))[0]
            try:
                uri = repo[key][index]
            except KeyError:
                wmsg = 'Ignoring unregistered test result {!r}.'
                warnings.warn(wmsg.format(key))
                continue
            with temp_png(key) as expected_fname:
                processed = True
                resource = requests.get(uri)
                if resource.status_code == 200:
                    with open(expected_fname, 'wb') as fo:
                        fo.write(resource.content)
                else:
                    # Perhaps the uri has not been pushed into the repo yet,
                    # so check if a local "developer" copy is available ...
                    local_fname = os.path.join(result_dir,
                                               os.path.basename(uri))
                    if not os.path.isfile(local_fname):
                        emsg = 'Bad URI {!r} for test {!r}.'
                        raise ValueError(uri, key)
                    else:
                        # The temporary expected filename has the test name
                        # baked into it, and is used in the diff plot title.
                        # So copy the local file to the exected file to
                        # maintain this helpfulness.
                        shutil.copy(local_fname, expected_fname)
                mcompare.compare_images(expected_fname, result_fname, tol=_TOL)
                diff_fname = os.path.splitext(result_fname)[0] + _POSTFIX_DIFF
                diff_viewer(repo, key, repo_fname, expected_fname,
                            result_fname, diff_fname)
        if not processed:
            msg = '\n{}: There are no iris test result images to process.\n'
            print(msg.format(prog))


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
    help = ('Compare result image with the oldest or last registered '
            'expected test image')
    parser.add_argument('--last', '-l',
                        action='store_true',
                        default=False,
                        help=help)
    args = parser.parse_args()
    result_dir = args.resultdir
    if not os.path.isdir(result_dir):
        emsg = 'Invalid results directory: {}'
        raise ValueError(emsg.format(result_dir))
    index = -1 if args.last else 0
    step_over_diffs(result_dir, index)
