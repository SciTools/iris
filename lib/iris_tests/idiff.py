#!/usr/bin/env python
# (C) British Crown Copyright 2010 - 2012, Met Office
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

import multiprocessing
import optparse
import os
import os.path
import shutil

import matplotlib.cm as cm
import matplotlib.pyplot as plt


DIFF_DIR = 'diff_images'


class Difference(object):
    def __init__(self, name, message):
        self.name = name
        self.message = message

    def __str__(self):
        return "%s\n   %s" % (self.name, self.message)


def compare_files(dir_path1, dir_path2, name):
    result = None
    
    path1 = os.path.join(dir_path1, name)
    path2 = os.path.join(dir_path2, name)
    
    if not os.path.isfile(path1) or not os.path.isfile(path2):
        result = Difference(name, 'missing file')
    else:
        image1 = plt.imread(path1)
        image2 = plt.imread(path2)
        if image1.shape != image2.shape:
            result = Difference(name, "shape: %s -> %s" % (image1.shape, image2.shape))
        else:
            diff = (image1 != image2).any(axis=2)
            if diff.any():
                diff_path = os.path.join(DIFF_DIR, name)
                diff_path = os.path.realpath(diff_path)
                plt.figure(figsize=reversed(diff.shape), dpi=1)
                plt.figimage(diff, cmap=cm.gray)
                plt.savefig(diff_path, dpi=1)
                result = Difference(name, "diff: %s" % diff_path)
    return result


def map_compare_files(args):
    return compare_files(*args)


def compare_dirs(dir_path1, dir_path2):
    if not os.path.isdir(dir_path1) or not os.path.isdir(dir_path2):
        raise ValueError('Can only compare directories')

    # Prepare the results directory
    if os.path.isdir(DIFF_DIR):
        shutil.rmtree(DIFF_DIR)
    os.makedirs(DIFF_DIR)
    
    pool = multiprocessing.Pool()
    names = set(name for name in os.listdir(dir_path1))
    names = names.union(name for name in os.listdir(dir_path2))
    args = [(dir_path1, dir_path2, name) for name in names if name.endswith('.png')]
    diffs = pool.map(map_compare_files, args)
    for diff in diffs:
        if diff is not None:
            print diff
            pass
    pool.close()
    pool.join()

def step_over_diffs(d_path1, d_path2):
    import matplotlib.pyplot as plt
    import matplotlib.image as mimg

    for fname in os.listdir(DIFF_DIR):
        print fname
        plt.figure(figsize=(16, 16))
        plt.suptitle(fname)
        ax = plt.subplot(221)
        plt.imshow(mimg.imread(os.path.join(d_path2, fname)))
        ax = plt.subplot(222, sharex=ax, sharey=ax)
        plt.imshow(mimg.imread(os.path.join(d_path1, fname)))
        ax = plt.subplot(223, sharex=ax, sharey=ax)
        plt.imshow(mimg.imread(os.path.join(DIFF_DIR, fname)))
        plt.show()


if __name__ == '__main__':
    usage = "usage: %prog [options] <dir1> <dir2>"
    description = "Compare directories of PNG images, producing black and white mask images of the differences." \
    " Designed to compare image output from different branches, created with 'python test_thing.py -sf'." \
    " Example: python idiff.py <trunk>/lib/iris_tests/image_results <branch>/lib/iris_tests/image_results"
    parser = optparse.OptionParser(usage=usage, description=description)
    parser.add_option('-o', '--output', dest='output', help='output directory', metavar='DIR')
    parser.add_option('-n', dest='compute_diffs', action="store_false", help='Enable flag to disable diff creation, simply do other things instead (such as view pre computed diffs.)')
    parser.add_option('-v', dest='view_diffs', action="store_true", help='view diffs')
    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error('Incorrect number of arguments')
    if options.output:
        DIFF_DIR = options.output
    if options.compute_diffs != False:
        compare_dirs(*args)
    if options.view_diffs:
        step_over_diffs(*args)
