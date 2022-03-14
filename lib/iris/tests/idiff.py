# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
# !/usr/bin/env python
"""
Provides "diff-like" comparison of images.

Currently relies on matplotlib for image processing so limited to PNG format.

"""

import argparse
import contextlib
from glob import glob
import hashlib
import os.path
import sys
import warnings

# Force iris.tests to use the ```tkagg``` backend by using the '-d'
# command-line argument as idiff is an interactive tool that requires a
# gui interface.
sys.argv.append("-d")
from PIL import Image  # noqa
import imagehash  # noqa
import matplotlib.image as mimg  # noqa
import matplotlib.pyplot as plt  # noqa
import matplotlib.testing.compare as mcompare  # noqa
from matplotlib.testing.exceptions import ImageComparisonFailure  # noqa
import matplotlib.widgets as mwidget  # noqa
import numpy as np  # noqa

import iris.tests  # noqa
import iris.util as iutil  # noqa

_POSTFIX_DIFF = "-failed-diff.png"


@contextlib.contextmanager
def temp_png(suffix=""):
    if suffix:
        suffix = "-{}".format(suffix)
    fname = iutil.create_temp_filename(suffix + ".png")
    try:
        yield fname
    finally:
        os.remove(fname)


def image_exists_with_prefix(image_path, directory, prefix):
    """
    Does the given image already exist in the given directory with the given
    prefix?
    """
    check_image_hash = hashlib.sha256(
        open(image_path, "rb").read()
    ).hexdigest()

    image_found = False

    target = os.path.join(directory, f"{prefix}*")

    for image_name in glob(target):
        dir_image_hash = hashlib.sha256(
            open(os.path.join(directory, image_name), "rb").read()
        ).hexdigest()
        if dir_image_hash == check_image_hash:
            image_found = True
            break

    return image_found


def diff_viewer(
    key,
    phash,
    status,
    expected_fname,
    result_fname,
    diff_fname,
):
    fig = plt.figure(figsize=(14, 12))
    plt.suptitle(os.path.basename(expected_fname))
    ax = plt.subplot(221)
    ax.imshow(mimg.imread(expected_fname))
    ax = plt.subplot(222, sharex=ax, sharey=ax)
    ax.imshow(mimg.imread(result_fname))
    ax = plt.subplot(223, sharex=ax, sharey=ax)
    ax.imshow(mimg.imread(diff_fname))

    result_dir = os.path.dirname(result_fname)

    def accept(event):
        # TODO: This check should include the dir we're working in, and the main dir
        if not image_exists_with_prefix(result_fname, result_dir, key):
            # Ensure to maintain strict time order where the first uri
            # associated with the repo key is the oldest, and the last
            # uri is the youngest
            # TODO: Increment the index of the result by 1
            out_file = os.path.join(
                result_dir, os.path.basename(expected_fname)
            )
            # os.rename(result_fname, out_file)
            print(f"would rename {result_fname} to {out_file}")
            msg = "ACCEPTED:  {} -> {}"
            print(
                msg.format(
                    os.path.basename(result_fname),
                    os.path.basename(expected_fname),
                )
            )
        else:
            msg = "DUPLICATE: {} -> {} (ignored)"
            print(
                msg.format(
                    os.path.basename(result_fname),
                    os.path.basename(expected_fname),
                )
            )
            os.remove(result_fname)
        os.remove(diff_fname)
        plt.close()

    def reject(event):
        if not image_exists_with_prefix(result_fname, result_dir, key):
            print("REJECTED:  {}".format(os.path.basename(result_fname)))
        else:
            msg = "DUPLICATE: {} -> {} (ignored)"
            print(
                msg.format(
                    os.path.basename(result_fname),
                    os.path.basename(expected_fname),
                )
            )
        os.remove(result_fname)
        os.remove(diff_fname)
        plt.close()

    def skip(event):
        # Let's keep both the result and the diff files.
        print("SKIPPED:   {}".format(os.path.basename(result_fname)))
        plt.close()

    ax_accept = plt.axes([0.59, 0.05, 0.1, 0.075])
    ax_reject = plt.axes([0.7, 0.05, 0.1, 0.075])
    ax_skip = plt.axes([0.81, 0.05, 0.1, 0.075])
    baccept = mwidget.Button(ax_accept, "Accept")
    baccept.on_clicked(accept)
    breject = mwidget.Button(ax_reject, "Reject")
    breject.on_clicked(reject)
    bskip = mwidget.Button(ax_skip, "Skip")
    bskip.on_clicked(skip)
    plt.text(0.59, 0.15, status, transform=fig.transFigure)
    plt.show()


def _calculate_hit(image_paths, phash, action):

    expected = [
        imagehash.phash(
            Image.open(image_path), hash_size=iris.tests._HASH_SIZE
        )
        for image_path in image_paths
    ]

    # Calculate the hamming distance vector for the result hash.
    distances = [e - phash for e in expected]

    if action == "first":
        index = 0
    elif action == "last":
        index = -1
    elif action == "similar":
        index = np.argmin(distances)
    elif action == "different":
        index = np.argmax(distances)
    else:
        emsg = "Unknown action: {!r}"
        raise ValueError(emsg.format(action))

    return index, distances[index]


def step_over_diffs(result_dir, action, display=True):
    processed = False

    if action in ["first", "last"]:
        kind = action
    elif action in ["similar", "different"]:
        kind = "most {}".format(action)
    else:
        emsg = "Unknown action: {!r}"
        raise ValueError(emsg.format(action))
    if display:
        msg = (
            "\nComparing the {!r} expected image with "
            "the test result image."
        )
        print(msg.format(kind))

    # Remove old image diff results.
    target = os.path.join(result_dir, "*{}".format(_POSTFIX_DIFF))
    for fname in glob(target):
        os.remove(fname)

    # Filter out all non-test result image files.
    target_glob = os.path.join(result_dir, "result-*.png")
    results = []
    for fname in sorted(glob(target_glob)):
        # We only care about PNG images.
        try:
            im = Image.open(fname)
            if im.format != "PNG":
                # Ignore - it's not a png image.
                continue
        except IOError:
            # Ignore - it's not an image.
            continue
        results.append(fname)

    count = len(results)

    reference_images = glob(
        os.path.join(iris.tests.get_data_path("images"), "*")
    )

    for count_index, result_fname in enumerate(results):
        key = os.path.splitext("-".join(result_fname.split("result-")[1:]))[0]

        try:
            # Calculate the test result perceptual image hash.
            phash = imagehash.phash(
                Image.open(result_fname), hash_size=iris.tests._HASH_SIZE
            )
            relevant_image_names = [
                x
                for x in filter(
                    lambda x: os.path.basename(x).startswith(key),
                    reference_images,
                )
            ]
            hash_index, distance = _calculate_hit(
                relevant_image_names, phash, action
            )
            uri = relevant_image_names[hash_index]
        except KeyError:
            wmsg = "Ignoring unregistered test result {!r}."
            warnings.warn(wmsg.format(key))
            continue

        processed = True

        # Look in test data for our image
        local_fname = os.path.join(
            iris.tests.get_data_path("images"), os.path.basename(uri)
        )
        if not os.path.isfile(local_fname):
            emsg = "Bad URI {!r} for test {!r}."
            raise ValueError(emsg.format(local_fname, key))

        try:
            mcompare.compare_images(local_fname, result_fname, tol=0)
        except Exception as e:
            if isinstance(e, ValueError) or isinstance(
                e, ImageComparisonFailure
            ):
                print("Could not compare {}: {}".format(result_fname, e))
                continue
            else:
                # Propagate the exception, keeping the stack trace
                raise
        diff_fname = os.path.splitext(result_fname)[0] + _POSTFIX_DIFF
        args = local_fname, result_fname, diff_fname
        if display:
            msg = "Image {} of {}: hamming distance = {} " "[{!r}]"
            status = msg.format(count_index + 1, count, distance, kind)
            prefix = key, phash, status
            yield prefix + args
        else:
            yield args
    if display and not processed:
        print("\nThere are no iris test result images to process.\n")


if __name__ == "__main__":
    default = os.path.join(
        os.path.dirname(iris.tests.__file__), "result_image_comparison"
    )
    description = "Iris graphic test difference tool."
    formatter_class = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(
        description=description, formatter_class=formatter_class
    )
    help = "path to iris tests result image directory (default: %(default)s)"
    parser.add_argument("--resultdir", "-r", default=default, help=help)
    help = 'force "iris.tests" to use the tkagg backend (default: %(default)s)'
    parser.add_argument("-d", action="store_true", default=True, help=help)
    help = """
first     - compare result image with first (oldest) expected image
last      - compare result image with last (youngest) expected image
similar   - compare result image with most similar expected image (default)
different - compare result image with most unsimilar expected image
"""
    choices = ("first", "last", "similar", "different")
    parser.add_argument(
        "action", nargs="?", choices=choices, default="similar", help=help
    )
    args = parser.parse_args()
    result_dir = args.resultdir
    if not os.path.isdir(result_dir):
        emsg = "Invalid results directory: {}"
        raise ValueError(emsg.format(result_dir))
    for args in step_over_diffs(result_dir, args.action):
        diff_viewer(*args)
