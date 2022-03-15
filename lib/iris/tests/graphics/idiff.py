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
import hashlib
from pathlib import Path
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
import iris.tests.graphics as graphics

_POSTFIX_DIFF = "-failed-diff.png"


def hash_image(image_path):
    """
    Get the sha256 of the contents of an image
    """
    return hashlib.sha256(open(image_path, "rb").read()).hexdigest()


def image_already_present(check_path, image_dir):
    """
    Check if an image is already in the given directory
    """
    check_hash = hash_image(check_path)
    for dir_image_path in image_dir.iterdir():
        if check_hash == hash_image(dir_image_path):
            return True
    return False


def diff_viewer(
    key,
    status,
    expected_path,
    result_path,
    diff_fname,
):
    fig = plt.figure(figsize=(14, 12))
    plt.suptitle(expected_path.name)
    ax = plt.subplot(221)
    ax.imshow(mimg.imread(expected_path))
    ax = plt.subplot(222, sharex=ax, sharey=ax)
    ax.imshow(mimg.imread(result_path))
    ax = plt.subplot(223, sharex=ax, sharey=ax)
    ax.imshow(mimg.imread(diff_fname))

    result_dir = result_path.parent
    reference_image_lookup = graphics._get_reference_image_lookup(
        expected_path.parent
    )

    def accept(event):
        if not image_already_present(result_path, expected_path.parent):
            # Ensure to maintain strict time order where the first uri
            # associated with the repo key is the oldest, and the last
            # uri is the youngest
            out_file = result_dir / graphics._next_reference_image_name(
                reference_image_lookup, key
            )
            result_path.rename(out_file)
            msg = f"ACCEPTED:  {result_path.name} -> {out_file.name}"
            print(msg)
        else:
            msg = f"DUPLICATE: {result_path.name} -> {expected_path.name} (ignored)"
            print(msg)
            result_path.unlink()
        diff_fname.unlink()
        plt.close()

    def reject(event):
        if not image_already_present(result_path, expected_path.parent):
            print(f"REJECTED:  {result_path.name}")
        else:
            msg = f"DUPLICATE: {result_path.name} -> {expected_path.name} (ignored)"
            print(msg)
        result_path.unlink()
        diff_fname.unlink()
        plt.close()

    def skip(event):
        # Let's keep both the result and the diff files.
        print(f"SKIPPED:   {result_path.name}")
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
        imagehash.phash(Image.open(image_path), hash_size=graphics._HASH_SIZE)
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
    for fname in result_dir.glob(f"*{_POSTFIX_DIFF}"):
        fname.unlink()

    # Filter out all non-test result image files.
    results = []
    for fname in sorted(result_dir.glob(f"{graphics._RESULT_PREFIX}*.png")):
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

    reference_image_dir = Path(iris.tests.get_data_path("images"))

    reference_images = graphics._get_reference_image_lookup(
        reference_image_dir
    )

    for count_index, result_path in enumerate(results):
        test_key = graphics.extract_test_key(result_path.name)

        try:
            # Calculate the test result perceptual image hash.
            phash = imagehash.phash(
                Image.open(result_path), hash_size=graphics._HASH_SIZE
            )
            relevant_image_paths = reference_images[test_key]
            hash_index, distance = _calculate_hit(
                relevant_image_paths, phash, action
            )
            relevant_image_path = reference_images[test_key][hash_index]
        except KeyError:
            wmsg = "Ignoring unregistered test result {!r}."
            warnings.warn(wmsg.format(test_key))
            continue

        processed = True

        try:
            # Creates the diff file when the images aren't identical
            mcompare.compare_images(relevant_image_path, result_path, tol=0)
        except Exception as e:
            if isinstance(e, ValueError) or isinstance(
                e, ImageComparisonFailure
            ):
                print(f"Could not compare {result_path}: {e}")
                continue
            else:
                # Propagate the exception, keeping the stack trace
                raise
        diff_path = result_dir / Path(f"{result_path.stem}{_POSTFIX_DIFF}")
        args = relevant_image_path, result_path, diff_path
        if display:
            status = f"Image {count_index + 1} of {count}: hamming distance = {distance} [{kind}]"
            prefix = test_key, status
            yield prefix + args
        else:
            yield args
    if display and not processed:
        print("\nThere are no iris test result images to process.\n")


if __name__ == "__main__":
    default = Path(iris.tests.__file__).parent / Path(
        "result_image_comparison"
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
    result_dir = Path(args.resultdir)
    if not result_dir.is_dir():
        emsg = f"Invalid results directory: {result_dir}"
        raise ValueError(emsg)
    for args in step_over_diffs(result_dir, args.action):
        diff_viewer(*args)
