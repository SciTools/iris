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
from pathlib import Path
import re
import sys
import warnings

# Force iris.tests to use the ```tkagg``` backend by using the '-d'
# command-line argument as idiff is an interactive tool that requires a
# gui interface.
sys.argv.append("-d")
from PIL import Image  # noqa
import matplotlib.image as mimg  # noqa
import matplotlib.pyplot as plt  # noqa
import matplotlib.testing.compare as mcompare  # noqa
from matplotlib.testing.exceptions import ImageComparisonFailure  # noqa
import matplotlib.widgets as mwidget  # noqa

import iris.tests  # noqa
import iris.tests.graphics as graphics  # noqa

# Allows restoration of test id from result image name
_RESULT_NAME_PATTERN = re.compile(graphics.RESULT_PREFIX + r"(.*).png")


def extract_test_key(result_image_name):
    """
    Extracts the name of the test which a result image refers to
    """
    name_match = _RESULT_NAME_PATTERN.match(str(result_image_name))
    if name_match:
        test_key = name_match.group(1)
    else:
        emsg = f"Incorrectly named image in result dir: {result_image_name}"
        raise ValueError(emsg)
    return test_key


_POSTFIX_DIFF = "-failed-diff.png"


def diff_viewer(
    test_id,
    status,
    phash,
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

    repo = graphics.read_repo_json()

    def accept(event):
        if test_id not in repo:
            repo[test_id] = phash
            graphics.write_repo_json(repo)
            out_file = result_dir / (test_id + ".png")
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
        if test_id not in repo:
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


def step_over_diffs(result_dir, display=True):
    processed = False

    if display:
        msg = "\nComparing the expected image with the test result image."
        print(msg)

    # Remove old image diff results.
    for fname in result_dir.glob(f"*{_POSTFIX_DIFF}"):
        fname.unlink()

    reference_image_dir = Path(iris.tests.get_data_path("images"))
    repo = graphics.read_repo_json()

    # Filter out all non-test result image files.
    results = []
    for fname in sorted(result_dir.glob(f"{graphics.RESULT_PREFIX}*.png")):
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

    for count_index, result_path in enumerate(results):
        test_key = extract_test_key(result_path.name)
        test_key = graphics.fully_qualify(test_key, repo)
        reference_image_path = reference_image_dir / (test_key + ".png")

        try:
            # Calculate the test result perceptual image hash.
            phash = graphics.get_phash(result_path)
            distance = graphics.get_phash(reference_image_path) - phash
        except FileNotFoundError:
            wmsg = "Ignoring unregistered test result {!r}."
            warnings.warn(wmsg.format(test_key))
            continue

        processed = True

        try:
            # Creates the diff file when the images aren't identical
            mcompare.compare_images(reference_image_path, result_path, tol=0)
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
        args = phash, reference_image_path, result_path, diff_path
        if display:
            status = f"Image {count_index + 1} of {count}: hamming distance = {distance}"
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
    args = parser.parse_args()
    result_dir = Path(args.resultdir)
    if not result_dir.is_dir():
        emsg = f"Invalid results directory: {result_dir}"
        raise ValueError(emsg)

    for args in step_over_diffs(result_dir):
        diff_viewer(*args)
