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
import codecs
import contextlib
from glob import glob
import json
import os.path
import shutil
import sys
import warnings

# Force iris.tests to use the ```tkagg``` backend by using the '-d'
# command-line argument as idiff is an interactive tool that requires a
# gui interface.
sys.argv.append("-d")
from PIL import Image  # noqa
import filelock  # noqa
import imagehash  # noqa
import matplotlib.image as mimg  # noqa
import matplotlib.pyplot as plt  # noqa
import matplotlib.testing.compare as mcompare  # noqa
from matplotlib.testing.exceptions import ImageComparisonFailure  # noqa
import matplotlib.widgets as mwidget  # noqa
import numpy as np  # noqa
import requests  # noqa

import iris.tests  # noqa
import iris.util as iutil  # noqa

_POSTFIX_DIFF = "-failed-diff.png"
_POSTFIX_JSON = os.path.join("results", "imagerepo.json")
_POSTFIX_LOCK = os.path.join("results", "imagerepo.lock")


@contextlib.contextmanager
def temp_png(suffix=""):
    if suffix:
        suffix = "-{}".format(suffix)
    fname = iutil.create_temp_filename(suffix + ".png")
    try:
        yield fname
    finally:
        os.remove(fname)


def diff_viewer(
    repo,
    key,
    repo_fname,
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
    fname = "{}.png".format(phash)
    base_uri = "https://scitools.github.io/test-iris-imagehash/images/v4/{}"
    uri = base_uri.format(fname)
    phash_fname = os.path.join(result_dir, fname)

    def accept(event):
        if uri not in repo[key]:
            # Ensure to maintain strict time order where the first uri
            # associated with the repo key is the oldest, and the last
            # uri is the youngest
            repo[key].append(uri)
            # Update the image repo.
            with open(repo_fname, "wb") as fo:
                json.dump(
                    repo,
                    codecs.getwriter("utf-8")(fo),
                    indent=4,
                    sort_keys=True,
                )
            os.rename(result_fname, phash_fname)
            msg = "ACCEPTED:  {} -> {}"
            print(
                msg.format(
                    os.path.basename(result_fname),
                    os.path.basename(phash_fname),
                )
            )
        else:
            msg = "DUPLICATE: {} -> {} (ignored)"
            print(
                msg.format(
                    os.path.basename(result_fname),
                    os.path.basename(phash_fname),
                )
            )
            os.remove(result_fname)
        os.remove(diff_fname)
        plt.close()

    def reject(event):
        if uri not in repo[key]:
            print("REJECTED:  {}".format(os.path.basename(result_fname)))
        else:
            msg = "DUPLICATE: {} -> {} (ignored)"
            print(
                msg.format(
                    os.path.basename(result_fname),
                    os.path.basename(phash_fname),
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


def _calculate_hit(uris, phash, action):
    # Extract the hex basename strings from the uris.
    hexes = [os.path.splitext(os.path.basename(uri))[0] for uri in uris]
    # Create the expected perceptual image hashes from the uris.
    to_hash = imagehash.hex_to_hash
    expected = [to_hash(uri_hex) for uri_hex in hexes]
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
    dname = os.path.dirname(iris.tests.__file__)
    lock = filelock.FileLock(os.path.join(dname, _POSTFIX_LOCK))
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

    with lock.acquire(timeout=30):
        # Load the imagerepo.
        repo_fname = os.path.join(dname, _POSTFIX_JSON)
        with open(repo_fname, "rb") as fi:
            repo = json.load(codecs.getreader("utf-8")(fi))

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

        for count_index, result_fname in enumerate(results):
            key = os.path.splitext(
                "-".join(result_fname.split("result-")[1:])
            )[0]
            try:
                # Calculate the test result perceptual image hash.
                phash = imagehash.phash(
                    Image.open(result_fname), hash_size=iris.tests._HASH_SIZE
                )
                uris = repo[key]
                hash_index, distance = _calculate_hit(uris, phash, action)
                uri = uris[hash_index]
            except KeyError:
                wmsg = "Ignoring unregistered test result {!r}."
                warnings.warn(wmsg.format(key))
                continue
            with temp_png(key) as expected_fname:
                processed = True
                resource = requests.get(uri)
                if resource.status_code == 200:
                    with open(expected_fname, "wb") as fo:
                        fo.write(resource.content)
                else:
                    # Perhaps the uri has not been pushed into the repo yet,
                    # so check if a local "developer" copy is available ...
                    local_fname = os.path.join(
                        result_dir, os.path.basename(uri)
                    )
                    if not os.path.isfile(local_fname):
                        emsg = "Bad URI {!r} for test {!r}."
                        raise ValueError(emsg.format(uri, key))
                    else:
                        # The temporary expected filename has the test name
                        # baked into it, and is used in the diff plot title.
                        # So copy the local file to the exected file to
                        # maintain this helpfulness.
                        shutil.copy(local_fname, expected_fname)
                try:
                    mcompare.compare_images(
                        expected_fname, result_fname, tol=0
                    )
                except Exception as e:
                    if isinstance(e, ValueError) or isinstance(
                        e, ImageComparisonFailure
                    ):
                        print(
                            "Could not compare {}: {}".format(result_fname, e)
                        )
                        continue
                    else:
                        # Propagate the exception, keeping the stack trace
                        raise
                diff_fname = os.path.splitext(result_fname)[0] + _POSTFIX_DIFF
                args = expected_fname, result_fname, diff_fname
                if display:
                    msg = "Image {} of {}: hamming distance = {} " "[{!r}]"
                    status = msg.format(count_index + 1, count, distance, kind)
                    prefix = repo, key, repo_fname, phash, status
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
