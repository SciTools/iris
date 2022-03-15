# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
# !/usr/bin/env python
"""
Contains Iris graphic testing utilities

By default, this module sets the matplotlib backend to "agg". But when
this module is imported it checks ``sys.argv`` for the flag "-d". If
found, it is removed from ``sys.argv`` and the matplotlib backend is
switched to "tkagg" to allow the interactive visual inspection of
graphical test results.
"""

from collections import defaultdict
import io
import os
from pathlib import Path
import re
import sys
import threading
import unittest

import numpy as np

# Test for availability of matplotlib.
# (And remove matplotlib as an iris.tests dependency.)
try:
    import matplotlib

    # Override any user settings e.g. from matplotlibrc file.
    matplotlib.rcdefaults()
    # Set backend *after* rcdefaults, as we don't want that overridden (#3846).
    matplotlib.use("agg")
    # Standardise the figure size across matplotlib versions.
    # This permits matplotlib png image comparison.
    matplotlib.rcParams["figure.figsize"] = [8.0, 6.0]
    import matplotlib.pyplot as plt
except ImportError:
    MPL_AVAILABLE = False
else:
    MPL_AVAILABLE = True

# Whether to display matplotlib output to the screen.
_DISPLAY_FIGURES = False

if MPL_AVAILABLE and "-d" in sys.argv:
    sys.argv.remove("-d")
    plt.switch_backend("tkagg")
    _DISPLAY_FIGURES = True

#: Default perceptual hash size.
_HASH_SIZE = 16
#: Default maximum perceptual hash hamming distance.
_HAMMING_DISTANCE = 2
# Prefix for image test results (that aren't yet verified as good to add to
# reference images)
_RESULT_PREFIX = "result-"


def _results_dir():
    test_results_dir = Path(__file__).parents[1] / Path(
        "result_image_comparison"
    )

    if not os.access(test_results_dir, os.W_OK):
        if not os.access(Path("."), os.W_OK):
            raise IOError(
                "Write access to a local disk is required "
                "to run image tests. Run the tests from a "
                "current working directory you have write "
                "access to to avoid this issue."
            )
        else:
            test_results_dir = Path(".") / Path("iris_image_test_output")

    return test_results_dir


_IMAGE_NAME_PATTERN = re.compile(r"(.*)_([0-9]+).png")


def _get_reference_image_lookup(reference_image_dir):
    tmp_storage = defaultdict(dict)

    reference_image_dir = Path(reference_image_dir)
    for reference_image_path in reference_image_dir.iterdir():
        name_match = _IMAGE_NAME_PATTERN.match(reference_image_path.name)
        if name_match:
            test_name = name_match.group(1)
            image_index = int(name_match.group(2))
            tmp_storage[test_name][image_index] = reference_image_path
        else:
            emsg = f"Incorrectly named image in reference dir: {reference_image_path}"
            raise ValueError(emsg)

    reference_image_lookup = {}

    for test_name, index_dict in tmp_storage.items():
        path_list = [None] * (max(index_dict.keys()) + 1)
        try:
            for ind, image_path in index_dict.items():
                path_list[ind] = image_path
            assert None not in path_list
        except (KeyError, AssertionError):
            emsg = f"Reference images for {test_name} numbered incorrectly"
            raise ValueError(emsg)
        reference_image_lookup[test_name] = path_list

    return reference_image_lookup


def _next_reference_image_name(reference_image_lookup, test_id):
    try:
        image_index = len(reference_image_lookup[test_id])
    except KeyError:
        image_index = 0
    fname = Path(f"{test_id}_{image_index}.png")
    return fname


_RESULT_NAME_PATTERN = re.compile(r"result-(.*).png")


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


def check_graphic(test_obj):
    """
    Check the hash of the current matplotlib figure matches the expected
    image hash for the current graphic test.

    To create missing image test results, set the IRIS_TEST_CREATE_MISSING
    environment variable before running the tests. This will result in new
    and appropriately "<hash>.png" image files being generated in the image
    output directory, and the imagerepo.json file being updated.

    """

    from PIL import Image
    import imagehash

    reference_image_lookup = _get_reference_image_lookup(
        test_obj.get_data_path("images")
    )

    test_id = test_obj._unique_id()

    dev_mode = os.environ.get("IRIS_TEST_CREATE_MISSING")

    try:
        #: The path where the images generated by the tests should go.
        test_results_dir = _results_dir()

        test_results_dir.mkdir(exist_ok=True)

        result_path = test_results_dir / Path(f"{_RESULT_PREFIX}{test_id}.png")

        # Check if test_id is fully qualified, if it's not then try to work
        # out what it should be
        if test_id not in reference_image_lookup:

            test_id_candidates = [
                x for x in reference_image_lookup.keys() if x.endswith(test_id)
            ]

            if len(test_id_candidates) == 1:
                (test_id,) = test_id_candidates

        def _create_missing():

            fname = _next_reference_image_name(test_id)

            output_path = test_results_dir / fname

            print(f"Creating image file: {output_path}")
            figure.savefig(output_path)

        # Calculate the test result perceptual image hash.
        buffer = io.BytesIO()
        figure = plt.gcf()
        figure.savefig(buffer, format="png")
        buffer.seek(0)
        phash = imagehash.phash(Image.open(buffer), hash_size=_HASH_SIZE)

        reference_image_names = reference_image_lookup[test_id]

        if reference_image_names:

            expected = [
                imagehash.phash(
                    Image.open(test_results_dir / image_name),
                    hash_size=_HASH_SIZE,
                )
                for image_name in reference_image_names
            ]

            # Calculate hamming distance vector for the result hash.
            distances = [e - phash for e in expected]

            if np.all([hd > _HAMMING_DISTANCE for hd in distances]):
                if dev_mode:
                    _create_missing()
                else:
                    figure.savefig(result_path)
                    msg = (
                        "Bad phash {} with hamming distance {} " "for test {}."
                    )
                    msg = msg.format(phash, distances, test_id)
                    if _DISPLAY_FIGURES:
                        emsg = "Image comparison would have failed: {}"
                        print(emsg.format(msg))
                    else:
                        emsg = "Image comparison failed: {}"
                        raise AssertionError(emsg.format(msg))
        else:
            if dev_mode:
                _create_missing()
            else:
                figure.savefig(result_path)
                emsg = "Missing image test result: {}."
                raise AssertionError(emsg.format(test_id))

        if _DISPLAY_FIGURES:
            plt.show()

    finally:
        plt.close()


# Threading non re-entrant blocking lock to ensure thread-safe plotting.
_lock = threading.Lock()


class GraphicsTestMixin:

    # nose directive: dispatch tests concurrently.
    _multiprocess_can_split_ = True

    def setUp(self):
        # Acquire threading non re-entrant blocking lock to ensure
        # thread-safe plotting.
        _lock.acquire()
        # Make sure we have no unclosed plots from previous tests before
        # generating this one.
        if MPL_AVAILABLE:
            plt.close("all")

    def tearDown(self):
        # If a plotting test bombs out it can leave the current figure
        # in an odd state, so we make sure it's been disposed of.
        if MPL_AVAILABLE:
            plt.close("all")
        # Release the non re-entrant blocking lock.
        _lock.release()


def skip_plot(fn):
    """
    Decorator to choose whether to run tests, based on the availability of the
    matplotlib library.

    Example usage:
        @skip_plot
        class MyPlotTests(test.GraphicsTest):
            ...

    """
    skip = unittest.skipIf(
        condition=not MPL_AVAILABLE,
        reason="Graphics tests require the matplotlib library.",
    )

    return skip(fn)
