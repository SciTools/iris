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

import codecs
import io
import json
import os
from pathlib import Path
import sys
import threading
from typing import Callable, Dict, Union
import unittest

import filelock

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

# Threading non re-entrant blocking lock to ensure thread-safe plotting in the
# GraphicsTestMixin.
_lock = threading.Lock()

#: Default perceptual hash size.
HASH_SIZE = 16
#: Default maximum perceptual hash hamming distance.
HAMMING_DISTANCE = 2
# Prefix for image test results (that aren't yet verified as good to add to
# reference images)
RESULT_PREFIX = "result-"
# Name of the imagerepo json and associated file lock
IMAGE_REPO_DIR = Path(__file__).parents[1] / "results"
IMAGE_REPO_PATH = IMAGE_REPO_DIR / "imagerepo.json"
IMAGE_REPO_LOCK_PATH = IMAGE_REPO_DIR / "imagerepo.lock"


__all__ = [
    "GraphicsTestMixin",
    "MPL_AVAILABLE",
    "RESULT_PREFIX",
    "check_graphic",
    "fully_qualify",
    "generate_repo_from_baselines",
    "get_phash",
    "read_repo_json",
    "repos_equal",
    "skip_plot",
    "write_repo_json",
]


def _output_dir() -> Path:
    test_output_dir = Path(__file__).parents[1] / Path(
        "result_image_comparison"
    )

    if not os.access(test_output_dir, os.W_OK):
        if not os.access(Path("."), os.W_OK):
            raise IOError(
                "Write access to a local disk is required "
                "to run image tests. Run the tests from a "
                "current working directory you have write "
                "access to to avoid this issue."
            )
        else:
            test_output_dir = Path(".") / "iris_image_test_output"

    return test_output_dir


def read_repo_json() -> Dict[str, str]:
    with open(IMAGE_REPO_PATH, "rb") as fi:
        repo: Dict[str, str] = json.load(codecs.getreader("utf-8")(fi))
    return repo


def write_repo_json(data: Dict[str, str]) -> None:
    string_data = {}
    for key, val in data.items():
        string_data[key] = str(val)
    with open(IMAGE_REPO_PATH, "wb") as fo:
        json.dump(
            string_data,
            codecs.getwriter("utf-8")(fo),
            indent=4,
            sort_keys=True,
        )


def repos_equal(repo1: Dict[str, str], repo2: Dict[str, str]) -> bool:
    if sorted(repo1.keys()) != sorted(repo2.keys()):
        return False
    for key, val in repo1.items():
        if str(val) != str(repo2[key]):
            return False
    return True


def get_phash(input: Path) -> str:
    from PIL import Image
    import imagehash

    return imagehash.phash(Image.open(input), hash_size=HASH_SIZE)


def generate_repo_from_baselines(baseline_image_dir: Path) -> Dict[str, str]:
    repo = {}
    for path in baseline_image_dir.iterdir():
        phash = get_phash(path)
        repo[path.stem] = phash
    return repo


def fully_qualify(test_id: str, repo: str) -> Dict[str, str]:
    # If the test_id isn't in the repo as it stands, look for it
    if test_id not in repo:
        test_id_candidates = [x for x in repo.keys() if x.endswith(test_id)]
        if len(test_id_candidates) == 1:
            (test_id,) = test_id_candidates
    return test_id


def check_graphic(test_id: str, results_dir: Union[str, Path]) -> None:
    """
    Check the hash of the current matplotlib figure matches the expected
    image hash for the current graphic test.

    To create missing image test results, set the IRIS_TEST_CREATE_MISSING
    environment variable before running the tests. This will result in new
    and appropriately "<hash>.png" image files being generated in the image
    output directory, and the imagerepo.json file being updated.

    """
    from imagehash import hex_to_hash

    dev_mode = os.environ.get("IRIS_TEST_CREATE_MISSING")

    #: The path where the images generated by the tests should go.
    test_output_dir = _output_dir()
    test_output_dir.mkdir(exist_ok=True)

    # The path where the image matching this test should be saved if necessary
    result_path = test_output_dir / f"{RESULT_PREFIX}{test_id}.png"

    results_dir = Path(results_dir)
    repo = read_repo_json()

    # Check if test_id is fully qualified, if it's not then try to work
    # out what it should be
    test_id = fully_qualify(test_id, repo)

    try:

        def _create_missing(phash: str) -> None:

            output_path = test_output_dir / (test_id + ".png")

            print(f"Creating image file: {output_path}")
            figure.savefig(output_path)

            msg = "Creating imagerepo entry: {} -> {}"
            print(msg.format(test_id, phash))
            # The imagerepo.json file is a critical resource, so ensure
            # thread safe read/write behaviour via platform independent
            # file locking.
            lock = filelock.FileLock(IMAGE_REPO_LOCK_PATH)
            with lock.acquire(timeout=600):
                # Read the file again in case it changed, then edit before
                # releasing lock
                repo = read_repo_json()
                repo[test_id] = phash
                write_repo_json(repo)

        # Calculate the test result perceptual image hash.
        buffer = io.BytesIO()
        figure = plt.gcf()
        figure.savefig(buffer, format="png")
        buffer.seek(0)
        phash = get_phash(buffer)

        if test_id in repo:

            expected = hex_to_hash(repo[test_id])

            # Calculate hamming distance vector for the result hash.
            distance = expected - phash

            if distance > HAMMING_DISTANCE:
                if dev_mode:
                    _create_missing(phash)
                else:
                    figure.savefig(result_path)
                    msg = (
                        "Bad phash {} with hamming distance {} " "for test {}."
                    )
                    msg = msg.format(phash, distance, test_id)
                    if _DISPLAY_FIGURES:
                        emsg = "Image comparison would have failed: {}"
                        print(emsg.format(msg))
                    else:
                        emsg = "Image comparison failed: {}"
                        raise AssertionError(emsg.format(msg))
        else:
            if dev_mode:
                _create_missing(phash)
            else:
                figure.savefig(result_path)
                emsg = "Missing image test result: {}."
                raise AssertionError(emsg.format(test_id))

        if _DISPLAY_FIGURES:
            plt.show()

    finally:
        plt.close()


class GraphicsTestMixin:
    def setUp(self) -> None:
        # Acquire threading non re-entrant blocking lock to ensure
        # thread-safe plotting.
        _lock.acquire()
        # Make sure we have no unclosed plots from previous tests before
        # generating this one.
        if MPL_AVAILABLE:
            plt.close("all")

    def tearDown(self) -> None:
        # If a plotting test bombs out it can leave the current figure
        # in an odd state, so we make sure it's been disposed of.
        if MPL_AVAILABLE:
            plt.close("all")
        # Release the non re-entrant blocking lock.
        _lock.release()


def skip_plot(fn: Callable) -> Callable:
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
