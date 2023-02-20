# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Provides testing capabilities for installed copies of Iris.

"""

# Because this file is imported by setup.py, there may be additional runtime
# imports later in the file.
import os
import sys


# NOTE: Do not inherit from object as distutils does not like it.
class TestRunner:
    """Run the Iris tests under pytest and pytest-xdist for performance"""

    description = (
        "Run tests under pytest and pytest-xdist for performance. "
        "Default behaviour is to run all non-gallery tests. "
        "Specifying one or more test flags will run *only* those "
        "tests."
    )
    user_options = [
        (
            "no-data",
            "n",
            "Override the paths to the data repositories so it "
            "appears to the tests that it does not exist.",
        ),
        ("stop", "x", "Stop running tests after the first error or failure."),
        ("system-tests", "s", "Run the limited subset of system tests."),
        ("gallery-tests", "e", "Run the gallery code tests."),
        ("default-tests", "d", "Run the default tests."),
        (
            "num-processors=",
            "p",
            "The number of processors used for running " "the tests.",
        ),
        ("create-missing", "m", "Create missing test result files."),
        ("coverage", "c", "Enable coverage testing"),
    ]
    boolean_options = [
        "no-data",
        "system-tests",
        "stop",
        "gallery-tests",
        "default-tests",
        "create-missing",
        "coverage",
    ]

    def initialize_options(self):
        self.no_data = False
        self.stop = False
        self.system_tests = False
        self.gallery_tests = False
        self.default_tests = False
        self.num_processors = None
        self.create_missing = False
        self.coverage = False

    def finalize_options(self):
        # These environment variables will be propagated to all the
        # processes that pytest-xdist creates.
        if self.no_data:
            print("Running tests in no-data mode...")
            import iris.config

            iris.config.TEST_DATA_DIR = None
        if self.create_missing:
            os.environ["IRIS_TEST_CREATE_MISSING"] = "true"

        tests = []
        if self.system_tests:
            tests.append("system")
        if self.default_tests:
            tests.append("default")
        if self.gallery_tests:
            tests.append("gallery")
        if not tests:
            tests.append("default")
        print("Running test suite(s): {}".format(", ".join(tests)))
        if self.stop:
            print("Stopping tests after the first error or failure")
        if self.num_processors is None:
            self.num_processors = "auto"
        else:
            self.num_processors = int(self.num_processors)

    def run(self):
        import pytest

        if hasattr(self, "distribution") and self.distribution.tests_require:
            self.distribution.fetch_build_eggs(self.distribution.tests_require)

        tests = []
        if self.system_tests:
            tests.append("lib/iris/tests/system_test.py")
        if self.default_tests:
            tests.append("lib/iris/tests")
        if self.gallery_tests:
            import iris.config

            default_doc_path = os.path.join(sys.path[0], "docs")
            doc_path = iris.config.get_option(
                "Resources", "doc_dir", default=default_doc_path
            )
            gallery_path = os.path.join(doc_path, "gallery_tests")
            if os.path.exists(gallery_path):
                tests.append(gallery_path)
            else:
                print(
                    "WARNING: Gallery path %s does not exist." % (gallery_path)
                )
        if not tests:
            tests.append("lib/iris/tests")

        args = [
            None,
            f"-n={self.num_processors}",
        ]

        if self.stop:
            args.append("-x")

        if self.coverage:
            args.extend(["--cov=lib/iris", "--cov-report=xml"])

        result = True
        for test in tests:
            args[0] = test
            print()
            print(
                f"Running test discovery on {test} with {self.num_processors} processors."
            )
            retcode = pytest.main(args=args)
            result &= retcode.value == 0
        if result is False:
            exit(1)
