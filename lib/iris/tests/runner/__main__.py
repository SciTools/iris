# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Provides testing capabilities for installed copies of Iris.

"""

import argparse

from ._runner import TestRunner

parser = argparse.ArgumentParser(
    "iris.tests", description=TestRunner.description
)
for long_opt, short_opt, help_text in TestRunner.user_options:
    long_opt = long_opt.strip("=")
    if long_opt in TestRunner.boolean_options:
        parser.add_argument(
            "--" + long_opt,
            "-" + short_opt,
            action="store_true",
            help=help_text,
        )
    else:
        parser.add_argument("--" + long_opt, "-" + short_opt, help=help_text)
args = parser.parse_args()

runner = TestRunner()

runner.initialize_options()
for long_opt, short_opt, help_text in TestRunner.user_options:
    arg = long_opt.replace("-", "_").strip("=")
    setattr(runner, arg, getattr(args, arg))
runner.finalize_options()

runner.run()
