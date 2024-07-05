# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Installer for custom benchmarks.

Provided since having multiple pyproject.toml files confuses some tools.
"""

import setuptools

setuptools.setup(
    name="custom_bms",
    version="1.0",
    packages=["asv_bench"],
)
