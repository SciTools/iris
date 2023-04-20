# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# This script will process all .rst files that have been created by
# sphinxcontrib.apidoc extension and perform minor changes, specifically:
#
# - Remove the suffix for "package" and " module".
#

import ntpath
from pathlib import Path


def main_api_rst_formatting(app):
    src_dir = Path("generated/api")

    print(f"[{ntpath.basename(__file__)}] Processing RST files", end="")

    for file in src_dir.iterdir():
        print(f".", end="")

        with open(file, "r") as f:
            lines = f.read()

        lines = lines.replace(" package\n=", "\n")
        lines = lines.replace(" module\n=", "\n")

        with open(file, "w") as f:
            f.write(lines)
    print("")

def setup(app):
    app.connect("builder-inited", main_api_rst_formatting)
