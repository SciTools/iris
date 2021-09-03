# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
A command line utility for downloading cartopy resources e.g.,

    > python cartopy_feature_download.py --help


"""

import argparse
import os
import pathlib
import sys
import tempfile
import urllib.request

import cartopy
from cartopy import config

FEATURE_DOWNLOAD_URL = f"https://raw.githubusercontent.com/SciTools/cartopy/v{cartopy.__version__}/tools/feature_download.py"
# This will be the (more stable) cartopy resource endpoint from v0.19.0.post1+
# See https://github.com/SciTools/cartopy/pull/1833
URL_TEMPLATE = "https://naturalearth.s3.amazonaws.com/{resolution}_{category}/ne_{resolution}_{name}.zip"
SHP_NE_SPEC = ("shapefiles", "natural_earth")


def main(target_dir, features, dry_run):
    target_dir = pathlib.Path(target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    cwd = pathlib.Path.cwd()

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)

        # Download cartopy feature_download tool, which is not bundled
        # within a cartopy package, and make it importable.
        urllib.request.urlretrieve(FEATURE_DOWNLOAD_URL, "feature_download.py")
        with open("__init__.py", "w"):
            pass
        sys.path.append(tmpdir)

        from feature_download import download_features

        # Configure the cartopy resource cache.
        config["pre_existing_data_dir"] = str(target_dir)
        config["data_dir"] = str(target_dir)
        config["repo_data_dir"] = str(target_dir)
        # Force use of stable endpoint for pre-v0.20 cartopy.
        config["downloaders"][SHP_NE_SPEC].url_template = URL_TEMPLATE

        # Perform download, or dry-run.
        download_features(features, dry_run=dry_run)

        os.chdir(cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download cartopy data for caching."
    )
    parser.add_argument(
        "--dryrun",
        "-d",
        action="store_true",
        help="perform a dry-run of the download",
    )
    parser.add_argument(
        "--feature",
        "-f",
        nargs="+",
        default=["physical"],
        help=(
            "specify one or more features to download [cultural|cultural-extra|gshhs|physical], "
            'default is "physical"'
        ),
    )
    parser.add_argument(
        "--nowarn",
        "-n",
        action="store_true",
        help="ignore cartopy DownloadWarning warnings",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="save datasets in the specified directory",
    )

    args = parser.parse_args()

    if args.nowarn:
        import warnings

        warnings.filterwarnings("ignore", category=cartopy.io.DownloadWarning)

    main(args.output, args.feature, args.dryrun)
