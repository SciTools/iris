# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
# !/usr/bin/env python
"""
Updates imagerepo.json based on the baseline images

"""

import argparse
from pathlib import Path

import iris.tests
import iris.tests.graphics as graphics


def update_json(baseline_image_dir: Path, dry_run: bool = False):
    old_repo = graphics.read_repo_json()
    new_repo = graphics.generate_repo_from_baselines(baseline_image_dir)

    if graphics.repos_equal(old_repo, new_repo):
        msg = (
            f"No change in contents of {graphics.IMAGE_REPO_PATH} based on "
            f"{baseline_image_dir}"
        )
        print(msg)
    else:
        for key in set(old_repo.keys()) | set(new_repo.keys()):
            old_val = old_repo.get(key, None)
            new_val = new_repo.get(key, None)
            if str(old_val) != str(new_val):
                print(key)
                print(f"\t{old_val} -> {new_val}")
        if not dry_run:
            graphics.write_repo_json(new_repo)


if __name__ == "__main__":
    default_baseline_image_dir = Path(
        iris.tests.IrisTest.get_data_path("images")
    )
    description = "Update imagerepo.json based on contents of the baseline image directory"
    formatter_class = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(
        description=description, formatter_class=formatter_class
    )
    help = "path to iris tests result image directory (default: %(default)s)"
    parser.add_argument(
        "--image-dir", default=default_baseline_image_dir, help=help
    )
    help = "dry run (don't actually update imagerepo.json)"
    parser.add_argument("--dry-run", action="store_true", help=help)
    args = parser.parse_args()
    update_json(
        args.image_dir,
        args.dry_run,
    )
