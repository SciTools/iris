#!/usr/bin/env python
# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Updates imagerepo.json based on the baseline images."""

import argparse
from pathlib import Path

from imagehash import hex_to_hash

from iris.tests import _shared_utils
import iris.tests.graphics as graphics


def update_json(baseline_image_dir: Path, dry_run: bool = False):
    repo = graphics.read_repo_json()
    suggested_repo = graphics.generate_repo_from_baselines(baseline_image_dir)

    if graphics.repos_equal(repo, suggested_repo):
        msg = (
            f"No change in contents of {graphics.IMAGE_REPO_PATH} based on "
            f"{baseline_image_dir}"
        )
        print(msg)
    else:
        for key in sorted(set(repo.keys()) | set(suggested_repo.keys())):
            old_val = repo.get(key)
            new_val = suggested_repo.get(key)
            if old_val is None:
                repo[key] = suggested_repo[key]
                print(key)
                print(f"\t{old_val} -> {new_val}")
            elif new_val is None:
                del repo[key]
                print(key)
                print(f"\t{old_val} -> {new_val}")
            else:
                difference = hex_to_hash(str(old_val)) - hex_to_hash(str(new_val))
                if difference > 0:
                    print(key)
                    print(f"\t{old_val} -> {new_val} ({difference})")
                    repo[key] = suggested_repo[key]
        if not dry_run:
            graphics.write_repo_json(repo)


if __name__ == "__main__":
    default_baseline_image_dir = Path(_shared_utils.get_data_path("images"))
    description = (
        "Update imagerepo.json based on contents of the baseline image directory"
    )
    formatter_class = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(
        description=description, formatter_class=formatter_class
    )
    help = "path to iris tests result image directory (default: %(default)s)"
    parser.add_argument("--image-dir", default=default_baseline_image_dir, help=help)
    help = "dry run (don't actually update imagerepo.json)"
    parser.add_argument("--dry-run", action="store_true", help=help)
    args = parser.parse_args()
    update_json(
        args.image_dir,
        args.dry_run,
    )
