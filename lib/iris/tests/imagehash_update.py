import argparse
import json
from pathlib import Path

from PIL import Image
import imagehash

import iris.tests

DEFAULT_IMAGEHASH_DIR = Path("../../../../test-iris-imagehash")


def updateImageRepoData(image_repo_data, imagehash_dir):

    for test_entry in image_repo_data:

        for valid_image in test_entry:

            image_uri = imagehash_dir / Path(valid_image["image_uri"])

            current_hash = imagehash.phash(
                Image.open(image_uri), hash_size=iris.tests._HASH_SIZE
            )

            if current_hash not in valid_image["known_hashes"]:
                valid_image["known_hashes"].append(current_hash)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imagehash-dir",
        "-i",
        type=Path,
        default=DEFAULT_IMAGEHASH_DIR,
        help="Base directory of test-iris-imagehash repo",
    )

    p = parser.parse_args()

    repo_fname = Path(iris.tests._RESULT_PATH) / Path("imagerepo.json")

    with open(repo_fname) as irj:
        current_image_repo = json.load(irj)

    updateImageRepoData(current_image_repo, p.imagehash_dir)

    with open(repo_fname, "w+") as irj:
        json.dump(current_image_repo, irj)
