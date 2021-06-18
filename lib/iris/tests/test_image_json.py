# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import codecs
import itertools
import json
import os

import requests


@tests.skip_inet
class TestImageFile(tests.IrisTest):
    def test_resolve(self):
        listingfile_uri = (
            "https://raw.githubusercontent.com/SciTools/test-iris-imagehash"
            "/gh-pages/v4_files_listing.txt"
        )
        req = requests.get(listingfile_uri)
        if req.status_code != 200:
            raise ValueError(
                "GET failed on image listings file: {}".format(listingfile_uri)
            )

        listings_text = req.content.decode("utf-8")
        reference_image_filenames = [
            line.strip() for line in listings_text.split("\n")
        ]
        base = "https://scitools.github.io/test-iris-imagehash/images/v4"
        reference_image_uris = set(
            "{}/{}".format(base, name) for name in reference_image_filenames
        )

        imagerepo_json_filepath = os.path.join(
            os.path.dirname(__file__), "results", "imagerepo.json"
        )
        with open(imagerepo_json_filepath, "rb") as fi:
            imagerepo = json.load(codecs.getreader("utf-8")(fi))

        # "imagerepo" maps key: list-of-uris. Put all the uris in one big set.
        tests_uris = set(itertools.chain.from_iterable(imagerepo.values()))

        missing_refs = list(tests_uris - reference_image_uris)
        n_missing_refs = len(missing_refs)
        if n_missing_refs > 0:
            amsg = (
                "Missing images: These {} image uris are referenced in "
                "imagerepo.json, but not listed in {} : "
            )
            amsg = amsg.format(n_missing_refs, listingfile_uri)
            amsg += "".join("\n        {}".format(uri) for uri in missing_refs)
            # Always fails when we get here: report the problem.
            self.assertEqual(n_missing_refs, 0, msg=amsg)


if __name__ == "__main__":
    tests.main()
