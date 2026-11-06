# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

from pathlib import Path

from iris.tests import _shared_utils
import iris.tests.graphics as graphics


@_shared_utils.skip_data
class TestImageFile:
    def test_json(self):
        # get test names from json
        repo_names = [*graphics.read_repo_json().keys()]
        # get file names from test data
        test_data_names = [
            pp.stem for pp in Path(_shared_utils.get_data_path(["images"])).iterdir()
        ]
        # compare
        repo_name_set = set(repo_names)
        assert len(repo_names) == len(repo_name_set)
        test_data_name_set = set(test_data_names)
        assert len(test_data_names) == len(test_data_name_set)
        missing_from_json = test_data_name_set - repo_name_set
        if missing_from_json:
            amsg = (
                "Missing images: Images are present in the iris-test-data "
                "repo, that are not referenced in imagerepo.json"
            )
            # Always fails when we get here: report the problem.
            assert missing_from_json == set(), amsg
        missing_from_test_data = repo_name_set - test_data_name_set
        if missing_from_test_data:
            amsg = (
                "Missing images: Image names are referenced in "
                "imagerepo.json, that are not present in the iris-test-data "
                "repo"
            )
            # Always fails when we get here: report the problem.
            assert missing_from_test_data == set(), amsg
