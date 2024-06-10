#!/usr/bin/env python3
# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""
A do-nothing script to hand-hold through the Iris release process.

https://blog.danslimmon.com/2019/07/15/do-nothing-scripting-the-key-to-gradual-automation/

"""
from datetime import datetime
from enum import Enum
from pathlib import Path
import re
from sys import stderr
from time import sleep
import typing


class ReleaseTypes(Enum):
    MAJOR = 0
    MINOR = 1
    PATCH = 2


valid_release_types = typing.Literal["major", "minor", "patch"]


class ReleaseStrings:
    """An easy way to pass the various flavours of release string between functions."""

    def __init__(self, input_tag: str):
        version_mask = r"v\d+\.\d+\.\d+\D*.*"
        regex_101 = "https://regex101.com/r/dLVaNH/1"
        if re.fullmatch(version_mask, input_tag) is None:
            message = (
                "Release tag does not match the input mask:\n"
                f"{version_mask}\n"
                f"({regex_101})"
            )
            raise ValueError(message)
        else:
            self.tag = input_tag  # v1.2.3rc0

        self.series = ".".join(self.tag.split(".")[:2])  # v1.2
        self.branch = self.series + ".x"  # v1.2.x
        self.release = self.tag[1:]  # 1.2.3rc0


class WhatsNewRsts:
    """An easy way to pass the paths of various What's New files between functions."""

    def __init__(self, release_strings: ReleaseStrings):
        src_dir = Path(__file__).parents[1] / "docs" / "src"
        whatsnew_dir = src_dir / "whatsnew"
        assert whatsnew_dir.is_dir()

        self.latest = whatsnew_dir / "latest.rst"
        self.release = whatsnew_dir / (release_strings.series[1:] + ".rst")
        self.index = whatsnew_dir / "index.rst"
        self.template = self.latest.with_suffix(".rst.template")


def _break_print(message: str):
    print()
    print(message)
    # Help with flow/visibility by waiting 1secs before proceeding.
    sleep(1)


def _mark_section(section_number: int):
    _break_print(f"SECTION {section_number} ...")


def _get_input(message: str, expected_inputs: str) -> str:
    _break_print(message)
    return input(expected_inputs + " : ")


def _wait_for_done(message: str):
    _break_print(message)
    done = False
    while not done:
        done = (
            input("Step complete? y / [n] : ").casefold() == "y".casefold()
        )


def _report_problem(message: str):
    print(message, file=stderr)
    # To ensure correct sequencing of messages.
    sleep(0.5)


def get_release_type() -> ReleaseTypes:
    release_type = None
    release_types_str = " ".join(
        [f"{m.name}={m.value}" for m in ReleaseTypes.__members__.values()]
    )
    message = "What type of release are you preparing?\nhttps://semver.org/"
    while release_type is None:
        input_type = _get_input(message, release_types_str)
        try:
            release_type = ReleaseTypes(int(input_type))
        except ValueError:
            _report_problem("Invalid release type. Please try again ...")
    _break_print(f"{release_type} confirmed.")
    return release_type


def get_release_tag() -> ReleaseStrings:
    # TODO: automate using setuptools_scm.
    release_strings = None
    message = (
        "Input the release tag you are creating today, including any release "
        "candidate suffix.\n"
        "https://semver.org/\n"
        "https://scitools-iris.readthedocs.io/en/latest/developers_guide/release.html?highlight=candidate#release-candidate"
    )
    while release_strings is None:
        input_tag = _get_input(message, "e.g. v1.2.3rc0")
        try:
            release_strings = ReleaseStrings(input_tag)
        except ValueError as err:
            _report_problem(str(err))
    return release_strings


def check_release_candidate(
    release_type: ReleaseTypes, release_strings: ReleaseStrings
) -> bool:
    is_release_candidate = "rc" in release_strings.tag

    message = "Checking tag for release candidate: "
    if is_release_candidate:
        message += "DETECTED\nThis IS a release candidate."
    else:
        message += "NOT DETECTED\nThis IS NOT a release candidate."
    _break_print(message)

    if release_type == ReleaseTypes.PATCH and is_release_candidate:
        message = (
            "Release candidates are not expected for PATCH releases. "
            "Are you sure you want to continue?"
        )
        if _get_input(message, "y / [n]").casefold() != "y".casefold():
            exit()
    return is_release_candidate


def check_first_in_series(
    release_type: ReleaseTypes,
    release_strings: ReleaseStrings,
    is_release_candidate: bool,
) -> bool:
    first_in_series = False
    if release_type != ReleaseTypes.PATCH:
        message = (
            "Have there been any prior releases in the "
            f"{release_strings.series} series, including release candidates?"
        )
        first_in_series = (
            _get_input(message, "[y] / n").casefold() == "n".casefold()
        )
        if first_in_series:
            _break_print("First in series confirmed.")
            if not is_release_candidate:
                message = (
                    "The first release in a series is expected to be a "
                    "release candidate, but this is not. Are you sure you "
                    "want to continue?"
                )
                if _get_input(message, "y / [n]").casefold() != "y".casefold():
                    exit()
        else:
            _break_print("Existing series confirmed.")
    return first_in_series


def update_standard_names(first_in_series: bool) -> None:
    if first_in_series:
        message = (
            "Update the file ``etc/cf-standard-name-table.xml`` to the latest CF "
            "standard names, via a new Pull Request.\n"
            "(This is used during build to automatically generate the sourcefile "
            "``lib/iris/std_names.py``).\n"
            "Latest standard names:\n"
            'wget "https://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml";'
        )
        _wait_for_done(message)


def check_deprecations(release_type: ReleaseTypes) -> None:
    if release_type == ReleaseTypes.MAJOR:
        message = (
            "This is a MAJOR release - be sure to finalise all deprecations "
            "and FUTUREs from previous releases, via a new Pull Request.\n"
            "https://scitools-iris.readthedocs.io/en/latest/developers_guide/contributing_deprecations.html"
        )
        _wait_for_done(message)


def _delete_local_branch(branch_name: str):
    message = (
        "Before the next step, avoid a name clash by deleting any "
        "existing local branch, if one exists.\n"
        f"git branch -D {branch_name};\n"
        f"git push -d origin {branch_name};"
    )
    _wait_for_done(message)


def create_release_branch(
    release_strings: ReleaseStrings, first_in_series: bool
) -> None:
    # TODO: automate

    _break_print("Release branch management ...")

    if first_in_series:
        message = (
            "Visit https://github.com/SciTools/iris and create the"
            f"``{release_strings.branch}`` release branch from ``main``."
        )
        _wait_for_done(message)

    else:
        message = (
            "Cherry-pick any specific commits that are needed from ``main`` "
            f"onto {release_strings.branch} , to get the CI passing.\n"
            "E.g. a new dependency pin may have been introduced since "
            f"{release_strings.branch} was last updated from ``main``.\n"
            "DO NOT squash-merge - want to preserve the original commit SHA's."
        )
        _wait_for_done(message)


def finalise_whats_new(
    release_type: ReleaseTypes,
    release_strings: ReleaseStrings,
    is_release_candidate: bool,
    first_in_series: bool,
) -> WhatsNewRsts:
    _break_print("What's New finalisation ...")

    working_branch = release_strings.branch + ".updates"
    _delete_local_branch(working_branch)
    message = (
        f"Checkout a local branch from the official {release_strings.branch} branch.\n"
        "git fetch upstream;\n"
        f"git checkout upstream/{release_strings.branch} -b "
        f"{working_branch};"
    )
    _wait_for_done(message)

    rsts = WhatsNewRsts(release_strings)

    # TODO: automate
    if first_in_series:
        message = (
            "'Cut' the What's New for the release.\n"
            f"git mv {rsts.latest.absolute()} {rsts.release.absolute()};"
        )
        _wait_for_done(message)

        message = (
            f"In {rsts.index.absolute()}:\n"
            f"Replace references to {rsts.latest.name} with {rsts.release.name}"
        )
        _wait_for_done(message)

    _break_print(f"What's New file path = {rsts.release}")

    if not release_type == ReleaseTypes.PATCH:
        whatsnew_title = f"{release_strings.series} ({datetime.today().strftime('%d %b %Y')})"
        if is_release_candidate:
            whatsnew_title += " [release candidate]"
        # TODO: automate
        message = f"In {rsts.release.name}: set the page title to:\n{whatsnew_title}\n"
        if not is_release_candidate:
            message += (
                "\nBe sure to remove any existing mentions of release "
                "candidate from the title.\n"
            )
        _wait_for_done(message)

        message = (
            f"In {rsts.release.name}: ensure the page title underline is "
            "the exact same length as the page title text."
        )
        _wait_for_done(message)

        dropdown_title = f"\n{release_strings.series} Release Highlights\n"
        message = (
            f"In {rsts.release.name}: set the sphinx-design dropdown title to:{dropdown_title}"
        )
        _wait_for_done(message)

        message = (
            f"Review {rsts.release.name} to ensure it is a good reflection of "
            f"what is new in {release_strings.series}."
        )
        _wait_for_done(message)

        message = (
            "Work with the development team to populate the Release "
            f"Highlights dropdown section at the top of {rsts.release.name}."
        )
        _wait_for_done(message)

    else:
        message = (
            "Create a patch dropdown section at the top of "
            f"{rsts.release.name}.\n"
            f"See {rsts.template} for how this should be written."
        )
        _wait_for_done(message)

    if first_in_series:
        # TODO: automate
        message = (
            "Remove the What's New template file.\n"
            f"git rm {rsts.template.absolute()};"
        )
        _wait_for_done(message)

    message = (
        "Commit and push all the What's New changes.\n"
        f"git commit -am \"What's new updates for {release_strings.tag} .\";\n"
        f"git push -u origin {working_branch};"
    )
    _wait_for_done(message)

    message = (
        f"Follow the Pull Request process to get {working_branch} "
        f"merged into upstream/{release_strings.branch} .\n"
        "Make sure the documentation is previewed during this process."
    )
    _wait_for_done(message)

    return rsts


def cut_release(
    release_strings: ReleaseStrings, is_release_candidate: bool
) -> None:
    _break_print("The release ...")

    message = (
        "Visit https://github.com/SciTools/iris/releases/new to open "
        "a blank new-release web page."
    )
    _wait_for_done(message)

    message = (
        f"Select {release_strings.branch} as the Target.\n"
        f"Input {release_strings.tag} as the new tag to create, and also as "
        "the Release title."
    )
    _wait_for_done(message)

    message = (
        "Craft an appropriate release description in the main text box.\n"
        "Be sure to mention the What's New entry, conda-forge and PyPI - you "
        "will need to return later to make these into links.\n"
        "Be careful to change the appropriate words if copying from a "
        "previous release description."
    )
    _wait_for_done(message)

    if is_release_candidate:
        message = (
            "This is a release candidate - include the following instructions "
            "for installing with conda or pip:\n"
            f"conda install -c conda-forge/label/rc_iris iris={release_strings.release}\n"
            f"pip install scitools-iris=={release_strings.release}"
        )
        _wait_for_done(message)

        message = (
            "This is a release candidate - tick the box to set this as a "
            "pre-release."
        )
        _wait_for_done(message)

    else:
        message = "Tick the box to set this as the latest release."
        _wait_for_done(message)

    message = "Click: Publish release !"
    _wait_for_done(message)

    message = (
        "The CI will now run against this new tag, including automatically "
        "publishing to PyPI."
    )
    _break_print(message)

    url = "https://github.com/SciTools/iris/actions/workflows/ci-wheels.yml"
    message = (
        f"Visit {url} to monitor the building, testing and publishing of "
        "the Iris sdist and binary wheel to PyPI."
    )
    _wait_for_done(message)


def check_rtd(
    release_strings: ReleaseStrings, is_release_candidate: bool
) -> None:
    _break_print("Read the Docs checks ...")

    message = (
        "Visit https://readthedocs.org/projects/scitools-iris/versions/ and "
        "make sure you are logged in."
    )
    _wait_for_done(message)

    message = f"Set {release_strings.tag} to Active, un-Hidden."
    _wait_for_done(message)

    message = f"Set {release_strings.branch} to Active, Hidden."
    _wait_for_done(message)

    message = (
        "Keep the latest 2 branch doc builds active - those formatted 0.0.x - "
        "deactivate older ones."
    )
    _wait_for_done(message)

    message = (
        f"Visit https://scitools-iris.readthedocs.io/en/{release_strings.tag} "
        "to confirm:\n\n"
        "- The docs have rendered.\n"
        f"- The version badge in the top left reads: {release_strings.tag} .\n"
        "   (this demonstrates that setuptools_scm has worked correctly).\n"
        "- The What's New looks correct.\n"
        f"- {release_strings.tag} is available in RTD's version switcher.\n\n"
        "NOTE: the docs can take several minutes to finish building."
    )
    if not is_release_candidate:
        message += (
            "- Selecting 'stable' in the version switcher also brings up the "
            f"{release_strings.tag} render."
        )
    _wait_for_done(message)

    message = (
        f"Visit https://scitools-iris.readthedocs.io/en/{release_strings.branch} "
        "to confirm:\n\n"
        "- The docs have rendered\n"
        f"- The version badge in the top left includes: {release_strings.branch} .\n"
        f"- {release_strings.branch} is NOT available in RTD's version switcher.\n\n"
        "NOTE: the docs can take several minutes to finish building."
    )
    _wait_for_done(message)


def check_pypi(
    release_strings: ReleaseStrings, is_release_candidate: bool
) -> str:
    _break_print("PyPI checks ...")
    _break_print("If anything goes wrong, manual steps are in the documentation.")

    message = (
        "Confirm that the following URL is correctly populated:\n"
        f"https://pypi.org/project/scitools-iris/{release_strings.release}/"
    )
    _wait_for_done(message)

    message = (
        f"Confirm that {release_strings.release} is at the top of this page:\n"
        "https://pypi.org/project/scitools-iris/#history"
    )
    _wait_for_done(message)

    if is_release_candidate:
        message = (
            f"Confirm that {release_strings.release} is marked as a "
            f"pre-release on this page:\n"
            "https://pypi.org/project/scitools-iris/#history"
        )
    else:
        message = (
            f"Confirm that {release_strings.release} is the tag shown on the "
            "scitools-iris PyPI homepage:\n"
            "https://pypi.org/project/scitools-iris/"
        )
    _wait_for_done(message)

    message = (
        f"Visit the below and click `view hashes` for the Source Distribution"
        f"(`.tar.gz`):\n"
        f"https://pypi.org/project/scitools-iris/{release_strings.release}#files\n"
    )
    sha256 = _get_input(message, "Input the SHA256 hash")

    message = (
        "Confirm that pip install works as expected:\n"
        f"pip install scitools-iris=={release_strings.release};"
    )
    _wait_for_done(message)

    return sha256


def update_conda_forge(
    release_strings: ReleaseStrings, is_release_candidate: bool, sha256: str
) -> None:
    _break_print("conda-forge updates ...")

    if not is_release_candidate:
        message = (
            "NOTE: after several hours conda-forge automation will create a "
            "Pull Request against conda-forge/iris-feedstock (via the "
            "regro-cf-autotick-bot). Quicker to sort it now, manually ..."
        )
        _break_print(message)

    message = (
        "Make sure you have a GitHub fork of:\n"
        "https://github.com/conda-forge/iris-feedstock"
    )
    _wait_for_done(message)

    message = (
        "Make sure you have a local clone of your iris-feedstock fork.\n"
        "`cd` into your clone."
    )
    _wait_for_done(message)

    if is_release_candidate:
        message = (
            "Visit the conda-forge feedstock branches page:\n"
            "https://github.com/conda-forge/iris-feedstock/branches"
        )
        _wait_for_done(message)

        message = (
            "Find the release candidate branch - "
            "`rc`/`release-candidate`/similar.\n"
        )
        rc_branch = _get_input(
            message,
            "Input the name of the release candidate branch"
        )

        message = (
            f"Is the latest commit on {rc_branch} over 1 month ago?"
        )
        archive_rc = None
        while archive_rc is None:
            age_check = _get_input(message, "y / n")
            if age_check.casefold() == "y".casefold():
                archive_rc = True
            elif age_check.casefold() == "n".casefold():
                archive_rc = False
            else:
                _report_problem("Invalid entry. Please try again ...")

        if archive_rc:
            # We chose this odd handling of release candidate branches because
            #  a persistent branch will gradually diverge as `main` receives
            #  automatic and manual maintenance (where recreating these on
            #  another branch is often beyond Iris dev expertise). Advised
            #  practice from conda-forge is also liable to evolve over time.
            #  Since there is no benefit to a continuous Git history on the
            #  release candidate branch, the simplest way to keep it aligned
            #  with best practice is to regularly create a fresh branch from
            #  `main`.

            date_string = datetime.today().strftime("%Y%m%d")
            message = (
                f"Archive the {rc_branch} branch by appending _{date_string} "
                "to its name.\n"
                f"e.g. rc_{date_string}\n\n"
                f"({__file__} includes an explanation of this in the comments)."
            )
            _wait_for_done(message)

            message = (
                "Follow the latest conda-forge guidance for creating a new "
                "release candidate branch from the `main` branch:\n"
                "https://conda-forge.org/docs/maintainer/knowledge_base.html#pre-release-builds\n\n"
                "Config file(s) should point to the `rc_iris` label.\n"
            )
            rc_branch = _get_input(message, "Input the name of your new branch")

        upstream_branch = rc_branch
    else:
        upstream_branch = "main"

    # TODO: automate
    message = (
        "Checkout a new branch for the conda-forge changes for this release:\n"
        "git fetch upstream;\n"
        f"git checkout upstream/{upstream_branch} -b {release_strings.tag};\n"
    )
    _wait_for_done(message)

    message = (
        "Update ./recipe/meta.yaml:\n\n"
        f"- The version at the very top of the file: {release_strings.release}\n"
        f"- The sha256 hash: {sha256}\n"
        "- Requirements: align the packages and pins with those in the Iris repo\n"
        "- Maintainers: update with any changes to the dev team\n"
        "- MAKE SURE everything else is correct - plenty of other things "
        "might need one-off changes.\n"
    )
    _wait_for_done(message)

    # TODO: automate
    message = (
        "No other file normally needs changing in iris-feedstock, so push up "
        "the changes to prepare for a Pull Request:\n"
        f'git commit -am "Recipe updates for {release_strings.tag} .";\n'
        f"git push -u origin {release_strings.tag};"
    )
    _wait_for_done(message)

    message = (
        f"Follow the Pull Request process to get {release_strings.tag} branch "
        f"merged into upstream/{upstream_branch} .\n"
        "Specific conda-forge guidance will be automatically given once the "
        "PR is created."
    )
    _wait_for_done(message)

    message = (
        f"Confirm that {release_strings.release} appears in this list:\n"
        "https://anaconda.org/conda-forge/iris/files"
    )
    _wait_for_done(message)

    if not is_release_candidate:
        message = (
            f"Confirm that {release_strings.release} is displayed on this "
            "page as the latest available:\n"
            "https://anaconda.org/conda-forge/iris"
        )
        _wait_for_done(message)

    if is_release_candidate:
        channel_command = " -c conda-forge/label/rc_iris "
    else:
        channel_command = " "
    message = (
        "Confirm that conda (or mamba) install works as expected:\n"
        f"conda create -n tmp_iris{channel_command}iris={release_strings.release};\n"
        f"conda remove -n tmp_iris --all;"
    )
    _wait_for_done(message)


def update_links(release_strings: ReleaseStrings) -> None:
    _break_print("Link updates ...")

    message = (
        "Revisit the GitHub release:\n"
        f"https://github.com/SciTools/iris/releases/tag/{release_strings.tag}\n"
        "You have confirmed that Read the Docs, PyPI and conda-forge have all "
        "updated correctly. Include the following links in the release "
        "notes:\n\n"
        f"https://scitools-iris.readthedocs.io/en/{release_strings.tag}/\n"
        f"https://pypi.org/project/scitools-iris/{release_strings.release}/\n"
        f"https://anaconda.org/conda-forge/iris?version={release_strings.release}\n"
    )
    _wait_for_done(message)

    message = (
        "Update the release page in GitHub discussions, with the above links "
        "and anything else appropriate.\n"
        "https://github.com/SciTools/iris/discussions"
    )
    _wait_for_done(message)


def twitter_announce(
    release_strings: ReleaseStrings, first_in_series: bool
) -> None:
    message = (
        "Announce the release via https://twitter.com/scitools_iris, and any "
        "other appropriate message boards (e.g. Yammer).\n"
        "Any content used for the announcement should be stored in the "
        "SciTools/twitter-scitools-iris GitHub repo.\n"
    )
    if not first_in_series:
        message += (
            f"Consider replying within an existing {release_strings.series} "
            "announcement thread, if appropriate."
        )
    _wait_for_done(message)


def update_citation(
    release_strings: ReleaseStrings, is_release_candidate: bool
) -> None:
    if not is_release_candidate:
        src_dir = Path(__file__).parents[1] / "docs" / "src"
        citation_rst = src_dir / "userguide" / "citation.rst"
        assert citation_rst.is_file()
        message = (
            f"Follow the Pull Request process to update {citation_rst.name} "
            "with the correct dates, DOI and version string for "
            f"{release_strings.tag}.\n"
            f"{citation_rst.absolute()}\n\n"
            f"The PR should target {release_strings.branch} (prior to merge-back)."
        )
        _wait_for_done(message)


def merge_back(
    release_strings: ReleaseStrings, first_in_series: bool, rsts: WhatsNewRsts
) -> None:
    _break_print("Branch merge-back ...")

    merge_commit = (
        "BE SURE TO MERGE VIA A MERGE-COMMIT (not a squash-commit), to "
        "preserve the commit SHA's."
    )

    if first_in_series:
        # TODO: automate

        working_branch = release_strings.branch + ".mergeback"
        _delete_local_branch(working_branch)
        message = (
            "Checkout a local branch from the official ``main`` branch.\n"
            "git fetch upstream;\n"
            f"git checkout upstream/main -b {working_branch};"
        )
        _wait_for_done(message)

        message = (
            f"Merge in the commits from {release_strings.branch}.\n"
            f"{merge_commit}\n"
            f"git merge upstream/{release_strings.branch} --no-ff "
            '-m "Merging release branch into main";'
        )
        _wait_for_done(message)

        message = (
            "Recreate the following files, which are present in ``main``, but "
            f"are currently deleted from {working_branch}:\n"
            f"{rsts.latest.absolute()}\n"
            f"{rsts.template.absolute()}\n"
            "THEN:\n"
            f"git add {rsts.latest.absolute()};\n"
            f"git add {rsts.template.absolute()};\n"
        )
        _wait_for_done(message)

        message = (
            f"In {rsts.index.absolute()}:\n"
            f"Add {rsts.latest.name} to the top of the list of .rst files, "
            f"and set the top include:: to be {rsts.latest.name} ."
        )
        _wait_for_done(message)

        message = (
            "Commit and push all the What's New changes.\n"
            "git commit -am \"Restore latest What's New files.\";\n"
            f"git push -u origin {working_branch};"
        )
        _wait_for_done(message)

        message = (
            "Follow the Pull Request process to get "
            f"{working_branch} merged into upstream/main .\n"
            "Make sure the documentation is previewed during this process.\n"
            f"{merge_commit}"
        )
        _wait_for_done(message)

    else:
        message = (
            f"Propose a merge-back from {release_strings.branch} into ``main`` by "
            f"visiting this URL and clicking `Create pull request`:\n"
            f"https://github.com/SciTools/iris/compare/main...{release_strings.branch}\n"
            f"{merge_commit}"
        )
        _wait_for_done(message)
        message = (
            f"Once the pull request is merged ensure that the {release_strings.branch} "
            "release branch is restored.\n"
            "GitHub automation rules may have automatically deleted the release branch."
        )
        _wait_for_done(message)


def main():
    _mark_section(1)
    release_type = get_release_type()

    _mark_section(2)
    release_strings = get_release_tag()

    _mark_section(3)
    is_release_candidate = check_release_candidate(
        release_type,
        release_strings,
    )

    _mark_section(4)
    is_first_in_series = check_first_in_series(
        release_type,
        release_strings,
        is_release_candidate,
    )

    _mark_section(5)
    update_standard_names(
        is_first_in_series,
    )

    _mark_section(6)
    check_deprecations(
        release_type,
    )

    _mark_section(7)
    create_release_branch(
        release_strings,
        is_first_in_series,
    )

    _mark_section(8)
    whats_new_rsts = finalise_whats_new(
        release_type,
        release_strings,
        is_release_candidate,
        is_first_in_series,
    )

    _mark_section(9)
    cut_release(
        release_strings,
        is_release_candidate,
    )

    _mark_section(10)
    check_rtd(
        release_strings,
        is_release_candidate,
    )

    _mark_section(11)
    sha256 = check_pypi(
        release_strings,
        is_release_candidate,
    )

    _mark_section(12)
    update_conda_forge(
        release_strings,
        is_release_candidate,
        sha256,
    )

    _mark_section(13)
    update_links(
        release_strings,
    )

    _mark_section(14)
    twitter_announce(
        release_strings,
        is_first_in_series,
    )

    _mark_section(15)
    update_citation(
        release_strings,
        is_release_candidate,
    )

    _mark_section(16)
    merge_back(
        release_strings,
        is_first_in_series,
        whats_new_rsts,
    )

    _break_print("RELEASE COMPLETE. Congratulations! 🎉")


if __name__ == "__main__":
    main()
