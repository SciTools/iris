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
from enum import IntEnum
from pathlib import Path
import re
import shlex
import subprocess
import typing

from packaging.version import InvalidVersion, Version

try:
    from nothing import Progress
except ImportError:
    install_message = (
        "This script requires the `nothing` package to be installed:\n"
        "pip install git+https://github.com/SciTools-incubator/nothing.git"
    )
    raise ImportError(install_message)


class IrisVersion(Version):
    def __str__(self):
        return f"v{super().__str__()}"

    @property
    def series(self) -> str:
        # TODO: find an alternative word which is meaningful to everyone
        #  while not being ambiguous.
        return f"v{self.major}.{self.minor}"

    @property
    def branch(self) -> str:
        return f"{self.series}.x"


class IrisRelease(Progress):
    class ReleaseTypes(IntEnum):
        MAJOR = 0
        MINOR = 1
        PATCH = 2

    github_scitools: str = "upstream"
    github_fork: str = "origin"
    github_user: typing.Optional[str] = None
    patch_min_max_tag: typing.Optional[tuple[str, str]] = None
    git_tag: typing.Optional[str] = None  # v1.2.3rc0
    sha256: typing.Optional[str] = None

    @classmethod
    def get_cmd_description(cls) -> str:
        return "Do-nothing workflow for the Iris release process."

    @classmethod
    def get_steps(cls) -> list[typing.Callable[..., None]]:
        return [
            cls.analyse_remotes,
            # cls.parse_tags,
            cls.get_release_tag,
            cls.get_all_patches,
            cls.apply_patches,
            cls.validate,
            cls.update_standard_names,
            cls.check_deprecations,
            cls.create_release_branch,
            cls.finalise_whats_new,
            cls.cut_release,
            cls.check_rtd,
            cls.check_pypi,
            cls.update_conda_forge,
            cls.update_links,
            cls.bluesky_announce,
            cls.merge_back,
            cls.next_release,
        ]

    @staticmethod
    def _git_remote_v() -> str:
        # Factored out to assist with testing.
        return subprocess.check_output(shlex.split("git remote -v"), text=True)

    def _git_remote_get_url(self) -> str:
        # Factored out to assist with testing.
        return subprocess.check_output(
            shlex.split(f"git remote get-url {self.github_fork}"), text=True
        )

    def analyse_remotes(self):
        self.print("Analysing Git remotes ...")

        class Remote(typing.NamedTuple):
            name: str
            url: str
            fetch: bool

        remotes_raw = self._git_remote_v().splitlines()
        remotes_split = [line.split() for line in remotes_raw]
        remotes = [
            Remote(name=parts[0], url=parts[1], fetch=parts[2] == "(fetch)")
            for parts in remotes_split
        ]

        scitools_regex = re.compile(r"github\.com[:/]SciTools/iris\.git")
        self.github_scitools = [
            r.name for r in remotes
            if r.fetch and scitools_regex.search(r.url)
        ][0]

        possible_forks = [
            r for r in remotes
            if not r.fetch and r.name != self.github_scitools
        ]
        assert len(possible_forks) > 0

        def number_to_fork(input_number: str) -> str | None:
            try:
                result = possible_forks[int(input_number)].name
            except (ValueError, IndexError):
                result = None
                self.report_problem("Invalid number. Please try again ...")
            return result

        numbered_forks = " | ".join(
            [f"{ix}: {r.name}" for ix, r in enumerate(possible_forks)]
        )
        self.set_value_from_input(
            key="github_fork",
            message="Which remote is your Iris fork?",
            expected_inputs=f"Choose a number {numbered_forks}",
            post_process=number_to_fork,
        )

        fork_url = self._git_remote_get_url()
        self.github_user = re.search(
            r"(?<=github\.com[:/])([a-zA-Z0-9-]+)(?=/)",
            fork_url,
        ).group(0)
        if self.github_user is None:
            message = f"Error deriving GitHub username from URL: {fork_url}"
            raise RuntimeError(message)

    def _git_ls_remote_tags(self) -> str:
        # Factored out to assist with testing.
        return subprocess.check_output(
            shlex.split(f"git ls-remote --tags {self.github_scitools}"),
            text=True,
        )

    def _get_tagged_versions(self) -> list[IrisVersion]:
        tag_regex = re.compile(r"(?<=refs/tags/).*$")
        scitools_tags_raw = self._git_ls_remote_tags().splitlines()
        scitools_tags_searched = [
            tag_regex.search(line) for line in scitools_tags_raw
        ]
        scitools_tags = [
            search.group(0) for search in scitools_tags_searched
            if search is not None
        ]

        def get_version(tag: str) -> IrisVersion | None:
            try:
                return IrisVersion(tag)
            except InvalidVersion:
                return None

        versions = [get_version(tag) for tag in scitools_tags]
        tagged_versions = [v for v in versions if v is not None]
        if len(tagged_versions) == 0:
            message = (
                "Error: unable to find any valid version tags in the "
                f"{self.github_scitools} remote."
            )
            raise RuntimeError(message)
        return tagged_versions

    def get_release_tag(self):
        def validate(input_tag: str) -> str | None:
            result = None
            try:
                version = IrisVersion(input_tag)
            except InvalidVersion as err:
                self.report_problem(
                    f"Packaging error: {err}\n"
                    "Please try again ..."
                )
            else:
                if version in self._get_tagged_versions():
                    self.report_problem(
                        f"Version {version} already exists as a git tag. "
                        "Please try again ..."
                    )
                else:
                    result= input_tag  # v1.2.3rc0
            return result

        message = (
            "Input the release tag you are creating today, including any "
            "release "
            "candidate suffix.\n"
            "https://semver.org/\n"
            "https://scitools-iris.readthedocs.io/en/latest/developers_guide"
            "/release.html?highlight=candidate#release-candidate"
        )
        self.set_value_from_input(
            key="git_tag",
            message=message,
            expected_inputs="e.g. v1.2.3rc0",
            post_process=validate,
        )

    @property
    def version(self) -> IrisVersion:
        # Implemented like this since the Version class cannot be JSON serialised.
        return IrisVersion(self.git_tag)

    @property
    def is_latest_tag(self) -> bool:
        return all(self.version >= v for v in self._get_tagged_versions())

    @property
    def release_type(self) -> ReleaseTypes:
        if self.version.micro == 0:
            if self.version.minor == 0:
                release_type = self.ReleaseTypes.MAJOR
            else:
                release_type = self.ReleaseTypes.MINOR
        else:
            release_type = self.ReleaseTypes.PATCH
        return release_type

    @property
    def is_release_candidate(self) -> bool:
        return self.version.is_prerelease and self.version.pre[0] == "rc"

    @property
    def first_in_series(self) -> bool:
        return self.version.series not in [v.series for v in self._get_tagged_versions()]

    def get_all_patches(self):
        if self.release_type is self.ReleaseTypes.PATCH:
            message = (
                "PATCH release detected. Sometimes a patch needs to be applied "
                "to multiple series."
            )
            self.print(message)

            tagged_versions = self._get_tagged_versions()
            series_all = [v.series for v in sorted(tagged_versions)]
            series_unique = sorted(set(series_all), key=series_all.index)
            series_numbered = "\n".join(f"{i}: {s}" for i, s in enumerate(series_unique))

            def numbers_to_new_patches(
                input_numbers: str
            ) -> tuple[str, str] | None:
                try:
                    first_str, last_str = input_numbers.split(",")
                    first, last = int(first_str), int(last_str)
                except ValueError:
                    self.report_problem(
                        "Invalid input, expected two integers comma-separated. "
                        "Please try again ..."
                    )
                    return None

                try:
                    series_min = series_unique[first]
                    series_max = series_unique[last]
                except IndexError:
                    self.report_problem("Invalid numbers. Please try again ...")
                    return None

                def series_new_patch(series: str) -> str:
                    latest = max(v for v in tagged_versions if v.series == series)
                    iris_version = IrisVersion(
                        f"{latest.major}.{latest.minor}.{latest.micro + 1}"
                    )
                    return str(iris_version)

                return (series_new_patch(series_min), series_new_patch(series_max))

            self.set_value_from_input(
                key="patch_min_max_tag",
                message=(
                    f"{series_numbered}\n\n"
                    "Input the earliest and latest series that need patching."
                ),
                expected_inputs=f"Choose two numbers from above e.g. 0,2",
                post_process=numbers_to_new_patches,
            )

            first_patch = self.patch_min_max[0]
            if self.version > first_patch:
                message = (
                    f"Starting with {first_patch}. ({self.version} will be "
                    "covered in sequence)"
                )
                self.print(message)
                self.git_tag = str(first_patch)

    @property
    def patch_min_max(self) -> tuple[IrisVersion, IrisVersion] | None:
        if self.patch_min_max_tag is None:
            result = None
        else:
            assert len(self.patch_min_max_tag) == 2
            result = (
                IrisVersion(self.patch_min_max_tag[0]),
                IrisVersion(self.patch_min_max_tag[1]),
            )
        return result

    @property
    def more_patches_after_this_one(self) -> bool:
        return(
            self.release_type is self.ReleaseTypes.PATCH and
            self.patch_min_max is not None and
            self.version < self.patch_min_max[1]
        )

    def apply_patches(self):
        if self.release_type is self.ReleaseTypes.PATCH:
            message = (
                f"Input the {self.github_scitools} branch name where the patch "
                "change commit(s) exist, or make no input if nothing has been "
                "merged yet."
            )
            patch_branch = self.get_input(
                message=message,
                expected_inputs="",
            )
            match patch_branch:
                case self.version.branch:
                    message = (
                        "The patch change(s) are on the ideal branch to avoid later"
                        f"Git conflicts: {self.version.branch} . Continue ..."
                    )
                case "":
                    message = (
                        f"Propose the patch change(s) against {self.version.branch} via "
                        f"pull request(s). Targeting {self.version.branch} will "
                        "avoid later Git conflicts."
                    )
                case _:
                    message = (
                        "Create pull request(s) cherry-picking the patch change(s) "
                        f"from {patch_branch} into {self.version.branch} .\n"
                        "cherry-picking will cause Git conflicts later in the "
                        "release process; in future consider targeting the patch "
                        "change(s) directly at the release branch."
                    )

            self.wait_for_done(message)

    def validate(self) -> None:
        self.print("Validating release details ...")

        message_template = (
            f"{self.version} corresponds to a {{}} release. This script cannot "
            "handle such releases."
        )
        if self.version.is_devrelease:
            message = message_template.format("development")
            raise RuntimeError(message)
        if self.version.is_postrelease:
            message = message_template.format("post")
            raise RuntimeError(message)

        if self.version.is_prerelease and self.version.pre[0] != "rc":
            message = (
                "The only pre-release type that this script can handle is 'rc' "
                f"(for release candidate), but got '{self.version.pre[0]}'."
            )
            raise RuntimeError(message)

        if self.release_type is self.ReleaseTypes.PATCH and self.is_release_candidate:
            message = (
                f"{self.version} corresponds to a PATCH release AND a release "
                "candidate. This script cannot handle that combination."
            )
            raise RuntimeError(message)

        if self.first_in_series:
            message_pre = (
                f"No previous releases found in the {self.version.series} series."
            )
            if self.release_type is self.ReleaseTypes.PATCH:
                message = (
                    f"{message_pre} This script cannot handle a PATCH release "
                    f"that is the first in a series."
                )
                raise RuntimeError(message)

            if not self.is_release_candidate:
                message = (
                    f"{message_pre} The first release in a series is expected "
                    f"to be a release candidate, but this is not. Are you sure "
                    f"you want to continue?"
                )
                if self.get_input(message, "y / [n]").casefold() != "y".casefold():
                    exit()

        status = {
            "GitHub user": self.github_user,
            "SciTools remote": self.github_scitools,
            "Fork remote": self.github_fork,
            "Release tag": self.git_tag,
            "Release type": self.release_type.name,
            "Release candidate?": self.is_release_candidate,
            f"First release in {self.version.series} series?": self.first_in_series,
            "Current latest Iris release": max(self._get_tagged_versions()),
        }
        if self.release_type is self.ReleaseTypes.PATCH and self.patch_min_max is not None:
            status["Series being patched"] = (
                f"{self.patch_min_max[0].series} to {self.patch_min_max[1].series}"
            )
        message = (
            "\n".join(f"- {k}: {v}" for k, v in status.items()) + "\n\n"
            "Confirm that the details above are correct.\n"
            "Consider temporary/permanent edits to the do-nothing script if "
            "necessary."
        )
        self.wait_for_done(message)

    def _create_pr(
        self,
        base_org: str,
        base_repo: str,
        base_branch: str,
        head_branch: str
    ) -> None:
        """Instruct user to create a PR with a specified base and head.

        Parameters
        ----------
        base_org : str
            The name of the GitHub organisation that owns the `base_repo` that
            owns the `base_branch`.
        base_repo : str
            The name of the GitHub repository (within the `base_org`) that owns
            the `base_branch`.
        base_branch : str
            The name of the branch (within the `base_repo`) that will be the
            base of the PR.
        head_branch : str
            The name of the branch (within the user's fork of `base_repo`) that
            will be the head of the PR.
        """
        repo_url = f"https://github.com/{base_org}/{base_repo}"
        diff_url = f"{base_branch}...{self.github_user}:{base_repo}:{head_branch}"
        full_url = f"{repo_url}/compare/{diff_url}"

        pr_message = (
            "Create a Pull Request for your changes by visiting this URL "
            "and clicking `Create pull request`:\n"
            f"{full_url}"
        )
        self.wait_for_done(pr_message)

    def update_standard_names(self):
        if self.first_in_series:
            working_branch = self.version.branch + ".standard_names"
            self._delete_local_branch(working_branch)
            message = (
                "Checkout a local branch from the official ``main`` branch.\n"
                f"git fetch {self.github_scitools};\n"
                f"git checkout {self.github_scitools}/main -b {working_branch};"
            )
            self.wait_for_done(message)

            url = "https://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml"
            file = Path(__file__).parents[1] / "etc" / "cf-standard-name-table.xml"
            message = (
                "Update the CF standard names table to the latest version:\n"
                f'wget "{url}" -O {file};\n'
                f"git add {file};\n"
                "git commit -m 'Update CF standard names table.';\n"
                f"git push -u {self.github_fork} {working_branch};"
            )
            self.wait_for_done(message)

            self._create_pr(
                base_org="SciTools",
                base_repo="iris",
                base_branch="main",
                head_branch=working_branch,
            )
            message = "Work with the development team to get the PR merged."
            self.wait_for_done(message)

    def check_deprecations(self):
        if self.release_type is self.ReleaseTypes.MAJOR:
            message = (
                "This is a MAJOR release - be sure to finalise all deprecations "
                "and FUTUREs from previous releases, via a new Pull Request.\n"
                "https://scitools-iris.readthedocs.io/en/latest/developers_guide"
                "/contributing_deprecations.html"
            )
            self.wait_for_done(message)

    def create_release_branch(self):
        # TODO: automate
        print("Release branch management ...")

        if self.first_in_series:
            message = (
                "Visit https://github.com/SciTools/iris and create the"
                f"``{self.version.branch}`` release branch from ``main``."
            )
            self.wait_for_done(message)

        else:
            message = (
                "If necessary: "
                "cherry-pick any specific commits that are needed from ``main`` "
                f"onto {self.version.branch} , to get the CI passing.\n"
                "E.g. a new dependency pin may have been introduced since "
                f"{self.version.branch} was last updated from ``main``.\n"
                "Note that cherry-picking will cause Git conflicts later in "
                "the release process."
            )
            self.wait_for_done(message)

    def _delete_local_branch(self, branch_name: str):
        message = (
            "Before the next step, avoid a name clash by deleting any "
            "existing local branch, if one exists.\n"
            f"git branch -D {branch_name};\n"
            f"git push -d {self.github_fork} {branch_name};"
        )
        IrisRelease.wait_for_done(message)

    class WhatsNewRsts(typing.NamedTuple):
        latest: Path
        release: Path
        index_: Path
        template: Path

    @property
    def whats_news(self) -> WhatsNewRsts:
        src_dir = Path(__file__).parents[1] / "docs" / "src"
        whatsnew_dir = src_dir / "whatsnew"
        assert whatsnew_dir.is_dir()
        latest = whatsnew_dir / "latest.rst"

        return self.WhatsNewRsts(
            latest=latest,
            release=whatsnew_dir / (self.version.series[1:] + ".rst"),
            index_=whatsnew_dir / "index.rst",
            template=latest.with_suffix(".rst.template"),
        )

    def finalise_whats_new(self):
        self.print("What's New finalisation ...")

        working_branch = self.version.branch + ".updates"
        self._delete_local_branch(working_branch)
        message = (
            f"Checkout a local branch from the official {self.version.branch} "
            f"branch.\n"
            f"git fetch {self.github_scitools};\n"
            f"git checkout {self.github_scitools}/{self.version.branch} -b "
            f"{working_branch};"
        )
        self.wait_for_done(message)

        # TODO: automate
        if self.first_in_series:
            message = (
                "'Cut' the What's New for the release.\n"
                f"git mv {self.whats_news.latest.absolute()} "
                f"{self.whats_news.release.absolute()};"
            )
            self.wait_for_done(message)

            message = (
                f"In {self.whats_news.index_.absolute()}:\n"
                f"Replace references to {self.whats_news.latest.name} with "
                f"{self.whats_news.release.name}"
            )
            self.wait_for_done(message)

        self.print(f"What's New file path = {self.whats_news.release}")

        if not self.release_type is self.ReleaseTypes.PATCH:
            whatsnew_title = (
                f"{self.version.series} ({datetime.today().strftime('%d %b %Y')}"
            )
            if self.is_release_candidate:
                whatsnew_title += " [release candidate]"
            whatsnew_title += ")"
            # TODO: automate
            message = (
                f"In {self.whats_news.release.name}: set the page title to:\n"
                f"{whatsnew_title}\n"
            )
            if not self.is_release_candidate:
                message += (
                    "\nBe sure to remove any existing mentions of release "
                    "candidate from the title.\n"
                )
            self.wait_for_done(message)

            message = (
                f"In {self.whats_news.release.name}: ensure the page title "
                "underline is the exact same length as the page title text."
            )
            self.wait_for_done(message)

            dropdown_title = f"\n{self.version.series} Release Highlights\n"
            message = (
                f"In {self.whats_news.release.name}: set the sphinx-design "
                f"dropdown title to:{dropdown_title}"
            )
            self.wait_for_done(message)

            message = (
                f"Review {self.whats_news.release.name} to ensure it is a good "
                f"reflection of what is new in {self.version.series}.\n"
                "I.e. all significant work you are aware of should be "
                "present, such as a major dependency pin, a big new feature, "
                "a known performance change. You can not be expected to know "
                "about every single small change."
            )
            self.wait_for_done(message)

            message = (
                "Work with the development team to populate the Release "
                f"Highlights dropdown section at the top of "
                f"{self.whats_news.release.name}."
            )
            self.wait_for_done(message)

        else:
            message = (
                "Create a patch dropdown section at the top of "
                f"{self.whats_news.release.name}.\n"
                f"See {self.whats_news.template} for how this should be written."
            )
            self.wait_for_done(message)

        if self.first_in_series:
            # TODO: automate
            message = (
                "Remove the What's New template file.\n"
                f"git rm {self.whats_news.template.absolute()};"
            )
            self.wait_for_done(message)

        message = (
            "Commit and push all the What's New changes.\n"
            f"git add {self.whats_news.release.absolute()};\n"
            f"git add {self.whats_news.index_.absolute()};\n"
            f'git commit -m "Whats-New updates for {self.version} .";\n'
            f"git push -u {self.github_fork} {working_branch};"
        )
        self.wait_for_done(message)

        self._create_pr(
            base_org="SciTools",
            base_repo="iris",
            base_branch=self.version.branch,
            head_branch=working_branch,
        )
        message = (
            "Work with the development team to get the PR merged.\n"
            "Make sure the documentation is previewed during this process.\n"
            "Make sure you are NOT targeting the `main` branch."
        )
        self.wait_for_done(message)

    def cut_release(self):
        self.print("The release ...")

        message = (
            "Visit https://github.com/SciTools/iris/releases/new to open "
            "a blank new-release web page."
        )
        self.wait_for_done(message)

        message = (
            f"Select {self.version.branch} as the Target.\n"
            f"Input {self.version} as the new tag to create, and also as "
            "the Release title.\n"
            "Make sure you are NOT targeting the `main` branch."
        )
        self.wait_for_done(message)

        message = (
            "Populate the main text box.\n"
            "- Usual approach: copy from the last similar release, and "
            "THOROUGHLY check for all references to the old release - change "
            "these.\n"
            "- Alternatively: craft a new release description from scratch. "
            "Be sure to mention the What's New entry, conda-forge and PyPI; "
            "note that you will need to return later to make these into "
            "links.\n"
        )
        self.wait_for_done(message)

        if self.is_release_candidate:
            message = (
                "This is a release candidate - include the following "
                "instructions for installing with conda or pip:\n"
                f"conda install -c conda-forge/label/rc_iris iris={self.version.public}\n"
                f"pip install scitools-iris=={self.version.public}"
            )
            self.wait_for_done(message)

            message = (
                "This is a release candidate - tick the box to set this as a "
                "pre-release."
            )
            self.wait_for_done(message)

        else:
            if self.is_latest_tag:
                message = "Tick the box to set this as the latest release."
            else:
                message = "Un-tick the latest release box."
            self.wait_for_done(message)

        message = "Click: Publish release !"
        self.wait_for_done(message)

        message = (
            "The CI will now run against this new tag, including automatically "
            "publishing to PyPI."
        )
        self.print(message)

        url = "https://github.com/SciTools/iris/actions/workflows/ci-wheels.yml"
        message = (
            f"Visit {url} to monitor the building, testing and publishing of "
            "the Iris sdist and binary wheel to PyPI."
        )
        self.wait_for_done(message)

    def check_rtd(self):
        self.print("Read the Docs checks ...")

        message = (
            "Visit https://readthedocs.org/projects/scitools-iris/versions/ "
            "and make sure you are logged in."
        )
        self.wait_for_done(message)

        message = f"Set {self.version} to Active, un-Hidden."
        self.wait_for_done(message)

        message = f"Set {self.version.branch} to Active, Hidden."
        self.wait_for_done(message)

        message = (
            "Keep only the latest 2 branch doc builds active - "
            f"'{self.version.branch}' and the previous one - deactivate older "
            "ones."
        )
        self.wait_for_done(message)

        message = (
            f"Visit https://scitools-iris.readthedocs.io/en/{self.version} "
            "to confirm:\n\n"
            "- The docs have rendered.\n"
            "- The version badge in the top left reads:\n"
            f"  'version (archived) | {self.version}'\n"
            "   (this demonstrates that setuptools_scm has worked correctly).\n"
            "- The What's New looks correct.\n"
            f"- {self.version} is available in RTD's version switcher.\n"
        )
        if not self.is_release_candidate and self.is_latest_tag:
            message += (
                "- Selecting 'stable' in the version switcher also brings up "
                f"the {self.version} render.\n"
            )
        message += "\nNOTE: the docs can take several minutes to finish building."
        self.wait_for_done(message)

        message = (
            f"Visit https://scitools-iris.readthedocs.io/en/{self.version.branch} "
            "to confirm:\n\n"
            "- The docs have rendered\n"
            f"- The version badge in the top left includes: {self.version.branch} .\n"
            f"- {self.version.branch} is NOT available in RTD's version switcher.\n\n"
            "NOTE: the docs can take several minutes to finish building."
        )
        self.wait_for_done(message)

    def check_pypi(self):
        self.print("PyPI checks ...")
        self.print("If anything goes wrong, manual steps are in the documentation.")

        message = (
            "Confirm that the following URL is correctly populated:\n"
            f"https://pypi.org/project/scitools-iris/{self.version.public}/"
        )
        self.wait_for_done(message)

        if self.is_latest_tag:
            message = (
                f"Confirm that {self.version.public} is at the top of this page:\n"
                "https://pypi.org/project/scitools-iris/#history"
            )
            self.wait_for_done(message)

        if self.is_release_candidate:
            message = (
                f"Confirm that {self.version.public} is marked as a "
                f"pre-release on this page:\n"
                "https://pypi.org/project/scitools-iris/#history"
            )
            self.wait_for_done(message)
        elif self.is_latest_tag:
            message = (
                f"Confirm that {self.version.public} is the tag shown on the "
                "scitools-iris PyPI homepage:\n"
                "https://pypi.org/project/scitools-iris/"
            )
            self.wait_for_done(message)

        def validate(sha256_string: str) -> str | None:
            valid = True
            try:
                _ = int(sha256_string, 16)
            except ValueError:
                valid = False
            valid = valid and len(sha256_string) == 64

            if not valid:
                self.report_problem("Invalid SHA256 hash. Please try again ...")
                result = None
            else:
                result = sha256_string
            return result

        message = (
            f"Visit the below and click `view details` for the Source Distribution"
            f"(`.tar.gz`):\n"
            f"https://pypi.org/project/scitools-iris/{self.version.public}#files\n"
        )
        self.set_value_from_input(
            key="sha256",
            message=message,
            expected_inputs="Input the SHA256 hash",
            post_process=validate,
        )

        message = (
            "Confirm that pip install works as expected:\n"
            "conda create -y -n tmp_iris pip cf-units;\n"
            "conda activate tmp_iris;\n"
            f"pip install scitools-iris=={self.version.public};\n"
            'python -c "import iris; print(iris.__version__)";\n'
            "conda deactivate;\n"
            "conda remove -n tmp_iris --all;\n"
        )
        self.wait_for_done(message)

    def update_conda_forge(self):
        self.print("conda-forge checks ...")

        if not self.is_release_candidate:
            message = (
                "NOTE: after several hours conda-forge automation will "
                "create a "
                "Pull Request against conda-forge/iris-feedstock (via the "
                "regro-cf-autotick-bot). Quicker to sort it now, manually ..."
            )
            self.print(message)

        message = (
            "Make sure you have a GitHub fork of:\n"
            "https://github.com/conda-forge/iris-feedstock"
        )
        self.wait_for_done(message)

        if self.is_release_candidate:
            message = (
                "Visit the conda-forge feedstock branches page:\n"
                "https://github.com/conda-forge/iris-feedstock/branches"
            )
            self.wait_for_done(message)

            message = (
                "Find the release candidate branch - typical names:\n"
                "`rc` / `release-candidate` / similar .\n"
            )
            rc_branch = self.get_input(
                message,
                "Input the name of the release candidate branch"
            )

            message = (
                f"Is the latest commit on {rc_branch} over 1 month ago?"
            )
            archive_rc = None
            while archive_rc is None:
                valid_entries = ["y", "n"]
                age_check = self.get_input(message, " / ".join(valid_entries))
                match = [age_check.casefold() == e.casefold() for e in valid_entries]
                if not any(match):
                    self.report_problem("Invalid entry. Please try again ...")
                else:
                    archive_rc = match[0]

            if archive_rc:
                # We chose this odd handling of release candidate branches because
                #  a persistent branch will gradually diverge as `main` receives
                #  automatic and manual maintenance (where recreating these on
                #  another branch is often beyond Iris dev expertise). Advised
                #  practice from conda-forge is also liable to evolve over time.
                #  Since there is no benefit to a continuous Git history on the
                #  release candidate branch, the simplest way to keep it  aligned
                #  with best practice is to regularly create a fresh branch from
                #  `main`.

                date_string = datetime.today().strftime("%Y%m%d")
                message = (
                    f"Archive the {rc_branch} branch by appending _"
                    f"{date_string} "
                    "to its name.\n"
                    f"e.g. rc_{date_string}\n\n"
                    f"({__file__} includes an explanation of this in the "
                    f"comments)."
                )
                self.wait_for_done(message)

                message = (
                    "Follow the latest conda-forge guidance for creating a new "
                    "release candidate branch from the `main` branch:\n"
                    "https://conda-forge.org/docs/maintainer/knowledge_base.html#pre-release-builds\n\n"
                    "If you need to change any feedstock files: a pull "
                    "request is coming in the the next steps so you can make "
                    "those changes at that point.\n\n"
                    "DEVIATION FROM GUIDANCE: config file(s) should point to "
                    "the `rc_iris` label (this is not the name that "
                    "conda-forge suggest).\n"
                )
                rc_branch = self.get_input(message, "Input the name of your new branch")

            upstream_branch = rc_branch
        else:
            upstream_branch = "main"

        # TODO: automate
        message = (
            "Checkout a new branch for the conda-forge changes for this "
            "release:\n"
            "git fetch upstream;\n"
            f"git checkout upstream/{upstream_branch} -b "
            f"{self.version};\n"
        )
        self.wait_for_done(message)

        message = (
            "Update ./recipe/meta.yaml:\n\n"
            f"- The version at the very top of the file: "
            f"{self.version.public}\n"
            f"- The sha256 hash: {self.sha256}\n"
            "- Requirements: align the packages and pins with those in the "
            "Iris repo\n"
            "- Maintainers: update with any changes to the dev team\n"
            "- Skim read the entire file to see if anything else is out of"
            "date, e.g. is the licence info still correct? Ask the lead "
            "Iris developers if unsure.\n"
        )
        if not self.is_latest_tag:
            message += (
                f"\nNOTE: {self.version} is not the latest Iris release, so "
                "you may need to restore settings from an earlier version "
                f"(check previous {self.version.series} releases)."
            )
        self.wait_for_done(message)

        # TODO: automate
        message = (
            "No other file normally needs changing in iris-feedstock, "
            "so push up "
            "the changes to prepare for a Pull Request:\n"
            f"git add recipe/meta.yaml;\n"
            f'git commit -m "Recipe updates for {self.version} .";\n'
            f"git push -u origin {self.version};"
        )
        self.wait_for_done(message)

        self._create_pr(
            base_org="conda-forge",
            base_repo="iris-feedstock",
            base_branch=upstream_branch,
            head_branch=f"{self.version}",
        )

        if self.is_release_candidate:
            readme_url = f"https://github.com/{self.github_user}/iris-feedstock/blob/{self.version}/README.md"
            rc_evidence = (
                "\n\nConfirm that conda-forge knows your changes are for the "
                "release candidate channel by checking the below README file. "
                "This should make multiple references to the `rc_iris` label:\n"
                f"{readme_url}"
            )
        else:
            rc_evidence = ""
        message = (
            "Follow the automatic conda-forge guidance for further populating "
            f"your Pull Request.{rc_evidence}"
        )
        self.wait_for_done(message)

        message = "Work with your fellow feedstock maintainers to get the PR merged."
        self.wait_for_done(message)

        message = (
            "After the PR is merged, wait for the CI to complete, after which "
            "the new version of Iris will be on conda-forge's servers.\n"
            "https://dev.azure.com/conda-forge/feedstock-builds/_build?definitionId=464"
        )
        self.wait_for_done(message)

        message = (
            f"Confirm that {self.version.public} appears in this list:\n"
            "https://anaconda.org/conda-forge/iris/files"
        )
        self.wait_for_done(message)

        if not self.is_release_candidate and self.is_latest_tag:
            message = (
                f"Confirm that {self.version.public} is displayed on this "
                "page as the latest available:\n"
                "https://anaconda.org/conda-forge/iris"
            )
            self.wait_for_done(message)

        if self.is_release_candidate:
            channel_command = " -c conda-forge/label/rc_iris "
        else:
            channel_command = " -c conda-forge "

        message = (
            "The new release will now undergo testing and validation in the "
            "cf-staging channel. Once this is complete, the release will be "
            "available in the standard conda-forge channel. This can "
            "sometimes take minutes, or up to an hour.\n"
            "Confirm that the new release is available for use from "
            "conda-forge by running the following command:\n"
            f"conda search{channel_command}iris=={self.version.public};"
        )
        self.wait_for_done(message)

        message = (
            "Confirm that conda (or mamba) install works as expected:\n"
            f"conda create -n tmp_iris{channel_command}iris="
            f"{self.version.public};\n"
            "conda activate tmp_iris;\n"
            'python -c "import iris; print(iris.__version__)";\n'
            "conda deactivate;\n"
            f"conda remove -n tmp_iris --all;"
        )
        self.wait_for_done(message)

        if not self.is_latest_tag and not self.more_patches_after_this_one:
            latest_version = max(self._get_tagged_versions())
            message = (
                f"{self.version} is not the latest Iris release, so the "
                f"{upstream_branch} branch needs to be restored to reflect "
                f"{latest_version}, to minimise future confusion.\n"
                "Do this via a new pull request. So long as the version number "
                "and build number match the settings from the latest release, "
                "no new conda-forge release will be triggered.\n"
            )
            self.wait_for_done(message)

    def update_links(self):
        self.print("Link updates ...")

        message = (
            "Revisit the GitHub release:\n"
            f"https://github.com/SciTools/iris/releases/tag/{self.version}\n"
            "You have confirmed that Read the Docs, PyPI and conda-forge have all "
            "updated correctly. Include the following links in the release "
            "notes:\n\n"
            f"https://scitools-iris.readthedocs.io/en/{self.version}/\n"
            f"https://pypi.org/project/scitools-iris/{self.version.public}/\n"
            f"https://anaconda.org/conda-forge/iris?version={self.version.public}\n"
        )
        self.wait_for_done(message)

        message = (
            "What is the URL for the GitHub discussions page of this "
            "release?\n"
            "https://github.com/SciTools/iris/discussions\n"
        )
        discussion_url = self.get_input(message, "Input the URL")

        message = (
            f"Update {discussion_url}, with the above "
            "links and anything else appropriate.\n"
            "The simplest way is to copy appropriate content from a previous "
            "release, then edit it to match the current release."
        )
        self.wait_for_done(message)

        message = (
            f"Comment on {discussion_url} to notify anyone watching that "
            f"{self.version} has been released."
        )
        self.wait_for_done(message)

    def bluesky_announce(self):
        message = (
            "Announce the release via https://bsky.app/profile/scitools.bsky.social, "
            "and any "
            "other appropriate message boards (e.g. Viva Engage).\n"
            "Any content used for the announcement should be stored in the "
            "SciTools/bluesky-scitools GitHub repo.\n"
        )
        if not self.first_in_series:
            message += (
                f"Consider replying within an existing "
                f"{self.version.series} "
                "announcement thread, if appropriate."
            )
        self.wait_for_done(message)

    def merge_back(self):
        self.print("Branch merge-back ...")

        merge_commit = (
            "BE SURE TO MERGE VIA A MERGE-COMMIT (not a squash-commit), to "
            "preserve the commit SHA's."
        )

        def next_series_patch() -> IrisVersion:
            tagged_versions = self._get_tagged_versions()
            series_all = sorted(set(v.series for v in tagged_versions))
            try:
                next_series = series_all[series_all.index(self.version.series) + 1]
            except (IndexError, ValueError):
                message = f"Error finding next series after {self.version.series} ."
                raise RuntimeError(message)

            series_latest = max(
                v for v in tagged_versions if v.series == next_series
            )
            return IrisVersion(
                f"{series_latest.major}.{series_latest.minor}.{series_latest.micro + 1}"
            )

        if self.more_patches_after_this_one:
            message = (
                "More series need patching. Merge into the next series' branch ..."
            )
            self.print(message)
            next_patch = next_series_patch()
            target_branch = next_patch.branch
            working_branch = f"{self.version}-to-{target_branch}"
        else:
            next_patch = None
            target_branch = "main"
            working_branch = self.version.branch + ".mergeback"

        # TODO: automate
        self._delete_local_branch(working_branch)
        message = (
            "Checkout a local branch from the official branch.\n"
            f"git fetch {self.github_scitools};\n"
            f"git checkout {self.github_scitools}/{target_branch} -b {working_branch};"
        )
        self.wait_for_done(message)

        message = (
            f"Merge in the commits from {self.version.branch}.\n"
            f"{merge_commit}\n"
            f"git merge {self.github_scitools}/{self.version.branch} --no-ff "
            f'-m "Merging {self.version.branch} into {target_branch}";'
        )
        self.wait_for_done(message)

        if self.first_in_series:
            message = (
                "Recreate the What's New template from ``main``:\n"
                f"git checkout {self.github_scitools}/main {self.whats_news.template.absolute()};\n"
            )
            self.wait_for_done(message)

            message = (
                "Recreate the What's New latest from the template:\n"
                f"cp {self.whats_news.template.absolute()} "
                f"{self.whats_news.latest.absolute()};\n"
                f"git add {self.whats_news.latest.absolute()};\n"
            )
            self.wait_for_done(message)

            message = (
                f"Follow any guidance in {self.whats_news.latest.name} to "
                "complete the recreation-from-template.\n"
                "E.g. removing the bugfix section."
            )
            self.wait_for_done(message)

            message = (
                f"In {self.whats_news.index_.absolute()}:\n"
                f"Add {self.whats_news.latest.name} to the top of the list of .rst "
                f"files, "
                f"and set the top include:: to be {self.whats_news.latest.name} ."
            )
            self.wait_for_done(message)

            message = (
                "Commit and push all the What's New changes.\n"
                f"git add {self.whats_news.index_.absolute()};\n"
                'git commit -m "Restore latest Whats-New files.";\n'
                f"git push -u {self.github_fork} {working_branch};"
            )
            self.wait_for_done(message)

        self._create_pr(
            base_org="SciTools",
            base_repo="iris",
            base_branch=target_branch,
            head_branch=working_branch,
        )

        message = (
            "COMBINING BRANCHES CAN BE RISKY; confirm that only the expected "
            "commits are in the PR."
        )
        self.wait_for_done(message)

        message = (
            "Work with the development team to get the PR merged.\n"
            f"If {self.version.branch} includes any cherry-picks, there may be "
            "merge conflicts to resolve.\n"
            "Make sure the documentation is previewed during this process.\n"
            f"{merge_commit}"
        )
        self.wait_for_done(message)

        if self.more_patches_after_this_one:
            self.print("Moving on to the next patch ...")
            assert self.version != next_patch

            # Create a special new progress file which is set up for stepping
            #  through the next patch release.
            next_patch_str = str(next_patch).replace(".", "_")
            next_patch_stem = self._get_file_stem().with_stem(next_patch_str)

            class NextPatch(IrisRelease):
                @classmethod
                def _get_file_stem(cls) -> Path:
                    return next_patch_stem

                def run(self):
                    pass

            next_patch_kwargs = self.state | dict(
                git_tag=str(next_patch),
                sha256=None,
                latest_complete_step=NextPatch.get_steps().index(NextPatch.validate) - 1,
            )
            next_patch_script = NextPatch(**next_patch_kwargs)
            next_patch_script.save()

            new_command = (
                f"python {Path(__file__).absolute()} load "
                f"{next_patch_script._file_path}"
            )
            message = (
                "Run the following command in a new terminal to address "
                f"{next_patch} next:\n"
                f"{new_command}"
            )
            self.wait_for_done(message)

    def next_release(self):
        if self.release_type is not self.ReleaseTypes.PATCH and not self.is_release_candidate:
            self.print("Prep next release ...")

            message = (
                "Confirm that there is a release manager in place for the "
                "next minor (or major) release."
            )
            self.wait_for_done(message)

            message = (
                "Confirm that the next release manager has set up a "
                "milestone for their release.\n"
                "https://github.com/SciTools/iris/milestones"
            )
            self.wait_for_done(message)

            message = (
                "Confirm that the next release manager has set up a "
                "discussion page for their release.\n"
                "https://github.com/SciTools/iris/discussions/categories/releases"
            )
            self.wait_for_done(message)

            message = (
                "Confirm that the next release manager has arranged "
                "some team development time (e.g. sprints) for "
                "delivering Iris improvements in their release.\n\n"
                "The UK Met Office has some Confluence guidance for this."
            )
            self.wait_for_done(message)

            message = (
                "Remind the next release manager about the importance "
                "of regularly championing their release (e.g. during "
                "Peloton meetings).\n\n"
                "Relying solely on a few focussed weeks cannot deliver "
                "many improvements."
            )
            self.wait_for_done(message)


if __name__ == "__main__":
    IrisRelease.main()
