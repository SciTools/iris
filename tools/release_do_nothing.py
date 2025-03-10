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
import typing

try:
    from nothing import Progress
except ImportError:
    install_message = (
        "This script requires the `nothing` package to be installed:\n"
        "pip install git+https://github.com/SciTools-incubator/nothing.git"
    )
    raise ImportError(install_message)


class IrisRelease(Progress):
    class ReleaseTypes(IntEnum):
        MAJOR = 0
        MINOR = 1
        PATCH = 2

    github_user: str = None
    release_type: ReleaseTypes = None
    git_tag: str = None  # v1.2.3rc0
    first_in_series: bool = None
    sha256: str = None

    @classmethod
    def get_cmd_description(cls) -> str:
        return "Do-nothing workflow for the Iris release process."

    @classmethod
    def get_steps(cls) -> list[typing.Callable[..., None]]:
        return [
            cls.get_github_user,
            cls.get_release_type,
            cls.get_release_tag,
            cls.check_release_candidate,
            cls.check_first_in_series,
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
        ]

    def get_github_user(self):
        def validate(input_user: str) -> str | None:
            if not re.fullmatch(r"[a-zA-Z0-9-]+", input_user):
                self.report_problem("Invalid GitHub username. Please try again ...")
            else:
                return input_user

        message = (
            "Please input your GitHub username.\n"
            "This is used in the URLs for creating pull requests."
        )
        self.set_value_from_input(
            key="github_user",
            message=message,
            expected_inputs="Username",
            post_process=validate,
        )
        self.print(f"GitHub username = {self.github_user}")

    def get_release_type(self):
        def validate(input_value: str) -> IrisRelease.ReleaseTypes | None:
            try:
                return self.ReleaseTypes(int(input_value))
            except ValueError:
                self.report_problem("Invalid release type. Please try again ...")

        self.set_value_from_input(
            key="release_type",
            message="What type of release are you preparing?\nhttps://semver.org/",
            expected_inputs=f"Choose a number {tuple(self.ReleaseTypes)}",
            post_process=validate,
        )
        self.print(f"{repr(self.release_type)} confirmed.")

    def get_release_tag(self):
        # TODO: automate using setuptools_scm.

        def validate(input_tag: str) -> str | None:
            # TODO: use the packaging library?
            version_mask = r"v\d+\.\d+\.\d+\D*.*"
            regex_101 = "https://regex101.com/r/dLVaNH/1"
            if re.fullmatch(version_mask, input_tag) is None:
                problem_message = (
                    "Release tag does not match the input mask:\n"
                    f"{version_mask}\n"
                    f"({regex_101})"
                )
                self.report_problem(problem_message)
            else:
                return input_tag  # v1.2.3rc0

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

    class Strings(typing.NamedTuple):
        series: str
        branch: str
        release: str

    @property
    def strings(self) -> Strings:
        series = ".".join(self.git_tag.split(".")[:2])  # v1.2
        return self.Strings(
            series=series,
            branch=series + ".x",  # v1.2.x
            release=self.git_tag[1:],  # 1.2.3rc0
        )

    @property
    def is_release_candidate(self) -> bool:
        return "rc" in self.git_tag

    def check_release_candidate(self):
        message = "Checking tag for release candidate: "
        if self.is_release_candidate:
            message += "DETECTED\nThis IS a release candidate."
        else:
            message += "NOT DETECTED\nThis IS NOT a release candidate."
        self.print(message)

        if self.release_type == self.ReleaseTypes.PATCH and self.is_release_candidate:
            message = (
                "Release candidates are not expected for PATCH releases. "
                "Are you sure you want to continue?"
            )
            if self.get_input(message, "y / [n]").casefold() != "y".casefold():
                exit()

    def check_first_in_series(self):
        if self.release_type != self.ReleaseTypes.PATCH:
            message = (
                f"Is this the first release in the {self.strings.series} "
                f"series, including any release candidates?"
            )
            self.set_value_from_input(
                key="first_in_series",
                message=message,
                expected_inputs="y / n",
                post_process=lambda x: x.casefold() == "y".casefold(),
            )
            if self.first_in_series:
                self.print("First in series confirmed.")
                if not self.is_release_candidate:
                    message = (
                        "The first release in a series is expected to be a "
                        "release candidate, but this is not. Are you sure you "
                        "want to continue?"
                    )
                    if self.get_input(message, "y / [n]").casefold() != "y".casefold():
                        exit()
            else:
                self.print("Existing series confirmed.")

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
            working_branch = self.strings.branch + ".standard_names"
            self._delete_local_branch(working_branch)
            message = (
                "Checkout a local branch from the official ``main`` branch.\n"
                "git fetch upstream;\n"
                f"git checkout upstream/main -b {working_branch};"
            )
            self.wait_for_done(message)

            url = "https://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml"
            file = Path(__file__).parents[1] / "etc" / "cf-standard-name-table.xml"
            message = (
                "Update the CF standard names table to the latest version:\n"
                f'wget "{url}" -O {file};\n'
                f"git add {file};\n"
                "git commit -m 'Update CF standard names table.';\n"
                f"git push -u origin {working_branch};"
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
        if self.release_type == self.ReleaseTypes.MAJOR:
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
                f"``{self.strings.branch}`` release branch from ``main``."
            )
            self.wait_for_done(message)

        else:
            message = (
                "Cherry-pick any specific commits that are needed from ``main`` "
                f"onto {self.strings.branch} , to get the CI passing.\n"
                "E.g. a new dependency pin may have been introduced since "
                f"{self.strings.branch} was last updated from ``main``.\n"
                "DO NOT squash-merge - want to preserve the original commit "
                "SHA's."
            )
            self.wait_for_done(message)

    @staticmethod
    def _delete_local_branch(branch_name: str):
        message = (
            "Before the next step, avoid a name clash by deleting any "
            "existing local branch, if one exists.\n"
            f"git branch -D {branch_name};\n"
            f"git push -d origin {branch_name};"
        )
        IrisRelease.wait_for_done(message)

    class WhatsNewRsts(typing.NamedTuple):
        latest: Path
        release: Path
        index: Path
        template: Path

    @property
    def whats_news(self) -> WhatsNewRsts:
        src_dir = Path(__file__).parents[1] / "docs" / "src"
        whatsnew_dir = src_dir / "whatsnew"
        assert whatsnew_dir.is_dir()
        latest = whatsnew_dir / "latest.rst"

        return self.WhatsNewRsts(
            latest=latest,
            release=whatsnew_dir / (self.strings.series[1:] + ".rst"),
            index=whatsnew_dir / "index.rst",
            template=latest.with_suffix(".rst.template"),
        )

    def finalise_whats_new(self):
        self.print("What's New finalisation ...")

        working_branch = self.strings.branch + ".updates"
        self._delete_local_branch(working_branch)
        message = (
            f"Checkout a local branch from the official {self.strings.branch} "
            f"branch.\n"
            "git fetch upstream;\n"
            f"git checkout upstream/{self.strings.branch} -b "
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
                f"In {self.whats_news.index.absolute()}:\n"
                f"Replace references to {self.whats_news.latest.name} with "
                f"{self.whats_news.release.name}"
            )
            self.wait_for_done(message)

        self.print(f"What's New file path = {self.whats_news.release}")

        if not self.release_type == self.ReleaseTypes.PATCH:
            whatsnew_title = (
                f"{self.strings.series} ({datetime.today().strftime('%d %b %Y')}"
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

            dropdown_title = f"\n{self.strings.series} Release Highlights\n"
            message = (
                f"In {self.whats_news.release.name}: set the sphinx-design "
                f"dropdown title to:{dropdown_title}"
            )
            self.wait_for_done(message)

            message = (
                f"Review {self.whats_news.release.name} to ensure it is a good "
                f"reflection of what is new in {self.strings.series}.\n"
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
            f"git add {self.whats_news.index.absolute()};\n"
            f'git commit -m "Whats new updates for {self.git_tag} .";\n'
            f"git push -u origin {working_branch};"
        )
        self.wait_for_done(message)

        self._create_pr(
            base_org="SciTools",
            base_repo="iris",
            base_branch=self.strings.branch,
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
            f"Select {self.strings.branch} as the Target.\n"
            f"Input {self.git_tag} as the new tag to create, and also as "
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
                f"conda install -c conda-forge/label/rc_iris iris={self.strings.release}\n"
                f"pip install scitools-iris=={self.strings.release}"
            )
            self.wait_for_done(message)

            message = (
                "This is a release candidate - tick the box to set this as a "
                "pre-release."
            )
            self.wait_for_done(message)

        else:
            message = "Tick the box to set this as the latest release."
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

        message = f"Set {self.git_tag} to Active, un-Hidden."
        self.wait_for_done(message)

        message = f"Set {self.strings.branch} to Active, Hidden."
        self.wait_for_done(message)

        message = (
            "Keep only the latest 2 branch doc builds active - "
            f"'{self.strings.branch}' and the previous one - deactivate older "
            "ones."
        )
        self.wait_for_done(message)

        message = (
            f"Visit https://scitools-iris.readthedocs.io/en/{self.git_tag} "
            "to confirm:\n\n"
            "- The docs have rendered.\n"
            "- The version badge in the top left reads:\n"
            f"  'version (archived) | {self.git_tag}'\n"
            "   (this demonstrates that setuptools_scm has worked correctly).\n"
            "- The What's New looks correct.\n"
            f"- {self.git_tag} is available in RTD's version switcher.\n\n"
            "NOTE: the docs can take several minutes to finish building."
        )
        if not self.is_release_candidate:
            message += (
                "- Selecting 'stable' in the version switcher also brings up "
                f"the {self.git_tag} render."
            )
        self.wait_for_done(message)

        message = (
            f"Visit https://scitools-iris.readthedocs.io/en/{self.strings.branch} "
            "to confirm:\n\n"
            "- The docs have rendered\n"
            f"- The version badge in the top left includes: {self.strings.branch} .\n"
            f"- {self.strings.branch} is NOT available in RTD's version switcher.\n\n"
            "NOTE: the docs can take several minutes to finish building."
        )
        self.wait_for_done(message)

    def check_pypi(self):
        self.print("PyPI checks ...")
        self.print("If anything goes wrong, manual steps are in the documentation.")

        message = (
            "Confirm that the following URL is correctly populated:\n"
            f"https://pypi.org/project/scitools-iris/{self.strings.release}/"
        )
        self.wait_for_done(message)

        message = (
            f"Confirm that {self.strings.release} is at the top of this page:\n"
            "https://pypi.org/project/scitools-iris/#history"
        )
        self.wait_for_done(message)

        if self.is_release_candidate:
            message = (
                f"Confirm that {self.strings.release} is marked as a "
                f"pre-release on this page:\n"
                "https://pypi.org/project/scitools-iris/#history"
            )
        else:
            message = (
                f"Confirm that {self.strings.release} is the tag shown on the "
                "scitools-iris PyPI homepage:\n"
                "https://pypi.org/project/scitools-iris/"
            )
        self.wait_for_done(message)

        def validate(sha256_string: str) -> str:
            valid = True
            try:
                _ = int(sha256_string, 16)
            except ValueError:
                valid = False
            valid = valid and len(sha256_string) == 64

            if not valid:
                self.report_problem("Invalid SHA256 hash. Please try again ...")
            else:
                return sha256_string

        message = (
            f"Visit the below and click `view hashes` for the Source Distribution"
            f"(`.tar.gz`):\n"
            f"https://pypi.org/project/scitools-iris/{self.strings.release}#files\n"
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
            f"pip install scitools-iris=={self.strings.release};\n"
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
            f"{self.git_tag};\n"
        )
        self.wait_for_done(message)

        message = (
            "Update ./recipe/meta.yaml:\n\n"
            f"- The version at the very top of the file: "
            f"{self.strings.release}\n"
            f"- The sha256 hash: {self.sha256}\n"
            "- Requirements: align the packages and pins with those in the "
            "Iris repo\n"
            "- Maintainers: update with any changes to the dev team\n"
            "- Skim read the entire file to see if anything else is out of"
            "date, e.g. is the licence info still correct? Ask the lead "
            "Iris developers if unsure.\n"
        )
        self.wait_for_done(message)

        # TODO: automate
        message = (
            "No other file normally needs changing in iris-feedstock, "
            "so push up "
            "the changes to prepare for a Pull Request:\n"
            f"git add recipe/meta.yaml;\n"
            f'git commit -m "Recipe updates for {self.git_tag} .";\n'
            f"git push -u origin {self.git_tag};"
        )
        self.wait_for_done(message)

        self._create_pr(
            base_org="conda-forge",
            base_repo="iris-feedstock",
            base_branch=upstream_branch,
            head_branch=self.git_tag,
        )

        if self.is_release_candidate:
            readme_url = f"https://github.com/{self.github_user}/iris-feedstock/blob/{self.git_tag}/README.md"
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
            f"Confirm that {self.strings.release} appears in this list:\n"
            "https://anaconda.org/conda-forge/iris/files"
        )
        self.wait_for_done(message)

        if not self.is_release_candidate:
            message = (
                f"Confirm that {self.strings.release} is displayed on this "
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
            f"conda search{channel_command}iris=={self.strings.release};"
        )
        self.wait_for_done(message)

        message = (
            "Confirm that conda (or mamba) install works as expected:\n"
            f"conda create -n tmp_iris{channel_command}iris="
            f"{self.strings.release};\n"
            "conda activate tmp_iris;\n"
            'python -c "import iris; print(iris.__version__)";\n'
            "conda deactivate;\n"
            f"conda remove -n tmp_iris --all;"
        )
        self.wait_for_done(message)

    def update_links(self):
        self.print("Link updates ...")

        message = (
            "Revisit the GitHub release:\n"
            f"https://github.com/SciTools/iris/releases/tag/{self.git_tag}\n"
            "You have confirmed that Read the Docs, PyPI and conda-forge have all "
            "updated correctly. Include the following links in the release "
            "notes:\n\n"
            f"https://scitools-iris.readthedocs.io/en/{self.git_tag}/\n"
            f"https://pypi.org/project/scitools-iris/{self.strings.release}/\n"
            f"https://anaconda.org/conda-forge/iris?version={self.strings.release}\n"
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
            f"{self.git_tag} has been released."
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
                f"{self.strings.series} "
                "announcement thread, if appropriate."
            )
        self.wait_for_done(message)

    def merge_back(self):
        self.print("Branch merge-back ...")

        merge_commit = (
            "BE SURE TO MERGE VIA A MERGE-COMMIT (not a squash-commit), to "
            "preserve the commit SHA's."
        )

        if self.first_in_series:
            # TODO: automate

            working_branch = self.strings.branch + ".mergeback"
            self._delete_local_branch(working_branch)
            message = (
                "Checkout a local branch from the official ``main`` branch.\n"
                "git fetch upstream;\n"
                f"git checkout upstream/main -b {working_branch};"
            )
            self.wait_for_done(message)

            message = (
                f"Merge in the commits from {self.strings.branch}.\n"
                f"{merge_commit}\n"
                f"git merge upstream/{self.strings.branch} --no-ff "
                '-m "Merging release branch into main";'
            )
            self.wait_for_done(message)

            message = (
                "Recreate the What's New template from ``main``:\n"
                f"git checkout upstream/main {self.whats_news.template.absolute()};\n"
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
                f"In {self.whats_news.index.absolute()}:\n"
                f"Add {self.whats_news.latest.name} to the top of the list of .rst "
                f"files, "
                f"and set the top include:: to be {self.whats_news.latest.name} ."
            )
            self.wait_for_done(message)

            message = (
                "Commit and push all the What's New changes.\n"
                f"git add {self.whats_news.index.absolute()};\n"
                'git commit -m "Restore latest Whats New files.";\n'
                f"git push -u origin {working_branch};"
            )
            self.wait_for_done(message)

            self._create_pr(
                base_org="SciTools",
                base_repo="iris",
                base_branch="main",
                head_branch=working_branch,
            )
            message = (
                "Work with the development team to get the PR merged.\n"
                "Make sure the documentation is previewed during this process.\n"
                f"{merge_commit}"
            )
            self.wait_for_done(message)

        else:
            message = (
                f"Propose a merge-back from {self.strings.branch} into "
                f"``main`` by "
                f"visiting this URL and clicking `Create pull request`:\n"
                f"https://github.com/SciTools/iris/compare/main..."
                f"{self.strings.branch}\n"
                f"{merge_commit}"
            )
            self.wait_for_done(message)
            message = (
                f"Once the pull request is merged ensure that the "
                f"{self.strings.branch} "
                "release branch is restored.\n"
                "GitHub automation rules may have automatically deleted the "
                "release branch."
            )
            self.wait_for_done(message)


if __name__ == "__main__":
    IrisRelease.main()
