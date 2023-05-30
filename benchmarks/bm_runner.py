# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Argparse conveniences for executing common types of benchmark runs.
"""

from abc import ABC, abstractmethod
import argparse
from argparse import ArgumentParser
from datetime import datetime
from importlib import import_module
from os import environ
from pathlib import Path
import re
import subprocess
from tempfile import NamedTemporaryFile
from textwrap import dedent
from typing import Literal

# The threshold beyond which shifts are 'notable'. See `asv compare`` docs
#  for more.
COMPARE_FACTOR = 1.2

BENCHMARKS_DIR = Path(__file__).parent

# Common ASV arguments for all run_types except `custom`.
ASV_HARNESS = (
    "run {posargs} --attribute rounds=4 --interleave-rounds --strict "
    "--show-stderr"
)


def echo(echo_string: str):
    # Use subprocess for printing to reduce chance of printing out of sequence
    #  with the subsequent calls.
    subprocess.run(["echo", f"BM_RUNNER DEBUG: {echo_string}"])


def _subprocess_runner(args, **kwargs):
    if kwargs.pop("asv", False):
        args.insert(0, "asv")
        kwargs["cwd"] = BENCHMARKS_DIR
    echo(" ".join(args))
    kwargs.setdefault("check", True)
    return subprocess.run(args, **kwargs)


def _subprocess_runner_capture(args, **kwargs) -> str:
    result = _subprocess_runner(args, capture_output=True, **kwargs)
    return result.stdout.decode()


def _check_requirements(package: str) -> None:
    try:
        import_module(package)
    except ImportError as exc:
        message = (
            f"No {package} install detected. Benchmarks can only "
            f"be run in an environment including {package}."
        )
        raise Exception(message) from exc


def _prep_data_gen_env() -> None:
    """
    Create/access a separate, unchanging environment for generating test data.
    """

    root_dir = BENCHMARKS_DIR.parent
    python_version = "3.11"
    data_gen_var = "DATA_GEN_PYTHON"
    if data_gen_var in environ:
        echo("Using existing data generation environment.")
    else:
        echo("Setting up the data generation environment ...")
        # Get Nox to build an environment for the `tests` session, but don't
        #  run the session. Will re-use a cached environment if appropriate.
        _subprocess_runner(
            [
                "nox",
                f"--noxfile={root_dir / 'noxfile.py'}",
                "--session=tests",
                "--install-only",
                f"--python={python_version}",
            ]
        )
        # Find the environment built above, set it to be the data generation
        #  environment.
        data_gen_python = next(
            (root_dir / ".nox").rglob(f"tests*/bin/python{python_version}")
        ).resolve()
        environ[data_gen_var] = str(data_gen_python)

        echo("Installing Mule into data generation environment ...")
        mule_dir = data_gen_python.parents[1] / "resources" / "mule"
        if not mule_dir.is_dir():
            _subprocess_runner(
                [
                    "git",
                    "clone",
                    "https://github.com/metomi/mule.git",
                    str(mule_dir),
                ]
            )
        _subprocess_runner(
            [
                str(data_gen_python),
                "-m",
                "pip",
                "install",
                str(mule_dir / "mule"),
            ]
        )

        echo("Data generation environment ready.")


def _setup_common() -> None:
    _check_requirements("asv")
    _check_requirements("nox")

    _prep_data_gen_env()

    echo("Setting up ASV ...")
    _subprocess_runner(["machine", "--yes"], asv=True)

    echo("Setup complete.")


def _asv_compare(*commits: str, overnight_mode: bool = False) -> None:
    """Run through a list of commits comparing each one to the next."""
    commits = [commit[:8] for commit in commits]
    for i in range(len(commits) - 1):
        before = commits[i]
        after = commits[i + 1]
        asv_command = (
            f"compare {before} {after} --factor={COMPARE_FACTOR} --split"
        )

        _subprocess_runner(asv_command.split(" "), asv=True)
        # Now store the results in variables.
        comparison = _subprocess_runner_capture(
            asv_command.split(" "), asv=True
        )
        shifts = _subprocess_runner_capture(
            [*asv_command.split(" "), "--only-changed"], asv=True
        )

        # Write the comparisons report to a file.
        comps_dir = BENCHMARKS_DIR / ".asv" / "performance-comparisons"
        comps_dir.mkdir(exist_ok=True, parents=True)
        comps_path = (comps_dir / f"{after}-vs-{before}").with_suffix(".txt")
        comps_path.write_text(comparison)

        if shifts or (not overnight_mode):
            # For the overnight run: only post if there are shifts.
            _gh_post_results(after, comparison, shifts)


def _gh_post_results(commit_sha: str, results_full: str, results_shifts: str):
    """
    Post the results on GitHub, if running under GitHub Actions.

    If benchmarking a pull request: post as a new comment. Otherwise: post as
    a new issue.
    """
    if "GITHUB_ACTIONS" not in environ:
        # Only run when within GHA.
        return

    on_pull_request = "GITHUB_BASE_REF" in environ
    run_id = environ["GITHUB_RUN_ID"]
    repo = environ["GITHUB_REPOSITORY"]
    gha_run_link = (
        f"[`{run_id}`](https://github.com/{repo}/actions/runs/{run_id})"
    )

    performance_report = dedent(
        (
            f"""
            ### Performance Benchmark Report: {commit_sha}

            <details>
            <summary>Performance shifts</summary>

            ```
            {results_shifts}
            ```

            </details>

            <details>
            <summary>Full benchmark results</summary>

            ```
            {results_full}
            ```

            </details>

            Generated by GHA run {gha_run_link}
            """
        )
    )

    body_file = NamedTemporaryFile("w")
    body_file_path = Path(body_file.name)

    if on_pull_request:
        # Post the report as a comment on the active PR.
        body_file_path.write_text(performance_report)
        pr_number = _subprocess_runner_capture(
            ["echo", "${{github.event.number}}"]
        ).rstrip
        command = (
            f"gh pr comment {pr_number} "
            f"--body-file {body_file_path} "
            f"--repo {repo}"
        )
        _subprocess_runner(command.split(" "))
    else:
        # Post the report as new issue.
        commit_msg = _subprocess_runner_capture(
            f"git log {commit_sha}^! --oneline".split(" ")
        )
        # Intended for benchmarking commits on trunk - should include a PR
        #  number due to our squash policy.
        pr_tag_match = re.search("#[0-9]*", commit_msg)

        assignee = ""
        pr_tag = "pull request number unavailable"
        if pr_tag_match is not None:
            pr_tag = pr_tag_match.group(0)

            for login_type in ("author", "mergedBy"):
                gh_query = f'.["{login_type}"]["login"]'
                command = (
                    f"gh pr view {pr_tag[1:]} "
                    f"--json {login_type} -q '{gh_query}' "
                    f"--repo {repo}"
                )
                login = _subprocess_runner_capture(command.split(" "))

                command = [
                    "curl",
                    "-s",
                    f"https://api.github.com/users/{login}",
                ]
                login_info = _subprocess_runner_capture(command)
                is_user = '"type": "User"' in login_info
                if is_user:
                    assignee = login
                    break

        title = f"Performance Shift(s): `{commit_sha}`"
        body = dedent(
            (
                f"""
                Benchmark comparison has identified performance shifts at:

                * commit {commit_sha} ({pr_tag}).

                <p>
                Please review the report below and 
                take corrective/congratulatory action as appropriate 
                :slightly_smiling_face:
                </p>
                """
            )
        )
        body += performance_report
        body_file_path.write_text(body)

        # Forced to use the less-readable list construction here due to spaces
        #  in some of the arguments.
        command = [
            "gh",
            "issue",
            "create",
            f"--title={title}",
            f"--body-file={body_file_path}",
            f"--assignee={assignee}",
            '--label="Bot"',
            '--label="Type: Performance"',
            f"--repo={repo}",
        ]
        _subprocess_runner(command)

    body_file.close()


class _SubParserGenerator(ABC):
    """Convenience for holding all the necessary argparse info in 1 place."""

    name: str = NotImplemented
    description: str = NotImplemented
    epilog: str = NotImplemented

    def __init__(self, subparsers: ArgumentParser.add_subparsers) -> None:
        self.subparser: ArgumentParser = subparsers.add_parser(
            self.name,
            description=self.description,
            epilog=self.epilog,
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self.add_arguments()
        self.subparser.add_argument(
            "asv_args",
            nargs=argparse.REMAINDER,
            help="Any number of arguments to pass down to ASV.",
        )
        self.subparser.set_defaults(func=self.func)

    @abstractmethod
    def add_arguments(self) -> None:
        """All self.subparser.add_argument() calls."""
        _ = NotImplemented

    @staticmethod
    @abstractmethod
    def func(args: argparse.Namespace):
        """
        The function to return when the subparser is parsed.

        `func` is then called, performing the user's selected sub-command.

        """
        _ = args
        return NotImplemented


class Overnight(_SubParserGenerator):
    name = "overnight"
    description = (
        "Benchmarks all commits between the input **first_commit** to ``HEAD``, "
        "comparing each to its parent for performance shifts. Commit"
        "performance comparisons are saved to a file in:\n"
        "``.asv/performance-comparisons/``\n\n"
        "If running on GitHub Actions: performance shift(s) will be reported "
        "in a new issue.\n"
        "Designed for checking the previous 24 hours' commits, typically in a "
        "scheduled script."
    )
    epilog = (
        "e.g. python bm_runner.py overnight a1b23d4\n"
        "e.g. python bm_runner.py overnight a1b23d4 --bench=regridding"
    )

    def add_arguments(self) -> None:
        self.subparser.add_argument(
            "first_commit",
            type=str,
            help="The first commit in the benchmarking commit sequence.",
        )

    @staticmethod
    def func(args: argparse.Namespace) -> None:
        _setup_common()

        commit_range = f"{args.first_commit}^^.."
        asv_command = ASV_HARNESS.format(posargs=commit_range)
        _subprocess_runner([*asv_command.split(" "), *args.asv_args], asv=True)

        # git rev-list --first-parent is the command ASV uses.
        git_command = f"git rev-list --first-parent {commit_range}"
        commit_string = _subprocess_runner_capture(git_command.split(" "))
        commit_list = commit_string.rstrip().split("\n")
        _asv_compare(*reversed(commit_list), overnight_mode=True)


class Branch(_SubParserGenerator):
    name = "branch"
    description = (
        "Performs the same operations as ``overnight``, but always on two commits "
        "only - ``HEAD``, and ``HEAD``'s merge-base with the input "
        "**base_branch**. Commit performance comparisons are saved to a file in:\n"
        "``.asv/performance-comparisons/``\n\n"
        "If running on GitHub Actions: HEAD will be GitHub's merge commit and"
        "merge-base will be the merge target. Performance results will be posted "
        "on the relevant pull request.\n"
        "Designed "
        "for testing if the active branch's changes cause performance shifts - "
        "anticipating what would be caught by ``overnight`` once merged.\n\n"
        "**For maximum accuracy, avoid using the machine that is running this "
        "session. Run time could be >1 hour for the full benchmark suite.**"
    )
    epilog = (
        "e.g. python bm_runner.py branch upstream/main\n"
        "e.g. python bm_runner.py branch upstream/main --bench=regridding"
    )

    def add_arguments(self) -> None:
        self.subparser.add_argument(
            "base_branch",
            type=str,
            help="A branch that has the merge-base with ``HEAD`` - ``HEAD`` will be benchmarked against that merge-base.",
        )

    @staticmethod
    def func(args: argparse.Namespace) -> None:
        _setup_common()

        git_command = f"git merge-base HEAD {args.base_branch}"
        merge_base = _subprocess_runner_capture(git_command.split(" "))[:8]

        with NamedTemporaryFile("w") as hashfile:
            hashfile.writelines([merge_base, "\n", "HEAD"])
            hashfile.flush()
            commit_range = f"HASHFILE:{hashfile.name}"
            asv_command = ASV_HARNESS.format(posargs=commit_range)
            _subprocess_runner(
                [
                    *asv_command.split(" "),
                    "--bench",
                    "aux_factory",
                    "--attribute",
                    "rounds=1",
                    *args.asv_args,
                ],
                asv=True,
            )

        _asv_compare(merge_base, "HEAD")


class _CSPerf(_SubParserGenerator, ABC):
    """Common code used by both CPerf and SPerf."""

    description = (
        "Run the on-demand {} suite of benchmarks (part of the UK Met "
        "Office NG-VAT project) for the ``HEAD`` of ``upstream/main`` only, "
        "and publish the results to the input **publish_dir**, within a "
        "unique subdirectory for this run."
    )
    epilog = (
        "e.g. python bm_runner.py {0} my_publish_dir\n"
        "e.g. python bm_runner.py {0} my_publish_dir --bench=regridding"
    )

    def add_arguments(self) -> None:
        self.subparser.add_argument(
            "publish_dir",
            type=str,
            help="HTML results will be published to a sub-dir in this dir.",
        )

    @staticmethod
    def csperf(
        args: argparse.Namespace, run_type: Literal["cperf", "sperf"]
    ) -> None:
        _setup_common()

        publish_dir = Path(args.publish_dir)
        if not publish_dir.is_dir():
            message = (
                f"Input 'publish directory' is not a directory: {publish_dir}"
            )
            raise NotADirectoryError(message)
        publish_subdir = (
            publish_dir
            / f"{run_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        publish_subdir.mkdir()

        # Activate on demand benchmarks (C/SPerf are deactivated for
        #  'standard' runs).
        environ["ON_DEMAND_BENCHMARKS"] = "True"
        commit_range = "upstream/main^!"

        asv_command = (
            ASV_HARNESS.format(posargs=commit_range) + f" --bench={run_type}"
        )
        # C/SPerf benchmarks are much bigger than the CI ones:
        # Don't fail the whole run if memory blows on 1 benchmark.
        asv_command = asv_command.replace(" --strict", "")
        # Only do a single round.
        asv_command = re.sub(r"rounds=\d", "rounds=1", asv_command)
        _subprocess_runner([*asv_command.split(" "), *args.asv_args], asv=True)

        asv_command = f"publish {commit_range} --html-dir={publish_subdir}"
        _subprocess_runner(asv_command.split(" "), asv=True)

        # Print completion message.
        location = BENCHMARKS_DIR / ".asv"
        print(
            f'New ASV results for "{run_type}".\n'
            f'See "{publish_subdir}",'
            f'\n  or JSON files under "{location / "results"}".'
        )


class CPerf(_CSPerf):
    name = "cperf"
    description = _CSPerf.description.format("CPerf")
    epilog = _CSPerf.epilog.format("cperf")

    @staticmethod
    def func(args: argparse.Namespace) -> None:
        _CSPerf.csperf(args, "cperf")


class SPerf(_CSPerf):
    name = "sperf"
    description = _CSPerf.description.format("SPerf")
    epilog = _CSPerf.epilog.format("sperf")

    @staticmethod
    def func(args: argparse.Namespace) -> None:
        _CSPerf.csperf(args, "sperf")


class Custom(_SubParserGenerator):
    name = "custom"
    description = (
        "Run ASV with the input **ASV sub-command**, without any preset "
        "arguments - must all be supplied by the user. So just like running "
        "ASV manually, with the convenience of re-using the runner's "
        "scripted setup steps."
    )
    epilog = "e.g. python bm_runner.py custom continuous a1b23d4 HEAD --quick"

    def add_arguments(self) -> None:
        self.subparser.add_argument(
            "asv_sub_command",
            type=str,
            help="The ASV command to run.",
        )

    @staticmethod
    def func(args: argparse.Namespace) -> None:
        _setup_common()
        _subprocess_runner([args.asv_sub_command, *args.asv_args], asv=True)


def main():
    parser = ArgumentParser(
        description="Run the Iris performance benchmarks (using Airspeed Velocity).",
        epilog="More help is available within each sub-command.",
    )
    subparsers = parser.add_subparsers(required=True)

    for gen in (Overnight, Branch, CPerf, SPerf, Custom):
        _ = gen(subparsers).subparser

    parsed = parser.parse_args()
    parsed.func(parsed)


if __name__ == "__main__":
    main()
