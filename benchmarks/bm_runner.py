from abc import ABC, abstractmethod
import argparse
from argparse import ArgumentParser
from datetime import datetime
from os import environ
from pathlib import Path
import re
import subprocess
from tempfile import NamedTemporaryFile
from typing import Literal

from pkg_resources import parse_version

# The threshold beyond which shifts are 'notable'. See `asv compare`` docs
#  for more.
COMPARE_FACTOR = 1.2


# Common ASV arguments for all run_types except `custom`.
ASV_HARNESS = (
    "asv run {posargs} --attribute rounds=4 --interleave-rounds --strict "
    "--show-stderr"
)


def prep_data_gen_env():
    # TODO: docstring

    python_version = "3.10"
    data_gen_var = "DATA_GEN_PYTHON"
    if data_gen_var in environ:
        print("Using existing data generation environment.")
    else:
        print("Setting up the data generation environment...")
        # Get Nox to build an environment for the `tests` session, but don't
        #  run the session. Will re-use a cached environment if appropriate.
        subprocess.run(
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

        mule_dir = data_gen_python.parents[1] / "resources" / "mule"
        if not mule_dir.is_dir():
            print("Installing Mule into data generation environment...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/metomi/mule.git",
                    str(mule_dir),
                ]
            )
        subprocess.run(
            [
                str(data_gen_python),
                "-m",
                "pip",
                "install",
                str(mule_dir / "mule"),
            ]
        )

        print("Data generation environment ready.")


def setup_common():
    prep_data_gen_env()

    print("Setting up ASV...")
    subprocess.run(["asv", "machine", "--yes"])

    print("Setup complete.")


def asv_compare(*commits: str, overnight_mode: bool = False):
    """Run through a list of commits comparing each one to the next."""
    commits = [commit[:8] for commit in commits]
    shifts_dir = Path(".asv") / "performance-shifts"
    for i in range(len(commits) - 1):
        before = commits[i]
        after = commits[i + 1]
        asv_command = (
            f"asv compare {before} {after} --factor={COMPARE_FACTOR} --split"
        )
        subprocess.run(asv_command.split(" "))

        if overnight_mode:
            # Record performance shifts.
            # Run the command again but limited to only showing performance
            #  shifts.
            shifts = subprocess.run(
                [*asv_command.split(" "), "--only-changed"],
                capture_output=True,
                text=True,
            ).stdout
            if shifts:
                # Write the shifts report to a file.
                # Dir is used by .github/workflows/benchmarks.yml,
                #  but not cached - intended to be discarded after run.
                shifts_dir.mkdir(exist_ok=True, parents=True)
                shifts_path = (shifts_dir / after).with_suffix(".txt")
                with shifts_path.open("w") as shifts_file:
                    shifts_file.write(shifts)


class SubParserGenerator(ABC):
    """Convenience for holding all the necessary info in 1 place."""

    name: str = NotImplemented
    description: str = NotImplemented

    def __init__(self, subparsers: ArgumentParser.add_subparsers):
        self.subparser: ArgumentParser = subparsers.add_parser(
            self.name,
            description=self.description,
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
        _ = args
        return NotImplemented


class Overnight(SubParserGenerator):
    name = "overnight"
    description = (
        "Benchmarks all commits between the input **first_commit** to ``HEAD``, "
        "comparing each to its parent for performance shifts. If a commit causes "
        "shifts, the output is saved to a file:\n"
        "``.asv/performance-shifts/<commit-sha>``\n\n"
        "Designed for checking the previous 24 hours' commits, typically in a "
        "scheduled script."
    )

    def add_arguments(self) -> None:
        self.subparser.add_argument(
            "first_commit",
            type=str,
            help="The first commit in the benchmarking commit sequence.",
        )

    @staticmethod
    def func(args: argparse.Namespace):
        setup_common()

        commit_range = f"{args.first_commit}^^.."
        asv_command = ASV_HARNESS.format(posargs=commit_range)
        subprocess.run(*asv_command.split(" "), *args.asv_args)

        # git rev-list --first-parent is the command ASV uses.
        git_command = f"git rev-list --first-parent {commit_range}"
        commit_string = subprocess.run(
            git_command.split(" "), capture_output=True, text=True
        ).stdout
        commit_list = commit_string.rstrip().split("\n")
        asv_compare(*reversed(commit_list), overnight_mode=True)


class Branch(SubParserGenerator):
    name = "branch"
    description = (
        "Performs the same operations as ``overnight``, but always on two commits "
        "only - ``HEAD``, and ``HEAD``'s merge-base with the input "
        "**base_branch**. Output from this run is never saved to a file. Designed "
        "for testing if the active branch's changes cause performance shifts - "
        "anticipating what would be caught by ``overnight`` once merged.\n\n"
        "**For maximum accuracy, avoid using the machine that is running this "
        "session. Run time could be >1 hour for the full benchmark suite.**"
    )

    def add_arguments(self) -> None:
        self.subparser.add_argument(
            "base_branch",
            type=str,
            help="A branch that has the merge-base with ``HEAD`` - ``HEAD`` will be benchmarked against that merge-base.",
        )

    @staticmethod
    def func(args: argparse.Namespace):
        setup_common()

        git_command = f"git merge-base HEAD {args.base_branch}"
        merge_base = subprocess.run(
            git_command.split(" "), capture_output=True, text=True
        ).stdout[:8]

        with NamedTemporaryFile("w") as hashfile:
            hashfile.writelines([merge_base, "\n", "HEAD"])
            hashfile.flush()
            commit_range = f"HASHFILE:{hashfile.name}"
            asv_command = ASV_HARNESS.format(posargs=commit_range)
            subprocess.run([*asv_command.split(" "), *args.asv_args])

        asv_compare(merge_base, "HEAD")


class CSPerf(SubParserGenerator, ABC):
    description = (
        "Run the on-demand {} suite of benchmarks (part of the UK Met "
        "Office NG-VAT project) for the ``HEAD`` of ``upstream/main`` only, "
        "and publish the results to the input **publish_dir**, within a "
        "unique subdirectory for this run."
    )

    def add_arguments(self) -> None:
        self.subparser.add_argument(
            "publish_dir",
            type=str,
            help="HTML results will be published to a sub-dir in this dir.",
        )

    @staticmethod
    def csperf(args: argparse.Namespace, run_type: Literal["cperf", "sperf"]):
        setup_common()

        if not args.publish_dir.is_dir():
            message = f"Input 'publish directory' is not a directory: {args.publish_dir}"
            raise NotADirectoryError(message)
        publish_subdir = (
            args.publish_dir
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
        subprocess.run([*asv_command.split(" "), *args.asv_args])

        asv_command = f"asv publish {commit_range} --html-dir={publish_subdir}"
        subprocess.run(asv_command.split(" "))

        # Print completion message.
        location = Path().cwd() / ".asv"
        print(
            f'New ASV results for "{run_type}".\n'
            f'See "{publish_subdir}",'
            f'\n  or JSON files under "{location / "results"}".'
        )


class CPerf(CSPerf):
    name = "cperf"
    description = CSPerf.description.format("CPerf")

    @staticmethod
    def func(args: argparse.Namespace):
        super().csperf(args, "cperf")


class SPerf(CSPerf):
    name = "sperf"
    description = CSPerf.description.format("SPerf")

    @staticmethod
    def func(args: argparse.Namespace):
        super().csperf(args, "sperf")


class Custom(SubParserGenerator):
    name = "custom"
    description = (
        "Run ASV with the input **ASV sub-command**, without any preset "
        "arguments - must all be supplied by the user. So just like running "
        "ASV manually, with the convenience of re-using the runners' "
        "scripted setup steps."
    )

    def add_arguments(self) -> None:
        self.subparser.add_argument(
            "asv_sub_command",
            type=str,
            help="The ASV command to run.",
        )

    @staticmethod
    def func(args: argparse.Namespace):
        setup_common()
        subprocess.run(["asv", args.asv_sub_command, *args.asv_args])


def main():
    try:
        import asv
    except ImportError as exc:
        message = (
            "No Airspeed Velocity (ASV) install detected. Benchmarks can only "
            "be run in an environment including ASV."
        )
        raise Exception(message) from exc

    parser = ArgumentParser(
        description="Run the Iris performance benchmarks (using Airspeed Velocity).",
    )
    subparsers = parser.add_subparsers()

    for gen in (Overnight, Branch, CPerf, SPerf, Custom):
        _ = gen(subparsers).subparser

    parsed = parser.parse_args()
    parsed.func(parsed)
    pass


if __name__ == "__main__":
    main()
