from abc import ABC, abstractmethod
import argparse
from argparse import ArgumentParser

# Common ASV arguments for all run_types except `custom`.
ASV_HARNESS = (
    "asv run {posargs} --attribute rounds=4 --interleave-rounds --strict "
    "--show-stderr"
)


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
        _ = args


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
        _ = args


class CPerf(SubParserGenerator):
    name = "cperf"
    description = (
        "Run the on-demand CPerf suite of benchmarks (part of the UK Met "
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
    def func(args: argparse.Namespace):
        _ = args


class SPerf(CPerf):
    name = "sperf"
    description = CPerf.description.replace("CPerf", "SPerf")


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
        _ = args


def main():
    parser = ArgumentParser(
        description="Run the Iris performance benchmarks (using Airspeed Velocity).",
    )
    subparsers = parser.add_subparsers()

    for gen in (Overnight, Branch, CPerf, SPerf, Custom):
        _ = gen(subparsers).subparser

    _ = parser.parse_args()


if __name__ == "__main__":
    main()
