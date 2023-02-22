from abc import ABC
import argparse
from argparse import ArgumentParser
from typing import List, Tuple

# Common ASV arguments for all run_types except `custom`.
ASV_HARNESS = (
    "asv run {posargs} --attribute rounds=4 --interleave-rounds --strict "
    "--show-stderr"
)

parser = ArgumentParser(
    description="Run the Iris performance benchmarks (using Airspeed Velocity).",
)

# TODO: help.
subparsers = parser.add_subparsers(help="")


class SubParserSpec:
    """Used to make it easier to hold all the necessary info in 1 place."""

    name: str = NotImplemented
    description: str = NotImplemented
    argument_args: List[Tuple[tuple, dict]] = NotImplemented

    @staticmethod
    def func(args: argparse.Namespace):
        _ = args
        return NotImplemented

    @classmethod
    def get_subparser(cls):
        sb = subparsers.add_parser(
            cls.name,
            description=cls.description,
            formatter_class=argparse.RawTextHelpFormatter,
        )
        for args, kwargs in cls.argument_args:
            sb.add_argument(*args, **kwargs)
        sb.set_defaults(func=cls.func)
        return sb


class OvernightSpec(SubParserSpec):
    name = "overnight"
    description = (
        "Benchmarks all commits between the input **first_commit** to ``HEAD``, "
        "comparing each to its parent for performance shifts. If a commit causes "
        "shifts, the output is saved to a file:\n"
        "``.asv/performance-shifts/<commit-sha>``\n\n"
        "Designed for checking the previous 24 hours' commits, typically in a "
        "scheduled script."
    )
    argument_args = [
        (
            ("first_commit",),
            dict(
                type=str,
                help="The first commit in the benchmarking commit sequence.",
            ),
        )
    ]

    @staticmethod
    def func(args: argparse.Namespace):
        _ = args


class BranchSpec(SubParserSpec):
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
    argument_args = [
        (
            ("base_branch",),
            dict(
                type=str,
                help="A branch that has the merge-base with ``HEAD`` - ``HEAD`` will be benchmarked against that merge-base.",
            ),
        )
    ]

    @staticmethod
    def func(args: argparse.Namespace):
        _ = args


# ################################
# # overnight
#
# overnight = subparsers.add_parser("overnight")
# overnight.description = (
#     "Benchmarks all commits between the input **first_commit** to ``HEAD``, "
#     "comparing each to its parent for performance shifts. If a commit causes "
#     "shifts, the output is saved to a file:\n"
#     "``.asv/performance-shifts/<commit-sha>``\n\n"
#     "Designed for checking the previous 24 hours' commits, typically in a "
#     "scheduled script."
# )
# overnight.add_argument(
#     "first_commit",
#     type=str,
#     help="The first commit in the benchmarking commit sequence.",
# )
#
#
# # def func_overnight(args: argparse.Namespace):
# #     pass
# #
# #
# # overnight.set_defaults(func=func_overnight)
#
#
# ################################
# # branch
#
# branch = subparsers.add_parser("branch")
# branch.description = (
#     "Performs the same operations as ``overnight``, but always on two commits "
#     "only - ``HEAD``, and ``HEAD``'s merge-base with the input "
#     "**base_branch**. Output from this run is never saved to a file. Designed "
#     "for testing if the active branch's changes cause performance shifts - "
#     "anticipating what would be caught by ``overnight`` once merged.\n\n"
#     "**For maximum accuracy, avoid using the machine that is running this "
#     "session. Run time could be >1 hour for the full benchmark suite.**"
# )
# branch.add_argument(
#     "base_branch",
#     type=str,
#     help="A branch that has the merge-base with ``HEAD`` - ``HEAD`` will be benchmarked against that merge-base.",
# )
#
# cperf = subparsers.add_parser("cperf")
# cperf.description = (
#     "Run the on-demand CPerf suite of benchmarks (part of the UK Met Office "
#     "NG-VAT project) for the ``HEAD`` of ``upstream/main`` only, and publish "
#     "the results to the input **publish_dir**, within a unique "
#     "subdirectory for this run."
# )
#
# sperf = subparsers.add_parser("sperf")
# sperf.description = cperf.description.replace("CPerf", "SPerf")
#
# for subparser in [cperf, sperf]:
#     subparser.add_argument(
#         "publish_dir",
#         type=str,
#         help="HTML results will be published to a sub-dir in this dir.",
#     )
#
# custom = subparsers.add_parser("custom")
# custom.description = (
#     "Run the on-demand CPerf suite of benchmarks (part of the UK Met Office "
#     "NG-VAT project) for the ``HEAD`` of ``upstream/main`` only, and publish "
#     "the results to the input **publish_dir**, within a unique "
#     "subdirectory for this run."
# )
# custom.add_argument(
#     "asv_sub_command",
#     type=str,
#     help="The ASV command to run.",
# )

overnight = OvernightSpec.get_subparser()
branch = BranchSpec.get_subparser()

args = parser.parse_args()
exit()
