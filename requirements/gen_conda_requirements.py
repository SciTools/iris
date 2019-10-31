# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

import argparse
import os.path


REQS_DIR = os.path.dirname(__file__)
CONDA_PATTERN = "#conda:"


def read_conda_reqs(fname):
    lines = []
    with open(fname, "r") as fh:
        for line in fh:
            line = line.strip()
            if CONDA_PATTERN in line:
                line_start = line.index(CONDA_PATTERN) + len(CONDA_PATTERN)
                line = line[line_start:].strip()
            lines.append(line)
    return lines


def compute_requirements(requirement_names=("core",)):
    conda_reqs_lines = []

    for req_name in requirement_names:
        fname = os.path.join(REQS_DIR, "{}.txt".format(req_name))
        if not os.path.exists(fname):
            raise RuntimeError(
                "Unable to find the requirements file for {} "
                "in {}".format(req_name, fname)
            )
        conda_reqs_lines.extend(read_conda_reqs(fname))
        conda_reqs_lines.append("")

    return conda_reqs_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbosity", help="increase output verbosity")
    parser.add_argument(
        "--groups",
        nargs="*",
        default=[],
        help=(
            "Gather requirements for these given named groups "
            "(as found in the requirements/ folder)"
        ),
    )

    args = parser.parse_args()

    requirement_names = args.groups
    requirement_names.insert(0, "core")
    requirement_names.insert(0, "setup")

    print("\n".join(compute_requirements(requirement_names)))


if __name__ == "__main__":
    main()
