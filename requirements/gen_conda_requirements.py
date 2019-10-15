# (C) British Crown Copyright 2017 - 2019, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import os.path


REQS_DIR = os.path.dirname(__file__)
CONDA_PATTERN = '#conda:'


def read_conda_reqs(fname):
    lines = []
    with open(fname, 'r') as fh:
        for line in fh:
            line = line.strip()
            if CONDA_PATTERN in line:
                line_start = line.index(CONDA_PATTERN) + len(CONDA_PATTERN)
                line = line[line_start:].strip()
            lines.append(line)
    return lines


def compute_requirements(requirement_names=('core', )):
    conda_reqs_lines = []

    for req_name in requirement_names:
        fname = os.path.join(REQS_DIR, '{}.txt'.format(req_name))
        if not os.path.exists(fname):
            raise RuntimeError('Unable to find the requirements file for {} '
                               'in {}'.format(req_name, fname))
        conda_reqs_lines.extend(read_conda_reqs(fname))
        conda_reqs_lines.append('')

    return conda_reqs_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbosity", help="increase output verbosity")
    parser.add_argument(
            "--groups", nargs='*', default=[],
            help=("Gather requirements for these given named groups "
                  "(as found in the requirements/ folder)"))

    args = parser.parse_args()

    requirement_names = args.groups
    requirement_names.insert(0, 'core')
    requirement_names.insert(0, 'setup')

    print('\n'.join(compute_requirements(requirement_names)))


if __name__ == '__main__':
    main()
