# (C) British Crown Copyright 2017, Met Office
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


def read_conda_reqs(fname, options):
    lines = []
    with open(fname, 'r') as fh:
        for line in fh:
            line = line.strip()
            if CONDA_PATTERN in line:
                line_start = line.index(CONDA_PATTERN) + len(CONDA_PATTERN)
                line = line[line_start:].strip()
                if 'only python=2' in line:
                    if 'python=2' in options:
                        line = line.replace('(only python=2)', '')
                        lines.append(line)
                    else:
                        continue
                else:
                    lines.append(line)
            else:
                lines.append(line)
    return lines


def compute_requirements(requirement_names=('core', ), options=None):
    conda_reqs_lines = []

    if 'python=2' in options:
        conda_reqs_lines.append('python=2.*')
    else:
        conda_reqs_lines.append('# Python 3 conda configuration')

    for req_name in requirement_names:
        fname = os.path.join(REQS_DIR, '{}.txt'.format(req_name))
        if not os.path.exists(fname):
            raise RuntimeError('Unable to find the requirements file for {} '
                               'in {}'.format(req_name, fname))
        conda_reqs_lines.extend(read_conda_reqs(fname, options))
        conda_reqs_lines.append('')

    return conda_reqs_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbosity", help="increase output verbosity")
    parser.add_argument(
            "--groups", nargs='*', default=[],
            help=("Gather requirements for these given named groups "
                  "(as found in the requirements/ folder)"))
    parser.add_argument(
            "--py2", action="store_true",
            help="Build the conda requirements for a python 2 installation")

    args = parser.parse_args()

    requirement_names = args.groups
    requirement_names.insert(0, 'core')
    requirement_names.insert(0, 'setup')

    options = []
    if args.py2:
        options.append('python=2')

    print('\n'.join(compute_requirements(requirement_names, options)))


if __name__ == '__main__':
    main()
