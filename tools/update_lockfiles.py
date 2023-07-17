# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
A command line utility for generating conda-lock files for the environments
that nox uses for testing each different supported version of python.
Typical usage:

    python tools/update_lockfiles.py -o requirements/locks requirements/py*.yml


"""

import argparse
from pathlib import Path
import subprocess
import sys
from warnings import warn


message = (
    "Iris' large requirements may require Mamba to successfully solve. If you "
    "don't want to install Mamba, consider using the workflow_dispatch on "
    "Iris' GitHub action."
)
warn(message)


try:
    import conda_lock
except:
    print("conda-lock must be installed.")
    exit(1)

parser = argparse.ArgumentParser(
    "Iris Lockfile Generator",
)

parser.add_argument('files', nargs='+',
    help="List of environment.yml files to lock")
parser.add_argument('--output-dir', '-o', default='.',
    help="Directory to save output lock files")

args = parser.parse_args()

for infile in args.files:
    print(f"generating lockfile for {infile}", file=sys.stderr)

    fname = Path(infile).name
    ftype = fname.split('.')[-1]
    if ftype.lower() in ('yaml', 'yml'):
        fname = '.'.join(fname.split('.')[:-1])

    # conda-lock --filename-template expects a string with a "...{platform}..."
    # placeholder in it, so we have to build the .lock filename without
    # using .format
    ofile_template = Path(args.output_dir) / (fname+'-{platform}.lock')
    subprocess.call([
        'conda-lock',
        'lock',
        '--filename-template', ofile_template,
        '--file', infile,
        '-k', 'explicit',
        '--platform', 'linux-64'
    ])
    print(f"lockfile saved to {ofile_template}".format(platform='linux-64'),
        file=sys.stderr)
