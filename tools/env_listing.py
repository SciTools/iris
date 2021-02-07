#!/usr/bin/env python
# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Produce a listing output of the calling Python environment.

Combine "conda list" with a "pip list" to describe all installed packages.

"""
from subprocess import check_output
import beautifultable as bt


def sanitise_lines(lines, header_line_hint=None):
    """
    Split string by spaces, removing leading, trailing and repeated whitespace.
    Remove header lines up to the last one containing 'header_line_hint'.

    """
    pkg_info = []
    for line in lines:
        line = line.strip()
        while '\t' in line:
            line = line.replace('\t', ' ')
        while '  ' in line:
            ...
            line = line.replace('  ', ' ')
        line = line.split(' ')
        if len(line) > 0 and line[0] not in ('', '#'):
            pkg_info.append(line)

    pkg_keys = [pkg[0] for pkg in pkg_info]
    if header_line_hint is not None:
        header_inds = [ind for ind, key in enumerate(pkg_keys)
                       if header_line_hint in key]
        if len(header_inds) > 0:
            pkg_info = pkg_info[header_inds[-1] + 1:]

    result = {pkg[0]: pkg[1:] for pkg in pkg_info}
    return result


def scan_env():
    """
    Get package listings from conda and pip.

    Return:
        package_names, conda_info, pip_info

    The package names are sorted.
    'pip_info' is a dict : name --> version
    'conda_info' is a dict : name --> version, build, channel
    Any package may be present in either dict, or both.

    """
    conda_list_bytes = check_output(['conda list'], shell=True)
    conda_list_lines = conda_list_bytes.decode().split('\n')
    pip_list_bytes = check_output(['pip list'], shell=True)
    pip_list_lines = pip_list_bytes.decode().split('\n')

    conda_info = sanitise_lines(conda_list_lines)
    pip_info = sanitise_lines(pip_list_lines, header_line_hint='---')
    conda_keys = set(conda_info.keys())
    pip_keys = set([pkg[0] for pkg in pip_info])
    all_keys = sorted(set(conda_info.keys()) | set(pip_info.keys()))

    return all_keys, conda_info, pip_info


def make_package_table(package_names, conda_info, pip_info):
    """
    Turn the package info returned from 'scan_env' into a printable table.

    """
    table = bt.BeautifulTable()
    table.columns.header = [
        'Package', 'source', 'version(s)', 'conda-version', 'conda-channel']

    def version_summary(package, pip_info, conda_info):
        """
        Extract + combine info from pip and conda about 'package'.
        Return a list of 5 strings :  package, source, version, build, channel.

        """
        pipver = pip_info[package][0] if package in pip_info else None
        condaver = conda_info[package][0] if package in conda_info else None
        columns = [package] + [''] * 4
        if condaver:
            conda_extra_columns = conda_info[package][1:]
            assert len(conda_extra_columns) == 2
            columns[3:] = conda_extra_columns
        if pipver is not None and condaver is None:
            source, version = "pip", pipver
        elif pipver is None and condaver is not None:
            source, version = "conda", condaver
        else:
            if (pipver == condaver):
                source, version = "both", pipver
            else:
                source, version = '**CONFLICT**', f'pip={pipver}  conda={condaver}'
        if 'CONFLICT' not in source:
            version = '= ' + version
        columns[1:3] = [source, version]
        return columns

    for key in package_names:
        table.rows.append(version_summary(key,
                                          pip_info=pip_info,
                                          conda_info=conda_info))

    # Pre-style the table output.
    table.maxwidth = 9999
    table.columns.alignment = bt.ALIGN_LEFT
    table.set_style(bt.STYLE_COMPACT)
    return table  # Ready to print


if __name__ == '__main__':
    env_info = scan_env()
    table = make_package_table(*env_info)
    print(str(table))
