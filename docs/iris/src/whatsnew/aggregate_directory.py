# (C) British Crown Copyright 2015, Met Office
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
"""
Build a release file from files in a contributions directory.

Looks for directories "<...whatsnew>/contributions_<xx.xx>".
Takes specified "xx.xx" as version, or latest found (alphabetic).
Writes a file "<...whatsnew>/<xx.xx>.rst".

Valid contributions filenames are of the form:
    <category>_<date>_summary.txt
Where <summary> can be any valid chars, and
<category> is one of :
   "newfeature" "bugfix" "incompatiblechange" "deprecate" "docchange", and
<date> is in the style "2001-Jan-23".

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import datetime
from glob import glob
import os
import re
import argparse
import warnings
from operator import itemgetter
from distutils import version

# Regular expressions: CONTRIBUTION_REGEX matches the filenames of
# contribution snippets. It is split into three sections separated by _
# 0. String for the category. 1. ISO8601 date. 2. String for the feature name.
# RELEASE_REGEX matches the directory names, returning the release.
CONTRIBUTION_REGEX_STRING = r'(?P<category>.*)'
CONTRIBUTION_REGEX_STRING += r'_(?P<isodate>\d{4}-\w{3}-\d{2})'
CONTRIBUTION_REGEX_STRING += r'_(?P<summary>.*)\.txt$'
CONTRIBUTION_REGEX = re.compile(CONTRIBUTION_REGEX_STRING)
RELEASEDIR_PREFIX = r'contributions_'
_RELEASEDIR_REGEX_STRING = RELEASEDIR_PREFIX + r'(?P<release>.*)$'
RELEASE_REGEX = re.compile(_RELEASEDIR_REGEX_STRING)
SOFTWARE_NAME = 'Iris'
EXTENSION = '.rst'
VALID_CATEGORIES = [
    {'Prefix': 'newfeature', 'Title': 'Features'},
    {'Prefix': 'bugfix', 'Title': 'Bugs Fixed'},
    {'Prefix': 'incompatiblechange', 'Title': 'Incompatible Changes'},
    {'Prefix': 'deprecate', 'Title': 'Deprecations'},
    {'Prefix': 'docchange', 'Title': 'Documentation Changes'}
]
VALID_CATEGORY_PREFIXES = [cat['Prefix'] for cat in VALID_CATEGORIES]


def _self_root_directory():
    return os.path.abspath(os.path.dirname(__file__))


def _decode_contribution_filename(file_name):
    file_name_elements = CONTRIBUTION_REGEX.match(file_name)
    category = file_name_elements.group('category')
    if category not in VALID_CATEGORY_PREFIXES:
        # This is an error
        raise ValueError('Unknown category in contribution filename.')
    isodate = file_name_elements.group('isodate')
    date_of_item = datetime.datetime.strptime(isodate, '%Y-%b-%d').date()
    return category, isodate, date_of_item


def is_release_directory(directory_name, release):
    '''Returns True if a given directory name matches the requested release.'''
    result = False
    directory_elements = RELEASE_REGEX.match(directory_name)
    try:
        release_string = directory_elements.group('release')
        directory_release = version.StrictVersion(release_string)
    except (AttributeError, ValueError):
        pass
    else:
        if directory_release == release:
            result = True
    return result


def is_compiled_release(root_directory, release):
    '''Returns True if the requested release.rst file exists.'''
    result = False
    compiled_filename = '{!s}{}'.format(release, EXTENSION)
    compiled_filepath = os.path.join(root_directory, compiled_filename)
    if os.path.exists(compiled_filepath) and os.path.isfile(compiled_filepath):
        result = True
    return result


def get_latest_release(root_directory=None):
    """
    Implement default=latest release identification.

    Returns a valid release code.

    """
    if root_directory is None:
        root_directory = _self_root_directory()
    directory_contents = os.listdir(root_directory)
    # Default release to latest visible dir.
    possible_release_dirs = [releasedir_name
                             for releasedir_name in directory_contents
                             if RELEASE_REGEX.match(releasedir_name)]
    if len(possible_release_dirs) == 0:
        dirspec = os.path.join(root_directory, RELEASEDIR_PREFIX + '*')
        msg = 'No valid release directories found, i.e. {!r}.'
        raise ValueError(msg.format(dirspec))
    release_dirname = sorted(possible_release_dirs)[-1]
    release = RELEASE_REGEX.match(release_dirname).group('release')
    return release


def find_release_directory(root_directory, release=None,
                           fail_on_existing=True):
    '''
    Returns the matching contribution directory or raises an exception.

    Defaults to latest-found release (from release directory names).
    Optionally, fail if the matching release file already exists.
    *Always* fail if no release directory exists.

    '''
    if release is None:
        # Default to latest release.
        release = get_latest_release(root_directory)

    if fail_on_existing:
        compiled_release = is_compiled_release(root_directory, release)
        if compiled_release:
            msg = ('Specified release {!r} is already compiled : '
                   '{!r} already exists.')
            compiled_filename = str(release) + EXTENSION
            raise ValueError(msg.format(release, compiled_filename))

    directory_contents = os.listdir(root_directory)
    result = None
    for inode in directory_contents:
        node_path = os.path.join(root_directory, inode)
        if os.path.isdir(node_path):
            release_directory = is_release_directory(inode, release)
            if release_directory:
                result = os.path.join(root_directory, inode)
                break
    if not result:
        msg = 'Contribution folder for release {!s} does not exist : no {!r}.'
        release_dirname = RELEASEDIR_PREFIX + str(release) + '/'
        release_dirpath = os.path.join(root_directory, release_dirname)
        raise ValueError(msg.format(release, release_dirpath))
    return result


def generate_header(release):
    '''Return a list of text lines that make up a header for the document.'''
    header_text = []
    title_template = 'What\'s New in {} {!s}\n'
    title_line = title_template.format(SOFTWARE_NAME, release)
    title_underline = ('=' * (len(title_line) - 1)) + '\n'
    isodatestamp = datetime.date.today().strftime('%Y-%m-%d')
    header_text.append(title_line)
    header_text.append(title_underline)
    header_text.append('\n')
    header_text.append(':Release: {!s}\n'.format(release))
    header_text.append(':Date: {}\n'.format(isodatestamp))
    header_text.append('\n')
    description_template = 'This document explains the new/changed features '\
                           'of {} in version {!s}\n'
    header_text.append(description_template.format(SOFTWARE_NAME, release))
    header_text.append('(:doc:`View all changes <index>`.)')
    header_text.append('\n')
    return header_text


def read_directory(directory_path):
    '''Parse the items in a specified directory and return their metadata.'''
    directory_contents = os.listdir(directory_path)
    compilable_files_unsorted = []
    misnamed_files = []
    for file_name in directory_contents:
        try:
            category, isodate, date_of_item = \
                _decode_contribution_filename(file_name)
        except (AttributeError, ValueError):
            misnamed_files.append(file_name)
            continue
        compilable_files_unsorted.append({'Category': category,
                                          'Date': date_of_item,
                                          'FileName': file_name})
    compilable_files = sorted(compilable_files_unsorted,
                              key=itemgetter('Date'),
                              reverse=True)
    if misnamed_files:
        msg = 'Found contribution file(s) with unexpected names :'
        for filename in misnamed_files:
            full_path = os.path.join(directory_path, filename)
            msg += '\n  {}'.format(full_path)
        warnings.warn(msg, UserWarning)

    return compilable_files


def compile_directory(directory, release):
    '''Read in source files in date order and compile the text into a list.'''
    source_text = read_directory(directory)
    compiled_text = []
    header_text = generate_header(release)
    compiled_text.extend(header_text)
    for count, category in enumerate(VALID_CATEGORIES):
        category_text = []
        subtitle_line = ''
        if count == 0:
            subtitle_line += '{} {!s} '.format(SOFTWARE_NAME, release)
        subtitle_line += category['Title'] + '\n'
        subtitle_underline = ('=' * (len(subtitle_line) - 1)) + '\n'
        category_text.append('\n')
        category_text.append(subtitle_line)
        category_text.append(subtitle_underline)
        category_items = [item for item in source_text
                          if item['Category'] == category['Prefix']]
        if not category_items:
            continue
        for file_description in category_items:
            entry_path = os.path.join(directory, file_description['FileName'])
            with open(entry_path, 'r') as content_object:
                category_text.extend(content_object.readlines())
        compiled_text.extend(category_text)
    return compiled_text


def check_all_contributions_valid(release=None, quiet=False):
    """"Scan the contributions directory for badly-named files."""
    root_directory = _self_root_directory()
    # Check there are *some* contributions directory(s), else silently pass.
    contribs_spec = os.path.join(root_directory, RELEASEDIR_PREFIX + '*')
    if len(glob(contribs_spec)) > 0:
        # There are some contributions directories: check latest / specified.
        if release is None:
            release = get_latest_release()
        if not quiet:
            msg = 'Checking whatsnew contributions for release "{!s}".'
            print(msg.format(release))
        release_directory = find_release_directory(root_directory, release,
                                                   fail_on_existing=False)
        # Run the directory scan, but convert any warning into an error.
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            compile_directory(release_directory, release)
    if not quiet:
        print('done.')


def run_compilation(release=None, quiet=False):
    '''Write a draft release.rst file given a specified uncompiled release.'''
    if release is None:
        # This must exist !
        release = get_latest_release()
    if not quiet:
        msg = 'Building release document for release "{!s}".'
        print(msg.format(release))
    root_directory = _self_root_directory()
    release_directory = find_release_directory(root_directory, release)
    compiled_text = compile_directory(release_directory, release)
    compiled_filename = str(release) + EXTENSION
    compiled_filepath = os.path.join(root_directory, compiled_filename)
    with open(compiled_filepath, 'w') as output_object:
        for string_line in compiled_text:
            output_object.write(string_line)
    if not quiet:
        print('done.')


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("release", help="Release number to be compiled",
                        nargs='?', type=version.StrictVersion)
    PARSER.add_argument(
        '-c', '--checkonly', action='store_true',
        help="Check contribution file names, do not build.")
    PARSER.add_argument(
        '-q', '--quiet', action='store_true',
        help="Do not print progress messages.")
    ARGUMENTS = PARSER.parse_args()
    release = ARGUMENTS.release
    quiet = ARGUMENTS.quiet
    if ARGUMENTS.checkonly:
        check_all_contributions_valid(release, quiet=quiet)
    else:
        run_compilation(release, quiet=quiet)
