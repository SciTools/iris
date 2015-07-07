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

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import datetime
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
CONTRIBUTION_REGEX_STRING += r'_(?P<isodate>\d{4}-\d{2}-\d{2})'
CONTRIBUTION_REGEX_STRING += r'_(?P<summary>.*)\.txt$'
CONTRIBUTION_REGEX = re.compile(CONTRIBUTION_REGEX_STRING)
RELEASE_REGEX = re.compile(r'contributions_(?P<release>.*)$')
SOFTWARE_NAME = 'Iris'
EXTENSION = '.rst'
VALID_CATEGORIES = [
    {'Prefix': 'newfeature', 'Title': 'Features'},
    {'Prefix': 'bugfix', 'Title': 'Bugs Fixed'},
    {'Prefix': 'incompatiblechange', 'Title': 'Incompatible Changes'},
    {'Prefix': 'deprecate', 'Title': 'Deprecations'},
    {'Prefix': 'docchange', 'Title': 'Documentation Changes'}
]


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


def find_release_directory(root_directory, release):
    '''Returns the matching contribution directory or rasises an exception.'''
    result = None
    directory_contents = os.listdir(root_directory)
    compiled_release = is_compiled_release(root_directory, release)
    if compiled_release:
        raise OSError("Specified release is already compiled.")
    for inode in directory_contents:
        if os.path.isdir(inode):
            release_directory = is_release_directory(inode, release)
            if release_directory:
                result = os.path.join(root_directory, inode)
                break
    if not result:
        raise OSError("Contribution folder for this release does not exist.")
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
        file_name_elements = CONTRIBUTION_REGEX.match(file_name)
        try:
            category = file_name_elements.group('category')
            isodate = file_name_elements.group('isodate')
        except (AttributeError, ValueError):
            misnamed_files.append(file_name)
            continue
        else:
            date_of_item = datetime.datetime.strptime(isodate,
                                                      '%Y-%m-%d').date()
        compilable_files_unsorted.append({'Category': category,
                                          'Date': date_of_item,
                                          'FileName': file_name})
    compilable_files = sorted(compilable_files_unsorted,
                              key=itemgetter('Date'),
                              reverse=True)
    if misnamed_files:
        warning_text = 'Skipped files: {!s}'.format(misnamed_files)
        warnings.warn(warning_text, UserWarning)
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


def run_compilation(release):
    '''Write a draft release.rst file given a specified uncompiled release.'''
    root_directory = os.getcwd()
    release_directory = find_release_directory(root_directory, release)
    compiled_text = compile_directory(release_directory, release)
    compiled_filename = '{!s}{}'.format(release, EXTENSION)
    compiled_filepath = os.path.join(root_directory, compiled_filename)
    with open(compiled_filepath, 'w') as output_object:
        for string_line in compiled_text:
            output_object.write(string_line)

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("release", help="Release number to be compiled",
                        type=version.StrictVersion)
    ARGUMENTS = PARSER.parse_args()
    run_compilation(ARGUMENTS.release)
