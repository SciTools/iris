# (C) British Crown Copyright 2010 - 2014, Met Office
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


'''
Generate the rst files for the examples
'''

from __future__ import (absolute_import, division, print_function)

import os
import re
import shutil
import sys


def out_of_date(original, derived):
    '''
    Returns True if derivative is out-of-date wrt original,
    both of which are full file paths.

    TODO: this check isn't adequate in some cases, e.g., if we discover
    a bug when building the examples, the original and derived will be
    unchanged but we still want to force a rebuild.
    '''
    return (not os.path.exists(derived) or
            os.stat(derived).st_mtime < os.stat(original).st_mtime)


docstring_regex = re.compile(r'[\'\"]{3}(.*?)[\'\"]{3}', re.DOTALL)


noplot_regex = re.compile(r'#\s*-\*-\s*noplot\s*-\*-')


def generate_example_rst(app):
    # Example code can be found at the same level as the documentation
    # src folder.
    rootdir = os.path.join(os.path.dirname(app.builder.srcdir), 'example_code')

    # Examples are built as a subfolder of the src folder.
    exampledir = os.path.join(app.builder.srcdir, 'examples')

    if not os.path.exists(exampledir):
        os.makedirs(exampledir)

    datad = {}
    for root, subFolders, files in os.walk(rootdir):
        for fname in files:
            if (fname.startswith('.') or fname.startswith('#') or
                    fname.startswith('_') or fname.find('.svn') >= 0 or
                    not fname.endswith('.py')):
                continue

            fullpath = os.path.join(root, fname)
            with open(fullpath) as fh:
                contents = fh.read()
            # indent
            relpath = os.path.split(root)[-1]
            datad.setdefault(relpath, []).append((fullpath, fname, contents))

    subdirs = sorted(datad.keys())

    fhindex = open(os.path.join(exampledir, 'index.rst'), 'w')
    fhindex.write('''\
Iris examples
=============

.. toctree::
    :maxdepth: 2

''')

    for subdir in subdirs:
        rstdir = os.path.join(exampledir, subdir)
        if not os.path.exists(rstdir):
            os.makedirs(rstdir)

        outputdir = os.path.join(app.builder.outdir, 'examples')
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        outputdir = os.path.join(outputdir, subdir)
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        subdirIndexFile = os.path.join(rstdir, 'index.rst')
        fhsubdirIndex = open(subdirIndexFile, 'w')
        fhindex.write('    {}/index.rst\n'.format(subdir))

        subdir_root_path = os.path.join(rootdir, subdir)

        # Use the __init__.py file's docstring for the subdir example page (if
        # __init__ exists).
        if os.path.exists(os.path.join(subdir_root_path, '__init__.py')):
            import imp
            mod = imp.load_source(
                subdir,
                os.path.join(subdir_root_path, '__init__.py'))
            fhsubdirIndex.writelines(mod.__doc__)
        else:
            line = 'Examples in {}\n'.format(subdir)
            fhsubdirIndex.writelines([line, '=' * len(line)])

        # Append the code to produce the toctree.
        fhsubdirIndex.write('''
.. toctree::
    :maxdepth: 1

''')

        sys.stdout.write(subdir + ', ')
        sys.stdout.flush()

        data = sorted(datad[subdir])

        for fullpath, fname, contents in data:
            basename, ext = os.path.splitext(fname)
            outputfile = os.path.join(outputdir, fname)

            rstfile = '{}.rst'.format(basename)
            outrstfile = os.path.join(rstdir, rstfile)

            fhsubdirIndex.write('    {}\n'.format(rstfile))

            if not out_of_date(fullpath, outrstfile):
                continue

            # Copy the example code to be in the src examples directory. This
            # means we can define a simple relative path in the plot directive,
            # which can also copy the file into the resulting build directory.
            shutil.copy(fullpath, rstdir)

            fh = open(outrstfile, 'w')
            fh.write('.. _{}-{}:\n\n'.format(subdir, basename))

            docstring_results = docstring_regex.search(contents)
            if docstring_results is not None:
                fh.write(docstring_results.group(1))
            else:
                title = '{} example code: {}'.format(subdir, fname)
                fh.write(title + '\n')
                fh.write('=' * len(title) + '\n\n')

            if not noplot_regex.search(contents):
                rel_example = os.path.relpath(outputfile, app.builder.outdir)
                fh.write('\n\n.. plot:: {}\n'.format(rel_example))
                fh.write('    :include-source:\n\n')
            else:
                fh.write('[`source code <{}>`_]\n\n'.format(fname))
                fh.write('.. literalinclude:: {}\n\n'.format(fname))

                # Write the .py file contents (we didn't need to do this for
                # plots as the plot directive does this for us.)
                with open(outputfile, 'w') as fhstatic:
                    fhstatic.write(contents)

            fh.close()

        fhsubdirIndex.close()

    fhindex.close()


def setup(app):
    app.connect('builder-inited', generate_example_rst)
