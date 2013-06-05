# (C) British Crown Copyright 2010 - 2013, Met Office
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
Generate the rst files for the examples
"""
import os
import re
import sys


def out_of_date(original, derived):
    """
    Returns True if derivative is out-of-date wrt original,
    both of which are full file paths.

    TODO: this check isn't adequate in some cases.  Eg, if we discover
    a bug when building the examples, the original and derived will be
    unchanged but we still want to force a rebuild.
    """
    return (not os.path.exists(derived) or
            os.stat(derived).st_mtime < os.stat(original).st_mtime)


docstring_regex = re.compile(r'[\'\"]{3}(.*?)[\'\"]{3}', re.DOTALL)


noplot_regex = re.compile(r"#\s*-\*-\s*noplot\s*-\*-")


def generate_example_rst(app):
    # example code can be found at the same level as the documentation src folder
    rootdir = os.path.join(os.path.dirname(app.builder.srcdir), 'example_code')

    # examples are build as a subfolder of the src folder
    exampledir = os.path.join(app.builder.srcdir, 'examples')

    if not os.path.exists(exampledir):
        os.makedirs(exampledir)

    datad = {}
    for root, subFolders, files in os.walk(rootdir):
        for fname in files:
            if ( fname.startswith('.') or fname.startswith('#') or fname.startswith('_') or
                 fname.find('.svn')>=0 or not fname.endswith('.py') ):
                continue

            fullpath = os.path.join(root,fname)
            contents = file(fullpath).read()
            # indent
            relpath = os.path.split(root)[-1]
            datad.setdefault(relpath, []).append((fullpath, fname, contents))

    subdirs = datad.keys()
    subdirs.sort()

    fhindex = file(os.path.join(exampledir, 'index.rst'), 'w')
    fhindex.write("""\
Iris examples
=============

.. toctree::
    :maxdepth: 2

""")

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
        fhsubdirIndex = file(subdirIndexFile, 'w')
        fhindex.write('    %s/index.rst\n\n' % subdir)

        subdir_root_path = os.path.join(rootdir, subdir)

        # use the __init__.py file's docstring for the subdir example page (if __init__ exists)
        if os.path.exists(os.path.join(subdir_root_path, '__init__.py')):
            import imp
            mod = imp.load_source(subdir, os.path.join(subdir_root_path, '__init__.py'))
            fhsubdirIndex.writelines(mod.__doc__)
        else:
            fhsubdirIndex.writelines(['Examples in %s\n' % subdir, '='*50])

        # append the code to produce the toctree
        fhsubdirIndex.write("""\

.. toctree::
    :maxdepth: 1

""")

        sys.stdout.write(subdir + ", ")
        sys.stdout.flush()

        data = datad[subdir]
        data.sort()

        for fullpath, fname, contents in data:
            basename, ext = os.path.splitext(fname)
            outputfile = os.path.join(outputdir, fname)
            #thumbfile = os.path.join(thumb_dir, '%s.png'%basename)
            #print '    static_dir=%s, basename=%s, fullpath=%s, fname=%s, thumb_dir=%s, thumbfile=%s'%(static_dir, basename, fullpath, fname, thumb_dir, thumbfile)

            rstfile = '%s.rst'%basename
            outrstfile = os.path.join(rstdir, rstfile)

            fhsubdirIndex.write('    %s\n'%rstfile)

            if not out_of_date(fullpath, outrstfile):
                continue

            fh = file(outrstfile, 'w')
            fh.write('.. _%s-%s:\n\n'%(subdir, basename))

            docstring_results = docstring_regex.search(contents)
            if docstring_results is not None:
                fh.write( docstring_results.group(1) )
            else:
                title = '%s example code: %s'%(subdir, fname)
                #title = '<img src=%s> %s example code: %s'%(thumbfile, subdir, fname)
                fh.write(title + '\n')
                fh.write('='*len(title) + '\n\n')

            if not noplot_regex.search(contents):
                fh.write("\n\n.. plot:: %s\n\n::\n\n" % fullpath)
            else:
                fh.write("[`source code <%s>`_]\n\n::\n\n" % fname)
                # write the py file contents (we didnt need to do this for plot as the plot directive does this for us.
                fhstatic = file(outputfile, 'w')
                fhstatic.write(contents)
                fhstatic.close()

            # indent the contents
            contents = '\n'.join(['    %s'%row.rstrip() for row in contents.split('\n')])
            fh.write(contents)
            fh.write('\n\n\n\n\n')

            fh.close()

        fhsubdirIndex.close()

    fhindex.close()

def setup(app):
    app.connect('builder-inited', generate_example_rst)
