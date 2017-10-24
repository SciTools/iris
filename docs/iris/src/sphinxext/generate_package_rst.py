# (C) British Crown Copyright 2010 - 2015, Met Office
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

import os
import sys
import re
import inspect


document_dict = {
    # Use autoclass for classes.
    'class': '''
{object_docstring}

..

    .. autoclass:: {object_name}
        :members:
        :undoc-members:
        :inherited-members:

''',
    'function': '''
.. autofunction:: {object_name}

''',
    # For everything else, let automodule do some magic...
    None: '''

.. autodata:: {object_name}

'''}


horizontal_sep = '''
.. raw:: html

    <p class="hr_p"><a href="#">&uarr;&#32&#32 top &#32&#32&uarr;</a></p>
    <!--

-----------

.. raw:: html

    -->

'''


def lookup_object_type(obj):
    if inspect.isclass(obj):
        return 'class'
    elif inspect.isfunction(obj):
        return 'function'
    else:
        return None


def auto_doc_module(file_path, import_name, root_package,
                    package_toc=None, title=None):
    doc = r'''.. _{import_name}:

{title_underline}
{title}
{title_underline}

{sidebar}

.. currentmodule:: {root_package}

.. automodule:: {import_name}

In this module:

{module_elements}


'''
    if package_toc:
        sidebar = '''
.. sidebar:: Modules in this package

{package_toc_tree}

    '''.format(package_toc_tree=package_toc)
    else:
        sidebar = ''

    try:
        mod = __import__(import_name)
    except ImportError as e:
        message = r'''.. error::

    This module could not be imported. Some dependencies are missing::

        ''' + str(e)
        return doc.format(title=title or import_name,
                          title_underline='=' * len(title or import_name),
                          import_name=import_name, root_package=root_package,
                          sidebar=sidebar, module_elements=message)

    mod = sys.modules[import_name]
    elems = dir(mod)

    if '__all__' in elems:
        document_these = [(attr_name, getattr(mod, attr_name))
                          for attr_name in mod.__all__]
    else:
        document_these = [(attr_name, getattr(mod, attr_name))
                          for attr_name in elems
                          if (not attr_name.startswith('_') and
                              not inspect.ismodule(getattr(mod, attr_name)))]

        def is_from_this_module(arg):
            name = arg[0]
            obj = arg[1]
            return (hasattr(obj, '__module__') and
                    obj.__module__ == mod.__name__)

        sort_order = {'class': 2, 'function': 1}

        # Sort them according to sort_order dict.
        def sort_key(arg):
            name = arg[0]
            obj = arg[1]
            return sort_order.get(lookup_object_type(obj), 0)

        document_these = filter(is_from_this_module, document_these)
        document_these = sorted(document_these, key=sort_key)

    lines = []
    for element, obj in document_these:
        object_name = import_name + '.' + element
        obj_content = document_dict[lookup_object_type(obj)].format(
            object_name=object_name,
            object_name_header_line='+' * len(object_name),
            object_docstring=inspect.getdoc(obj))
        lines.append(obj_content)

    lines = horizontal_sep.join(lines)

    module_elements = '\n'.join(' * :py:obj:`{}`'.format(element)
                                for element, obj in document_these)

    lines = doc + lines
    return lines.format(title=title or import_name,
                        title_underline='=' * len(title or import_name),
                        import_name=import_name, root_package=root_package,
                        sidebar=sidebar, module_elements=module_elements)


def auto_doc_package(file_path, import_name, root_package, sub_packages):
    max_depth = 1 if import_name == 'iris' else 2
    package_toc = '\n      '.join(sub_packages)
    package_toc = '''
   .. toctree::
      :maxdepth: {:d}
      :titlesonly:

      {}


'''.format(max_depth, package_toc)

    if '.' in import_name:
        title = None
    else:
        title = import_name.capitalize() + ' reference documentation'

    return auto_doc_module(file_path, import_name, root_package,
                           package_toc=package_toc, title=title)


def auto_package_build(app):
    root_package = app.config.autopackage_name
    if root_package is None:
        raise ValueError('set the autopackage_name variable in the '
                         'conf.py file')

    if not isinstance(root_package, list):
        raise ValueError('autopackage was expecting a list of packages to '
                         'document e.g. ["itertools"]')

    for package in root_package:
        do_package(package)


def do_package(package_name):
    out_dir = package_name + os.path.sep

    # Import the root package. If this fails then an import error will be
    # raised.
    module = __import__(package_name)
    root_package = package_name
    rootdir = os.path.dirname(module.__file__)

    package_folder = []
    module_folders = {}
    for root, subFolders, files in os.walk(rootdir):
        for fname in files:
            name, ext = os.path.splitext(fname)

            # Skip some non-relevant files.
            if (fname.startswith('.') or fname.startswith('#') or
                    re.search('^_[^_]', fname) or fname.find('.svn') >= 0 or
                    not (ext in ['.py', '.so'])):
                continue

            # Handle new shared library naming conventions
            if ext == '.so':
                name = name.split('.', 1)[0]

            rel_path = root_package + \
                os.path.join(root, fname).split(rootdir)[-1]
            mod_folder = root_package + \
                os.path.join(root).split(rootdir)[-1].replace('/', '.')

            # Only add this package to folder list if it contains an __init__
            # script.
            if name == '__init__':
                package_folder.append([mod_folder, rel_path])
            else:
                import_name = mod_folder + '.' + name
                mf_list = module_folders.setdefault(mod_folder, [])
                mf_list.append((import_name, rel_path))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for package, package_path in package_folder:
        if '._' in package or 'test' in package:
            continue

        paths = []
        for spackage, spackage_path in package_folder:
            # Ignore this packages, packages that are not children of this
            # one, test packages, private packages, and packages that are
            # subpackages of subpackages (they'll be part of the subpackage).
            if spackage == package:
                continue
            if not spackage.startswith(package):
                continue
            if spackage.count('.') > package.count('.') + 1:
                continue
            if 'test' in spackage:
                continue

            split_path = spackage.rsplit('.', 2)[-2:]
            if any(part[0] == '_' for part in split_path):
                continue

            paths.append(os.path.join(*split_path) + '.rst')

        paths.extend(os.path.join(os.path.basename(os.path.dirname(path)),
                                  os.path.basename(path).split('.', 1)[0])
                     for imp_name, path in module_folders.get(package, []))

        paths.sort()
        doc = auto_doc_package(package_path, package, root_package, paths)

        package_dir = out_dir + package.replace('.', os.path.sep)
        if not os.path.exists(package_dir):
            os.makedirs(out_dir + package.replace('.', os.path.sep))

        out_path = package_dir + '.rst'
        if not os.path.exists(out_path):
            print('Creating non-existent document {} ...'.format(out_path))
            with open(out_path, 'w') as fh:
                fh.write(doc)
        else:
            with open(out_path, 'r') as fh:
                existing_content = ''.join(fh.readlines())
            if doc != existing_content:
                print('Creating out of date document {} ...'.format(
                    out_path))
                with open(out_path, 'w') as fh:
                    fh.write(doc)

        for import_name, module_path in module_folders.get(package, []):
            doc = auto_doc_module(module_path, import_name, root_package)
            out_path = out_dir + import_name.replace('.', os.path.sep) + '.rst'
            if not os.path.exists(out_path):
                print('Creating non-existent document {} ...'.format(
                    out_path))
                with open(out_path, 'w') as fh:
                    fh.write(doc)
            else:
                with open(out_path, 'r') as fh:
                    existing_content = ''.join(fh.readlines())
                if doc != existing_content:
                    print('Creating out of date document {} ...'.format(
                        out_path))
                    with open(out_path, 'w') as fh:
                        fh.write(doc)


def setup(app):
    app.connect('builder-inited', auto_package_build)
    app.add_config_value('autopackage_name', None, 'env')
