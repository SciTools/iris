Custom Builds/Installs of Iris
==============================

Optional library: libmo_unpack

This library is property of the MetOffice and licensed separately.
It is used for decoding/unpacking PP files or Fields files that use an lbpack of 1 or 4.

Use of this library is not enabled by default and Iris can be installed using the following command:

  python setup.py install

If this library is available it's use can be enabled by installing Iris with the following command:

  python setup.py --with-unpack install

Note that if this library and/or its associated header files are installed in a custom location
then additional compiler arguments can be passed in to ensure that the Python extension
module linking against it builds correctly:

  python setup.py --with-unpack build_ext -I <custom include dir> -L <custom link-time libdir> -R <custom runtime libdir> install


