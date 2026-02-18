.. _s3_io:

Loading From and Saving To S3 Buckets
=====================================

For cloud computing, it is natural to want to access data storage based on URIs.
At the present time, by far the most widely used platform for this is
`Amazon S3 "buckets" <https://aws.amazon.com/s3/>`_.

It is common to treat an S3 bucket like a "disk", storing files as individual S3
objects.  S3 access urls can also contain a nested
`'prefix string' <https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-prefixes.html>`_
structure, which naturally mirrors sub-directories in a file-system.

While it would be possible for Iris to support S3 access directly, as it does the
"OpenDAP" protocol for netCDF data, this approach has some serious limitations : most
notably, each supported file format would have to be separately extended to support S3
urls in the place of file paths for loading and saving.

Instead, we have found that it is most practical to perform this access using a virtual
file system approach.  However, one drawback is that this is best controlled *outside*
the Python code -- see details below.


TL;DR
-----
Install s3-fuse and use its ``s3fs`` command, to create a file-system mount which maps
to an S3 bucket.  S3 objects can then be accessed as a regular files (read and write).


Fsspec, S3-fs, fuse and s3-fuse
--------------------------------
This approach depends on a set of related code solutions, as follows:

`fsspec <https://github.com/fsspec/filesystem_spec/blob/master/README.md>`_
is a  general framework for implementing Python-file-like access to alternative storage
resources.

`s3fs <https://github.com/fsspec/s3fs>`_
is a package based on fsspec, which enables Python to "open" S3 data objects as Python
file-like objects for reading and writing.

`fuse <https://github.com/libfuse/libfuse>`_
is an interface library that enables a data resource to be "mounted" as a Linux
filesystem, with user (not root) privilege.

`s3-fuse <https://github.com/s3fs-fuse/s3fs-fuse/blob/master/README.md>`_
is a utility based on s3fs and fuse, which provides a POSIX-compatible  "mount" so that
an S3 bucket can be accessed as a regular Unix file system.


Practical usage
---------------
Of the above, the only thing you actually need to know about is **s3-fuse**.

There is an initial one-time setup, and also actions to take in advance of launching
Python, and after exit, each time you want to access S3 from Python.

Prior requirements
^^^^^^^^^^^^^^^^^^

Install "s3-fuse"
~~~~~~~~~~~~~~~~~
The official
`installation instructions <https://github.com/s3fs-fuse/s3fs-fuse/blob/master/README.md#installation>`_
assume that you will perform a system installation with `apt`, `yum` or similar.

However, since you may well not have adequate 'sudo' or root access permissions
for this, it is simpler to instead install it only into your Python environment.
Though not suggested, this appears to work on Unix systems where we have tried it.

So, you can use conda or pip -- e.g.

.. code-block:: bash

    $ pip install s3-fuse

or

.. code-block:: bash

    $ conda install s3-fuse

( Or better, put it into a reusable 'spec file', with all other requirements, and then
use ``$ conda create --file ...``
).

Create an empty mount directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You need an empty directory in your existing filesystem tree, that you will map your
S3 bucket **onto** -- e.g.

.. code-block:: bash

    $ mkdir /home/self.me/s3_root/testbucket_mountpoint

The file system which this belongs to is presumably irrelevant, and will not affect
performance.

Setup AWS credentials
~~~~~~~~~~~~~~~~~~~~~
Provide S3 access credentials in an AWS credentials file, as described in
`this account <https://github.com/s3fs-fuse/s3fs-fuse/blob/master/README.md#examples>`_.


Before use (before each Python invocation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Activate your Python environment, which then gives access to the s3-fuse Linux
command (note: somewhat confusingly, this is called "s3fs").

Map your S3 bucket "into" the chosen empty directory -- e.g.

.. code-block:: bash

    $ s3fs my-test-bucket /home/self.me/s3_root/testbucket_mountpoint``

.. note::

    You can now freely list/access contents of your bucket at this path
    -- including updating or writing files.

.. note::

    This performs a Unix file-system "mount" operation, which temporarily
    modifies your system.  This change is not part of the current environment, and is not
    limited to the scope of the current process.

    If you reboot, the mount will disappear.  If you logout and login again, there can
    be problems : ideally you should avoid this by always "unmounting" (see below).


Within Python code
^^^^^^^^^^^^^^^^^^
Access files stored as S3 objects "under" the S3 url, appearing as files under the
mapped file-system path -- e.g.

.. code-block:: python

    >>> path = "/home/self.me/s3_root/testbucket_mountpoint/sub_dir/a_file.nc"
    >>> cubes = iris.load(path)


After use (after Python exit)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
At some point, you should "forget" the mounted S3 filesystem by **unmounting** it -- e.g.

.. code-block:: bash

    $ umount /home/self.me/s3_root/testbucket_mountpoint

.. note::

    The "umount" is a standard Unix command.  It may not always succeed, in which case
    some kind of retry may be needed -- see detail notes below.

    The mount created will not survive a system reboot, nor does it function correctly
    if the user logs out + logs in again.

    Presumably, problems could occur if repeated operation were to create a very large
    number of mounts, so unmounting after use does seem advisable.


Some Pros and Cons of this approach
-----------------------------------

PROs
^^^^

*   s3fs supports random access to "parts" of a file, allowing efficient handling of
    datasets larger than memory without requiring the data to be explicitly sharded
    in storage.

*   s3-fuse is transparent to file access within Python, including Iris load+save or
    other files accessed via a Python 'open' : the S3 data appears to be files in a
    regular file-system.

*   the file-system virtualisation approach works for all file formats, since the
    mapping occurs in the O.S. rather than in Iris, or Python.

*   "mounting" avoids the need for the Python code to dynamically connect to /
    disconnect from an S3 bucket

*   the "unmount problem" (see below) is managed at the level of the operating system,
    where it occurs, instead of trying to allow for it in Python code.  This means it
    could be managed differently in different operating systems, if needed.

CONs
^^^^

*   this solution is specific to S3 storage

*   possibly the virtualisation is not perfect :  some file-system operations might not
    behave as expected, e.g. with regard to file permissions or system information

*   it requires user actions *outside* the Python code

*   the user must manage the mount/umount context


Background Notes and Details
----------------------------

*   The file-like objects provided by **fsspec** replicate nearly *all* the behaviours
    of a regular Python file.

    However, this is still hard to integrate with regular file access, since you
    cannot create one from a regular Python "open" call -- still less
    when opening a file with an underlying file-format such as netCDF4 or HDF5
    (since these are usually implemented in other languages such as C).

    So, the key benefit offered by **s3-fuse** is that all the functions are mapped
    onto regular O.S. file-system calls -- so the file-format never needs to
    know that the data is not a "real" file.

*   It would be possible, instead, to copy data into an *actual* file on disk, but the
    s3-fuse approach avoids the need for copying, and thus in a cloud environment also
    the cost and maintenance of a "local disk".

    s3fs also allows the software to access only *required* parts of a file, without
    copying the whole content.  This is obviously essential for efficient use of large
    datasets, e.g. when larger than available memory.

*   It is also possible to use "s3-fuse" to establish the mounts *from within Python*.
    However, we have considered integrating this into Iris and rejected it because of
    unavoidable problems : namely, the "umount problem" (see below).
    For details, see : https://github.com/SciTools/iris/pull/6731

*   "Unmounting" must be done via a shell ``umount`` command, and there is no easy way to
    guarantee that this succeeds, since it can often get a "target is busy" error, which
    can only be resolved by delay + retry.
    This "umount problem" is a known problem in Unix generally : see
    `here <https://stackoverflow.com/questions/tagged/linux%20umount>`_
