.. include:: common_links.inc
.. _iris_docs:


Iris
====

**A powerful, format-agnostic, community-driven Python package for analysing
and visualising Earth science data.**

Iris implements a data model based on the `CF conventions <http://cfconventions.org>`_
giving you a powerful, format-agnostic interface for working with your data.
It excels when working with multi-dimensional Earth Science data, where tabular
representations become unwieldy and inefficient.

For more information see :ref:`why_iris`.

..
    COMMENT: Image alignment: https://getbootstrap.com/docs/4.0/utilities/sizing/

.. todo::
    The below example grid-item-card should allow for a :class-img-top:
    but it will not build.  Waiting for a fix ideally from sphinx-design.

    :class-img-top: w-50 m-auto px-1 py-2


.. grid:: 3

    .. grid-item-card::
        :text-align: center
        :img-top: _static/icon_shuttle.svg
        :shadow: lg

        This above image should be nicely scaled and not too big.

        +++
        .. button-ref:: getting_started_index
            :ref-type: ref
            :class: btn-outline-info btn-block

                TEXT BUTTON


.. todo:: These cards work ok but they do not align the footer.


.. grid:: 3

    .. grid-item::

        .. card::
            :text-align: center
            :img-top: _static/icon_shuttle.svg
            :class-img-top: w-50 m-auto px-1 py-2
            :shadow: lg

            Information on Iris, how to install and a gallery of examples that
            create plots.

            +++
            .. button-ref:: getting_started_index
                :ref-type: ref
                :class: btn-outline-info btn-block

                    Getting Started

    .. grid-item::

        .. card::
            :text-align: center
            :img-top: _static/icon_instructions.svg
            :class-img-top: w-50 m-auto px-1 py-2
            :shadow: lg

            Learn how to use Iris, including loading, navigating, saving,
            plotting and more.

            +++
            .. button-ref:: user_guide_index
                :ref-type: ref
                :class: btn-outline-info btn-block

                    User Guide

    .. grid-item::

        .. card::
            :text-align: center
            :img-top: _static/icon_development.svg
            :class-img-top: w-50 m-auto px-1 py-2
            :shadow: lg

            Information on how you can contribute to Iris as a developer.

            +++
            .. button-ref:: development_where_to_start
                :ref-type: ref
                :class: btn-outline-info btn-block

                    Developers Guide


.. grid:: 3

    .. grid-item::

        .. card::
            :text-align: center
            :img-top: _static/icon_api.svg
            :class-img-top: w-50 m-auto px-1 py-2
            :shadow: lg

            Browse full Iris functionality by module.

            +++
            .. button-ref:: generated/api/iris
                :ref-type: doc
                :class: btn-outline-info btn-block

                    Iris API

    .. grid-item::

        .. card::
            :text-align: center
            :img-top: _static/icon_new_product.svg
            :class-img-top: w-50 m-auto px-1 py-2
            :shadow: lg

            Find out what has recently changed in Iris.

            +++
            .. button-ref:: iris_whatsnew
                :ref-type: ref
                :class: btn-outline-info btn-block

                    What's New

    .. grid-item::

        .. card::
            :text-align: center
            :img-top: _static/icon_thumb.png
            :class-img-top: w-50 m-auto px-1 py-2
            :shadow: lg

            Raise the profile of issues by voting on them.

            +++
            .. button-ref:: voted_issues_top
                :ref-type: ref
                :class: btn-outline-info btn-block

                    Voted Issues


Icons made by `FreePik <https://www.freepik.com>`_ from
`Flaticon <https://www.flaticon.com/>`_


.. _iris_support:

Support
~~~~~~~

We, the Iris developers have adopted `GitHub Discussions`_ to capture any
discussions or support questions related to Iris.

See also `StackOverflow for "How Do I? <https://stackoverflow.com/questions/tagged/python-iris>`_
that may be useful but we do not actively monitor this.

The legacy support resources:

* `Users Google Group <https://groups.google.com/forum/#!forum/scitools-iris>`_
* `Developers Google Group <https://groups.google.com/forum/#!forum/scitools-iris-dev>`_
* `Legacy Documentation`_ (Iris 2.4 or earlier).  This is an archive of zip
  files of past documentation.  You can download, unzip and view the
  documentation locally (index.html).  There may be some incorrect rendering
  and older javascvript (.js) files may show a warning when uncompressing, in
  which case we suggest you use a different unzip tool.


.. toctree::
   :caption: Getting Started
   :maxdepth: 1
   :hidden:

   getting_started


.. toctree::
   :caption: User Guide
   :maxdepth: 1
   :name: userguide_index
   :hidden:

   userguide/index


.. toctree::
   :caption: Developers Guide
   :maxdepth: 1
   :name: developers_index
   :hidden:

   developers_guide/contributing_getting_involved


.. toctree::
   :caption: Iris API
   :maxdepth: 1
   :hidden:

   generated/api/iris


.. toctree::
   :caption: What's New in Iris
   :maxdepth: 1
   :name: whats_new_index
   :hidden:

   whatsnew/index

.. todolist::