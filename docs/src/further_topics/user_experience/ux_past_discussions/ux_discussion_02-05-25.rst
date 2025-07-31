Iris UX Discussion 02-05-25
###########################

Community
*********

Low User Interaction
=====================

* Iris has always been a bit difficult to discover, as it covers quite a niche area.
  We think that with the more popular and general-fit xarray, not only are we hard to discover,
  but our ideal users might be finding xarray, calling it "good enough", and not putting in the
  extra effort to find the better fitting packages.

* It might also be that Iris' user base is largely Met Office employees, who would rely on
  support and Viva Engage. We *DO NOT* think that we should put more focus onto Met Office
  employees, as that would skew the balance and dissuade further users from joining us.

* **ACTION**: Have a discussion about how much resource we can commit to publicity.
  This could involve visiting and/or presenting at conferences,
  having a greater social media presence, etc.

Documentation
*************

Ease of Cross-Package Development
=================================

We do currently have the Community tab, which includes valuable documentation for xarray and
pandas documentation. This is hard to find however, and misses links for things such as
GeoVista and ncdata. We think this should be more immediate, or less buried. We also wonder
if we should sell Iris as both a complete product *and a toolbox*, for using with xarray.

* **ACTION**: Create an issue for the above on Iris.

Accessibility and Completeness of Documentation
===============================================

We need to offer a wider and more balanced approach. Diataxis caters to all learning styles,
so would be a good start. We also think that a how-to section, like in ncdata, and more visible
tutorials/examples should be added.

* **ACTION**: Create an issue for the above on Iris.

Development Process Transparency
================================

We need a clear page on our philosophy, including why we've made certain decisions and how
that will affect users. An example is where we stand on CF standards.

We should also offer a public roadmap of our plans around Iris, and more transparency of our approach.

* **ACTION**: Create issues for the above on Iris.

Codebase
********

Ecosystem Alignment
===================

We've found numerous examples of methods etc. that follow a different pattern than is used across most other popular repos. These should all be aligned, unless it significantly hinders our users, or if the payoff wouldn't be worth the work.

# Most cube methods *could* just be root level. There's pros and cons to this. Most operations require a cube anyway, so one school of thought suggests making them methods is redundant, and many repos have moved away from this. **However**, it was raised that chaining methods are a useful utility, and that this move might not make much difference to users. We didn't come to a decision on this, but we were leaning against.

  * **ACTION** - *???*: Make a formal decision to either close the issue as unplanned, or to put it in the next major release.

# We have a number of in-place operations, that really shouldn't be. We have an issue for this, but we think this should definately be in the next major release.

  * **ACTION** - *???*: Formalise it being in the next major release (new milestone?)

# cube.copy() currently offers the ability to overwrite the data of the cube by adding the new data as a parameter. This is not the same as other packages, and adds unnecessary confusion to dataless cubes. There is an issue exiting for this.

  * **ACTION** - *???*: Formalise it being in the next major release (new milestone?)

Ease of Debugging
=================

We've been bitten a few times by silently failing bugs, either because errors aren't clear enough, or that Iris doesn't raise an error at all when something doesn't work.

* **ACTION** - *CULTURE*: Ensure when workarounds are created (often in surgeries), that these are either documented on Iris, or there is a related issue raised.

* **ACTION** - *CULTURE*: Create an issue if any unclear issues are found. These could have the UX label.

Outdated API
============

* Many of the methods that perform operations on dimensional metadata have useless duplication.
  For example, adding cell_measures requires a unique method call, even though it shares an API
  with most other metadata.
   * **ACTION**: Create an issue for the above. This should probably be a dragon.

* The way we approach dask is confusing in places. Our approach to data realisation is based on a
  legacy understanding. We want to ensure you can't accidentally realise data.

  # Why is it a method to access the dask array, and a property to realise the data?!

    * **ACTION** - *???*: Create an issue to make realising data a method, and accessing the lazy array a property.

  # Why do we have cube.data.rechunk, rather than cube.rechunk. Is this just a convenience that will open us up to further troubles?

    * Would a thin wrapper be the way forward?

      * **ACTION** - *???*: Create a discussion for the above.


Culture
*******

* **ACTION**: Create the a UX label to track related work. As a follow on task, consider some existing issues or PRs which would fit having the label.

* **ACTION**: Plan a regular session for discussing the state of the user experience.
  We discussed making this yearly, although it could be more or less frequent as needed.

Further Notes
*************

We currently have a number of ongoing PRs that we think make large strides in including our UX:

  * Merge Concatenate work

  * Iris Loading work

    * We think the approach on this was very well thought out, it maintains our CF safety net
      whilst giving users a more approachable loading experience.