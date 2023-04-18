.. include:: ../common_links.inc

.. _contributing.benchmarks:

Benchmarking
============
Iris includes architecture for benchmarking performance and other metrics of
interest. This is done using the `Airspeed Velocity`_ (ASV) package.

Full detail on the setup and how to run or write benchmarks is in
`benchmarks/README.md`_ in the Iris repository.

Continuous Integration
----------------------
The primary purpose of `Airspeed Velocity`_, and Iris' specific benchmarking
setup, is to monitor for performance changes using statistical comparison
between commits, and this forms part of Iris' continuous integration.

Accurately assessing performance takes longer than functionality pass/fail
tests, so the benchmark suite is not automatically run against open pull
requests, instead it is **run overnight against each the commits of the
previous day** to check if any commit has introduced performance shifts.
Detected shifts are reported in a new Iris GitHub issue.

If a pull request author/reviewer suspects their changes may cause performance
shifts, a convenience script is available to replicate the
overnight benchmark run but comparing the current ``HEAD`` with a requested
branch (e.g. ``upstream/main``). Read more in `benchmarks/README.md`_.

Other Uses
----------
Even when not statistically comparing commits, ASV's accurate execution time
results - recorded using a sophisticated system of repeats - have other
applications.

* Absolute numbers can be interpreted providing they are recorded on a
  dedicated resource.
* Results for a series of commits can be visualised for an intuitive
  understanding of when and why changes occurred.

  .. image:: asv_example_images/commits.png
     :width: 300

* Parameterised benchmarks make it easy to visualise:

  * Comparisons

    .. image:: asv_example_images/comparison.png
       :width: 300

  * Scalability

    .. image:: asv_example_images/scalability.png
       :width: 300

This also isn't limited to execution times. ASV can also measure memory demand,
and even arbitrary numbers (e.g. file size, regridding accuracy), although
without the repetition logic that execution timing has.


.. _Airspeed Velocity: https://github.com/airspeed-velocity/asv
.. _benchmarks/README.md: https://github.com/SciTools/iris/blob/main/benchmarks/README.md
