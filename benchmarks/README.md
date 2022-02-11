[//]: # (the final pieces of this setup are still falling into place, so have
         included several TODO's)

# Iris Performance Benchmarking

Iris uses an [Airspeed Velocity](https://github.com/airspeed-velocity/asv)
(ASV) setup to benchmark performance. This is primarily designed to check for
performance shifts between commits using statistical analysis, but can also
be easily repurposed for manual comparative and scalability analyses.

The benchmarks are automatically run overnight
[by a GitHub Action](../.github/workflows/benchmark.yml), with any notable
shifts in performance being flagged in a new GitHub issue.

## Running benchmarks

`asv ...` commands must be run from this directory. You will need to have ASV
installed, as well as Nox (see 
[Benchmark environments](#benchmark-environments)).

[Iris' noxfile](../noxfile.py) includes a `benchmarks` session that provides
conveniences for setting up, and even replicating the automated overnight run
locally. See the session docstring for detail.

[//]: # (TODO: ### Environment variables section)

## Writing benchmarks

[//]: # (TODO: ### Data generation section)

[See the ASV docs](https://asv.readthedocs.io/) for full detail.

### ASV re-run behaviour

Note that ASV re-runs a benchmark multiple times between its `setup()` routine.
This is a problem for benchmarking certain Iris operations such as data
realisation, since the data will no longer be lazy after the first run.
Consider writing extra steps to restore objects' original state _within_ the
benchmark itself.

If adding steps to the benchmark will skew the result too much then re-running
can be disabled by setting an attribute on the benchmark: `number = 1`. To
maintain result accuracy this should be accompanied by increasing the number of
repeats _between_ `setup()` calls using the `repeat` attribute.
`warmup_time = 0` is also advisable since ASV performs independent re-runs to
estimate run-time, and these will still be subject to the original problem.

### Scaling / non-Scaling Performance Differences

When comparing performance between commits/file-type/whatever it can be helpful
to know if the differences exist in scaling or non-scaling parts of the Iris
functionality in question. This can be done using a size parameter, setting
one value to be as small as possible (e.g. a scalar `Cube`), and the other to
be significantly larger (e.g. a 1000x1000 `Cube`). Performance differences
might only be seen for the larger value, or the smaller, or both, getting you
closer to the root cause.

## Benchmark environments

We have disabled ASV's standard environment management, instead using an
environment built using the same Nox scripts as Iris' test environments. This
is done using ASV's plugin architecture - see
[asv_delegated_conda.py](asv_delegated_conda.py) and the extra config items in
[asv.conf.json](asv.conf.json).

(ASV is written to control the environment(s) that benchmarks are run in -
minimising external factors and also allowing it to compare between a matrix
of dependencies (each in a separate environment). We have chosen to sacrifice
these features in favour of testing each commit with its intended dependencies,
controlled by Nox + lock-files).
