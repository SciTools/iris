# Iris custom benchmarks

Structured and written in accordance with the [ASV guidelines](https://asv.readthedocs.io/projects/asv-runner/en/latest/development/benchmark_plugins.html).

To work, these benchmarks must be installed into the environment where the
benchmarks are run (i.e. not the environment containing ASV + Nox, but the
one built to the same specifications as the Tests environment). This is done
by modifying `sys.path` in 
[the benchmarks `__init__`](../benchmarks/__init__.py).
