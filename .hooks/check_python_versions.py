# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Pre-commit hook: ensure Python versions are consistent across config files.

Python versions must be updated consistently everywhere they appear in the
repository (pyproject.toml, noxfile.py, CI workflows, requirements files, etc.).

This hook runs when any version-related file is modified and validates that
all versions are in sync across the entire repository.

Usage (pre-commit passes staged file paths as arguments)::

    python .hooks/check_python_versions.py [file ...]

Exit codes:
    0 – all versions are consistent
    1 – version inconsistencies detected
"""

from pathlib import Path
from packaging.version import Version
import sys


# Files that contain Python version information
_VERSION_CONFIG_FILES = {
    "pyproject.toml",
    "noxfile.py",
    ".github/workflows/ci-wheels.yml",
    ".github/workflows/ci-tests.yml",
    "benchmarks/bm_runner.py",
}


def _find_repo_root(start_path: Path) -> Path:
    """Find the repository root by looking for .git."""
    current = start_path.resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return start_path


def _extract_supported_versions(root_dir: Path) -> list[str] | None:
    """Extract supported Python versions from pyproject.toml.
    
    Returns None if versions cannot be determined.
    """
    pyproject = root_dir / "pyproject.toml"
    if not pyproject.exists():
        return None

    try:
        content = pyproject.read_text()
        # Look for "Programming Language :: Python :: 3.X" entries
        import re
        matches = re.findall(
            r'"Programming Language :: Python :: (3\.\d+)"',
            content
        )
        if matches:
            return sorted(set(matches))
    except (OSError, UnicodeDecodeError):
        pass

    return None


def check_consistency(root_dir: Path) -> list[str]:
    """Check that Python versions are consistent across all config files.
    
    Returns a list of violation strings, empty if all consistent.
    """
    # Get the authoritative version list from pyproject.toml
    all_supported = _extract_supported_versions(root_dir)
    if not all_supported:
        # Can't validate without knowing what versions should be supported
        return []

    _parsed = [Version(v) for v in all_supported]
    latest_supported = str(max(_parsed))

    violations = []
    checks = []

    # pyproject.toml - version classifiers
    pyproject_toml_file = root_dir / "pyproject.toml"
    if pyproject_toml_file.exists():
        expected = "\n    ".join(
            [f'"Programming Language :: Python :: {ver}",' for ver in all_supported]
        )
        checks.append((pyproject_toml_file, expected, "pyproject.toml version classifiers"))

    # noxfile.py - _PY_VERSIONS_ALL
    nox_file = root_dir / "noxfile.py"
    if nox_file.exists():
        expected = "_PY_VERSIONS_ALL = [" + ", ".join([f'"{ver}"' for ver in all_supported])
        checks.append((nox_file, expected, "noxfile.py _PY_VERSIONS_ALL"))

    # CI workflows
    ci_wheels_file = root_dir / ".github" / "workflows" / "ci-wheels.yml"
    if ci_wheels_file.exists():
        expected = "python-version: [" + ", ".join([f'"{ver}"' for ver in all_supported])
        checks.append((ci_wheels_file, expected, "ci-wheels.yml python-version"))

    ci_tests_file = root_dir / ".github" / "workflows" / "ci-tests.yml"
    if ci_tests_file.exists():
        expected = (
            f'python-version: ["{latest_supported}"]\n'
            f'{" " * 8}session: ["doctest", "gallery"]'
        )
        checks.append((ci_tests_file, expected, "ci-tests.yml doctest/gallery"))

    benchmarks_dir = root_dir / "benchmarks"
    benchmark_runner_file = benchmarks_dir / "bm_runner.py"
    if benchmark_runner_file.exists():
        expected = f'python_version = "{latest_supported}"'
        checks.append((benchmark_runner_file, expected, "bm_runner.py python_version"))

    # Requirements files
    requirements_dir = root_dir / "requirements"
    if requirements_dir.exists():
        for ver in all_supported:
            req_yaml = requirements_dir / f"py{ver.replace('.', '')}.yml"
            if req_yaml.exists():
                expected = f"- python ={ver}"
                checks.append((req_yaml, expected, f"requirements/py{ver.replace('.', '')}.yml"))

    # CI tests file version entries
    if ci_tests_file.exists():
        for ver in all_supported:
            expected = f'python-version: "{ver}"\n{" " * 12}session: "tests"'
            checks.append((ci_tests_file, expected, f"ci-tests.yml py{ver} tests"))

    # Run all checks
    for path, search, description in checks:
        try:
            content = path.read_text()
            if search not in content:
                violations.append(
                    f"{path}: missing expected version entry for {description}"
                )
        except (OSError, UnicodeDecodeError):
            pass

    return violations


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Find repository root
    if not argv:
        # No files provided, can't determine repo root
        # In this case, do nothing (let normal test handle it)
        return 0

    paths = [Path(p) for p in argv]
    repo_root = _find_repo_root(paths[0])

    # Check if any of the version-related files are in the staged files
    is_version_file_staged = any(
        str(p).endswith(tuple(f"/{name}" for name in _VERSION_CONFIG_FILES))
        or str(p).endswith(tuple(_VERSION_CONFIG_FILES))
        for p in paths
    )

    if not is_version_file_staged:
        # Only run check if a version-related file is being modified
        return 0

    violations = check_consistency(repo_root)

    if violations:
        print(
            "Python version consistency check failed.\n"
            "Python versions must be updated consistently across all config files.\n"
        )
        for v in violations:
            print(v)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
