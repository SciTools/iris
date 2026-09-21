# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Pre-commit hook: ensure Python files have proper license headers.

All Python files (with some exceptions) must start with the Iris license
header.

Usage (pre-commit passes staged file paths as arguments)::

    python .hooks/check_license_headers.py [file ...]

Exit codes:
    0 – all clear
    1 – one or more files lack the required license header
"""

from fnmatch import fnmatch
from pathlib import Path
import subprocess
import sys


LICENSE_TEMPLATE = """# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details."""

# Patterns of files to exclude from license header check
_EXCLUDE_PATTERNS = (
    "setup.py",
    "noxfile.py",
    "build/*",
    "dist/*",
    "docs/gallery_code/*/*.py",
    "docs/src/developers_guide/documenting/*.py",
    "docs/src/user_manual/tutorial/plotting_examples/*.py",
    "docs/src/user_manual/tutorial/regridding_plots/*.py",
    "docs/src/_build/*",
    "lib/iris/analysis/_scipy_interpolate.py",
)


def _find_repo_root(start_path: Path) -> Path | None:
    """Find the repository root by looking for .git."""
    current = start_path.resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return None


def _should_check(rel_path: str) -> bool:
    """Return True if relative path should be checked for a license header."""
    if not rel_path.endswith(".py"):
        return False

    return not any(fnmatch(rel_path, pat) for pat in _EXCLUDE_PATTERNS)


def _get_license_header_content(path: Path) -> str:
    """Get the content to check for license header, accounting for shebang."""
    content = path.read_text()
    if content.startswith("#!"):
        # Strip shebang line
        content = "\n".join(content.split("\n")[1:])
    return content


def check_file(path: Path, repo_root: Path | None = None) -> list[str]:
    """Return a list of violation strings for *path*, empty if clean.
    
    Args:
        path: Path to check (can be relative or absolute).
        repo_root: Repository root. If None, will be inferred.
    
    Returns:
        List of violation strings.
    """
    # Ensure path is absolute for consistent comparison
    path = path.resolve()
    
    if repo_root is None:
        repo_root = _find_repo_root(path)
        if not repo_root:
            repo_root = path.parents[2] if len(path.parents) > 2 else path.parent
    
    repo_root = repo_root.resolve()

    try:
        rel_path = path.relative_to(repo_root).as_posix()
    except ValueError:
        # Path is not under repo root
        return []

    if not _should_check(rel_path):
        return []

    try:
        content = _get_license_header_content(path)
    except (OSError, UnicodeDecodeError):
        # Let other tools handle file read errors
        return []

    if not content.startswith(LICENSE_TEMPLATE):
        return [f"{path}: missing or incorrect license header"]

    return []


def _get_all_tracked_files(repo_root: Path) -> list[Path]:
    """Get all Python files tracked by git using git ls-files.
    
    This is much faster than recursive glob for checking entire repo.
    """
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "*.py"],
            cwd=repo_root,
            text=True,
        )
        return [repo_root / line.strip() for line in output.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # git not available or not a git repo; fall back to glob
        return list(repo_root.rglob("*.py"))


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # If files provided, check only those files (pre-commit mode)
    if argv:
        paths = [Path(p) for p in argv]
        repo_root = _find_repo_root(paths[0])
        if not repo_root:
            repo_root = paths[0].parents[2] if len(paths[0].parents) > 2 else paths[0].parent

        all_violations: list[str] = []
        for path in paths:
            all_violations.extend(check_file(path, repo_root))
    else:
        # No files provided; check entire repo (test mode)
        # This requires finding the repo root first
        cwd = Path.cwd()
        repo_root = _find_repo_root(cwd)
        if not repo_root:
            # Try to find .git by going up from cwd
            current = cwd.resolve()
            while current != current.parent:
                if (current / ".git").exists():
                    repo_root = current
                    break
                current = current.parent
        
        if not repo_root:
            print("Error: Could not find git repository root")
            return 1

        all_violations: list[str] = []
        for path in _get_all_tracked_files(repo_root):
            all_violations.extend(check_file(path, repo_root))

    if all_violations:
        print(
            "License header check failed.\n"
            "All Python files must start with the Iris license header.\n"
        )
        for v in all_violations:
            print(v)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
