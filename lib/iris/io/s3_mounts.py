# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Code to manage access to files stored in S3 buckets.

S3 buckets are mounted by name in temporary directories.
These mounts must persist, so that any data-proxies can re-open the mapped files.
They are removed either at system exit, or under the control of a context manager.
Authentication, access control, and the storage region supplied are controlled outside of
Iris/Python, by user configuration (e.g. "aws configure").

"""

import atexit
from pathlib import Path
import shutil
import subprocess
from subprocess import run
import time
from typing import List

_DO_DEBUG = True


def _DEBUG(*args, **kwargs):
    if _DO_DEBUG:
        print(*args, **kwargs)


class _S3MountsManager:
    _N_UNMOUNT_RETRIES = 3
    _NULL_PATH = Path("/<nonexist>")

    def __init__(self, base_path: str | Path | None = None):
        self.mounts_basepath: Path = self._NULL_PATH
        self.mount_paths: List[Path] = []

    def _ensure_basepath(self) -> Path:
        if self.mounts_basepath is self._NULL_PATH:
            base_path = Path("/var/tmp/__iris_s3mounts__")
            if not base_path.exists():
                base_path.mkdir(parents=True, exist_ok=False)
            self.mounts_basepath = base_path
            assert self.mounts_basepath.exists()
        return self.mounts_basepath

    def bucket_mountpath(self, bucket_name: str) -> Path:
        self._ensure_basepath()
        mount_path = self.mounts_basepath / bucket_name
        if mount_path not in self.mount_paths:
            _DEBUG(f"\nCreating S3 mount-path dir {mount_path} ...")
            mount_path.mkdir(parents=True, exist_ok=False)
            _DEBUG("...done.\n")

            _DEBUG(f"\nMounting S3 bucket {bucket_name} ...")
            try:
                run(
                    f"s3fs {bucket_name} {mount_path}",
                    shell=True,
                    capture_output=True,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                print(f"Error mounting s3 bucket {bucket_name} at {mount_path}:")
                print(exc.stderr.decode())
                raise
            _DEBUG("...done.\n")
            self.mount_paths.append(mount_path)

        return mount_path

    def unmount_bucket(self, mount_path: Path, n_tries_done: int) -> bool:
        """Attempt to unmount the specified S3 mount, or force if timed out."""
        success = False
        if n_tries_done >= self._N_UNMOUNT_RETRIES:
            _DEBUG(
                f"Unmount of {mount_path} out of retries - final attempt with lazy ..."
            )
            try:
                run(
                    f"umount {mount_path} -l",
                    shell=True,
                    capture_output=True,
                    check=True,
                )
                _DEBUG("...succeeded.\n")
                success = True
            except subprocess.CalledProcessError as exc:
                msg = exc.stderr.decode()
                print(f"Unknown error in 'umount {mount_path} -l' :", msg)
                raise
        else:
            try:
                _DEBUG(
                    f"\nUnmounting mount path {mount_path}, attempt #{n_tries_done} ..."
                )
                run(f"umount {mount_path}", shell=True, capture_output=True, check=True)
                _DEBUG("...succeeded.\n")
                success = True
            except subprocess.CalledProcessError as exc:
                msg = exc.stderr.decode()
                if "busy" in msg:
                    # This is OK. We will just pause before retrying.
                    _DEBUG("Unmount request failed with 'busy': error=", msg)
                else:
                    print(f"Unknown error attempting to unmount {mount_path}:", msg)
                    raise
        return success

    def unmount_all(self):
        # Cleanup handler
        unmount_tries = {name: 0 for name in self.mount_paths}
        while unmount_tries:
            # try once (each pass) to unmount each outstanding mount.
            to_do = list(unmount_tries.keys())
            for unmount in to_do:
                n_tries = unmount_tries[unmount]
                unmount_tries[unmount] = n_tries + 1
                try:
                    if self.unmount_bucket(unmount, n_tries):
                        del unmount_tries[unmount]
                except subprocess.CalledProcessError as exc:
                    print("Other failure on unmount: ", exc.stderr.decode())
                    raise

            if unmount_tries:
                # If any remain, pause before doing a batch of retries.
                time.sleep(1.0)

        # All finally gone: remove the base-path also
        try:
            shutil.rmtree(self.mounts_basepath)
        finally:
            self.mounts_basepath = self._NULL_PATH


# Singleton object that holds all the info.
s3_mounter = _S3MountsManager()


# Fix so that we clean up at the exit of the main program.
@atexit.register
def _cleanup():
    _DEBUG("\n\nFINAL CLEANUP.\n")
    try:
        s3_mounter.unmount_all()
    except Exception:
        pass  # ignore any exception with the cleanup handler


# #
# # ALSO: force cleanup if an uncaught Exception causes exit.
# #
# _orig_excepthook = sys.excepthook
#
# def _excepthook(*args):
#     global _orig_excepthook
#     try:
#         _cleanup()
#     finally:
#         _orig_excepthook(*args)
#
# sys.excepthook = _excepthook
