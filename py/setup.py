# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
"""Build unilinalg._core, a Cython extension over the UniLinalg C ABI.

Normal development: run `nimble pyLib` first so the library is at the repo
root, then any setup.py command. Installing from the sdist -- no repo root,
just this py/ project extracted standalone -- builds the vendored Nim source
under _nimsrc/ automatically via `nimble`; Nim and nimble must be on PATH
(https://nim-lang.org/install.html)."""
import os
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from Cython.Build import cythonize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG_DIR = os.path.join(HERE, "unilinalg")
VENDOR_DIR = os.path.join(HERE, "_nimsrc")
NIMBLE_FILE = "UniLinalg.nimble"
VENDOR_FILES = [NIMBLE_FILE, "config.nims"]
VENDOR_DIRS = ["src", "include"]

# Windows: link a vcc static lib, since MSVC CPython cannot link MinGW output.
# Elsewhere: bundle the shared lib in the package, found through an rpath
# relative to the extension. macOS rejects distutils' -R, hence extra_link_args.
if sys.platform == "win32":
    LIB_NAME, BUNDLED = "UniLinalg.lib", False
    RUNTIME_DIRS, LINK_ARGS, NIMBLE_TASK = [], [], "clibMsvc"
elif sys.platform == "darwin":
    LIB_NAME, BUNDLED = "libUniLinalg.dylib", True
    RUNTIME_DIRS, LINK_ARGS, NIMBLE_TASK = [], ["-Wl,-rpath,@loader_path"], "clib"
else:
    LIB_NAME, BUNDLED = "libUniLinalg.so", True
    RUNTIME_DIRS, LINK_ARGS, NIMBLE_TASK = ["$ORIGIN"], [], "clib"


def vendor_nim_source():
    """Copy the Nim source tree into py/_nimsrc/ so it travels inside the
    sdist -- setuptools only packages files under the project directory
    (py/), never a parent via `../`."""
    if os.path.exists(VENDOR_DIR):
        shutil.rmtree(VENDOR_DIR)
    os.makedirs(VENDOR_DIR)
    for f in VENDOR_FILES:
        shutil.copy2(os.path.join(ROOT, f), os.path.join(VENDOR_DIR, f))
    for d in VENDOR_DIRS:
        shutil.copytree(os.path.join(ROOT, d), os.path.join(VENDOR_DIR, d))


def nim_project_dir():
    """Where UniLinalg.nimble lives: the real repo root in a normal checkout,
    or the vendored copy when building from an extracted sdist (which has
    no parent repo, just this project standalone)."""
    if os.path.exists(os.path.join(ROOT, NIMBLE_FILE)):
        return ROOT
    if os.path.exists(os.path.join(VENDOR_DIR, NIMBLE_FILE)):
        return VENDOR_DIR
    return None


def ensure_lib_built():
    """Return the path to the built lib, compiling it via nimble first when
    installing from an sdist (no prebuilt lib shipped, source-only)."""
    prebuilt = os.path.join(ROOT, LIB_NAME)
    if os.path.exists(prebuilt):
        return prebuilt
    proj = nim_project_dir()
    if proj is None:
        raise SystemExit(
            f"setup.py: {prebuilt} not found — run `nimble {NIMBLE_TASK}` first."
        )
    built = os.path.join(proj, LIB_NAME)
    if os.path.exists(built):
        return built
    nimble = shutil.which("nimble")
    if nimble is None:
        raise SystemExit(
            "setup.py: `nimble` not found on PATH. Building unilinalg from "
            "source needs Nim (https://nim-lang.org/install.html)."
        )
    try:
        subprocess.check_call([nimble, "install", "--depsOnly", "-y"], cwd=proj)
        subprocess.check_call([nimble, NIMBLE_TASK], cwd=proj)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"setup.py: `nimble {NIMBLE_TASK}` failed: {e}") from e
    if not os.path.exists(built):
        raise SystemExit(f"setup.py: `nimble {NIMBLE_TASK}` did not produce {built}")
    return built


# `sdist` packages source only -- it never compiles anything, so it must not
# require a prebuilt lib. Every other command (build_ext, bdist_wheel, ...)
# needs a real lib to link against, built locally or, from an sdist, via
# nimble -- including a combined invocation like `sdist bdist_wheel`, so the
# sole-command check below (not a bare `"sdist" in sys.argv`) is what decides
# this, ignoring option flags like `--formats=gztar`.
commands = [a for a in sys.argv[1:] if not a.startswith("-")]
if commands == ["sdist"]:
    vendor_nim_source()
    INCLUDE, LIB_DIR = os.path.join(ROOT, "include"), ROOT
else:
    lib_path = ensure_lib_built()
    LIB_DIR = os.path.dirname(lib_path)
    INCLUDE = os.path.join(ROOT, "include")
    if not os.path.isdir(INCLUDE):
        INCLUDE = os.path.join(VENDOR_DIR, "include")
    if BUNDLED:
        os.makedirs(PKG_DIR, exist_ok=True)
        shutil.copy2(lib_path, os.path.join(PKG_DIR, LIB_NAME))

# The sdist ships the pre-transpiled unilinalg/_core.c, not the .pyx (Cython
# rewrites Extension.sources from .pyx to .c when it builds the sdist, so the
# .pyx is never actually collected). Cythonize only when the .pyx is present
# (a normal git checkout); an sdist install compiles the shipped .c directly,
# needing no Cython.
pyx = os.path.join("unilinalg", "_core.pyx")
core_c = os.path.join("unilinalg", "_core.c")
ext = Extension(
    "unilinalg._core",
    # setuptools requires sources to be relative to setup.py's own directory
    # (an absolute path here fails bdist_wheel's manifest step) -- the
    # existence check below still needs an absolute path since it must be
    # correct regardless of the caller's cwd.
    sources=[pyx if os.path.exists(os.path.join(HERE, pyx)) else core_c],
    include_dirs=[INCLUDE],
    library_dirs=[LIB_DIR],
    runtime_library_dirs=RUNTIME_DIRS,
    extra_link_args=LINK_ARGS,
    libraries=["UniLinalg"],
)
ext_modules = cythonize([ext], language_level=3) if ext.sources[0].endswith(".pyx") else [ext]

setup(
    ext_modules=ext_modules,
    include_package_data=True,
    package_data={"unilinalg": [LIB_NAME] if BUNDLED else []},
    exclude_package_data={"unilinalg": ["_core.c"]},
    zip_safe=False,
)
