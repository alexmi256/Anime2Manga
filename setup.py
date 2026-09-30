"""Build the native seam-carving extension.

The pure-Python engine in ``src/anime2manga/seam_carving.py`` is always the
fallback; this just compiles the optional accelerator.  ``pip install .`` /
``pip install -e .`` builds it, and ``import anime2manga.seam_carving`` falls
back cleanly when it is missing.

When no C++ compiler is on ``PATH`` the extension is skipped with a warning so
installation still succeeds; a compiler that *is* present but fails to build the
engine is a real error and is allowed to fail the install.
"""

import os
import platform
import shutil
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def _has_cxx_compiler() -> bool:
    names = (os.environ.get("CXX"), "c++", "g++", "clang++", "cl")
    return any(name and shutil.which(name) for name in names)


def _extra_compile_args() -> list[str]:
    """Optimisation flags for the accelerator.

    The engine is compiled on the machine it runs on, so ``-march=native`` is
    safe and roughly 1.7x faster than the default baseline target.  Set
    ``ANIME2MANGA_PORTABLE_BUILD=1`` to produce a machine-independent (slower)
    extension, e.g. before building a wheel.
    """
    args = ["-O3", "-std=c++17"]
    machine = platform.machine().lower()
    if (
        os.environ.get("ANIME2MANGA_PORTABLE_BUILD") != "1"
        and sys.platform.startswith("linux")
        and machine in {"x86_64", "amd64", "aarch64", "arm64"}
    ):
        args.append("-march=native")
    return args


class OptionalBuildExt(build_ext):
    """Skip the accelerator only when there is no C++ compiler at all."""

    def build_extension(self, ext):
        if not _has_cxx_compiler():
            self.warn(
                "no C++ compiler found; skipping the native seam-carving extension "
                "(the pure-Python engine will be used)"
            )
            return
        super().build_extension(ext)


setup(
    ext_modules=[
        Extension(
            "anime2manga._seamcarve",
            sources=["src/anime2manga/_seamcarve.cpp"],
            language="c++",
            extra_compile_args=_extra_compile_args(),
        )
    ],
    cmdclass={"build_ext": OptionalBuildExt},
)
