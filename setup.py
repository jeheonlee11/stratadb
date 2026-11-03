"""Build hook for the optional native HNSW library.

Everything else is declared in pyproject.toml. The extension is a plain C
shared library loaded through ctypes (no Python/numpy C API), marked
``optional`` so an install without a C compiler still succeeds and StrataDB
falls back to the pure-Python index. ``python -m stratadb.native build``
compiles it later.
"""

import sys

from setuptools import Extension, setup

extra = [] if sys.platform == "win32" else ["-O3", "-std=c11"]

setup(
    ext_modules=[
        Extension(
            "stratadb._hnsw_native",
            sources=["stratadb/native/hnsw.c"],
            extra_compile_args=extra,
            optional=True,
        )
    ],
)
