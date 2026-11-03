"""Loader for the optional native HNSW library (``hnsw.c``).

The library is built by ``pip install`` when a C compiler is available (see
``setup.py``), or on demand with ``python -m stratadb.native build``. It is
loaded through ``ctypes``; nothing here needs the Python or numpy C API.

Set ``STRATADB_NATIVE=0`` to ignore the library and use the pure-Python index;
``STRATADB_NATIVE_LIB=/path/to/lib.so`` loads that build instead of the
installed one.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent.parent
_SOURCE = Path(__file__).resolve().parent / "hnsw.c"
_ENV_FLAG = "STRATADB_NATIVE"


def library_candidates() -> list[Path]:
    """Shared-library files that could hold the native index, most specific first."""
    patterns = ["_hnsw_native*.so", "_hnsw_native*.pyd", "_hnsw_native*.dylib", "_hnsw_native*.dll"]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(Path(p) for p in glob.glob(str(_PKG_DIR / pattern)))
    return found


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    c = ctypes
    p = c.c_void_p
    sig = {
        "hnsw_new": (p, [c.c_int, c.c_int, c.c_int, c.c_int, c.c_int64]),
        "hnsw_free": (None, [p]),
        "hnsw_len": (c.c_int64, [p]),
        "hnsw_deleted_count": (c.c_int64, [p]),
        "hnsw_entry": (c.c_int32, [p]),
        "hnsw_max_level": (c.c_int32, [p]),
        "hnsw_vectors": (p, [p]),
        "hnsw_levels": (p, [p]),
        "hnsw_deleted": (p, [p]),
        "hnsw_mark_deleted": (None, [p, c.c_int64]),
        "hnsw_is_deleted": (c.c_int, [p, c.c_int64]),
        "hnsw_add_batch": (c.c_int64, [p, p, p, c.c_int64]),
        "hnsw_search": (c.c_int64, [p, p, c.c_int64, c.c_int64, p, p, p]),
        "hnsw_brute_force": (c.c_int64, [p, p, c.c_int64, p, p, p]),
        "hnsw_links_total": (c.c_int64, [p]),
        "hnsw_export_links": (None, [p, p, p]),
        "hnsw_import": (c.c_int, [p, c.c_int64, p, c.c_int, p, p, p, p, c.c_int32, c.c_int32]),
    }
    for name, (restype, argtypes) in sig.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
    return lib


_lib: ctypes.CDLL | None = None
_tried = False
_error: str | None = None


def load() -> ctypes.CDLL | None:
    """Return the bound library, or None when it is unavailable or disabled."""
    global _lib, _tried, _error
    if _tried:
        return _lib
    _tried = True
    if os.environ.get(_ENV_FLAG, "1").lower() in ("0", "false", "no", "off"):
        _error = f"disabled by {_ENV_FLAG}"
        return None
    override = os.environ.get(_ENV_FLAG + "_LIB")  # a specific build, for development
    for path in ([Path(override)] if override else library_candidates()):
        try:
            _lib = _bind(ctypes.CDLL(str(path)))
            return _lib
        except (OSError, AttributeError) as exc:
            _error = f"{path.name}: {exc}"
    if _error is None:
        _error = "library not built (run: python -m stratadb.native build)"
    return None


def available() -> bool:
    return load() is not None


def status() -> str:
    lib = load()
    if lib is not None:
        return f"native HNSW loaded from {lib._name}"
    return f"native HNSW unavailable: {_error}"


def build(verbose: bool = True) -> Path:
    """Compile ``hnsw.c`` into the package directory with the system C compiler."""
    import shutil
    import subprocess

    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        raise RuntimeError("no C compiler found (set CC or install cc/gcc/clang)")
    suffix = ".dll" if sys.platform == "win32" else ".so"
    out = _PKG_DIR / f"_hnsw_native{suffix}"
    cmd = [cc, "-O3", "-std=c11", "-fPIC", "-shared", "-o", str(out), str(_SOURCE)]
    if sys.platform == "darwin":
        cmd.insert(1, "-dynamiclib")
        cmd.remove("-shared")
    if verbose:
        print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    global _tried, _lib
    _tried, _lib = False, None
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["build"]:
        out = build()
        print(f"built {out}")
        print(status())
        return 0
    print(status())
    return 0 if available() else 1


if __name__ == "__main__":
    raise SystemExit(main())
