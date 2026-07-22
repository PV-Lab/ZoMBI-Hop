"""Warm-start sampling and greedy optima finding on the probability simplex."""

# This suite drives shared code (src/, optimize/) that was written on a UTF-8 Linux
# HPC node and logs freely with non-ASCII characters (e.g. the "->" arrow U+2192 in
# src/utils/gp_simplex.determine_penalty_ellipsoid's radius-cap message). On Windows
# the default console/redirect encoding is cp1252, so such a print raises
# UnicodeEncodeError — and because that print happens *inside* the optimizer's
# needle-declaration path, the exception aborts the whole ZoMBI run mid-trial. The
# warm arm hits the radius cap far more often than the cold arm, so left unfixed this
# silently truncates one side of the comparison. Forcing stdio to UTF-8 here — once,
# at package import, before any entry point (validate/compare/analyze) runs — makes
# the codebase's unicode logging safe regardless of the host locale, rather than
# chasing individual characters through the shared production path.
import sys as _sys

for _stream in (_sys.stdout, _sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError):
            # Stream doesn't support reconfigure (e.g. already-wrapped); leave as-is.
            pass
