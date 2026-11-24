#!/usr/bin/env python3
"""Build a zero-copy symlinked Loop-Aware Hybrid model directory (`12.8GB` core + `8.9GB` post-loop `L18..23`)."""

from __future__ import annotations

import sys

from inkling_mlx_vision.accel import build_hybrid_cli

if __name__ == "__main__":
    raise SystemExit(build_hybrid_cli(sys.argv[1:]))
