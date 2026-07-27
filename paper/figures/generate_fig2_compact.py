#!/usr/bin/env python3
"""Thin wrapper: fig2_compact is generated ONLY by generate_all.fig2_compact
(the authoritative, compliance-hooked implementation). This script exists so
older invocations keep working without producing a divergent PDF."""
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "generate_all", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "generate_all.py"))
ga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ga)

if __name__ == "__main__":
    ga.fig2_compact()
    ga.verify_min_effective()
