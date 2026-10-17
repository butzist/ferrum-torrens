"""Load the FreeToken probe when FT_PROBE_OUT is set.

This directory is put on PYTHONPATH for the reference server; Python imports
sitecustomize automatically in every spawned process. Nothing is patched unless
the output directory is requested, so the module is inert for normal runs.
"""

import os

if os.environ.get("FT_PROBE_OUT"):
    from probe_impl import install

    install()
