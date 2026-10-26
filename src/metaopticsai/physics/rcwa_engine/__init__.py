"""Vendored metabox3 (TF-based RCWA simulator).

Copy your existing metabox3/ folder here verbatim. Only
`physics/backends/metabox.py` should import from this package.
"""

# Re-export the canonical submodules — at runtime these only resolve once
# the actual metabox3 sources are copied into this directory.
try:
    # Order matters: utils defines the type aliases that rcwa/raster/modeling/
    # assembly re-import from the package (`from . import CoordType, ...`).
    # Hoist them onto the package namespace before those submodules are loaded.
    from . import utils  # noqa: F401
    from .utils import (  # noqa: F401
        CoordType, Feature, Incidence, ParameterType,
    )
    from . import rcwa, modeling, assembly  # noqa: F401
except ImportError as _e:  # pragma: no cover
    import warnings
    warnings.warn(
        f"vendor.metabox3 not populated yet ({_e}); copy your metabox3/ "
        "directory into src/metaopticsai/vendor/metabox3/."
    )
