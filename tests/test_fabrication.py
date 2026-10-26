"""Test 11: GDS export (skipped if gdsfactory unavailable)."""
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pytest


def _gdsfactory_available() -> bool:
    try:
        import gdsfactory  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _gdsfactory_available(), reason="gdsfactory not installed")
def test_gds_engine_generates_file():
    """Test 11 — GDSEngine produces a non-empty .gds file."""
    from metaopticsai.fabrication.gds import GDSEngine

    quantized = np.ones((10, 10), dtype=int)
    bins = list(range(1, 9))
    dimensions = list(range(50, 330, 35))
    phases = [i * 45.0 for i in range(8)]

    engine = GDSEngine()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "test_lens")
        stats = engine.generate(
            quantized_mask=quantized,
            bins=bins,
            dimensions=dimensions,
            phases=phases,
            period=350.0,
            pixel_size_um=0.35,
            unitcells_per_pixel=1,
            shape="Cylinder",
            is_circular=False,
            output_path=out_path,
        )
        assert Path(out_path + ".gds").exists()
        assert stats["file_size_kb"] > 0
