"""FpvGateMarker (FGM): a striped marker for FPV drone race gates, and its detector.

FpvGateMarker（FGM）: FPV ドローンのゲートに貼る縦縞のマーカーと、その検出。
"""
from .detector import FGM_LEFT, FGM_RIGHT, FGM_TOL, Detection, FgmDetector

__version__ = "0.1.0"
__all__ = ["FGM_LEFT", "FGM_RIGHT", "FGM_TOL", "Detection", "FgmDetector", "__version__"]
