"""FpvGateMarker (FGM) detection. See docs/spec.md for the specification.

FpvGateMarker（FGM）の検出。仕様は docs/spec.md。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except ImportError as e:
    # OpenCV is not a dependency (opencv-python and opencv-contrib-python conflict), so say how to install it
    # OpenCV は依存に入れていない（opencv-python と opencv-contrib-python がぶつかる）ので、入れ方を示す
    raise ImportError("fgm needs OpenCV. Install it with: pip install fpv-gate-marker[opencv] "
                      "(or install opencv-python / opencv-contrib-python yourself)") from e

# Black/white widths from left to right (modules), starting with black. At least 1 module of white outside.
# 黒白の幅を左から並べたもの（マス）。黒から始まる。外側に白1以上の余白
FGM_LEFT = (3, 1, 1, 1, 1)    # left post, thick black on the outside (left) / ゲート左の柱。太い黒が外側（左）
FGM_RIGHT = (1, 1, 1, 1, 3)   # right post, thick black on the outside (right) / ゲート右の柱。太い黒が外側（右）
FGM_TOL = 0.5                 # tolerance in modules / 1マスに対する許容誤差
FGM_MAX_GAP = 1               # unreadable scan lines allowed in a vertical chain / 縦に続いているとみなすとき、読めなくてもよい走査線の数
# Vertical extent must be at least this times the width (rejects short matches such as OSD characters or one ArUco module)
# 縦に続いた長さが幅のこの倍以上（OSD の文字や ArUco の1マス分だけの一致を除く）
FGM_MIN_HEIGHT = 0.7
FGM_PAIR_GAP = 0.2            # left-right gap must be at least this times the FGM width / 左と右の間が、FGM の幅のこの倍以上空いていること
FGM_PAIR_SIZE = 2.0           # left/right width ratio must be within this factor / 左と右の幅の比がこの倍以内であること


@dataclass
class Detection:
    # Whether the gate is visible (True from when both sides are seen until both are gone)
    # ゲートが見えているか（左右の両方が見えてから、両方とも消えるまで True）
    visible: bool
    corners: list = field(default_factory=list)   # corners of each FGM found (for drawing, float32 4x2) / 見つかった FGM ごとの四隅（描画用、float32 の 4×2）
    ids: list = field(default_factory=list)       # "L" (left) / "R" (right), in the same order as corners / corners と同じ順に "L"（左）/ "R"（右）


def _similar_edges(pattern):
    """Distances (modules) from the start of black to the next start of black, and white to the next white.
    They do not change when black bleeds wider or narrower (the way barcodes are read).

    黒の始まり→次の黒の始まり、白の始まり→次の白の始まりの距離（マス）。
    黒がにじんで太っても細っても変わらない（バーコードの読み方）。"""
    return [pattern[k] + pattern[k + 1] for k in range(len(pattern) - 1)]


def _binarize_rows(frame: np.ndarray, line_step: int) -> np.ndarray:
    """Binarize every line_step-th scan line against its local brightness (black is True; works even when backlit).
    Only the scan lines are converted to gray. The mean stays uint8 (rounded integer) for the comparison.

    line_step 画素おきの走査線を、行ごとに周りの明るさで2値化する（黒が True。逆光で全体が暗くても読めるように）。
    グレーにするのは走査線だけ。平均は uint8（四捨五入した整数）のまま比べる。"""
    rows = frame[line_step // 2::line_step]
    if rows.ndim == 3:
        rows = cv2.cvtColor(rows, cv2.COLOR_BGR2GRAY)
    rows = cv2.GaussianBlur(rows, (5, 1), 0)      # blur horizontally only to reduce noise / 横だけぼかしてノイズを減らす
    w = rows.shape[1]
    mean = cv2.blur(rows, (max(31, (w // 4) | 1), 1), borderType=cv2.BORDER_REPLICATE)
    return rows < cv2.subtract(mean, 5)           # 0 when mean < 5 (never black) / mean < 5 のときは 0（黒にならない）


def _find_patterns(black: np.ndarray, min_width: float, patterns) -> list:
    """For each pattern, return the runs where black/white starting with black match its ratios,
    as [(scan line index, left x, right x), ...]. There must be white (quiet zone) outside.
    Color edges are found once; cheap conditions filter candidates before the ratios are checked.
    Ratios are compared with integers.

    黒から始まる黒白の並びが patterns のそれぞれの比に合う区間を、並びごとに
    [(走査線番号, 左端x, 右端x), ...] で返す。外側には白（余白）があること。
    色の境目は1回だけ求め、軽い条件で候補を絞ってから比を調べる。比は整数で比べる。"""
    n = len(patterns[0])
    w = black.shape[1]
    flat = black.ravel()
    k = np.flatnonzero(flat[1:] != flat[:-1]) + 1   # first pixel after a color change (flattened index) / 色が変わった最初の画素（平らにした位置）
    keep = k % w != 0                                # a row start is compared with the previous row's end, so drop it / 行の頭は前の行の終わりと比べているので除く
    rr, cc = k[keep] // w, k[keep] % w
    if len(cc) < n + 1:
        return [[] for _ in patterns]
    m = len(cc) - n
    # Same scan line, wide enough, starting with black / 同じ走査線で、幅が足りて、黒から始まるもの
    i = np.flatnonzero((rr[:m] == rr[n:]) & (cc[n:] - cc[:m] >= min_width))
    i = i[black[rr[i], cc[i]]]
    e = [cc[i + j] for j in range(n + 1)]            # edges e0..en / 境目 e0〜en
    # Width of the white (quiet zone) outside. The row ends count as quiet zone.
    # 外側の白（余白）の幅。行の端は余白ありとみなす
    pi = np.maximum(i - 1, 0)
    prev = e[0] - np.where((i > 0) & (rr[pi] == rr[i]), cc[pi], 0)
    ni = np.minimum(i + n + 1, len(cc) - 1)
    nxt = np.where((i + n + 1 < len(cc)) & (rr[ni] == rr[i]), cc[ni], w) - e[n]
    out = []
    for pattern in patterns:
        # One module u = d / s. e0 -> e(n-1) is "start of black -> start of black", so it is robust to bleeding.
        # "x is within FGM_TOL*u of a*u" is multiplied by s on both sides: |s*x - a*d| <= FGM_TOL*d (no division error).
        # 1マス u = d / s。e0→e(n-1) は「黒の始まり→黒の始まり」なのでにじみに強い。
        # 「x と a·u の差が FGM_TOL·u 以内」を両辺 s 倍して |s·x − a·d| ≤ FGM_TOL·d で比べる（割り算の誤差を出さない）
        s = sum(pattern[:n - 1])
        d = e[n - 1] - e[0]
        tol = FGM_TOL * d
        ok = (s * prev >= tol) & (s * nxt >= tol)
        for j, a in enumerate(_similar_edges(pattern)):
            ok &= np.abs(s * (e[j + 2] - e[j]) - a * d) <= tol
        sel = i[ok]
        out.append(list(zip(rr[sel].tolist(), cc[sel].tolist(), cc[sel + n].tolist())))
    return out


class FgmDetector:
    """Finds a left and right pair of FpvGateMarkers (FGM) on sparse scan lines.

    On each scan line it looks for black/white runs matching the ratios. If a match continues at the same
    position on at least `min_lines` scan lines and its extent is at least FGM_MIN_HEIGHT times its width,
    it is one marker. When both are seen, left to the left of right and not touching, the gate is
    "visible". Once visible, it stays visible until both sides are gone (one side leaving the frame as you
    approach is not a pass). Create one per video stream (camera); it keeps state from the previous frame.
    A round gate ring (fgm.markers.gate_ring_pages) is read too, because a horizontal line through its center
    crosses the same patterns on the left and right.

    左右2枚の FpvGateMarker（FGM）を、間引いた走査線で探す。

    走査線ごとに黒白の並びが比率に合う所を探し、`min_lines` 本以上の走査線で同じ位置に
    続いていて、その長さが幅の FGM_MIN_HEIGHT 倍以上なら1枚のマーカーとする。
    左が右より左に、くっつかずに離れて、両方見えたら「見えた」。いったん見えたら、
    左右の両方とも見えなくなるまで「見えた」のまま（近づいて片方が画面から出ても通過にしない）。
    映像（カメラ）ごとに1つ作る（前のフレームの状態を持つ）。
    円形ゲートの輪（fgm.markers.gate_ring_pages）も、中心を通る横線の左右がこの並びになるので読める。
    """

    def __init__(self, line_step: int = 4, min_lines: int = 3, min_size_pct: float = 5.0):
        self.line_step = max(1, int(line_step))
        self.min_lines = max(1, int(min_lines))
        self.min_size_pct = float(min_size_pct)
        self.seen = False    # True from when both sides are seen until both are gone / 左右両方が見えてから、両方とも見えなくなるまで True

    def reset(self) -> None:
        self.seen = False

    def detect(self, frame: np.ndarray) -> Detection:
        h = frame.shape[0]
        ys = np.arange(self.line_step // 2, h, self.line_step)
        black = _binarize_rows(frame, self.line_step)
        left, right = _find_patterns(black, h * self.min_size_pct / 100.0, (FGM_LEFT, FGM_RIGHT))
        found = {"L": self._chains(left, ys), "R": self._chains(right, ys)}
        corners, ids = [], []
        for name, chains in found.items():
            for c in chains:
                corners.append(c["pts"])
                ids.append(name)
        if any(self._is_gate(l, r) for l in found["L"] for r in found["R"]):
            self.seen = True
        elif not ids:
            self.seen = False
        return Detection(self.seen, corners, ids)

    @staticmethod
    def _is_gate(left: dict, right: dict) -> bool:
        """Whether a left and right FGM look like one gate. Rejects adjacent matches such as OSD characters.

        左右の FGM が1つのゲートに見えるか。OSD の文字のように隣り合った一致を除く。"""
        wl, wr = left["width"], right["width"]
        if max(wl, wr) > FGM_PAIR_SIZE * min(wl, wr):
            return False
        return right["x0"] - left["x1"] >= FGM_PAIR_GAP * (wl + wr) / 2

    def _chains(self, cands: list, ys: np.ndarray) -> list:
        """Chain vertically continuing candidates and return those with at least min_lines lines.

        縦に続く候補をつなぎ、min_lines 本以上のものを返す。"""
        chains = []          # {"rows": [...], "x0": [...], "x1": [...]}
        for r, x0, x1 in sorted(cands):
            cx, width = (x0 + x1) / 2, x1 - x0
            best = None
            for c in chains:
                gap = r - c["rows"][-1]
                if gap < 1 or gap > FGM_MAX_GAP + 1:
                    continue
                pw = c["x1"][-1] - c["x0"][-1]
                # The center may shift by the scan line spacing so a tilted drone can still read it
                # 機体が傾いていても読めるよう、中心は走査線の間隔ぶんずれてよい
                if abs(cx - (c["x0"][-1] + c["x1"][-1]) / 2) <= self.line_step * gap + width / 7 \
                        and 0.7 <= width / pw <= 1.43:
                    best = c
                    break
            if best is None:
                best = {"rows": [], "x0": [], "x1": []}
                chains.append(best)
            best["rows"].append(r)
            best["x0"].append(x0)
            best["x1"].append(x1)
        out = []
        for c in chains:
            if len(c["rows"]) < self.min_lines:
                continue
            y0, y1 = float(ys[c["rows"][0]]), float(ys[c["rows"][-1]])
            width = float(np.mean(np.subtract(c["x1"], c["x0"])))
            if y1 - y0 + self.line_step < FGM_MIN_HEIGHT * width:
                continue
            pts = np.array([[c["x0"][0], y0], [c["x1"][0], y0], [c["x1"][-1], y1], [c["x0"][-1], y1]],
                           dtype=np.float32)
            out.append({"pts": pts, "width": width, "x0": float(np.mean(c["x0"])),
                        "x1": float(np.mean(c["x1"]))})
        return out
