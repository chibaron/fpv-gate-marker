"""FpvGateMarker（FGM）の検出。仕様は docs/spec.md。"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# 黒白の幅を左から並べたもの（マス）。黒から始まる。外側に白1以上の余白
FGM_LEFT = (3, 1, 1, 1, 1)    # ゲート左の柱。太い黒が外側（左）
FGM_RIGHT = (1, 1, 1, 1, 3)   # ゲート右の柱。太い黒が外側（右）
FGM_TOL = 0.5                 # 1マスに対する許容誤差
FGM_MAX_GAP = 1               # 縦に続いているとみなすとき、読めなくてもよい走査線の数
FGM_MIN_HEIGHT = 0.7          # 縦に続いた長さが幅のこの倍以上（OSD の文字や ArUco の1マス分だけの一致を除く）
FGM_PAIR_GAP = 2.0            # 左と右の間が、FGM の幅のこの倍以上空いていること（ゲートの幅）
FGM_PAIR_SIZE = 3.0           # 左と右の幅の比がこの倍以内であること


@dataclass
class Detection:
    visible: bool                 # ゲートが見えているか（左右の両方が見えてから、両方とも消えるまで True）
    corners: list = field(default_factory=list)   # 見つかった FGM ごとの四隅（描画用、float32 の 4×2）
    ids: list = field(default_factory=list)       # corners と同じ順に "L"（左）/ "R"（右）


def _similar_edges(pattern):
    """黒の始まり→次の黒の始まり、白の始まり→次の白の始まりの距離（マス）。
    黒がにじんで太っても細っても変わらない（バーコードの読み方）。"""
    return [pattern[k] + pattern[k + 1] for k in range(len(pattern) - 1)]


def _binarize_rows(gray: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """走査線 ys を、行ごとに周りの明るさで2値化する（黒が True。逆光で全体が暗くても読めるように）。"""
    w = gray.shape[1]
    rows = cv2.GaussianBlur(gray[ys], (5, 1), 0).astype(np.float32)   # 横だけぼかしてノイズを減らす
    mean = cv2.blur(rows, (max(31, (w // 4) | 1), 1), borderType=cv2.BORDER_REPLICATE)
    return rows < mean - 5


def _find_pattern(black: np.ndarray, min_width: float, pattern) -> list:
    """黒から始まる黒白の並びが pattern の比に合う区間を (走査線番号, 左端x, 右端x) で返す。
    外側には白（余白）があること。numpy でまとめて処理する。"""
    n = len(pattern)
    w = black.shape[1]
    rr, cc = np.nonzero(black[:, 1:] != black[:, :-1])
    cc = cc + 1                              # 色が変わった最初の画素
    if len(cc) < n + 1:
        return []
    m = len(cc) - n
    e = [cc[j:j + m] for j in range(n + 1)]  # 境目 e0〜en
    ok = (rr[:m] == rr[n:]) & black[rr[:m], e[0]]   # 同じ走査線で、黒から始まる
    # 1マスの大きさ。e0→e(n-1) は「黒の始まり→黒の始まり」なのでにじみに強い
    u = (e[n - 1] - e[0]) / float(sum(pattern[:n - 1]))
    ok &= (e[n] - e[0]) >= min_width
    for j, d in enumerate(_similar_edges(pattern)):
        ok &= np.abs((e[j + 2] - e[j]) - d * u) <= FGM_TOL * u
    # 外側の白（余白）が1マスくらいはあること。行の端は余白ありとみなす
    idx = np.arange(m)
    pi = np.maximum(idx - 1, 0)
    prev = np.where((idx > 0) & (rr[pi] == rr[:m]), cc[pi], 0)
    ni = np.minimum(idx + n + 1, len(cc) - 1)
    nxt = np.where((idx + n + 1 < len(cc)) & (rr[ni] == rr[:m]), cc[ni], w)
    ok &= (e[0] - prev >= FGM_TOL * u) & (nxt - e[n] >= FGM_TOL * u)
    sel = np.nonzero(ok)[0]
    return [(int(rr[i]), int(e[0][i]), int(e[n][i])) for i in sel]


class FgmDetector:
    """左右2枚の FpvGateMarker（FGM）を、間引いた走査線で探す。

    走査線ごとに黒白の並びが比率に合う所を探し、`min_lines` 本以上の走査線で同じ位置に
    続いていて、その長さが幅の FGM_MIN_HEIGHT 倍以上なら1枚のマーカーとする。
    左が右より左に、ゲートの幅ぶん離れて、両方見えたら「見えた」。いったん見えたら、
    左右の両方とも見えなくなるまで「見えた」のまま（近づいて片方が画面から出ても通過にしない）。
    映像（カメラ）ごとに1つ作る（前のフレームの状態を持つ）。
    円形ゲートの輪（fgm.markers.gate_ring_pages）も、中心を通る横線の左右がこの並びになるので読める。
    """

    def __init__(self, line_step: int = 4, min_lines: int = 3, min_size_pct: float = 5.0):
        self.line_step = max(1, int(line_step))
        self.min_lines = max(1, int(min_lines))
        self.min_size_pct = float(min_size_pct)
        self.seen = False    # 左右両方が見えてから、両方とも見えなくなるまで True

    def reset(self) -> None:
        self.seen = False

    def detect(self, frame: np.ndarray) -> Detection:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = gray.shape
        ys = np.arange(self.line_step // 2, h, self.line_step)
        black = _binarize_rows(gray, ys)
        min_width = h * self.min_size_pct / 100.0
        found = {"L": self._chains(_find_pattern(black, min_width, FGM_LEFT), ys),
                 "R": self._chains(_find_pattern(black, min_width, FGM_RIGHT), ys)}
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
        """左右の FGM が1つのゲートに見えるか。OSD の文字のように隣り合った一致を除く。"""
        wl, wr = left["width"], right["width"]
        if max(wl, wr) > FGM_PAIR_SIZE * min(wl, wr):
            return False
        return right["x0"] - left["x1"] >= FGM_PAIR_GAP * (wl + wr) / 2

    def _chains(self, cands: list, ys: np.ndarray) -> list:
        """縦に続く候補をつなぎ、min_lines 本以上のものを返す。"""
        chains = []          # {"rows": [...], "x0": [...], "x1": [...]}
        for r, x0, x1 in sorted(cands):
            cx, width = (x0 + x1) / 2, x1 - x0
            best = None
            for c in chains:
                gap = r - c["rows"][-1]
                if gap < 1 or gap > FGM_MAX_GAP + 1:
                    continue
                pw = c["x1"][-1] - c["x0"][-1]
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
