"""FGM の印刷用画像を作る（左右の PNG、円形ゲート用の輪の PDF）。

    fgm-make-markers                       # markers/ に出力（python -m fgm.markers でも同じ）
    fgm-make-markers --module 100 --height 24
    fgm-make-markers --gate-inner 700 --gate-module 15   # 円形ゲート（内径700mm）に貼る輪

円形ゲート用（marker_fgm_ring.pdf）は、ゲートの輪に沿って貼る同心の縞。中心を通る横線で切ると、
左側が FGM 左、右側が FGM 右と同じ並びになるので、FgmDetector で読める
（円なので機体が何度傾いても読める）。A4 に入るように扇形に分けて実寸で出す。100%（実際のサイズ）で印刷する。
PDF には Pillow が要る（pip install fpv-gate-marker[pdf]）。
"""
from __future__ import annotations

import argparse
import math
import os

import cv2
import numpy as np

from .detector import FGM_LEFT, FGM_RIGHT

FGM_QUIET = 2   # FGM の外側の白（マス）。読むのに必要なのは1マスだが、ボケに備えて2マス
GATE_QUIET = 1  # 円形ゲートの輪の、一番外と一番内の白（マス）


def fgm_marker(pattern, module_px: int, height_modules: int) -> np.ndarray:
    """縦縞の FGM。左右に FGM_QUIET マスの白い余白を付ける。"""
    cols = [255] * FGM_QUIET
    for k, n in enumerate(pattern):
        cols += [0 if k % 2 == 0 else 255] * n
    cols += [255] * FGM_QUIET
    row = np.repeat(np.array(cols, np.uint8), module_px)
    return np.tile(row, (height_modules * module_px, 1))


def _gate_ring_color(r_mm, inner_mm: float, module_mm: float):
    """円形ゲートの輪で、中心からの距離 r_mm の色（黒 0 / 白 255）。輪の外は -1。
    外側から 白1 → FGM_LEFT（太い黒が外側）→ 白1。"""
    radial = [GATE_QUIET] + list(FGM_LEFT) + [GATE_QUIET]    # 外側→内側の幅（マス）
    r_in = inner_mm / 2
    r_out = r_in + sum(radial) * module_mm
    k = np.searchsorted(np.cumsum(radial) * module_mm, r_out - r_mm, side="right")
    color = np.where(k % 2 == 1, 0, 255)                      # 0番目（一番外）が白、次が黒…
    return np.where((r_mm >= r_in) & (r_mm <= r_out), color, -1)


def gate_ring_image(inner_px: int, module_px: float) -> np.ndarray:
    """円形ゲートの輪の全体（画素単位。プレビューとテスト用）。輪の外は灰色。"""
    r_out = inner_px / 2 + (sum(FGM_LEFT) + 2 * GATE_QUIET) * module_px
    size = int(math.ceil(r_out * 2)) + 2
    yy, xx = np.mgrid[0:size, 0:size]
    c = _gate_ring_color(np.hypot(xx - (size - 1) / 2, yy - (size - 1) / 2), inner_px, module_px)
    return np.where(c < 0, 128, c).astype(np.uint8)


def gate_ring_pages(inner_mm: float, module_mm: float, dpi: int, paper=(210, 297), margin=10):
    """円形ゲートの輪を、紙に入る扇形に分けて実寸で描く。(ページの画像のリスト, 分割数) を返す。"""
    r_in = inner_mm / 2
    r_out = r_in + (sum(FGM_LEFT) + 2 * GATE_QUIET) * module_mm
    pw, ph = paper[0] - 2 * margin, paper[1] - 2 * margin
    for n in range(3, 200):                  # 紙（縦か横）に入る最小の分割数
        half = math.pi / n
        bw, bh = 2 * r_out * math.sin(half), r_out - r_in * math.cos(half)
        if bw <= pw and bh <= ph:
            size = paper
            break
        if bw <= ph and bh <= pw:
            size = (paper[1], paper[0])      # 横向き
            break
    else:
        raise ValueError("輪の幅が紙に入りません。1マス（gate_module）を小さくしてください")
    ppm = dpi / 25.4                         # 1mm の画素数
    W, H = int(size[0] * ppm), int(size[1] * ppm)
    # 扇形を上向きに置き、紙の中央に。円の中心は扇形の下（紙の外）にある
    cx, cy = W / 2, H / 2 + ((r_out + r_in * math.cos(half)) / 2) * ppm
    yy, xx = np.mgrid[0:H, 0:W]
    x, y = (xx - cx) / ppm, (cy - yy) / ppm
    c = _gate_ring_color(np.hypot(x, y), inner_mm, module_mm)
    c[np.abs(np.arctan2(x, y)) > half] = -1
    page = np.where(c < 0, 255, c).astype(np.uint8)
    # 切り取り線（灰色）。扇形の外周・内周と両端
    t = np.linspace(-half, half, 200)
    outline = np.concatenate([np.stack([np.sin(t), np.cos(t)], 1) * r_out,
                              np.stack([np.sin(t[::-1]), np.cos(t[::-1])], 1) * r_in])
    pts = np.stack([cx + outline[:, 0] * ppm, cy - outline[:, 1] * ppm], 1).astype(np.int32)
    cv2.polylines(page, [pts], True, 160, max(1, int(0.3 * ppm)), cv2.LINE_AA)
    pages = []
    for k in range(n):
        img = page.copy()
        label = f"{k + 1}/{n}   gate inner {inner_mm:g}mm   module {module_mm:g}mm   print 100%"
        cv2.putText(img, label, (int(margin * ppm), H - int(margin * ppm / 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.012 * dpi, 0, max(1, dpi // 100), cv2.LINE_AA)
        pages.append(img)
    return pages, n


def write_markers(out: str, module_px: int = 80, height_modules: int = 16, gate_inner: float = 600,
                  gate_module: float = 15, dpi: int = 200) -> list[str]:
    """左右の PNG・円形ゲートの PDF・その見本を out に書き、書いたファイルのパスを返す。"""
    os.makedirs(out, exist_ok=True)
    paths = []
    for name, pattern in (("left", FGM_LEFT), ("right", FGM_RIGHT)):
        path = os.path.join(out, f"marker_fgm_{name}.png")
        cv2.imwrite(path, fgm_marker(pattern, module_px, height_modules))
        paths.append(path)
    from PIL import Image
    pages, _ = gate_ring_pages(gate_inner, gate_module, dpi)
    path = os.path.join(out, "marker_fgm_ring.pdf")
    imgs = [Image.fromarray(pg) for pg in pages]
    imgs[0].save(path, "PDF", resolution=dpi, save_all=True, append_images=imgs[1:])
    paths.append(path)
    path = os.path.join(out, "marker_fgm_ring_preview.png")
    cv2.imwrite(path, gate_ring_image(400, 400 / gate_inner * gate_module))   # 全体の見本（縮小）
    paths.append(path)
    return paths


def main(argv=None):
    p = argparse.ArgumentParser(description="FGM の印刷用画像を作る")
    p.add_argument("--module", type=int, default=80, help="FGM の1マスの画素数")
    p.add_argument("--height", type=int, default=16, help="FGM の高さ（マス）")
    p.add_argument("--gate-inner", type=float, default=600, help="円形ゲートの内径（mm）。輪はこの外側に貼る")
    p.add_argument("--gate-module", type=float, default=15, help="円形ゲートの輪の1マス（mm）")
    p.add_argument("--dpi", type=int, default=200, help="円形ゲート用 PDF の解像度")
    p.add_argument("--out", default="markers")
    a = p.parse_args(argv)
    try:
        paths = write_markers(a.out, a.module, a.height, a.gate_inner, a.gate_module, a.dpi)
    except ValueError as e:
        raise SystemExit(str(e))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
