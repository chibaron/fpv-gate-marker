"""FGM の動作確認（合成映像）。

    python -m pytest tests  または  python tests/test_fgm.py
"""
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fgm import FGM_LEFT, FGM_RIGHT, FgmDetector   # noqa: E402
from fgm.markers import fgm_marker, gate_ring_image, gate_ring_pages, write_markers  # noqa: E402


def frame_with_fgm(left=True, right=True, module=5, angle=0.0, swap=False, height=20, xs=(180, 460),
                   w=640, h=480):
    """左右の FGM（高さ height マス、中心 xs）。swap なら左右を入れ替えて貼る（向きが逆）。"""
    img = np.full((h, w), 90, np.uint8)
    pats = [(FGM_LEFT, left, xs[0]), (FGM_RIGHT, right, xs[1])]
    if swap:
        pats = [(FGM_RIGHT, right, xs[0]), (FGM_LEFT, left, xs[1])]
    for pat, on, cx in pats:
        if on:
            m = fgm_marker(pat, module, height)
            y, x = (h - m.shape[0]) // 2, cx - m.shape[1] // 2
            img[y:y + m.shape[0], x:x + m.shape[1]] = m
    if angle:
        img = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1), (w, h),
                             borderValue=90)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def blank(w=640, h=480):
    return np.full((h, w, 3), 90, np.uint8)


def frame_with_aruco(ids=(0, 1), size=120, w=640, h=480):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = blank(w, h)
    for k, i in enumerate(ids):
        m = cv2.aruco.generateImageMarker(d, i, size)
        m = cv2.copyMakeBorder(m, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        y, x = 100, 60 + k * 260
        img[y:y + m.shape[0], x:x + m.shape[1]] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    return img


def test_detector():
    det = FgmDetector(line_step=4, min_lines=3, min_size_pct=5)

    def fresh(img):          # 前のフレームの状態なしで読む
        det.reset()
        return det.detect(img).visible

    d = det.detect(frame_with_fgm())
    assert d.visible and sorted(d.ids) == ["L", "R"] and len(d.corners) == 2
    assert fresh(frame_with_fgm(angle=30))                            # 機体が傾いても読める
    assert not fresh(frame_with_fgm(right=False))                     # 片方だけでは見えたとしない
    assert not fresh(frame_with_fgm(left=False))
    assert not fresh(frame_with_fgm(swap=True))                       # 左右が逆
    assert not fresh(frame_with_fgm(height=4))                        # 背が低い（OSD の文字など）
    assert not fresh(frame_with_fgm(xs=(250, 330)))                   # 左右が近すぎる（ゲートでない）
    assert fresh(frame_with_fgm(xs=(150, 500)))
    assert not fresh(blank())
    assert not fresh(frame_with_aruco())                              # ArUco は FGM として読まない
    assert not FgmDetector(min_size_pct=40).detect(frame_with_fgm()).visible   # 小さすぎ
    assert not FgmDetector(min_lines=100).detect(frame_with_fgm()).visible     # 縦の本数不足
    img = frame_with_fgm()
    img[::2] = 90                                                     # 横線のノイズで半分の行が読めない
    assert FgmDetector(line_step=2).detect(img).visible
    assert FgmDetector().detect(cv2.cvtColor(frame_with_fgm(), cv2.COLOR_BGR2GRAY)).visible   # グレーでも


def test_hysteresis():
    """見えた後は、左右の両方とも見えなくなるまで「見えた」のまま（近づいて片方が画面から出たとき）。"""
    det = FgmDetector()
    assert det.detect(frame_with_fgm()).visible
    assert det.detect(frame_with_fgm(right=False)).visible
    assert det.detect(frame_with_fgm(left=False)).visible
    assert not det.detect(blank()).visible
    assert not det.detect(frame_with_fgm(left=False)).visible         # 両方見えるまでは片方では見えない
    assert det.detect(frame_with_fgm()).visible
    det.reset()
    assert not det.detect(frame_with_fgm(left=False)).visible


def test_gate_ring():
    """円形ゲートに貼る輪は、FGM（左右）として読める。円なので機体が傾いても読める。"""
    det = FgmDetector()
    g = gate_ring_image(300, 4.5)                    # 内径600mm・1マス15mm を 0.5px/mm で
    for angle in (0, 35, 70):
        img = np.full((480, 640), 90, np.uint8)
        m = cv2.warpAffine(g, cv2.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), angle, 1),
                           (g.shape[1], g.shape[0]), borderValue=128)
        y, x = (480 - m.shape[0]) // 2, (640 - m.shape[1]) // 2
        roi = img[y:y + m.shape[0], x:x + m.shape[1]]
        roi[m != 128] = m[m != 128]
        det.reset()
        assert det.detect(cv2.GaussianBlur(img, (0, 0), 1.0)).visible, angle
    pages, n = gate_ring_pages(600, 15, 50)          # A4 に入る扇形に分ける
    assert n >= 3 and len(pages) == n and max(pages[0].shape) == int(297 * 50 / 25.4)


def test_write_markers():
    with tempfile.TemporaryDirectory() as d:
        paths = write_markers(d, module_px=4, dpi=50)
        assert [os.path.basename(p) for p in paths] == [
            "marker_fgm_left.png", "marker_fgm_right.png", "marker_fgm_ring.pdf", "marker_fgm_ring_preview.png"]
        assert all(os.path.getsize(p) > 0 for p in paths)
        left = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
        assert left.shape == (16 * 4, (sum(FGM_LEFT) + 4) * 4)       # 余白2マスずつ・高さ16マス


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
