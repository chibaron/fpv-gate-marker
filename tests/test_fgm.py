"""FGM tests (synthetic video).

    python -m pytest tests  or  python tests/test_fgm.py

FGM の動作確認（合成映像）。

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
    """Left and right FGM (height modules tall, centered at xs). With swap, left and right are exchanged (wrong way round).

    左右の FGM（高さ height マス、中心 xs）。swap なら左右を入れ替えて貼る（向きが逆）。"""
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

    def fresh(img):          # read without previous frame state / 前のフレームの状態なしで読む
        det.reset()
        return det.detect(img).visible

    d = det.detect(frame_with_fgm())
    assert d.visible and sorted(d.ids) == ["L", "R"] and len(d.corners) == 2
    assert fresh(frame_with_fgm(angle=30))                            # readable when tilted / 機体が傾いても読める
    assert not fresh(frame_with_fgm(right=False))                     # one side alone is not visible / 片方だけでは見えたとしない
    assert not fresh(frame_with_fgm(left=False))
    assert not fresh(frame_with_fgm(swap=True))                       # left and right swapped / 左右が逆
    assert not fresh(frame_with_fgm(height=4))                        # too short (OSD characters etc.) / 背が低い（OSD の文字など）
    assert not fresh(frame_with_fgm(xs=(250, 330)))                   # too close together (not a gate) / 左右が近すぎる（ゲートでない）
    assert fresh(frame_with_fgm(xs=(150, 500)))
    assert not fresh(blank())
    assert not fresh(frame_with_aruco())                              # ArUco is not read as FGM / ArUco は FGM として読まない
    assert not FgmDetector(min_size_pct=40).detect(frame_with_fgm()).visible   # too small / 小さすぎ
    assert not FgmDetector(min_lines=100).detect(frame_with_fgm()).visible     # not enough lines / 縦の本数不足
    img = frame_with_fgm()
    img[::2] = 90                                                     # line noise makes half the rows unreadable / 横線のノイズで半分の行が読めない
    assert FgmDetector(line_step=2).detect(img).visible
    assert FgmDetector().detect(cv2.cvtColor(frame_with_fgm(), cv2.COLOR_BGR2GRAY)).visible   # grayscale input too / グレーでも


def test_hysteresis():
    """Once visible, stays visible until both sides are gone (when one side leaves the frame as you approach).

    見えた後は、左右の両方とも見えなくなるまで「見えた」のまま（近づいて片方が画面から出たとき）。"""
    det = FgmDetector()
    assert det.detect(frame_with_fgm()).visible
    assert det.detect(frame_with_fgm(right=False)).visible
    assert det.detect(frame_with_fgm(left=False)).visible
    assert not det.detect(blank()).visible
    assert not det.detect(frame_with_fgm(left=False)).visible         # one side is not enough until both are seen / 両方見えるまでは片方では見えない
    assert det.detect(frame_with_fgm()).visible
    det.reset()
    assert not det.detect(frame_with_fgm(left=False)).visible


def test_gate_ring():
    """The round gate ring reads as FGM (left and right). Being a circle, it reads when tilted.

    円形ゲートに貼る輪は、FGM（左右）として読める。円なので機体が傾いても読める。"""
    det = FgmDetector()
    g = gate_ring_image(300, 4.5)                    # 600 mm inner, 15 mm module at 0.5 px/mm / 内径600mm・1マス15mm を 0.5px/mm で
    for angle in (0, 35, 70):
        img = np.full((480, 640), 90, np.uint8)
        m = cv2.warpAffine(g, cv2.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), angle, 1),
                           (g.shape[1], g.shape[0]), borderValue=128)
        y, x = (480 - m.shape[0]) // 2, (640 - m.shape[1]) // 2
        roi = img[y:y + m.shape[0], x:x + m.shape[1]]
        roi[m != 128] = m[m != 128]
        det.reset()
        assert det.detect(cv2.GaussianBlur(img, (0, 0), 1.0)).visible, angle
    pages, n = gate_ring_pages(600, 15, 50)          # split into arc segments that fit on A4 / A4 に入る扇形に分ける
    assert n >= 3 and len(pages) == n and max(pages[0].shape) == int(297 * 50 / 25.4)


def test_lap_timer():
    """tools/laptimer.py: a pass is the last visible time, confirmed after flicker_ms; short gaps are not passes.

    tools/laptimer.py: 通過は最後に見えた時刻で、flicker_ms 見えなければ決まる。短い途切れは通過にしない。"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    from laptimer import LapTimer

    timer = LapTimer(flicker_ms=100, min_lap_s=2.0)
    seq = [(0.0, True), (0.5, True), (0.55, False), (0.6, True),     # 50 ms gap is not a pass / 50ms の途切れは通過でない
           (1.0, True), (1.05, False), (1.2, False),                 # pass at 1.0 / 1.0 に通過
           (2.0, True), (2.1, False), (2.3, False),                  # 1.0 s after the previous: ignored / 前から1秒: 数えない
           (6.0, True), (6.2, False),                                # pass at 6.0, lap 5.0 / 6.0 に通過、ラップ 5.0
           (11.5, True), (11.7, False)]                           # pass at 11.5, lap 5.5 / 11.5 に通過、ラップ 5.5
    passes = [tp for t, v in seq if (tp := timer.update(t, v)) is not None]
    assert passes == [1.0, 6.0, 11.5] and timer.laps == [5.0, 5.5]
    assert timer.running(13.0) == 1.5
    timer.reset()
    assert timer.passes == [] and timer.running(1.0) is None


def test_write_markers():
    with tempfile.TemporaryDirectory() as d:
        paths = write_markers(d, module_px=4, dpi=50)
        assert [os.path.basename(p) for p in paths] == [
            "marker_fgm_left.png", "marker_fgm_right.png", "marker_fgm_ring.pdf", "marker_fgm_ring_preview.png"]
        assert all(os.path.getsize(p) > 0 for p in paths)
        left = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
        assert left.shape == (16 * 4, (sum(FGM_LEFT) + 4) * 4)       # 2-module margins, 16 modules tall / 余白2マスずつ・高さ16マス


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
