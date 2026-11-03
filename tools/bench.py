"""ArUco と FpvGateMarker（FGM）の性能を比べる（処理時間・認識率・誤検出）。

    python tools/bench.py                         # 合成映像での処理時間と認識率
    python tools/bench.py --videos a.mp4 b.mp4    # 加えて、マーカーの無い動画で処理時間と誤検出を数える

認識率は合成映像（640×480、背景は滑らかな模様）にマーカーを置いて測る。大きさは
「黒い部分の幅」（ArUco は1辺、FGM は黒5本の幅）を px でそろえる。設定は既定値（最小サイズ 5%）。
誤検出は、動画の全フレームで「見えた」とした数と、通過の数（見えた→100ms 見えない。= 誤ったラップ）。
ArUco は DICT_4X4_50 の ID 0〜3（TinyViewPlus のマーカーと同じ）。
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fgm import FGM_LEFT, FGM_RIGHT, FgmDetector  # noqa: E402
from fgm.markers import fgm_marker, gate_ring_image  # noqa: E402

W, H = 640, 480
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


class Detection:
    def __init__(self, visible, corners):
        self.visible, self.corners = visible, corners


class MarkerDetector:
    """比べる相手の ArUco。ID 0〜3 で、1辺がフレーム高さの min_size_pct% 以上なら「見えた」。"""

    def __init__(self, ids=(0, 1, 2, 3), min_size_pct=5.0):
        self.detector = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
        self.ids, self.min_size_pct = set(ids), min_size_pct

    def detect(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return Detection(False, [])
        ok = [c.reshape(4, 2) for c, i in zip(corners, ids.flatten()) if int(i) in self.ids]
        ok = [p for p in ok if np.mean([np.linalg.norm(p[k] - p[(k + 1) % 4]) for k in range(4)])
              >= frame.shape[0] * self.min_size_pct / 100]
        return Detection(bool(ok), ok)


class PassCounter:
    """見えた → flicker_ms 見えない、で1回の通過と数える。"""

    def __init__(self, flicker_ms=100):
        self.flicker, self.last = flicker_ms / 1000, None

    def update(self, t, visible):
        if visible:
            self.last = t
        elif self.last is not None and t - self.last >= self.flicker:
            self.last = None
            return True
        return False

# (名前, ボケσ, ノイズσ, 櫛状のずれpx)
QUALITY = [("きれい", 0, 0, 0), ("ボケ1+ノイズ8", 1.0, 8, 0),
           ("ボケ1.5+ノイズ12+櫛3", 1.5, 12, 3), ("ボケ2+ノイズ15+櫛6", 2.0, 15, 6)]
# (名前, 傾き°, 横の縮み) 横の縮み = 斜めから見たとき
POSE = [("正面", 0, 1.0), ("傾き30°", 30, 1.0), ("傾き60°", 60, 1.0), ("斜め(横0.6)", 0, 0.6)]


def background(rng):
    return cv2.GaussianBlur(rng.integers(60, 200, (H, W)).astype(np.uint8), (0, 0), 8)


def paste(img, m, mask, cx, cy, angle, squash):
    """m（mask の所だけ）を、横に squash 倍・angle 度回して (cx, cy) に置く。"""
    m = cv2.resize(m, None, fx=squash, fy=1, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
    M = cv2.getRotationMatrix2D((m.shape[1] / 2, m.shape[0] / 2), angle, 1)
    M[0, 2] += cx - m.shape[1] / 2
    M[1, 2] += cy - m.shape[0] / 2
    mw = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST)
    img[mw > 0] = cv2.warpAffine(m, M, (W, H))[mw > 0]
    return np.count_nonzero(mw) / max(1, np.count_nonzero(mask))   # 画面に入った割合


def degrade(img, blur, noise, comb, rng):
    img = img.astype(np.float32)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=blur, sigmaY=0.3)   # アナログは横のボケが強い
    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    if comb:
        img[1::2] = np.roll(img[1::2], comb, axis=1)                   # インターレースの櫛状のずれ
    return img


def scene_aruco(width, angle, squash, rng):
    img = background(rng)
    cell = width / 6.0
    m = cv2.aruco.generateImageMarker(DICT, int(rng.integers(0, 4)), 600)
    m = cv2.copyMakeBorder(m, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)   # 白1マス
    m = cv2.resize(m, None, fx=cell / 100, fy=cell / 100, interpolation=cv2.INTER_AREA)
    paste(img, m, np.full_like(m, 255), W / 2 + rng.uniform(-60, 60), H / 2 + rng.uniform(-40, 40), angle, squash)
    return img


def scene_fgm_posts(width, angle, squash, rng):
    """柱に貼った FGM 左右（高さ16マス）。柱の中心の間は FGM の幅の5倍。左右が画面に入らなければ None。"""
    img = background(rng)
    mod = width / 7.0
    gap = width * 5 * squash
    cx, cy = W / 2 + rng.uniform(-30, 30), H / 2 + rng.uniform(-30, 30)
    rad = np.deg2rad(angle)
    for pat, sgn in ((FGM_LEFT, -1), (FGM_RIGHT, 1)):
        m = fgm_marker(pat, 20, 16)
        m = cv2.resize(m, None, fx=mod / 20, fy=mod / 20, interpolation=cv2.INTER_AREA)
        # 機体が傾くと、左右の柱は中心のまわりに回る
        dx, dy = sgn * gap / 2 * np.cos(rad), -sgn * gap / 2 * np.sin(rad)
        if paste(img, m, np.full_like(m, 255), cx + dx, cy + dy, angle, squash) < 0.98:
            return None
    return img


def scene_fgm_ring(width, angle, squash, rng):
    """円形ゲートの FGM の輪（内径600mm・1マス15mm の比）。"""
    img = background(rng)
    mod = width / 7.0
    g = gate_ring_image(int(600 / 15 * mod), mod)
    if g.shape[0] > H - 10:
        return None                  # 輪が画面に入らない
    paste(img, g, (g != 128).astype(np.uint8) * 255, W / 2 + rng.uniform(-20, 20),
          H / 2 + rng.uniform(-20, 20), angle, squash)
    return img


def recognition(detectors, widths, trials, seed=0):
    rng = np.random.default_rng(seed)
    results = {}
    for name, (det, scene) in detectors.items():
        rows = []
        for width in widths:
            row = []
            for _, angle, squash in POSE:
                for _, blur, noise, comb in QUALITY:
                    ok = 0
                    for _ in range(trials):
                        img = scene(width, angle, squash, rng)
                        if img is None:
                            ok = np.nan
                            break
                        if hasattr(det, "reset"):
                            det.reset()      # FGM は前のフレームの状態を持つ
                        ok += det.detect(degrade(img, blur, noise, comb, rng)).visible
                    row.append(ok / trials)
            rows.append(row)
        results[name] = np.array(rows)
    return results


def speed(det, frames, repeat=1):
    t = time.perf_counter()
    for _ in range(repeat):
        for f in frames:
            det.detect(f)
    return (time.perf_counter() - t) / (repeat * len(frames)) * 1000


def video_frames(path, fps=30):
    """動画を 640×480・30fps 相当で読む。"""
    c = cv2.VideoCapture(path)
    step = max(1, round((c.get(cv2.CAP_PROP_FPS) or fps) / fps))
    i = 0
    while True:
        ok, f = c.read()
        if not ok:
            return
        i += 1
        if i % step == 0:
            yield cv2.cvtColor(cv2.resize(f, (W, H)), cv2.COLOR_BGR2GRAY)


def false_positives(dets, videos, save_dir=None):
    stats = {k: {"frames": 0, "visible": 0, "passes": 0, "ms": 0.0} for k in dets}
    for path in videos:
        timers = {k: PassCounter(100) for k in dets}
        for det in dets.values():
            if hasattr(det, "reset"):
                det.reset()
        for n, f in enumerate(video_frames(path)):
            for k, det in dets.items():
                t = time.perf_counter()
                d = det.detect(f)
                s = stats[k]
                s["ms"] += (time.perf_counter() - t) * 1000
                s["frames"] += 1
                s["visible"] += d.visible
                s["passes"] += timers[k].update(n / 30, d.visible)
                if d.visible and save_dir:
                    img = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                    for pts in d.corners:
                        cv2.polylines(img, [np.array(pts, np.int32)], True, (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(save_dir, f"{k}_{stats[k]['visible']:04d}.jpg"), img)
        print(f"  {os.path.basename(path)}", flush=True)
    return stats


def _pct(v):
    return f"{v * 100:15.0f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", nargs="*", default=[], help="マーカーの無い動画（誤検出と処理時間を数える）")
    p.add_argument("--trials", type=int, default=10, help="認識率の条件ごとの試行回数")
    p.add_argument("--save", help="誤検出したフレームを保存するフォルダ")
    a = p.parse_args()
    if a.save:
        os.makedirs(a.save, exist_ok=True)

    widths = [16, 20, 24, 30, 40, 60, 100, 160]
    dets = {"ArUco": (MarkerDetector(), scene_aruco),
            "FGM(柱)": (FgmDetector(), scene_fgm_posts),
            "FGM(円形ゲート)": (FgmDetector(), scene_fgm_ring)}

    print("== 処理時間（1フレーム, ms。640×480）==")
    rng = np.random.default_rng(1)
    blank = [background(rng) for _ in range(20)]
    for name, (det, scene) in dets.items():
        with_marker = [degrade(scene(40, 0, 1.0, rng), 1.0, 8, 0, rng) for _ in range(20)]
        print(f"  {name:14s} マーカー無し {speed(det, blank, 5):5.1f}   マーカー有り {speed(det, with_marker, 5):5.1f}")

    print(f"\n== 認識率（各条件 {a.trials} 回）==")
    res = recognition(dets, widths, a.trials)
    print("  条件ごと（画面に入る大きさの平均）")
    print("  " + " " * 34 + "".join(f"{n:>16s}" for n in dets))
    for j, (pose, *_ ) in enumerate([(pn, q) for pn, *_ in POSE for q in QUALITY]):
        pn = POSE[j // len(QUALITY)][0]
        qn = QUALITY[j % len(QUALITY)][0]
        vals = "".join(_pct(np.nanmean(res[n][:, j])) for n in dets)
        print(f"  {pn:10s} {qn:22s}{vals}")
    common = [i for i, wpx in enumerate(widths) if wpx in (30, 40)]
    print("\n  条件ごと（3つとも画面に入る 30px・40px だけの平均）")
    print("  " + " " * 34 + "".join(f"{n:>16s}" for n in dets))
    for j in range(len(POSE) * len(QUALITY)):
        pn, qn = POSE[j // len(QUALITY)][0], QUALITY[j % len(QUALITY)][0]
        print(f"  {pn:10s} {qn:22s}" + "".join(_pct(np.nanmean(res[n][common, j])) for n in dets))
    print("\n  大きさごと（黒い部分の幅 px。全部の条件の平均）")
    print("  " + " " * 8 + "".join(f"{n:>16s}" for n in dets))
    for i, wpx in enumerate(widths):
        print(f"  {wpx:4d}px " + "".join(_pct(np.nanmean(res[n][i])) if not np.isnan(res[n][i]).all()
                                         else f"{'画面外':>13s}" for n in dets))

    if a.videos:
        print("\n== 誤検出（マーカーの無い動画）==")
        fp_dets = {"ArUco": MarkerDetector(), "FGM": FgmDetector()}
        st = false_positives(fp_dets, a.videos, a.save)
        for k, s in st.items():
            mins = s["frames"] / 30 / 60
            print(f"  {k:6s} {s['frames']}フレーム（{mins:.1f}分）: 見えた {s['visible']}フレーム"
                  f"（{s['visible'] / s['frames'] * 100:.2f}%）、誤った通過 {s['passes']}回"
                  f"（{s['passes'] / mins:.2f}回/分）、処理時間 平均 {s['ms'] / s['frames']:.1f}ms")


if __name__ == "__main__":
    main()
