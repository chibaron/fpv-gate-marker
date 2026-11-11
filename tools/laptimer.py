"""Lap timer for testing: run camera (or video) frames through FgmDetector and time the laps.

    python tools/laptimer.py                      # find cameras; choose one if there are several
    python tools/laptimer.py --camera 1           # camera 1 (e.g. an analog video capture device)
    python tools/laptimer.py --list-cameras       # list the cameras found
    python tools/laptimer.py --camera a.mp4       # a video file (times follow the video's own clock)
    python tools/laptimer.py --camera a.mp4 --headless --log laps.csv

A pass is counted when the gate goes from visible to not visible and stays not visible for --flicker-ms.
The pass time is the last frame the gate was visible. Laps are the times between passes; the first pass starts
the timer. Passes closer than --min-lap seconds to the previous one are ignored.
Each pass beeps (--no-beep to turn it off).
Keys in the window: q / Esc quit, r reset laps.
Without --camera, cameras 0-9 are probed. If several are found, a window shows live video from each and you press
its number (with --headless, type the number in the terminal).

テスト用のラップタイマー。カメラ（または動画）の映像を FgmDetector に通してラップを計る。

    python tools/laptimer.py                      # カメラを探す。複数あれば選ぶ
    python tools/laptimer.py --camera 1           # カメラ 1（アナログ映像のキャプチャー機器など）
    python tools/laptimer.py --list-cameras       # 見つかったカメラを並べる
    python tools/laptimer.py --camera a.mp4       # 動画ファイル（時刻は動画の中の時間）
    python tools/laptimer.py --camera a.mp4 --headless --log laps.csv

ゲートが「見えた」→「見えない」になり、--flicker-ms のあいだ見えないままなら1回の通過とする。
通過の時刻は、最後に見えたフレームの時刻。ラップは通過と通過の間で、最初の通過で計り始める。
前の通過から --min-lap 秒以内の通過は数えない。
通過のたびに beep を鳴らす（--no-beep で鳴らさない）。
窓のキー: q / Esc で終わる、r でラップを消す。
--camera が無ければカメラ 0〜9 を探す。複数見つかったら、それぞれの映像を動画で並べた窓を出すので、番号のキーで選ぶ
（--headless のときはターミナルで番号を入れる）。
"""
import argparse
import csv
import os
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fgm import FGM_TOL, FgmDetector  # noqa: E402

COLORS = {"L": (255, 200, 0), "R": (255, 0, 255)}   # BGR: left cyan-ish, right magenta / 左は水色、右は紫


class LapTimer:
    """Turns per-frame visible flags into passes and laps.

    フレームごとの「見えた」を、通過とラップに変える。"""

    def __init__(self, flicker_ms=100, min_lap_s=2.0):
        self.flicker = flicker_ms / 1000
        self.min_lap = min_lap_s
        self.reset()

    def reset(self):
        self.last_visible = None    # time the gate was last seen (None = no pass pending) / 最後に見えた時刻（None なら通過待ちでない）
        self.passes = []            # pass times (s) / 通過の時刻（秒）
        self.laps = []              # lap times (s) / ラップ（秒）

    def update(self, t, visible):
        """Feed one frame at time t (s). Returns the pass time when a pass is counted, otherwise None.

        時刻 t（秒）のフレームを1つ入れる。通過を数えたらその時刻を、それ以外は None を返す。"""
        if visible:
            self.last_visible = t
            return None
        if self.last_visible is None or t - self.last_visible < self.flicker:
            return None
        tp, self.last_visible = self.last_visible, None
        if self.passes and tp - self.passes[-1] < self.min_lap:
            return None                 # too soon after the previous pass / 前の通過から近すぎる
        if self.passes:
            self.laps.append(tp - self.passes[-1])
        self.passes.append(tp)
        return tp

    def running(self, t):
        """Time since the last pass (the current lap), or None before the first pass.

        前の通過からの時間（今のラップ）。最初の通過の前は None。"""
        return t - self.passes[-1] if self.passes else None


def _open_camera(index):
    # DirectShow opens cameras much faster than the default on Windows
    # Windows では DirectShow のほうがカメラを開くのがずっと速い
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    return cv2.VideoCapture(index, backend)


def find_cameras(max_index=10):
    """Probe camera indices 0..max_index-1. Returns [(index, open capture, first frame), ...] for those that
    give a frame. The captures stay open; release the ones you do not use.

    カメラ番号 0〜max_index-1 を試し、映像が取れたものを [(番号, 開いた capture, 最初の画像), ...] で返す。
    capture は開いたまま。使わないものは release する。"""
    found = []
    for i in range(max_index):
        cap = _open_camera(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if ok and frame is not None:
            found.append((i, cap, frame))
        else:
            cap.release()
    return found


def _camera_grid(tiles_src):
    """Tile [(index, frame), ...] into one image with index and resolution labels.

    [(番号, 画像), ...] を、番号と解像度を付けて1枚に並べる。"""
    tiles = []
    for i, frame in tiles_src:
        t = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        cv2.rectangle(t, (0, 0), (320, 30), (0, 0, 0), -1)
        cv2.putText(t, f"[{i}]  {w}x{h}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(t)
    cols = min(3, len(tiles))
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    grid = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
    bar = np.zeros((30, grid.shape[1], 3), np.uint8)
    cv2.putText(bar, "Press the camera number (q / Esc to quit)", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)
    return np.vstack([bar, grid])


def choose_camera(found, headless):
    """Let the user pick one of several cameras. Returns the index, or None if cancelled.
    The window shows live video from every camera.

    複数のカメラから1つ選んでもらう。番号を返す。やめたら None。窓には全部のカメラの映像を動画で出す。"""
    indices = [i for i, _, _ in found]
    if headless:
        while True:
            s = input(f"Choose a camera {indices} (empty to quit): ").strip()
            if not s:
                return None
            if s.isdigit() and int(s) in indices:
                return int(s)
    last = {i: frame for i, _, frame in found}    # keep the last frame if a camera skips a read / 読めなかったら前の画像のまま
    title = "Select camera"
    try:
        while True:
            for i, cap, _ in found:
                ok, frame = cap.read()
                if ok and frame is not None:
                    last[i] = frame
            cv2.imshow(title, _camera_grid([(i, last[i]) for i in indices]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return None
            if ord("0") <= key <= ord("9") and key - ord("0") in indices:
                return key - ord("0")
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None         # window closed / 窓を閉じた
    finally:
        cv2.destroyWindow(title)


def beep(freq=1000, ms=150):
    """Beep without blocking the detection loop (winsound.Beep blocks, so it runs in a thread).

    検出の処理を止めずに beep を鳴らす（winsound.Beep は鳴り終わるまで戻らないのでスレッドで鳴らす）。"""
    if sys.platform == "win32":
        import winsound
        threading.Thread(target=winsound.Beep, args=(freq, ms), daemon=True).start()
    else:
        print("\a", end="", flush=True)     # terminal bell / 端末のベル


def open_source(src):
    """Open a camera index or a video file / URL. Returns (capture, is_file).

    カメラ番号か、動画ファイル・URL を開く。(capture, ファイルか) を返す。"""
    if src.isdigit():
        return _open_camera(int(src)), False
    return cv2.VideoCapture(src), True


def draw(img, d, timer, t, det_ms, fps):
    """Draw detections, state and laps onto img.

    検出した枠・状態・ラップを img に描く。"""
    for pts, name in zip(d.corners, d.ids):
        p = np.array(pts, np.int32)
        cv2.polylines(img, [p], True, COLORS[name], 2)
        cv2.putText(img, name, tuple(p[0] + [0, -4]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[name], 2)
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 28), (0, 0, 0), -1)
    state, color = ("GATE", (0, 255, 0)) if d.visible else ("----", (160, 160, 160))
    cv2.putText(img, state, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    run = timer.running(t)
    info = f"lap {run:7.2f}s" if run is not None else "waiting for first pass"
    cv2.putText(img, f"{info}   det {det_ms:4.1f}ms   {fps:4.1f}fps", (80, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    best = min(timer.laps) if timer.laps else None
    if timer.laps:
        cv2.rectangle(img, (w - 140, 32), (w, 36 + 20 * min(10, len(timer.laps))), (0, 0, 0), -1)
    for k, lap in enumerate(timer.laps[-10:]):
        n = len(timer.laps) - min(10, len(timer.laps)) + k + 1
        c = (0, 255, 255) if lap == best else (255, 255, 255)
        cv2.putText(img, f"{n:3d}  {lap:7.3f}", (w - 130, 50 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)


def summary(timer, det_times):
    print(f"\npasses {len(timer.passes)}, laps {len(timer.laps)}")
    if timer.laps:
        print(f"best {min(timer.laps):.3f}s, average {np.mean(timer.laps):.3f}s")
    if det_times:
        print(f"detection {np.mean(det_times):.2f} ms average, {np.max(det_times):.2f} ms max "
              f"({len(det_times)} frames)")


def main(argv=None):
    p = argparse.ArgumentParser(description="Lap timer for testing FGM with a camera or video")
    p.add_argument("--camera", help="camera index, or a video file / URL (default: find cameras and choose)")
    p.add_argument("--list-cameras", action="store_true", help="list the cameras found and exit")
    p.add_argument("--size", default="640x480",
                   help="resize frames to WxH before detection (default 640x480; 'none' keeps the original)")
    p.add_argument("--flicker-ms", type=float, default=100, help="not-visible time that confirms a pass (ms)")
    p.add_argument("--min-lap", type=float, default=2.0, help="ignore passes closer than this to the previous (s)")
    p.add_argument("--line-step", type=int, default=4, help="FgmDetector line_step")
    p.add_argument("--min-lines", type=int, default=3, help="FgmDetector min_lines")
    p.add_argument("--min-size-pct", type=float, default=5.0, help="FgmDetector min_size_pct")
    p.add_argument("--tol", type=float, default=FGM_TOL, help="FgmDetector tol (tolerance in modules)")
    p.add_argument("--threads", type=int, default=1, help="cv2.setNumThreads (1 is fastest for small frames)")
    p.add_argument("--log", help="write passes and laps to this CSV file")
    p.add_argument("--headless", action="store_true", help="no window; print laps only")
    p.add_argument("--no-beep", dest="beep", action="store_false", help="do not beep on each pass")
    a = p.parse_args(argv)

    size = None if a.size.lower() == "none" else tuple(int(v) for v in a.size.lower().split("x"))
    cv2.setNumThreads(a.threads)
    cap = None
    if a.camera is None or a.list_cameras:
        found = find_cameras()
        for i, _, frame in found:
            print(f"camera {i}: {frame.shape[1]}x{frame.shape[0]}")
        choice = None
        if not a.list_cameras and found:
            choice = found[0][0] if len(found) == 1 else choose_camera(found, a.headless)
        # Keep the chosen camera open and use it as is; close the others
        # 選んだカメラは開いたまま使い、ほかは閉じる
        for i, c, _ in found:
            if i == choice:
                cap = c
            else:
                c.release()
        if a.list_cameras:
            return
        if not found:
            raise SystemExit("No camera found. Use --camera to give an index, a video file or a URL.")
        if choice is None:
            return
        a.camera = str(choice)
        print(f"using camera {a.camera}", flush=True)
    if cap is None:
        cap, is_file = open_source(a.camera)
    else:
        is_file = False
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {a.camera}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    det = FgmDetector(a.line_step, a.min_lines, a.min_size_pct, a.tol)
    timer = LapTimer(a.flicker_ms, a.min_lap)
    log = None
    if a.log:
        log = open(a.log, "w", newline="")
        writer = csv.writer(log)
        writer.writerow(["pass", "time_s", "lap_s"])
    det_times, frame_n = [], 0
    t0 = time.perf_counter()
    last_frame_wall, fps = t0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.perf_counter()
            # Files use the video clock so laps match the recording; cameras use the wall clock
            # 動画は動画の中の時刻（録画どおりのラップになる）、カメラは実際の時刻
            t = frame_n / video_fps if is_file else now - t0
            frame_n += 1
            fps = 0.9 * fps + 0.1 / max(now - last_frame_wall, 1e-6) if frame_n > 1 else 0.0
            last_frame_wall = now
            if size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            s = time.perf_counter()
            d = det.detect(frame)
            det_ms = (time.perf_counter() - s) * 1000
            det_times.append(det_ms)
            tp = timer.update(t, d.visible)
            if tp is not None:
                if a.beep:
                    beep()
                lap = timer.laps[-1] if len(timer.passes) > 1 else None
                n = len(timer.passes)
                print(f"pass {n:3d} at {tp:8.3f}s" + (f"   lap {len(timer.laps):3d}  {lap:7.3f}s" if lap else "   start"),
                      flush=True)
                if log:
                    writer.writerow([n, f"{tp:.3f}", f"{lap:.3f}" if lap else ""])
                    log.flush()
            if a.headless:
                continue
            draw(frame, d, timer, t, det_ms, fps)
            cv2.imshow("FGM lap timer", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                timer.reset()
                print("reset", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if log:
            log.close()
        if not a.headless:
            cv2.destroyAllWindows()
    summary(timer, det_times)


if __name__ == "__main__":
    main()
