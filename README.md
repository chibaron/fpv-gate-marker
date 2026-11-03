<div align="right"><a href="./README_jp.md">日本語はこちら</a></div>

# FpvGateMarker (FGM)

A striped marker for FPV drone race gates, and a detector (Python / OpenCV) that finds it in the onboard camera feed.

## Features

- One marker on each side post of a gate. For round gates, a ring of stripes goes all the way around.
- Easy to make: it is just black and white vertical stripes. No printer needed — black tape on a white board works.
- Lightweight: it only reads a few sparse horizontal lines (about 0.5–2 ms at 640×480, a fraction of ArUco).
- Robust to blur, noise and interlace combing in analog video.

The full specification (marker dimensions, detection steps, performance) is in [docs/spec.md](docs/spec.md).

![FGM](docs/fgm.png)

## Markers

| | Horizontal pattern (left to right, in modules) |
| --- | --- |
| Left post | white ≥1 : black 3 : white 1 : black 1 : white 1 : black 1 : white ≥1 |
| Right post | white ≥1 : black 1 : white 1 : black 1 : white 1 : black 3 : white ≥1 |
| Round gate ring (outside → inside) | white 1 : black 3 : white 1 : black 1 : white 1 : black 1 : white 1 |

Mount the stripes vertically, with the thick black band on the outer side of the gate.
Taller markers are easier to read and still work when the drone is tilted.

### Making one with tape

Stick black vinyl tape vertically on a white board or sheet of paper. If one tape width is one module, the left post is
3 tapes of black, a 1-tape gap, 1 tape, a gap, 1 tape (the right post is the mirror image).
The detector reads ratios, so small errors are fine. Leave at least one module of white on the outer sides.

### Printing

```
pip install ".[opencv,pdf]"            # in this folder
fgm-make-markers                       # writes to markers/
```

- `marker_fgm_left.png` / `marker_fgm_right.png`: for the left and right posts.

## Detection

```python
from fgm import FgmDetector

det = FgmDetector()                 # one per video stream (camera)
d = det.detect(frame)               # BGR or grayscale image
d.visible                           # is the gate visible
d.corners, d.ids                    # corners of each FGM found, and "L" / "R" (for drawing)
```

`visible` turns True when both the left and right markers are seen, and stays True until both are gone
(it does not turn False when one side leaves the frame first as you approach the gate). For lap timing, count a pass when
`visible` goes from True to False (waiting about 100 ms helps ignore single missed frames).

| Argument | Default | Meaning |
| --- | --- | --- |
| `line_step` | 4 | Read every Nth pixel row |
| `min_lines` | 3 | Number of consecutive rows needed to accept a marker |
| `min_size_pct` | 5.0 | Minimum marker width (black part) as % of frame height. Ignores distant gates |

For small images such as 640×480, OpenCV runs faster with a single thread (`cv2.setNumThreads(1)`).
This is a process-wide setting, so the detector does not change it.

OpenCV is not a required dependency (installing both `opencv-python` and `opencv-contrib-python` breaks things).
If you do not have it, install with the `[opencv]` extra.

## Development

```
pip install -e ".[opencv,pdf]"
python tests/test_fgm.py                    # tests on synthetic video
python tools/bench.py --videos a.mp4        # comparison with ArUco (speed, detection rate, false positives)
```

Used by: d-plactice-ArUco (an FPV drone lap timer)

## License

MIT
