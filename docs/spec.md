<div align="right"><a href="./spec_jp.md">日本語はこちら</a></div>

# FpvGateMarker (FGM) Specification

A vertically striped marker for FPV drone race gates, and how to detect it. From the onboard camera feed
(analog 5.8 GHz video is assumed), the detector decides for each frame whether the gate is "visible" or "not visible".
Lap timing (counting a pass at the moment the gate goes from visible to not visible) is up to the application.

This document lists every number so that the same detector can be implemented in other languages (C++, etc.).
The Python implementation is `fgm/detector.py`.
**If this document and the code disagree, fix one of them so they match.** This English version is the reference;
the Japanese version (`spec_jp.md`) is kept in sync with it.

## 1. Markers

### 1.1 FGM for the side posts

The stripes are vertical and are read horizontally. A left and a right marker form a pair, mounted so that the thick
black band is on the outer side of the gate. The width is defined by the black/white ratios (in modules); the height
can be as long as you like.

| | Horizontal pattern (left to right, in modules) |
| --- | --- |
| Left marker (left post) | white ≥1 : **black 3 : white 1 : black 1 : white 1 : black 1** : white ≥1 |
| Right marker (right post) | white ≥1 : **black 1 : white 1 : black 1 : white 1 : black 3** : white ≥1 |

- The bold part is 7 modules wide. This is called the "FGM width".
- The printable markers (`fgm-make-markers`) have 2 modules of white on the outer sides (to allow for blur) and are
  16 modules tall (`marker_fgm_left.png` / `marker_fgm_right.png`).
- Taller is more stable and still readable when the drone is tilted. As long as one horizontal line crosses all five
  stripes, tilting does not change the ratios (every crossed width is stretched by the same factor).
  About 45° with a height of 16 modules, about 55° with two markers stacked vertically.

### 1.2 Round gate ring

Concentric stripes along the ring of a round gate. From outside to inside:

**white 1 : black 3 : white 1 : black 1 : white 1 : black 1 : white 1** (modules)

- A horizontal line through the center crosses the left side as a left marker and the right side as a right marker,
  so detection is the same as 1.1 (there is no ring-specific processing). Because it is a circle, it can be read at any
  tilt angle.
- The outermost and innermost white modules are printed too (do not trim them). They act as quiet zones regardless of
  the gate frame color.
- Given the gate inner diameter (`--gate-inner`, default 600 mm) and module size (`--gate-module`, default 15 mm),
  `fgm-make-markers` splits the ring into arc segments that fit on A4 and writes an actual-size PDF
  (`marker_fgm_ring.pdf`). 10 pages for 600 mm / 15 mm. **Print at 100% (actual size).**
  A preview of the whole ring is `marker_fgm_ring_preview.png`.

## 2. Detection

The input is one frame (grayscale, or color converted to grayscale; only the lines in (1) need to be converted).
Let `h` be its height and `w` its width. Pixel values are integers from 0 to 255.

### 2.1 Parameters

| Name | Default | Meaning |
| --- | --- | --- |
| `line_step` | 4 | Read every `line_step`-th pixel row |
| `min_lines` | 3 | A marker needs this many consecutive lines |
| `min_size_pct` | 5.0 | Minimum FGM width (7 black modules) as % of frame height |

### 2.2 Constants

| Name | Value | Meaning |
| --- | --- | --- |
| `FGM_LEFT` | (3, 1, 1, 1, 1) | Black/white widths of the left marker (starting with black) |
| `FGM_RIGHT` | (1, 1, 1, 1, 3) | Right marker |
| `FGM_TOL` | 0.5 | Tolerance, in modules |
| `FGM_MAX_GAP` | 1 | Number of unreadable lines allowed when chaining vertically |
| `FGM_MIN_HEIGHT` | 0.7 | Vertical extent must be at least this times the width |
| `FGM_PAIR_GAP` | 2.0 | Gap between left and right must be at least this times the width (average of both) |
| `FGM_PAIR_SIZE` | 3.0 | Ratio of left and right widths must be within this factor |

### 2.3 Steps

**(1) Lines to read**: y = `line_step // 2`, `line_step // 2 + line_step`, … (< h).

**(2) Binarization** (per line)
1. Horizontal 5-pixel Gaussian blur (no vertical blur; OpenCV `GaussianBlur(ksize=(5,1), sigma=0)`, result rounded
   to integers).
2. `mean` = the moving average of the result of 1 over `max(31, (w // 4) | 1)` pixels horizontally (edges replicate
   the edge value), rounded to the nearest integer (the window size is odd, so it is never exactly .5;
   same as OpenCV `blur` on uint8).
3. A pixel is black if the result of 1 is less than `mean − 5`, otherwise white.

**(3) Finding the pattern in a line** (for each pattern `P = (p0, …, p4)`, left and right)
1. Let e0, e1, … be the positions where the color changes (x where pixels x and x−1 differ), in order.
2. Each run of 6 consecutive edges e0…e5 where e0 is the start of black is a candidate
   (e0–e1 black, e1–e2 white, …, e4–e5 black).
3. One module `u = (e4 − e0) / (p0 + p1 + p2 + p3)` (start of the first black to start of the last black).
4. Width `e5 − e0 ≥ h × min_size_pct / 100`.
5. Each distance from the start of a color to the start of the next run of the same color, `e(j+2) − e(j)`
   (j = 0–3), is within ±`FGM_TOL × u` of `(p(j) + p(j+1)) × u` (inclusive).
   That is 4:2:2:2 for left and 2:2:2:4 for right. These distances do not change when black bleeds wider or narrower
   (the way barcodes are read).
6. Quiet zones: from the edge before e0 (x = 0 if there is none in the line) to e0, and from e5 to the edge after e5
   (x = w if there is none), are both at least `FGM_TOL × u`.
7. Each match is recorded as (line index, x0 = e0, x1 = e5).

Dividing by u in 5 and 6 makes exact boundary cases depend on rounding, so multiply both sides by
`s = p0 + p1 + p2 + p3` and compare using `d = e4 − e0`: 5 becomes
`|s × (e(j+2) − e(j)) − (p(j) + p(j+1)) × d| ≤ FGM_TOL × d`, and 6 becomes `s × quiet zone ≥ FGM_TOL × d`
(the left sides are integers; `FGM_TOL × d` is a multiple of 0.5, so it is exact in floating point too).

**(4) Chaining vertically into one marker** (left and right separately; candidates are visited in line order,
then by x0 within a line)
1. For each candidate, look at the existing chains in creation order and append it to the first chain that satisfies
   all of the following. If none does, start a new chain.
   - The line index difference `gap` from the chain's last line is between 1 and `FGM_MAX_GAP + 1`.
   - The center `(x0 + x1) / 2` moves by at most `line_step × gap + width / 7` (so the chain continues when the drone
     is tilted).
   - The width ratio (this candidate / the chain's last) is between 0.7 and 1.43.
2. Keep chains with at least `min_lines` lines.
3. Let y0, y1 be the y of the chain's first and last lines and W the average width. Drop the chain if
   `y1 − y0 + line_step < FGM_MIN_HEIGHT × W` (this removes short matches such as OSD characters or a single ArUco
   module; 0.5 still produced false positives on OSD characters).
4. Each remaining chain is one FGM. x0 and x1 are the chain averages; the corners are (first x0, y0), (first x1, y0),
   (last x1, y1), (last x0, y1).

**(5) Gate check**: "both left and right are visible" if at least one pair of a left FGM `L` and a right FGM `R`
satisfies:
- `max(L.W, R.W) ≤ FGM_PAIR_SIZE × min(L.W, R.W)`
- `R.x0 − L.x1 ≥ FGM_PAIR_GAP × (L.W + R.W) / 2` (left is to the left of right and they are a gate width apart;
  this removes adjacent matches such as OSD characters)

**(6) Visible / not visible** (keeps the previous frame's state `seen`, initially false)
- If "both left and right are visible", `seen = true`.
- Otherwise, if there is no FGM at all (neither left nor right), `seen = false`.
- Otherwise (only one side, or no valid pair), keep the previous frame's value.
- `seen` is the result for the frame.

In other words, seeing only one side while not visible does not make it visible. Once visible, it stays visible until
both sides are gone. When approaching a gate, one side leaves the frame first; that is not a pass — the pass happens
when the last marker disappears. Because it keeps state, **create one detector per video stream (camera)**,
and call `reset()` when the stream switches.

### 2.4 Output

- `visible`: `seen` from (6).
- `corners`, `ids`: the corners of each FGM found and `"L"` / `"R"` (reported even for one side only; used to draw
  frames when setting up gates).

## 3. Performance (`tools/bench.py`)

Compared with ArUco (DICT_4X4_50, IDs 0–3). Sizes are matched by the width of the black part (one side for ArUco,
7 black modules for FGM), with default settings.

- Processing time (640×480, per frame): on synthetic video with markers, ArUco about 13 ms, FGM about 2 ms; without
  markers, ArUco about 1.4 ms, FGM about 0.5 ms. On FPV practice videos, ArUco 4.8 ms and FGM 2.5 ms on average
  (FGM measured before the speedup). The images are small, so OpenCV with a single thread (`cv2.setNumThreads(1)`)
  is faster still (about 0.3–1 ms on synthetic video).
- Detection rate (synthetic video, 30 and 40 px, 10 trials each)
  - Clean and lightly blurred: 100% for ArUco, FGM (posts) and FGM (round gate).
  - Strong blur + noise + interlace combing (σ2, noise 15, 6 px): ArUco 0%, FGM (posts and round gate) 100%.
  - Tilted 60°: FGM posts 0% (ArUco and the round gate can be read).
  - Oblique (0.6× horizontally): falls below the minimum size, about 50% for all.
- Minimum readable size: ArUco about 30 px, FGM (posts) about 24 px, FGM (round gate) about 20 px. ArUco can be read
  from a single marker, so it still works when the gate is so close that it overflows the frame.
- False positives (12 practice videos without markers, about 14 minutes): 0 for both. The first version read OSD
  characters (white text with black outlines) as a left/right pair and counted 6 false passes, so the conditions on
  vertical extent and left/right spacing (2.3 (4)-3 and (5)) were added.

Why it is robust to analog video: interlace combing does not matter within a single line because a line belongs to one
field. Noise on a line only costs one readable line. It is also robust to vertical motion blur.

The detection rate on video of real FGM markers has not been measured yet.
