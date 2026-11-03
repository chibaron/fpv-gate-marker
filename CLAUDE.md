# fpv-gate-marker

FpvGateMarker（FGM）: FPV ドローンのゲートに貼る縦縞のマーカーと、その検出器（Python パッケージ `fgm`）。
d-plactice-ArUco（`C:\USER\d-plactice-ArUco`）から分けた。d-plactice-ArUco は `pip install -e ../fpv-gate-marker` で使う。

- 仕様書: `docs/spec.md`（英語、正本）と `docs/spec_jp.md`（日本語版）。ほかの言語でも同じ検出を作れるよう数値を全部書く。コードを変えたら必ず両方合わせる
- 公開する（MIT）。d-plactice-ArUco の設定や Firebase のことは書かない

## 実行とテスト

```
pip install -e ".[opencv,pdf]"
python tests/test_fgm.py                    # 変えたら通す
python tools/bench.py --videos a.mp4        # ArUco との比較（処理時間・認識率・誤検出）
fgm-make-markers                            # 印刷用（python -m fgm.markers でも同じ）
```

## 構成

| ファイル | 役割 |
| --- | --- |
| `fgm/detector.py` | `FgmDetector`（間引いた横線で左右の FGM を探す。前のフレームの状態 `seen` を持つ）、`Detection`、定数 |
| `fgm/markers.py` | 左右の PNG、円形ゲートの輪（A4 に分けた実寸 PDF）、`fgm-make-markers` |
| `tools/bench.py` | ArUco との比較。ArUco の検出と通過の数え方は d-plactice-ArUco と同じものを自前で持つ |

## 決まっていること

- 左 = 黒3:白1:黒1:白1:黒1、右はその鏡像、外側に白1以上。円形ゲートの輪は外から 白1:黒3:白1:黒1:白1:黒1:白1
- 「見えた」= 左右の両方が見えた。「見えない」= 左右の両方とも見えない（その間は前のフレームのまま）
- OpenCV は依存に入れない（`[opencv]` の extra。opencv-python と opencv-contrib-python がぶつかるため）
- コメント・文書は日本語。ただし `README.md` と `docs/spec.md` は英語（日本語版は `README_jp.md` と `docs/spec_jp.md`。変えたら両方合わせる）
