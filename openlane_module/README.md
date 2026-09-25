# openlane_module

把 **OpenLane**（OpenDriveLab 的 3D 車道資料集，蓋在 Waymo Open Dataset 上）
轉成本專案既有的資料集格式，以及確認這份資料能拿來驗證什麼的工具。

pipeline 與 GT 模組**一行未改** —— 相容性全部在轉換器裡解決。

資料集的完整事實表（規模、篩選漏斗、GT 精度、已知失效）在
[WWH-18](https://linear.app/wwhale/issue/WWH-18)。

## 檔案

| 檔案 | 做什麼 |
|---|---|
| `convert_openlane.py` | 轉成 `images/ + measurements.csv + road_profile.csv` 三件套 |
| `frame_tags.py` | 逐幀變因表（天氣/時段/地景、六個 case tag、實測曲率、標線組合、GT 寬） |
| `frame_check.py` | 判別 `xyz` 在哪個座標系（投影殘差：光學系 0.90 px vs 車體系 11.75 px） |
| `gt_width_noise.py` | 車道寬 GT 自身的雜訊，兩個獨立方法（都得到約 12 mm） |
| `measure_paint_inset.py` | **直接量**漆線內縮 inset（GT 標中心、`w_real` 要內緣，差一個 inset）。不經過 pipeline 的寬度所以不循環；`--carla` 用已知真值驗量測方法本身 |
| `nearfield_feasibility.py` | 近場錨在高相機上的可行性：視窗在不在影像裡、移遠要付多少代價 |
| `export_updown.py` | 官方「上下坡」標籤（`test/1000_updown.txt`，5,518 幀 / 33 段）**不篩選**全部轉出；轉換器會踢掉的原因寫進 `drop_reasons` 欄，剖面只用漆線算（路緣不當路面高度）。另出 `index.csv`（逐段）與 `frames.csv`（逐幀） |
| `select_updown.py` | 從上面的輸出挑片段：去掉大部分是路緣（無漆線幀 ≥ 50%）與只有單邊（兩側漆線幀 < 25%）的段，以目錄連結放進新根目錄，不複製影像 |
| `select_updown_straight.py` | 同一份輸出**逐幀**篩：兩側內側線都是漆線（轉換器收錄條件）、50 m 內 sagitta ≤ 0.15 m（Tier A 門檻）、非夜間、非路口（官方 tag，`--keep-night` / `--keep-intersection` 放回）。frame_id 重編、`collect_dist_m` 保留，影像用硬連結。2026-09-24：留下 ≥10 幀的 8 段 221 幀、38 個連續 run |
| `convert_openlane_culane.py` | OpenLane 2D 標註（`uv`）→ CULane 格式（`.lines.txt`＋分割遮罩＋`list/<split>_gt.txt`），給 CLRNet 等 2D 偵測器訓練／評估（WWH-25）。只收 `attribute` 1–4（左左／左／右／右右），路緣從不帶 attribute 所以自動排除；影像硬連結、不縮放不裁切。⚠ OpenLane 標註通常沒延伸到影像底部（從 8–14 m 才開始）；`--extend-bottom` 把每條線依最近那段直線延伸到底部（CULane 的標註習慣），預設不延伸，微調時兩種都試。空標籤幀依原因分類（`frame_kind`；`--stats-only` 只統計）：驗證集 42.7% 空（路緣 19.3%、沒線 9.2%、遠處漆線 11.4%、自車道位置上有漆線但沒標 2.7%），預設排除最後一類。⚠ 最後一類 96% 在路口，是 OpenLane「路口裡不定義自車道」的慣例而非漏標；正式訓練前要決定是否改成保留 |

全部從 repo 根目錄執行，預設讀 `D:/datasets/openlane`：

```bash
# 只轉標註（不需要影像）
python openlane_module/convert_openlane.py --openlane D:/datasets/openlane --out <OUT> --no-images

# 只轉指定的幀（分層實驗用；CSV 需有 segment / frame 欄，由 frame_tags.py 產生）
python openlane_module/convert_openlane.py --openlane D:/datasets/openlane --out <OUT> \
    --frame-list debug/outputs/tierA_frames.csv

python openlane_module/frame_tags.py            # -> debug/outputs/openlane_frame_tags.csv
python openlane_module/frame_check.py
python openlane_module/gt_width_noise.py
python openlane_module/nearfield_feasibility.py

# 漆線內縮：Tier A 全部，或用 CARLA 的已知真值驗量測方法
python -m openlane_module.measure_paint_inset
python -m openlane_module.measure_paint_inset --carla inference_datasets/<dataset> --limit 40

# 官方上下坡子集：全部轉出，再挑兩側都有漆線的片段（WWH-22）
python -m openlane_module.export_updown --openlane D:/datasets/openlane --out D:/datasets/openlane_updown
python -m openlane_module.select_updown --src D:/datasets/openlane_updown --out D:/datasets/openlane_updown_twoside

# 2D 偵測器的訓練資料（CULane 格式）；輸出路徑要短（Windows 260 字元上限）
python -m openlane_module.convert_openlane_culane --split training \
    --openlane <OpenLane 根目錄> --out D:/datasets/openlane_culane
```

## 三個設計決定

1. **相機位姿寫成單位姿態、原點 0** → 世界座標就等於相機座標。因為
   `lane_lines[i]['xyz']` 本來就在相機光學座標系（已用 `frame_check.py` 判別），
   所以 `z_gt = x`、`h_gt = z`，不需要 offset、弧長換算或車身姿態。
2. **虛擬內參重採樣**：202 段有 68 組不同內參、主點不置中，等效 −0.42°~+0.69°
   的常數 pitch 偏誤（比整體 MAE 還大）。每幀套 `H = K_virt · K_orig⁻¹` 重採樣到
   同一台虛擬相機，輸出後整個資料集是同一組內參。
3. **逐 segment 一個輸出目錄**：近場 calibrator 與路線剖面圖都假設一條連續路線。

## ⚠ 授權

CC BY-NC-SA + Waymo Non-Commercial。非商業用途；**影像與衍生影像不可進版控**。
轉換器只讀本機路徑，輸出請放在 repo 之外。
