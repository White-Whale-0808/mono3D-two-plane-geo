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
