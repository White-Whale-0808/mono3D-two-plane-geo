# TODO — mono3D-two-plane-geo

> **最後對照程式碼：2026-07-28**（WWH-7 / WWH-8 / WWH-9 + 標定修正之後）。
> 行號皆為當下實測。本檔 2026-07-03 首版的部分項目已失效，改寫時保留了
> 「已完成 / 已失效」的紀錄以免重複討論。

三大區塊：
1. [`lane_segmentation.py` 優化](#lane_segmentationpy-優化)
2. [Repo 層級待辦](#repo-層級待辦)
3. [`get_carlaDataset.py` 需要補採的資料](#get_carladatasetpy-需要補採的資料)

---

# lane_segmentation.py 優化

## 高優先

### 1. 把具名常數搬進 config（影響準度，跨相機高度可調）
模組頂部 **L35–L49** 的常數命名清楚、有物理意義，但全部寫死、無法調整。
同一份 code 要跑 CARLA(相機 2.4m) 和 dataset(1.08m) 兩種高度，這些值卻不能隨場景變。
建議搬進 `config/inference_road_lane_segmentation.yaml` 的 `lane_segmentation` 區塊，
或做成 `split_left_right_lines` 的 keyword 參數。

- [ ] `_TOL_LANE_FRACTION = 0.10`（關聯容差 = 局部車道寬的 10%）
- [ ] `_TOL_PX_FLOOR = 3.0`（ELSED 端點噪聲下限 px）
- [ ] `_CROSS_LANE_FRACTION = 0.40`（搜尋不超過 40% 車道寬）
- [ ] `_SEED_DELTA = 0.5`（自車橫向位置不確定性）
- [ ] `_SEED_X_MAX = 8.0` / `_NOISE_X_MAX = 16.0`（斜率閘門的橫向公尺數）
- [ ] `_MODEL_MEMORY_M = 4.0`（局部模型擬合的深度範圍）
- [ ] `_RESET_GAP_M = 2.0`（深度跳變 → 可能換平面的閾值）
- [ ] `_MAX_GRADE_DEG = 15.0`（最壞路面坡度）
- [ ] `_GRADE_RAMP_Z0 = 6.0` / `_GRADE_RAMP_SPAN = 6.0`（坡度 slack 的 ramp）
- [ ] `_SUPPORT_MIN_LEN_PX = 60.0`（WWH-6 新增，2026-07-03 首版未列）

#### 1a. 這些常數是否有依據？(是否算 magic number)
重點結論：**幾何/論文給的是「縮放形式」(threshold 隨 y、車道寬、深度怎麼變)，
不是「係數本身」。** 所以即使 docstring 標為 "geometry-derived"，多數 scalar
仍是手選 → 嚴格定義下還是 magic number，只是「有動機的」。

分類（依依據強度）：

| 常數 | 形式來源 | scalar 本身 | 判定 | 可 ref |
|---|---|---|---|---|
| `_SEED_DELTA = 0.5` | 幾何 | 真推導：車可在自車道內任意橫向 → 內側標線 X∈(0±0.5)·w | **有依據（最壞界）** | 自身幾何；但保守，見 L136 TODO |
| `_MAX_GRADE_DEG = 15.0` | 工程標準 | 15°(~27%) 是道路最大縱坡的保守上界 | **有依據（外部標準）** | 道路幾何設計規範（如 AASHTO 縱坡上限） |
| `_SUPPORT_MIN_LEN_PX = 60.0` | 實測分佈 | 註解記了依據：電線桿/山坡 32–54 px，合法遠段 ≥77 px | **有依據（實測，樣本數未知）** | 註解 L47–49；建議補樣本數 |
| `_TOL_PX_FLOOR = 3.0` | 感測器噪聲 | 3px 安全下限，可對應 ELSED 端點抖動 | **半 magic（經驗有依據）** | `docs/papers/ELSED_*.pdf`（定位精度） |
| `_TOL_LANE_FRACTION = 0.10` | 幾何(∝車道寬) | 10% 係數手選 | **magic（有動機）** | 無；建議用標線寬/ELSED 誤差統計推 |
| `_CROSS_LANE_FRACTION = 0.40` | 幾何(∝車道寬) | 40%<50% 中線給 margin，係數手選 | **magic（有動機）** | 無 |
| `_SEED_X_MAX = 8.0` / `_NOISE_X_MAX = 16.0` | 斜率↔橫向距離換算是幾何 | 8m/16m 距離手選 | **magic** | 無 |
| `_MODEL_MEMORY_M = 4.0` | — | 局部窗長手選 | **magic** | 無 |
| `_RESET_GAP_M = 2.0` | — | 平面變化門檻手選 | **magic** | 無 |
| `_GRADE_RAMP_Z0 / SPAN = 6.0` | — | 近場視為自車平面的距離手選 | **magic** | 無 |

小結：真正站得住腳的只有 `_SEED_DELTA`、`_MAX_GRADE_DEG`、`_SUPPORT_MIN_LEN_PX`
(部分 `_TOL_PX_FLOOR`)；其餘 6 項仍應當作 magic number。投影/兩平面模型本身有
repo 內論文背書（`Lin_&_Tsai_IEEETPAMI_1991.pdf`、`AI-Enhanced_Mono-View_*.pdf`），
但**沒有任一篇規定這些係數的具體數值**。

- [ ] 待辦：對「magic（有動機）」這 6 項，用 CARLA GT 量化校準（如標線寬、橫向偏移
      p95、實際換面距離分佈），把手選值換成資料推導值，並在註解標明來源
      → 需要的資料見[第 3 區塊](#get_carladatasetpy-需要補採的資料)
- [ ] 待辦：對「有依據」的項目，在註解補上明確 ref（規範名稱 / repo 論文路徑）

### 2. 修失效的文件參照（成本最低）
模組 docstring **L7、L11** 仍指向：
- `docs/lane_segmentation_issues.md`（問題 4）
- `docs/lane_segmentation_parameter_problem.md`

但 `docs/` 底下只有 `papers/` 和 `diagrams/`，這兩個檔仍不存在（**2026-07-28 覆核仍失效**）。

- [ ] 補回文件，或更新註解指向 `docs/diagrams/lane_segmentation_flow.drawio`
      / `docs/papers/lane_segmentation_design_logic.drawio`

## 中優先

### 3. 抽出仍硬寫、且連名字都沒有的數字
這些比第 1 點更值得抽出，因為完全沒有說明（皆為 magic number）。行號已於 2026-07-28 更新：

- [ ] **L130** legacy gate：`0.5 * min_slope * (mid_y / img_height)` — 0.5 無說明
      （僅 `geom is None` 的 legacy 分支會走到）
- [ ] **L161 / L173 / L176** `_fit_x_of_y`：`last_n=8`、最少 `>= 4` 點
- [ ] **L294–295** legacy `assoc_window`：`2.0 + 1.5*missed`、`0.18 * center_x`
- [ ] **L363**：`missed > max(4, track_bands // 3)`
- [ ] **L374**：`missed >= 2`；**L376**：`track_points[-2:]`
- [ ] **L468**：`track_bands = max(int(track_bands), 16)`
      （WWH-7 已把參數名 `num_bands` → `track_bands` 並在 config 設 16，
      所以「默默改成 16」的坑已緩解；但 clamp 本身仍未說明理由）
- [ ] **L65**：`min_y_margin=0.05`

### 3b.（新增 2026-07-28）`lane_fitting.py` / `pitch_estimation.py` 的常數
WWH-9 重寫後新增了一批常數，2026-07-03 首版未涵蓋。它們的註解**普遍比
lane_segmentation 那批好**（多數記了實測依據與受影響幀號），但同樣全部寫死：

`lane_fitting.py`
- [ ] `_SHADOW_MARGIN_PX = 3.0` / `_SHADOW_MIN_OVERLAP_ROWS = 8`（L10–11）
- [ ] `_FRAG_MAX_STEP_PX = 4.0`（L16）— **隱藏耦合**：註解說它是從
      `min_slope = 0.3` 推的，但 `min_slope` 在 config 裡可調，改了不會連動
- [ ] `_JUNCTION_TOL_PX = 20.0` / `_JUNCTION_SLOPE_ROWS = 10`（L23、L26）
- [ ] `_REFINE_SEARCH_PX = 3` / `_REFINE_MIN_GRAD = 4.0`（L33、L39）

`pitch_estimation.py`
- [ ] `WINDOW_FRAC = 0.15` / `WINDOW_MIN_M = 1.0`（L7–8）—— 這兩個是 windowed
      估測器**明示的空間解析度**，最該進 config
- [ ] 函式預設值：`z_cap_m=45.0`、`min_valid_range_m=0.5`、
      `min_window_points=4`、`n_pitch_samples=200`、`resid_mad_k=5.0`(spline 路徑)

### 4. 殘留 TODO 與死參數
- [ ] **L136** TODO：`replace 1.0 multiplier with p95 lateral offset from CARLA GT` 尚未完成
- [ ] `roi_far`（**L421**）仍標明 unused、僅為簽名相容保留 → 評估清掉
- [ ] `roi_near` / `min_slope` / `lane_band_tolerance` 仍僅 legacy 路徑使用 →
      幾何模式啟用後是雜訊，考慮集中到 legacy 分支或清理
      （注意 `min_slope` 另被 `_FRAG_MAX_STEP_PX` 的推導引用，見 3b）

## 低優先

### 5. 效能
- [ ] `_track_side`（L269 起）每個 band 對全部 `infos` 線性掃描，左右各一次，
      複雜度 O(bands × N)。線段量大時可先按 `y` 對 infos 建索引/排序加速。
      目前 lane segmentation 只佔 ~16 ms，非瓶頸（真正的瓶頸見 B）。

---

# Repo 層級待辦

## 高優先

### A.（已改寫 2026-07-28）two-plane 模型與 `infer_one` 的定位
**原始條目已失效**：`fit_two_plane_model` 在 WWH-7 就從 `pitch_estimation.py`
**整個移除**了（只剩 `carla_module/realtime_test.py:38` 還 import 它 → 壞的，
屬 WWH-10）。`pipeline.py` 的 docstring/return 不一致也已修好。

改為以下三項：

- [ ] **`infer_one` 目前沒有任何呼叫者**。兩個 runner（`utils/*.py`）都各自
      inline 展開了五個階段，`carla_module` 走自己的路徑。但 `CLAUDE.md:40`
      和 `README.md:28` 都還宣稱它是 "core entry point"。
      → 決定：讓 runner 改用 `infer_one`，還是承認它是範例碼並更新文件
- [ ] **`pipeline.py` 的 docstring 又過期了**：L9 和 L66 都寫
      "continuous spline pitch(z)"，但 WWH-9 之後預設估測器是 **windowed**
- [ ] **`infer_one` 無法選 pitch 估測器**：`estimate_pitch_from_curves` 有
      `method` 參數（`windowed`/`spline`），但 `infer_one` 沒往下傳，永遠用預設
- [ ] 專案名稱仍是 "two-plane geometry"，但現在的輸出是連續 pitch(z) 曲線，
      沒有 near/far 兩平面 + knee 的概念了 → 決定要不要重新引入，或更新命名/文件

### B.（已完成）ELSED+PIDNet 的 MAE 驗證
- [x] batch inference + profile MAE 已跑過多輪。最新基準（2026-07-28，
      標定修正後）：**mean 0.2331 / median 0.1845 / p90 0.4291 / max 0.7223**，
      544 幀 0 例外。修正前基準備份於 `debug/outputs/pre_calibration_baseline/`

### B2.（新增）windowed pitch 的 95 ms 延遲
- [ ] `estimate_pitch_windowed` 每幀呼叫 200 次 `theilslopes`，pitch 階段
      從 5 ms 變 ~95–135 ms。批次無所謂，**CARLA 即時前必須處理**（屬 WWH-10）

## 中優先

### C. 清掉已死的註解區塊
- [ ] `libs/inference/road_segmentation.py` **L56–L72**：整段被註解掉的舊
      Resnet101 推論程式碼，已不使用，可刪。另 **L84**、**L89–90** 的
      erosion / GaussianBlur 死碼（L81–82 已有說明為何放棄，可保留說明刪死碼）
- [x] `lane_fitting.py` 的 RANSAC import 與用法註解 —— **已於重寫時清掉**

### D. 沒有任何單元測試
**2026-07-28 覆核：repo 內仍無自己的 test。**（`carla_module/realtime_test.py`
是需要 CARLA server 的整合測試；`elsed_src/pybind11/tests/` 是 vendored 的第三方測試。）
WWH-8 期間跑過的等價性測試是臨時腳本，沒有進版控。

- [ ] 至少為 `pitch_estimation`（windowed / spline / degenerate fallback）與
      `lane_segmentation` 幾何推導加幾個 pinhole 合成資料的單元測試
- [ ] `debug/` 底下的診斷腳本（`check_width_calibration.py` 等）有現成的
      合成/實測驗證邏輯，可以抽成正式測試

## 低優先

### E. config 與文件一致性
- [ ] `config` 內仍留 3 個被註解的舊參數：`threshold`(L15)、
      `mask_erosion_kernel`(L16)、`roi`(L24) → 確認不再需要後清理
      （`ransac_residual_threshold` 已不存在；L5 的替代 `weight_path` 是刻意保留）
- [ ] README.md 內容是否同步到目前 pipeline，待對照一次
      （已知至少 L28 的 `infer_one` 說法要改，見 A）

### F. Linear integration verified
- [x] WWH-5: Linear-GitHub 連動測試完成

### G.（新增 2026-07-28）profile 圖依 z 切成固定跨距
- [ ] 不同幀的 z 範圍差很多（2.5–10 m vs 9.5–45 m），同樣 8 inch 寬的圖
      造成 m/inch 差 5 倍，遠距幀的細節看不見。原型已做在
      `debug/viz_profile_split.py`（每張固定 5 m，`--yscale shared|tile`，
      建議 tile）。**使用者 2026-07-27 指示先不動畫圖**，待日後決定是否進 libs

---

# `get_carlaDataset.py` 需要補採的資料

**背景（2026-07-28）**：診斷 `Y_3d` vs GT height 的全域偏移時，發現兩個誤差
（`w_real` 3.5→3.216、GT 對齊少算相機前移 1.5 m）。修正後 batch MAE
0.3265→0.2331。完整推導與實驗見記憶 `calibration-bias-2026-07-27`；
決定性的量測工具是 `debug/check_width_calibration.py`。

但這次的結論**有一項是靠間接論證撐起來的**：影像只能約束 `W/h` 比值
（`w_px = f_x·W/(f_y·h)·(y−cy)`），所以 9 cm 的偏移可以讀成「車道寬 W 太大」
或「相機高 h 太小」，兩者從影像上分不出來。我是用 A/B 的 MAE 排序否決後者的
——很強，但終究不是直接量測。**下面第 1 組就是為了一槌定音。**

現況 CSV 只有 4 欄：`frame_id, gt_pitch_deg, gt_speed_mps, collect_dist_m`
（`DatasetWriter._CSV_HEADER`，L193）。

## 第 1 組：一次性驗證（跑一次、印出來、確認完就好，不必進 CSV）

目的：直接驗證 `w_real = 3.216` 與 `camera_height = 1.08`，解除上述簡併。

- [ ] **`waypoint.lane_width`** — 確認 CARLA 的 3.5 m 定義是**車道邊界中心到中心**
      ```python
      wp = world.get_map().get_waypoint(vehicle.get_location())
      print(wp.lane_width)
      ```
- [ ] **標線寬** `wp.left_lane_marking.width` / `wp.right_lane_marking.width`
      驗證式：`lane_width − (left.width + right.width)/2 == 3.216`？
      目前反推出來的差值是 **0.284 m**。注意雙黃線那側的 `.width` 語意
      （`SolidSolid` 是否含間隙）要看清楚
- [ ] **相機到路面的實際高度** — 這是解簡併的關鍵
      ```python
      cam_z  = camera.get_transform().location.z
      road_z = world.get_map().get_waypoint(vehicle.get_location()).transform.location.z
      print(cam_z - road_z)   # 若 ≈1.08 → w_real 3.216 成立；若 ≈1.18 → 反之
      ```
      `CAMERA_HEIGHT = 1.08`（L55）是相對**車輛原點**掛載的，車輛原點不見得在地面
- [ ] **確認相機無畸變、主點在正中心** — CARLA `sensor.camera.rgb` 是理想針孔，
      主點應嚴格等於影像中心（resize 後 cy = 256）。這會坐實「平坦幀量到
      cy = 253.3 px 是車身俯仰假象」的判斷（A/B 已否決 cy 修正）

## 第 2 組：CSV 要加的欄位（影響 GT 品質，值得重採資料集）

- [ ] **路面坡度（與車身姿態分開）**
      目前 `gt_pitch_deg` 記的是 `transform.rotation.pitch` = **車身姿態**（含懸吊），
      但 pipeline 量的是**路面**。加一欄 `road_pitch_deg`：
      ```python
      wp = carla_map.get_waypoint(transform.location)
      road_pitch = wp.transform.rotation.pitch
      ```
      證據：平坦路段的 10 幀裡 `cy_true` 在 252.5↔254.5 px 漂移（≈車身俯仰），
      同批幀 `W_est` 卻穩定在 3.216（std 0.003）。兩者分開記才能量化這個誤差源。
      註：profile 的**參考基準** p0 應該仍用車身姿態（相機剛性固定在車上），
      但**前方的坡度**應該用路面 → 記兩欄才能兩邊都對
- [ ] **車輛世界座標 `loc_x, loc_y, loc_z`**
      目前只存累積距離。有了 (x,y,z) 就能：
      (a) 分辨**弧長 vs 水平距離**（`collect_dist_m` 是 3D 弧長，但相機的 z 是水平距離）
      (b) **直接**重建 GT 高度剖面，取代現在 `gt_height_profile` 那個
          「積分 tan(pitch)」的近似 → 積分誤差整個消失
- [ ] **相機掛載參數寫進 metadata**
      現在 `camera_forward_offset: 1.5` 和 `camera_height: 1.08` 是手動從
      `get_carlaDataset.py:296` 抄進推論 config 的，兩邊沒有任何連結，改一邊
      另一邊不會知道。採集時輸出 `metadata.json`：
      `{camera_x, camera_z, fov, img_width, img_height, map_name, vehicle_bp}`

## 第 3 組：真正一勞永逸的做法（建議優先評估）

- [ ] **直接採樣前方路面剖面，取代「用行駛歷史回推」**
      現在 `gt_pitch_profile` 是拿「車子後來開到那裡時的 pitch」當作前方 GT
      （`pitch_visualization.py` 模組 docstring）。這帶來三個問題：
      前移 1.5 m 的對齊、弧長 vs 水平距離、以及**開到終點的幀沒有前方 GT**。
      改成採集當下直接問地圖：
      ```python
      wp = carla_map.get_waypoint(camera_world_location)
      for d in range(0, 51):           # 每 1 m
          nxt = wp.next(d)             # 記 z 與 rotation.pitch
      ```
      存成每幀一列的 `road_profile.csv`（或 npz）。這樣 GT 是**當下量測**而非
      重建，上述三個問題一次消失，`gt_pitch_profile` / `gt_height_profile`
      可以整個簡化掉

## 第 4 組：既有待辦（沿用）

- [ ] 連 CARLA server 看 HUD，確認坡度符號（上坡為正）—— WWH-8 遺留
- [ ] `lane_segmentation` 的 6 個 "magic（有動機）" 常數，用 CARLA GT 量化校準
      （標線寬、橫向偏移 p95、實際換面距離分佈）—— 見本文件第 1a 節

## 注意

- `carla_module/` 目前仍 **DEFERRED**（WWH-10）：`realtime_test.py` /
  `carla_visualization.py` 都 import 了已刪除的函式。動 CARLA 時要一起遷移到
  `lane_curve` / `sample_widths_from_curves` / `estimate_pitch_from_curves`
- `realtime_test.py` 也讀 `w_real`，遷移時記得新的語意是**內側邊到內側邊**
- 若第 1 組驗出 `w_real` 該是別的值，**現有 `outputs/measurements.csv` 的
  MAE 基準要整批重跑**；修正前的基準備份在
  `debug/outputs/pre_calibration_baseline/`
