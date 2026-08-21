# TODO — mono3D-two-plane-geo

> **最後對照程式碼：2026-07-28**（WWH-7 / WWH-8 / WWH-9 + 標定修正之後）。
> 行號皆為當下實測。本檔 2026-07-03 首版的部分項目已失效，改寫時保留了
> 「已完成 / 已失效」的紀錄以免重複討論。
>
> **2026-08-18 更新**：CARLA 採集與標定全部完成，第三區塊已大幅精簡（過程中
> 被推翻的假說不再保留於此，完整記錄在 Linear WWH-12 / WWH-13）。行號未重新對照。

三大區塊：
1. [`lane_segmentation.py` 優化](#lane_segmentationpy-優化)
2. [Repo 層級待辦](#repo-層級待辦)
3. [CARLA 資料採集與標定](#carla-資料採集與標定)

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
現行基準（2026-08-18，新資料集 + 量測式 GT + `w_real` 3.25）：
**mean 0.2720 / median 0.1889 / p90 0.6817 / max 0.8608**，453 幀 0 例外。
與更早的數字**不可直接比較**（資料集、GT、`w_real` 都換過）。

### B2.（新增）windowed pitch 的 95 ms 延遲
- [ ] `estimate_pitch_windowed` 每幀呼叫 200 次 `theilslopes`，pitch 階段
      從 5 ms 變 ~95–135 ms。批次無所謂，**CARLA 即時前必須處理**（屬 WWH-10）

## 中優先

### C. 清掉已死的註解區塊
- [ ] `libs/inference/road_segmentation.py` **L56–L72**：整段被註解掉的舊
      Resnet101 推論程式碼，已不使用，可刪。另 **L84**、**L89–90** 的
      erosion / GaussianBlur 死碼（L81–82 已有說明為何放棄，可保留說明刪死碼）

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

### G.（新增 2026-07-28）profile 圖依 z 切成固定跨距
- [ ] 不同幀的 z 範圍差很多（2.5–10 m vs 9.5–45 m），同樣 8 inch 寬的圖
      造成 m/inch 差 5 倍，遠距幀的細節看不見。原型已做在
      `debug/viz_profile_split.py`（每張固定 5 m，`--yscale shared|tile`，
      建議 tile）。**使用者 2026-07-27 指示先不動畫圖**，待日後決定是否進 libs

---

# CARLA 資料採集與標定

**2026-08-18 全部完成。** 採集器已補齊 GT，資料已重採，pipeline 已改用量測式 GT。
完整論證見 Linear WWH-12 / WWH-13，這裡只留結論與未解項。

## 已定案的標定值

| 參數 | 值 | 依據 |
|---|---|---|
| `camera_height` | **1.08** | 直接量到 1.0816（坡度修正後，三種量法全距 0.0001） |
| `f_x` / `f_y` | **512 / 455** | fov 90° @1280 反推 f=640，resize 後；`lens_circle_multiplier=0` 理想針孔 |
| `cy` | **256** | 主點嚴格在中心；用量測 GT 反解逐幀 std 僅 0.30 px |
| `camera_forward_offset` | **1.5** | 沿車輛前進軸恆定；二維掃描顯示是尖銳最佳值 |
| `w_real` | **3.25** | 四條獨立幾何路徑收斂到 3.243~3.250（見下） |

`w_real` 的四條路徑：高度殘差最小化 3.248、平均殘差反推 −H·δ 得 3.246、
隱含車道寬（正向投影 2-15 m）3.243~3.260、`cy=256` 約束 3.2432；
物理幾何（地圖 + 漆緣）3.25。

> ⚠ **3.25 綁定這條路的標線組合**：本資料集是**左雙黃、右單白**。`w_real` 是內緣到
> 內緣，兩側從 3.5 m 的邊界中心寬各內縮不同的量（左 0.1875、右 0.0625，合計 0.25）。
> 同樣 3.5 m 車道下：單+單 3.375、**雙+單 3.25**（本資料集）、雙+雙 3.125；
> 換 `lane_width` 又另計。**不要把 3.25 帶到別的地圖或別條車道。**

## 資料集與 GT

現行資料集 `inference_datasets/carla_dataset_Town03_20260818_195221/`（495 幀），
除 `images/` 與 `measurements.csv`（17 欄）外另有：

- `road_profile.csv` —— 採集當下直接問地圖的前方路面剖面，存 waypoint 的
  **世界座標**（不是預先算好的 pitch，免得把座標系的選擇燒死在資料裡）
- `metadata.json` —— 掛載參數 / fov / 解析度 / 地圖名 / 採集設定

`libs/road_profile_gt.py` 依有無 `road_profile.csv` 自動選 GT 來源：

```
v    = P_world − cam_world
z_gt = v · forward     ← 沿光軸的深度，正是 pipeline 的 z
h_gt = v · up          ← 垂直於光軸的高度，正是 pipeline 的 Y_3d
```

**不需要 offset 常數、不需要弧長換算、不經過車身姿態。** 舊資料集自動退回
回推式 GT，舊基準仍可重現。

## 未解項

### ⚠ 最高優先：新 GT 可能漏掉真實的路面起伏（2026-08-18 發現）

**現象**：以 frame 400 為例，**預測的 pitch(z) 有波浪狀起伏、舊 GT 也有，但新 GT
完全平滑**。使用者目視發現。

**證據（間接但一致）**：

- 把「預測 − 新GT」的殘差**依世界里程分箱**（每箱至少 5 幀看到）：
  箱平均值的變異 **0.310°** vs 箱內跨幀散布 **0.203°**，比值 **1.53**。
  不同幀在**不同深度**看同一段路卻看到同樣的偏差 → **波浪釘在世界座標上**，
  屬於那段路，不屬於處理流程
- `車身pitch − 地圖路面pitch` 的空間自相關，滯後 1 m 仍有 **0.930** ——
  是平滑的空間函數，不是隨機的懸吊抖動

**假說**：`waypoint` 回傳的是 **OpenDRIVE 的解析中心線**，但**相機看到的是渲染
網格、車子壓過的是物理網格**。網格由解析曲線三角化而來，有離散化誤差。
三者不完全相同，所以只有問地圖的新 GT 沒有波浪。

**這代表舊 GT 不是單純比較差** —— 它量的是**車實際壓過的表面**，含有新 GT 沒有
的真實資訊。2026-08-21 量化證實了這點：車身 pitch 有 99.3% 的變異來自地形，
殘差與縱向加速度無關（corr −0.004），而殘差對得上「車輛實際高度偏離解析
中心線」的斜率（corr +0.662，量級 0.333° vs 0.356°）。**懸吊雜訊小到量不出來。**

**已查的兩個候選，都不足以解釋（2026-08-19）**：

- [x] ~~網格 vs 解析曲線~~ —— **量了，太小**。用 `verify_carla_geometry`
      那趟的 `h_wp_cam`（到 waypoint）減 `h_ray`（射線打到實際網格），
      400 幀 `ray_label` 全是 `Roads`、橫向偏移 median 0.00014 m：
      網格 − 解析曲線 median **−0.28 mm**、std **3.59 mm**，等效 pitch 誤差
      std 0.144°。**但空間相關長度只有約 0.3 m**（自相關 1 m 時 0.233、
      3 m 時 0.068），在 2 m 分箱內平均後只剩約 0.06°，**比觀察到的 0.310°
      小五倍**
- [x] ~~右側是虛線、曲線在空白處內插~~ —— **右側是實線**（使用者確認），
      機制不存在

**目前最有希望的候選**：pipeline 反推的**隱含車道寬隨世界位置變動**。
用 GT 深度做正向投影（不是拿 pipeline 深度反算，那是恆等式），依世界位置分箱：
箱平均變異 **0.0302 m** vs 箱內跨幀散布 **0.0156 m**，比值 **1.93**。
量級約 1% → 換算 pitch 約 0.1°，**解釋得了一部分，不是全部**。

> ⚠ 但中段（24-46 m，取樣最充分）其實穩定在 3.242~3.262（±0.3%），
> 大幅偏離集中在路線兩端，那裡貢獻幀數少、深度範圍窄，**選樣偏差未控制**。

### ✅ GT 對齊到網格 —— **已完成，改用網格版**（2026-08-22，WWH-14）

**結論：`ground_truth.height_source` 預設改為 `auto`（有 `z_mesh` 就用 mesh，
舊資料集自動退回 analytic）。** 使用者 2026-08-19 判斷「waypoint 太理想化，
相機看到的是網格」——**這個判斷是對的**。

資料集 `carla_dataset_Town03_20260822_000716`（446 幀，`mesh_label` 100%
`Roads`、`z_mesh` 零空值）。

**三條獨立證據**：

1. **偏差是真的幾何，不是 `cast_ray` 的假象** —— `LegacyProfileGT`（車身姿態，
   **動力學**）與 mesh（射線，**幾何**）相對 analytic 多出來的結構，逐幀去均值
   後 corr median **+0.990**、**100%** 的幀 >0.5。兩條完全獨立的量測路徑
2. **相機看得到** —— 預測相對 analytic 的偏離 vs 網格偏差，逐幀去均值 corr
   median **+0.909**、81.2% >0.5。排列對照（別幀的形狀 / 隨機平滑曲線）只有
   −0.084 / −0.005（32%）。逐幀迴歸斜率 **+0.588**
3. **`w_real` 判定更接近幾何值** —— 零平均高度殘差：mesh **3.2522**、
   analytic 3.2486，幾何值 3.2500

**官方批次（446 幀、0 例外、0 問題幀）**：

| GT | mean | median | p90 | max |
|---|---|---|---|---|
| analytic | 0.2688 | 0.1723 | 0.7063 | 0.9206 |
| **mesh** | **0.2367** | 0.1779 | **0.4647** | 0.7165 |
| legacy（參考） | 0.2428 | 0.1788 | 0.4528 | 0.9606 |

**網格偏差的性質**：局部的，不是整條路的起伏。偏差值直方圖**雙峰** ——
62.8% 落在 ±5 mm、13.8% 落在 +45~+50 mm。平坦段（s=5~50）std 只有 1.67 mm，
凹陷段（s=52~56）平均 −53.24 mm，抬升段（s=62~76）平均 +39.75 mm；二階差分的
尖峰只在區段**邊界**。像相鄰路面資產之間的接縫高差，不是三角化誤差。

> ⚠⚠ **這一節被我改寫三次，前兩版的結論都是錯的。錯誤原因記在這裡，不要重蹈**：
>
> 1. **拿絕對高度 MAE 當判準** —— GT 是 `P_world − cam_world`，換 `z_mesh` 只改
>    剖面點；短視距幀的窗口整個落在偏差區裡，GT 曲線幾乎整條**平移**，絕對高度
>    MAE 爆掉 +132%。那是平移不是形狀，扣掉每幀常數項後 mesh 反而較好。
>    （這個坑是使用者問「高度 GT 用 waypoint 本來就會有偏差」才發現的）
> 2. **pooled 相關 +0.13 是垃圾** —— 混進了 339 幀「mesh 與 analytic 完全相同」
>    （零訊號純雜訊），又沒扣每幀常數項。正確做法是**逐幀去均值 + 只看真的有
>    偏差的幀**，得到 +0.909
> 3. **把「兩個指標反向」讀成「mesh 是巧合」** —— 正確的讀法是「其中一個指標壞了」
> 4. **短窗口的相關要有排列對照** —— 兩條平滑曲線在 6 m 窗口上 |corr|>0.5 的
>    機率有 64%。但對照分佈**對稱於 0**，實際是單邊 +0.909，所以訊號是真的

**未解**：相機只看到約 **59%** 的振幅。可能是碰撞網格與渲染網格仍有細部差異，
也可能是 pipeline 已知的振幅衰減（見「其他」節）。本資料集分不開 —— 偏差區段
正好落在**坡頂**，能看到它的幀全部是短視距（可見深度中位數 6.14 m vs 其他
16.95 m；篩掉可見深度 >10 m 後一幀不剩）。這是幾何上的必然，不是取樣不巧。

**採集密度已改並實跑**：`_PROFILE_STEP_M` 1.0 → **0.125 m**，對齊 legacy GT 的
密度（18 km/h @ 40 fps，相機每幀前進 0.1246 m）。新資料集
`carla_dataset_Town03_20260822_021432`（459 幀、每幀 401 點、`mesh_label` 100%
`Roads`、`z_mesh` 零空值）。

**車速不受影響**（先前的顧慮得到實測確認）：全程 std 看似變差是**起步暫態**
（前 5 m 從 12.2 加速到 18 km/h）。里程 >5 m 之後：

| 資料集 | 車速 std |
|---|---|
| step=1.0（000716） | 0.079 |
| step=0.125（021432） | 0.088 |
| 更舊（195221） | 0.082 |

三者無法區分 —— 同步模式隔離 RPC 成本這件事成立。

### ⚠ 加密踩到一個 bug：`pitch_at` 的逐點微分

加密後 mesh GT 的 MAE **反而惡化** 0.2545 → 0.2834，且「網格結構」與 legacy GT
的相關從 **+0.946 崩到 +0.312**。原因不在資料，在 `RoadProfileGT.pitch_at`：

它原本用逐點 `np.gradient`。在 1 m 間距下**那本身就是 ±1 m 的中央差分**，
無意間在當低通濾波器。改成 0.125 m 之後那個隱含的低通消失，射線高度的量化
誤差（0.1 mm 級）被除以 0.125 m 放大成斜率雜訊。高頻雜訊隨間距是 **U 形**的：

| 間距 | 0.12 | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| 高頻雜訊（度） | 0.350 | 0.185 | **0.181** | 0.227 | 0.345 | 0.774 |

**修法**：`pitch_at` 改用**固定物理視窗**的最小平方斜率，預設 ±1.0 m
（`RoadProfileGT.PITCH_WINDOW_M`），**不隨採樣密度變**。

- 向後相容：舊資料集（1 m）的 MAE 一模一樣（analytic 0.2688 / mesh 0.2367）
- 新資料集 mesh 由 0.2834 復原到 **0.2545**，相關由 0.312 回到 **0.946**
- **analytic 對視窗不敏感**（0.2706~0.2719）—— 解析曲線本來就平滑，
  受影響的只有 mesh
- 為何不用估測器同款的 ±max(1, 0.15z)：結果差不多（mesh 0.2539）但遠處過度
  平滑會沖淡結構（預測與網格的相關 0.653 → 0.521），而且會讓 GT 耦合到
  估測器的參數

> ⚠ **教訓**：「用更密的取樣」不是免費的。逐點微分的雜訊放大與間距成反比，
> 換密度時必須同時檢查所有**隱含依賴取樣間距**的算式。我一開始還用「相鄰點
> pitch 跳動」當平滑度指標，那在不同間距下根本不可比（平滑曲線本來就會小
> 8 倍），差點得出「加密沒問題」的錯誤結論。

**試過但行不通的替代方案**：離線把所有幀的剖面點在世界座標下合併可到 1 cm，
但更差 —— 單幀那批點是同一瞬間、同一相機姿態打出來的、內部一致；跨幀合併會
混入配準誤差（用**里程**對齊更糟，7.7 mm 假散布 vs 用 (x,y) 的 0.27 mm）。
**密度只能在採集時給。**

代價是每幀 `next()`/`cast_ray` 從 51 次變 401 次，同步模式下只花牆鐘時間。
可用 `--profile-step-m` 調整。

腳本：`debug/diag_mesh_vs_analytic.py`、`debug/eval_mesh_gt{,2,3}.py`、
`debug/eval_mesh_gt_height.py`、`debug/dense_mesh_gt.py`、
`debug/plot_frame_gt_compare.py`（`debug/` 未進版控）。
示範幀：380 / 390（相機在凹陷邊緣，窗口正好蓋住它）。

### 〔已完成的實作紀錄〕

**理由**：相機看到的、車子壓過的都是**網格**；`waypoint` 給的是解析曲線。
目前量到的「網格 − 解析曲線只差 3.6 mm」是**在相機腳下量的**（射線往下打），
而相機看的是**前方 5-40 m** —— 遠處的網格偏差（三角化密度、LOD）**完全沒量過**。

- [x] ~~`sample_road_profile()` 增加射線模式~~ —— 已完成。多收一個
      `world` 參數；`road_profile.csv` 新增 `z_mesh` / `mesh_label` 兩欄，
      解析高度仍留在原本的 `z` 欄（`libs/road_profile_gt.py` 讀的就是那欄，
      舊 reader 與舊資料集完全不受影響）。射線從解析曲面**上方 2 m** 起打，
      往下 8 m —— 起點必須抬高，否則從曲面上起算有可能整條射線都在網格下方
- [x] ~~射線要用 label 過濾~~ —— 已完成。`_ray_ground_z` 從
      `verify_carla_geometry.py` 移進 `get_carlaDataset.py` 成為共用的
      `ray_ground_z` / `GROUND_LABELS`（兩支腳本不再各留一份）。打不到地面
      語意就留空並記 `no-ground:<看到的標籤>`，不猜
- [x] ~~效能：怕 51 次 cast_ray 拖垮同步迴圈、車速守不住~~ —— **這個顧慮是
      多餘的**（2026-08-21 更正）。採集器跑的是同步模式且
      `fixed_delta_seconds = 1/camera_fps`，每次 `world.tick()` 推進固定的
      模擬時間，RPC 再慢也**不會**改變物理、PID 的 dt 或車速曲線；多出來的
      只有牆鐘時間，也就是「這趟採集在現實裡要跑比較久」。所以
      **車速 std 不該拿來當射線模式的驗收指標**。HUD 那行
      `Profile: X ms/frame wall-clock` 是用來估採集時長的，不是看有沒有超時；
      真的趕時間才用 `--no-profile-ray`
- [ ] 離線驗證已過：`uv run --no-sync python debug/stub_test_collector.py`
      涵蓋射線長度/起點、車身命中被跳過、打不到地面時留空、以及 CSV 兩欄並存
      （順手修好 stub 缺 `Vector3D.dot` 的既有壞掉問題 —— commit `8444727`
      改用 CARLA 內建向量運算後就沒跟上）

> ⚠ **驗收條件已更正（2026-08-21）**。原本寫「舊 GT 的波浪是懸吊，新 GT
> **不該**重現它，重現了反而是錯的」—— **那是錯的，方向剛好相反**，量過了：
>
> | 量測 | 結果 |
> |---|---|
> | 車身 pitch vs 解析路面斜率 | corr **0.9966**（地形解釋 99.3% 變異） |
> | 殘差 vs 縱向加速度 | corr **−0.004** → 不是煞車點頭/加速後仰 |
> | `veh_z` − 解析中心線高度 | std **16.2 mm**、全距 −54~+16 mm，自相關 3m=+0.77 / 12m=+0.69 |
> | 該高度偏差的斜率 vs 車身殘差 | std 0.333° vs 0.356°，corr **+0.662** |
>
> 也就是說**舊 GT 的波浪是車子跟著一個偏離解析中心線的真實表面走**，
> 不是懸吊雜訊。所以對齊網格的 GT **本來就應該部分重現它**。
> 這反而讓本項的前提更強：0.333° 與待解的 0.310° 波浪量級吻合。
>
> 仍然成立的只有一句：新 GT 建在相機座標系、用實測 transform，所以**車身
> 姿態在座標系層面會抵消** —— 但這推不出「舊 GT 的波浪是假的」。
>
> 已排除的替代解釋：
>
> - **載荷轉移**（上坡壓後懸吊）—— 坡度是位置的函數，所以這機制也會釘在
>   世界座標上、光看空間錨定分不出來。但扣掉坡度只從 **16.17 → 15.73 mm**
>   （不到 3% 變異，斜率僅 0.88 mm/deg），再扣縱向加速度也只到 15.66 mm
> - **側向偏移 / 路拱** —— 車到最近解析點平面距離 median 0.4 cm、p95 4.2 cm，
>   就算 2% 路拱也只有 0.8 mm
> - **懸吊共振** —— 等速下也會釘在世界座標上，但共振會在半波長處讓自相關
>   轉負，這裡到 12 m 都還是 +0.69，不像共振
>
> ⚠ **還沒對上的一個數字**：相機腳下射線量到的網格偏差只有 std 3.59 mm
> （相關長度 0.3 m），車輛高度偏差卻是 15.66 mm（相關長度數公尺），差 4.4 倍。
> 一部分可能是路段不同 —— 3.59 mm 來自 `20260730_235516` 那趟驗證，不是現行
> 資料集的路線。本項實跑後 `z_mesh` 會在**同一條路**上直接給出答案。
>
> 仍分不開的：`veh_z − 解析高度` 混了物理網格偏差 + 懸吊行程 + 輪胎變形。
> 診斷腳本：`debug/diag_body_vs_road_pitch.py`、
> `debug/diag_veh_height_vs_analytic.py`（`debug/` 未進版控）

### 其他候選（可先在現有資料上做）

- [ ] 在**同一批影像**上直接量漆緣間距（WWH-12 的
      `debug/measure_paint_edges.py` 手法），依世界位置分箱，分辨是
      「漆的間距真的在變」還是「pipeline 量錯」。新資料集的影像本身就是乾淨原圖
- [ ] 評估這條對現有結論的影響。`w_real` 的判定是用**高度殘差**做的，
      零均值的起伏不影響尺度，但會墊高 MAE、也可能污染振幅缺口的量測

### 其他

- [ ] **pitch 振幅比真實路面低 0.5~1.5%**（逐幀比較）。等價地，反解的 `cy` 有
      **+0.031 px/m** 的深度漂移。嫌疑在 pipeline 側，尚未定位
- [ ] **MAE 判準與幾何不一致**：同資料集同 GT 下 MAE 仍偏好 3.216
      （0.2541 vs 0.2720）。比例 −1.05% 正好對應上面那個振幅缺口 ——
      MAE 用「把 z 縮小」去補償 pipeline 把坡度估平的部分。
      **所以 `w_real` 要用高度/幾何定，MAE 只用來報告精度。**
      可否證的預測：修掉振幅缺口後，MAE 應與幾何一致
- [ ] **z > 15~20 m 沒有獨立幾何驗證** —— 反解深度與正向投影兩支診斷在地平線
      附近都會病態（會給出 `w_real` 5.66、W = 14 m 這種值），是診斷失效不是
      pipeline 的誤差，但遠段目前確實沒有第二個證據
- [ ] 連 CARLA server 看 HUD，確認坡度符號（上坡為正）—— WWH-8 遺留
- [ ] `lane_segmentation` 的 6 個 "magic（有動機）" 常數，用 CARLA GT 量化校準
      —— 見第 1a 節

## 兩條方法論教訓

1. **不要拿兩個都不準的量互相反解**（雙黃線間隙先後推出 0.1834、0.028，都錯；
   直接量才定案）
2. **不要把所有取樣點倒在一起做迴歸**，也不要拿沒有空間對齊的量做迴歸。
   本次三個錯誤結論（平滑偏差 0.9617、車身/路面振幅比 1.0213、3% 振幅缺口）
   全是這一類：pooled 迴歸的權重被不同幀的取樣密度扭曲，或兩欄記錄位置差 1.5 m

## 注意

- `carla_module/` 的推論路徑仍 **DEFERRED**（WWH-10）：`realtime_test.py` /
  `carla_visualization.py` 都 import 了已刪除的函式。動 CARLA 時要一起遷移到
  `lane_curve` / `sample_widths_from_curves` / `estimate_pitch_from_curves`
- `realtime_test.py` 也讀 `w_real`，遷移時記得語意是**內側邊到內側邊**
- 基準備份：`debug/outputs/pre_w325_baseline/`（舊資料集 + 回推 GT + 3.216，
  0.2331）與 `debug/outputs/pre_calibration_baseline/`（更早）
