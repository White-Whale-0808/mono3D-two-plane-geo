# TODO — mono3D-two-plane-geo

兩大區塊：
1. [`lane_segmentation.py` 優化](#lane_segmentationpy-優化)
2. [Repo 層級待辦](#repo-層級待辦)

---

# lane_segmentation.py 優化

## 高優先

### 1. 把具名常數搬進 config（影響準度，跨相機高度可調）
模組頂部 34–46 行的常數命名清楚、有物理意義，但全部寫死、無法調整。
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

#### 1a. 這些常數是否有依據？(是否算 magic number)
重點結論：**幾何/論文給的是「縮放形式」(threshold 隨 y、車道寬、深度怎麼變)，
不是「係數本身」。** 所以即使 docstring 標為 "geometry-derived"，多數 scalar
仍是手選 → 嚴格定義下還是 magic number，只是「有動機的」。

分類（依依據強度）：

| 常數 | 形式來源 | scalar 本身 | 判定 | 可 ref |
|---|---|---|---|---|
| `_SEED_DELTA = 0.5` | 幾何 | 真推導：車可在自車道內任意橫向 → 內側標線 X∈(0±0.5)·w | **有依據（最壞界）** | 自身幾何；但保守，見 L133 TODO |
| `_MAX_GRADE_DEG = 15.0` | 工程標準 | 15°(~27%) 是道路最大縱坡的保守上界 | **有依據（外部標準）** | 道路幾何設計規範（如 AASHTO 縱坡上限） |
| `_TOL_PX_FLOOR = 3.0` | 感測器噪聲 | 3px 安全下限，可對應 ELSED 端點抖動 | **半 magic（經驗有依據）** | `docs/papers/ELSED_*.pdf`（定位精度） |
| `_TOL_LANE_FRACTION = 0.10` | 幾何(∝車道寬) | 10% 係數手選 | **magic（有動機）** | 無；建議用標線寬/ELSED 誤差統計推 |
| `_CROSS_LANE_FRACTION = 0.40` | 幾何(∝車道寬) | 40%<50% 中線給 margin，係數手選 | **magic（有動機）** | 無 |
| `_SEED_X_MAX = 8.0` / `_NOISE_X_MAX = 16.0` | 斜率↔橫向距離換算是幾何 | 8m/16m 距離手選 | **magic** | 無 |
| `_MODEL_MEMORY_M = 4.0` | — | 局部窗長手選 | **magic** | 無 |
| `_RESET_GAP_M = 2.0` | — | 平面變化門檻手選 | **magic** | 無 |
| `_GRADE_RAMP_Z0 / SPAN = 6.0` | — | 近場視為自車平面的距離手選 | **magic** | 無 |

小結：真正站得住腳的只有 `_SEED_DELTA`、`_MAX_GRADE_DEG`(部分 `_TOL_PX_FLOOR`)；
其餘 6 項仍應當作 magic number。投影/兩平面模型本身有 repo 內論文背書
（`Lin_&_Tsai_IEEETPAMI_1991.pdf`、`AI-Enhanced_Mono-View_*.pdf`），但**沒有任一篇
規定這些係數的具體數值**。

- [ ] 待辦：對「magic（有動機）」這 6 項，用 CARLA GT 量化校準（如標線寬、橫向偏移
      p95、實際換面距離分佈），把手選值換成資料推導值，並在註解標明來源
- [ ] 待辦：對「有依據」3 項，在註解補上明確 ref（規範名稱 / repo 論文路徑）

### 2. 修失效的文件參照（成本最低）
模組 docstring（第 7、11 行）和註解指向：
- `docs/lane_segmentation_issues.md`（問題 4）
- `docs/lane_segmentation_parameter_problem.md`

但 `docs/` 底下只剩 `papers/`，這兩個檔不存在，讀 code 的人找不到設計依據。

- [ ] 補回文件，或更新註解指向正確位置（例如 `docs/papers/lane_segmentation_design_logic.drawio`）

## 中優先

### 3. 抽出仍硬寫、且連名字都沒有的數字
這些比第 1 點更值得抽出，因為完全沒有說明（皆為 magic number）：

- [ ] L127 legacy gate：`0.5 * min_slope * (mid_y / img_height)` — 0.5 無說明
- [ ] L158 / L173 `_fit_x_of_y`：`last_n=8`、最少 `>= 4` 點
- [ ] L283–284 legacy `assoc_window`：`2.0 + 1.5*missed`、`0.18 * center_x`
- [ ] L343：`missed > max(4, track_bands // 3)`
- [ ] L354：`missed >= 2`；L356：`track_points[-2:]`
- [ ] L431：`track_bands = max(int(num_bands), 16)` — 把 config 的 `num_bands=3` 默默改成 16，呼叫端不易察覺
- [ ] L62：`min_y_margin=0.05`

### 4. 殘留 TODO 與死參數
- [ ] L133 TODO：`replace 1.0 multiplier with p95 lateral offset from CARLA GT` 尚未完成
- [ ] `roi_far`（L386）標明 unused，僅為簽名相容保留 → 評估清掉
- [ ] `roi_near` / `min_slope` / `lane_band_tolerance` 僅 legacy 路徑使用 →
      幾何模式啟用後是雜訊，考慮集中到 legacy 分支或清理

## 低優先

### 5. 效能
- [ ] `_track_side` 每個 band 對全部 `infos` 線性掃描（L328、L236），左右各一次，
      複雜度 O(bands × N)。線段量大時可先按 `y` 對 infos 建索引/排序加速。
      目前資料量未必是瓶頸，屬低優先。

---

# Repo 層級待辦

## 高優先

### A. 兩平面模型只用在 CARLA，離線 pipeline 沒用到
專案名稱是 "two-plane geometry"，但 `fit_two_plane_model` 只在
`carla_module/realtime_test.py` 被呼叫；離線的 `pipeline.py::infer_one()`
只回傳 `pitch_per_depth`（per-depth-band Theil-Sen），沒算 two-plane。
- [ ] 決定離線路徑是否也要輸出 two-plane near/far pitch + knee
- [ ] `pipeline.py` L65–66 的 docstring 提到 `two_plane` 輸出，但實際 return
      只有 `pitch_per_depth` → 註解與程式碼不一致，需修正

### B. 完成本機 ELSED+PIDNet 的 MAE 驗證
`lane_segmentation.py` 2026-06-10 改寫後，本機 ELSED+PIDNet 的 MAE 驗證仍未做
（見記憶筆記）。改寫對準度的影響還沒量化。
- [ ] 跑 batch inference + profile MAE，與改寫前 baseline 比較

## 中優先

### C. 清掉已死的註解區塊
- [ ] `libs/inference/road_segmentation.py` L58–92：整段被註解掉的舊
      Resnet101 推論 + mask erosion/blur 程式碼，已不使用，可刪
- [ ] `libs/inference/lane_fitting.py` L3、L42：被註解掉的 RANSAC import 與用法
      （已有註解說明為何不用，可保留一行說明、刪掉死碼）

### D. 沒有任何單元測試
repo 內沒有自己的 test（`carla_module/realtime_test.py` 是整合測試，需 CARLA server）。
- [ ] 至少為 `pitch_estimation`（two-plane / 單平面 fallback）與
      `lane_segmentation` 幾何推導加幾個 pinhole 合成資料的單元測試

## 低優先

### E. config 與文件一致性
- [ ] `config` 內仍留多個被註解的舊參數（`threshold`、`mask_erosion_kernel`、
      `roi`、`ransac_residual_threshold`）→ 確認不再需要後清理
- [ ] README.md（398 行）內容是否同步到目前 pipeline，待對照一次
