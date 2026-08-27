# TODO — mono3D-two-plane-geo

> **本檔只留「還沒做的事」。** 已完成的項目與被推翻的假說一律移出，論證留在
> Linear 票（WWH-7 ~ WWH-15）與 commit 訊息裡。
>
> 行號最後對照程式碼是 **2026-08-27**（WWH-15 之後），動到相關檔案時請順手更新。
>
> **2026-08-22**：第三區塊改寫 —— CARLA 採集/標定/GT 全部定案。
> **2026-08-27**：WWH-15 結案後整理 —— 移出三個已完成項（下坡兩個失效模式、
> `infer_one` 無呼叫者、`infer_one` 不能選估測器）、刪除一個判定不重要項
> （`_track_side` 效能）、補上 WWH-15 新增的常數、全檔行號重新對照。

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

> ⚠ **2026-08-27 未決的設計爭議**：WWH-15 刻意把 `paint_evidence.py` 的門檻
> **留在模組常數**，理由是「有物理推導的門檻不進 config，以免邀請使用者手調
> 本來不該手調的東西」。本節與那個決定直接矛盾。建議先裁決分界線：
> **只搬「真的跟場景/相機綁定」的**（下表 magic 那 6 項，且應在校準後才搬），
> **有依據的（`_SEED_DELTA`、`_MAX_GRADE_DEG`、`_SUPPORT_MIN_LEN_PX`）留常數
> 但補推導註解**。決定之前不要動手。

- [ ] `_TOL_LANE_FRACTION = 0.10`（關聯容差 = 局部車道寬的 10%）
- [ ] `_TOL_PX_FLOOR = 3.0`（ELSED 端點噪聲下限 px）
- [ ] `_CROSS_LANE_FRACTION = 0.40`（搜尋不超過 40% 車道寬）
- [ ] `_SEED_DELTA = 0.5`（自車橫向位置不確定性）
- [ ] `_SEED_X_MAX = 8.0` / `_NOISE_X_MAX = 16.0`（斜率閘門的橫向公尺數）
- [ ] `_MODEL_MEMORY_M = 4.0`（局部模型擬合的深度範圍）
- [ ] `_RESET_GAP_M = 2.0`（深度跳變 → 可能換平面的閾值）
- [ ] `_MAX_GRADE_DEG = 15.0`（最壞路面坡度，WWH-15 後住在 `geometry.py`）
- [ ] `_GRADE_RAMP_Z0 = 6.0` / `_GRADE_RAMP_SPAN = 6.0`（坡度 slack 的 ramp，同上）
- [ ] `_SUPPORT_MIN_LEN_PX = 60.0`（WWH-6 新增，2026-07-03 首版未列）

#### 1a. 這些常數是否有依據？(是否算 magic number)
重點結論：**幾何/論文給的是「縮放形式」(threshold 隨 y、車道寬、深度怎麼變)，
不是「係數本身」。** 所以即使 docstring 標為 "geometry-derived"，多數 scalar
仍是手選 → 嚴格定義下還是 magic number，只是「有動機的」。

分類（依依據強度）：

| 常數 | 形式來源 | scalar 本身 | 判定 | 可 ref |
|---|---|---|---|---|
| `_SEED_DELTA = 0.5` | 幾何 | 真推導：車可在自車道內任意橫向 → 內側標線 X∈(0±0.5)·w | **有依據（最壞界）** | 自身幾何；但保守，見 L87 TODO |
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
      → 見[第 3 區塊](#carla-資料採集與標定)的未解項 4
- [ ] 待辦：對「有依據」的項目，在註解補上明確 ref（規範名稱 / repo 論文路徑）

### 2. 修失效的文件參照（成本最低）
模組 docstring **L7、L11** 仍指向：
- `docs/lane_segmentation_issues.md`（問題 4）
- `docs/lane_segmentation_parameter_problem.md`

但 `docs/` 底下只有 `papers/` 和 `diagrams/`，這兩個檔仍不存在（**2026-08-27 覆核仍失效**）。

- [ ] 補回文件，或更新註解指向 `docs/diagrams/lane_segmentation_flow.drawio`
      / `docs/papers/lane_segmentation_design_logic.drawio`

## 中優先

### 3. 抽出仍硬寫、且連名字都沒有的數字
這些比第 1 點更值得抽出，因為完全沒有說明（皆為 magic number）。行號已於 2026-08-27 更新：

- [ ] **L81** legacy gate：`0.5 * min_slope * (mid_y / img_height)` — 0.5 無說明
      （僅 `geom is None` 的 legacy 分支會走到）
- [ ] **L112 / L127** `_fit_x_of_y`：`last_n=8`、最少 `>= 4` 點
- [ ] **L245–246** legacy `assoc_window`：`2.0 + 1.5*missed`、`0.18 * center_x`
- [ ] **L314**：`missed > max(4, track_bands // 3)`
- [ ] **L325**：`missed >= 2`；同段 `track_points[-2:]`
- [ ] **L419**：`track_bands = max(int(track_bands), 16)`
      （WWH-7 已把參數名 `num_bands` → `track_bands` 並在 config 設 16，
      所以「默默改成 16」的坑已緩解；但 clamp 本身仍未說明理由）
- [ ] **`geometry.py` L31**：`min_y_margin=0.05`（WWH-15 抽出 `CameraGeometry`
      時從 lane_segmentation 搬過去的）

### 3b. `lane_fitting.py` / `pitch_estimation.py` / `paint_evidence.py` 的常數
WWH-9 與 WWH-15 各新增一批常數。它們的註解**普遍比 lane_segmentation 那批好**
（多數記了實測依據與受影響幀號），但同樣全部寫死：

`lane_fitting.py`
- [ ] `_SHADOW_MARGIN_PX = 3.0` / `_SHADOW_MIN_OVERLAP_ROWS = 8`（L10–11）
- [ ] `_FRAG_MAX_STEP_PX = 4.0`（L16）— **隱藏耦合**：註解說它是從
      `min_slope = 0.3` 推的，但 `min_slope` 在 config 裡可調，改了不會連動
- [ ] `_JUNCTION_TOL_PX = 20.0` / `_JUNCTION_SLOPE_ROWS = 10`（L23、L26）
- [ ] `_REFINE_SEARCH_PX = 3` / `_REFINE_MIN_GRAD = 4.0`（L33、L39）
- [ ] **（WWH-15 新增）** `_ZJUMP_ABS_M = 3.0` / `_ZJUMP_FRAC = 0.3`（L242–243）、
      `_ZJUMP_EXTRAP_FACTOR = 1.5`（L250）—— 深度連續性截斷的門檻

`paint_evidence.py`（**WWH-15 新增，L56–61**）
- [ ] `_STRIPE_M = 0.125`（CARLA 實測標線寬）、`_RIDGE_THR = 10.0`、`_PEAK_PX = 4`、
      `_FAR_CAP_PX = 60`、`_SEG_SAMPLES = 9`、`_TRUNC_MIN_RUN = 5`
      ⚠ **這批是刻意不進 config 的**（見第 1 點的設計爭議）；列在這裡是為了
      清單完整，不是說一定要搬。`_STRIPE_M` 是唯一真的綁地圖的（換地圖要改）

`pitch_estimation.py`
- [ ] `WINDOW_FRAC = 0.15` / `WINDOW_MIN_M = 1.0`（L7–8）—— 這兩個是 windowed
      估測器**明示的空間解析度**，最該進 config
- [ ] 函式預設值：`z_cap_m=45.0`（L142、L211）、`min_valid_range_m=0.5`（L141、L210）、
      `min_window_points=4`（L145）、`n_pitch_samples=200`（L146、L215）、
      `resid_mad_k=5.0`（L212，spline 路徑）

### 4. 殘留 TODO 與死參數
- [ ] **L87** TODO：`replace 1.0 multiplier with p95 lateral offset from CARLA GT` 尚未完成
- [ ] `roi_far`（**L372**）仍標明 unused、僅為簽名相容保留 → 評估清掉
- [ ] `roi_near`（L371）/ `min_slope`（L368）/ `lane_band_tolerance`（L370）仍僅
      legacy 路徑使用 → 幾何模式啟用後是雜訊，考慮集中到 legacy 分支或清理
      （注意 `min_slope` 另被 `_FRAG_MAX_STEP_PX` 的推導引用，見 3b）

---

# Repo 層級待辦

## 高優先

### A. `pipeline.py` 與文件的一致性
> WWH-15（2026-08-27）已完成本節原本的兩項：batch runner 改走 `infer_one`、
> `infer_one` 加了 `method` 參數。以下是剩下的。

- [ ] **`pipeline.py` 的 docstring 又過期了**：L9 和 L91 都寫
      "continuous spline pitch(z)"，但 WWH-9 之後預設估測器是 **windowed**
- [ ] **單張 runner 仍攤開 pipeline**：`utils/inference_road_lane_segmentation.py`
      要畫中間產物，所以 WWH-15 選擇「保持攤開但照 pipeline 原樣插入三道閘門」。
      → 要消除這份拷貝，得讓 `infer_one` 有 debug 模式吐中間產物。
      **教訓：改 pipeline 記得有兩份拷貝要同步**
- [ ] 專案名稱仍是 "two-plane geometry"，但現在的輸出是連續 pitch(z) 曲線，
      沒有 near/far 兩平面 + knee 的概念了 → 決定要不要重新引入，或更新命名/文件
      （純命名決策，沒有技術債後果，不急）

## 中優先

### C. 清掉已死的註解區塊
- [ ] `libs/inference/road_segmentation.py` **L56–L75**：整段被註解掉的舊
      Resnet101 推論程式碼，已不使用，可刪。另 **L84–L90** 的
      erosion / GaussianBlur / morphology 死碼（L79–82 已有說明為何放棄，
      可保留說明刪死碼）

### D. 沒有任何單元測試
**2026-08-27 覆核：repo 內仍無自己的 test。**（`carla_module/realtime_test.py`
是需要 CARLA server 的整合測試；`elsed_src/pybind11/tests/` 是 vendored 的第三方測試。）
WWH-8 的等價性測試、WWH-15 的煙霧測試都是臨時腳本，沒有進版控。

- [ ] 至少為 `pitch_estimation`（windowed / spline / degenerate fallback）與
      `lane_segmentation` 幾何推導加幾個 pinhole 合成資料的單元測試
- [ ] `debug/` 底下的診斷腳本（`check_width_calibration.py` 等）有現成的
      合成/實測驗證邏輯，可以抽成正式測試
- [ ] WWH-15 的三道閘門（`filter_paint_segments` / `truncate_at_evidence_break` /
      `truncate_at_depth_jump`）都是純函式、易測，且各自有踩過坑的邊界案例
      （leading-drop、大列距缺口），優先補

## 低優先

### E. config 與文件一致性
- [ ] `config` 內仍留 3 個被註解的舊參數：`threshold`(L15)、
      `mask_erosion_kernel`(L16)、`roi`(L23) → 確認不再需要後清理
      （`ransac_residual_threshold` 已不存在；L5 的替代 `weight_path` 是刻意保留）
- [ ] README.md 內容是否同步到目前 pipeline，待對照一次。已知 **L28** 說
      `infer_one` 是 "core entry point" —— batch runner 現在確實走它，但單張
      runner 仍自己攤開，說法只對一半

### B2. windowed pitch 的 95 ms 延遲（隨 CARLA 一起解凍）
- [ ] `estimate_pitch_windowed` 每幀呼叫 200 次 `theilslopes`，pitch 階段
      從 5 ms 變 ~95–135 ms。批次無所謂，**CARLA 即時前必須處理**。
      `carla_module/` 整個 DEFERRED（WWH-10），所以這項也一起壓著

### G. profile 圖依 z 切成固定跨距（原型已備，等決策）
- [ ] 不同幀的 z 範圍差很多（2.5–10 m vs 9.5–45 m），同樣 8 inch 寬的圖
      造成 m/inch 差 5 倍，遠距幀的細節看不見。原型已做在
      `debug/viz_profile_split.py`（每張固定 5 m，`--yscale shared|tile`，
      建議 tile）。**使用者 2026-07-27 指示先不動畫圖**，待日後決定是否進 libs

---

# CARLA 資料採集與標定

**採集、標定、GT 來源全部定案**（WWH-11 / 12 / 13 / 14 均已 merge）。本節只留
**還在用的參考值**與**未解項**；已完成的實作紀錄、被推翻的假說與完整論證留在
Linear 票裡，不再佔這裡的篇幅。

## 名詞

| 名詞 | 意思 |
|---|---|
| **analytic GT**（`z` 欄） | `road_profile.csv` 的 waypoint 高度 = OpenDRIVE 解析中心線。平滑，不含實作出來的路面細節 |
| **mesh GT**（`z_mesh` 欄） | 採集當下從剖面點上方 2 m 往下打射線，命中路面網格的高度。**現行預設**（`height_source: auto`） |
| **legacy GT** | 沒有 `road_profile.csv` 時的退路：拿車子後來開到那裡的**車身** pitch 當前方 GT |
| **可見深度 / 跨距** | 該幀能量到車道寬的 z 範圍（`z_visible_min..max`）與其長度 |
| **曲率** | GT pitch 在該幀可見視窗內的最大變化率（°/m）。**目前 MAE 的主要驅動量** |
| **`dy_far`** | 追蹤到的車道列 − GT 說「該深度的路面應該在的列」，取遠半段中位數。正常 −1~−5 px |
| **空間錨定** | 依世界位置分箱，比較「箱平均的變異」與「箱內跨幀散布」，判斷某偏差屬於那段路還是屬於處理流程 |

## 已定案的標定值

| 參數 | 值 | 依據 |
|---|---|---|
| `camera_height` | **1.08** | 直接量到 1.0816（坡度修正後，三種量法全距 0.0001） |
| `f_x` / `f_y` | **512 / 455** | fov 90° @1280 反推 f=640，resize 後；`lens_circle_multiplier=0` 理想針孔 |
| `cy` | **256** | 主點嚴格在中心；用量測 GT 反解逐幀 std 僅 0.30 px |
| `camera_forward_offset` | **1.5** | 沿車輛前進軸恆定；二維掃描顯示是尖銳最佳值 |
| `w_real` | **3.25** | 四條獨立幾何路徑收斂到 3.243~3.250 |

> ⚠ **3.25 綁定這條路的標線組合**：左雙黃、右單白。`w_real` 是內緣到內緣，兩側
> 從 3.5 m 的邊界中心寬各內縮不同的量（左 0.1875、右 0.0625，合計 0.25）。
> 同樣 3.5 m 車道下：單+單 3.375、**雙+單 3.25**（本資料集）、雙+雙 3.125。
> **不要把 3.25 帶到別的地圖或別條車道。**

`libs/road_profile_gt.py` 把剖面投影到相機座標系：`v = P_world − cam_world`，
`z_gt = v·forward`、`h_gt = v·up`。**不需要 offset 常數、不需要弧長換算、
不經過車身姿態。** 舊資料集自動退回 legacy GT。

## 資料集與現行基準（三條路線，數字為 WWH-15 之後）

三份都是 0.125 m 剖面間距、401 點/幀、`z_mesh` 零空值、`mesh_label` 100% `Roads`。

| 資料集 | 幀 | 路線 | road_pitch | 有效幀 | mean | median | p90 | >2° |
|---|---|---|---|---|---|---|---|---|
| `..._uphile` | 459 | 57 m 上坡 | +0.2 ~ **+12.4°** | 436 | **0.2500** | 0.1764 | 0.5216 | 0 |
| `..._down_hile` | 591 | 74 m 下坡 | **−12.4** ~ +4.4° | 447 | **0.3010** | 0.2307 | 0.5897 | 0 |
| `..._full_road` | 1371 | 171 m 全段 | −23.3 ~ +9.6° | 1115 | **0.2072** | 0.1340 | 0.4210 | 0 |

WWH-15 之前的 baseline（供對照）：mean 0.2545 / 0.4633 / 0.2143，>2° 為 0 / 10 / 1。
⚠ **不同資料集的絕對值不可互比**（不同路段、不同曲率分佈）。

**`down_hile` 是 `uphile` 的反方向**（pitch 極值鏡像、對向車道，y 差 1.75 m）。
兩趟在世界座標 x≈68 都量到同一個網格凹陷（−69 / −67 mm）。

**沒有輸出的幀不是 bug，是路口**：`down_hile` 0-121、`full_road` 871-1047 都在
同一個坡頂十字路口（`veh_z` 8.02、無車道標線），`uphile` 最後 23 幀在坡頂。
方法需要兩條內緣線，那裡本來就給不出東西。另有 `down_hile` 5 幀
（122/134/136/141/243）被 WWH-15 的漆料閘門判為「右側整條無可驗證漆」而誠實棄權。

`libs/visualization/route_profile_visualization.py` 會在 batch 末尾畫整條路線的
地形剖面（高度 analytic vs mesh、`mesh − analytic`、每幀 MAE 對齊到里程）。

## 未解項

### 1. MAE 有一部分量的是**解析度差異**，不是估測誤差

排除下坡失效模式後，誤差幾乎全由 **GT pitch 的曲率**解釋，與可見跨距無關
（下表是 WWH-15 之前算的，三份合計 n≈1970，格內為 MAE mean；閘門修掉的正是
右下角那些高曲率格，趨勢本身不變）：

| 跨距＼曲率 | ≤0.5 | 0.5–1 | 1–2 | >2 °/m |
|---|---|---|---|---|
| 0–3 m | 0.102 | 0.228 | 0.321 | **0.714** |
| 3–6 m | 0.109 | 0.345 | 0.417 | 0.502 |
| 6–12 m | 0.109 | 0.242 | 0.442 | 0.547 |
| >12 m | 0.122 | 0.202 | 0.193 | — |

偏相關：`MAE~曲率 |` 控制 1/跨距 = **+0.39**；`MAE~1/跨距 |` 控制曲率 = **−0.06**。
→ **「短視距造成誤差」是假關聯**（短視距與高曲率都發生在坡頂）。

機制：估測器用 ±max(1 m, 0.15z) 的 Theil-Sen 視窗，GT 用固定 ±1 m 最小平方，
路面 pitch 變化快的地方兩者本來就會差，差多少正比於曲率。
平坦區（曲率 <0.5°/m）三份都是 **0.10~0.12°** —— 這才是方法本身的精度。

- [ ] 把 GT 換成估測器同款視窗**當診斷跑一次**，看曲率那條斜率剩多少，
      藉此拆出「真的估錯」的部分。⚠ 只當診斷 —— WWH-14 已否決把預設 GT 耦合
      到估測器參數

### 2. 網格振幅缺口：相機只看到約 59%

WWH-14 未解項。逐幀迴歸斜率 +0.588：相機確實看得到 `z_mesh` 的凹陷，但只有
約 59% 的振幅。可能是碰撞網格與渲染網格仍有細部差異，也可能是 pipeline 已知的
振幅衰減（見第 4 項）。WWH-14 的資料集分不開 —— 偏差區段正好在坡頂，看得到它
的幀全是短視距。

- [ ] **`down_hile` 正好提供了缺的觀測**：同一個凹陷（世界 x≈68）這次是從
      **下坡方向、長視距**看到的。可以直接拆
- [ ] 原本的顧慮是「那一帶（s=18~30 m）的幀被坡頂遮蔽污染」—— WWH-15 已修掉
      該失效模式，直接用 gated 的 `outputs/measurements_down_hile.csv` 即可

### 3. 隱含車道寬隨世界位置變動

用 GT 深度做正向投影（不是拿 pipeline 深度反算，那是恆等式），依世界位置分箱：
箱平均變異 **0.0302 m** vs 箱內跨幀散布 **0.0156 m**，比值 **1.93**，量級約 1%
→ 換算 pitch 約 0.1°。⚠ 中段（24-46 m，取樣最充分）其實穩定在 3.242~3.262
（±0.3%），大幅偏離集中在路線兩端，**選樣偏差未控制**。

- [ ] 在**同一批影像**上直接量漆緣間距（WWH-12 的 `debug/measure_paint_edges.py`
      手法），依世界位置分箱，分辨是「漆的間距真的在變」還是「pipeline 量錯」

### 4. 其他

- [ ] **pitch 振幅比真實路面低 0.5~1.5%**（逐幀比較）。等價地，反解的 `cy` 有
      **+0.031 px/m** 的深度漂移。嫌疑在 pipeline 側，尚未定位
- [ ] **MAE 判準與幾何不一致**：同資料集同 GT 下 MAE 仍偏好 3.216
      （0.2541 vs 0.2720）。比例 −1.05% 正好對應上面那個振幅缺口 —— MAE 用
      「把 z 縮小」去補償 pipeline 把坡度估平的部分。**所以 `w_real` 要用
      高度/幾何定，MAE 只用來報告精度。** 可否證的預測：修掉振幅缺口後，
      MAE 應與幾何一致
- [ ] **z > 15~20 m 沒有獨立幾何驗證** —— 反解深度與正向投影兩支診斷在地平線
      附近都會病態，是診斷失效不是 pipeline 的誤差，但遠段確實缺第二個證據
- [ ] 連 CARLA server 看 HUD，確認坡度符號（上坡為正）—— WWH-8 遺留
- [ ] `lane_segmentation` 的 6 個「magic（有動機）」常數，用 CARLA GT 量化校準
      —— 見第 1a 節

## 方法論教訓

1. **不要拿兩個都不準的量互相反解**（雙黃線間隙先後推出 0.1834、0.028，都錯；
   直接量才定案）
2. **不要把所有取樣點倒在一起做迴歸**，也不要拿沒有空間對齊的量做迴歸。
   pooled 迴歸的權重會被不同幀的取樣密度扭曲
3. **對齊要用世界座標，不要用里程**（里程 7.7 mm 假散布 vs (x,y) 的 0.27 mm）。
   路線會立體交叉時**最近鄰要算 3D** —— `full_road` 先走天橋、後來從橋下穿過，
   只比 (x, y) 會長出一個 4 m 深的假深谷
4. **換取樣密度時要檢查所有隱含依賴間距的算式**。逐點微分的雜訊放大與間距成
   反比（`pitch_at` 就踩過），這種依賴不會出現在介面上
5. **比較「散布」前先確認坡度已消掉**。0.25 m 的格子跨在 12° 坡上，光坡度就讓
   格內高度差 50 mm；要比就比偏差量（`z_mesh − z`）或每單位長度的值
6. **相關的兩個解釋量要做偏相關**。「短視距造成誤差」看起來相關 0.46，控制曲率
   後只剩 −0.06
7. **（WWH-15）門檻的絕對值/相對值在「大缺口」上不可判定**。z 跳截斷在 105 列
   的大缺口上誤觸發（連續路面走那麼多列本來就前進 3.5 m），必須配「連續路面
   外推上界」`z_exp = z1·(y1−cy)/(y2−cy)` 才能分辨。同理，光度閘的亮峰窗必須
   固定 px 而非由 z 換算 —— 換算值在地平線附近被坡度餘裕撐到 23 px 就失效

## 注意

- `carla_module/` 的推論路徑仍 **DEFERRED**（WWH-10）：`realtime_test.py` /
  `carla_visualization.py` 都 import 了已刪除的函式。動 CARLA 時要一起遷移到
  `lane_curve` / `sample_widths_from_curves` / `estimate_pitch_from_curves`，
  並補上 WWH-15 的三道閘門
- `realtime_test.py` 也讀 `w_real`，遷移時記得語意是**內側邊到內側邊**
- 基準備份：`debug/outputs/pre_w325_baseline/`（舊資料集 + 回推 GT + 3.216，
  0.2331）與 `debug/outputs/pre_calibration_baseline/`（更早）
