# Change Log — carla_module 整合與 PIDNet 遷移

## 日期：2026-04-29

---

## 1. 模型遷移：ResNet101 DeepLabV3 → PIDNet

**舊做法**
- 使用 `libs/model/resnet101.py` 的 `build_inference_model()` 載入 DeepLabV3-ResNet101
- 輸出為 1-channel sigmoid 機率圖，搭配 `threshold=0.5` 產生二元遮罩
- 需從 config 讀取 `model.model_path`

**新做法**
- 使用 `libs/inference/road_segmentation.py` 的 `load_pidnet()` 載入 PIDNet-S
- 輸出為 19-class Cityscapes logits，以 `argmax` 取 class 0（road）產生遮罩
- 需從 config 讀取 `model.model_name` 和 `model.weight_path`
- 不再需要 `threshold` 參數

**Config 異動（`config/inference_road_lane_segmentation.yaml`）**
```yaml
# 新增
model_name: "pidnet_s"
weight_path: "pidnet_pretrained_model/PIDNet_S_Cityscapes_val.pt"
device: "cuda"   # 原為 "cpu"

# 保留（已棄用，僅供參考）
model_path: "models/road_segmentation_best.pth"
```

---

## 2. 檔案重組：散落檔案整合進 `carla_module/`

### 新增檔案

| 路徑 | 說明 |
|------|------|
| `carla_module/__init__.py` | 套件識別檔 |
| `carla_module/realtime_test.py` | 主腳本（從根目錄 `carla_realtime_test.py` 移入並更新） |
| `carla_module/carla_road_segmentation.py` | PIDNet 版 `predict_road_from_pil()`，取代舊 ResNet101 版本 |
| `carla_module/carla_visualization.py` | 車道視覺化（從 `libs/visualization/` 移入，內容不變） |
| `carla_module/change.md` | 本檔案 |

### 刪除檔案

| 路徑 | 原因 |
|------|------|
| `carla_realtime_test.py` | 移入 `carla_module/realtime_test.py` |
| `libs/inference/carla_road_segmentation.py` | ResNet101 版本已棄用，由新版取代 |
| `libs/visualization/carla_visualization.py` | 移入 `carla_module/carla_visualization.py` |

---

## 3. Import 路徑變更

`carla_module/realtime_test.py` 中的 import 異動：

```python
# 移除
from libs.model.resnet101 import build_inference_model
from libs.inference.carla_road_segmentation import predict_road_from_pil
from libs.visualization.carla_visualization import render_piecewise_fits_to_array

# 新增
from libs.inference.road_segmentation import load_pidnet, apply_road_mask
from carla_module.carla_road_segmentation import predict_road_from_pil
from carla_module.carla_visualization import render_piecewise_fits_to_array
```

---

## 4. sys.path 修正

腳本從根目錄移入子資料夾後，`__file__.parent` 改變：

```python
# 舊（在根目錄時）
sys.path.insert(0, str(pathlib.Path(__file__).parent))

# 新（在 carla_module/ 內）
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
```

---

## 5. 執行指令變更

```bash
# 舊
python carla_realtime_test.py

# 新
python carla_module/realtime_test.py
```
