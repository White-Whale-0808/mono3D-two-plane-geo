"""OpenLane (Waymo) -> 本專案的資料集格式（images / measurements.csv / road_profile.csv）。

pipeline 與 GT 模組**完全不用改**，靠三個轉換達成：

1. **GT 座標**：OpenLane 的 ``lane_lines[i]['xyz']`` 已經在相機光學座標系
   （x 前、y 左、z 上，原點在相機）—— 實測 extrinsic 旋轉 ≈ 單位矩陣（掛載
   pitch 僅 0.17°），且 ``u = fx·(−y)/x + cx``、``v = fy·(−z)/x + cy`` 能重現
   標註的 ``uv``。:class:`RoadProfileGT` 算的是 ``v = P_world − cam_world`` 再投
   影到相機基底，所以只要把相機位姿寫成**單位姿態、原點在 0**，世界座標就等於
   相機座標，``z_gt = x``、``h_gt = z``，不需要任何 offset 或弧長換算。

2. **虛擬內參**（本轉換器的重點）：OpenLane 202 段有 **68 組不同內參**，主點
   ``cy`` 介於 624.8~665.0（影像高 1280，中心 640）。本專案的幾何假設理想針孔
   ``cy = H/2``，那個偏移等效於 **−0.42°~+0.69° 的常數 θ0 偏誤**，比整體 MAE
   還大。所以每幀用單應矩陣 ``H = K_virt · K_orig⁻¹`` 重採樣到同一組虛擬內參
   （主點置中、焦距統一、順便完成縮放），輸出後整個資料集就是**同一台校準過的
   相機**，config 只要一組 f_x / f_y。

3. **逐 segment 一個資料集目錄**：近場自標定的 calibrator 與路線剖面圖都假設
   「一條連續路線」，把 99 段混成一個目錄會讓 ``collect_dist_m`` 失去意義。

篩選條件見 :data:`GRADE_MIN` 等常數與 ``--mode``；預設連虛線也收，
``measurements.csv`` 會留下 ``left_category`` / ``right_category``，
下游要再篩實線隨時可以，不必重轉。

用法::

    # 只轉標註（不需要影像，可先驗證 GT 串接）
    python scripts/convert_openlane.py --openlane D:/datasets/openlane \\
        --out D:/datasets/openlane_converted --no-images --limit-segments 3

    # 影像解開後連影像一起轉
    python scripts/convert_openlane.py --openlane D:/datasets/openlane \\
        --out D:/datasets/openlane_converted
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.env_setup import setup_env  # noqa: E402

# warp_image 會 import cv2（C extension），依 repo 規則必須先設定 DLL 搜尋路徑。
setup_env()

# ---------------------------------------------------------------- 標註常數
CAT = {0: 'unknown', 1: 'white-dash', 2: 'white-solid', 3: 'dbl-white-dash',
       4: 'dbl-white-solid', 5: 'w-ldash-rsolid', 6: 'w-lsolid-rdash',
       7: 'yellow-dash', 8: 'yellow-solid', 9: 'dbl-yellow-dash',
       10: 'dbl-yellow-solid', 11: 'y-ldash-rsolid', 12: 'y-lsolid-rdash',
       20: 'left-curbside', 21: 'right-curbside'}
SOLID = {2, 4, 6, 8, 10, 12}
DASHED = {1, 3, 5, 7, 9, 11}
PAINT = SOLID | DASHED          # curbside(20/21) 不是漆，paint_evidence 會擋掉

# ---------------------------------------------------------------- 篩選門檻
PROBE_X = 12.0        # 在這個深度決定「哪一條是內側線」（標註大多從 8 m 才開始）
GRADE_MIN = 0.30      # 60 m 內至少要有這麼多高程變化（m），否則沒有坡可估
GRADE_SPAN = 60.0
WIDTH_MIN, WIDTH_MAX = 2.5, 4.5   # 內側線間距的合理範圍（漆線中心對中心）

# ---------------------------------------------------------------- 剖面取樣
PROFILE_STEP_M = 0.25
PROFILE_MAX_M = 80.0

# ---------------------------------------------------------------- 虛擬相機
# 輸出影像大小；沿用本專案 config 的 resize_size = [512, 1024]（高, 寬）。
OUT_W, OUT_H = 1024, 512
SRC_W, SRC_H = 1920, 1280
# 虛擬焦距 = 202 段的中位數焦距（fx == fy 恆成立）依各軸縮放比例換算。
# 主點固定在正中央，這正是 pipeline 的理想針孔假設。
_F_SRC_MEDIAN = 2071.4
VIRT_FX = _F_SRC_MEDIAN * OUT_W / SRC_W     # 1104.75
VIRT_FY = _F_SRC_MEDIAN * OUT_H / SRC_H     # 828.56
VIRT_CX, VIRT_CY = OUT_W / 2, OUT_H / 2


def virtual_K():
    return np.array([[VIRT_FX, 0.0, VIRT_CX],
                     [0.0, VIRT_FY, VIRT_CY],
                     [0.0, 0.0, 1.0]])


def inner_pair(lane_lines, probe_x=PROBE_X):
    """自車道左右兩條內側線，各為 ``(y_at_probe, category, xyz)``；缺一回 None。"""
    left = right = None
    for lane in lane_lines:
        xyz = np.asarray(lane['xyz'], dtype=float)
        if xyz.shape[1] < 5 or not (xyz[0].min() <= probe_x <= xyz[0].max()):
            continue
        order = np.argsort(xyz[0])
        xyz = xyz[:, order]
        y = float(np.interp(probe_x, xyz[0], xyz[1]))
        if y > 0 and (left is None or y < left[0]):
            left = (y, lane['category'], xyz)
        elif y < 0 and (right is None or y > right[0]):
            right = (y, lane['category'], xyz)
    return left, right


def frame_record(data, accept):
    """把一幀標註變成 (量測列, 剖面陣列)；不合格回 None。

    剖面 = 兩條內側線高度的平均，也就是**車道中心的路面**，深度均勻取樣。
    """
    left, right = inner_pair(data['lane_lines'])
    if not (left and right):
        return None
    if left[1] not in accept or right[1] not in accept:
        return None

    width = left[0] - right[0]
    if not (WIDTH_MIN <= width <= WIDTH_MAX):
        return None

    xl, xr = left[2], right[2]
    z0 = max(xl[0].min(), xr[0].min())
    z1 = min(xl[0].max(), xr[0].max(), PROFILE_MAX_M)
    if z1 - z0 < 5.0:
        return None

    depth = np.arange(z0, z1 + 1e-9, PROFILE_STEP_M)
    h_left = np.interp(depth, xl[0], xl[2])
    h_right = np.interp(depth, xr[0], xr[2])
    height = 0.5 * (h_left + h_right)

    near = depth <= z0 + GRADE_SPAN
    grade = float(height[near].max() - height[near].min())
    if grade < GRADE_MIN:
        return None

    meta = dict(
        left_category=CAT.get(left[1], left[1]),
        right_category=CAT.get(right[1], right[1]),
        left_solid=int(left[1] in SOLID), right_solid=int(right[1] in SOLID),
        lane_width_gt=round(width, 4),
        z_gt_min=round(float(z0), 2), z_gt_max=round(float(z1), 2),
        grade_60m=round(grade, 4),
    )
    return meta, depth, height


def warp_image(src_path, dst_path, K_src):
    """用 H = K_virt · K_src⁻¹ 把影像重採樣到虛擬相機（含主點置中與縮放）。"""
    import cv2
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        return False
    H = virtual_K() @ np.linalg.inv(np.asarray(K_src, dtype=float))
    out = cv2.warpPerspective(img, H, (OUT_W, OUT_H), flags=cv2.INTER_AREA)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), out)
    return True


def convert_segment(seg_dir, out_dir, image_root, accept, want_images,
                    keep=None):
    """轉一個 segment；回傳寫出的幀數。

    ``keep`` 若給定，只轉 (segment, frame_stem) 在集合裡的幀（分層實驗用）。
    里程 ``travelled`` 仍累加**所有**幀，所以 collect_dist_m 不受篩選影響。
    """
    seg_dir, out_dir = Path(seg_dir), Path(out_dir)
    rows, profile_rows = [], []
    travelled, prev_pos = 0.0, None
    frame_id = 0

    for path in sorted(seg_dir.glob('*.json')):
        # scene.zip 夾帶的 macOS 資源檔（``._name.json``）長得像 JSON 但不是，
        # 而且在 Windows 上會用 cp950 解碼直接爆掉 —— 明確跳過並固定 UTF-8。
        if path.name.startswith('._'):
            continue
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)

        pos = np.asarray(data['pose'], dtype=float)[:3, 3]
        if prev_pos is not None:
            travelled += float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos

        if keep is not None and (seg_dir.name, path.stem) not in keep:
            continue
        got = frame_record(data, accept)
        if got is None:
            continue
        meta, depth, height = got

        if want_images:
            src = Path(image_root) / data['file_path']
            if not src.exists():
                continue
            if not warp_image(src, out_dir / 'images' / ('%06d.png' % frame_id),
                              data['intrinsic']):
                continue

        # 相機位姿寫成單位姿態、原點在 0 -> 世界座標即相機座標
        rows.append(dict(
            frame_id=frame_id,
            cam_x=0.0, cam_y=0.0, cam_z=0.0,
            cam_pitch_deg=0.0, cam_yaw_deg=0.0, cam_roll_deg=0.0,
            collect_dist_m=round(travelled, 4),
            source_frame=path.stem, **meta))
        for d, h in zip(depth, height):
            profile_rows.append((frame_id, round(float(d), 3),
                                 round(float(d), 4), 0.0, round(float(h), 5)))
        frame_id += 1

    if not rows:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / 'measurements.csv', index=False)
    pd.DataFrame(profile_rows,
                 columns=['frame_id', 'd_req_m', 'x', 'y', 'z']
                 ).to_csv(out_dir / 'road_profile.csv', index=False)
    with open(out_dir / 'metadata.json', 'w') as fh:
        json.dump(dict(source='OpenLane', segment=seg_dir.name,
                       frames=len(rows),
                       f_x=round(VIRT_FX, 4), f_y=round(VIRT_FY, 4),
                       c_x=VIRT_CX, c_y=VIRT_CY,
                       resize_size=[OUT_H, OUT_W],
                       camera_height=2.116, camera_forward_offset=0.0,
                       note='virtual intrinsics; camera pose written as identity'),
                  fh, indent=2)
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--openlane', required=True, type=Path,
                    help='解壓後的 OpenLane 根目錄（含 validation/ 與 images/）')
    ap.add_argument('--out', required=True, type=Path, help='輸出根目錄')
    ap.add_argument('--split', default='validation')
    ap.add_argument('--mode', choices=['paint', 'solid'], default='paint',
                    help="paint（預設）連虛線也收；solid 只收兩側實線")
    ap.add_argument('--no-images', action='store_true',
                    help='只轉標註，不處理影像（影像還沒下載時用）')
    ap.add_argument('--limit-segments', type=int, default=None)
    ap.add_argument('--frame-list', type=Path, default=None,
                    help='CSV（需有 segment / frame 欄），只轉列出的幀；'
                         '分層實驗用，見 debug/openlane_frame_tags.py')
    args = ap.parse_args(argv)

    accept = SOLID if args.mode == 'solid' else PAINT
    keep = None
    if args.frame_list:
        fl = pd.read_csv(args.frame_list, dtype={'frame': str})
        keep = set(zip(fl['segment'], fl['frame']))
        print('frame-list: %d frames / %d segments'
              % (len(keep), fl['segment'].nunique()))
    segments = sorted((args.openlane / args.split).glob('segment-*'))
    if keep is not None:
        wanted = {s for s, _ in keep}
        segments = [s for s in segments if s.name in wanted]
    if args.limit_segments:
        segments = segments[:args.limit_segments]

    total = 0
    kept_segments = 0
    for seg in segments:
        n = convert_segment(seg, args.out / seg.name, args.openlane,
                            accept, not args.no_images, keep)
        if n:
            kept_segments += 1
            total += n
            print('%5d frames  %s' % (n, seg.name))
    print('\n%d frames from %d/%d segments -> %s'
          % (total, kept_segments, len(segments), args.out))
    print('config: f_x=%.2f  f_y=%.2f  camera_height=2.116  '
          'camera_forward_offset=0.0  resize_size=[%d, %d]'
          % (VIRT_FX, VIRT_FY, OUT_H, OUT_W))


if __name__ == '__main__':
    main()
