"""路面 GT 的兩種來源，統一成同一個介面（``pitch_at`` / ``height_at``）。

``RoadProfileGT``（量測式，2026-08-18 起）
    資料集若有 ``road_profile.csv``，就用採集當下直接問地圖得到的前方路面
    剖面。剖面存的是 waypoint 的**世界座標**，配上每幀相機的完整 transform，
    投影到相機座標系即可::

        v    = P_world − cam_world
        z_gt = v · forward     ← 沿光軸的深度，正是 pipeline 的 z
        h_gt = v · up          ← 垂直於光軸的高度，正是 pipeline 的 Y_3d

    **不需要 offset 常數、不需要弧長換算、不經過車身姿態。**

``LegacyProfileGT``（回推式，舊資料集用）
    沒有 ``road_profile.csv`` 時退回 ``pitch_visualization`` 原本的做法：拿
    「車子後來開到那裡時的**車身** pitch」當前方 GT。保留原路徑不動，舊基準
    才能重現。

兩者的差異已量化（見 to-do.md 與 WWH-13）：相關 0.99742，但回推式在遠處偏
平（z=30 處差 +0.367°），且開到終點的幀沒有前方 GT。`w_real` 的物理值 3.25
就是換成量測式之後才浮現的。
"""
from pathlib import Path

import numpy as np
import pandas as pd

from libs.visualization.pitch_visualization import (gt_height_profile,
                                                    gt_pitch_profile)


def carla_basis(pitch_deg, yaw_deg, roll_deg):
    """CARLA/UE 慣例的 (forward, right, up) 單位向量，角度為度。

    已驗證：用車輛姿態加上掛載 (x=1.5, z=1.08) 重建相機世界座標，與採集時
    記錄的 ``cam_x/y/z`` 相差 max 0.13 mm。
    """
    p, y, r = np.radians(pitch_deg), np.radians(yaw_deg), np.radians(roll_deg)
    cp, sp, cy, sy, cr, sr = (np.cos(p), np.sin(p), np.cos(y),
                              np.sin(y), np.cos(r), np.sin(r))
    fwd   = np.stack([cp * cy, cp * sy, sp], -1)
    right = np.stack([cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, -cp * sr], -1)
    up    = np.stack([-cy * sp * cr - sy * sr, -sy * sp * cr + cy * sr, cp * cr], -1)
    return fwd, right, up


class RoadProfileGT:
    """量測式 GT：每幀的路面剖面，表達在該幀的相機座標系。"""

    def __init__(self, dataset_dir):
        dataset_dir = Path(dataset_dir)
        m = pd.read_csv(dataset_dir / "measurements.csv")
        prof = pd.read_csv(dataset_dir / "road_profile.csv")
        fwd, _, up = carla_basis(m.cam_pitch_deg.to_numpy(),
                                 m.cam_yaw_deg.to_numpy(),
                                 m.cam_roll_deg.to_numpy())
        cam = m[["cam_x", "cam_y", "cam_z"]].to_numpy()
        pos = {int(f): i for i, f in enumerate(m.frame_id)}

        self.dataset_dir = dataset_dir
        self._prof = {}
        for frame_id, sub in prof.groupby("frame_id"):
            i = pos.get(int(frame_id))
            if i is None:
                continue
            v = sub.sort_values("d_req_m")[["x", "y", "z"]].to_numpy() - cam[i]
            z, h = v @ fwd[i], v @ up[i]
            order = np.argsort(z)          # 投影後 z 未必單調（彎道/路拱）
            z, h = z[order], h[order]
            keep = np.concatenate(([True], np.diff(z) > 1e-9))
            self._prof[int(frame_id)] = (z[keep], h[keep])

    def describe(self):
        return (f"measured road profile ({len(self._prof)} frames, "
                f"{self.dataset_dir.name})")

    def has(self, frame_id):
        return int(frame_id) in self._prof

    def z_range(self, frame_id):
        z, _ = self._prof[int(frame_id)]
        return float(z[0]), float(z[-1])

    def height_at(self, frame_id, distances):
        """相機座標中的路面高度；超出剖面範圍回 NaN。"""
        if not self.has(frame_id):
            return np.full(len(np.atleast_1d(distances)), np.nan)
        z, h = self._prof[int(frame_id)]
        d = np.atleast_1d(np.asarray(distances, float))
        return np.interp(d, z, h, left=np.nan, right=np.nan)

    def pitch_at(self, frame_id, distances):
        """路面 pitch（度），由剖面的局部斜率求得；超出範圍回 NaN。

        注意這是**點取樣的真實斜率**，而 pipeline 的 pitch 經過 windowed
        Theil-Sen 平滑。兩者的不對稱在有曲率的路段值得留意（實測全域約 10%
        的 MAE 差異），但它解釋不掉尺度問題——見 WWH-13。
        """
        if not self.has(frame_id):
            return np.full(len(np.atleast_1d(distances)), np.nan)
        z, h = self._prof[int(frame_id)]
        d = np.atleast_1d(np.asarray(distances, float))
        slope = np.interp(d, z, np.gradient(h, z), left=np.nan, right=np.nan)
        return np.degrees(np.arctan(slope))


class LegacyProfileGT:
    """回推式 GT：舊資料集沒有 road_profile.csv 時的退路。"""

    def __init__(self, measurements, camera_offset_m=0.0, camera_height=None):
        self.measurements = measurements
        self.camera_offset_m = camera_offset_m
        self.camera_height = camera_height

    def describe(self):
        return "legacy GT rebuilt from collect_dist_m + body pitch"

    def has(self, frame_id):
        return bool((self.measurements["frame_id"] == int(frame_id)).any())

    def pitch_at(self, frame_id, distances):
        return gt_pitch_profile(self.measurements, int(frame_id), distances,
                                self.camera_offset_m)

    def height_at(self, frame_id, distances):
        if self.camera_height is None:
            raise ValueError("camera_height required for legacy height GT")
        return gt_height_profile(self.measurements, int(frame_id), distances,
                                 self.camera_height,
                                 camera_offset_m=self.camera_offset_m)


def load_profile_gt(measurements_csv, *, camera_offset_m=0.0, camera_height=None,
                    measurements=None):
    """依資料集內容挑 GT 來源：有 road_profile.csv 就用量測式，否則回推式。"""
    path = Path(measurements_csv)
    if (path.parent / "road_profile.csv").exists():
        return RoadProfileGT(path.parent)
    if measurements is None:
        measurements = pd.read_csv(path)
    return LegacyProfileGT(measurements, camera_offset_m, camera_height)
