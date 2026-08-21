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
    """量測式 GT：每幀的路面剖面，表達在該幀的相機座標系。

    ``height_source`` 選剖面點的高度要用哪一欄：

    ``"analytic"``
        ``z`` —— waypoint 的高度，也就是 OpenDRIVE 的**解析中心線**。
    ``"mesh"``（預設）
        ``z_mesh`` —— 採集當下向下射線打到路面網格的高度（WWH-14 起才有這欄；
        舊資料集沒有，會直接報錯而不是默默退回解析值）。

    **預設是** ``"mesh"``（WWH-14，2026-08-22 定案）。資料集
    `carla_dataset_Town03_20260822_000716`，三條獨立證據：

    1. 偏差是真的幾何，不是 ``cast_ray`` 的假象 —— :class:`LegacyProfileGT`
       （車身姿態，動力學）與 mesh（射線，幾何）相對 analytic 多出來的結構，
       逐幀去均值後 corr median **+0.990**、**100%** 的幀 >0.5。兩條完全獨立
       的量測路徑記錄到同一個東西。
    2. 相機看得到 —— 預測相對 analytic 的偏離 vs 網格偏差，逐幀去均值 corr
       median **+0.909**、81.2% >0.5；排列對照（別幀的形狀 / 隨機平滑曲線）
       只有 −0.084 / −0.005（32%）。逐幀迴歸斜率 **+0.588**。
    3. ``w_real`` 判定更接近幾何值 —— 零平均高度殘差：mesh **3.2522**、
       analytic 3.2486，幾何值 3.2500。

    官方批次（446 幀）：analytic mean 0.2688 / p90 0.7063；
    mesh mean **0.2367** / p90 **0.4647**。

    ⚠ **不要用絕對高度 MAE 判定這件事。** GT 是 ``P_world − cam_world``，換
    ``z_mesh`` 只改剖面點；短視距幀的窗口整個落在偏差區裡，GT 曲線幾乎整條
    平移，絕對高度 MAE 會爆掉（+132%）—— 但那是平移不是形狀，扣掉每幀常數項
    後 mesh 反而較好。這個坑我踩過，一度得出「mesh 明顯更差」的錯誤結論。

    未解：相機只看到約 **59%** 的振幅。可能是碰撞網格與渲染網格仍有細部差異，
    也可能是 pipeline 已知的振幅衰減（pitch 振幅比真實低 0.5~1.5%）。本資料集
    分不開 —— 偏差區段正好落在坡頂，能看到它的幀全部是短視距（可見深度中位數
    6.14 m vs 其他 16.95 m）。

    網格偏差的性質：偏差值直方圖是**雙峰** —— 62.8% 落在 ±5 mm、13.8% 落在
    +45~+50 mm。平坦段（s=5~50）std 只有 1.67 mm，抬升段（s=62~76）平均
    +39.75 mm，凹陷段（s=52~56）平均 −53.24 mm；二階差分的尖峰只出現在區段
    **邊界**。像相鄰路面資產之間的接縫高差，不是三角化誤差。

    ``x`` / ``y`` 兩欄不受影響：射線是垂直往下打的，只換高度。
    """

    #: pitch 的預設視窗半寬（m）。見 :meth:`pitch_at` —— 固定物理長度，
    #: 不隨 road_profile.csv 的採樣間距變動。
    PITCH_WINDOW_M = 1.0

    def __init__(self, dataset_dir, height_source=None, pitch_window_m=None):
        """``height_source=None``（預設）= 自動：有 ``z_mesh`` 就用，沒有就
        退回 ``"analytic"``。明確指定 ``"mesh"`` 而資料集沒有那欄則報錯 ——
        自動選擇是為了讓 WWH-14 之前的舊資料集照常可用，不是讓打錯的設定
        默默生效。實際選到哪一個看 :meth:`describe`。"""
        if height_source == "auto":      # config 用字串表示「自動」
            height_source = None
        if height_source not in (None, "analytic", "mesh"):
            raise ValueError(f"height_source must be 'auto'/None, 'analytic' "
                             f"or 'mesh', got {height_source!r}")
        dataset_dir = Path(dataset_dir)
        m = pd.read_csv(dataset_dir / "measurements.csv")
        prof = pd.read_csv(dataset_dir / "road_profile.csv")

        if height_source is None:
            height_source = "mesh" if "z_mesh" in prof.columns else "analytic"
        z_col = "z" if height_source == "analytic" else "z_mesh"
        if z_col not in prof.columns:
            raise ValueError(
                f"{dataset_dir.name}/road_profile.csv 沒有 {z_col!r} 欄"
                f"（height_source={height_source!r}）。射線高度是 WWH-14 起才"
                f"採集的，舊資料集只能用 height_source='analytic'。")
        n_before = len(prof)
        prof = prof[prof[z_col].notna()]
        self.n_dropped = n_before - len(prof)

        fwd, _, up = carla_basis(m.cam_pitch_deg.to_numpy(),
                                 m.cam_yaw_deg.to_numpy(),
                                 m.cam_roll_deg.to_numpy())
        cam = m[["cam_x", "cam_y", "cam_z"]].to_numpy()
        pos = {int(f): i for i, f in enumerate(m.frame_id)}

        self.dataset_dir = dataset_dir
        self.height_source = height_source
        self.pitch_window_m = (self.PITCH_WINDOW_M if pitch_window_m is None
                               else float(pitch_window_m))
        self._prof = {}
        for frame_id, sub in prof.groupby("frame_id"):
            i = pos.get(int(frame_id))
            if i is None:
                continue
            v = sub.sort_values("d_req_m")[["x", "y", z_col]].to_numpy() - cam[i]
            z, h = v @ fwd[i], v @ up[i]
            order = np.argsort(z)          # 投影後 z 未必單調（彎道/路拱）
            z, h = z[order], h[order]
            keep = np.concatenate(([True], np.diff(z) > 1e-9))
            self._prof[int(frame_id)] = (z[keep], h[keep])

    def describe(self):
        drop = f", {self.n_dropped} pts dropped" if self.n_dropped else ""
        return (f"measured road profile [{self.height_source}] "
                f"({len(self._prof)} frames, {self.dataset_dir.name}{drop}, "
                f"pitch window ±{self.pitch_window_m:g} m)")

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

    def pitch_at(self, frame_id, distances, window_m=None):
        """路面 pitch（度）：剖面在 ±``window_m`` 內的最小平方斜率。

        **視窗是固定的物理長度，不隨採樣密度變** —— 這是刻意的。原本這裡用
        逐點 ``np.gradient``，在 ``_PROFILE_STEP_M = 1.0`` 的舊資料集上沒問題，
        因為 1 m 間距的中央差分**本身就是** ±1 m 視窗。採集密度改成 0.125 m
        之後那個隱含的低通消失，射線高度的量化誤差（0.1 mm 級）被除以 0.125 m
        放大成斜率雜訊，mesh GT 的 MAE 從 0.2545 惡化到 0.2834，而「網格結構」
        與 legacy GT 的相關從 +0.946 崩到 +0.312。改成顯式視窗就復原了。

        analytic 來源對視窗不敏感（解析曲線本來就平滑，實測 0.2706~0.2719），
        受影響的只有 mesh。

        預設 1.0 m 的理由：正好重現舊資料集的行為（可比較），且在實測中最能
        保留訊號。改用估測器同款的 ±max(1, 0.15z) 差不多（mesh MAE 0.2539），
        但遠處過度平滑會沖淡結構（預測與網格的相關 0.653 -> 0.521），而且會讓
        GT 耦合到估測器的參數。

        ⚠ 這仍與 pipeline 的 pitch 不完全對等：後者是 Theil-Sen（穩健中位數
        斜率），這裡是最小平方。有曲率的路段兩者會有差異。
        """
        if not self.has(frame_id):
            return np.full(len(np.atleast_1d(distances)), np.nan)
        z, h = self._prof[int(frame_id)]
        d = np.atleast_1d(np.asarray(distances, float))
        half = self.pitch_window_m if window_m is None else float(window_m)

        lo = np.searchsorted(z, d - half)
        hi = np.searchsorted(z, d + half, side="right")
        slope = np.full(len(d), np.nan)
        for i, (a, b) in enumerate(zip(lo, hi)):
            if b - a >= 3:
                x, y = z[a:b], h[a:b]
                var = x.var()
                if var > 1e-12:
                    slope[i] = (np.mean(x * y) - x.mean() * y.mean()) / var
        # 視窗湊不到 3 點（剖面太稀、或查詢點落在剖面兩端）才退回逐點微分
        gap = ~np.isfinite(slope)
        if gap.any():
            slope[gap] = np.interp(d[gap], z, np.gradient(h, z),
                                   left=np.nan, right=np.nan)
        out_of_range = (d < z[0]) | (d > z[-1])
        slope[out_of_range] = np.nan
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
                    measurements=None, height_source=None):
    """依資料集內容挑 GT 來源：有 road_profile.csv 就用量測式，否則回推式。

    ``height_source`` 只對量測式有意義，見 :class:`RoadProfileGT`。
    """
    path = Path(measurements_csv)
    if (path.parent / "road_profile.csv").exists():
        return RoadProfileGT(path.parent, height_source=height_source)
    if measurements is None:
        measurements = pd.read_csv(path)
    return LegacyProfileGT(measurements, camera_offset_m, camera_height)
