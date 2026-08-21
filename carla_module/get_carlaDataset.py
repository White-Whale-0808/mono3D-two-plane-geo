"""
carla_module/get_carlaDataset.py
CARLA 資料集蒐集腳本：蒐集 RGB 影像與 GT pitch／速度／行駛距離

使用方式：
    uv run python carla_module/get_carlaDataset.py [--host HOST] [--port PORT]
        [--speed KMH] [--camera-fps N] [--z-offset N]
        [--align-frames N] [--warmup-frames N] [--lookahead-m N]
        [--kp F] [--ki F] [--kd F] [--ff-gain F]

狀態機流程：
    生成車輛
       ↓
    [ALIGNING]   TM autopilot 對齊車道（連續 --align-frames 幀 steer < 0.05）
       ↓
    [WARMUP]     關閉 TM，PID+FF 控制 + pure-pursuit 橫向修正，等 --warmup-frames 幀讓物理穩定
       ↓
    [COLLECTING] PID+FF 控制 + pure-pursuit 橫向修正，存圖 + GT 量測，按 'q' 停止

    橫向控制說明：ALIGNING 只保證 TM 轉向輸出連續幾幀 < 0.05，不保證航向零
    誤差；早期版本切到手動後直接 steer=0 寫死，假設接下來的路完全筆直，路
    稍有彎或殘留航向誤差就會在整段收集距離上不受控漂移，可能跨到隔壁車道
    （carla_module/verify_carla_geometry.py 的驗證報告第⑥項可以量出這個問題）。
    現在改用 `_pure_pursuit_steer`（沿車道中心線前視 `--lookahead-m`，預設
    6m）持續修正，縱向（throttle/brake）仍是同一套 PID+FF 不變。

儲存路徑：outputs/carla_dataset_{Map}_{YYYYMMDD_HHMMSS}/
    ├── images/         000000.png, 000001.png, ...
    ├── measurements.csv  逐幀：車身姿態 / 路面姿態 / 速度 / 距離 / 車輛與相機世界座標
    ├── road_profile.csv  逐幀 × 逐採樣點：相機前方路面的世界座標（採集當下直接問地圖）
    └── metadata.json     相機掛載參數、fov、解析度、地圖名、採集設定

GT 的設計原則：**存最生的量，之後再推導**。前方剖面存 waypoint 的世界座標而
不是預先算好的 pitch，相機存完整 transform 而不是只有位置 —— 這樣「沿光軸的
深度」「高度剖面」「相對哪個座標系的 pitch」都能離線重算，不必為了換一種
推導方式而重採。詳見 DatasetWriter 與 sample_road_profile 的 docstring，
以及 to-do.md 第 2/3 組。
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils.env_setup import setup_env
setup_env()

_whl = os.environ.get("CARLA_WHL_PATH", "")
if _whl:
    _carla_api_dir = str(pathlib.Path(_whl).parent.parent)
    if _carla_api_dir not in sys.path:
        sys.path.insert(0, _carla_api_dir)

import argparse
import csv
import datetime
import json
import math
import queue
from typing import Optional

import carla
import cv2
import numpy as np


# ── 相機規格（比照 get_data.py）────────────────────────────────────────────────
IMG_WIDTH     = 1280
IMG_HEIGHT    = 720
CAMERA_FOV    = 90.0
CAMERA_HEIGHT = 1.08   # 相對「車輛原點」的掛載高度，車輛原點不一定在地面
CAMERA_FWD_X  = 1.5    # 相機前移量：GT 距離記在車輛原點，推論端需用此值對齊
PHYSICS_WARMUP_TICKS = 30   # 生成後讓物理系統落穩的預熱 tick 數


# ── 狀態機 ────────────────────────────────────────────────────────────────────

class State:
    ALIGNING   = "ALIGNING"    # TM autopilot 對齊車道中
    WARMUP     = "WARMUP"      # 關閉 TM 後等物理穩定
    COLLECTING = "COLLECTING"  # PID+FF 控制，開始存資料


# ── PID + Feed-forward 控制器 ─────────────────────────────────────────────────

class PIDController:
    """
    針對坡道定速設計的 PID 控制器。
    - P 項：即時反應速差
    - I 項：消除上坡重力造成的穩態誤差（取代舊版固定偏置 0.3）
    - D 項：抑制超調與震盪
    搭配 Feed-forward 坡度補償，進坡瞬間不掉速。
    """

    def __init__(self, kp: float = 1.0, ki: float = 0.25, kd: float = 0.15,
                 dt: float = 0.025, integral_limit: float = 3.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral_limit = integral_limit  # anti-windup

        self._integral  = 0.0
        self._prev_error: Optional[float] = None

    def reset(self) -> None:
        self._integral   = 0.0
        self._prev_error = None

    def step(self, error: float) -> float:
        """輸入速差 (m/s)，輸出 [-1, 1] 控制量（正=油門，負=煞車）"""
        self._integral += error * self.dt
        self._integral  = max(-self.integral_limit,
                              min(self.integral_limit, self._integral))

        # 首步（或 reset 後）沒有前值可差分，D 項歸零，
        # 避免 (error-0)/dt 的 derivative kick 造成油門/煞車瞬間打滿
        if self._prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self._prev_error) / self.dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(max(-1.0, min(1.0, output)))


def _get_slope_deg(vehicle: carla.Vehicle) -> float:
    """回傳車輛當前坡度（度），上坡為正、下坡為負。"""
    # CARLA/UE 慣例：上坡（機頭朝上）pitch 為正值，直接沿用
    return vehicle.get_transform().rotation.pitch


def _slope_feedforward(slope_deg: float, ff_gain: float) -> float:
    """
    坡度前饋補償量（油門側）。
    物理原理：上坡阻力 ∝ sin(θ)，直接預補油門，
    不需等積分累積，進坡瞬間不掉速。
    下坡回傳 0（不主動提前煞車）。
    """
    return float(max(0.0, math.sin(math.radians(slope_deg)) * ff_gain * 100))


def _compute_pid_ff(
    vehicle:    carla.Vehicle,
    target_mps: float,
    pid:        PIDController,
    ff_gain:    float,
) -> tuple[float, float, float, float]:
    """算 PID + Feed-forward 縱向控制量，不呼叫 apply_control。回傳 (throttle, brake, 當前速度 m/s, 坡度 °)。"""
    vel       = vehicle.get_velocity()
    speed     = vel.length()
    error     = target_mps - speed
    slope_deg = _get_slope_deg(vehicle)

    pid_out     = pid.step(error)
    feedforward = _slope_feedforward(slope_deg, ff_gain)
    raw_output  = pid_out + feedforward

    if raw_output >= 0.0:
        throttle = min(1.0, raw_output)
        brake    = 0.0
    else:
        throttle = 0.0
        brake    = min(1.0, abs(raw_output) * 0.6)

    return throttle, brake, speed, slope_deg


def _wrap_deg(angle: float) -> float:
    """正規化角度差到 (-180, 180]，避免 0°/360° 之類的等價角度被算成 -360°。"""
    return (angle + 180.0) % 360.0 - 180.0


def _pure_pursuit_steer(vehicle: carla.Vehicle, carla_map: carla.Map,
                        lookahead_m: float) -> float:
    """
    取代早期版本寫死的 steer=0.0。ALIGNING 只保證 TM 轉向輸出
    連續幾幀 < 0.05，不保證航向零誤差；WARMUP/COLLECTING 加起來往往是幾十到
    上百公尺，路稍有彎或殘留一點航向誤差，steer=0 會讓車子不受控地橫向漂移，
    可能跨到隔壁車道，把整批資料的 GT 對應關係搞錯。

    做法：沿目前車道中心線往前 lookahead_m 找一個目標點，算目標點相對車頭
    朝向的夾角，轉成 [-1, 1] 的 steer 修正量（簡化版 pure-pursuit）。
    """
    tf = vehicle.get_transform()
    wp = carla_map.get_waypoint(tf.location, project_to_road=True)
    if wp is None:
        return 0.0
    nxt = wp.next(lookahead_m)
    if not nxt:
        return 0.0
    target = nxt[0].transform.location

    fwd = tf.get_forward_vector()
    heading      = math.degrees(math.atan2(fwd.y, fwd.x))
    to_target_x  = target.x - tf.location.x
    to_target_y  = target.y - tf.location.y
    target_angle = math.degrees(math.atan2(to_target_y, to_target_x))

    angle_err_deg = _wrap_deg(target_angle - heading)
    # 經驗值：滿舵對應 ~30° 航向誤差，避免小誤差被過度放大造成震盪
    return max(-1.0, min(1.0, angle_err_deg / 30.0))


# ── 前方路面剖面（採集當下直接問地圖）──────────────────────────────────────────

_PROFILE_MAX_D_M = 50.0   # 採樣到相機前方幾公尺（略大於 pitch_estimation 的 z_cap 45）
_PROFILE_STEP_M  = 1.0    # 採樣間距

# 向下射線可接受的地面語意標籤（label 名稱字串比對，跨 CARLA 版本較穩）。
# verify_carla_geometry.py 也用這份，改這裡兩邊一起動。
GROUND_LABELS = {"Roads", "RoadLines", "Ground", "Terrain", "Sidewalks"}


def ray_ground_z(world: carla.World, loc: carla.Location,
                 depth: float = 6.0) -> tuple[Optional[float], Optional[str]]:
    """
    從 loc 垂直向下射線，回傳第一個「地面語意」命中點的 z 與其標籤。

    **必須**用 label 過濾：從相機處起算的射線會先穿過引擎蓋/底盤，剖面上的
    射線則可能打到路上的車輛或雜物。找不到地面標籤就回 None —— 不做退而求
    其次的猜測，寧缺勿錯（否則會把車身高度當成路面高度）。

    回傳的第二項在失敗時是 `"no-ground:<看到的標籤>"`，方便離線分辨「射線沒
    命中」與「命中了但都不是地面」。
    """
    try:
        end    = carla.Location(x=loc.x, y=loc.y, z=loc.z - depth)
        points = world.cast_ray(loc, end)
    except Exception:                                          # noqa: BLE001
        return None, None
    if not points:
        return None, None

    labels_seen: list[str] = []
    for pt in points:
        label = str(getattr(pt, "label", "")).split(".")[-1]
        z     = float(pt.location.z)
        if z > loc.z + 1e-6:          # 射線可能回傳起點上方的命中，忽略
            continue
        labels_seen.append(label)
        if label in GROUND_LABELS:
            return z, label
    return None, ("no-ground:" + ",".join(labels_seen) if labels_seen else None)


def sample_road_profile(
    carla_map:  carla.Map,
    cam_loc:    carla.Location,
    max_d_m:    float = _PROFILE_MAX_D_M,
    step_m:     float = _PROFILE_STEP_M,
) -> list[tuple[float, carla.Waypoint]]:
    """相機前方路面剖面：沿車道往前走，回傳 (要求距離, waypoint) 序列。

    取代「用行駛歷史回推前方 GT」的做法（`pitch_visualization.gt_pitch_profile`
    是拿車子後來開到那裡時的 pitch 當前方 GT）。歷史回推有三個問題：車身/懸吊
    的遲滯（實測 body = 1.0213 x road - 0.48 度，且逐坡度帶差值不是常數）、
    前移量的對齊、以及開到終點的幀沒有前方 GT。直接問地圖三者一次消失。

    起點是**相機所在處**而非車輛所在處：相機前移 CAMERA_FWD_X = 1.5 m，
    在 9.5 度坡上那裡的路面高 0.25 m，用車輛 waypoint 會系統性量錯
    （見 carla_module/verify_carla_geometry.py 的第 1 組驗證）。

    逐步走而不是直接 `wp.next(d)`：後者在路口會分岔，且每次都從頭走一遍。
    分岔時選航向最接近前一段的分支，避免剖面拐進岔路。
    """
    wp = carla_map.get_waypoint(cam_loc, project_to_road=True)
    if wp is None:
        return []

    out = [(0.0, wp)]
    prev_fwd = wp.transform.get_forward_vector()
    d = 0.0
    while d < max_d_m - 1e-6:
        nxts = wp.next(step_m)
        if not nxts:
            break   # 路盡頭：剖面就到此為止，不猜測
        wp = max(nxts, key=lambda w: w.transform.get_forward_vector().dot(prev_fwd))
        prev_fwd = wp.transform.get_forward_vector()
        d += step_m
        out.append((d, wp))
    return out


def apply_pid_ff_control_with_steering(
    vehicle:     carla.Vehicle,
    carla_map:   carla.Map,
    target_mps:  float,
    pid:         PIDController,
    ff_gain:     float,
    lookahead_m: float,
) -> tuple[float, float]:
    """
    PID+FF 縱向 + pure-pursuit 橫向，同一輪算好後一次 apply_control 送出。

    早期版本是「先送出縱向控制（steer 寫死 0）→ 用 vehicle.get_control()
    讀回來 → 疊加 steer → 再 apply_control 一次」，看似安全（CARLA 只認每個
    tick 最後一次 apply_control），實際上有 race condition：同步模式下
    get_control() 要等下一次 world.tick() 才會反映剛剛 apply_control() 送出的
    值，讀回來的其實是上一輪的舊值。實測這會讓油門被舊值鎖住、車速衝過頭
    （曾經衝到 37km/h，目標只有 18），才被大幅延遲的煞車拉回來變成 bang-bang
    震盪。修法是完全不讀 get_control()，縱向/橫向控制量在這裡當場算好、
    一次送出。
    """
    throttle, brake, speed, slope_deg = _compute_pid_ff(vehicle, target_mps, pid, ff_gain)
    steer = _pure_pursuit_steer(vehicle, carla_map, lookahead_m)
    vehicle.apply_control(carla.VehicleControl(
        throttle=throttle, steer=steer, brake=brake,
        hand_brake=False, manual_gear_shift=False))
    return speed, slope_deg


# ── 引數 ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CARLA 資料集蒐集")
    p.add_argument("--host",          default="127.0.0.1", help="CARLA 伺服器位址")
    p.add_argument("--port",          type=int,   default=2000,  help="埠號")
    p.add_argument("--timeout",       type=float, default=20.0,  help="連線逾時秒數")
    p.add_argument("--speed",         type=float, default=18.0,  help="目標車速 km/h")
    p.add_argument("--camera-fps",    type=int,   default=40,    help="同步模式 FPS（預設：40）")
    p.add_argument("--z-offset",      type=float, default=0.0,
                   help="spectator z 偏移量（預設 0，依場景高度調整）")
    p.add_argument("--align-frames",  type=int,   default=20,
                   help="連續幾幀 steer<0.05 才視為對齊完成（預設：20）")
    p.add_argument("--warmup-frames", type=int,   default=10,
                   help="切換 TM→手動後再等幾幀才開始存檔（預設：10）")
    p.add_argument("--lookahead-m",   type=float, default=6.0,
                   help="pure-pursuit 橫向修正的前視距離（預設 6 m），"
                        "取代寫死 steer=0 的假設")
    # ── PID + FF 參數（可在命令列覆寫）──────────────────────────────────────
    p.add_argument("--kp",      type=float, default=1.0,
                   help="PID 比例增益（預設 1.0，調大反應更快但易震盪）")
    p.add_argument("--ki",      type=float, default=0.25,
                   help="PID 積分增益（預設 0.25）")
    p.add_argument("--kd",      type=float, default=0.15,
                   help="PID 微分增益（預設 0.15）")
    p.add_argument("--ff-gain", type=float, default=0.015,
                   help="坡度前饋增益（預設 0.015）")
    return p.parse_args()


# ── 資料集寫入器 ───────────────────────────────────────────────────────────────

class DatasetWriter:
    """資料集寫入器：影像 + 逐幀量測 + 前方路面剖面 + 掛載中繼資料。

    設計原則是**存最生的量，之後再推導**。前方剖面存 waypoint 的世界座標
    (x, y, z) 而不是預先算好的 pitch —— 存 pitch 會把「相對哪個座標系」這個
    選擇燒死在資料裡，選錯就得重採；存世界座標的話，深度／高度剖面／pitch
    都能離線重算，連「弧長 vs 沿光軸深度」這種問題都變成算術而非假說。
    同理相機存的是完整 transform（位置 + 旋轉）而非只有位置：pipeline 的 z
    是沿光軸的距離，光軸方向由相機旋轉定義。
    """

    _CSV_HEADER = [
        "frame_id",
        # 車身姿態（含懸吊）—— 原有欄位，語意與名稱刻意不變，舊工具仍可讀
        "gt_pitch_deg",
        "gt_speed_mps",
        "collect_dist_m",
        # 路面姿態（相機所在處）—— 與車身分開記，才能量化懸吊造成的誤差。
        # 實測 body = 1.0213 x road - 0.48 度，車身振幅超出 2.13%
        "road_pitch_deg",
        "body_roll_deg",
        # 世界座標：有了它才能直接重建高度剖面，取代「積分 tan(pitch)」的近似
        "veh_x", "veh_y", "veh_z",
        "cam_x", "cam_y", "cam_z",
        "cam_pitch_deg", "cam_yaw_deg", "cam_roll_deg",
        "lane_id", "road_id",
    ]

    # 每幀 × 每個採樣點一列
    _PROFILE_HEADER = [
        "frame_id", "d_req_m", "x", "y", "z", "wp_pitch_deg", "lane_id", "road_id",
    ]

    def __init__(self, root: pathlib.Path, map_name: str, metadata: dict) -> None:
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = root / "outputs" / f"carla_dataset_{map_name}_{ts_str}"
        self.img_dir  = self.save_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self.save_dir / "measurements.csv"
        self._csv_f  = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_f)
        self._writer.writerow(self._CSV_HEADER)

        prof_path = self.save_dir / "road_profile.csv"
        self._prof_f = open(prof_path, "w", newline="", encoding="utf-8")
        self._prof_writer = csv.writer(self._prof_f)
        self._prof_writer.writerow(self._PROFILE_HEADER)

        # 掛載參數寫進資料集本身：現在推論 config 的 camera_forward_offset /
        # camera_height 是手抄過去的，兩邊沒有任何連結，改一邊另一邊不會知道
        with open(self.save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self._count = 0
        print(f"[資料集] 儲存至：{self.save_dir}")

    @property
    def count(self) -> int:
        return self._count

    def save(
        self,
        image:          carla.Image,
        transform:      carla.Transform,
        speed_mps:      float,
        collect_dist_m: float,
        cam_tf:         carla.Transform,
        profile:        list,
    ) -> None:
        """存一幀。`profile` 來自 `sample_road_profile`，其第 0 點就是相機
        所在處的 waypoint，所以不必再查一次地圖。"""
        img_path = str(self.img_dir / f"{self._count:06d}.png")
        image.save_to_disk(img_path)

        wp_cam = profile[0][1] if profile else None
        vl, cl, cr = transform.location, cam_tf.location, cam_tf.rotation

        # pitch 一律 wrap 到 (-180, 180]：CARLA 的平路 waypoint pitch 實測會
        # 回 360.0，不 wrap 的話下游會讀到「路面 pitch +360 度」
        row = [
            self._count,
            f"{_wrap_deg(transform.rotation.pitch):.4f}",
            f"{speed_mps:.4f}",
            f"{collect_dist_m:.4f}",
            "" if wp_cam is None else f"{_wrap_deg(wp_cam.transform.rotation.pitch):.4f}",
            f"{_wrap_deg(transform.rotation.roll):.4f}",
            f"{vl.x:.4f}", f"{vl.y:.4f}", f"{vl.z:.4f}",
            f"{cl.x:.4f}", f"{cl.y:.4f}", f"{cl.z:.4f}",
            f"{_wrap_deg(cr.pitch):.4f}", f"{_wrap_deg(cr.yaw):.4f}", f"{_wrap_deg(cr.roll):.4f}",
            "" if wp_cam is None else int(wp_cam.lane_id),
            "" if wp_cam is None else int(wp_cam.road_id),
        ]
        self._writer.writerow(row)
        self._csv_f.flush()

        for d_req, wp in profile:
            loc = wp.transform.location
            self._prof_writer.writerow([
                self._count, f"{d_req:.2f}",
                f"{loc.x:.4f}", f"{loc.y:.4f}", f"{loc.z:.4f}",
                f"{_wrap_deg(wp.transform.rotation.pitch):.4f}",
                int(wp.lane_id), int(wp.road_id),
            ])
        self._prof_f.flush()

        self._count += 1

    def close(self) -> None:
        self._csv_f.close()
        self._prof_f.close()
        print(f"[資料集] 共儲存 {self._count} 幀。路徑：{self.save_dir}")


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main() -> None:
    args     = parse_args()
    root_dir = pathlib.Path(__file__).parent.parent

    # ── 連線 ──────────────────────────────────────────────────────────────────
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world     = client.get_world()
    carla_map = world.get_map()
    print(f"[初始化] 連線至地圖：{carla_map.name}")
    bp_lib = world.get_blueprint_library()

    # ── 生成車輛 ──────────────────────────────────────────────────────────────
    vehicle_bp      = bp_lib.find("vehicle.tesla.model3")
    spectator       = world.get_spectator()
    spawn_transform = spectator.get_transform()
    spawn_transform.location.z    += args.z_offset
    spawn_transform.rotation.pitch = 0.0
    spawn_transform.rotation.roll  = 0.0

    vehicle: carla.Vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
    if vehicle is None:
        raise RuntimeError(
            "無法在 spectator 位置生成車輛，請移動鏡頭到可通行路面，"
            "或以 --z-offset 調整偏移量（預設 0）"
        )
    vehicle.set_autopilot(False)
    print(f"[初始化] 車輛生成於：{spawn_transform.location}")

    TARGET_SPEED_MPS = args.speed / 3.6

    # ── 同步模式 ──────────────────────────────────────────────────────────────
    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 1.0 / args.camera_fps
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.CloudySunset)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    # ── PID + FF 控制器（dt 對齊同步模式 FPS）────────────────────────────────
    pid = PIDController(
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        dt=1.0 / args.camera_fps,   # 與 tick 週期一致，微分項才準確
        integral_limit=3.0,
    )
    ff_gain = args.ff_gain
    print(f"[PID+FF] kp={args.kp}  ki={args.ki}  kd={args.kd}  ff_gain={ff_gain}")

    # ── 感測器 ────────────────────────────────────────────────────────────────
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMG_WIDTH))
    cam_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
    cam_bp.set_attribute("fov", str(CAMERA_FOV))
    cam_tf = carla.Transform(carla.Location(x=CAMERA_FWD_X, z=CAMERA_HEIGHT))
    camera: carla.Sensor = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

    image_queue: queue.Queue = queue.Queue()
    camera.listen(image_queue.put)

    # ── 物理預熱 ──────────────────────────────────────────────────────────────
    print(f"[初始化] 物理預熱 {PHYSICS_WARMUP_TICKS} ticks...")
    for _ in range(PHYSICS_WARMUP_TICKS):
        world.tick()

    # ── 啟動 TM 對齊 ──────────────────────────────────────────────────────────
    vehicle.set_autopilot(True, tm.get_port())
    tm.ignore_lights_percentage(vehicle, 100.0)
    tm.ignore_signs_percentage(vehicle,  100.0)
    tm.set_desired_speed(vehicle, args.speed)

    state          = State.ALIGNING
    align_counter  = 0
    warmup_counter = 0

    # ── 距離追蹤 ──────────────────────────────────────────────────────────────
    total_distance_m   = 0.0   # 累計總行駛距離（含 ALIGNING/WARMUP）
    collect_distance_m = 0.0   # 僅 COLLECTING 階段的距離
    prev_location: Optional[carla.Location] = None

    map_name = carla_map.name.split("/")[-1]
    metadata = {
        "created":    datetime.datetime.now().isoformat(timespec="seconds"),
        "map":        carla_map.name,
        "vehicle_bp": vehicle_bp.id,
        # 推論端的 camera_forward_offset / camera_height / f_x / f_y 都從這裡推
        "camera": {
            "x_forward_m": CAMERA_FWD_X,
            "z_height_m":  CAMERA_HEIGHT,
            "fov_deg":     CAMERA_FOV,
            "img_width":   IMG_WIDTH,
            "img_height":  IMG_HEIGHT,
        },
        "road_profile": {
            "max_d_m":     _PROFILE_MAX_D_M,
            "step_m":      _PROFILE_STEP_M,
            "origin":      "camera",   # waypoint 起點是相機所在處，不是車輛原點
            "stored":      "world xyz of each waypoint (raw); pitch is a convenience column",
        },
        "collection": {
            "target_speed_kmh": args.speed,
            "camera_fps":       args.camera_fps,
            "lookahead_m":      args.lookahead_m,
            "pid":              {"kp": args.kp, "ki": args.ki, "kd": args.kd},
            "ff_gain":          args.ff_gain,
        },
    }
    writer   = DatasetWriter(root_dir, map_name, metadata)
    win_name = "CARLA Dataset Collection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    print(f"[{state}] TM 對齊車道中（需連續 {args.align_frames} 幀 steer < 0.05）")

    try:
        while True:
            target_frame = world.tick()

            # ── 取出與本 tick 對齊的相機幀 ────────────────────────────────────
            image: Optional[carla.Image] = None
            try:
                while True:
                    temp = image_queue.get(timeout=2.0)
                    if temp.frame == target_frame:
                        image = temp
                        break
                    if temp.frame > target_frame:
                        break
            except queue.Empty:
                print(f"[警告] 幀 {target_frame} 影像遺失，跳過")
                continue

            if image is None:
                continue

            transform    = vehicle.get_transform()
            vel          = vehicle.get_velocity()
            speed_mps    = vel.length()
            cur_location = transform.location

            # ── 距離累加（每幀位移，同步模式下精度高）────────────────
            if prev_location is not None:
                frame_dist = cur_location.distance(prev_location)
                total_distance_m += frame_dist
                if state == State.COLLECTING:
                    collect_distance_m += frame_dist
            prev_location = cur_location

            # ── 狀態機 ────────────────────────────────────────────────────────
            slope_deg = 0.0

            if state == State.ALIGNING:
                steer = vehicle.get_control().steer
                if abs(steer) < 0.05:
                    align_counter += 1
                else:
                    align_counter = 0

                if align_counter >= args.align_frames:
                    vehicle.set_autopilot(False)
                    pid.reset()   # 切換後清除積分，避免舊值干擾
                    state          = State.WARMUP
                    warmup_counter = 0
                    print(f"[{state}] TM 已關閉，等待 {args.warmup_frames} 幀物理穩定...")

            elif state == State.WARMUP:
                speed_mps, slope_deg = apply_pid_ff_control_with_steering(
                    vehicle, carla_map, TARGET_SPEED_MPS, pid, ff_gain, args.lookahead_m)
                warmup_counter += 1
                if warmup_counter >= args.warmup_frames:
                    state = State.COLLECTING
                    print(f"[{state}] 開始存檔！按 'q' 停止")

            elif state == State.COLLECTING:
                speed_mps, slope_deg = apply_pid_ff_control_with_steering(
                    vehicle, carla_map, TARGET_SPEED_MPS, pid, ff_gain, args.lookahead_m)
                cam_tf  = camera.get_transform()
                profile = sample_road_profile(carla_map, cam_tf.location)
                writer.save(image, transform, speed_mps, collect_distance_m,
                            cam_tf, profile)

            # ── 顯示視窗 ──────────────────────────────────────────────────────
            arr     = np.frombuffer(image.raw_data, dtype=np.uint8)
            display = arr.reshape((image.height, image.width, 4))[:, :, :3].copy()
            h, font = display.shape[0], cv2.FONT_HERSHEY_SIMPLEX

            state_color = {
                State.ALIGNING:   (0, 165, 255),
                State.WARMUP:     (0, 255, 255),
                State.COLLECTING: (0, 255, 0),
            }[state]

            cv2.putText(display,
                        f"[{state}]  Saved: {writer.count}",
                        (10, 30),     font, 0.8, state_color,   2, cv2.LINE_AA)
            cv2.putText(display,
                        f"GT Pitch: {transform.rotation.pitch:+.2f} deg  "
                        f"Slope: {slope_deg:+.1f} deg",
                        (10, 62),     font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display,
                        f"Speed: {speed_mps * 3.6:.1f} km/h  "
                        f"Steer: {vehicle.get_control().steer:+.3f}",
                        (10, 94),     font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display,
                        f"Dist(total): {total_distance_m:.1f} m  "
                        f"Dist(collect): {collect_distance_m:.1f} m",
                        (10, 126),    font, 0.8, (180, 255, 180), 2, cv2.LINE_AA)
            cv2.putText(display,
                        "Press Q to stop",
                        (10, h - 12), font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.imshow(win_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        print("[清理] 恢復非同步模式...")
        s = world.get_settings()
        s.synchronous_mode    = False
        s.fixed_delta_seconds = None
        world.apply_settings(s)
        tm.set_synchronous_mode(False)

        camera.stop()
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        writer.close()
        print(f"[距離統計] 總行駛：{total_distance_m:.1f} m  "
              f"採集階段：{collect_distance_m:.1f} m")


if __name__ == "__main__":
    main()
