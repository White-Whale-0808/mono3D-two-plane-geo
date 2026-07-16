"""
carla_module/get_carlaDataset.py
CARLA 資料集蒐集腳本：蒐集 RGB 影像與 GT pitch／速度／行駛距離

使用方式：
    uv run python carla_module/get_carlaDataset.py [--host HOST] [--port PORT]
        [--speed KMH] [--camera-fps N] [--z-offset N]
        [--align-frames N] [--warmup-frames N]
        [--kp F] [--ki F] [--kd F] [--ff-gain F]

狀態機流程：
    生成車輛
       ↓
    [ALIGNING]   TM autopilot 對齊車道（連續 --align-frames 幀 steer < 0.05）
       ↓
    [WARMUP]     關閉 TM，PID+FF 控制直行，等 --warmup-frames 幀讓物理穩定
       ↓
    [COLLECTING] steer=0 PID+FF 控制，存圖 + GT 量測，按 'q' 停止

儲存路徑：根目錄 carla_dataset_{Map}_{YYYYMMDD_HHMMSS}/
    ├── images/  000000.png, 000001.png, ...
    └── measurements.csv
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
import math
import queue
from typing import Optional

import carla
import cv2
import numpy as np


# ── 相機規格（比照 get_data.py）────────────────────────────────────────────────
IMG_WIDTH     = 1280
IMG_HEIGHT    = 720
CAMERA_HEIGHT = 1.08
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


def apply_pid_ff_control(
    vehicle:    carla.Vehicle,
    target_mps: float,
    pid:        PIDController,
    ff_gain:    float,
) -> tuple[float, float]:
    """
    執行 PID + Feed-forward 定速控制，steer 固定為 0。
    回傳 (當前速度 m/s, 坡度 °)。
    """
    vel       = vehicle.get_velocity()
    speed     = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
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

    vehicle.apply_control(carla.VehicleControl(
        throttle=throttle,
        steer=0.0,
        brake=brake,
        hand_brake=False,
        manual_gear_shift=False,
    ))
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
    _CSV_HEADER = ["frame_id", "gt_pitch_deg", "gt_speed_mps", "collect_dist_m"]

    def __init__(self, root: pathlib.Path, map_name: str) -> None:
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = root / "output" / f"carla_dataset_{map_name}_{ts_str}"
        self.img_dir  = self.save_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self.save_dir / "measurements.csv"
        self._csv_f  = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_f)
        self._writer.writerow(self._CSV_HEADER)
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
    ) -> None:
        img_path = str(self.img_dir / f"{self._count:06d}.png")
        image.save_to_disk(img_path)

        row = [
            self._count,
            f"{transform.rotation.pitch:.4f}",
            f"{speed_mps:.4f}",
            f"{collect_dist_m:.4f}",
        ]
        self._writer.writerow(row)
        self._csv_f.flush()
        self._count += 1

    def close(self) -> None:
        self._csv_f.close()
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
    cam_bp.set_attribute("fov", "90")
    cam_tf = carla.Transform(carla.Location(x=1.5, z=CAMERA_HEIGHT))
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
    writer   = DatasetWriter(root_dir, map_name)
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
            speed_mps    = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            cur_location = transform.location

            # ── 距離累加（每幀位移，同步模式下精度高）────────────────
            if prev_location is not None:
                dx = cur_location.x - prev_location.x
                dy = cur_location.y - prev_location.y
                dz = cur_location.z - prev_location.z
                frame_dist = math.sqrt(dx**2 + dy**2 + dz**2)
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
                speed_mps, slope_deg = apply_pid_ff_control(
                    vehicle, TARGET_SPEED_MPS, pid, ff_gain)
                warmup_counter += 1
                if warmup_counter >= args.warmup_frames:
                    state = State.COLLECTING
                    print(f"[{state}] 開始存檔！按 'q' 停止")

            elif state == State.COLLECTING:
                speed_mps, slope_deg = apply_pid_ff_control(
                    vehicle, TARGET_SPEED_MPS, pid, ff_gain)
                writer.save(image, transform, speed_mps, collect_distance_m)

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
