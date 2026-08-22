"""
carla_module/pick_route.py
互動式路線選擇：在俯視路網圖上點選起點／終點，規劃並預覽路線，存成 JSON
給 get_carlaDataset.py --route-file 採集。

使用方式：
    # 1) 先開著 CARLA server，然後選路線（會跳出一個視窗）
    uv run python carla_module/pick_route.py

    # 2) 拿存好的 JSON 去採集
    uv run python carla_module/get_carlaDataset.py --route-file outputs/routes/xxx.json

操作：
    左鍵第一下   設起點（綠）
    左鍵第二下   設終點（紅），立刻用 GlobalRoutePlanner 規劃並畫出路線
    S            存成 JSON
    R            清除重選
    Q / 關視窗   離開

    只設起點就按 S，會存一條從起點沿車道直走 --straight-m 公尺的路線
    （每遇分岔選最接近直走的分支），適合採直路資料。

為什麼要這個工具：`--route-dest X,Y` 得先知道終點的世界座標，而那個座標
沒有好方法可以查 —— 只能在 CARLA 裡把 spectator 開過去再抄下來。俯視圖上
點兩下直覺得多，而且可以先看到路線長什麼樣、會經過哪些路口，再決定要不要採。

座標與投影：CARLA 是左手座標系（x 右、y 下），所以圖上把 y 軸反向，看起來
才跟遊戲裡的俯視地圖一致。點擊拿到的 event.xdata/ydata 仍是 CARLA 的 x/y，
反轉軸不影響資料座標。

高架路段的已知限制：點擊只有 (x, y)，沒有高度。路網點是**用 2D 距離**去
找最近的，所以立體交叉處（一條路壓在另一條上）可能挑到不是你要的那一層。
挑完之後圖上會標出該點的 z，對不上就往旁邊一點、避開重疊處再點。
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import argparse
import datetime
import json

# get_carlaDataset 在 import 時會做 setup_env() 與 CARLA_WHL_PATH 的 sys.path
# 設定（agents 套件要靠它才找得到），所以一定要在 import carla 之前
from carla_module.get_carlaDataset import (  # noqa: E402
    _ROUTE_STEP_M,
    ROUTE_FILE_VERSION,
    Route,
    build_route_straight,
    build_route_to,
)

import carla                                  # noqa: E402
import matplotlib.pyplot as plt               # noqa: E402
import numpy as np                            # noqa: E402


# matplotlib 預設字型沒有中文字符，圖上的標題／圖例會整排變成豆腐框。
# 依序試 Windows 內建的中文字型，都沒有才退回預設（英數仍可讀）。
# unicode_minus 要關掉：中文字型多半沒有 U+2212，負號會跟著變豆腐
plt.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei", "Microsoft YaHei", "PingFang TC",
    "Noto Sans CJK TC", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# 採集端的節奏，用來把「路線長度」換算成「大概幾幀」。與 get_carlaDataset 的
# 預設一致（18 km/h、40 fps）：每幀前進 18/3.6/40 = 0.125 m
_DEFAULT_SPEED_KMH = 18.0
_DEFAULT_FPS       = 40


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CARLA 互動式路線選擇")
    p.add_argument("--host",       default="127.0.0.1", help="CARLA 伺服器位址")
    p.add_argument("--port",       type=int,   default=2000, help="埠號")
    p.add_argument("--timeout",    type=float, default=20.0, help="連線逾時秒數")
    p.add_argument("--net-step-m", type=float, default=2.0,
                   help="底圖路網的取樣間距（公尺，預設 2.0）。"
                        "調小圖更細但點更多、拖動更慢")
    p.add_argument("--straight-m", type=float, default=250.0,
                   help="只設起點時，直走路線要規劃多長（公尺，預設 250）")
    p.add_argument("--out",        default=None,
                   help="輸出 JSON 路徑（預設 outputs/routes/route_{Map}_{時間}.json）")
    return p.parse_args()


def load_network(carla_map: carla.Map, step_m: float) -> np.ndarray:
    """所有可行駛車道的中心線取樣點，回傳 shape (N, 3) 的 x/y/z 陣列。

    `generate_waypoints` 一次給出全地圖每條車道每 step_m 一點，畫成散點就是
    一張路網俯視圖，也直接拿來做點擊吸附。
    """
    wps = carla_map.generate_waypoints(step_m)
    if not wps:
        raise RuntimeError("地圖沒有回傳任何 waypoint，確認連到的是有路網的地圖")
    return np.array([[w.transform.location.x,
                      w.transform.location.y,
                      w.transform.location.z] for w in wps])


def snap(net: np.ndarray, x: float, y: float) -> carla.Location:
    """把點擊位置吸附到最近的路網點（2D 距離，見模組 docstring 的高架限制）。"""
    i = int(np.argmin((net[:, 0] - x) ** 2 + (net[:, 1] - y) ** 2))
    return carla.Location(x=float(net[i, 0]), y=float(net[i, 1]), z=float(net[i, 2]))


def frames_estimate(length_m: float) -> int:
    """路線長度換算成採集幀數，讓人在按 S 之前就知道會產生多少張圖。"""
    per_frame_m = (_DEFAULT_SPEED_KMH / 3.6) / _DEFAULT_FPS
    return int(length_m / per_frame_m)


def save_route(route: Route, carla_map: carla.Map, mode: str,
               start: carla.Location, dest: carla.Location | None,
               out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version":   ROUTE_FILE_VERSION,
        "created":   datetime.datetime.now().isoformat(timespec="seconds"),
        "map":       carla_map.name,
        "mode":      mode,
        "start":     [round(start.x, 3), round(start.y, 3), round(start.z, 3)],
        "dest":      None if dest is None else
                     [round(dest.x, 3), round(dest.y, 3), round(dest.z, 3)],
        "planned_m": round(route.length_m, 2),
        "step_m":    _ROUTE_STEP_M,
        "points":    [[round(p.x, 3), round(p.y, 3), round(p.z, 3)]
                      for p in route.points],
    }
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n[存檔] {out_path}")
    print(f"[存檔] {route.length_m:.1f} m / {len(route)} 點，"
          f"預估 ~{frames_estimate(route.length_m)} 幀")
    print(f"\n接著跑：\n"
          f"  uv run python carla_module/get_carlaDataset.py "
          f"--route-file {out_path}\n")


class RoutePicker:
    """俯視圖上的點選狀態機：起點 → 終點 → 規劃 → 存檔。"""

    def __init__(self, carla_map: carla.Map, net: np.ndarray,
                 straight_m: float, out_path: pathlib.Path) -> None:
        self.carla_map  = carla_map
        self.net        = net
        self.straight_m = straight_m
        self.out_path   = out_path

        self.start: carla.Location | None = None
        self.dest:  carla.Location | None = None
        self.route: Route | None          = None
        self.mode:  str                   = "straight"

        # 深色底 + 亮色路網：路要夠顯眼才點得準，路線與起訖點也才跳得出來
        self.fig, self.ax = plt.subplots(figsize=(11, 10), facecolor="#1f2226")
        self.fig.canvas.manager.set_window_title(f"CARLA 路線選擇 — {carla_map.name}")
        self.ax.set_facecolor("#2a2e33")

        self.ax.scatter(net[:, 0], net[:, 1], s=1.5, c="#c7ccd1", linewidths=0)
        self.ax.set_aspect("equal")
        self.ax.invert_yaxis()          # CARLA 是左手系，反轉才跟遊戲俯視圖一致
        self.ax.set_xlabel("world x (m)", color="#c7ccd1")
        self.ax.set_ylabel("world y (m)", color="#c7ccd1")
        self.ax.tick_params(colors="#9aa0a6")
        for spine in self.ax.spines.values():
            spine.set_color("#4a4f55")
        self.ax.grid(alpha=0.12, color="#c7ccd1")

        # 這幾個 artist 之後就地更新，不重畫整張底圖
        self._start_art = self.ax.plot([], [], "o", ms=11, mfc="#22c55e",
                                       mec="black", mew=1.2, zorder=5,
                                       label="起點 start")[0]
        self._dest_art  = self.ax.plot([], [], "o", ms=11, mfc="#ef4444",
                                       mec="black", mew=1.2, zorder=5,
                                       label="終點 dest")[0]
        self._route_art = self.ax.plot([], [], "-", lw=2.8, c="#38bdf8",
                                       zorder=4, label="規劃路線 route")[0]
        leg = self.ax.legend(loc="upper right", framealpha=0.92,
                             facecolor="#1f2226", edgecolor="#4a4f55")
        for txt in leg.get_texts():
            txt.set_color("#e5e7eb")

        self._status = self.ax.set_title("", color="#e5e7eb")
        self._refresh_title("左鍵點第一下：設起點")

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        # 保險：CallbackRegistry 只用弱參考存 bound method，光靠它撐不住這個
        # 實例。呼叫端已經接住回傳值了，這裡再讓 figure 也拉一條硬參考，
        # 之後有人重構 main() 也不會又變成「點了沒反應」
        self.fig._route_picker = self

    # ── 繪圖 ──────────────────────────────────────────────────────────────
    def _refresh_title(self, hint: str) -> None:
        bits = []
        if self.start is not None:
            bits.append(f"起點 ({self.start.x:.1f}, {self.start.y:.1f}, z={self.start.z:.1f})")
        if self.dest is not None:
            bits.append(f"終點 ({self.dest.x:.1f}, {self.dest.y:.1f}, z={self.dest.z:.1f})")
        if self.route is not None:
            bits.append(f"路線 {self.route.length_m:.0f} m "
                        f"(~{frames_estimate(self.route.length_m)} 幀)")
        head = "  |  ".join(bits) if bits else "尚未選點"
        self._status.set_text(f"{head}\n{hint}   [S] 存檔  [R] 重選  [Q] 離開")
        self.fig.canvas.draw_idle()

    def _draw_route(self) -> None:
        if self.route is None or len(self.route) == 0:
            self._route_art.set_data([], [])
            return
        pts = self.route.points
        self._route_art.set_data([p.x for p in pts], [p.y for p in pts])

    # ── 規劃 ──────────────────────────────────────────────────────────────
    def _plan(self) -> None:
        """依目前已選的點規劃路線。失敗只提示、不中斷，讓人可以重選。"""
        try:
            if self.dest is not None:
                self.route = build_route_to(self.carla_map, self.start, self.dest)
                self.mode  = "dest"
            else:
                self.route = build_route_straight(
                    self.carla_map, self.start, self.straight_m)
                self.mode  = "straight"
        except Exception as exc:                                # noqa: BLE001
            self.route = None
            self._draw_route()
            self._refresh_title(f"規劃失敗：{exc}　按 R 重選")
            return

        if len(self.route) < 2:
            self.route = None
            self._draw_route()
            self._refresh_title("規劃不出路線（起點與終點之間可能不連通）　按 R 重選")
            return

        self._draw_route()
        self._refresh_title("路線 OK，按 S 存檔；不滿意按 R 重選")

    # ── 事件 ──────────────────────────────────────────────────────────────
    def on_click(self, event) -> None:
        if event.inaxes is not self.ax or event.button != 1:
            return

        # 縮放／平移開著時不設點，否則每拉一次框都會被當成選點。但要講出來 ——
        # 預設沉默的話，使用者只會看到「點了沒反應」，跟壞掉分不出來
        toolbar = self.fig.canvas.toolbar
        if toolbar is not None and toolbar.mode:
            self._refresh_title(
                f"工具列的「{toolbar.mode}」還開著，先在工具列把它關掉再點選")
            return

        loc = snap(self.net, event.xdata, event.ydata)
        if self.start is None:
            self.start = loc
            self._start_art.set_data([loc.x], [loc.y])
            self._refresh_title("左鍵點第二下：設終點（或直接按 S 存直走路線）")
            # 先畫出直走路線當預覽，不用等第二下
            self._plan()
        elif self.dest is None:
            self.dest = loc
            self._dest_art.set_data([loc.x], [loc.y])
            self._plan()
        else:
            self._refresh_title("起點與終點都已設定，按 R 重選")

    def on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key == "r":
            self.start = self.dest = self.route = None
            self._start_art.set_data([], [])
            self._dest_art.set_data([], [])
            self._draw_route()
            self._refresh_title("已清除。左鍵點第一下：設起點")
        elif key == "s":
            if self.route is None:
                self._refresh_title("還沒有可存的路線，先點起點")
                return
            save_route(self.route, self.carla_map, self.mode,
                       self.start, self.dest, self.out_path)
            self._refresh_title(f"已存到 {self.out_path.name}　可以關掉視窗了")
        elif key == "q":
            plt.close(self.fig)


def main() -> None:
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world     = client.get_world()
    carla_map = world.get_map()
    print(f"[初始化] 連線至地圖：{carla_map.name}")

    net = load_network(carla_map, args.net_step_m)
    print(f"[初始化] 路網取樣點：{len(net)}（間距 {args.net_step_m} m）")

    if args.out:
        out_path = pathlib.Path(args.out)
    else:
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = (pathlib.Path(__file__).parent.parent / "outputs" / "routes"
                    / f"route_{carla_map.name.split('/')[-1]}_{ts}.json")

    print("\n操作：左鍵設起點 → 左鍵設終點 → S 存檔（R 重選、Q 離開）")
    print("只設起點就按 S，會存一條直走 "
          f"{args.straight_m:.0f} m 的路線\n")

    # 這個 picker 一定要接住！matplotlib 的 CallbackRegistry 對 bound method
    # 是用**弱參考**存的，沒人持有實例的話它會被 GC 回收，回呼跟著失效 ——
    # 症狀是視窗跟圖都正常，但點擊完全沒反應
    picker = RoutePicker(carla_map, net, args.straight_m, out_path)
    plt.show()
    del picker      # 明確用掉，免得被當成沒用到的變數清掉


if __name__ == "__main__":
    main()
