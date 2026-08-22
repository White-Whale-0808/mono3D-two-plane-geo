"""整條路線的地形剖面圖：把每幀的前方剖面沿世界座標拼回一條路。

`road_profile.csv` 是「逐幀 × 逐採樣點」的前方剖面（每幀 0~50 m）。同一個
地點會被很多幀從不同深度看到，所以把所有點按**世界位置**分箱平均，就得到
整條路線的地形剖面 —— 這正是 WWH-14 的空間錨定量測（格內跨幀散布 0.27 mm）
所依據的性質，圖上會把實際的格內散布印出來當健康檢查。

畫三格，共用 x 軸（沿路里程 s）：

(a) 地形高度：`z`（OpenDRIVE 解析中心線）vs `z_mesh`（射線打到的路面網格），
    右軸疊上路面坡度（±1 m 最小平方斜率，與 ``RoadProfileGT.pitch_at`` 同定義）
(b) `z_mesh − z`（mm）—— 網格相對解析曲線的偏差，看它落在路線的哪幾段
(c) 每幀的 profile MAE，畫在**該幀相機所在的 s** 上，於是誤差熱點與地形對齊
    （只有拿到 batch 的 df 時才畫）

用法（batch runner）::

    from libs.visualization.route_profile_visualization import plot_route_profile
    plot_route_profile(dataset_dir, df, save_path=None)   # 檔名自動帶資料集名
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

C_ANALYTIC, C_MESH, C_MAE = "#2e8b57", "#c1440e", "#1f4e79"


def route_stations(points_xyz, track_xyz, s_track):
    """把世界座標點投影到相機軌跡折線上，回傳沿路里程 s（m）。

    **不用** ``collect_dist_m + d_req_m`` 相加。那是里程對齊，會把每幀的里程
    誤差帶進來，同一個地點在不同幀落到不同的 s（實測 7.7 mm 假散布，改用
    世界座標只剩 0.27 mm）。這裡取最近的軌跡頂點，再沿當地軌跡方向做次頂點
    投影。

    **最近鄰必須算 3D**：路線會在立體交叉處自己壓過自己（`full_road` 就是
    先走天橋、後來從橋下穿過）。只比 (x, y) 的話，橋下那段的剖面點會被吸到
    橋上的站點，圖上會冒出一個 4 m 深的假深谷。高度差一拉開就分得乾淨。
    """
    dist, j = cKDTree(track_xyz).query(points_xyz)
    j0 = np.clip(j - 1, 0, len(track_xyz) - 1)
    j1 = np.clip(j + 1, 0, len(track_xyz) - 1)
    u = track_xyz[j1] - track_xyz[j0]
    length = np.linalg.norm(u, axis=1)
    t = np.zeros(len(points_xyz))
    ok = length > 1e-9
    t[ok] = ((points_xyz[ok] - track_xyz[j[ok]]) * u[ok]).sum(1) / length[ok]
    return s_track[j] + t, dist


def _slope_deg(s, h, half_m=1.0):
    """±``half_m`` 內的最小平方斜率（度），與 ``RoadProfileGT.pitch_at`` 同定義。"""
    lo = np.searchsorted(s, s - half_m)
    hi = np.searchsorted(s, s + half_m, side="right")
    out = np.full(len(s), np.nan)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b - a >= 3:
            x, y = s[a:b], h[a:b]
            var = x.var()
            if var > 1e-12:
                out[i] = (np.mean(x * y) - x.mean() * y.mean()) / var
    return np.degrees(np.arctan(out))


def plot_route_profile(dataset_dir, df=None, save_path=None, bin_m=0.25,
                       mae_col="profile_mae", max_offset_m=2.0):
    """畫整條路線的地形剖面 + 網格偏差（+ 每幀 MAE），存成 PNG 回傳路徑。

    Parameters
    ----------
    dataset_dir : str | Path
        含 ``measurements.csv`` 與 ``road_profile.csv`` 的資料集目錄。
    df : pd.DataFrame | None
        batch runner 的輸出（要有 ``frame_id`` 與 ``profile_mae``）。給了才畫
        第三格。
    save_path : str | Path | None
        ``None`` = ``outputs/route_profile_<資料集名>.png``（帶資料集名，跑多份
        資料集才不會互相覆蓋）。
    bin_m : float
        沿路分箱寬度（m）。0.25 m 遠小於網格偏差的空間相關長度（3~6 m）。
    max_offset_m : float
        剖面點離駕駛軌跡超過這個 3D 距離就丟掉 —— 路口處 ``waypoint.next()``
        會岔到別條路上，那些點不屬於這條路線。
    """
    dataset_dir = Path(dataset_dir)
    m = pd.read_csv(dataset_dir / "measurements.csv").sort_values("frame_id")
    prof = pd.read_csv(dataset_dir / "road_profile.csv")

    # 軌跡的高度用「該幀相機正下方的路面」（剖面的 d_req_m = 0 那點），
    # 這樣軌跡與剖面點在同一個面上，3D 最近鄰才不會被 1.08 m 的相機高度歪掉。
    prof = prof[prof["z"].notna()].copy()
    under_cam = (prof.sort_values("d_req_m").groupby("frame_id")["z"].first())
    z_track = m["frame_id"].astype(int).map(under_cam)
    z_track = z_track.fillna(m["cam_z"] - (m["cam_z"] - m["veh_z"]).median())
    track = np.column_stack([m["cam_x"], m["cam_y"], z_track])
    s_track = np.concatenate(([0.0], np.cumsum(          # 沿路水平距離
        np.linalg.norm(np.diff(track[:, :2], axis=0), axis=1))))
    prof["s"], off = route_stations(prof[["x", "y", "z"]].to_numpy(),
                                    track, s_track)
    # 剖面是靠 waypoint.next() 往前長的，遇到路口會岔到別條路（或匝道）上去。
    # 那些點的世界座標離駕駛軌跡很遠，硬投影回里程只會在圖上長出假的尖峰。
    keep = off <= max_offset_m
    n_pts, n_off_route = len(prof), int((~keep).sum())
    prof = prof[keep]

    prof["bin"] = np.floor(prof["s"] / bin_m) * bin_m + bin_m / 2
    grp = prof.groupby("bin")
    binned = grp[["z"]].mean()
    binned["n_frames"] = grp["frame_id"].nunique()
    has_mesh = "z_mesh" in prof.columns and prof["z_mesh"].notna().any()
    if has_mesh:
        # 格內散布要看 z_mesh − z，不能看 z 本身：0.25 m 的格子跨在 12° 的坡上，
        # 光是坡度就讓格內高度差 50 mm，那不是量測分歧。偏差量把坡度消掉了，
        # 剩下的才是「不同幀從不同深度看同一格」的不一致（WWH-14 量到 0.27 mm）。
        prof["dev_mm"] = (prof["z_mesh"] - prof["z"]) * 1000
        grp = prof.groupby("bin")
        binned["z_mesh"] = grp["z_mesh"].mean()
        binned["dev_mm"] = grp["dev_mm"].mean()
        binned["scatter_mm"] = grp["dev_mm"].std()
    else:
        binned["scatter_mm"] = grp["z"].std() * 1000
    binned = binned[binned["n_frames"] >= 3]
    s = binned.index.to_numpy()

    n_panel = 3 if (df is not None and mae_col in df.columns) else 2
    fig, axes = plt.subplots(n_panel, 1, figsize=(13.5, 3.1 * n_panel + 1.2),
                             sharex=True, constrained_layout=True)

    # (a) 地形高度 + 坡度
    ax = axes[0]
    ax.plot(s, binned["z"], "-", color=C_ANALYTIC, lw=2,
            label="analytic  (z, OpenDRIVE centreline)")
    if has_mesh:
        ax.plot(s, binned["z_mesh"], "--", color=C_MESH, lw=1.6,
                label="mesh  (z_mesh, downward ray)")
    ax.set_ylabel("road height (m, world)")
    ax.grid(alpha=.3)
    axg = ax.twinx()
    grade_src = binned["z_mesh"] if has_mesh else binned["z"]
    axg.plot(s, _slope_deg(s, grade_src.to_numpy()), "-", color="0.55", lw=1.0,
             alpha=.8, label="grade (±1 m LS slope)")
    axg.set_ylabel("road grade (deg)", color="0.4")
    axg.tick_params(axis="y", colors="0.4")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axg.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    scatter_label = ("mesh − analytic" if has_mesh else "height")
    ax.set_title(f"Route terrain profile — {dataset_dir.name}   "
                 f"({len(m)} frames, {s_track[-1]:.1f} m route, {bin_m:g} m bins)")
    ax.text(.995, .04,
            f"within-bin scatter of {scatter_label}: "
            f"median {binned['scatter_mm'].median():.2f} mm\n"
            f"unmapped points: {100 * n_off_route / n_pts:.0f}% "
            f"(junction branches / beyond route end)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="0.45",
            bbox=dict(facecolor="white", alpha=.65, edgecolor="none", pad=1.5))

    # (b) 網格 − 解析
    ax = axes[1]
    if has_mesh:
        ax.fill_between(s, 0, binned["dev_mm"], color=C_MESH, alpha=.35)
        ax.plot(s, binned["dev_mm"], "-", color=C_MESH, lw=1.0)
        ax.axhline(0, color=C_ANALYTIC, lw=1.2)
        dev = binned["dev_mm"]
        ax.set_title(f"mesh − analytic:  std {dev.std():.1f} mm, "
                     f"p5/p95 {dev.quantile(.05):+.1f} / {dev.quantile(.95):+.1f} mm, "
                     f"max |dev| {dev.abs().max():.1f} mm", fontsize=10)
    else:
        ax.text(.5, .5, "no z_mesh in this dataset (pre-WWH-14)",
                ha="center", va="center", transform=ax.transAxes, color="0.4")
    ax.set_ylabel("mesh − analytic (mm)")
    ax.grid(alpha=.3)

    # (c) 每幀 MAE，畫在該幀相機所在的 s
    if n_panel == 3:
        ax = axes[2]
        s_cam = dict(zip(m["frame_id"].astype(int), s_track))
        mae = pd.to_numeric(df[mae_col], errors="coerce")
        fid = df["frame_id"].astype(int)
        keep = mae.notna() & fid.isin(s_cam)
        x = fid[keep].map(s_cam).to_numpy()
        y = mae[keep].to_numpy()
        order = np.argsort(x)
        x, y = x[order], y[order]
        # 少數幾幀的 MAE 可以到 10°（路口／越過坡頂看到對面路面），不夾的話
        # 整條 rolling median 會被壓成一條貼地的線
        cap = max(1.0, 4 * np.percentile(y, 90))
        over = y > cap
        ax.scatter(x[~over], y[~over], s=14, color=C_MAE, alpha=.6,
                   label="profile MAE per frame")
        if over.any():
            ax.scatter(x[over], np.full(over.sum(), cap), marker="^", s=36,
                       color="tab:red", clip_on=False, zorder=5,
                       label=f"clipped > {cap:.2f}° (n={int(over.sum())})")
            ax.set_ylim(0, cap)
        if len(y) >= 21:
            roll = pd.Series(y).rolling(21, center=True, min_periods=5).median()
            # 路口那種整段沒有輸出的地方要把線斷開，否則會拉一條假的水平線過去
            roll[np.concatenate(([False], np.diff(x) > 2.0))] = np.nan
            ax.plot(x, roll, "-", color="#d95f02", lw=1.8, label="rolling median (21 frames)")
        ax.axhline(np.mean(y), color="0.5", ls="--", lw=1.0,
                   label=f"mean {np.mean(y):.4f}°")
        ax.set_ylabel("profile MAE (deg)")
        ax.grid(alpha=.3)
        # 可見深度：MAE 的熱點多半是視距被地形切短的地方，兩條疊在一起才看得出來
        if "z_visible_max" in df.columns:
            zv = pd.to_numeric(df.loc[keep, "z_visible_max"], errors="coerce")
            axv = ax.twinx()
            axv.plot(x, zv.to_numpy()[order], "-", color="0.55", lw=1.0, alpha=.8,
                     label="visible depth z_max (m)")
            axv.set_ylabel("visible depth z_max (m)", color="0.4")
            axv.tick_params(axis="y", colors="0.4")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = axv.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
        else:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("route position s (m, along camera track)")

    if save_path is None:
        save_path = Path("outputs") / f"route_profile_{dataset_dir.name}.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return str(save_path)
