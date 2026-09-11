"""近場錨在高相機（OpenLane）上的可行性：視窗在哪、移過去要付多少代價。

兩個問題，都只用 GT 算，不必跑 pipeline：

1. **視窗在不在影像裡** —— 地面深度 z 對應影像列 v = cy + f_y·h/z，所以列距
   地平線的距離正比於 f_y·h。CARLA 是 455×1.08 = 491，OpenLane 是
   828.56×2.116 = 1753，差 3.6 倍 -> z∈[2,5] m 落在 v 607–1133，影像只有
   512 列。看得到的最近地面是 f_y·h/(H/2) = 6.86 m。

2. **移到 7–12 m 的代價** —— 錨假設「那塊路面在 Y = −h」，偏誤就是
   z_h/z_gt − 1，其中 z_h = f_y·h_nom/(v−cy) 是錨假設的深度、z_gt 是 GT 路面
   與該影像列射線的真實交點。再模擬估計器本身（中位數 vs Theil-Sen 截距），
   看 θ0 修正能吸收多少、以及 THETA0_GATE_DEG 會擋掉多少幀。

2026-09-11 在 Tier A 308 幀上的結果：
    視窗 [7,12] m  w_real_med 4.87% (166 mm)   w_real_z0 1.33% (45 mm)
                   |θ0| 中位 0.78°   過 0.3° 閘門者僅 7%
    對照: CARLA z∈[2,5] 實測 8–20 mm；固定 w_real=3.25 是 3.05% (104 mm)
結論：要在高相機上跑近場自標定，(a) 視窗需由 f_y·h 推導、(b) 主估計要改
w_real_z0、(c) 閘門門檻要重新推導 —— 即 to-do §3 第 1 項的 θ0 截距修正。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse
import numpy as np, pandas as pd
from scipy.stats import theilslopes

MIN_PTS = 8


def visibility(f_y, h, H):
    """回傳 (看得到的最近地面深度, z∈[2,5] 視窗的可用列數)。"""
    cy = H / 2.0
    z_bottom = f_y * h / (H - 0.5 - cy)
    rows = sum(1 for v in range(int(cy) + 1, H)
               if 2.0 <= f_y * h / (v - cy) <= 5.0)
    return z_bottom, rows


def load_frames(root, f_y, cy, H, h_nom, inset):
    """逐幀產生 (z_h, z_gt, w_true)。"""
    for seg in sorted(p for p in pathlib.Path(root).iterdir() if p.is_dir()):
        prof = pd.read_csv(seg / "road_profile.csv")
        meas = pd.read_csv(seg / "measurements.csv").set_index("frame_id")
        for fid, g in prof.groupby("frame_id"):
            if fid not in meas.index:
                continue
            z_gt, h_gt = g.x.to_numpy(), g.z.to_numpy()
            v = cy + f_y * (-h_gt) / z_gt          # GT 路面投影到的影像列
            m = (v > cy) & (v < H)
            if m.sum() < 3:
                continue
            yield (f_y * h_nom / (v[m] - cy), z_gt[m],
                   float(meas.loc[fid, "lane_width_gt"]) - inset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='D:/datasets/openlane_tierA')
    ap.add_argument('--f-y', type=float, default=828.56)
    ap.add_argument('--f-x', type=float, default=1104.75)
    ap.add_argument('--height', type=float, default=2.116)
    ap.add_argument('--image-height', type=int, default=512)
    ap.add_argument('--inset', type=float, default=0.25)
    ap.add_argument('--w-nominal', type=float, default=3.4,
                    help='把相對誤差換算成 mm 時用的代表寬度')
    args = ap.parse_args()
    H, cy = args.image_height, args.image_height / 2.0

    print("=== 1. 視窗在不在影像裡 ===")
    for name, f_y, h in [("CARLA", 455.0, 1.08), ("本次", args.f_y, args.height)]:
        zb, n = visibility(f_y, h, H)
        print(f"  {name:6s} f_y·h={f_y*h:7.1f}  最近可見地面 {zb:5.2f} m  "
              f"z∈[2,5] 可用列數 {n}")

    frames = list(load_frames(args.root, args.f_y, cy, H, args.height, args.inset))
    print(f"\n=== 2. 錨假設的深度誤差（{len(frames)} 幀）===")
    z_all = np.concatenate([z for _, z, _ in frames])
    r_all = np.concatenate([zh / zg for zh, zg, _ in frames])
    print(f"{'z 區間':>12s} {'n':>7s} {'偏誤中位':>9s} {'|偏誤| p90':>10s} {'換算 mm':>9s}")
    for lo, hi in [(6.8, 8), (8, 10), (10, 12), (12, 15), (15, 20), (20, 30)]:
        s = (z_all >= lo) & (z_all < hi)
        if s.sum() < 50:
            continue
        e = r_all[s] - 1
        print(f"{lo:5.1f}–{hi:<5.0f} {s.sum():7d} {100*np.median(e):+8.2f}% "
              f"{100*np.percentile(np.abs(e),90):9.2f}% "
              f"{1000*args.w_nominal*np.median(np.abs(e)):8.0f}")

    print(f"\n=== 3. 估計器模擬（w_real_med vs θ0-free 的 w_real_z0）===")
    print(f"{'視窗':>12s} {'幀':>6s} {'w_real_med':>16s} {'w_real_z0':>16s} "
          f"{'|θ0|中位':>9s} {'過0.3°':>7s}")
    for lo, hi in [(7, 10), (7, 11), (7, 12), (8, 12), (8, 15)]:
        me, ze, th = [], [], []
        for z_h, z_gt, w_true in frames:
            sel = (z_h >= lo) & (z_h <= hi)
            if sel.sum() < MIN_PTS:
                continue
            zz = z_h[sel]
            w_est = w_true * (z_h / z_gt)[sel]     # 估計器看到的逐列寬度
            if np.ptp(zz) < 1e-9:
                continue
            fit = theilslopes(w_est, zz)
            me.append(np.median(w_est) / w_true - 1)
            ze.append(fit.intercept / w_true - 1)
            th.append(np.degrees(np.arctan(fit.slope * args.height / fit.intercept))
                      if fit.intercept > 0 else np.nan)
        if not me:
            print(f"  [{lo},{hi}] m -> 0 幀"); continue
        me, ze, th = np.array(me), np.array(ze), np.array(th)
        mm = lambda a: 1000 * args.w_nominal * np.median(np.abs(a))
        print(f"  [{lo:2d},{hi:2d}] m {len(me):6d} "
              f"{100*np.median(np.abs(me)):7.2f}% ({mm(me):4.0f} mm) "
              f"{100*np.median(np.abs(ze)):7.2f}% ({mm(ze):4.0f} mm) "
              f"{np.nanmedian(np.abs(th)):8.2f}° {100*np.mean(np.abs(th)<=0.3):6.0f}%")


if __name__ == '__main__':
    main()
