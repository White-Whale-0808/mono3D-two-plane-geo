"""直接量 OpenLane 的漆線內縮 inset —— 量距離，不套公式。

要解決的問題（WWH-18 / WWH-19）
    GT 標的是漆線**中心**，``w_real`` 要的是**內緣**，兩者差一個 inset。
    目前 OpenLane 的 inset 是沿用 CARLA 的漆寬 0.125 m 推出來的，整組是猜的：
    美國 MUTCD 常規線 4–6 吋 → 總內縮 0.203–0.305 m，**±5 cm ＝ 3.3 m 車道上的
    1.5%**，比要驗收的 1.05% 還大。不確定度大於要驗收的量，就該先去量它。

    ⚠ 不能用「pipeline 的 bias 只有 13 mm 所以 inset=0.25 對」反推 —— 那假設了
    pipeline 無偏差，是循環論證。這支腳本完全不經過 pipeline 的寬度，只用
    影像 ＋ GT 的橫向位置與深度，所以不循環。

量什麼
    對每個探測深度，把 GT 標的線中心投影到影像得到 ``u0``，再在該列的水平
    亮度剖面上找出這條標線（單線或雙線整組）的**漆料範圍**，取靠車道中心那
    一側的邊緣 ``u_inner``：

        inset = (u_inner − u0) · z / f_x        （左線；右線取相反方向）

    順帶得到整組漆料的跨距（單線＝線寬、雙線＝2×線寬＋間隙），可以反過來
    檢查「間隙≈線寬」這個假設成不成立。

邊緣怎麼定
    半高寬：背景取窗外兩側的中位數，門檻 = 背景 + 0.5×(峰值 − 背景)，
    跨越點做線性內插到次像素。雙線的中心落在中間的間隙（暗），所以不是找
    「包含 u0 的那一段」，而是找**整組的最內側邊界** —— 從窗內最靠內的那個
    「亮→暗」跨越點開始算，單線與雙線因此走同一套邏輯，不必先知道是哪一種。

解析度
    虛擬相機 f_x = 1104.75，10 m 處 1 px = 9 mm、14 m 處 1 px = 13 mm。
    次像素內插後單次量測約 ±1 px，所以中位數值的不確定度遠小於要消掉的 ±5 cm。

CARLA 驗證模式（``--carla``）
    同一套邊緣邏輯跑在 CARLA 影像上，錨點改用 pipeline 自己的內緣曲線，量
    整組漆料的跨距。真值已知：右側單白 0.125 m、左側雙黃 2×0.125 ＋ 間隙
    ≈ 0.375 m。這是在驗**量測方法本身**，不是驗 OpenLane。

用法::

    python -m openlane_module.measure_paint_inset                     # Tier A 全部
    python -m openlane_module.measure_paint_inset --limit-segments 3
    python -m openlane_module.measure_paint_inset --carla inference_datasets/carla_dataset_Town03_uphile --limit 40
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.env_setup import setup_env  # noqa: E402

setup_env()

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from openlane_module.convert_openlane import (CAT, inner_pair, VIRT_FX, VIRT_FY,  # noqa: E402
                                              VIRT_CX, VIRT_CY)

PROBE_Z = np.arange(8.0, 14.01, 1.0)   # 探測深度（m）：8 m 是 GT 標註的最近處，
                                       # 14 m 之後單線只剩 10 px，半高寬不可靠
GROUP_HALF_M = 0.50                    # 視窗半寬（m）：要蓋得住雙線整組
BG_OUTER_M = (0.60, 1.00)              # 背景取樣區（m，離 u0 這麼遠的路面）
MIN_CONTRAST = 8.0                     # 峰值 − 背景的下限（gray levels）


def _gray(img):
    """Luma of an RGB image.

    Both callers flip the cv2 BGR frame with [:, :, ::-1] before calling, so the
    input is RGB and the weights must be R,G,B = 0.299, 0.587, 0.114 — the same
    as paint_evidence._gray. This used to carry the BGR order, which swapped red
    and blue: yellow paint (high R, almost no B) lost most of its contrast, so
    yellow-line rows were either rejected by MIN_CONTRAST or had their edges
    misplaced. White lines were barely affected.
    """
    g = np.asarray(img, dtype=np.float32)
    return g @ np.array([0.299, 0.587, 0.114], np.float32) if g.ndim == 3 else g


def paint_group_edges(gray, v, u0, px_per_m, inner_sign):
    """回傳 (u_inner, u_outer, contrast)；找不到回 None。

    ``inner_sign`` = +1 表示車道中心在 u 增加的方向（左側線），−1 則相反。
    """
    h, w = gray.shape
    vi = int(round(v))
    if not (1 <= vi <= h - 2):
        return None
    strip = gray[vi - 1:vi + 2].mean(axis=0)
    half = int(round(GROUP_HALF_M * px_per_m))
    lo, hi = int(round(u0)) - half, int(round(u0)) + half
    bg_lo, bg_hi = (int(round(BG_OUTER_M[0] * px_per_m)),
                    int(round(BG_OUTER_M[1] * px_per_m)))
    if lo - bg_hi < 0 or hi + bg_hi >= w or half < 3:
        return None

    bg = float(np.median(np.concatenate([
        strip[int(round(u0)) - bg_hi:int(round(u0)) - bg_lo],
        strip[int(round(u0)) + bg_lo:int(round(u0)) + bg_hi]])))
    win = strip[lo:hi + 1]
    peak = float(win.max())
    if peak - bg < MIN_CONTRAST:
        return None
    thr = bg + 0.5 * (peak - bg)
    mask = win >= thr
    if not mask.any():
        return None

    # 亮區先切成連續段，再依「雙線的間隙 ≈ 線寬」合併 —— 不能直接取窗內最外
    # 與最內的亮像素，那會把窗內任何不相干的亮東西（另一條車道線、補丁、反光）
    # 一起吃進來，單線會被量成雙線那麼寬。
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    starts = np.r_[0 if mask[0] else np.array([], int), edges[mask[edges + 1]] + 1]
    ends = np.r_[edges[~mask[edges + 1]], len(mask) - 1 if mask[-1] else np.array([], int)]
    runs = [(int(s), int(e)) for s, e in zip(np.sort(starts), np.sort(ends)) if e >= s]
    if not runs:
        return None
    c0 = u0 - lo
    k = int(np.argmin([abs(0.5 * (s + e) - c0) for s, e in runs]))
    s, e = runs[k]
    for step in (-1, 1):                       # 最多各併一段（雙線就是一對）
        j = k + step
        if 0 <= j < len(runs):
            gap = runs[j][0] - e - 1 if step > 0 else s - runs[j][1] - 1
            width = max(e - s, runs[j][1] - runs[j][0]) + 1
            if 0 <= gap <= 1.5 * width:
                s, e = min(s, runs[j][0]), max(e, runs[j][1])
    if (e - s + 1) > 0.5 * px_per_m:           # 整組超過 0.5 m 就不是標線
        return None
    i_in, i_out = (e, s) if inner_sign > 0 else (s, e)

    def cross(i, step):
        """從亮像素 i 往 step 方向找門檻跨越點，線性內插到次像素。"""
        j = i + step
        if not (0 <= j < len(win)):
            return lo + i + 0.5 * step
        a, b = win[i], win[j]
        t = 0.5 if a == b else (a - thr) / (a - b)
        return lo + i + step * float(np.clip(t, 0.0, 1.0))

    u_inner = cross(i_in, 1 if inner_sign > 0 else -1)
    u_outer = cross(i_out, -1 if inner_sign > 0 else 1)
    return u_inner, u_outer, peak - bg


def run_openlane(args):
    import cv2
    fl = pd.read_csv(args.frame_list, dtype={'frame': str})
    # --frame-list 選的是「幀」，不是「段」。只挑段的話，一份完整（非 Tier A）
    # 的轉換會把那些段裡的虛線/路口/彎道幀通通混進 inset 統計裡 —— 那正是
    # Tier A 篩掉的東西。語意與 convert_openlane.convert_segment 的 keep 一致。
    keep = set(zip(fl.segment, fl.frame))
    segs = sorted(fl.segment.unique())
    if args.limit_segments:
        segs = segs[:args.limit_segments]
    rows = []
    for seg in segs:
        seg_dir = pathlib.Path(args.root) / seg
        if not (seg_dir / 'measurements.csv').exists():
            print(f"  skip {seg[:30]} (未轉出)")
            continue
        meas = pd.read_csv(seg_dir / 'measurements.csv', dtype={'source_frame': str})
        for r in meas.itertuples():
            if (seg, r.source_frame) not in keep:
                continue
            js = pathlib.Path(args.openlane) / 'validation' / seg / f'{r.source_frame}.json'
            img_path = seg_dir / 'images' / f'{int(r.frame_id):06d}.png'
            if not (js.exists() and img_path.exists()):
                continue
            data = json.load(open(js, encoding='utf-8'))
            L, R = inner_pair(data['lane_lines'])
            if not (L and R):
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = _gray(img[:, :, ::-1])
            for side, line, sign in (('left', L, +1), ('right', R, -1)):
                xyz, cat = line[2], CAT.get(line[1], line[1])
                for z in PROBE_Z:
                    if not (xyz[0].min() <= z <= xyz[0].max()):
                        continue
                    y = float(np.interp(z, xyz[0], xyz[1]))
                    hgt = float(np.interp(z, xyz[0], xyz[2]))
                    u0 = VIRT_FX * (-y) / z + VIRT_CX
                    v = VIRT_FY * (-hgt) / z + VIRT_CY
                    got = paint_group_edges(gray, v, u0, VIRT_FX / z, sign)
                    if got is None:
                        continue
                    u_in, u_out, contrast = got
                    rows.append(dict(
                        segment=seg, frame_id=int(r.frame_id), side=side,
                        category=cat, z=float(z),
                        inset_m=sign * (u_in - u0) * z / VIRT_FX,
                        group_m=abs(u_in - u_out) * z / VIRT_FX,
                        contrast=contrast, px_per_m=VIRT_FX / z))
        print(f"  {seg[:34]}  累計 {len(rows)} 筆", flush=True)
    return pd.DataFrame(rows)


def run_carla(args):
    """方法驗證：同一套邊緣邏輯量 CARLA 的漆料跨距，真值 0.125 / 0.375。"""
    import cv2
    import yaml
    from libs.inference.road_segmentation import load_pidnet
    from libs.inference.pipeline import infer_one

    cfg = yaml.safe_load(open('config/inference_road_lane_segmentation.yaml',
                              encoding='utf-8'))
    pe, mo, lf = cfg['pitch_estimation'], cfg['model'], cfg['lane_fitting']
    f_x, f_y, cam_h = pe['f_x'], pe['f_y'], pe['camera_height']
    resize = tuple(cfg['input']['resize_size'])
    model = load_pidnet(mo['model_name'], mo['weight_path'], mo['device'])
    ds = pathlib.Path(args.carla)
    meas = pd.read_csv(ds / 'measurements.csv')
    if args.limit:
        meas = meas.head(args.limit)

    rows = []
    for r in meas.itertuples():
        img_path = ds / 'images' / f'{int(r.frame_id):06d}.png'
        if not img_path.exists():
            continue
        try:
            dbg = infer_one(model, str(img_path), mo['device'], resize,
                            cfg['line_segmentation']['min_segment_length_near'],
                            cfg['line_segmentation']['min_segment_length_far'],
                            lf['num_samples'], f_x, f_y, pe['w_real'], cam_h,
                            samples_per_meter=lf.get('samples_per_meter'),
                            track_bands=cfg['lane_segmentation'].get('track_bands', 16),
                            method=pe.get('method', 'windowed'),
                            return_debug=True)['debug']
        except Exception:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gray = _gray(cv2.resize(img[:, :, ::-1], (resize[1], resize[0])))
        for side, key, sign in (('left', 'left_curve', +1), ('right', 'right_curve', -1)):
            cv = dbg.get(key)
            if not (isinstance(cv, dict) and len(cv.get('y', ()))):
                continue
            # 曲線是逐列的 {'y': 列, 'x': 內緣欄}；取量測深度範圍內最近的幾列，那裡漆最寬。
            # ⚠ 必須先限定 z 再取最近的列，順序不能反：左線的內緣鏈常一路延伸到影像
            # 底部（z ≈ 1.9 m），若先取最近 40 列再過濾 z >= 3，40 列會全部被刷掉 ——
            # 雙黃線那一側因此從來沒產生過任何一筆，驗證只驗到了單白線。
            pts = np.column_stack([np.asarray(cv['x'], float),
                                   np.asarray(cv['y'], float)])
            zs = f_y * cam_h / np.maximum(pts[:, 1] - resize[0] / 2.0, 1e-6)
            pts = pts[(zs >= 3.0) & (zs <= 12.0)]
            pts = pts[np.argsort(-pts[:, 1])][:40]
            for u_edge, v in pts[::8]:
                z = f_y * cam_h / max(v - resize[0] / 2.0, 1e-6)
                got = paint_group_edges(gray, v, u_edge, f_x / z, sign)
                if got is None:
                    continue
                u_in, u_out, contrast = got
                rows.append(dict(frame_id=int(r.frame_id), side=side, z=z,
                                 group_m=abs(u_in - u_out) * z / f_x,
                                 contrast=contrast))
    return pd.DataFrame(rows)


def blur_diagnostic(df, f_x, label=""):
    """半高寬有沒有被模糊撐大？看 group_m 對深度的斜率。

    影像模糊會讓半高寬多出**固定的像素數** δ，換算成公尺就是 δ·z/f_x ——
    也就是 group_m 對 z 的斜率 = δ/f_x。真正的漆寬與深度無關，斜率應為 0。
    這是 2026-09-12 在 CARLA 上發現的：那裡的漆線在近場只有 12 px 寬、與模糊
    尺度相當，量到 +11.6 mm/m（δ≈+7~10 px），單白量成 0.204 m 而不是 0.125；
    把斜率外插回 z=0 得 0.137 m，反而接近真值。OpenLane 的漆線在 8 m 處有
    25 px，實測斜率 −0.2 mm/m（δ≈−0.2 px），沒有這個問題。
    """
    if len(df) < 20 or df.z.nunique() < 3:
        return
    slope, intercept = np.polyfit(df.z, df.group_m, 1)
    delta_px = slope * f_x
    flag = "⚠ 半高寬被模糊撐大" if abs(delta_px) > 1.0 else "OK"
    print(f"\n模糊診斷{label}: group_m 對 z 斜率 {1000*slope:+.1f} mm/m "
          f"(等效 {delta_px:+.1f} px)  z→0 截距 {intercept:.4f} m  -> {flag}")


def summarise(df, key, cols):
    g = df.groupby(key)
    out = g.agg(n=('z', 'size'), **{c: (c, 'median') for c in cols})
    out['mad_mm'] = g[cols[0]].apply(
        lambda s: 1000 * 1.4826 * np.median(np.abs(s - s.median()))).round(1)
    return out.round(4)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='D:/datasets/openlane_tierA')
    ap.add_argument('--openlane', default='D:/datasets/openlane')
    ap.add_argument('--frame-list', default='debug/outputs/tierA_frames.csv')
    ap.add_argument('--limit-segments', type=int, default=None)
    ap.add_argument('--carla', default=None, help='改跑 CARLA 方法驗證模式')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.carla:
        df = run_carla(args)
        out = pathlib.Path(args.out or 'debug/outputs/paint_group_carla.csv')
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\n{len(df)} 筆 -> {out}")
        if len(df):
            print(summarise(df, 'side', ['group_m', 'contrast']).to_string())
            import yaml as _yaml
            _fx = _yaml.safe_load(open('config/inference_road_lane_segmentation.yaml',
                                       encoding='utf-8'))['pitch_estimation']['f_x']
            # Per side: the two sides carry different true widths (single white
            # 0.125, double yellow 0.375) at different depth mixes, so one pooled
            # group_m-vs-z fit measures the side mix, not blur. It only looked
            # sane before because the left side never produced a single row.
            for _side in ("left", "right"):
                _d = df[df.side == _side]
                if len(_d) >= 3:
                    blur_diagnostic(_d, _fx, f"（CARLA {_side}）")
            print("\n真值：右側單白 0.125 m、左側雙黃 2×0.125＋間隙 ≈ 0.375 m")
        return

    df = run_openlane(args)
    out = pathlib.Path(args.out or 'debug/outputs/openlane_paint_inset.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df)} 筆量測 -> {out}")
    if not len(df):
        return
    blur_diagnostic(df[df.category == 'white-solid'], VIRT_FX, "（單白實線）")
    print("\n逐標線類別（中位數，m）：")
    print(summarise(df, 'category', ['inset_m', 'group_m', 'contrast']).to_string())
    print("\n逐側：")
    print(summarise(df, 'side', ['inset_m', 'group_m', 'contrast']).to_string())
    per_frame = df.pivot_table(index=['segment', 'frame_id'], columns='side',
                               values='inset_m', aggfunc='median').dropna()
    if len(per_frame):
        tot = per_frame['left'] + per_frame['right']
        print(f"\n逐幀總內縮（左＋右）中位 {tot.median():.4f} m  "
              f"p5..p95 {tot.quantile(.05):.4f}..{tot.quantile(.95):.4f}  n={len(tot)}")
        print(f"對照：目前沿用 CARLA 的假設值 0.25 m（單白 0.0625 ＋ 雙黃 0.1875）")


if __name__ == '__main__':
    main()
