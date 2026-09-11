"""OpenLane 車道寬 GT 自身的雜訊 —— 兩個獨立方法互相印證。

⚠ 教訓（WWH-18，2026-09-08）：量誤差時**不能用重疊視窗的一階差分 std**，
那量到的是估計量自身的平滑度，視窗越寬數字越漂亮。這裡兩個方法都避開了：

A（幀內）: 同一幀在 8–30 m 多個深度量寬，對二次趨勢的殘差。
   仍會混進「真實寬度沿路變化」，所以只是**上界**。
B（跨幀）: 同一段**世界位置**的路在連續幀被重複觀測（車前進，同一塊路從
   30 m 逐漸變成 8 m）。依世界里程分箱，箱內跨幀散布 = 純標註雜訊，
   **不需要任何平滑假設** —— 這是主要結論的依據。

2026-09-11 在 Tier A 308 幀上的結果：A σ=13 mm、B 箱內 σ 中位 12 mm
-> 單次觀測雜訊 ≈ 12 mm。對照同一份標註的 pitch 雜訊底 0.25–0.3°：寬度是
兩個橫向位置作差、pitch 要沿深度微分高程，微分放大雜訊，作差不會。
所以主要不確定度不是 GT 雜訊，而是 inset 的 ±2–3 cm（漆寬未知），且那是
系統性偏移 —— 影響絕對 bias，不影響跨段 slope 與「自適應 vs 固定」的比較。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json
from collections import defaultdict
import numpy as np, pandas as pd
from openlane_module.convert_openlane import inner_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--openlane', type=pathlib.Path, default=pathlib.Path('D:/datasets/openlane'))
    ap.add_argument('--split', default='validation')
    ap.add_argument('--frame-list', type=pathlib.Path,
                    default=pathlib.Path('debug/outputs/tierA_frames.csv'))
    ap.add_argument('--bin-m', type=float, default=1.0, help='方法 B 的世界位置分箱寬度')
    args = ap.parse_args()
    depths = np.arange(8.0, 30.01, 1.0)

    fl = pd.read_csv(args.frame_list, dtype={'frame': str})
    resA, binsB = [], defaultdict(list)
    for seg, grp in fl.groupby('segment'):
        want = set(grp.frame)
        js = [p for p in sorted((args.openlane / args.split / seg).glob('*.json'))
              if not p.name.startswith('._')]
        data = {p.stem: json.load(open(p, encoding='utf-8')) for p in js}
        stems = [p.stem for p in js]
        pos = [np.asarray(data[s]['pose'], dtype=float)[:3, 3] for s in stems]
        dist = np.cumsum(np.r_[0.0, [np.linalg.norm(pos[i+1] - pos[i])
                                     for i in range(len(pos) - 1)]])
        dmap = dict(zip(stems, dist))
        for stem in stems:
            if stem not in want:
                continue
            L, R = inner_pair(data[stem]['lane_lines'])
            if not (L and R):
                continue
            xl, xr = L[2], R[2]
            lo = max(xl[0].min(), xr[0].min())
            hi = min(xl[0].max(), xr[0].max())
            dd = depths[(depths >= lo) & (depths <= hi)]
            if len(dd) < 6:
                continue
            w = np.interp(dd, xl[0], xl[1]) - np.interp(dd, xr[0], xr[1])
            resA.extend((w - np.polyval(np.polyfit(dd, w, 2), dd)).tolist())
            for dep, wi in zip(dd, w):
                binsB[(seg, round((dmap[stem] + dep) / args.bin_m))].append(wi)

    resA = np.array(resA)
    mad = 1.4826 * np.median(np.abs(resA - np.median(resA)))
    print(f"方法 A（幀內對二次趨勢的殘差）n={len(resA)}")
    print(f"  σ = {1000*np.std(resA):.0f} mm    MAD*1.4826 = {1000*mad:.0f} mm  （上界）")

    sds = np.array([np.std(v, ddof=1) for v in binsB.values() if len(v) >= 4])
    ns = [len(v) for v in binsB.values() if len(v) >= 4]
    print(f"\n方法 B（同一世界位置跨幀重複觀測）{len(sds)} 個 {args.bin_m} m 分箱，"
          f"每箱 {np.median(ns):.0f} 次觀測")
    print(f"  箱內 σ: 中位 {1000*np.median(sds):.0f} mm   p90 {1000*np.percentile(sds,90):.0f} mm")
    print(f"  => 單次觀測的標註雜訊 ≈ {1000*np.median(sds):.0f} mm")


if __name__ == '__main__':
    main()
