"""xyz 到底在哪個座標系？用投影殘差判別。

假說 A：xyz 已在相機光學系  ->  u = fx(-y)/x + cx, v = fy(-z)/x + cy 直接成立
假說 B：xyz 在車體系        ->  要先套 extrinsic 的逆旋轉才對

uv 與 xyz 不逐點對應，所以比「點到投影折線的距離」而不是逐索引。
"""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
OL = pathlib.Path("D:/datasets/openlane")

def proj(xyz, K):
    x, y, z = xyz
    u = K[0,0]*(-y)/x + K[0,2]
    v = K[1,1]*(-z)/x + K[1,2]
    return np.vstack([u, v])

def pt2poly(pts, poly):
    """每個 pts 到 poly 折線的最短距離（poly: 2xN）。"""
    out = []
    A, B = poly[:, :-1], poly[:, 1:]
    AB = B - A
    L2 = (AB**2).sum(0); L2[L2 == 0] = 1e-12
    for p in pts.T:
        t = np.clip(((p[:, None] - A) * AB).sum(0) / L2, 0, 1)
        d = np.linalg.norm(A + t*AB - p[:, None], axis=0)
        out.append(d.min())
    return np.array(out)

resA, resB, rots, heights = [], [], [], []
segs = sorted((OL/"validation").glob("segment-*"))[:12]
for seg in segs:
    for p in sorted(x for x in seg.glob("*.json") if not x.name.startswith("._"))[::20]:
        d = json.load(open(p, encoding="utf-8"))
        K = np.asarray(d['intrinsic'], float)
        E = np.asarray(d['extrinsic'], float)
        R = E[:3, :3]
        rots.append(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))))
        for lane in d['lane_lines']:
            xyz = np.asarray(lane['xyz'], float)
            uv  = np.asarray(lane['uv'],  float)
            if xyz.shape[1] < 5 or uv.shape[1] < 5 or xyz[0].min() <= 0:
                continue
            heights.append(np.median(xyz[2]))
            resA.append(np.median(pt2poly(uv, proj(xyz, K))))
            xyz_b = R.T @ xyz               # 假說 B：先轉回光學系
            if xyz_b[0].min() <= 0: continue
            resB.append(np.median(pt2poly(uv, proj(xyz_b, K))))

resA, resB = np.array(resA), np.array(resB)
print(f"{len(resA)} 條車道線 / {len(segs)} 段\n")
print(f"extrinsic 旋轉角: 中位 {np.median(rots):.3f}°  max {np.max(rots):.3f}°")
print(f"xyz 的 z（高度）中位數: {np.median(heights):+.3f} m  "
      f"-> 原點在{'相機高度' if np.median(heights) < -1 else '地面'}")
print(f"\n投影殘差（標註 uv 到投影折線的距離，px）:")
print(f"  假說 A（xyz 已在光學系，只用內參）: 中位 {np.median(resA):6.3f}  "
      f"p90 {np.percentile(resA,90):6.3f}")
print(f"  假說 B（先套 extrinsic 逆旋轉）    : 中位 {np.median(resB):6.3f}  "
      f"p90 {np.percentile(resB,90):6.3f}")
print(f"\n  A 比 B 好 {np.median(resB)/np.median(resA):.1f} 倍" if np.median(resA) < np.median(resB)
      else f"\n  B 比 A 好 {np.median(resA)/np.median(resB):.1f} 倍")
