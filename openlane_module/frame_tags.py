"""逐幀變因表：天氣/時段/地景、OpenLane 六個 case tag（逐幀）、GT 實測曲率、
標線組合、GT 車道寬。同時統計轉換器各道篩選踢掉多少幀。"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from openlane_module.convert_openlane import (inner_pair, PAINT, SOLID, CAT,
                                      WIDTH_MIN, WIDTH_MAX, GRADE_MIN,
                                      GRADE_SPAN, PROFILE_MAX_M, PROFILE_STEP_M)

OL = Path("D:/datasets/openlane")
CURV_SPAN = 50.0

scene = json.load(open(OL / "scene/scene.json", encoding="utf-8"))
cases = {}
for p in (OL / "test").glob("1000_*.txt"):
    tag = p.stem.replace("1000_", "").replace("_case", "")
    s = set()
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            parts = line.split("/")
            s.add((parts[1], Path(parts[2]).stem))
    cases[tag] = s
print("case tags:", {k: len(v) for k, v in cases.items()})

reject = dict(no_pair=0, not_paint=0, width=0, short_span=0, flat=0, ok=0)
rows = []
segs = sorted((OL / "validation").glob("segment-*"))
for i, seg in enumerate(segs):
    if i % 40 == 0:
        print(f"  [{i}/{len(segs)}]", flush=True)
    for jf in sorted(seg.glob("*.json")):
        if jf.name.startswith("._"):
            continue
        data = json.load(open(jf, encoding="utf-8"))
        L, R = inner_pair(data["lane_lines"])
        if not (L and R):
            reject["no_pair"] += 1; continue
        if L[1] not in PAINT or R[1] not in PAINT:
            reject["not_paint"] += 1; continue
        width = L[0] - R[0]
        xl, xr = L[2], R[2]
        z0 = max(xl[0].min(), xr[0].min())
        z1 = min(xl[0].max(), xr[0].max(), PROFILE_MAX_M)
        if not (WIDTH_MIN <= width <= WIDTH_MAX):
            reject["width"] += 1; continue
        if z1 - z0 < 5.0:
            reject["short_span"] += 1; continue
        depth = np.arange(z0, z1 + 1e-9, PROFILE_STEP_M)
        h = 0.5 * (np.interp(depth, xl[0], xl[2]) + np.interp(depth, xr[0], xr[2]))
        near = depth <= z0 + GRADE_SPAN
        grade = float(h[near].max() - h[near].min())
        if grade < GRADE_MIN:
            reject["flat"] += 1; continue
        reject["ok"] += 1

        # 曲率
        c1 = min(xl[0].max(), xr[0].max(), z0 + CURV_SPAN)
        if c1 - z0 >= 10.0:
            x = np.arange(z0, c1, 0.5)
            yc = 0.5 * (np.interp(x, xl[0], xl[1]) + np.interp(x, xr[0], xr[1]))
            chord = np.interp(x, [x[0], x[-1]], [yc[0], yc[-1]])
            sag = float(np.abs(yc - chord).max())
            co = np.polyfit(x, yc, 2)
            head = float(np.degrees(np.arctan(np.polyval(np.polyder(co), x[-1]))
                                    - np.arctan(np.polyval(np.polyder(co), x[0]))))
        else:
            sag = head = np.nan
        key = (seg.name, jf.stem)
        sc = scene.get(seg.name, {})
        rows.append(dict(
            segment=seg.name, frame=jf.stem,
            weather=sc.get("weather", "?"), hours=sc.get("hours", "?"),
            scene=sc.get("scene", "?"),
            **{f"tag_{t}": int(key in s) for t, s in sorted(cases.items())},
            left_cat=CAT.get(L[1], L[1]), right_cat=CAT.get(R[1], R[1]),
            both_solid=int(L[1] in SOLID and R[1] in SOLID),
            w_gt=round(width, 4), sagitta=round(sag, 4) if sag == sag else np.nan,
            heading_deg=round(head, 3) if head == head else np.nan,
            z_gt_min=round(float(z0), 2), z_gt_max=round(float(z1), 2),
            grade_60m=round(grade, 4)))

df = pd.DataFrame(rows)
out = REPO / "debug/outputs/openlane_frame_tags.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print("\n篩選漏斗:", reject)
print(f"{len(df)} frames / {df.segment.nunique()} segments -> {out}")
