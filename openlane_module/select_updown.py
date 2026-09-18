"""從 export_updown 的輸出挑片段：去掉「大部分是路緣」與「只有單邊」的片段。

逐片段把幀分成三類（依 frames.csv 的 profile_source）：
    both    兩條內側線都是漆線
    single  只有一側是漆線
    none    兩側都不是漆線（只有路緣、或 12 m 處根本沒標註）

去掉：
    none 比例 ≥ NONE_MAX      大部分是路緣／沒標線 —— 我們的方法（漆線寬）與剖面真值都沒有依據
    both 比例 <  BOTH_MIN     幾乎只有單邊 —— 量不到車道寬，pipeline 出不了值

留下的片段以 Windows 目錄連結（junction）放進 --out，不複製影像；原本吃
「一個根目錄底下每個片段一個資料夾」的工具（eval / batch）可以直接指過去。
同時寫 --out/selection.csv（全部 33 段的比例與去留原因）。

    python -m openlane_module.select_updown
"""
import argparse
import subprocess
from pathlib import Path

import pandas as pd

NONE_MAX = 0.5
BOTH_MIN = 0.25


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, default=Path('D:/datasets/openlane_updown'))
    ap.add_argument('--out', type=Path, default=Path('D:/datasets/openlane_updown_twoside'))
    ap.add_argument('--none-max', type=float, default=NONE_MAX)
    ap.add_argument('--both-min', type=float, default=BOTH_MIN)
    a = ap.parse_args(argv)

    f = pd.read_csv(a.src / 'frames.csv')
    f['kind'] = f.profile_source.map({'both': 'both', 'left': 'single',
                                      'right': 'single', 'none': 'none'})
    t = pd.crosstab(f.segment, f.kind, normalize='index').reindex(
        columns=['both', 'single', 'none'], fill_value=0.0)
    t['frames'] = f.groupby('segment').size()
    t['frames_both'] = f[f.kind == 'both'].groupby('segment').size().reindex(t.index, fill_value=0)
    t['grade_60m_median'] = f.groupby('segment').grade_60m.median()
    reason = []
    for r in t.itertuples():
        why = []
        if r.none >= a.none_max:
            why.append(f'mostly_no_paint({r.none:.0%})')
        if r.both < a.both_min:
            why.append(f'mostly_single_side(both {r.both:.0%})')
        reason.append('|'.join(why))
    t['drop_reason'] = reason
    t['keep'] = t.drop_reason == ''
    t = t.round(3).sort_values(['keep', 'both'], ascending=False)

    a.out.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out / 'selection.csv')
    for seg in t[t.keep].index:
        link = a.out / seg
        if not link.exists():
            subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(a.src / seg)],
                           check=True, capture_output=True)
    k = t[t.keep]
    print(f'keep {len(k)}/{len(t)} segments, {int(k.frames.sum())} frames '
          f'({int(k.frames_both.sum())} with both lines) -> {a.out}')


if __name__ == '__main__':
    main()
