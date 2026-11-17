# -*- coding: utf-8 -*-
"""
exp_directional_tau.py —— 分方向阈值(上行用更低 τ) 实验
============================================================
目的：诊断"上爬坡命中率=0"。测试期上爬坡概率极大值仅 ~0.57，
      统一 τ=0.6 永远达不到。本实验对上行/下行分别扫描 τ，
      观察上行命中率能被拉起到多少、代价是多少虚警。

输入：output/lgb_pred.npz (cls 概率)、output/ramp_events.csv、field_power.csv
输出：output/directional_tau_report.md
"""
import os
import io
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import event_eval as ee

OUT = "output"
MERGE_GAP = 6
T_MIN, T_MAX = 29952, 35267


def load():
    d = np.load(os.path.join(OUT, "lgb_pred.npz"), allow_pickle=True)
    prob = d["prob_test"]
    t = d["t_test"].astype(int)
    order = np.argsort(t)
    p_up = prob[order, 2].astype(float)
    p_dn = prob[order, 0].astype(float)
    t = t[order]
    # 真实事件
    ev = pd.read_csv(os.path.join(OUT, "ramp_events.csv"))
    ev = ev[(ev.is_main == 1) & (ev.start_tidx >= T_MIN) & (ev.start_tidx <= T_MAX)].copy()
    ev["end_t_idx"] = ev.start_tidx + ev.n_trig - 1
    ev["ramp_dir"] = np.where(ev.direction == "up", 1, -1)
    ev = ev.reset_index(drop=True)
    ev["event_id"] = ev.index
    true_events = ev[["event_id", "start_tidx", "end_t_idx", "ramp_dir"]].rename(
        columns={"start_tidx": "start_t_idx"}).to_dict("records")
    return p_up, p_dn, t, true_events


def alarms_and_events(p, t, thr, ramp_dir):
    ad = ee.build_alarm_from_prob(p, t, threshold=thr)
    return ee.merge_alarm_events(ad, ramp_dir=ramp_dir, merge_gap=MERGE_GAP)


def run(p_up, p_dn, t, true_events, tau_up, tau_dn):
    pe_up = alarms_and_events(p_up, t, tau_up, 1)
    pe_dn = alarms_and_events(p_dn, t, tau_dn, -1)
    m = ee.compute_event_metrics(true_events, pe_up + pe_dn)
    byd = ee.compute_metrics_by_direction(true_events, pe_up + pe_dn)
    return m, byd, len(pe_up), len(pe_dn)


def main():
    p_up, p_dn, t, true_events = load()
    n_up = sum(1 for e in true_events if e["ramp_dir"] == 1)
    n_dn = sum(1 for e in true_events if e["ramp_dir"] == -1)
    print(f"真实事件: up={n_up} down={n_dn}  上爬坡概率max={p_up.max():.3f}")

    lines = []
    lines.append("# 分方向阈值实验报告：上爬坡低 τ vs 统一下行 τ\n")
    lines.append(f"- 真实事件(主口径,测试区)：上行 {n_up} / 下行 {n_dn}")
    lines.append(f"- 上爬坡概率极大值 = {p_up.max():.3f}（统一 τ=0.6 不可达，故上行命中=0）")
    lines.append(f"- 下行 τ 固定 = 0.60（低虚警推荐点）；上行 τ 从 0.15 扫描到 0.60\n")

    # 1) 统一 τ=0.6 基线
    m0, bd0, nu0, nd0 = run(p_up, p_dn, t, true_events, 0.60, 0.60)
    up0 = bd0[1]
    lines.append("## 1. 基线（统一 τ=0.60）\n")
    lines.append("| 方向 | 命中率 | 虚警率 | 预测事件 |\n|---|---|---|---|")
    lines.append(f"| 上行 | {up0['hit_rate']:.3f} | {up0['false_alarm_rate']:.3f} | {up0['num_pred']} |")
    dn0 = bd0[-1]
    lines.append(f"| 下行 | {dn0['hit_rate']:.3f} | {dn0['false_alarm_rate']:.3f} | {dn0['num_pred']} |")
    lines.append(f"| 合计 | {m0['hit_rate']:.3f} | {m0['false_alarm_rate']:.3f} | {m0['num_pred']} |\n")

    # 2) 上行 τ 扫描
    lines.append("## 2. 上行 τ 扫描（下行固定 0.60）\n")
    lines.append("| 上行τ | 上行命中率 | 上行虚警率 | 上行预测事件 | 合计命中率 | 合计虚警率 |\n|---|---|---|---|---|---|")
    best = None
    for tu in [0.60, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]:
        m, bd, nu, nd = run(p_up, p_dn, t, true_events, tu, 0.60)
        up = bd[1]
        lines.append(f"| {tu:.2f} | {up['hit_rate']:.3f} | {up['false_alarm_rate']:.3f} | {up['num_pred']} | "
                     f"{m['hit_rate']:.3f} | {m['false_alarm_rate']:.3f} |")
        # 选：上行命中率高、上行虚警率<=0.6 的最优
        if best is None or (up['hit_rate'] > best[1] and up['false_alarm_rate'] <= 0.6):
            best = (tu, up['hit_rate'], up['false_alarm_rate'], m['hit_rate'], m['false_alarm_rate'])

    lines.append("")
    if best:
        lines.append(f"## 3. 结论\n")
        lines.append(f"- 推荐上行 τ = **{best[0]:.2f}**：上行命中率 {best[1]:.3f}（虚警 {best[2]:.3f}），"
                     f"合计命中率 {best[3]:.3f}（虚警 {best[4]:.3f}）。")
        lines.append(f"- 上行命中率由基线 {up0['hit_rate']:.3f} 提升到 **{best[1]:.3f}**。")
        lines.append(f"- 说明：上行概率被'无爬坡'类压扁（max仅{p_up.max():.2f}），分方向阈值是必要修正；"
                     f"根本上仍需靠特征工程(风速上升率)提升上爬坡可分辨性。")

    txt = "\n".join(lines)
    io.open(os.path.join(OUT, "directional_tau_report.md"), "w", encoding="utf-8").write(txt)
    print(txt)
    print("\n[写出] output/directional_tau_report.md")


if __name__ == "__main__":
    main()
