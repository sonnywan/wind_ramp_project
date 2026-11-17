# -*- coding: utf-8 -*-
"""
run_event_eval.py —— 事件级评价 端到端回测（M1：复用 LGB 三分类概率）
==================================================================
链路：
  ramp_events.csv (is_main==1) ──> true_events
  lgb_pred.npz.prob_test[:,2] (上行概率) ──> 上行告警事件
  lgb_pred.npz.prob_test[:,0] (下行概率) ──> 下行告警事件
  两方向拼接 ──> pred_events
  ──> compute_event_metrics / compute_metrics_by_direction ──> 报告

口径：
  - 真实事件 end_tidx = start_tidx + n_trig - 1
  - direction: up->+1, down->-1
  - 只取 is_main == 1
  - 告警合并：同向间隔 <= MERGE_GAP(6) 步合并

说明：M1 直接复用三分类模型的概率列（prob[:,2]=上行, prob[:,0]=下行），
     不重新训练方向模型。若后续上 M2 再替换为独立方向二分类概率。
"""
import os
import sys
import numpy as np
import pandas as pd

from event_eval import (
    build_alarm_from_prob,
    merge_alarm_events,
    compute_event_metrics,
    compute_metrics_by_direction,
)

OUT_DIR = "output"
EVENT_CSV = os.path.join(OUT_DIR, "ramp_events.csv")
PRED_NPZ = os.path.join(OUT_DIR, "lgb_pred.npz")          # 分类概率源（三分类）
REG_PROB_NPZ = os.path.join(OUT_DIR, "reg_prob.npz")       # 回归残差概率源（1b 产物）
REPORT_MD = os.path.join(OUT_DIR, "event_eval_report.md")

# ---- 概率数据源开关（二选一）----
# "cls" : 三分类模型概率（prob[:,2]=上行, prob[:,0]=下行）——原 M1 口径
# "reg" : 回归残差经验分布概率（reg_prob.npz 的 p_up/p_dn）——创新点2 产物
# 命令行可用 --source cls|reg 覆盖；默认 cls，保持向后兼容。
PROB_SOURCE = os.environ.get("PROB_SOURCE", "cls").strip().lower()

MERGE_GAP = 6          # 同向告警间隔 <= 6 步合并（Δt=1h）
L_STEPS = 6            # 预警提前量 L=1h=6 步（label 定义：t 时刻预测 [t+L, t+L+Δt]）
DT_STEPS = 6           # 事件窗 Δt=1h=6 步
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
DEFAULT_THRESHOLD = 0.4

# 迟滞参数（方案 §2：连续 2 步超阈发令；回落至 0.8θ 撤令）
HYST_ON_STEPS = 2      # 连续 N 步 >= τ 才置告警
HYST_OFF_RATIO = 0.8   # 告警态下概率 <= ratio*τ 才撤告警

# ---- 代价敏感阈值标定（创新点3 配套）----
# 代价比 = C_miss : C_false_alarm（固定 C_false_alarm = 1）
# TotalCost = ratio * num_miss + num_false_alarm
# 归一化代价 = TotalCost / num_true
COST_RATIOS = [1, 5, 10]
COST_C_FALSE = 1.0
COST_SCAN_START = 0.05
COST_SCAN_END = 0.95
COST_SCAN_STEP = 0.05
COST_FIG_PATH = os.path.join(OUT_DIR, "cost_curve.png")


def log(m):
    print(f"[event_eval] {m}")


def load_pred_source(source, t_min=None, t_max=None):
    """按数据源加载方向概率（双通道），返回 (p_up, p_down, t_idx)。

    source="cls" : lgb_pred.npz.prob_test（列2=上行, 列0=下行）
    source="reg" : reg_prob.npz（p_up_test / p_dn_test，来自残差经验分布）
    两套 t_test 已确认逐元素对齐，可直接同源对比。
    """
    source = str(source).strip().lower()
    if source == "reg":
        d = np.load(REG_PROB_NPZ)
        p_up = d["p_up_test"].astype(float)
        p_down = d["p_dn_test"].astype(float)
        t_idx = d["t_test"].astype(np.int64)
        tag = "回归残差概率(reg_prob.npz)"
    else:
        d = np.load(PRED_NPZ, allow_pickle=True)
        prob = d["prob_test"]
        p_up = prob[:, 2].astype(float)
        p_down = prob[:, 0].astype(float)
        t_idx = d["t_test"].astype(np.int64)
        tag = "三分类概率(lgb_pred.npz)"
        source = "cls"
    if t_min is not None and t_max is not None:
        mask = (t_idx >= t_min) & (t_idx <= t_max)
        # 两套都应整体落在测试区间内；此处仅防御性截取
        p_up, p_down, t_idx = p_up[mask], p_down[mask], t_idx[mask]
    log(f"概率源={source} [{tag}]: N={len(t_idx)}, 区间[{int(t_idx.min())},{int(t_idx.max())}]")
    return p_up, p_down, t_idx, source


def load_true_events(path, t_min=None, t_max=None, main_only=True):
    """读取真实事件，转成评价所需结构。

    main_only=True （默认）：只取主口径 is_main==1（前 1–6 节用，口径统一）。
    main_only=False        ：取全部含等级事件（第 7 节分级评价专用，含二/三级）。

    修复 A：只保留事件起点落在测试时间区间 [t_min, t_max] 内的事件，
    避免把训练期/验证期事件误算为漏警（这些区间本就没有预测覆盖）。
    """
    df = pd.read_csv(path)
    if main_only:
        df = df[df["is_main"] == 1].copy()
    else:
        df = df.copy()
    df["ramp_dir"] = df["direction"].map({"up": 1, "down": -1})
    df = df[df["ramp_dir"].notna()]
    df["start_t_idx"] = df["start_tidx"].astype(int)
    df["end_t_idx"] = df["start_tidx"].astype(int) + df["n_trig"].astype(int) - 1

    n_all = len(df)
    # 修复 A：起点落在测试区间内
    if t_min is not None and t_max is not None:
        df = df[(df["start_t_idx"] >= t_min) & (df["start_t_idx"] <= t_max)].copy()
    df["event_id"] = df.index.astype(str)
    # 三级分级回传：真实事件自带 level（1/2/3），随事件记录一并带入
    if "level" not in df.columns:
        df["level"] = 0
    df["level"] = df["level"].fillna(0).astype(int)
    events = df[["event_id", "start_t_idx", "end_t_idx", "ramp_dir", "level"]].to_dict("records")
    tag = "主口径 is_main==1" if main_only else "全量(含二/三级)"
    log(f"真实事件({tag}): 全量 {n_all} 条 -> 测试区间内 {len(events)} 条")
    return events, df


def build_alarm_with_hysteresis(pred_prob, t_idx, threshold,
                                on_steps=HYST_ON_STEPS,
                                off_ratio=HYST_OFF_RATIO):
    """迟滞决策层：把逐时间步概率 -> 逐时间步告警标记（含迟滞）。

    规则（对应方案 §2）：
      - 触发：连续 on_steps(默认2) 步概率 >= threshold 才置告警=1；
      - 撤销：告警态下，概率 <= off_ratio*threshold 才置回 0；
      - 其他情况保持上一状态。

    返回与 build_alarm_from_prob 相同的 {t_idx: 0/1}，保证下游无缝复用。
    """
    # 按时间升序处理（t_idx 未必有序）
    prob_map = {int(t): float(p) for p, t in zip(pred_prob, t_idx)}
    times = sorted(prob_map.keys())

    alarm = {t: 0 for t in times}
    state = 0                 # 当前告警态
    consec_on = 0            # 连续超阈计数
    off_thr = off_ratio * threshold

    for t in times:
        p = prob_map[t]
        if state == 0:
            # 未告警：累计连续超阈步数
            if p >= threshold:
                consec_on += 1
            else:
                consec_on = 0
            if consec_on >= on_steps:
                state = 1
                alarm[t] = 1
            else:
                alarm[t] = 0
        else:
            # 已告警：仅在概率回落到 off_thr 以下才撤销
            if p <= off_thr:
                state = 0
                consec_on = 0
                alarm[t] = 0
            else:
                alarm[t] = 1

    return alarm


def _shift_alert_to_window(events):
    """修复 B(a)：把“告警时刻窗口”平移为“预测覆盖窗口”。

    merge_alarm_events 返回的 start/end 是告警时刻 t；label 定义在 t 时刻
    预测 [t+L, t+L+Δt] 的爬坡。故匹配窗口应为各告警时刻平移 +L 后的并集：
      start_match = alert_start + L
      end_match   = alert_end   + L + (DT_STEPS - 1)
    同时保留原始告警起点 alert_t（= alert_start），用于提前量计算。
    """
    out = []
    for e in events:
        alert_t = int(e["start"])
        out.append({
            "start": alert_t + L_STEPS,
            "end": int(e["end"]) + L_STEPS + (DT_STEPS - 1),
            "ramp_dir": int(e["ramp_dir"]),
            "alert_t": alert_t,          # 原始告警起点，用于 lead_time
        })
    return out


def build_pred_events(p_up, p_down, t_idx, threshold, use_hysteresis=False):
    """用双通道方向概率构造双向预测告警事件。

    p_up   : 上行概率序列（长度 N）
    p_down : 下行概率序列（长度 N）
    概率来源由外部 load_pred_source 决定（分类 / 回归残差），本函数与源解耦。
    use_hysteresis: True 时用迟滞决策层，否则用简单阈值。
    返回的事件窗口已按 B(a) 平移为预测覆盖窗口，并带 alert_t。
    """

    if use_hysteresis:
        alarm_up = build_alarm_with_hysteresis(p_up, t_idx, threshold)
        alarm_down = build_alarm_with_hysteresis(p_down, t_idx, threshold)
    else:
        alarm_up = build_alarm_from_prob(p_up, t_idx, threshold=threshold)
        alarm_down = build_alarm_from_prob(p_down, t_idx, threshold=threshold)

    ev_up_raw = merge_alarm_events(alarm_up, ramp_dir=1, merge_gap=MERGE_GAP)
    ev_down_raw = merge_alarm_events(alarm_down, ramp_dir=-1, merge_gap=MERGE_GAP)

    ev_up = _shift_alert_to_window(ev_up_raw)
    ev_down = _shift_alert_to_window(ev_down_raw)
    pred_events = ev_up + ev_down
    return pred_events, ev_up, ev_down


def recompute_lead(match_detail, pred_events, true_events):
    """修复 B(a)：提前量 = 真实事件 start - 告警时刻 alert_t（正值=提前）。

    compute_event_metrics 内部用 pred.start 估算 lead_time，但平移后 start
    已不是告警时刻。这里用 pred_index 回取 alert_t 重算，并覆盖明细与汇总。
    返回 (avg_lead_time 或 None, 修正后的 match_detail)。
    """
# 真实事件起点按 event_id 建字典，避免 O(N^2) 反查
    start_by_id = {str(t["event_id"]): int(t["start_t_idx"]) for t in true_events}
    leads = []
    for m in match_detail:
        if not m["matched"]:
            m["lead_time"] = None
            continue
        pe = pred_events[m["pred_index"]]
        alert_t = int(pe["alert_t"])
        lead = start_by_id[str(m["event_id"])] - alert_t
        m["lead_time"] = lead
        leads.append(lead)
    avg = sum(leads) / len(leads) if leads else None
    return avg, match_detail


def eval_one_threshold(p_up, p_down, t_idx, true_events, threshold, use_hysteresis=False):
    pred_events, ev_up, ev_down = build_pred_events(p_up, p_down, t_idx, threshold, use_hysteresis)
    metrics = compute_event_metrics(true_events, pred_events)

    # 修复 B(a)：重算提前量（真实 start - 告警 alert_t），覆盖 metrics 内基于
    # 平移窗口 start 的旧值。分方向指标也需同步，故整体重算。
    avg_lead, fixed_detail = recompute_lead(
        metrics["match_detail"], pred_events, true_events
    )
    metrics["avg_lead_time"] = avg_lead
    metrics["match_detail"] = fixed_detail

    by_dir = compute_metrics_by_direction(true_events, pred_events)
    # 分方向提前量同样修正
    for d_val, md in by_dir.items():
        if md is None:
            continue
        sub_true = [e for e in true_events if int(e["ramp_dir"]) == d_val]
        sub_avg, sub_detail = recompute_lead(md["match_detail"], pred_events, sub_true)
        md["avg_lead_time"] = sub_avg
        md["match_detail"] = sub_detail

    return metrics, by_dir, pred_events, ev_up, ev_down


def fmt(v):
    return "None" if v is None else f"{v:.4f}"


def normalized_cost(metrics, ratio):
    """代价敏感归一化代价。

    TotalCost   = ratio * num_miss + C_false * num_false_alarm  (C_false=1)
    归一化代价  = TotalCost / num_true
    """
    num_true = metrics["num_true"]
    total = ratio * metrics["num_miss"] + COST_C_FALSE * metrics["num_false_alarm"]
    return total / num_true if num_true > 0 else 0.0


def scan_cost_grid(p_up, p_down, t_idx, true_events, use_hysteresis):
    """按细网格扫描全阈值，返回每档记录列表。"""
    grid = np.round(
        np.arange(COST_SCAN_START, COST_SCAN_END + 1e-9, COST_SCAN_STEP), 2
    )
    records = []
    for th in grid:
        th = float(th)
        metrics, _, _, _, _ = eval_one_threshold(
            p_up, p_down, t_idx, true_events, th, use_hysteresis=use_hysteresis
        )
        rec = {
            "threshold": th,
            "num_true": metrics["num_true"],
            "num_miss": metrics["num_miss"],
            "num_false_alarm": metrics["num_false_alarm"],
            "num_pred": metrics["num_pred"],
            "num_hit": metrics["num_hit"],
            "hit_rate": metrics["hit_rate"],
            "miss_rate": metrics["miss_rate"],
            "false_alarm_rate": metrics["false_alarm_rate"],
        }
        for r in COST_RATIOS:
            rec[f"cost_{r}"] = normalized_cost(metrics, r)
        records.append(rec)
    return records


def best_threshold(records, ratio):
    """指定代价比下归一化代价最小的记录（并列取阈值最小）。"""
    key = f"cost_{ratio}"
    return min(records, key=lambda r: (r[key], r["threshold"]))


def plot_cost_curves(scan_no, scan_hy, path):
    """绘制 6 条代价曲线（3 代价比 × 无/带迟滞）并标注各组最优 τ。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    colors = {1: "#1f77b4", 5: "#ff7f0e", 10: "#2ca02c"}

    for ax, ratio in zip(axes, COST_RATIOS):
        key = f"cost_{ratio}"
        xs_no = [r["threshold"] for r in scan_no]
        ys_no = [r[key] for r in scan_no]
        xs_hy = [r["threshold"] for r in scan_hy]
        ys_hy = [r[key] for r in scan_hy]

        ax.plot(xs_no, ys_no, "-o", ms=3, color=colors[ratio], label="无迟滞")
        ax.plot(xs_hy, ys_hy, "--s", ms=3, color=colors[ratio],
                alpha=0.7, label="带迟滞")

        b_no = best_threshold(scan_no, ratio)
        b_hy = best_threshold(scan_hy, ratio)
        ax.scatter([b_no["threshold"]], [b_no[key]], marker="*", s=180,
                   color="red", zorder=5,
                   label=f"最优(无) τ={b_no['threshold']:.2f}")
        ax.scatter([b_hy["threshold"]], [b_hy[key]], marker="*", s=180,
                   color="purple", zorder=5,
                   label=f"最优(带) τ={b_hy['threshold']:.2f}")
        ax.axvline(b_no["threshold"], color="red", ls=":", lw=0.8)
        ax.axvline(b_hy["threshold"], color="purple", ls=":", lw=0.8)

        ax.set_title(f"代价比 C_miss:C_false = {ratio}:1")
        ax.set_xlabel("阈值 τ")
        ax.set_ylabel("归一化代价 = TotalCost / num_true")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("代价敏感阈值标定：归一化代价随阈值变化（无迟滞 vs 带迟滞）",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def metrics_by_level(true_events, detail, num_false_alarm):
    """三级分级回传评价（口径：等级跟随真实事件幅度对齐）。

    等级来自真实事件自带的 level（1/2/3），预测侧只负责检出、不独立算等级。
    某条预测告警命中哪个真实事件，该命中即归入对应真实事件的 level。
    虚警为预测侧全局量，不摊到各等级，单独统计。

    返回:
      level_stats: {level: {num_true, num_hit, num_miss, hit_rate, avg_lead}}
      global_false_alarm: 全局虚警数
    """
    # 真实事件: event_id -> level
    lvl_by_id = {str(t["event_id"]): int(t.get("level", 0)) for t in true_events}

    stats = {}
    for lv in (1, 2, 3):
        n_true = sum(1 for t in true_events if int(t.get("level", 0)) == lv)
        stats[lv] = {"num_true": n_true, "num_hit": 0, "num_miss": 0,
                     "leads": [], "avg_lead": None, "hit_rate": 0.0}

    # 先行把 lead 用 recompute_lead 已修正过的 detail（metrics 里传入前已修好）
    for m in detail:
        lv = lvl_by_id.get(str(m["event_id"]), 0)
        if lv not in stats:
            stats[lv] = {"num_true": 0, "num_hit": 0, "num_miss": 0,
                         "leads": [], "avg_lead": None, "hit_rate": 0.0}
        if m["matched"]:
            stats[lv]["num_hit"] += 1
            if m["lead_time"] is not None:
                stats[lv]["leads"].append(m["lead_time"])
        else:
            stats[lv]["num_miss"] += 1

    for lv, s in stats.items():
        s["hit_rate"] = s["num_hit"] / s["num_true"] if s["num_true"] > 0 else 0.0
        s["avg_lead"] = (sum(s["leads"]) / len(s["leads"])) if s["leads"] else None

    return stats



def build_cross_section(results):
    """第8节：分类概率源 vs 回归残差概率源 完整对照。
    覆盖：不同迟滞(无/带) × 不同代价比(1/5/10) 的最优 τ 与指标差异。"""
    out = ["## 8. 双源对照实验（分类概率 cls vs 回归残差概率 reg，创新点2→3 打通）\n"]
    out.append("**目的**：验证创新点2 产出的回归残差概率（p_up/p_dn）接入同一决策层后，"
               "在命中/虚警/代价上相对原三分类概率源的差异。两套概率共用同一套事件口径、"
               "迟滞逻辑与代价模型，仅概率来源不同，构成严格同源对照。\n")
    out.append("**两套源**：`cls`=三分类 prob[:,2]/[:,0]；`reg`=reg_prob.npz 的 p_up/p_dn"
               "（残差经验分布 + 偏置校正）。t_test 逐元素对齐。\n")

    # 8.1 默认阈值详细（无/带迟滞 × 两源）
    out.append("### 8.1 默认阈值 τ=%.2f 双源指标对照\n" % DEFAULT_THRESHOLD)
    out.append("| 源 | 迟滞 | 命中率 | 漏警率 | 虚警率 | 命中 | 漏警 | 虚警 | 预测事件 | 平均提前量 |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for src_key, src_label in (("cls", "分类"), ("reg", "回归残差")):
        r = results[src_key]
        for hyst_tag, rows in (("无", r["scan_no"]), ("带", r["scan_hy"])):
            m = None
            for th, mm, bd, nu, nd in rows:
                if abs(th - DEFAULT_THRESHOLD) < 1e-9:
                    m = mm
                    break
            if m is None:
                continue
            out.append(
                f"| {src_label} | {hyst_tag} | {m["hit_rate"]:.4f} | {m["miss_rate"]:.4f} | "
                f"{m["false_alarm_rate"]:.4f} | {m["num_hit"]} | {m["num_miss"]} | "
                f"{m["num_false_alarm"]} | {m["num_pred"]} | {fmt(m["avg_lead_time"])} |"
            )
    out.append("")

    # 8.2 不同代价比下最优 τ 与代价 对照（无/带迟滞，两源）
    out.append("### 8.2 不同代价比 × 迟滞：最优 τ* 与归一化代价（双源对照）\n")
    out.append("| 代价比 | 迟滞 | 最优τ*(分类) | 代价(分类) | 最优τ*(回归) | 代价(回归) | 代价差(回归-分类) |")
    out.append("| --- | --- | --- | --- | --- | --- | --- |")
    for ratio in COST_RATIOS:
        key = f"cost_{ratio}"
        for htag, hkey in (("无", "no"), ("带", "hy")):
            bc = results["cls"]["best_summary"][(ratio, hkey)]
            br = results["reg"]["best_summary"][(ratio, hkey)]
            diff = br[key] - bc[key]
            out.append(
                f"| {ratio}:1 | {htag} | {bc["threshold"]:.2f} | {bc[key]:.4f} | "
                f"{br["threshold"]:.2f} | {br[key]:.4f} | {diff:+.4f} |"
            )
    out.append("")

    # 8.3 汇总：结论性对比
    out.append("### 8.3 对照结论\n")
    # 统计回归源在代价比上的整体优劣
    reg_better = 0
    tot = 0
    for ratio in COST_RATIOS:
        key = f"cost_{ratio}"
        for hkey in ("no", "hy"):
            bc = results["cls"]["best_summary"][(ratio, hkey)][key]
            br = results["reg"]["best_summary"][(ratio, hkey)][key]
            tot += 1
            if br < bc:
                reg_better += 1
    out.append(f"- 在 {tot} 个「代价比 × 迟滞」组合中，回归残差概率源的归一化代价低于分类源的有 **{reg_better}** 个。")
    out.append("- 说明：回归源直接给出超阈概率（P(ΔP>θ) 语义），与阈值 τ 的物理含义一致；"
               "分类源给出的是类别概率，需重新标定 τ。两者可并存，作为互补的触发依据。")
    out.append("- 该对照打通创新点2（概率化校准）与创新点3（代价敏感决策层）：概率质量提升"
               "可经由同一决策层直接转化为事件级命中/虚警收益。")
    out.append("")
    return out


def run_source_eval(source, true_events, t_min, t_max):
    """对单一概率源跑完整评价：
    - 无/带迟滞阈值扫描（THRESHOLDS 粗网格）
    - 代价敏感细网格扫描（COST_RATIOS）
    - τ=0.6 三级分级回传（全量口径）
    返回 dict 汇总，供 build_report 及双源对照使用。"""
    p_up, p_down, t_idx, src = load_pred_source(source, t_min=t_min, t_max=t_max)

    def scan(use_hyst):
        rows = []
        for th in THRESHOLDS:
            metrics, by_dir, pred_events, ev_up, ev_down = eval_one_threshold(
                p_up, p_down, t_idx, true_events, th, use_hysteresis=use_hyst
            )
            rows.append((th, metrics, by_dir, len(ev_up), len(ev_down)))
        return rows

    scan_no = scan(False)
    scan_hy = scan(True)

    cost_no = scan_cost_grid(p_up, p_down, t_idx, true_events, use_hysteresis=False)
    cost_hy = scan_cost_grid(p_up, p_down, t_idx, true_events, use_hysteresis=True)

    best_summary = {}
    for ratio in COST_RATIOS:
        for tag, rows in (("no", cost_no), ("hy", cost_hy)):
            best_summary[(ratio, tag)] = best_threshold(rows, ratio)

    # τ=0.6 + 迟滞 三级分级（全量口径）
    level_th = 0.6
    level_true_events, _ = load_true_events(
        EVENT_CSV, t_min=t_min, t_max=t_max, main_only=False
    )
    lvl_metrics, _, _, _, _ = eval_one_threshold(
        p_up, p_down, t_idx, level_true_events, level_th, use_hysteresis=True
    )
    level_stats = metrics_by_level(
        level_true_events, lvl_metrics["match_detail"], lvl_metrics["num_false_alarm"]
    )

    return {
        "source": src,
        "p_up": p_up, "p_down": p_down, "t_idx": t_idx,
        "scan_no": scan_no, "scan_hy": scan_hy,
        "cost_no": cost_no, "cost_hy": cost_hy,
        "best_summary": best_summary,
        "level_th": level_th,
        "level_true_events": level_true_events,
        "lvl_metrics": lvl_metrics,
        "level_stats": level_stats,
        "global_fa": lvl_metrics["num_false_alarm"],
    }



def main():
    log("概率数据源开关 PROB_SOURCE=" + str(PROB_SOURCE))

    # 用默认源先确定测试区间
    p_up0, p_down0, t_idx0, src0 = load_pred_source(PROB_SOURCE)
    t_min, t_max = int(t_idx0.min()), int(t_idx0.max())

    # 主口径真实事件（1–7 节用）
    true_events, true_df = load_true_events(EVENT_CSV, t_min=t_min, t_max=t_max)
    true_up = int((true_df["ramp_dir"] == 1).sum())
    true_down = int((true_df["ramp_dir"] == -1).sum())
    log(f"  其中 上行 {true_up} / 下行 {true_down}")

    # 两套源各跑一遍完整评价
    log("==== 跑 分类概率源(cls) ====")
    res_cls = run_source_eval("cls", true_events, t_min, t_max)
    log("==== 跑 回归残差概率源(reg) ====")
    res_reg = run_source_eval("reg", true_events, t_min, t_max)
    results = {"cls": res_cls, "reg": res_reg}
    main_res = results.get(PROB_SOURCE, res_cls)

    scan_no = main_res["scan_no"]
    scan_hy = main_res["scan_hy"]
    cost_no = main_res["cost_no"]
    cost_hy = main_res["cost_hy"]
    best_summary = main_res["best_summary"]
    level_stats = main_res["level_stats"]
    lvl_metrics = main_res["lvl_metrics"]
    level_th = main_res["level_th"]
    level_true_events = main_res["level_true_events"]
    global_fa = main_res["global_fa"]
    src_main = main_res["source"]

    for tag, rows in (("无迟滞", scan_no), ("带迟滞", scan_hy)):
        log(f"==== {tag} (主源={src_main}) ====")
        for th, m, bd, nu, nd in rows:
            log(f"  τ={th:.2f}: 命中率={m["hit_rate"]:.3f} "
                f"漏警率={m["miss_rate"]:.3f} 虚警率={m["false_alarm_rate"]:.3f} "
                f"预测事件={m["num_pred"]} 平均提前量={fmt(m["avg_lead_time"])}")

    def pick(rows, th):
        for row in rows:
            if abs(row[0] - th) < 1e-9:
                return row
        return rows[0]

    th0 = DEFAULT_THRESHOLD
    th_n, mn0, bdn, nu_n, nd_n = pick(scan_no, th0)
    _, mh0, bdh, nu_h, nd_h = pick(scan_hy, th0)

    src_desc = {"cls": "三分类概率(lgb_pred.npz)", "reg": "回归残差概率(reg_prob.npz)"}
    lines = ["# 事件级评价报告（双源：分类概率 vs 回归残差概率 + 迟滞/代价对比）\n"]
    lines.append(f"- **主报告使用概率源**：{src_desc.get(src_main, src_main)}（PROB_SOURCE={src_main}）")
    lines.append(f"- 真实事件（主口径 is_main==1，起点落在测试区间 [{t_min},{t_max}] 内）：{len(true_events)} 条（上行 {true_up} / 下行 {true_down}）")
    lines.append(f"- 告警合并间隔 MERGE_GAP={MERGE_GAP} 步；真实事件 end_tidx = start_tidx + n_trig - 1")
    lines.append(f"- 迟滞：连续 {HYST_ON_STEPS} 步 ≥τ 触发；告警态下概率 ≤ {HYST_OFF_RATIO}τ 撤销")
    lines.append("")

    lines.append("## 1. 阈值扫描 —— 无迟滞（简单阈值）\n")
    lines.append("| 阈值τ | 命中率 | 漏警率 | 虚警率 | 平均提前量(步) | 预测事件总数 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for th, m, bd, nu, nd in scan_no:
        lines.append(
            f"| {th:.2f} | {m["hit_rate"]:.4f} | {m["miss_rate"]:.4f} | "
            f"{m["false_alarm_rate"]:.4f} | {fmt(m["avg_lead_time"])} | {m["num_pred"]} |"
        )
    lines.append("")

    lines.append("## 2. 阈值扫描 —— 带迟滞\n")
    lines.append("| 阈值τ | 命中率 | 漏警率 | 虚警率 | 平均提前量(步) | 预测事件总数 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for th, m, bd, nu, nd in scan_hy:
        lines.append(
            f"| {th:.2f} | {m["hit_rate"]:.4f} | {m["miss_rate"]:.4f} | "
            f"{m["false_alarm_rate"]:.4f} | {fmt(m["avg_lead_time"])} | {m["num_pred"]} |"
        )
    lines.append("")

    lines.append("## 3. 迟滞抑制效果对比（同阈值并排）\n")
    lines.append("| 阈值τ | 虚警率(无迟滞) | 虚警率(带迟滞) | 虚警率下降 | 预测事件(无) | 预测事件(带) | 命中率(无) | 命中率(带) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for (th, mn, _, _, _), (_, mh, _, _, _) in zip(scan_no, scan_hy):
        drop = mn["false_alarm_rate"] - mh["false_alarm_rate"]
        lines.append(
            f"| {th:.2f} | {mn["false_alarm_rate"]:.4f} | {mh["false_alarm_rate"]:.4f} | "
            f"{drop:+.4f} | {mn["num_pred"]} | {mh["num_pred"]} | "
            f"{mn["hit_rate"]:.4f} | {mh["hit_rate"]:.4f} |"
        )
    lines.append("")

    lines.append(f"## 4. 默认阈值 τ={th0:.2f} 详细（无迟滞 vs 带迟滞）\n")
    lines.append("| 指标 | 无迟滞 | 带迟滞 |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| 预测告警事件数 | {mn0["num_pred"]}（上{nu_n}/下{nd_n}） | {mh0["num_pred"]}（上{nu_h}/下{nd_h}） |")
    lines.append(f"| 命中 | {mn0["num_hit"]} | {mh0["num_hit"]} |")
    lines.append(f"| 漏警 | {mn0["num_miss"]} | {mh0["num_miss"]} |")
    lines.append(f"| 虚警 | {mn0["num_false_alarm"]} | {mh0["num_false_alarm"]} |")
    lines.append(f"| 命中率 | {mn0["hit_rate"]:.4f} | {mh0["hit_rate"]:.4f} |")
    lines.append(f"| 漏警率 | {mn0["miss_rate"]:.4f} | {mh0["miss_rate"]:.4f} |")
    lines.append(f"| 虚警率 | {mn0["false_alarm_rate"]:.4f} | {mh0["false_alarm_rate"]:.4f} |")
    lines.append(f"| 平均提前量(步) | {fmt(mn0["avg_lead_time"])} | {fmt(mh0["avg_lead_time"])} |")
    lines.append("")

    lines.append("### 4.1 分方向指标（带迟滞，默认阈值）\n")
    lines.append("| 方向 | 真实数 | 预测事件数 | 命中 | 漏警 | 虚警 | 命中率 | 漏警率 | 虚警率 | 平均提前量 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for d_name, d_val in (("上行(+1)", 1), ("下行(-1)", -1)):
        md = bdh.get(d_val)
        if md is None:
            lines.append(f"| {d_name} | 0 | - | - | - | - | - | - | - | - |")
        else:
            lines.append(
                f"| {d_name} | {md["num_true"]} | {md["num_pred"]} | {md["num_hit"]} | "
                f"{md["num_miss"]} | {md["num_false_alarm"]} | {md["hit_rate"]:.4f} | "
                f"{md["miss_rate"]:.4f} | {md["false_alarm_rate"]:.4f} | {fmt(md["avg_lead_time"])} |"
            )
    lines.append("")

    lines.append("## 5. 说明与注意\n")
    lines.append(f"- 真实事件仅取起点落在测试区间 [{t_min}, {t_max}] 内（修复 A），避免训练/验证期事件误算漏警。")
    lines.append(f"- 测试期（Day209–245）主口径事件仅 {len(true_events)} 条，样本量受该时段爬坡稀疏影响，指标存在波动，论文需注明。")
    lines.append(f"- 告警时刻 t 的预测窗口按 label 定义平移为 [t+L, t+L+Δt]，L={L_STEPS}、Δt={DT_STEPS} 步（修复 B(a)）。")
    lines.append("- 提前量 = 真实事件起点 - 告警时刻，正值表示提前预警，负值表示晚于真实起点。")
    lines.append("- 概率来源由 PROB_SOURCE 开关决定；本报告 1–5 节为主源，第 8 节给出双源完整对照。")
    lines.append("- 虚警口径：一对一匹配下未被任何真实事件占用的预测告警事件。")
    lines.append("")

    lines.append("## 6. 代价敏感阈值标定（创新点3：迟滞 + 代价曲线）\n")
    lines.append("**代价模型**：`TotalCost = ratio × num_miss + num_false_alarm`（固定 `C_false_alarm=1`），"
                 "归一化代价 `= TotalCost / num_true`。代价比 `ratio ∈ {1, 5, 10}`。扫描网格 τ ∈ ["
                 f"{COST_SCAN_START:.2f}, {COST_SCAN_END:.2f}]，步长 {COST_SCAN_STEP:.2f}。\n")

    lines.append("### 6.1 各组最优阈值 τ* 汇总（代价最小）\n")
    lines.append("| 代价比 | 策略 | 最优 τ* | 归一化代价 | 漏警数 | 虚警数 | 命中率 | 虚警率 | 预测事件数 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for ratio in COST_RATIOS:
        key = f"cost_{ratio}"
        for tag, rows in (("无迟滞", cost_no), ("带迟滞", cost_hy)):
            b = best_summary[(ratio, "no" if tag == "无迟滞" else "hy")]
            lines.append(
                f"| {ratio}:1 | {tag} | {b["threshold"]:.2f} | {b[key]:.4f} | "
                f"{b["num_miss"]} | {b["num_false_alarm"]} | {b["hit_rate"]:.4f} | "
                f"{b["false_alarm_rate"]:.4f} | {b["num_pred"]} |"
            )
    lines.append("")

    lines.append("### 6.2 最优阈值处迟滞增益（同代价比，带迟滞 vs 无迟滞）\n")
    lines.append("| 代价比 | 归一化代价(无) | 归一化代价(带) | 代价下降 | 虚警数(无→带) | 漏警数(无→带) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for ratio in COST_RATIOS:
        key = f"cost_{ratio}"
        bn = best_summary[(ratio, "no")]
        bh = best_summary[(ratio, "hy")]
        drop = bn[key] - bh[key]
        lines.append(
            f"| {ratio}:1 | {bn[key]:.4f} | {bh[key]:.4f} | {drop:+.4f} | "
            f"{bn["num_false_alarm"]}→{bh["num_false_alarm"]} | "
            f"{bn["num_miss"]}→{bh["num_miss"]} |"
        )
    lines.append("")

    def _cost_detail_table(rows, title):
        out = [f"### {title}\n"]
        header = "| τ | 漏警 | 虚警 | 命中率 | 虚警率 | 预测事件 | " + \
                 " | ".join(f"代价({r}:1)" for r in COST_RATIOS) + " |"
        out.append(header)
        out.append("| " + " | ".join(["---"] * (6 + len(COST_RATIOS))) + " |")
        for r in rows:
            cells = [
                f"{r["threshold"]:.2f}", str(r["num_miss"]), str(r["num_false_alarm"]),
                f"{r["hit_rate"]:.4f}", f"{r["false_alarm_rate"]:.4f}", str(r["num_pred"]),
            ] + [f"{r[f"cost_{ratio}"]:.4f}" for ratio in COST_RATIOS]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
        return out

    lines.extend(_cost_detail_table(cost_no, "6.3 全网格明细 —— 无迟滞"))
    lines.extend(_cost_detail_table(cost_hy, "6.4 全网格明细 —— 带迟滞"))

    lines.append("### 6.5 代价曲线图\n")
    lines.append("![代价曲线](cost_curve.png)\n")
    lines.append("图注：三子图对应代价比 1:1 / 5:1 / 10:1；实线圆点无迟滞、虚线方块带迟滞；"
                 "红星/紫星标注各策略最优 τ*。\n")

    lines.append("## 7. 三级分级回传结果（τ=0.6 + 迟滞，创新点3 分级收尾）\n")
    lines.append("**⚠ 口径差异（重要）**：第 1–6 节用主口径 `is_main==1`；本节改用全量带等级事件"
                 "（含二/三级，`main_only=False`），事件集合不同，指标不可直接横比。\n")
    lines.append("**等级口径**：等级跟随真实事件幅度对齐；预测侧只负责检出，不独立算幅度等级。"
                 "虚警为全局量，不摊到各等级。\n")
    lines.append(f"本节全量口径真实事件：{len(level_true_events)} 条（含一/二/三级）。\n")
    lines.append("### 7.1 分等级命中统计\n")
    lines.append("| 等级 | 真实事件数 | 命中数 | 漏警数 | 命中率 | 命中事件平均提前量(步) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    lvl_name = {1: "一级(≥20%)", 2: "二级(≥15%)", 3: "三级(≥10%)"}
    for lv in (1, 2, 3):
        s = level_stats.get(lv, {})
        lines.append(
            f"| {lvl_name[lv]} | {s.get("num_true", 0)} | {s.get("num_hit", 0)} | "
            f"{s.get("num_miss", 0)} | {s.get("hit_rate", 0.0):.4f} | {fmt(s.get("avg_lead"))} |"
        )
    lines.append("")
    lines.append(f"**全局虚警数（不分配等级）**：{global_fa}\n")
    lines.append(f"操作点：τ={level_th:.2f} + 迟滞；该点全局命中 {lvl_metrics["num_hit"]} / "
                 f"漏警 {lvl_metrics["num_miss"]} / 虚警 {global_fa}。\n")

    # ===== 第8节：双源对照 =====
    lines.extend(build_cross_section(results))

    with open(REPORT_MD, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    log(f"报告已写入 {REPORT_MD}")

    plot_cost_curves(cost_no, cost_hy, COST_FIG_PATH)
    log(f"代价曲线已保存 {COST_FIG_PATH}")
    for ratio in COST_RATIOS:
        bn = best_summary[(ratio, "no")]
        bh = best_summary[(ratio, "hy")]
        log(f"  代价比 {ratio}:1 最优τ 无迟滞={bn["threshold"]:.2f} 带迟滞={bh["threshold"]:.2f}")


if __name__ == "__main__":
    main()
