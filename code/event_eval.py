# -*- coding: utf-8 -*-
"""
event_eval.py —— 风电爬坡【事件级评价】核心逻辑（纯函数，不读写任何文件）

设计原则（对应方案 §2 / §6 W2“评价脚本必须先于模型”）：
  1. 完全方向感知：真实事件与预测告警事件都自带 ramp_dir（1 上 / -1 下），
     只有“时间窗交叠 + 方向一致”才算命中。
  2. 口径由调用方决定：本模块【不做】主口径过滤。调用方应在外层先筛选好
     真实事件（例如仅取主口径 >=20% 的事件），再传入本模块。
  3. 一对一匹配：每个预测告警事件最多命中一个真实事件，避免同一告警被
     多条真实事件重复计数；多余的同向告警按虚警计。
  4. 决策层只认“步数”，不认“分钟”（对应方案 §8.6）。

对外主接口：
  build_alarm_from_prob      —— 概率 -> 逐时间步告警标记
  merge_alarm_events         —— 告警标记 -> 预测告警事件（同向间隔 <= gap 合并）
  match_events               —— 真实事件 x 预测告警事件 一对一匹配
  compute_event_metrics      —— 总入口，返回事件级指标字典
  compute_metrics_by_direction —— 分方向指标（可选）

输入数据结构约定（均为 list[dict]，调用方自行组织）：
  真实事件 true_event:
    {
      'event_id'  : 任意,            # 事件唯一标识（可为 str/int）
      'start_t_idx': int,            # 事件窗口起点（时间索引）
      'end_t_idx'  : int,            # 事件窗口终点（时间索引，闭区间）
      'ramp_dir'   : 1 | -1,         # 方向：1 上爬坡 / -1 下爬坡
      # 其他字段（magnitude 等）本模块不使用，原样透传不影响
    }
  预测告警事件 pred_event（由 merge_alarm_events 生成）:
    {
      'start'   : int,               # 告警事件起点
      'end'     : int,               # 告警事件终点
      'ramp_dir': 1 | -1,            # 告警方向
    }
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# 1. 决策层：概率 -> 逐时间步告警标记
# ---------------------------------------------------------------------------
def build_alarm_from_prob(
    pred_prob: Sequence[float],
    t_idx: Sequence[int],
    threshold: float = 0.5,
) -> Dict[int, int]:
    """
    把逐时间步“爬坡概率”按阈值判定为逐时间步“告警标记”。

    注意：本函数只做最简单的阈值判定。如果你的决策层还有迟滞、平滑、
    分段阈值等逻辑，请在外部自行生成 alarm 字典后直接调用
    merge_alarm_events，无需经过本函数。

    参数
    ----
    pred_prob : 每时间步爬坡概率, 长度 N
    t_idx     : 每时间步对应的时间索引, 长度 N
    threshold : 概率阈值，>= threshold 视为告警（0/1）

    返回
    ----
    alarm_dict : {t_idx: 0/1}
        用 dict 保存，便于后续按时间排序、合并。
    """
    if len(pred_prob) != len(t_idx):
        raise ValueError(
            f"pred_prob 与 t_idx 长度不一致: {len(pred_prob)} vs {len(t_idx)}"
        )

    alarm: Dict[int, int] = {}
    for p, t in zip(pred_prob, t_idx):
        alarm[int(t)] = 1 if float(p) >= threshold else 0
    return alarm


# ---------------------------------------------------------------------------
# 2. 事件合并：告警标记 -> 预测告警事件
# ---------------------------------------------------------------------------
def merge_alarm_events(
    alarm_dict: Dict[int, int],
    ramp_dir: int = 1,
    merge_gap: int = 6,
) -> List[Dict]:
    """
    将同一方向、连续（或间隔 <= merge_gap 步）的告警合并为“预测告警事件”。

    合并规则（对应方案 §2 事件合并 + §8.6 步数口径）：
      - 仅告警为 1 的时间点参与事件；
      - 同向告警之间时间间隔 <= merge_gap 步 -> 合并为同一事件；
      - 间隔 > merge_gap -> 结束当前事件，另起新事件。
      - 事件起止 = 首/末告警的时间索引。

    重要：本函数只处理“单一方向”的告警。若你区分上/下行，请对每个方向
    分别调用本函数并把返回列表拼接，各次调用 ramp_dir 填对应方向。

    参数
    ----
    alarm_dict : {t_idx: 0/1}
    ramp_dir   : 本批告警代表的方向（1 上 / -1 下）
    merge_gap  : 允许合并的最大间隔步数（方案默认 6 步 = 1h）

    返回
    ----
    pred_events : list[dict]，每条 {'start', 'end', 'ramp_dir'}
    """
    if ramp_dir not in (1, -1):
        raise ValueError(f"ramp_dir 必须为 1 或 -1，收到 {ramp_dir}")
    if merge_gap < 0:
        raise ValueError(f"merge_gap 必须 >= 0，收到 {merge_gap}")

    alarm_times = sorted(t for t, a in alarm_dict.items() if a == 1)
    pred_events: List[Dict] = []
    if not alarm_times:
        return pred_events

    cur_start = alarm_times[0]
    cur_end = alarm_times[0]

    for t in alarm_times[1:]:
        if t - cur_end <= merge_gap:
            cur_end = t
        else:
            pred_events.append(
                {"start": cur_start, "end": cur_end, "ramp_dir": ramp_dir}
            )
            cur_start = t
            cur_end = t

    pred_events.append(
        {"start": cur_start, "end": cur_end, "ramp_dir": ramp_dir}
    )
    return pred_events


# ---------------------------------------------------------------------------
# 3. 匹配：真实事件 x 预测告警事件（一对一）
# ---------------------------------------------------------------------------
def _windows_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """闭区间交叠判定：[a_start,a_end] 与 [b_start,b_end] 是否有交集。"""
    return not (a_end < b_start or b_end < a_start)


def match_events(
    true_events: Sequence[Dict],
    pred_events: Sequence[Dict],
) -> List[Dict]:
    """
    真实事件与预测告警事件做【一对一】匹配。

    命中条件（必须同时满足）：
      1. 时间窗口交叠（闭区间）；
      2. 方向一致（ramp_dir 相同）。

    匹配策略：按真实事件顺序处理；对每个真实事件，在“同向 + 交叠 + 尚未
    被占用”的预测事件中，取【起点最早】的一个进行匹配。被占用的预测事件
    不会再次命中其它真实事件。

    返回
    ----
    match_result : list[dict]，与 true_events 等长、顺序一致，每条含：
        {
          'event_id'   : 真实事件 id,
          'ramp_dir'   : 方向,
          'matched'    : 是否命中(bool),
          'pred_index' : 命中时对应 pred_events 下标，否则 None,
          'lead_time'  : 命中时 = 预测告警起点 - 真实事件起点；否则 None,
        }
    """
    used_pred: set = set()
    match_result: List[Dict] = []

    for ev in true_events:
        t_start = int(ev["start_t_idx"])
        t_end = int(ev["end_t_idx"])
        t_dir = int(ev["ramp_dir"])

        candidates = []
        for pi, pe in enumerate(pred_events):
            if pi in used_pred:
                continue
            if int(pe["ramp_dir"]) != t_dir:
                continue
            if _windows_overlap(t_start, t_end, int(pe["start"]), int(pe["end"])):
                candidates.append(pi)

        if candidates:
            chosen = min(candidates, key=lambda i: int(pred_events[i]["start"]))
            used_pred.add(chosen)
            pred_start = int(pred_events[chosen]["start"])
            match_result.append(
                {
                    "event_id": ev.get("event_id"),
                    "ramp_dir": t_dir,
                    "matched": True,
                    "pred_index": chosen,
                    "lead_time": pred_start - t_start,
                }
            )
        else:
            match_result.append(
                {
                    "event_id": ev.get("event_id"),
                    "ramp_dir": t_dir,
                    "matched": False,
                    "pred_index": None,
                    "lead_time": None,
                }
            )

    return match_result


# ---------------------------------------------------------------------------
# 4. 总入口：计算事件级指标
# ---------------------------------------------------------------------------
def compute_event_metrics(
    true_events: Sequence[Dict],
    pred_events: Sequence[Dict],
) -> Dict:
    """
    事件级评价总入口。

    指标定义（对应方案 §2 主指标）：
      命中率   hit_rate  = 命中事件数 / 真实事件总数
      漏警率   miss_rate = 漏警事件数 / 真实事件总数
      虚警率   fa_rate   = 虚警事件数 / 预测告警事件总数
      平均提前量 avg_lead_time = 命中事件 lead_time 的均值（无命中为 None）

    方向区分：匹配已强制方向一致，可直接对“上/下爬坡混合”的事件列表调用；
    若需分方向指标，用 compute_metrics_by_direction。
    """
    num_true = len(true_events)
    num_pred = len(pred_events)

    match_detail = match_events(true_events, pred_events)

    hit_items = [m for m in match_detail if m["matched"]]
    miss_items = [m for m in match_detail if not m["matched"]]
    num_hit = len(hit_items)
    num_miss = len(miss_items)

    used_pred = {m["pred_index"] for m in hit_items}
    false_alarm_indices = [pi for pi in range(num_pred) if pi not in used_pred]
    num_false_alarm = len(false_alarm_indices)

    hit_rate = num_hit / num_true if num_true > 0 else 0.0
    miss_rate = num_miss / num_true if num_true > 0 else 0.0
    fa_rate = num_false_alarm / num_pred if num_pred > 0 else 0.0

    avg_lead: Optional[float]
    if num_hit > 0:
        avg_lead = sum(m["lead_time"] for m in hit_items) / num_hit
    else:
        avg_lead = None

    return {
        "num_true": num_true,
        "num_pred": num_pred,
        "num_hit": num_hit,
        "num_miss": num_miss,
        "num_false_alarm": num_false_alarm,
        "hit_rate": hit_rate,
        "miss_rate": miss_rate,
        "false_alarm_rate": fa_rate,
        "avg_lead_time": avg_lead,
        "match_detail": match_detail,
        "false_alarm_indices": false_alarm_indices,
    }


# ---------------------------------------------------------------------------
# 5. 分方向指标（可选，供论文分方向表格）
# ---------------------------------------------------------------------------
def compute_metrics_by_direction(
    true_events: Sequence[Dict],
    pred_events: Sequence[Dict],
) -> Dict[int, Optional[Dict]]:
    """
    按方向分别计算事件级指标，返回 {1: 上行指标, -1: 下行指标}。
    某方向无真实事件时该方向指标为 None。
    """
    result: Dict[int, Optional[Dict]] = {}
    for d in (1, -1):
        t_sub = [e for e in true_events if int(e["ramp_dir"]) == d]
        p_sub = [e for e in pred_events if int(e["ramp_dir"]) == d]
        result[d] = compute_event_metrics(t_sub, p_sub) if t_sub else None
    return result


if __name__ == "__main__":
    _true = [
        {"event_id": "T1", "start_t_idx": 100, "end_t_idx": 110, "ramp_dir": 1},
        {"event_id": "T2", "start_t_idx": 200, "end_t_idx": 205, "ramp_dir": -1},
    ]
    _pred = [
        {"start": 98, "end": 112, "ramp_dir": 1},
        {"start": 300, "end": 305, "ramp_dir": 1},
    ]
    _m = compute_event_metrics(_true, _pred)
    print("[event_eval self-test]")
    for k in ("num_true", "num_pred", "num_hit", "num_miss",
              "num_false_alarm", "hit_rate", "miss_rate",
              "false_alarm_rate", "avg_lead_time"):
        print(f"  {k:18s} = {_m[k]}")