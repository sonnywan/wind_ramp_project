# -*- coding: utf-8 -*-
"""
test_event_eval.py —— event_eval.py 的纯内存自检脚本（不读写任何数据文件）

运行：python code/test_event_eval.py
全部通过打印 "ALL TESTS PASSED"；任一失败抛 AssertionError。
"""

from __future__ import annotations

from event_eval import (
    build_alarm_from_prob,
    merge_alarm_events,
    match_events,
    compute_event_metrics,
    compute_metrics_by_direction,
)

EPS = 1e-12


def approx(a, b, tol=EPS):
    return abs(a - b) <= tol


# 用例 1：概率 -> 告警阈值判定
def test_build_alarm():
    prob = [0.1, 0.5, 0.49, 0.51, 0.9]
    tidx = [0, 1, 2, 3, 4]
    alarm = build_alarm_from_prob(prob, tidx, threshold=0.5)
    assert alarm == {0: 0, 1: 1, 2: 0, 3: 1, 4: 1}, alarm
    print("  [1] build_alarm_from_prob OK")


# 用例 2：告警 -> 事件合并
def test_merge():
    alarm = {0: 1, 1: 1, 2: 1, 3: 0, 9: 1, 11: 1}
    events = merge_alarm_events(alarm, ramp_dir=1, merge_gap=6)
    assert len(events) == 2, events
    assert events[0] == {"start": 0, "end": 2, "ramp_dir": 1}, events[0]
    assert events[1] == {"start": 9, "end": 11, "ramp_dir": 1}, events[1]

    # 间隔恰好 6 应合并
    ev2 = merge_alarm_events({0: 1, 6: 1}, merge_gap=6)
    assert len(ev2) == 1 and ev2[0]["start"] == 0 and ev2[0]["end"] == 6, ev2

    # 间隔 7 应拆分
    ev3 = merge_alarm_events({0: 1, 7: 1}, merge_gap=6)
    assert len(ev3) == 2, ev3

    # 空输入 / 全 0
    assert merge_alarm_events({}, merge_gap=6) == []
    assert merge_alarm_events({0: 0, 1: 0}, merge_gap=6) == []
    print("  [2] merge_alarm_events OK")


# 用例 3：一对一匹配（命中 + 漏警 + 方向不符）
def test_match_basic():
    true_events = [
        {"event_id": "T1", "start_t_idx": 100, "end_t_idx": 110, "ramp_dir": 1},
        {"event_id": "T2", "start_t_idx": 200, "end_t_idx": 205, "ramp_dir": -1},
    ]
    pred_events = [
        {"start": 98, "end": 112, "ramp_dir": 1},   # 命中 T1
        {"start": 198, "end": 207, "ramp_dir": 1},  # 方向不符，不能命中 T2
    ]
    res = match_events(true_events, pred_events)
    assert res[0]["matched"] is True, res[0]
    assert res[0]["pred_index"] == 0, res[0]
    assert res[0]["lead_time"] == 98 - 100, res[0]
    assert res[1]["matched"] is False, res[1]
    assert res[1]["pred_index"] is None
    print("  [3] match_events (hit/miss/direction) OK")


# 用例 4：交叠端点（闭区间边界相交也算交叠）
def test_overlap_boundary():
    true_events = [
        {"event_id": "T1", "start_t_idx": 100, "end_t_idx": 105, "ramp_dir": 1},
    ]
    res = match_events(true_events, [{"start": 105, "end": 110, "ramp_dir": 1}])
    assert res[0]["matched"] is True, res[0]

    res2 = match_events(true_events, [{"start": 106, "end": 110, "ramp_dir": 1}])
    assert res2[0]["matched"] is False, res2[0]
    print("  [4] overlap boundary OK")


# 用例 5：一对一占用（一个预测告警不被两条真实事件重复命中）
def test_one_to_one():
    true_events = [
        {"event_id": "T1", "start_t_idx": 100, "end_t_idx": 105, "ramp_dir": 1},
        {"event_id": "T2", "start_t_idx": 103, "end_t_idx": 108, "ramp_dir": 1},
    ]
    pred_events = [{"start": 101, "end": 104, "ramp_dir": 1}]
    res = match_events(true_events, pred_events)
    matched_flags = [r["matched"] for r in res]
    assert matched_flags.count(True) == 1, matched_flags
    assert matched_flags.count(False) == 1, matched_flags
    print("  [5] one-to-one occupancy OK")


# 用例 6：指标计算
def test_metrics():
    true_events = [
        {"event_id": "T1", "start_t_idx": 100, "end_t_idx": 110, "ramp_dir": 1},
        {"event_id": "T2", "start_t_idx": 200, "end_t_idx": 205, "ramp_dir": -1},
        {"event_id": "T3", "start_t_idx": 300, "end_t_idx": 310, "ramp_dir": 1},
    ]
    pred_events = [
        {"start": 90, "end": 99, "ramp_dir": 1},     # 与 T1 不相交 -> 虚警
        {"start": 195, "end": 200, "ramp_dir": -1},  # 命中 T2（200>=200 端点相交），提前量 -5
        {"start": 305, "end": 315, "ramp_dir": 1},   # 命中 T3，提前量 +5
    ]
    m = compute_event_metrics(true_events, pred_events)
    assert m["num_true"] == 3
    assert m["num_pred"] == 3
    assert m["num_hit"] == 2
    assert m["num_miss"] == 1
    assert m["num_false_alarm"] == 1
    assert approx(m["hit_rate"], 2 / 3), m["hit_rate"]
    assert approx(m["miss_rate"], 1 / 3), m["miss_rate"]
    assert approx(m["false_alarm_rate"], 1 / 3), m["false_alarm_rate"]
    assert approx(m["avg_lead_time"], 0.0), m["avg_lead_time"]
    assert m["false_alarm_indices"] == [0], m["false_alarm_indices"]
    print("  [6] compute_event_metrics OK")


# 用例 7：全漏警 / 空预测 / 空真实（防除零）
def test_edge_empty():
    true_events = [
        {"event_id": "T1", "start_t_idx": 10, "end_t_idx": 15, "ramp_dir": 1},
    ]
    m = compute_event_metrics(true_events, [])
    assert m["num_hit"] == 0 and m["num_miss"] == 1
    assert approx(m["miss_rate"], 1.0)
    assert approx(m["false_alarm_rate"], 0.0)
    assert m["avg_lead_time"] is None

    m2 = compute_event_metrics([], [{"start": 0, "end": 5, "ramp_dir": 1}])
    assert m2["num_true"] == 0 and m2["num_false_alarm"] == 1
    assert approx(m2["hit_rate"], 0.0)
    print("  [7] edge cases (empty / all miss) OK")


# 用例 8：分方向指标
def test_by_direction():
    true_events = [
        {"event_id": "U1", "start_t_idx": 100, "end_t_idx": 105, "ramp_dir": 1},
        {"event_id": "D1", "start_t_idx": 200, "end_t_idx": 205, "ramp_dir": -1},
    ]
    pred_events = [
        {"start": 98, "end": 102, "ramp_dir": 1},  # 命中上行
    ]
    res = compute_metrics_by_direction(true_events, pred_events)
    assert res[1] is not None and res[1]["num_hit"] == 1, res[1]
    assert res[-1] is not None and res[-1]["num_miss"] == 1, res[-1]
    print("  [8] compute_metrics_by_direction OK")


def main():
    print("Running event_eval self-tests ...")
    test_build_alarm()
    test_merge()
    test_match_basic()
    test_overlap_boundary()
    test_one_to_one()
    test_metrics()
    test_edge_empty()
    test_by_direction()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()