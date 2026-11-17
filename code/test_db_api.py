# -*- coding: utf-8 -*-
"""
test_db_api.py —— 数据库操作 + 后端接口 冒烟测试
===================================================
对应任务书（5）：数据库操作测试、接口通信测试。
无需外部服务：数据库测试直连 SQLite；接口测试直接调用处理函数（不起端口）。

用法:
  python code/test_db_api.py
"""
import os
import sys
import json
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database as db
import api_server as api

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name} {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def test_database():
    print("== 1. 数据库操作测试 ==")
    check("数据库文件存在", os.path.exists(db.DB_PATH))
    conn = sqlite3.connect(db.DB_PATH)
    cur = conn.cursor()
    # 表齐备
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = {r[0] for r in cur.fetchall()}
    for t in ("field_power", "ramp_events", "pred_prob", "alarm_log", "run_meta"):
        check(f"表存在: {t}", t in tables)

    # 行数
    for t, minimum in (("field_power", 30000), ("ramp_events", 1000),
                       ("pred_prob", 5000), ("alarm_log", 1)):
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        n = cur.fetchone()[0]
        check(f"{t} 行数>{minimum}", n > minimum, f"(={n})")

    # 主键完整性（插入重复应失败）
    try:
        cur.execute("INSERT INTO field_power(t_idx,day,step) VALUES (0,1,0)")
        conn.rollback()
        check("field_power 主键唯一约束", False)
    except sqlite3.IntegrityError:
        conn.rollback()
        check("field_power 主键唯一约束", True)

    # 条件查询：alarm_log 按命中过滤
    cur.execute("SELECT COUNT(*) FROM alarm_log WHERE is_hit=1")
    n_hit = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM alarm_log")
    n_all = cur.fetchone()[0]
    check("alarm_log 命中记录可查", 0 <= n_hit <= n_all, f"({n_hit}/{n_all})")

    # 事件主口径查询
    cur.execute("SELECT COUNT(*) FROM ramp_events WHERE is_main=1")
    n_main = cur.fetchone()[0]
    check("ramp_events 主口径可查", n_main > 0, f"(={n_main})")
    conn.close()


def test_api_queries():
    print("== 2. 后端接口（查询）测试 ==")
    # meta
    meta = api.q_meta()
    check("/meta 返回阈值", "tau" in meta, f"tau={meta.get('tau')}")
    check("/meta 概率源", meta.get("prob_source") in ("cls", "reg"))

    # timeseries
    ts = api.q_timeseries(30000, 30010)
    check("/timeseries 区间查询", len(ts) == 11, f"(={len(ts)})")
    check("/timeseries 字段完整", all(k in ts[0] for k in ("t_idx", "p_mw", "wspd_mean")))

    # events
    ev_all = api.q_events(main_only=True)
    ev_up = api.q_events(direction="up", main_only=True)
    check("/events 主口径", len(ev_all) > 0, f"(={len(ev_all)})")
    check("/events 方向过滤", all(e["direction"] == "up" for e in ev_up))

    # prob
    pr = api.q_prob(30000, 30002)
    check("/prob 区间查询", len(pr) == 3, f"(={len(pr)})")
    check("/prob 概率范围[0,1]", all(0 <= x["p_up"] <= 1 for x in pr))

    # alarms
    al = api.q_alarms(limit=5)
    check("/alarms 限量", len(al) <= 5)
    al_hit = api.q_alarms(is_hit=1)
    check("/alarms 命中过滤", all(a["is_hit"] == 1 for a in al_hit))

    # summary
    s = api.q_alarm_summary()
    check("/alarm_summary 字段", all(k in s for k in ("total_alarms", "hit_rate", "by_direction")))


def test_api_predict():
    print("== 3. 后端接口（模型调用）测试 ==")
    obj, code = api.do_predict({"t_idx": 30000})
    check("/predict 返回200", code == 200)
    check("/predict 含概率", "p_up" in obj and "p_dn" in obj)
    # 与数据库中 pred_prob 同点一致性
    pr = api.q_prob(30000, 30000)
    if pr:
        diff = abs(obj["p_up"][0] - round(pr[0]["p_up"], 4))
        check("/predict 与 pred_prob 一致", diff <= 5e-4, f"(diff={diff:.2e})")
    # 越界 t_idx
    obj2, code2 = api.do_predict({"t_idx": 1})
    check("/predict 越界返回400", code2 == 400)


def test_llm_explain():
    print("== 4. 大模型预警解释模块测试 ==")
    import llm_explain
    fact = llm_explain.build_warning_fact(
        t_start=30200, direction="up", level=1, p_up=0.72, p_dn=0.03,
        lead_steps=6, met_term=57.3, work_term=-14.9, wspd_mean=9.8, delta_p_mw=42.4,
    )
    out = llm_explain.explain(fact, use_live=False)
    check("解释含 summary", "summary" in out and len(out["summary"]) > 10)
    check("解释含 action", "action" in out and len(out["action"]) > 0)
    check("成因判定为气象型", fact["cause"] == "气象型")
    check("来源为 template", out.get("source") == "template")
    r = api.q_alarm_explain(1)
    check("/alarm_explain 返回解释", "summary" in r and "fact" in r)
    out2 = llm_explain.explain(fact, use_live=True)
    check("无配置自动回退模板", out2.get("source") == "template")


def test_predict_manual():
    print("== 5. 传感器输入在线判断（闭环）测试 ==")
    # 低风速：不预警
    r, code = api.do_predict_manual({"wspd": 6.0, "wind_thr": 12.0, "led": False, "save": False})
    check("低风速不预警", code == 200 and r["decision"]["is_alarm"] is False)
    check("低风速LED灭", r["led"]["on"] is False)
    # 高风速：风速硬阈值触发预警，且功率随曲线派生
    r2, _ = api.do_predict_manual({"wspd": 13.0, "wind_thr": 12.0, "led": False, "save": False})
    check("高风速触发预警", r2["decision"]["is_alarm"] is True)
    check("触发源=风速", r2["decision"]["by"] == "wind")
    check("高风速功率更高", r2["input"]["p_mw"] > r["input"]["p_mw"])
    # 风功率曲线：风速升高，派生功率升高
    low, _ = api.do_predict_manual({"wspd": 5.0, "led": False, "save": False})
    high, _ = api.do_predict_manual({"wspd": 10.0, "led": False, "save": False})
    check("风功率曲线单调(5<10m/s)", high["input"]["p_mw"] > low["input"]["p_mw"])
    # 模型路径：低 tau 下高上行概率触发
    r3, _ = api.do_predict_manual({"wspd": 11.5, "p_mw": 50, "wind_thr": 99, "tau": 0.2, "led": False, "save": False})
    check("模型路径可触发(低tau)", r3["decision"]["is_alarm"] is True and r3["decision"]["by"] == "model")
    # LED 开关一致性
    r4, _ = api.do_predict_manual({"wspd": 13.0, "wind_thr": 12.0, "led": True, "save": False})
    check("LED 与预警一致", r4["led"]["on"] == r4["decision"]["is_alarm"])


def main():
    print("=" * 55)
    print("数据库 + 后端接口 冒烟测试")
    print("=" * 55)
    test_database()
    test_api_queries()
    test_api_predict()
    test_llm_explain()
    test_predict_manual()
    print("-" * 55)
    print(f"结果: PASS={PASS}  FAIL={FAIL}")
    print("全部通过 OK" if FAIL == 0 else "存在失败 FAILED")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()