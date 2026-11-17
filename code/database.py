# -*- coding: utf-8 -*-
"""
database.py —— 风电爬坡预警系统【数据库层】（SQLite）
=====================================================================
任务书（4）要求：设计数据库存储风电功率、气象数据、预测结果及预警记录。
本模块用 SQLite 落地四张核心表 + 一张汇总视图，供后端接口 / 历史查询 / 测试使用。

设计原则：
  1. 与现有产物解耦：只从 output/ 下已生成的 CSV/NPZ 读入，不改动算法链路。
  2. 幂等：重复运行 = 重建表（DROP + CREATE + 灌数），便于回归。
  3. 表结构即论文"数据库设计"一节的依据（字段/类型/主键/索引见下方 DDL）。

表结构（5 张）：
  field_power  —— 场站级功率+气象时序（源自 field_power.csv）
  ramp_events  —— 真实爬坡事件清单（源自 ramp_events.csv，含成因分解）
  pred_prob    —— 逐时间步预测概率（源自 reg_prob.npz / lgb_pred.npz）
  alarm_log    —— 预警记录（由决策层生成，落库可回溯；每方向每事件一行）
  run_meta     —— 元信息（灌库时间、阈值、概率源、口径）

用法：
  python code/database.py            # 重建数据库 output/wind_ramp.db
  python code/database.py --check    # 只打印各表行数与样例
"""
import os
import sqlite3
import argparse
import datetime as dt

import numpy as np
import pandas as pd

OUT_DIR = "output"
DB_PATH = os.path.join(OUT_DIR, "wind_ramp.db")

FIELD_CSV = os.path.join(OUT_DIR, "field_power.csv")
EVENT_CSV = os.path.join(OUT_DIR, "ramp_events.csv")
REG_NPZ = os.path.join(OUT_DIR, "reg_prob.npz")
CLS_NPZ = os.path.join(OUT_DIR, "lgb_pred.npz")

STEPS_PER_DAY = 144
L_STEPS = 6            # 预警提前量 L = 1h
DT_STEPS = 6           # 事件窗 Δt = 1h
MERGE_GAP = 6          # 同向告警合并间隔
DEFAULT_TAU = 0.60     # 默认决策阈值
DEFAULT_SOURCE = "reg"
HYST_ON_STEPS = 2
HYST_OFF_RATIO = 0.8
THETA_MW = 40.2        # 20% × 201MW


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
DDL = """
PRAGMA journal_mode = WAL;

DROP TABLE IF EXISTS field_power;
CREATE TABLE field_power (
    t_idx       INTEGER PRIMARY KEY,   -- 全局时间步 (Day-1)*144+step
    day         INTEGER NOT NULL,
    step        INTEGER NOT NULL,
    p_mw        REAL,                  -- 场站总有功 (MW)
    cap_mw      REAL,                  -- 可用装机 (MW)
    wspd_mean   REAL,                  -- 平均风速 (m/s)
    ramp_rate   REAL,                  -- 当前爬坡率 R (MW/h)
    is_gap      INTEGER DEFAULT 0,     -- 数据缺口标记 0/1
    split       TEXT                   -- train/val/test
);
CREATE INDEX idx_field_day ON field_power(day);
CREATE INDEX idx_field_split ON field_power(split);

DROP TABLE IF EXISTS ramp_events;
CREATE TABLE ramp_events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    direction   TEXT NOT NULL,         -- up/down
    start_tidx  INTEGER NOT NULL,      -- 事件起点时间步
    start_day   INTEGER,
    start_step  INTEGER,
    end_day     INTEGER,
    p_start     REAL,                  -- 起始功率 (MW)
    p_end       REAL,                  -- 结束功率 (MW)
    delta_p     REAL,                  -- 功率变化 (MW)
    ramp_rate   REAL,                  -- 爬坡率 (MW/h)
    max_abs_r   REAL,
    level       INTEGER,               -- 1/2/3 幅度等级
    is_main     INTEGER DEFAULT 0,     -- 是否主口径 (≥20%)
    work_term   REAL,                  -- 成因分解: 可用装机项
    met_term    REAL,                  -- 成因分解: 气象项
    resid       REAL,                  -- 分解残差
    cls         TEXT,                  -- meteor/work 成因分类
    n_trig      INTEGER                -- 触发点数
);
CREATE INDEX idx_ev_start ON ramp_events(start_tidx);
CREATE INDEX idx_ev_dir ON ramp_events(direction);
CREATE INDEX idx_ev_main ON ramp_events(is_main);

DROP TABLE IF EXISTS pred_prob;
CREATE TABLE pred_prob (
    t_idx   INTEGER PRIMARY KEY,
    p_up    REAL,                      -- 上行超阈概率
    p_dn    REAL,                      -- 下行超阈概率
    source  TEXT                       -- cls/reg 概率源
);

DROP TABLE IF EXISTS alarm_log;
CREATE TABLE alarm_log (
    alarm_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    t_start     INTEGER NOT NULL,      -- 告警起点时间步
    t_end       INTEGER NOT NULL,      -- 告警终点时间步
    direction   TEXT NOT NULL,         -- up/down
    level       INTEGER,               -- 告警等级(按命中真实事件回填, 未命中=0)
    tau         REAL,                  -- 决策阈值
    source      TEXT,                  -- 概率源
    matched_event_id INTEGER,          -- 命中真实事件 id (可空)
    is_hit      INTEGER DEFAULT 0,     -- 是否命中真实事件
    lead_steps  INTEGER,               -- 提前量(步, 命中时回填)
    created_at  TEXT                   -- 落库时间
);
CREATE INDEX idx_alarm_t ON alarm_log(t_start);
CREATE INDEX idx_alarm_dir ON alarm_log(direction);

DROP TABLE IF EXISTS run_meta;
CREATE TABLE run_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def log(m):
    print(f"[database] {m}")


# ---------------------------------------------------------------------------
# 决策层（与 run_event_eval 同口径，独立实现以便落库）
# ---------------------------------------------------------------------------
def hysteresis_alarm(p, tau, on_steps=HYST_ON_STEPS, off_ratio=HYST_OFF_RATIO):
    n = len(p)
    alarm = np.zeros(n, dtype=int)
    state = 0
    consec = 0
    off_thr = off_ratio * tau
    for i in range(n):
        v = float(p[i])
        if state == 0:
            consec = consec + 1 if v >= tau else 0
            if consec >= on_steps:
                state = 1
                alarm[i] = 1
        else:
            if v <= off_thr:
                state = 0
                consec = 0
                alarm[i] = 0
            else:
                alarm[i] = 1
    return alarm


def merge_alarm(t_idx, alarm, direction, merge_gap=MERGE_GAP):
    """逐点告警 -> 合并事件列表 [(start,end,dir), ...]"""
    times = [int(t) for t, a in zip(t_idx, alarm) if a == 1]
    if not times:
        return []
    events = []
    cs = ce = times[0]
    for t in times[1:]:
        if t - ce <= merge_gap:
            ce = t
        else:
            events.append((cs, ce, direction))
            cs = ce = t
    events.append((cs, ce, direction))
    return events


def match_and_backfill(pred_events, true_events):
    """一对一匹配，回填 is_hit / matched_event_id / lead_steps / level。
    与 event_eval 口径一致：交叠 + 同向 + 起点最早者优先。"""
    used = set()
    out = []
    for (ps, pe, pdir) in pred_events:
        hit = None
        for i, ev in enumerate(true_events):
            if i in used:
                continue
            if ev["direction"] != pdir:
                continue
            if not (pe < ev["s"] or ev["e"] < ps):  # 交叠
                if hit is None:
                    hit = (i, ev)
        if hit is not None:
            i, ev = hit
            used.add(i)
            out.append({
                "t_start": ps, "t_end": pe, "direction": pdir,
                "level": int(ev.get("level", 0)),
                "matched_event_id": int(ev["event_id"]),
                "is_hit": 1,
                "lead_steps": int(ev["s"] - ps),
            })
        else:
            out.append({
                "t_start": ps, "t_end": pe, "direction": pdir,
                "level": 0, "matched_event_id": None,
                "is_hit": 0, "lead_steps": None,
            })
    return out


# ---------------------------------------------------------------------------
# 灌数
# ---------------------------------------------------------------------------
def load_field(conn):
    df = pd.read_csv(FIELD_CSV)
    cols = ["t_idx", "Day", "step", "P_mw", "C_mw", "Wspd_mean", "R", "is_gap", "split"]
    sub = df[cols].copy()
    sub.columns = ["t_idx", "day", "step", "p_mw", "cap_mw", "wspd_mean", "ramp_rate", "is_gap", "split"]
    sub.to_sql("field_power", conn, if_exists="append", index=False)
    log(f"field_power 灌入 {len(sub)} 行")
    return df


def load_events(conn):
    df = pd.read_csv(EVENT_CSV)
    sub = df.rename(columns={
        "direction": "direction", "start_tidx": "start_tidx",
        "start_day": "start_day", "start_step": "start_step", "end_day": "end_day",
        "P_start": "p_start", "P_end": "p_end", "delta_P": "delta_p",
        "R": "ramp_rate", "max_absR": "max_abs_r", "level": "level",
        "is_main": "is_main", "work_term": "work_term", "met_term": "met_term",
        "resid": "resid", "cls": "cls", "n_trig": "n_trig",
    })[["direction", "start_tidx", "start_day", "start_step", "end_day",
        "p_start", "p_end", "delta_p", "ramp_rate", "max_abs_r", "level",
        "is_main", "work_term", "met_term", "resid", "cls", "n_trig"]]
    sub.to_sql("ramp_events", conn, if_exists="append", index=False)
    log(f"ramp_events 灌入 {len(sub)} 行")
    return df


def load_pred_prob(conn, source=DEFAULT_SOURCE):
    if source == "reg":
        d = np.load(REG_NPZ)
        p_up = d["p_up_test"].astype(float)
        p_dn = d["p_dn_test"].astype(float)
        t_idx = d["t_test"].astype(int)
    else:
        d = np.load(CLS_NPZ, allow_pickle=True)
        prob = d["prob_test"]
        p_up = prob[:, 2].astype(float)
        p_dn = prob[:, 0].astype(float)
        t_idx = d["t_test"].astype(int)
    order = np.argsort(t_idx)
    df = pd.DataFrame({
        "t_idx": t_idx[order],
        "p_up": p_up[order],
        "p_dn": p_dn[order],
        "source": source,
    })
    df.to_sql("pred_prob", conn, if_exists="append", index=False)
    log(f"pred_prob 灌入 {len(df)} 行 (source={source})")
    return df


def build_alarm_log(conn, prob_df, events_df, source, tau):
    """由概率 + 决策层生成预警记录并落库（含命中回填）。"""
    t_idx = prob_df["t_idx"].to_numpy()
    # 决策层：双通道迟滞
    au = hysteresis_alarm(prob_df["p_up"].to_numpy(), tau)
    ad = hysteresis_alarm(prob_df["p_dn"].to_numpy(), tau)
    ev_up = merge_alarm(t_idx, au, "up")
    ev_dn = merge_alarm(t_idx, ad, "down")
    pred_events = ev_up + ev_dn

    # 真实事件（主口径），用于回填命中
    true_events = []
    tmin, tmax = int(t_idx.min()), int(t_idx.max())
    ed = events_df
    ed = ed[ed["is_main"] == 1]
    ed = ed[(ed["start_tidx"] >= tmin) & (ed["start_tidx"] <= tmax)]
    for k, (_, r) in enumerate(ed.iterrows()):
        true_events.append({
            "event_id": k,
            "s": int(r["start_tidx"]),
            "e": int(r["start_tidx"] + int(r["n_trig"]) - 1),
            "direction": r["direction"],
            "level": int(r["level"]) if pd.notna(r.get("level", 0)) else 0,
        })

    backfilled = match_and_backfill(pred_events, true_events)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [{
        "t_start": b["t_start"], "t_end": b["t_end"], "direction": b["direction"],
        "level": b["level"], "tau": tau, "source": source,
        "matched_event_id": b["matched_event_id"], "is_hit": b["is_hit"],
        "lead_steps": b["lead_steps"], "created_at": now,
    } for b in backfilled]
    pd.DataFrame(rows).to_sql("alarm_log", conn, if_exists="append", index=False)
    log(f"alarm_log 生成 {len(rows)} 条（命中 {sum(r['is_hit'] for r in rows)}）")
    return rows


def write_meta(conn, source, tau):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        "built_at": now,
        "prob_source": source,
        "tau": str(tau),
        "L_steps": str(L_STEPS),
        "dt_steps": str(DT_STEPS),
        "merge_gap": str(MERGE_GAP),
        "theta_mw": str(THETA_MW),
        "steps_per_day": str(STEPS_PER_DAY),
        "schema_version": "1.0",
    }
    pd.DataFrame(list(meta.items()), columns=["key", "value"]).to_sql(
        "run_meta", conn, if_exists="append", index=False
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_db(db_path=DB_PATH, source=DEFAULT_SOURCE, tau=DEFAULT_TAU):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(DDL)
        events_df = load_events(conn)
        load_field(conn)
        prob_df = load_pred_prob(conn, source=source)
        build_alarm_log(conn, prob_df, events_df, source, tau)
        write_meta(conn, source, tau)
        conn.commit()
    finally:
        conn.close()
    log(f"数据库已生成: {db_path}")


def check_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for tbl in ("field_power", "ramp_events", "pred_prob", "alarm_log", "run_meta"):
            cur.execute(f"SELECT COUNT(*) FROM {tbl}")
            n = cur.fetchone()[0]
            cur.execute(f"SELECT * FROM {tbl} LIMIT 2")
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            print(f"\n== {tbl} ({n} 行) ==")
            print("  columns:", cols)
            for r in rows:
                print("  ", r)
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查不重建")
    ap.add_argument("--source", default=DEFAULT_SOURCE, choices=["cls", "reg"])
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()
    if args.check:
        check_db(args.db)
    else:
        build_db(args.db, source=args.source, tau=args.tau)
        check_db(args.db)


if __name__ == "__main__":
    main()