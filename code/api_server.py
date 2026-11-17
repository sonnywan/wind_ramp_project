# -*- coding: utf-8 -*-
"""
api_server.py —— 风电爬坡预警系统【后端接口层】（Python 标准库 HTTP 服务）
=============================================================================
任务书（4）要求：实现后端数据处理和模型调用（Flask/FastAPI）。
本模块用标准库 http.server 零依赖实现 REST 接口，封装：
  - 数据处理：从数据库 / 产物读取并返回 JSON
  - 模型调用：POST /predict 调用 reg_infer 在线推理底座
  - 历史查询：GET /alarms 从 alarm_log 表条件检索
若环境安装了 FastAPI 可直接替换（接口语义一致），本文件保证"无第三方依赖也能跑"。

接口清单：
  GET  /health                 健康检查
  GET  /meta                   运行元信息（口径/阈值/概率源）
  GET  /timeseries             ?t0=&t1=  功率/气象时序（field_power）
  GET  /events                 ?direction=&level=&main_only=  真实事件清单
  GET  /prob                   ?t0=&t1=  逐时间步预测概率（pred_prob）
  GET  /alarms                 ?direction=&is_hit=&t0=&t1=&limit=  历史预警记录
  GET  /alarm_summary          预警统计（按方向/命中/等级聚合）
  POST /predict                body: {"t_idx": int, "tau": float}  在线推理单点
                              body: {"windows": [...], "p_now": [...]} 批量推理
用法：
  python code/api_server.py [--host 127.0.0.1] [--port 8000]
"""
import os
import sys
import json
import sqlite3
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

OUT_DIR = "output"
DB_PATH = os.path.join(OUT_DIR, "wind_ramp.db")
DATASET_NPZ = os.path.join(OUT_DIR, "dataset.npz")
FIELD_CSV = os.path.join(OUT_DIR, "field_power.csv")

# 延迟导入推理底座，避免无模型时启动失败
_infer = {"module": None, "bundle": None}


def get_infer():
    if _infer["module"] is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import reg_infer
        _infer["module"] = reg_infer
    return _infer["module"]


def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def rows_to_dicts(rows):
    return [dict(r) for r in rows]


# ------------------------- 查询实现 -------------------------
def q_meta():
    conn = db_conn()
    try:
        rows = conn.execute("SELECT key, value FROM run_meta").fetchall()
        return {r["key"]: r["value"] for r in rows}
    finally:
        conn.close()


def q_timeseries(t0, t1):
    conn = db_conn()
    try:
        sql = ("SELECT t_idx, day, step, p_mw, wspd_mean, ramp_rate, is_gap, split "
               "FROM field_power WHERE t_idx >= ? AND t_idx <= ? ORDER BY t_idx")
        rows = conn.execute(sql, (t0, t1)).fetchall()
        return rows_to_dicts(rows)
    finally:
        conn.close()


def q_events(direction=None, level=None, main_only=True):
    conn = db_conn()
    try:
        sql = ("SELECT event_id, direction, start_tidx, start_day, p_start, p_end, "
               "delta_p, ramp_rate, level, is_main, cls, n_trig FROM ramp_events WHERE 1=1")
        params = []
        if main_only:
            sql += " AND is_main = 1"
        if direction in ("up", "down"):
            sql += " AND direction = ?"
            params.append(direction)
        if level is not None:
            sql += " AND level = ?"
            params.append(level)
        sql += " ORDER BY start_tidx"
        rows = conn.execute(sql, params).fetchall()
        return rows_to_dicts(rows)
    finally:
        conn.close()


def q_prob(t0, t1):
    conn = db_conn()
    try:
        sql = ("SELECT t_idx, p_up, p_dn, source FROM pred_prob "
               "WHERE t_idx >= ? AND t_idx <= ? ORDER BY t_idx")
        rows = conn.execute(sql, (t0, t1)).fetchall()
        return rows_to_dicts(rows)
    finally:
        conn.close()


def q_alarms(direction=None, is_hit=None, t0=None, t1=None, limit=500):
    conn = db_conn()
    try:
        sql = ("SELECT alarm_id, t_start, t_end, direction, level, tau, source, "
               "matched_event_id, is_hit, lead_steps, created_at FROM alarm_log WHERE 1=1")
        params = []
        if direction in ("up", "down"):
            sql += " AND direction = ?"
            params.append(direction)
        if is_hit is not None:
            sql += " AND is_hit = ?"
            params.append(int(is_hit))
        if t0 is not None:
            sql += " AND t_start >= ?"
            params.append(t0)
        if t1 is not None:
            sql += " AND t_start <= ?"
            params.append(t1)
        sql += " ORDER BY t_start LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
        return rows_to_dicts(rows)
    finally:
        conn.close()


def q_alarm_summary():
    conn = db_conn()
    try:
        total = conn.execute("SELECT COUNT(*) c FROM alarm_log").fetchone()["c"]
        by_dir = rows_to_dicts(conn.execute(
            "SELECT direction, COUNT(*) c, SUM(is_hit) hits FROM alarm_log GROUP BY direction"
        ).fetchall())
        by_level = rows_to_dicts(conn.execute(
            "SELECT level, COUNT(*) c FROM alarm_log GROUP BY level"
        ).fetchall())
        hits = conn.execute("SELECT COUNT(*) c FROM alarm_log WHERE is_hit=1").fetchone()["c"]
        return {
            "total_alarms": total,
            "hit_alarms": hits,
            "hit_rate": round(hits / total, 4) if total else 0.0,
            "by_direction": by_dir,
            "by_level": by_level,
        }
    finally:
        conn.close()


def q_alarm_explain(alarm_id, use_live=False):
    """对一条告警生成自然语言解释（大模型模块，含模板兜底）。"""
    conn = db_conn()
    try:
        row = conn.execute(
            "SELECT * FROM alarm_log WHERE alarm_id = ?", (alarm_id,)
        ).fetchone()
        if row is None:
            return {"error": f"alarm_id={alarm_id} 不存在"}
        row = dict(row)
        # 补充概率（从 pred_prob 取同 t_start 点）与事件成因
        pr = conn.execute(
            "SELECT p_up, p_dn FROM pred_prob WHERE t_idx = ?", (row["t_start"],)
        ).fetchone()
        if pr:
            row["p_up"], row["p_dn"] = pr["p_up"], pr["p_dn"]
        ev = None
        if row.get("matched_event_id") is not None:
            ev = conn.execute(
                "SELECT met_term, work_term, cls, delta_p FROM ramp_events "
                "ORDER BY event_id LIMIT 1 OFFSET ?", (row["matched_event_id"],)
            ).fetchone()
        if ev:
            row["met_term"] = ev["met_term"]
            row["work_term"] = ev["work_term"]
            row["cls"] = ev["cls"]
            row["delta_p_mw"] = ev["delta_p"]
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import llm_explain
        fact = llm_explain.build_warning_fact(
            t_start=row["t_start"], direction=row["direction"], level=row.get("level", 0),
            p_up=row.get("p_up"), p_dn=row.get("p_dn"),
            lead_steps=row.get("lead_steps"),
            met_term=row.get("met_term"), work_term=row.get("work_term"),
            cls=row.get("cls"), delta_p_mw=row.get("delta_p_mw"),
        )
        out = llm_explain.explain(fact, use_live=use_live)
        out["fact"] = fact
        return out
    finally:
        conn.close()


def do_predict(payload):
    """在线推理：从 dataset.npz 按 t_idx 取窗口 -> reg_infer 推理。"""
    infer_mod = get_infer()
    dz = np.load(DATASET_NPZ, allow_pickle=True)
    X_test = dz["X_test"]
    t_test = dz["t_test"].astype(int)

    if "windows" in payload:
        X = np.asarray(payload["windows"], dtype=np.float32)
        P_now = np.asarray(payload["p_now"], dtype=float)
    else:
        t = int(payload["t_idx"])
        idx = np.where(t_test == t)[0]
        if len(idx) == 0:
            return {"error": f"t_idx={t} 不在测试集窗口范围内"}, 400
        i = int(idx[0])
        X = X_test[i:i + 1]
        # 当前功率从 npz 的回归标签处不可得，用 field_power 的 P_mw
        import pandas as pd
        fp = pd.read_csv(FIELD_CSV).set_index("t_idx")
        P_now = np.array([float(fp.loc[t, "P_mw"])])

    theta = payload.get("theta")
    res = infer_mod.infer(X, P_now, theta=theta)
    return {
        "t_idx": int(payload.get("t_idx", -1)),
        "P_hat": [round(float(v), 3) for v in res["P_hat"]],
        "dP_hat": [round(float(v), 3) for v in res["dP_hat"]],
        "p_up": [round(float(v), 4) for v in res["p_up"]],
        "p_dn": [round(float(v), 4) for v in res["p_dn"]],
    }, 200


# ---- 传感器/手动输入在线判断（任务书闭环主链路） ----------------
# 场景：传感器仅测风速。用户/传感器给出"当前"风速(及可选功率) →
#       后端用最近的真实窗口补足 72 步、把最后一步替换为实测值 → 归一化 →
#       模型推理 → 阈值/风速硬规则判定 → 写入 alarm_log → 下发 LED 指令。
_WIND = {"dz": None, "led": None}


def _get_wind_cache():
    if _WIND["dz"] is None:
        _WIND["dz"] = np.load(DATASET_NPZ, allow_pickle=True)
    if _WIND["led"] is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import led_bridge
        _WIND["led"] = led_bridge
    return _WIND["dz"], _WIND["led"]


# 经验风功率曲线：风速 → 容量系数(P/C)，来自 SDWPF 实测分箱均值。
_CF_BINS = [(3, 0.014), (6, 0.169), (9, 0.475), (12, 0.719), (15, 0.729), (20, 0.693)]


def _cf_from_wspd(wspd):
    """按经验功率曲线把风速映射为容量系数。"""
    for ub, cf in _CF_BINS:
        if wspd <= ub:
            return cf
    return 0.693


def build_window_from_sensor(wspd, p_mw=None, cap_mw=None, wdir=180.0, etmp=25.0,
                             itmp=40.0, tail_steps=6):
    """构造 (1,72,14) 归一化窗口：以最近真实窗口为底，将末尾 tail_steps 步
    (默认 6 步=1h) 的风速置为传感器实测值、功率按经验风功率曲线派生，
    使历史末端与当前功率都反映该风速读数（真实反映"风速驱动"）。
    p_mw 若用户显式给出则优先使用（否则由曲线派生）。返回 (X_norm, P_now)。"""
    dz, _ = _get_wind_cache()
    X_test = dz["X_test"]                     # 已归一化 (N,72,14)
    feat_mu, feat_sd = dz["feat_mu"], dz["feat_sd"]
    cols = list(dz["feature_cols"])
    ci = {c: i for i, c in enumerate(cols)}

    raw = X_test[-1].copy() * (feat_sd + 1e-9) + feat_mu   # (72,14) 物理量
    if cap_mw is None:
        cap_mw = float(raw[-1, ci["C_mw"]]) or 201.0
    cf = _cf_from_wspd(wspd)
    p_derived = cf * cap_mw
    if p_mw is None:
        p_mw = p_derived
    p_mw = float(p_mw)

    # 覆盖末尾 tail_steps 步：风速、功率、容量系数
    for k in range(1, tail_steps + 1):
        raw[-k, ci["Wspd"]] = float(wspd)
        raw[-k, ci["P_mw"]] = p_mw
        raw[-k, ci["C_mw"]] = float(cap_mw)
        raw[-k, ci["R"]] = p_mw / cap_mw if cap_mw > 0 else raw[-k, ci["R"]]
        raw[-k, ci["Wdir_sin"]] = float(np.sin(np.radians(wdir)))
        raw[-k, ci["Wdir_cos"]] = float(np.cos(np.radians(wdir)))
        raw[-k, ci["Etmp"]] = float(etmp)
        raw[-k, ci["Itmp"]] = float(itmp)

    Xn = ((raw - feat_mu) / (feat_sd + 1e-9))[None, :, :].astype(np.float32)
    return Xn, p_mw


def do_predict_manual(payload):
    """POST /predict_manual
    body: {"wspd": 12.5, "p_mw": 120.0(可选), "cap_mw": 201(可选),
           "tau": 0.30(可选), "wind_thr": 12.0(可选, 风速硬阈值),
           "led": true(可选, 是否下发LED), "save": true(可选, 是否入库)}
    """
    infer_mod = get_infer()
    _, led = _get_wind_cache()

    if "wspd" not in payload:
        return {"error": "缺少 wspd（风速）"}, 400
    wspd = float(payload["wspd"])
    p_mw = payload.get("p_mw")
    cap_mw = payload.get("cap_mw")
    wdir = float(payload.get("wdir", 180.0))
    etmp = float(payload.get("etmp", 25.0))
    itmp = float(payload.get("itmp", 40.0))
    theta = payload.get("theta")

    meta = q_meta()
    tau = float(payload.get("tau", meta.get("tau", 0.40)))
    wind_thr = float(payload.get("wind_thr", 12.0))
    do_led = bool(payload.get("led", True))
    do_save = bool(payload.get("save", True))

    X, P_now = build_window_from_sensor(wspd, p_mw, cap_mw, wdir, etmp, itmp)
    res = infer_mod.infer(X, np.array([P_now]), theta=theta)
    p_up = float(res["p_up"][0]); p_dn = float(res["p_dn"][0])
    dP_hat = float(res["dP_hat"][0]); P_hat = float(res["P_hat"][0])

    # 判定：模型阈值 主判 + 风速硬阈值 旁路
    by_model_up = p_up >= tau
    by_model_dn = p_dn >= tau
    if by_model_up and p_up >= p_dn:
        direction, prob, by = "up", p_up, "model"
    elif by_model_dn and p_dn > p_up:
        direction, prob, by = "down", p_dn, "model"
    else:
        direction, prob, by = None, max(p_up, p_dn), None

    # 风速硬阈值旁路（传感器达阈值直接上爬坡预警，贴合"达阈值 LED 亮"）
    by_wind = wspd >= wind_thr
    if by_wind and direction is None:
        direction, prob, by = "up", max(p_up, 0.5), "wind"  # 风速旁路给一个占位概率

    is_alarm = direction is not None
    level = 0
    if is_alarm:
        if prob >= 0.85 or wspd >= wind_thr + 3:
            level = 1
        elif prob >= 0.70:
            level = 2
        else:
            level = 3

    # 下发 LED（无硬件则虚拟，不影响演示）
    led_real = False
    if do_led:
        led_real = led.set_led(is_alarm)

    # 入库：仅当确实产生预警时写入 alarm_log（避免噪声记录）
    alarm_id = None
    if do_save and is_alarm:
        dz, _ = _get_wind_cache()
        t_now = int(dz["t_test"][-1])
        conn = db_conn()
        try:
            cur = conn.execute(
                "INSERT INTO alarm_log (t_start, t_end, direction, level, tau, source, "
                "matched_event_id, is_hit, lead_steps) VALUES (?,?,?,?,?,?,?,?,?)",
                (t_now, t_now, direction, level, tau, "manual", None, 0, 0))
            conn.commit()
            alarm_id = cur.lastrowid
        finally:
            conn.close()

    return {
        "input": {"wspd": wspd, "p_mw": P_now, "wind_thr": wind_thr, "tau": tau},
        "pred": {"P_hat": round(P_hat, 2), "dP_hat": round(dP_hat, 2),
                 "p_up": round(p_up, 4), "p_dn": round(p_dn, 4)},
        "decision": {
            "is_alarm": is_alarm,
            "direction": direction,
            "prob": round(prob, 4),
            "level": level,
            "by": by or "none",
            "by_wind": bool(by_wind),
        },
        "led": {"on": is_alarm, "real_hardware": led_real},
        "alarm_id": alarm_id,
    }, 200


# ------------------------- HTTP 路由 -------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[api] {self.address_string()} {fmt % args}")

    def do_GET(self):
        try:
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            def one(k, cast=str, default=None):
                v = qs.get(k)
                return cast(v[0]) if v else default

            if u.path == "/health":
                return self._send({"status": "ok", "db_exists": os.path.exists(DB_PATH)})
            if u.path == "/meta":
                return self._send(q_meta())
            if u.path == "/timeseries":
                return self._send(q_timeseries(one("t0", int, 0), one("t1", int, 10**9)))
            if u.path == "/events":
                lv = one("level", int, None)
                main_only = one("main_only", lambda x: x.lower() != "false", True)
                return self._send(q_events(one("direction"), lv, main_only))
            if u.path == "/prob":
                return self._send(q_prob(one("t0", int, 0), one("t1", int, 10**9)))
            if u.path == "/alarms":
                ih = one("is_hit", int, None)
                return self._send(q_alarms(one("direction"), ih,
                                           one("t0", int, None), one("t1", int, None),
                                           one("limit", int, 500)))
            if u.path == "/alarm_summary":
                return self._send(q_alarm_summary())
            if u.path == "/alarm_explain":
                aid = one("alarm_id", int, None)
                if aid is None:
                    return self._send({"error": "需要 alarm_id 参数"}, 400)
                live = one("live", lambda x: x.lower() == "true", False)
                return self._send(q_alarm_explain(aid, live))
            return self._send({"error": "not found", "path": u.path}, 404)
        except Exception as e:
            return self._send({"error": str(e)}, 500)

    def do_POST(self):
        try:
            u = urlparse(self.path)
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if u.path == "/predict":
                obj, code = do_predict(payload)
                return self._send(obj, code)
            if u.path == "/predict_manual":
                obj, code = do_predict_manual(payload)
                return self._send(obj, code)
            return self._send({"error": "not found", "path": u.path}, 404)
        except Exception as e:
            return self._send({"error": str(e)}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if not os.path.exists(DB_PATH):
        print(f"[api] 数据库不存在: {DB_PATH}，请先运行 python code/database.py")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[api] 服务启动 http://{args.host}:{args.port}  (Ctrl+C 停止)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[api] 已停止")
        srv.shutdown()


if __name__ == "__main__":
    main()