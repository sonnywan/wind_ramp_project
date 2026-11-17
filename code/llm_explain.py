# -*- coding: utf-8 -*-
"""
llm_explain.py —— 大模型驱动的预警解释生成模块（P1-8）
=====================================================================
任务书硬指标："了解常用大模型技术（如 GPT、BERT 等），能利用大模型技术实现某种功能"。

本模块用大语言模型（LLM）为每一次爬坡预警生成自然语言"预警解读"，面向调度人员：
  输入：一条告警/事件的结构化信息（方向、等级、超阈概率、提前量、成因分解、关键气象特征）
  输出：一段可读的预警解释 + 建议调度动作。

设计（工程可落地、离线可演示）：
  - 首选：调用 OpenAI 兼容 Chat Completions 接口（可用 GPT / 通义 / DeepSeek / 本地 Ollama 等
    任何 OpenAI 协议兼容服务）；通过环境变量配置 base_url / api_key / model。
  - 兜底：若未配置或调用失败，使用**规则模板 NLG**（TemplateGenerator）确定性生成解释，
    保证零依赖、离线可复现，作为演示与单元测试的稳定基线。

Prompt 工程：
  采用"系统角色 + 结构化事实 + 输出格式约束"的三段式提示，强制模型只依据给定事实生成，
  避免幻觉；并要求输出 JSON（summary / action / reason），便于前端展示。

用法：
  python code/llm_explain.py                 # 用一条示例预警演示（离线模板）
  set LLM_API_KEY=sk-xxx & set LLM_BASE_URL=https://... & python code/llm_explain.py --live
"""
import os
import json
import argparse

# ---------------------------------------------------------------------------
# 1. 预警事实的结构化表示
# ---------------------------------------------------------------------------
def build_warning_fact(t_start, direction, level, p_up, p_dn, lead_steps,
                       met_term=None, work_term=None, cls=None,
                       wspd_mean=None, delta_p_mw=None):
    """把一条预警整理成可解释的事实字典（供 LLM 与模板共用）。
    direction: 'up'/'down'；level: 1/2/3/0；p_up/p_dn: 超阈概率。"""
    dir_cn = "上爬坡" if direction == "up" else "下爬坡"
    level_cn = {3: "三级（关注）", 2: "二级（准备备用）", 1: "一级（立即调度）"}.get(level, "未分级")
    prob = p_up if direction == "up" else p_dn
    # 成因判定
    if met_term is not None and work_term is not None:
        am, aw = abs(met_term), abs(work_term)
        if am >= 2 * aw:
            cause = "气象型"
        elif aw >= 2 * am:
            cause = "工况型"
        else:
            cause = "混合型"
    else:
        cause = cls or "未知"
    return {
        "t_start": int(t_start),
        "direction": direction,
        "dir_cn": dir_cn,
        "level": int(level),
        "level_cn": level_cn,
        "prob": float(prob) if prob is not None else None,
        "lead_steps": int(lead_steps) if lead_steps is not None else None,
        "lead_hours": round(lead_steps * 10 / 60, 1) if lead_steps is not None else None,
        "cause": cause,
        "met_term": None if met_term is None else round(float(met_term), 2),
        "work_term": None if work_term is None else round(float(work_term), 2),
        "wspd_mean": None if wspd_mean is None else round(float(wspd_mean), 2),
        "delta_p_mw": None if delta_p_mw is None else round(float(delta_p_mw), 2),
    }


# ---------------------------------------------------------------------------
# 2. 规则模板 NLG（离线兜底）
# ---------------------------------------------------------------------------
def template_explain(fact):
    """基于模板的确定性解释生成，保证离线可运行。"""
    d = fact["dir_cn"]
    lv = fact["level_cn"]
    prob = fact["prob"]
    lead_h = fact["lead_hours"]
    cause = fact["cause"]

    parts = []
    parts.append(f"预计未来将出现一次【{d}】，预警等级为{lv}。")
    if prob is not None:
        parts.append(f"模型给出的超阈概率约 {prob*100:.0f}%。")
    if lead_h is not None:
        if lead_h > 0:
            parts.append(f"预警提前量约 {lead_h} 小时。")
        elif lead_h == 0:
            parts.append("预警与事件起点几乎同步。")
        # 负值（告警合并起点晚于事件起点）不写入解读，避免误导
    if fact.get("delta_p_mw") is not None:
        sign = "上升" if fact["direction"] == "up" else "下降"
        parts.append(f"参考本次事件功率{sign}幅度约 {abs(fact['delta_p_mw']):.1f} MW。")

    # 成因与依据
    if cause == "气象型":
        wspd = fact.get("wspd_mean")
        wtxt = f"（平均风速 {wspd} m/s）" if wspd is not None else ""
        parts.append(f"成因判定为气象型{wtxt}：主要由风速变化驱动的容量系数变动引起，属模型可预报范畴。")
        reason = "主导项为气象项，预测可信度较高"
    elif cause == "工况型":
        parts.append("成因判定为工况型：主要由限电/停机导致的可用装机容量变化引起，属模型难以从气象特征预测的部分，"
                     "系统对该类预警已降级处理，仅供参考。")
        reason = "主导项为工况项，建议核实场站运行状态"
    elif cause == "混合型":
        parts.append("成因判定为混合型：气象因素与工况因素共同作用，需结合现场运行情况综合判断。")
        reason = "气象项与工况项共同作用"
    else:
        reason = "成因信息不足"

    # 建议动作（按等级）
    action_map = {
        1: "建议立即启动调度预案：安排调峰备用、通知相关机组调整出力，并持续跟踪功率走势。",
        2: "建议进入准备状态：预留旋转备用、密切关注后续概率变化，必要时升级预警。",
        3: "建议保持关注：该级别为关注级，暂无需立即动作，但应监测是否升级。",
        0: "建议继续常规监测。",
    }
    parts.append(action_map.get(fact["level"], action_map[0]))

    return {
        "summary": "".join(parts),
        "action": action_map.get(fact["level"], action_map[0]),
        "reason": reason,
        "source": "template",
    }


# ---------------------------------------------------------------------------
# 3. LLM 生成（OpenAI 兼容接口）
# ---------------------------------------------------------------------------
def build_prompt(fact):
    """三段式 Prompt：系统角色 + 结构化事实 + 输出约束。"""
    system = (
        "你是风电场爬坡预警系统的调度辅助助手。你只能依据给定的结构化事实生成解释，"
        "不得编造未提供的数据；语气专业、简洁，面向电网调度人员。"
    )
    fact_json = json.dumps(fact, ensure_ascii=False, indent=2)
    user = (
        "下面是一条风电爬坡预警的结构化事实，请生成预警解读。\n"
        f"事实：\n{fact_json}\n\n"
        "请严格输出如下 JSON（不要额外文字）：\n"
        '{"summary": "一句话总体解读", "action": "建议的调度动作", "reason": "判定依据"}'
    )
    return system, user


def llm_explain(fact, base_url=None, api_key=None, model=None, timeout=20):
    """调用 OpenAI 兼容接口；失败或无配置则抛异常由上层兜底。"""
    base_url = base_url or os.environ.get("LLM_BASE_URL")
    api_key = api_key or os.environ.get("LLM_API_KEY")
    model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
    if not base_url or not api_key:
        raise RuntimeError("未配置 LLM_BASE_URL / LLM_API_KEY，回退模板")
    try:
        import requests
    except ImportError:
        raise RuntimeError("未安装 requests，回退模板")

    system, user = build_prompt(fact)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    url = base_url.rstrip("/") + "/chat/completions"
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}"},
                         json=payload, timeout=timeout)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        obj = {"summary": content, "action": "", "reason": ""}
    obj["source"] = "llm"
    return obj


def explain(fact, use_live=False, **kw):
    """统一入口：use_live 优先 LLM，失败自动回退模板。"""
    if use_live:
        try:
            return llm_explain(fact, **kw)
        except Exception as e:
            out = template_explain(fact)
            out["fallback_reason"] = str(e)
            return out
    return template_explain(fact)


# ---------------------------------------------------------------------------
# 4. 从数据库批量生成解释（供后端 /alarm_explain 使用）
# ---------------------------------------------------------------------------
def explain_alarm_row(row, use_live=False):
    """从 alarm_log 的一行（dict）构造事实并生成解释。"""
    fact = build_warning_fact(
        t_start=row.get("t_start"),
        direction=row.get("direction"),
        level=row.get("level", 0),
        p_up=row.get("p_up"),
        p_dn=row.get("p_dn"),
        lead_steps=row.get("lead_steps"),
    )
    return explain(fact, use_live=use_live)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="尝试调用真实 LLM")
    args = ap.parse_args()

    # 示例：一条一级上爬坡预警
    fact = build_warning_fact(
        t_start=30200, direction="up", level=1, p_up=0.72, p_dn=0.03,
        lead_steps=6, met_term=57.3, work_term=-14.9, wspd_mean=9.8, delta_p_mw=42.4,
    )
    print("[llm_explain] 结构化事实:")
    print(json.dumps(fact, ensure_ascii=False, indent=2))
    print("\n[llm_explain] 生成解释:")
    out = explain(fact, use_live=args.live)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()