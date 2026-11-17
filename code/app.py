# -*- coding: utf-8 -*-
# app.py —— 风电功率爬坡预警系统 Streamlit 双面板演示（美化版）
#  面板① 爬坡发生预警：方向无关，只回答"未来 1h 是否有爬坡"（强项，p_ramp AUC≈0.82）
#  面板② 方向判定：上行 / 下行（方向信号较弱，事件级上行命中率≈66.7%，v2 模型）
# 离线模式：静态窗口回放；实时模式：计时器驱动的滚动回放 test 段。

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm

# ============================================================
# 先注册字体（必须在 import pyplot 之前，否则 rcParams 拿不到）
# ============================================================
_FONT_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
_PACKAGED_FONT = os.path.join(_FONT_ASSETS, "NotoSansCJKsc-Regular.otf")
_cn_fonts = []

if os.path.exists(_PACKAGED_FONT):
    fm.fontManager.addfont(_PACKAGED_FONT)
    _packaged_name = fm.FontProperties(fname=_PACKAGED_FONT).get_name()
    _cn_fonts.append(_packaged_name)

# 补充系统可能有的中文字体
for f in fm.fontManager.ttflist:
    if any(k in f.name for k in ("YaHei", "Microsoft Ya", "SimHei", "SimSun", "KaiTi",
                                  "Noto Sans CJK", "WenQuanYi", "Source Han Sans")):
        if f.name not in _cn_fonts:
            _cn_fonts.append(f.name)

import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = (_cn_fonts + ["DejaVu Sans"]) if _cn_fonts else ["DejaVu Sans"]

# ---------- 统一配色（风电主题）----------
C_POWER    = "#2563eb"   # 功率曲线：风电蓝
C_RAMP     = "#7c3aed"   # 爬坡概率：紫色
C_UP       = "#dc2626"   # 上行：警示红
C_DN       = "#16a34a"   # 下行：森林绿
C_EVENT_UP = "#fca5a5"   # 上行事件色带（浅红）
C_EVENT_DN = "#bbf7d0"   # 下行事件色带（浅绿）
C_THRESH   = "#6b7280"   # 阈值线：灰
C_PRIMARY  = "#2563eb"   # 主色调（蓝色）
C_BG_DARK  = "#0f172a"   # 深色背景用

OUT_DIR = "output"
FIELD_CSV = os.path.join(OUT_DIR, "field_power.csv")
EVENT_CSV = os.path.join(OUT_DIR, "ramp_events.csv")
CLS_NPZ = os.path.join(OUT_DIR, "lgb_pred.npz")
REG_NPZ = os.path.join(OUT_DIR, "reg_prob.npz")
COST_PNG = os.path.join(OUT_DIR, "cost_curve.png")
REL_PNG = os.path.join(OUT_DIR, "reliability_diagram.png")

HYST_ON_STEPS = 2
HYST_OFF_RATIO = 0.8
LEAD_STEPS = 6   # 预警提前量 L=6 步，与 event_eval.py 口径一致


# ============================================================
# 全局 CSS 注入（统一主题 + 滑块/metric/卡片美化）
# ============================================================
def inject_css():
    st.markdown(f"""
    <style>
    /* --- 全局 --- */
    .stApp {{
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 30%);
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%);
        border-right: 1px solid #cbd5e1;
    }}

    /* --- 标题区 --- */
    .app-header {{
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 20px 24px;
        background: linear-gradient(135deg, {C_PRIMARY} 0%, #0891b2 100%);
        border-radius: 14px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 4px 16px rgba(37,99,235,0.25);
    }}
    .app-header .wind-icon {{
        font-size: 42px;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
    }}
    .app-header h1 {{
        margin: 0 !important;
        font-size: 26px !important;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}
    .app-header p {{
        margin: 4px 0 0 0;
        opacity: 0.9;
        font-size: 13px;
    }}

    /* --- Metric 卡片 --- */
    .metric-card {{
        background: white;
        border-radius: 12px;
        padding: 16px 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        transition: transform 0.15s, box-shadow 0.15s;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }}
    .metric-label {{
        color: #64748b;
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .metric-value {{
        font-size: 28px;
        font-weight: 700;
        margin-top: 6px;
        color: #0f172a;
    }}
    .metric-value.hit {{ color: {C_DN}; }}
    .metric-value.miss {{ color: {C_UP}; }}

    /* --- 滑块主题色 --- */
    [data-testid="stSlider"] > div > div > div > div {{
        background: {C_PRIMARY} !important;
    }}

    /* --- 状态灯 --- */
    .led {{
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 8px currentColor;
        animation: pulse 1.6s ease-in-out infinite;
    }}
    .led.green {{ background: {C_DN}; color: {C_DN}; }}
    .led.red   {{ background: {C_UP};   color: {C_UP}; }}
    .led.yellow{{ background: #d97706;  color: #d97706; }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50%      {{ opacity: 0.55; }}
    }}
    .status-banner {{
        padding: 12px 16px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .status-banner.safe   {{ background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }}
    .status-banner.alert  {{ background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }}
    .status-banner.warn   {{ background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }}

    /* --- 面板卡片 --- */
    .panel-card {{
        background: white;
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        margin-bottom: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    .panel-title {{
        font-size: 15px;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #f1f5f9;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .panel-title .badge {{
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 999px;
    }}
    .badge.strong {{ background: #dbeafe; color: #1d4ed8; }}
    .badge.sigweak{{ background: #fef3c7; color: #b45309; }}

    /* --- 隐藏 Streamlit 默认的一些丑陋元素 --- */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# 数据加载（带缓存）
# ============================================================
@st.cache_data
def load_field():
    return pd.read_csv(FIELD_CSV)


@st.cache_data
def load_source(source):
    if source == "cls":
        d = np.load(CLS_NPZ, allow_pickle=True)
        prob = d["prob_test"]
        p_up = prob[:, 2].astype(float)
        p_dn = prob[:, 0].astype(float)
        p_ramp = 1.0 - prob[:, 1].astype(float)
        t_idx = d["t_test"].astype(int)
    else:
        d = np.load(REG_NPZ)
        p_up = d["p_up_test"].astype(float)
        p_dn = d["p_dn_test"].astype(float)
        p_ramp = np.clip(p_up + p_dn, 0.0, 1.0)
        t_idx = d["t_test"].astype(int)
    order = np.argsort(t_idx)
    return p_up[order], p_dn[order], p_ramp[order], t_idx[order]


@st.cache_data
def load_events(t_min, t_max, main_only=True):
    df = pd.read_csv(EVENT_CSV)
    if main_only:
        df = df[df["is_main"] == 1].copy()
    df["start_t_idx"] = df["start_tidx"].astype(int)
    df["end_t_idx"] = df["start_tidx"].astype(int) + df["n_trig"].astype(int) - 1
    df = df[(df["start_t_idx"] >= t_min) & (df["start_t_idx"] <= t_max)]
    return df


def apply_hysteresis(p, tau, on_steps=HYST_ON_STEPS, off_ratio=HYST_OFF_RATIO):
    n = len(p)
    alarm = np.zeros(n, dtype=int)
    state = 0
    consec = 0
    off_thr = off_ratio * tau
    for i in range(n):
        v = p[i]
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


def simple_alarm(p, tau):
    return (p >= tau).astype(int)


# ============================================================
# 图表（美化版：统一配色 + 渐变色填充 + 精致网格）
# ============================================================
def _draw_events(ax, events):
    for _, ev in events.iterrows():
        s, e = int(ev["start_t_idx"]), int(ev["end_t_idx"])
        is_up = str(ev["direction"]).lower() == "up"
        color = C_EVENT_UP if is_up else C_EVENT_DN
        ax.axvspan(s, e, color=color, alpha=0.55, zorder=0)


def _style_axes(ax, ylabel=None, ylim=None):
    """统一的坐标轴美化。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.grid(True, alpha=0.25, color="#cbd5e1", linewidth=0.6, zorder=0)
    ax.tick_params(colors="#64748b", labelsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color="#475569", fontsize=10, fontweight=500)
    if ylim:
        ax.set_ylim(ylim)


def plot_ramp_panel(t_axis, P, p_ramp, alarm_ramp, events, tau_ramp):
    """面板① 爬坡发生预警。"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax0, ax1 = axes

    # 上部：功率
    ax0.plot(t_axis, P, color=C_POWER, lw=1.3, label="场站功率 P", zorder=3)
    ax0.fill_between(t_axis, P, alpha=0.08, color=C_POWER, zorder=2)
    _draw_events(ax0, events)
    _style_axes(ax0, ylabel="功率 (MW)")
    ax0.legend(loc="upper right", fontsize=8, framealpha=0.9,
               edgecolor="#e2e8f0", fancybox=True)

    # 下部：爬坡概率 + 告警
    ax1.plot(t_axis, p_ramp, color=C_RAMP, lw=1.2, label="爬坡概率", zorder=3)
    ax1.fill_between(t_axis, 0, p_ramp, alpha=0.18, color=C_RAMP, zorder=2)
    ax1.axhline(tau_ramp, color=C_THRESH, ls="--", lw=0.9,
                label=f"阈值 τ1={tau_ramp:.2f}", zorder=1)
    ar = np.where(alarm_ramp == 1)[0]
    if len(ar) > 0:
        ax1.scatter(t_axis[ar], p_ramp[ar], s=14, color=C_UP,
                    marker="o", zorder=5, label="爬坡预警", edgecolors="white", linewidths=0.6)
    _style_axes(ax1, ylabel="概率", ylim=(0, 1))
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9,
               edgecolor="#e2e8f0", fancybox=True, ncol=3)

    fig.tight_layout(pad=1.2)
    return fig


def plot_direction_panel(t_axis, P, p_up, p_dn, alarm_up, alarm_dn, events, tau_dir):
    """面板② 方向判定。"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax0, ax1 = axes

    # 上部：功率
    ax0.plot(t_axis, P, color=C_POWER, lw=1.3, label="场站功率 P", zorder=3)
    ax0.fill_between(t_axis, P, alpha=0.08, color=C_POWER, zorder=2)
    _draw_events(ax0, events)
    _style_axes(ax0, ylabel="功率 (MW)")
    ax0.legend(loc="upper right", fontsize=8, framealpha=0.9,
               edgecolor="#e2e8f0", fancybox=True)

    # 下部：上行/下行概率
    ax1.plot(t_axis, p_up, color=C_UP, lw=1.1, label="上行 p_up", zorder=3)
    ax1.plot(t_axis, p_dn, color=C_DN, lw=1.1, label="下行 p_dn", zorder=3)
    ax1.fill_between(t_axis, 0, p_up, alpha=0.12, color=C_UP, zorder=2)
    ax1.fill_between(t_axis, 0, p_dn, alpha=0.12, color=C_DN, zorder=2)
    ax1.axhline(tau_dir, color=C_THRESH, ls="--", lw=0.9,
                label=f"阈值 τ2={tau_dir:.2f}", zorder=1)
    au = np.where(alarm_up == 1)[0]
    ad = np.where(alarm_dn == 1)[0]
    if len(au) > 0:
        ax1.scatter(t_axis[au], p_up[au], s=16, color=C_UP, marker="v",
                    zorder=5, edgecolors="white", linewidths=0.6, label="上行告警")
    if len(ad) > 0:
        ax1.scatter(t_axis[ad], p_dn[ad], s=16, color=C_DN, marker="^",
                    zorder=5, edgecolors="white", linewidths=0.6, label="下行告警")
    _style_axes(ax1, ylabel="概率", ylim=(0, 1))
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9,
               edgecolor="#e2e8f0", fancybox=True, ncol=4)

    fig.tight_layout(pad=1.2)
    return fig


# ============================================================
# 指标计算（事件级 + 一对一匹配，与 run_event_eval.py 口径一致）
# ============================================================
def _merge_alarm_events(alarm_array, t_idx, merge_gap=6):
    """把逐点告警数组合并成告警事件列表（同向间隔 ≤ merge_gap 步合并）。"""
    times = t_idx[np.where(alarm_array == 1)[0]]
    if len(times) == 0:
        return []
    evts = []
    cur_s = cur_e = times[0]
    for t in times[1:]:
        if t - cur_e <= merge_gap:
            cur_e = t
        else:
            evts.append({"start": int(cur_s), "end": int(cur_e)})
            cur_s = cur_e = t
    evts.append({"start": int(cur_s), "end": int(cur_e)})
    return evts


def _match(true_events, pred_events):
    """一对一匹配：每个真实事件最多命中一个预测事件，按起点最早优先。"""
    used = set()
    hit = 0
    for te in true_events:
        best_pi = None
        best_start = float("inf")
        for pi, pe in enumerate(pred_events):
            if pi in used:
                continue
            # 时间窗交叠
            if not (pe["end"] < te["start_t_idx"] or te["end_t_idx"] < pe["start"]):
                if pe["start"] < best_start:
                    best_start = pe["start"]
                    best_pi = pi
        if best_pi is not None:
            used.add(best_pi)
            hit += 1
    return hit, len(pred_events) - len(used)   # hit, num_false_alarm


def compute_ramp_metrics(t_idx, alarm_ramp, events):
    """面板①：方向无关爬坡事件级命中率。"""
    # 真实事件（方向无关）
    true_list = [{"start_t_idx": int(r["start_t_idx"]), "end_t_idx": int(r["end_t_idx"])}
                 for _, r in events.iterrows()]
    pred_list = _merge_alarm_events(alarm_ramp, t_idx)
    hit, fa = _match(true_list, pred_list)
    n = len(true_list)
    return {"n_events": n,
            "hit_rate": hit / n if n else 0.0,
            "hit": hit, "false_alarm": fa,
            "n_pred": len(pred_list),
            "alarm_points": int(alarm_ramp.sum())}


def compute_quick_metrics(t_idx, alarm_up, alarm_dn, events):
    """面板②：分方向事件级命中率。"""
    up_true = [{"start_t_idx": int(r["start_t_idx"]), "end_t_idx": int(r["end_t_idx"])}
               for _, r in events.iterrows() if str(r["direction"]).lower() == "up"]
    dn_true = [{"start_t_idx": int(r["start_t_idx"]), "end_t_idx": int(r["end_t_idx"])}
               for _, r in events.iterrows() if str(r["direction"]).lower() == "down"]
    pred_up = _merge_alarm_events(alarm_up, t_idx)
    pred_dn = _merge_alarm_events(alarm_dn, t_idx)

    hit_up, fa_up = _match(up_true, pred_up)
    hit_dn, fa_dn = _match(dn_true, pred_dn)

    n_up = len(up_true)
    n_dn = len(dn_true)
    return {"n_up": n_up, "n_dn": n_dn,
            "hit_up": hit_up, "hit_dn": hit_dn,
            "hit_rate_up": hit_up / n_up if n_up else 0.0,
            "hit_rate_dn": hit_dn / n_dn if n_dn else 0.0,
            "fa_up": fa_up, "fa_dn": fa_dn,
            "n_pred_up": len(pred_up), "n_pred_dn": len(pred_dn),
            "alarm_points": int(alarm_up.sum() + alarm_dn.sum())}


# ============================================================
# Metric 卡片（带图标 + 颜色 + 动画悬停）
# ============================================================
def metric_card(label, value_str, icon="📊", tone=""):
    """tone: hit / miss / warn 控制数字颜色。"""
    cls_tone = f" {tone}" if tone else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value{cls_tone}">{value_str}</div>
    </div>"""


# ============================================================
# 主页面
# ============================================================
st.set_page_config(page_title="风电爬坡预警系统", layout="wide",
                   page_icon="🌬️")
inject_css()

# ---------- 顶部 Header ----------
st.markdown("""
<div class="app-header">
    <div class="wind-icon">🌬️</div>
    <div>
        <h1>风电功率爬坡预测与预警系统</h1>
        <p>SDWPF/KDD Cup 2022 · 134 台风机 · 201 MW · 5min/步 · LightGBM v2（方向聚焦 + 极值加权）</p>
    </div>
</div>""", unsafe_allow_html=True)

# ---------- 侧栏 ----------
with st.sidebar:
    st.markdown("## ⚙️ 配置")
    mode = st.radio("运行模式", ["离线回放", "实时模拟器"], index=0)
    source = st.selectbox("概率源", ["cls", "reg"], index=0,
                          format_func=lambda x: "分类概率(cls)" if x == "cls" else "回归校准概率(reg)")
    use_hyst = st.checkbox("启用迟滞", value=True)
    st.markdown("**阈值**")
    tau_ramp = st.slider("爬坡预警阈值 τ1", 0.05, 0.95, 0.50, 0.05)
    tau_dir = st.slider("方向判定阈值 τ2", 0.05, 0.95, 0.40, 0.05)
    st.caption("τ1 用于面板①；τ2 用于面板②（默认 0.40，v2 模型事件级命中率≈72%）。")

    p_up, p_dn, p_ramp, t_idx = load_source(source)
    t_min, t_max = int(t_idx.min()), int(t_idx.max())
    field = load_field()
    events = load_events(t_min, t_max, main_only=True)
    up_n = int((events.direction == "up").sum())
    dn_n = int((events.direction == "down").sum())
    st.markdown(f"**测试区间**：{t_min} ~ {t_max}（{len(t_idx)} 步 ≈ {len(t_idx)*5/60:.0f}h）")
    st.markdown(f"**真实事件(主口径)**：{len(events)} 条  🔺{up_n} / 🔻{dn_n}")

# ---------- 告警 ----------
if use_hyst:
    alarm_ramp = apply_hysteresis(p_ramp, tau_ramp)
    alarm_up = apply_hysteresis(p_up, tau_dir)
    alarm_dn = apply_hysteresis(p_dn, tau_dir)
else:
    alarm_ramp = simple_alarm(p_ramp, tau_ramp)
    alarm_up = simple_alarm(p_up, tau_dir)
    alarm_dn = simple_alarm(p_dn, tau_dir)

P_all = field.set_index("t_idx")["P_mw"]
P_hist = P_all.reindex(t_idx).to_numpy()


def render_charts(sl):
    tx = t_idx[sl]
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📡 面板① 爬坡发生预警 '
                '<span class="badge strong">强项</span></div>', unsafe_allow_html=True)
    fig1 = plot_ramp_panel(tx, P_hist[sl], p_ramp[sl], alarm_ramp[sl], events, tau_ramp)
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🧭 面板② 方向判定 '
                '<span class="badge sigweak">方向信号较弱</span></div>', unsafe_allow_html=True)
    fig2 = plot_direction_panel(tx, P_hist[sl], p_up[sl], p_dn[sl],
                                alarm_up[sl], alarm_dn[sl], events, tau_dir)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    st.markdown('</div>', unsafe_allow_html=True)


# ================= 离线回放 =================
if mode == "离线回放":
    st.markdown("### 📈 离线回放模式")
    col1, col2 = st.columns([3, 1])
    with col2:
        win = st.slider("窗口大小(步)", 100, len(t_idx), min(720, len(t_idx)), 50)
        start = st.slider("起始位置", 0, max(0, len(t_idx) - win), 0, 50)
    render_charts(slice(start, start + win))

    # ---- 指标区：用卡片替代 st.metric ----
    mr = compute_ramp_metrics(t_idx, alarm_ramp, events)
    m = compute_quick_metrics(t_idx, alarm_up, alarm_dn, events)

    st.markdown("---")
    st.markdown("### 🎯 全测试集指标（事件级 · 一对一匹配）")

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(metric_card("真实爬坡事件", f"{mr['n_events']}", icon="🎯"),
                    unsafe_allow_html=True)
    with g2:
        st.markdown(metric_card("爬坡命中率",
                                f"{mr['hit']}/{mr['n_events']} ({mr['hit_rate']:.1%})",
                                icon="⚡", tone="hit" if mr["hit_rate"] >= 0.7 else "warn"),
                    unsafe_allow_html=True)
    with g3:
        st.markdown(metric_card("告警事件数", f"{mr['n_pred']}", icon="📦"),
                    unsafe_allow_html=True)
    with g4:
        st.markdown(metric_card("虚警事件数", f"{mr['false_alarm']}",
                                icon="⚠️", tone="miss" if mr["false_alarm"] > mr["hit"] else ""),
                    unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(metric_card("🔺 上行真实", f"{m['n_up']}", icon="🔺"),
                    unsafe_allow_html=True)
    with g2:
        st.markdown(metric_card("上行命中率",
                                f"{m['hit_up']}/{m['n_up']} ({m['hit_rate_up']:.1%})",
                                icon="🎯", tone="hit" if m["hit_rate_up"] >= 0.6 else "miss"),
                    unsafe_allow_html=True)
    with g3:
        st.markdown(metric_card("🔻 下行真实", f"{m['n_dn']}", icon="🔻"),
                    unsafe_allow_html=True)
    with g4:
        st.markdown(metric_card("下行命中率",
                                f"{m['hit_dn']}/{m['n_dn']} ({m['hit_rate_dn']:.1%})",
                                icon="🎯", tone="hit" if m["hit_rate_dn"] >= 0.7 else "miss"),
                    unsafe_allow_html=True)

    st.markdown(
        "<span style='color:#64748b;font-size:12px;'>"
        "📐 口径说明：告警事件按同向 ≤6 步合并；匹配为一对一。"
        "完整事件级指标见 <code>output/event_eval_report.md</code></span>",
        unsafe_allow_html=True)
    st.markdown(f"<span style='color:#64748b;font-size:12px;'>"
                f"面板①虚警 = {mr['false_alarm']}，上行虚警 = {m['fa_up']}，下行虚警 = {m['fa_dn']}，"
                f"逐点告警总点数 = {mr['alarm_points']}</span>", unsafe_allow_html=True)

    with st.expander("📊 附加图表（代价曲线 / 可靠性图）", expanded=False):
        cc1, cc2 = st.columns(2)
        if os.path.exists(COST_PNG):
            cc1.image(COST_PNG, caption="代价敏感阈值标定（创新点3）", use_container_width=True)
        if os.path.exists(REL_PNG):
            cc2.image(REL_PNG, caption="概率校准可靠性图（创新点2）", use_container_width=True)

# ================= 实时模拟器 =================
else:
    st.markdown("### ⏱️ 实时模拟器（滚动回放 test 段）")
    st.caption("逐点回放模型输出，模拟「实时来数 → 算概率 → 触发告警」的在线推理链路")
    n_total = len(t_idx)
    speed = st.slider("每帧推进步数", 1, 60, 6, 1, key="sim_speed")
    max_steps = st.slider("回放总步数", 100, n_total, min(1440, n_total), 100, key="sim_max")
    run = st.toggle("▶ 开始 / 暂停", value=False, key="sim_run")

    if "cursor" not in st.session_state or st.session_state.get("mode") != mode:
        st.session_state.cursor = 0
        st.session_state.mode = mode

    @st.fragment(run_every="0.25s")
    def replay_frame():
        _run = st.session_state.get("sim_run", False)
        _speed = int(st.session_state.get("sim_speed", speed))
        _max = int(st.session_state.get("sim_max", max_steps))

        if _run and st.session_state.cursor < _max:
            st.session_state.cursor = min(st.session_state.cursor + _speed, _max)

        cur = st.session_state.cursor
        if cur == 0:
            cur = min(_speed, _max)
            st.session_state.cursor = cur

        # ---- 顶部实时读数 ----
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("t_idx", int(t_idx[cur - 1]))
        pw = P_hist[cur - 1]
        a2.metric("场站功率 (MW)", f"{pw:.1f}" if np.isfinite(pw) else "N/A")
        a3.metric("爬坡概率", f"{p_ramp[cur - 1]:.3f}")
        a4.metric("上行概率", f"{p_up[cur - 1]:.3f}")
        a5.metric("下行概率", f"{p_dn[cur - 1]:.3f}")
        st.progress(cur / _max, text=f"回放进度 {cur}/{_max} 步")

        # ---- 状态灯（自定义 LED 替代 st.error/success）----
        s1, s2 = st.columns(2)
        with s1:
            if alarm_ramp[cur - 1] == 1:
                st.markdown('<div class="status-banner alert">'
                            '<span class="led red"></span>🚨 面板①：检测到爬坡风险！</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-banner safe">'
                            '<span class="led green"></span>✅ 面板①：当前平稳</div>',
                            unsafe_allow_html=True)
        with s2:
            if alarm_up[cur - 1] == 1:
                st.markdown('<div class="status-banner alert">'
                            '<span class="led red"></span>🔺 面板②：上行爬坡告警</div>',
                            unsafe_allow_html=True)
            elif alarm_dn[cur - 1] == 1:
                st.markdown('<div class="status-banner warn">'
                            '<span class="led yellow"></span>🔻 面板②：下行爬坡告警</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-banner safe">'
                            '<span class="led green"></span>✅ 面板②：方向未定 / 无告警</div>',
                            unsafe_allow_html=True)

        render_charts(slice(max(0, cur - 360), cur))

    replay_frame()

# ---------- 页脚 ----------
st.markdown("---")
st.caption(
    "🖥️ 风电功率爬坡预测与预警系统 · 内蒙古工业大学毕业设计 · "
    "v2 模型（方向聚焦权重 + 极值加权 + 两阶段 LightGBM）· "
    "默认操作点 τ=0.40 · 事件级总体命中率 72.1% · "
    "完整指标见 `output/event_eval_report.md`"
)
