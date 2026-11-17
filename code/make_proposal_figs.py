# -*- coding: utf-8 -*-
"""make_proposal_figs.py —— 开题报告插图

输出到 output/figs/：
  fig1_usecase_operator.png   图1 风场运行值班员用例图
  fig2_usecase_admin.png      图2 系统管理员用例图
  fig3_modules.png            图3 系统功能模块图
  fig4_architecture.png       图4 系统总体架构图
  fig5_route.png              图5 技术路线图
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Circle, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join("output", "figs")
os.makedirs(OUT, exist_ok=True)

BLUE = "#1f4e79"
LIGHT = "#dce9f7"
GRAY = "#f2f2f2"


def actor(ax, x, y, label, scale=1.0):
    s = scale
    ax.add_patch(Circle((x, y + 1.15 * s), 0.26 * s, fc="white", ec=BLUE, lw=1.6, zorder=3))
    ax.plot([x, x], [y + 0.88 * s, y + 0.05 * s], color=BLUE, lw=1.6, zorder=3)
    ax.plot([x - 0.55 * s, x + 0.55 * s], [y + 0.62 * s, y + 0.62 * s], color=BLUE, lw=1.6, zorder=3)
    ax.plot([x, x - 0.42 * s], [y + 0.05 * s, y - 0.55 * s], color=BLUE, lw=1.6, zorder=3)
    ax.plot([x, x + 0.42 * s], [y + 0.05 * s, y - 0.55 * s], color=BLUE, lw=1.6, zorder=3)
    ax.text(x, y - 0.95 * s, label, ha="center", va="top", fontsize=10.5, color=BLUE)


def usecase(ax, cx, cy, text, w=4.3, h=0.88):
    ax.add_patch(Ellipse((cx, cy), w, h, fc=LIGHT, ec=BLUE, lw=1.3, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=10, zorder=4)


def usecase_fig(path, actor_label, cases):
    fig, ax = plt.subplots(figsize=(9.0, 5.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((2.35, 0.45), 7.35, 8.85,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec=BLUE, lw=1.4, ls="--", zorder=1))
    ax.text(6.02, 9.5, "风电功率爬坡事件预测与预警系统", ha="center", va="bottom",
            fontsize=10.5, color=BLUE, fontweight="bold")

    actor(ax, 1.15, 4.55, actor_label)

    n = len(cases)
    ys = [7.95 - i * (6.9 / max(n - 1, 1)) for i in range(n)]
    for y, t in zip(ys, cases):
        usecase(ax, 6.15, y, t)
        ax.plot([1.72, 4.0], [4.95, y], color=BLUE, lw=1.0, zorder=1)

    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


def module_fig(path):
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((2.45, 8.7), 5.3, 0.9, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=BLUE, ec=BLUE, zorder=2))
    ax.text(5.1, 9.15, "风电功率爬坡事件预测与预警系统", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold", zorder=3)

    mods = [
        ("数据接入与管理", "导入·清洗\n事件清单检索"),
        ("事件识别与归因", "阈值检测·合并\n分级·成因分解"),
        ("预测与概率化", "LightGBM 点预测\nP10–P90·超阈概率"),
        ("预警决策与发布", "代价敏感 τ*·迟滞\n方向·等级·LED"),
        ("可视化与解释", "ECharts 仪表盘\n历史告警·大模型"),
        ("实时采集与联调", "Arduino·串口\n滚动窗·在线推理"),
    ]
    w, h, gap = 1.46, 2.05, 0.19
    x0 = 0.28
    for i, (name, sub) in enumerate(mods):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 4.6), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=LIGHT, ec=BLUE, lw=1.2, zorder=2))
        ax.text(x + w / 2, 6.32, name, ha="center", va="center", fontsize=9.2,
                fontweight="bold", color=BLUE, zorder=3)
        ax.text(x + w / 2, 5.32, sub, ha="center", va="center", fontsize=7.8,
                color="#333333", zorder=3, linespacing=1.5)
        ax.plot([x + w / 2, x + w / 2], [8.15, 4.6 + h + 0.02], color=BLUE, lw=1.0, zorder=1)
    ax.plot([5.1, 5.1], [8.7, 8.15], color=BLUE, lw=1.0, zorder=1)
    ax.plot([x0 + w / 2, x0 + 5 * (w + gap) + w / 2], [8.15, 8.15], color=BLUE, lw=1.0, zorder=1)

    ax.add_patch(FancyBboxPatch((0.9, 1.75), 8.3, 0.78, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=GRAY, ec="#999999", lw=1.0, zorder=2))
    ax.text(5.05, 2.14, "支撑层：SQLite 数据库 ｜ REST 接口 ｜ 统一口径与决策层参数（步数时间基）",
            ha="center", va="center", fontsize=9.4, color="#333333", zorder=3)

    ax.annotate("", xy=(5.05, 2.6), xytext=(5.05, 4.5),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.1))
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


def arch_fig(path):
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    layers = [
        ("表现层", "ECharts 仪表盘（Bootstrap 5）· Streamlit 双模式界面", "#e8f1fb"),
        ("接口层", "REST 接口（/predict /alarms /alarm_explain 等）· 大模型解释（Prompt + 兜底）", "#eef6f0"),
        ("决策层", "τ 代价敏感标定 · 事件合并 · 迟滞 · 三级分级 · 方向判定", "#fdf3e6"),
        ("算法层", "14 维特征 → LightGBM 回归/分类 → 残差概率化 → 校准检验", "#f6ecf6"),
        ("数据层", "清洗与统一口径 → 事件检测与归因 → 滑窗数据集 → SQLite 落库", "#eaf3f6"),
        ("硬件层", "Arduino 采集（演示链路）→ 串口帧解析 → 72 步滚动窗口", "#f0f0f0"),
    ]
    y = 8.6
    for name, desc, color in layers:
        ax.add_patch(FancyBboxPatch((1.55, y - 1.0), 7.9, 1.0, boxstyle="round,pad=0.02,rounding_size=0.07",
                                    fc=color, ec=BLUE, lw=1.1, zorder=2))
        ax.text(1.78, y - 0.5, name, ha="left", va="center", fontsize=10.6,
                fontweight="bold", color=BLUE, zorder=3)
        ax.text(3.05, y - 0.5, desc, ha="left", va="center", fontsize=8.8,
                color="#222222", zorder=3)
        y -= 1.32

    ax.add_patch(FancyArrowPatch((0.95, 2.3), (0.95, 8.6), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.3, color=BLUE))
    ax.text(0.62, 5.4, "数据流与调用方向", rotation=90, ha="center", va="center",
            fontsize=9.2, color=BLUE)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


def route_fig(path):
    """技术路线图：单列流程 + 左侧阶段标注。"""
    steps = [
        ("SDWPF 公开数据集读取（134 台 × 245 天 × 10 min）", "数据层"),
        ("数据预处理：缺失插值 / 负值置零 / 停机判据 / 按时间划分", "数据层"),
        ("爬坡事件识别与成因归因：Δt=1 h、θ=20%、三级分级、ΔP 分解", "数据层"),
        ("特征工程：14 维特征 → 72 步滑窗 → 378 维展平", "算法层"),
        ("模型实验：LightGBM 主模型｜XGBoost·RF·LR 基线｜LSTM·TCN·级联", "算法层"),
        ("概率化与校准：残差经验分布 → P10–P90、Brier、可靠性图", "算法层"),
        ("代价敏感预警决策：τ* 标定 + 事件合并 + 迟滞 + 三级分级", "决策层"),
        ("落库与后端：SQLite 5 表 + REST 接口 + 在线推理 + 大模型解读", "系统层"),
        ("双模式前端：Bootstrap+ECharts 仪表盘 ／ Streamlit 回放", "系统层"),
        ("系统测试与验证：功能 / 一致性 / 事件级回测 / 提前量敏感性", "验证层"),
        ("毕业设计说明书撰写与答辩", "验证层"),
    ]
    colors = {"数据层": "#eaf3f6", "算法层": "#f6ecf6", "决策层": "#fdf3e6",
              "系统层": "#e8f1fb", "验证层": "#eef6f0"}
    fig, ax = plt.subplots(figsize=(8.2, 11.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.9, 19.7)
    ax.axis("off")

    n = len(steps)
    top = 19.2
    step = 1.62
    h = 1.15
    x0, w = 2.35, 6.9
    y = top
    for i, (text, stage) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x0, y - h), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    fc=colors[stage], ec=BLUE, lw=1.1, zorder=2))
        ax.text(x0 + w / 2, y - h / 2, text, ha="center", va="center", fontsize=8.6,
                zorder=3, linespacing=1.4)
        if i < n - 1:
            ax.annotate("", xy=(x0 + w / 2, y - h - 0.42), xytext=(x0 + w / 2, y - h - 0.02),
                        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.1))
        y -= step

    # 左侧阶段合并标注
    stage_span = {}
    for i, (_, st) in enumerate(steps):
        y_top = top - i * step
        stage_span.setdefault(st, [y_top, y_top - h])
        stage_span[st][1] = y_top - h
    for st, (yt, yb) in stage_span.items():
        ax.plot([2.0, 2.0], [yb, yt], color=BLUE, lw=1.6, solid_capstyle="butt", zorder=1)
        ax.text(1.72, (yt + yb) / 2, st, rotation=90, ha="center", va="center",
                fontsize=9.4, color=BLUE, fontweight="bold")

    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    usecase_fig(os.path.join(OUT, "fig1_usecase_operator.png"), "风场运行值班员", [
        "登录与权限校验",
        "浏览功率曲线与爬坡事件清单",
        "查看上下行概率与预警等级",
        "接收分级预警（页面 / LED）",
        "查询历史告警记录与命中情况",
        "查看预警解读（大模型生成）",
    ])
    usecase_fig(os.path.join(OUT, "fig2_usecase_admin.png"), "系统管理员", [
        "导入数据与重建数据库",
        "维护口径参数与事件清单",
        "配置阈值 τ、概率源与迟滞",
        "调用在线推理接口（/predict）",
        "查看告警统计与命中回填",
        "巡检系统状态与结果导出",
    ])
    module_fig(os.path.join(OUT, "fig3_modules.png"))
    arch_fig(os.path.join(OUT, "fig4_architecture.png"))
    route_fig(os.path.join(OUT, "fig5_route.png"))
