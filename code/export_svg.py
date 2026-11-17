# -*- coding: utf-8 -*-
"""export_svg.py —— 把开题报告的 5 张插图导出为 SVG（矢量）"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import rcParams
rcParams["svg.fonttype"] = "path"      # 文字转矢量轮廓，Word 里不依赖本机字体

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_proposal_figs as m

OUT = os.path.join("output", "figs")
os.makedirs(OUT, exist_ok=True)

m.usecase_fig(os.path.join(OUT, "fig1.svg"), "风场运行值班员", [
    "登录与权限校验",
    "浏览功率曲线与爬坡事件清单",
    "查看上下行概率与预警等级",
    "接收分级预警（页面 / LED）",
    "查询历史告警记录与命中情况",
    "查看预警解读（大模型生成）",
])
m.usecase_fig(os.path.join(OUT, "fig2.svg"), "系统管理员", [
    "导入数据与重建数据库",
    "维护口径参数与事件清单",
    "配置阈值 τ、概率源与迟滞",
    "调用在线推理接口（/predict）",
    "查看告警统计与命中回填",
    "巡检系统状态与结果导出",
])
m.module_fig(os.path.join(OUT, "fig3.svg"))
m.arch_fig(os.path.join(OUT, "fig4.svg"))
m.route_fig(os.path.join(OUT, "fig5.svg"))
