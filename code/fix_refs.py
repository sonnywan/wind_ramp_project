# -*- coding: utf-8 -*-
"""fix_refs.py —— 按已核实信息补全开题报告参考文献（学位论文加学校、补年份、改正文献类型）

只改第 11 行“八、参考文献”里指定的几条，其他段落、图表、表格一律不动。
输出到工作区，桌面原件不覆盖。
"""
import os
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = r"C:\Users\21912\Desktop\开题报告_王阳_风电功率爬坡事件预测与预警系统设计与实现(4)_矢量图(2).docx"
OUT = "开题报告_王阳_风电功率爬坡事件预测与预警系统设计与实现(4)_文献补全.docx"

# 序号 -> 新条目（序号即第 11 行里的第 n 段，标题占第 0 段）
FIX = {
    3: "严佳丽. 人工智能驱动的风电功率预测及其爬坡预测研究[D]. 中国矿业大学, 2025.",
    4: "刘雅婷. 考虑转折性天气的日前风电功率预测及爬坡事件检测方法研究[J]. 电力科学与技术学报, 2023.",
    5: "李凯. 考虑相似风电爬坡事件的风电功率修正预测[D]. 内蒙古工业大学, 2024.",
    6: "马易晨. 考虑爬坡事件与误差修正的短期风电功率集成预测方法研究[D]. 西安理工大学, 2025.",
    7: "许祖昕. 考虑功率爬坡事件的短期风电出力预测研究[D]. 南昌大学, 2025.",
    8: "王玮琨. 计及风电功率随机波动与爬坡事件的电力系统运行风险评估[D]. 重庆大学, 2025.",
    10: "熊予涵. 基于深度学习的风电爬坡事件及低出力事件预测识别研究[D]. 华中科技大学, 2024.",
    11: "刘铖铖. 基于深度学习的短期风电功率区间和爬坡预测方法研究[D]. 南京信息工程大学, 2025.",
    15: "董科韬. 风电机组功率爬坡预测方法研究[D]. 华北电力大学, 2025.",
}


def set_text(par_el, text):
    runs = par_el.findall(qn("w:r"))
    keep = runs[0]
    for r in runs[1:]:
        par_el.remove(r)
    for child in list(keep):
        if child.tag != qn("w:rPr"):
            keep.remove(child)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    keep.append(t)


def main():
    doc = Document(SRC)
    cell = doc.tables[0].rows[11].cells[0]
    paras = cell.paragraphs
    assert paras[0].text.startswith("八、参考文献"), paras[0].text[:20]
    for n, text in FIX.items():
        old = paras[n].text
        set_text(paras[n]._p, "[%d] %s" % (n, text))
        print("[%d] %s\n    -> %s" % (n, old[:46], paras[n].text[:46]))
    doc.save(OUT)
    print("\nsaved", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
