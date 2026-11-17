# -*- coding: utf-8 -*-
"""redesign_schedule.py —— 重排开题报告的“工作进度安排”

在你这版（桌面 (4)_矢量图(2)）上只替换第 12 行“工作进度安排”的条目，
其余段落、图表、表格、参考文献一律不动；输出到工作区，不覆盖桌面文件。
"""
import copy
import os
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = r"C:\Users\21912\Desktop\开题报告_王阳_风电功率爬坡事件预测与预警系统设计与实现(4)_矢量图(2).docx"
OUT = "开题报告_王阳_风电功率爬坡事件预测与预警系统设计与实现(4)_进度7条.docx"

# 按任务书 5.2 的 7 条编排；日期由任务书起止（2027-03-02 至 2027-06-05，共 14 周）推算
SCHEDULE = [
    "1)第1周（2027年3月2日—3月8日）：调研、学习、查询资料。",
    "2)第2周（2027年3月9日—3月15日）：需求分析与设计，提交开题报告。",
    "3)第3周—第7周（2027年3月16日—4月19日）：模型研究，原型系统开发、测试、调试。",
    "4)第8周（2027年4月20日—4月26日）：中期检查。",
    "5)第9周—第12周（2027年4月27日—5月24日）：完善系统，测试调试，撰写毕业设计说明书。",
    "6)第13周（2027年5月25日—5月31日）：指导教师、评阅人审阅毕业设计说明书。",
    "7)第14周（2027年6月1日—6月5日）：组织答辩、提交说明书终稿。",
]


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
    table = doc.tables[0]
    cell = table.rows[12].cells[0]
    paras = cell.paragraphs
    assert paras[0].text.startswith("工作进度安排"), paras[0].text[:20]
    proto = copy.deepcopy(paras[1]._p)          # 取原条目格式作模板
    for p in paras[1:]:
        p._p.getparent().remove(p._p)
    anchor = paras[0]._p
    for line in SCHEDULE:
        el = copy.deepcopy(proto)
        set_text(el, line)
        anchor.addnext(el)
        anchor = el
    doc.save(OUT)
    print("saved", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
