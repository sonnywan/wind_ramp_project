# -*- coding: utf-8 -*-
"""to_vector_images.py —— 把开题报告里的 5 张 PNG 插图换成 SVG 矢量图

做法（Word 2016+ 原生支持）：
  每张图同时保留原 PNG 作为回退底图，并在 <a:blip> 里追加 svgBlip 扩展，
  指向新增的 SVG 图像部件。Word 2016 及以上按 SVG 矢量渲染（无限放大不虚），
  旧版 Word / 部分 WPS 自动退回 PNG，显示效果与现在一致。

只改图片引用，不动文字、表格、样式、图片尺寸（必要时按 SVG 比例微调高度避免拉伸）。
输出：开题报告_王阳_...(4)_矢量图.docx（原文件保持不动）
"""
import os
import re
import sys
import hashlib
from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import qn, nsdecls
from docx.opc.part import Part
from docx.opc.packuri import PackURI
from docx.opc.constants import RELATIONSHIP_TYPE as RT

SRC = "开题报告_王阳_风电功率爬坡事件预测与预警系统设计与实现(4).docx"
OUT = "开题报告_王阳_风电功率爬坡事件预测与预警系统设计与实现(4)_矢量图.docx"
FIGDIR = os.path.join("output", "figs")
SVGS = [os.path.join(FIGDIR, f"fig{i}.svg") for i in range(1, 6)]
SVG_NS = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"
BLIP_EXT_URI = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"


def svg_size(path):
    """读 SVG 的 width/height（pt），返回 (w, h)。"""
    head = open(path, "r", encoding="utf-8").read(2000)
    w = re.search(r'width="([\d.]+)pt"', head)
    h = re.search(r'height="([\d.]+)pt"', head)
    return float(w.group(1)), float(h.group(1))


def main():
    doc = Document(SRC)
    blips = doc.element.body.findall(".//" + qn("a:blip"))
    # 只取图片关系（排除页眉页脚等），按文档顺序
    inlines = doc.element.body.findall(".//" + qn("wp:inline"))
    print("inline shapes:", len(inlines), "blips:", len(blips))
    assert len(inlines) == 5, len(inlines)

    for i, inline in enumerate(inlines):
        blip = inline.find(".//" + qn("a:blip"))
        rid_png = blip.get(qn("r:embed"))
        png_part = doc.part.related_parts[rid_png]
        png_bytes = png_part.blob
        print("图%d: 原图 %s, %d bytes" % (i + 1, png_part.partname, len(png_bytes)))

        svg_path = SVGS[i]
        svg_bytes = open(svg_path, "rb").read()
        sw, sh = svg_size(svg_path)

        # 1) 新增 SVG 图像部件并建立关系
        partname = "/word/media/proposal_fig%d.svg" % (i + 1)
        svg_part = Part(PackURI(partname), "image/svg+xml", blob=svg_bytes, package=doc.part.package)
        rid_svg = doc.part.relate_to(svg_part, RT.IMAGE)

        # 2) 在 blip 上追加 svgBlip 扩展（PNG 作为回退）
        ext = parse_xml(
            '<a:extLst %s><a:ext uri="%s">'
            '<asvg:svgBlip xmlns:asvg="%s" r:embed="%s"/>'
            "</a:ext></a:extLst>" % (nsdecls("a", "r"), BLIP_EXT_URI, SVG_NS, rid_svg)
        )
        for old in blip.findall(qn("a:extLst")):
            blip.remove(old)
        blip.append(ext)

        # 3) 按 SVG 比例校正高度，避免拉伸
        extent = inline.find(".//" + qn("wp:extent"))
        cx, cy = int(extent.get("cx")), int(extent.get("cy"))
        new_cy = int(round(cx * sh / sw))
        if abs(new_cy - cy) / cy > 0.002:
            print("   高度校正: %.2fcm -> %.2fcm" % (cy / 360000, new_cy / 360000))
            extent.set("cy", str(new_cy))
            a_ext = inline.find(".//" + qn("a:ext"))
            if a_ext is not None:
                a_ext.set("cy", str(new_cy))
        print("   已加矢量层 -> %s，尺寸 %.2f x %.2f cm" % (partname, cx / 360000, new_cy / 360000))

    doc.save(OUT)
    print("saved", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
