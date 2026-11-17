# -*- coding: utf-8 -*-
"""Dump a DOCX into a readable plain-text structure (paragraphs + tables)."""
import sys
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn


def iter_block_items(parent):
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)


def main(path):
    doc = Document(path)
    out = []
    pi = 0
    ti = 0
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            pi += 1
            text = block.text
            style = block.style.name if block.style is not None else ''
            fmt = block.paragraph_format
            align = fmt.alignment
            runs = [(r.text, r.bold, r.font.size.pt if r.font.size else None,
                     r.font.name) for r in block.runs if r.text]
            out.append(f"[P{pi:03d}][{style}][align={align}] {text!r}")
            if runs:
                out.append(f"      runs: {runs}")
        else:
            ti += 1
            out.append(f"[TABLE {ti}] rows={len(block.rows)} cols={len(block.columns)}")
            for ri, row in enumerate(block.rows):
                cells = []
                seen = set()
                for c in row.cells:
                    if id(c._tc) in seen:
                        continue
                    seen.add(id(c._tc))
                    cells.append(c.text.replace('\n', '\\n'))
                out.append(f"   R{ri}: " + " | ".join(cells))
    print("\n".join(out))
    print("\n--- SECTIONS ---")
    for s in doc.sections:
        print(s.page_width, s.page_height, s.left_margin, s.right_margin,
              s.top_margin, s.bottom_margin)


if __name__ == '__main__':
    main(sys.argv[1])
