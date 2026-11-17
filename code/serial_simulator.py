# -*- coding: utf-8 -*-
# serial_simulator.py —— 无硬件时的串口数据模拟器(备用, 暂未集成 app.py)
#
# 用 field_power.csv 的真实特征按协议拼帧, 输出到 stdout 或文件,
# 用于验证 serial_parser / serial_reader / reg_infer 整条链路。
#
# 用法:
#   python code/serial_simulator.py --n 200 --out output/_sim_frames.txt
#   # 然后把文件逐行喂给解析器做端到端自检

import argparse
import numpy as np
import pandas as pd

from serial_parser import build_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="生成帧数")
    ap.add_argument("--start", type=int, default=0, help="起始行(按 field_power 顺序)")
    ap.add_argument("--out", type=str, default=None, help="输出文件; 不给则打印到 stdout")
    ap.add_argument("--no-chk", action="store_true", help="不带校验")
    args = ap.parse_args()

    fp = pd.read_csv("output/field_power.csv").sort_values("t_idx").reset_index(drop=True)
    # field_power.csv 实际列名：Wspd_mean / C_mw / P_mw；缺口量用占位。
    def g(row, name, default=0.0):
        return float(row[name]) if name in row and pd.notna(row[name]) else default

    lines = []
    for i in range(args.start, min(args.start + args.n, len(fp))):
        row = fp.iloc[i]
        wspd = g(row, "Wspd_mean")   # 实际列名为 Wspd_mean
        wdir = g(row, "dir", 0.0) if "dir" in row and not isinstance(row.get("dir"), str) else 180.0
        etmp = g(row, "Etmp", 25.0)
        itmp = g(row, "Itmp", 40.0)
        ndir = wdir
        pab1 = g(row, "Pab1", 10.0); pab2 = g(row, "Pab2", 10.0); pab3 = g(row, "Pab3", 10.0)
        prtv = g(row, "Prtv", 0.0)
        cmw = g(row, "C_mw", 201.0)
        pmw = g(row, "P_mw", 0.0)
        lines.append(build_frame(wspd, wdir, etmp, itmp, ndir, pab1, pab2, pab3, prtv, cmw, pmw,
                                 with_chk=not args.no_chk))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"[sim] 写出 {len(lines)} 帧 -> {args.out}")
    else:
        for l in lines:
            print(l, end="")


if __name__ == "__main__":
    main()
