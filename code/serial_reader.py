# -*- coding: utf-8 -*-
# serial_reader.py —— 串口读取 -> 解析 -> 滚动窗口 -> 在线推理(备用, 暂未集成 app.py)
#
# 用法(硬件到货后):
#   python code/serial_reader.py --port COM3 --baud 115200
# 依赖: pyserial (pip install pyserial)
#
# 若暂无硬件, 可用 serial_simulator.py 产生虚拟串口/文件流测试解析链路。

import argparse
import sys
import numpy as np

from serial_parser import parse_frame, RollingWindow
import reg_infer


def load_norm_stats():
    dz = np.load("output/dataset.npz", allow_pickle=True)
    return dz["feat_mu"].astype(np.float32), dz["feat_sd"].astype(np.float32)


def run(port, baud, theta, T=72):
    try:
        import serial
    except ImportError:
        print("[serial_reader] 未安装 pyserial, 请先 pip install pyserial")
        return

    feat_mu, feat_sd = load_norm_stats()
    rw = RollingWindow(T=T)
    bundle = reg_infer.load_model()
    ser = serial.Serial(port, baud, timeout=2)
    print(f"[serial_reader] 打开 {port} @ {baud}, 等待窗口填满({T} 帧)...")
    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore")
            if not line or line.startswith("#"):
                continue
            feats, err = parse_frame(line)
            if err:
                print(f"[skip] {err}: {line.strip()}")
                continue
            rw.push(feats)
            if not rw.ready():
                continue
            X = rw.normalized_window(feat_mu, feat_sd)
            P_now = feats["P_mw"]
            out = reg_infer.infer(X, np.array([P_now]), bundle=bundle, theta=theta)
            print(
                f"P_now={P_now:.1f}MW  P_hat(t+H)={out['P_hat'][0]:.1f}MW  "
                f"dP_hat={out['dP_hat'][0]:+.1f}MW  "
                f"P_up={out['p_up'][0]:.3f}  P_dn={out['p_dn'][0]:.3f}"
            )
    except KeyboardInterrupt:
        print("\n[serial_reader] 用户中断")
    finally:
        ser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="如 COM3")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--theta", type=float, default=40.2)
    ap.add_argument("--T", type=int, default=72)
    args = ap.parse_args()
    run(args.port, args.baud, args.theta, args.T)


if __name__ == "__main__":
    main()
