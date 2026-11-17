# -*- coding: utf-8 -*-
# serial_parser.py —— Arduino 串口数据解析(备用模块, 暂未集成 app.py)
#
# 用途: 硬件到货后, 从串口读取 Arduino 上传的风速/风向/功率等, 拼成模型输入窗口。
#
# === Arduino -> PC 文本帧协议 (一行一帧, 逗号分隔, 换行结束) ===
#   帧格式: "$W,<Wspd>,<Wdir>,<Etmp>,<Itmp>,<Ndir>,<Pab1>,<Pab2>,<Pab3>,<Prtv>,<C_mw>,<P_mw>*<chk>\n"
#   - $W            : 帧头
#   - Wspd          : 风速 m/s
#   - Wdir          : 风向 deg (0-360)
#   - Etmp/Itmp     : 环境/机舱温度 (无传感可填 NAN)
#   - Ndir          : 机舱方向 deg
#   - Pab1/2/3      : 三相叶片角度 deg
#   - Prtv          : 变桨角 deg
#   - C_mw          : 可用容量 MW
#   - P_mw          : 当前有功 MW
#   - *chk          : 异或校验(可选; 为 "XX" 十六进制, 对 $ 与 * 之间字符逐字节异或)
#   解析器将派生: Wdir_sin/cos, Ndir_sin/cos, R(= P_mw / C_mw)
#
# 输出: 每帧一个 dict, 字段与 dataset.npz 的 feature_cols (14 维) 对齐。

import math
import numpy as np


def _xor_checksum(payload: str) -> int:
    c = 0
    for ch in payload.encode("utf-8"):
        c ^= ch
    return c


def parse_frame(line: str, verify_checksum=True):
    # 解析一行帧; 成功返回 dict, 失败返回 (None, 原因)
    line = line.strip()
    if not line or not line.startswith("$W"):
        return None, "非 $W 帧"
    body = line
    # 校验
    if "*" in body:
        body, chk_hex = body.rsplit("*", 1)
        if verify_checksum:
            try:
                want = int(chk_hex, 16)
            except ValueError:
                return None, "校验字段非十六进制"
            got = _xor_checksum(body[1:])  # 去掉 $
            if got != want:
                return None, "校验失败"
    parts = body.split(",")
    # parts[0]='$W'
    if len(parts) < 12:
        return None, "字段不足"
    try:
        vals = [float(x) if x.upper() != "NAN" else float("nan") for x in parts[1:12]]
    except ValueError:
        return None, "数值解析失败"
    (Wspd, Wdir, Etmp, Itmp, Ndir, Pab1, Pab2, Pab3, Prtv, C_mw, P_mw) = vals
    wdir_rad = math.radians(Wdir if not math.isnan(Wdir) else 0.0)
    ndir_rad = math.radians(Ndir if not math.isnan(Ndir) else 0.0)
    R = (P_mw / C_mw) if (C_mw and not math.isnan(C_mw) and C_mw > 0) else float("nan")
    feats = {
        "Wspd": Wspd,
        "Wdir_sin": math.sin(wdir_rad),
        "Wdir_cos": math.cos(wdir_rad),
        "Etmp": Etmp,
        "Itmp": Itmp,
        "Ndir_sin": math.sin(ndir_rad),
        "Ndir_cos": math.cos(ndir_rad),
        "Pab1": Pab1,
        "Pab2": Pab2,
        "Pab3": Pab3,
        "Prtv": Prtv,
        "C_mw": C_mw,
        "P_mw": P_mw,
        "R": R,
    }
    return feats, None


class RollingWindow:
    # 维护最近 T 步原始特征(未归一化), 用于拼模型输入
    def __init__(self, T=72, feature_order=None):
        self.T = T
        self.feature_order = feature_order or [
            "Wspd", "Wdir_sin", "Wdir_cos", "Etmp", "Itmp",
            "Ndir_sin", "Ndir_cos", "Pab1", "Pab2", "Pab3",
            "Prtv", "C_mw", "P_mw", "R",
        ]
        self.buf = []

    def push(self, feats):
        row = [feats.get(k, float("nan")) for k in self.feature_order]
        self.buf.append(row)
        if len(self.buf) > self.T:
            self.buf.pop(0)

    def ready(self):
        return len(self.buf) >= self.T

    def as_array(self):
        # 返回 (T, 14) 原始特征数组, 不足则 None
        if not self.ready():
            return None
        return np.array(self.buf[-self.T:], dtype=np.float32)

    def normalized_window(self, feat_mu, feat_sd):
        # 用训练集 mu/sd 归一化 (与 dataset.npz 一致), 返回 (1, T, 14)
        arr = self.as_array()
        if arr is None:
            return None
        norm = (arr - feat_mu) / (feat_sd + 1e-9)
        return norm[None, :, :].astype(np.float32)


def build_frame(Wspd, Wdir, Etmp, Itmp, Ndir, Pab1, Pab2, Pab3, Prtv, C_mw, P_mw, with_chk=True):
    # 反向工具: 按协议拼一帧(供 Arduino 模拟器/自检使用)
    body = "$W,%.3f,%.2f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f" % (
        Wspd, Wdir, Etmp, Itmp, Ndir, Pab1, Pab2, Pab3, Prtv, C_mw, P_mw)
    if with_chk:
        chk = _xor_checksum(body[1:])
        return body + "*%02X\n" % chk
    return body + "\n"


if __name__ == "__main__":
    # 自检: 拼帧 -> 解析 -> 校验
    frame = build_frame(8.5, 135.0, 25.0, 40.0, 130.0, 10.0, 11.0, 12.0, 5.0, 201.0, 88.5)
    print("frame:", frame.strip())
    feats, err = parse_frame(frame)
    print("parse err:", err)
    print("feats:", {k: round(v, 4) if isinstance(v, float) else v for k, v in feats.items()})
    # 校验篡改检测(改变单个字符, 避免 XOR 等量抵消)
    bad = frame.replace("135.00", "136.00")
    _, err2 = parse_frame(bad)
    print("tamper detect(? 期望报错):", err2)
    rw = RollingWindow(T=3)
    for _ in range(3):
        rw.push(feats)
    print("window ready:", rw.ready(), "shape:", rw.as_array().shape)
