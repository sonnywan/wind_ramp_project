# -*- coding: utf-8 -*-
"""
led_bridge.py —— 单片机 LED 亮灭桥接（串口指令下发）
=====================================================================
任务书闭环："前端输入(手动/传感器) → 后端判断 → 单片机 LED 亮"。

协议（向下行，PC → Arduino，一行一帧，\n 结尾）：
    LED:ON\n    点亮预警灯
    LED:OFF\n   熄灭
    LED:BLINK\n  闪烁（可选，表示事件进行中）
Arduino 端按行读取并控制对应引脚即可。

设计：
  - 无硬件 / 串口打开失败时自动降级为"虚拟 LED"，仅在日志中记录状态，
    不影响整条判断链路演示（前端网页红灯仍亮）。
  - 通过环境变量或参数指定串口：LED_SERIAL_PORT=COM3, 波特率默认 9600。
"""
import os

_state = {"ser": None, "enabled": False, "port": None}


def init(port=None, baud=9600):
    """尝试打开串口；失败则降级为虚拟 LED（返回 False）。"""
    port = port or os.environ.get("LED_SERIAL_PORT")
    if not port:
        print("[led] 未配置串口（LED_SERIAL_PORT），使用虚拟 LED")
        return False
    try:
        import serial  # pyserial
        _state["ser"] = serial.Serial(port, baud, timeout=1)
        _state["enabled"] = True
        _state["port"] = port
        print(f"[led] 串口已连接 {port}@{baud}")
        return True
    except Exception as e:
        print(f"[led] 串口打开失败（{e}），降级虚拟 LED")
        _state["enabled"] = False
        return False


def _send(cmd):
    if _state["enabled"] and _state["ser"] is not None:
        try:
            _state["ser"].write((cmd + "\n").encode("ascii"))
            _state["ser"].flush()
            return True
        except Exception as e:
            print(f"[led] 发送失败：{e}，降级虚拟 LED")
            _state["enabled"] = False
    return False


def set_led(on):
    """点亮/熄灭 LED；返回是否真实下发（False=虚拟）。"""
    cmd = "LED:ON" if on else "LED:OFF"
    real = _send(cmd)
    print(f"[led] {'→ ' + cmd if real else '（虚拟）' + cmd}")
    return real


def close():
    if _state["ser"] is not None:
        try:
            _state["ser"].close()
        except Exception:
            pass
    _state["ser"] = None
    _state["enabled"] = False


def is_real():
    return _state["enabled"]
