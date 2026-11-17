# -*- coding: utf-8 -*-
# reg_infer.py —— 回归模型在线推理底座(纯软件, 不依赖硬件)
# 加载 output/reg_lightgbm_model.pkl (train_reg_lightgbm.py 导出),
# 对新窗口特征 X (N, T, F) 计算 P_hat(t+H) 与 P(dP > theta) 概率。
# 供未来实时场景(串口/模拟器)调用; 当前 app.py 仍用回放模拟器。

import os
import numpy as np
import joblib

OUT_DIR = "output"
MODEL_PKL = os.path.join(OUT_DIR, "reg_lightgbm_model.pkl")

_cache = {"bundle": None}


def load_model(path=MODEL_PKL):
    if _cache["bundle"] is None:
        _cache["bundle"] = joblib.load(path)
    return _cache["bundle"]


def flatten_with_stats(X, keep_last=24):
    # 与训练完全一致的展平: 末 keep_last 步 + 均值/标准差/斜率
    N, T, F = X.shape
    tail = X[:, -keep_last:, :].reshape(N, keep_last * F)
    mu = X.mean(axis=1)
    sd = X.std(axis=1)
    slope = X[:, 0, :] - X[:, -1, :]
    return np.concatenate([tail, mu, sd, slope], axis=1).astype(np.float32)


def predict_power(X, bundle=None):
    # 返回 P_hat(t+H), 单位 MW。X: (N, T, F) 已归一化特征
    b = bundle or load_model()
    Feat = flatten_with_stats(X, b["keep_last"])
    return b["booster"].predict(Feat, num_iteration=b["best_iteration"])


def _corrected_resid_sorted(b):
    # 与 1b 完全一致: e_corr = e - mean(e) = e - bias, 再排序。
    if "_resid_corr" not in b:
        b["_resid_corr"] = np.sort(
            (b["resid_val_sorted"].astype(np.float64) - b["resid_bias"])
        )
    return b["_resid_corr"]


def prob_exceed(dP_hat, bundle=None, theta=None, direction="up"):
    # 方式一(经验分布, 非正态): P(dP > theta) 或 P(dP < -theta)
    # 用 1b 同口径的偏置校正残差 e_corr = e - mean(e), 保证与 reg_prob.npz 一致。
    b = bundle or load_model()
    if theta is None:
        theta = b["theta_default"]
    resid_sorted = _corrected_resid_sorted(b)
    n = len(resid_sorted)
    dP_hat = np.asarray(dP_hat, dtype=float)
    if direction == "up":
        cnt = np.searchsorted(resid_sorted, theta - dP_hat, side="right")
        return 1.0 - cnt / n
    else:
        cnt = np.searchsorted(resid_sorted, -theta - dP_hat, side="right")
        return cnt / n


def infer(X_current, P_now, bundle=None, theta=None):
    # 一站式: 给当前窗口 X_current(N,T,F) 与当前功率 P_now(N,), 返回 dict
    # dP_hat = P_hat(t+H) - P_now
    b = bundle or load_model()
    P_hat = predict_power(X_current, b)
    P_now = np.asarray(P_now, dtype=float)
    dP_hat = P_hat - P_now
    p_up = prob_exceed(dP_hat, b, theta=theta, direction="up")
    p_dn = prob_exceed(dP_hat, b, theta=theta, direction="down")
    return {"P_hat": P_hat, "dP_hat": dP_hat, "p_up": p_up, "p_dn": p_dn}


if __name__ == "__main__":
    # 自检: 用 dataset.npz 的 X_test 前 2000 样本推理, 与 reg_pred.npz.pred_test 对比
    dz = np.load(os.path.join(OUT_DIR, "dataset.npz"), allow_pickle=True)
    X_test = dz["X_test"][:2000]
    ref = np.load(os.path.join(OUT_DIR, "reg_pred.npz"))["pred_test"][:2000]
    b = load_model()
    pred = predict_power(X_test, b)
    diff = np.abs(pred - ref)
    print("[reg_infer selfcheck] n=%d  max|diff|=%.6f  mean|diff|=%.6f" % (len(pred), diff.max(), diff.mean()))
    print("  best_iter=%d keep_last=%d h_steps=%d bias=%.4f" % (
        b["best_iteration"], b["keep_last"], b["h_steps"], b["resid_bias"]))
    print("  一致性:", "PASS" if diff.max() < 1e-3 else "CHECK")
