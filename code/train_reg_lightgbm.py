# -*- coding: utf-8 -*-
"""
创新点2 第一步 (1a): LightGBM 回归模型训练
预测 t+H 时刻的场站总有功功率 P_mw (MW, 原始量纲, 不归一化)

因 lgb_pred.npz 只存了3分类概率, 无法做 残差经验分布,
故单独训练一个回归模型, 输出点预测 + 残差统计, 供 1b 做区间/概率校准.

用法: python code/train_reg_lightgbm.py
输出: output/reg_lightgbm_report.md, output/reg_pred.npz
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- 配置 ----------------
OUT_DIR = "output"
NPZ = os.path.join(OUT_DIR, "dataset.npz")
FIELD_CSV = os.path.join(OUT_DIR, "field_power.csv")
REPORT_MD = os.path.join(OUT_DIR, "reg_lightgbm_report.md")
PRED_NPZ = os.path.join(OUT_DIR, "reg_pred.npz")
MODEL_PKL = os.path.join(OUT_DIR, "reg_lightgbm_model.pkl")

H_STEPS = 12      # 预测步长: 12 步 = 2h
KEEP_LAST = 24    # flatten 保留最后 24 步

RANDOM_SEED = 42


def log(msg):
    print(msg, flush=True)


# ---------------- 特征展平 ----------------
def flatten_with_stats(X, keep_last=24):
    """
    X: (N, T, F)
    输出: (N, keep_last*F + F*3)
      - 最后 keep_last 步展平
      - 每个特征: 均值 / 标准差 / (首-末)斜率
    """
    N, T, F = X.shape
    tail = X[:, -keep_last:, :].reshape(N, keep_last * F)
    mu = X.mean(axis=1)          # (N, F)
    sd = X.std(axis=1)           # (N, F)
    slope = X[:, 0, :] - X[:, -1, :]  # (N, F) 首发末
    return np.concatenate([tail, mu, sd, slope], axis=1).astype(np.float32)


# ---------------- 回归标签 ----------------
def build_reg_labels(t_idx, P_series, is_gap_series, n_total):
    """
    t_idx: (N,) 每个样本对应的当前时刻索引
    P_series: (n_total,) 场站总有功 MW
    is_gap_series: (n_total,) 0/1 缺口标记
    y[i] = P_series[t+H], 若 t+H 存在 且 (t+1..t+H) 无 gap 且值有限
    """
    N = len(t_idx)
    y = np.full(N, np.nan, dtype=np.float32)
    valid = np.zeros(N, dtype=bool)
    for i, t in enumerate(t_idx):
        tf = t + H_STEPS
        if tf >= n_total:
            continue
        # 区间 (t, tf] 内不得有 gap
        seg = is_gap_series[t + 1: tf + 1]
        if seg.sum() > 0:
            continue
        val = P_series[tf]
        if not np.isfinite(val):
            continue
        y[i] = val
        valid[i] = True
    return y, valid


def eval_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    log(f"[{name}] n={len(y_true)}  MAE={mae:.3f} MW  RMSE={rmse:.3f} MW  R2={r2:.4f}")
    return dict(name=name, n=int(len(y_true)), mae=float(mae), rmse=rmse, r2=float(r2))


def main():
    log("=" * 60)
    log("LightGBM 回归 (点预测) 训练 - 创新点2 子步 1a")
    log("=" * 60)

    # ---- 读数据集 ----
    dz = np.load(NPZ, allow_pickle=True)
    X_train = dz["X_train"]; y_train_cls = dz["y_train"]; t_train = dz["t_train"]
    X_val = dz["X_val"];     y_val_cls = dz["y_val"];     t_val = dz["t_val"]
    X_test = dz["X_test"];   y_test_cls = dz["y_test"];   t_test = dz["t_test"]
    feature_cols = list(dz["feature_cols"]) if "feature_cols" in dz else []
    log(f"数据集: train={X_train.shape} val={X_val.shape} test={X_test.shape}")
    log(f"特征数={len(feature_cols)}  H_STEPS={H_STEPS}")

    # ---- 读场站功率序列 ----
    fp = pd.read_csv(FIELD_CSV)
    fp = fp.sort_values("t_idx").reset_index(drop=True)
    n_total = len(fp)
    # 校验 t_idx 连续
    assert np.array_equal(fp["t_idx"].values, np.arange(n_total)), "t_idx 不连续!"
    P_series = fp["P_mw"].values.astype(np.float64)
    is_gap_series = fp["is_gap"].values.astype(int)
    log(f"field_power: n_total={n_total}  P范围=[{np.nanmin(P_series):.1f}, {np.nanmax(P_series):.1f}] MW")

    # ---- 构建回归标签 ----
    y_train, v_train = build_reg_labels(t_train, P_series, is_gap_series, n_total)
    y_val,   v_val   = build_reg_labels(t_val,   P_series, is_gap_series, n_total)
    y_test,  v_test  = build_reg_labels(t_test,  P_series, is_gap_series, n_total)
    log(f"有效回归样本: train={v_train.sum()}/{len(v_train)} "
        f"val={v_val.sum()}/{len(v_val)} test={v_test.sum()}/{len(v_test)}")

    # ---- 特征展平 ----
    Ftr = flatten_with_stats(X_train, KEEP_LAST)
    Fva = flatten_with_stats(X_val, KEEP_LAST)
    Fte = flatten_with_stats(X_test, KEEP_LAST)
    log(f"展平后特征维度: {Ftr.shape[1]}")

    # ---- 取有效样本 ----
    Xtr, ytr, ttr = Ftr[v_train], y_train[v_train], t_train[v_train]
    Xva, yva, tva = Fva[v_val],   y_val[v_val],   t_val[v_val]
    Xte, yte, tte = Fte[v_test],  y_test[v_test],  t_test[v_test]

    # ---- 训练 ----
    params = dict(
        objective="regression",
        metric="l2",
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=60,
        feature_fraction=0.5,
        bagging_fraction=0.8,
        bagging_freq=5,
        lambda_l1=0.5,
        lambda_l2=2.0,
        verbose=-1,
        seed=RANDOM_SEED,
    )
    dtr = lgb.Dataset(Xtr, label=ytr)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    log("开始训练 LightGBM 回归 ...")
    booster = lgb.train(
        params, dtr,
        num_boost_round=1500,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)],
    )
    log(f"训练完成, best_iter={booster.best_iteration}")

    # ---- 预测 ----
    p_tr = booster.predict(Xtr, num_iteration=booster.best_iteration)
    p_va = booster.predict(Xva, num_iteration=booster.best_iteration)
    p_te = booster.predict(Xte, num_iteration=booster.best_iteration)

    m_tr = eval_metrics(ytr, p_tr, "train")
    m_va = eval_metrics(yva, p_va, "val")
    m_te = eval_metrics(yte, p_te, "test")

    # ---- 残差 ----
    e_tr = ytr - p_tr
    e_va = yva - p_va
    e_te = yte - p_te

    def resid_stats(e, name):
        return dict(
            name=name,
            mean=float(np.mean(e)),
            std=float(np.std(e)),
            p05=float(np.percentile(e, 5)),
            p50=float(np.percentile(e, 50)),
            p95=float(np.percentile(e, 95)),
            n=int(len(e)),
        )
    rs_tr = resid_stats(e_tr, "train")
    rs_va = resid_stats(e_va, "val")
    rs_te = resid_stats(e_te, "test")
    log(f"[resid train] mean={rs_tr['mean']:.3f} std={rs_tr['std']:.3f} "
        f"P05={rs_tr['p05']:.3f} P50={rs_tr['p50']:.3f} P95={rs_tr['p95']:.3f}")
    log(f"[resid val]   mean={rs_va['mean']:.3f} std={rs_va['std']:.3f} "
        f"P05={rs_va['p05']:.3f} P50={rs_va['p50']:.3f} P95={rs_va['p95']:.3f}")
    log(f"[resid test]  mean={rs_te['mean']:.3f} std={rs_te['std']:.3f} "
        f"P05={rs_te['p05']:.3f} P50={rs_te['p50']:.3f} P95={rs_te['p95']:.3f}")

    # ---- 保存 npz ----
    np.savez_compressed(
        PRED_NPZ,
        pred_train=p_tr.astype(np.float32), y_reg_train=ytr.astype(np.float32), t_train=ttr.astype(np.int64),
        pred_val=p_va.astype(np.float32),   y_reg_val=yva.astype(np.float32),   t_val=tva.astype(np.int64),
        pred_test=p_te.astype(np.float32),  y_reg_test=yte.astype(np.float32),  t_test=tte.astype(np.int64),
        resid_val=e_va.astype(np.float32),
        resid_test=e_te.astype(np.float32),
        h_steps=np.int64(H_STEPS),
    )
    log(f"已保存: {PRED_NPZ}")

    # ---- 写报告 ----
    lines = []
    lines.append("# LightGBM 回归 (点预测) 报告 — 创新点2 子步 1a\n")
    lines.append(f"- 预测目标: 场站总有功 P_mw(t+{H_STEPS})，单位 MW (原始量纲)")
    lines.append(f"- 特征: 复用 dataset.npz 归一化特征，flatten(keep_last={KEEP_LAST})+均值/标准差/斜率 → {Ftr.shape[1]} 维")
    lines.append(f"- 训练超参: lr=0.03, leaves=31, depth=6, min_child=60, ff=0.5, bag=0.8/5, L1=0.5, L2=2.0, rounds≤1500, early_stop=80")
    lines.append(f"- best_iter={booster.best_iteration}\n")

    lines.append("## 1. 点预测精度\n")
    lines.append("| 集合 | 样本数 | MAE (MW) | RMSE (MW) | R² |")
    lines.append("|---|---|---|---|---|")
    for m in (m_tr, m_va, m_te):
        lines.append(f"| {m['name']} | {m['n']} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['r2']:.4f} |")
    lines.append("")

    lines.append("## 2. 残差统计 e = P_obs − P̂ (MW)\n")
    lines.append("| 集合 | 样本数 | 均值 | 标准差 | P05 | P50 | P95 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in (rs_tr, rs_va, rs_te):
        lines.append(f"| {r['name']} | {r['n']} | {r['mean']:.3f} | {r['std']:.3f} | "
                     f"{r['p05']:.3f} | {r['p50']:.3f} | {r['p95']:.3f} |")
    lines.append("")
    lines.append("> 说明: 验证集残差经验分布将用于子步 1b 的 P10/P50/P90 区间与 P(ΔP>θ) 概率。")
    lines.append("> 若验证集残差均值接近 0 且分布近似对称，可直接用分位数平移构造区间。\n")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"已写报告: {REPORT_MD}")

    # ---- 导出模型 (.pkl) —— 供实时在线推理底座使用 ----
    # 打包 booster + 全部推理所需元数据, 加载后即可对新窗口 P̂(t+H) 与 P(ΔP>θ) 推理。
    resid_sorted_val = np.sort(e_va).astype(np.float64)
    model_bundle = {
        "booster": booster,
        "best_iteration": int(booster.best_iteration),
        "keep_last": int(KEEP_LAST),
        "h_steps": int(H_STEPS),
        "feature_cols": feature_cols,
        "n_features_in": int(Xtr.shape[1]),
        "resid_val_sorted": resid_sorted_val,   # 用于 P(ΔP>θ) 经验分布
        "resid_bias": float(np.mean(e_va)),      # 偏置校正量
        "theta_default": 40.2,                   # ≈20% 装机
        "train_meta": {
            "mae_test": m_te["mae"], "rmse_test": m_te["rmse"], "r2_test": m_te["r2"],
            "resid_val_mean": rs_va["mean"], "resid_val_std": rs_va["std"],
        },
    }
    joblib.dump(model_bundle, MODEL_PKL)
    log(f"已导出模型: {MODEL_PKL}  (joblib, 含 booster + keep_last/H/特征名/残差分布/偏置)")

    log("完成。")


if __name__ == "__main__":
    main()