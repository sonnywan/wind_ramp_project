# -*- coding: utf-8 -*-
"""
步骤 3a：LightGBM 基线（传统机器学习）
=========================================================
输入: output/dataset.npz (X_train, y_train, ...)
窗口 3D (N, 72, 14) 压平为 (N, 72*14)，并追加对窗口的统计特征(均值/标准差/斜率)
以增强树的判别力。

任务: 三分类 (-1 下爬坡 / 0 无 / +1 上爬坡)
严格按时间切分，不打乱。

输出: output/lightgbm_report.md + 混淆矩阵/特征重要性图
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

OUT_DIR = "output"
NPZ = os.path.join(OUT_DIR, "dataset.npz")
KEEP_LAST = 24   # 压平时保留的最近步数


def log(m):
    print(f"[lgb] {m}")


def flatten_with_stats(X, keep_last=24):
    """X: (N, T, F) -> (N, keep_last*F + 3*F)
    仅保留最近 keep_last 步的压平值（降低高维过拟合）
    + 每个特征的均值/标准差/首尾斜率（全局趋势）。"""
    N, T, F = X.shape
    flat = X[:, -keep_last:, :].reshape(N, keep_last * F)
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    slope = X[:, -1, :] - X[:, 0, :]   # 首尾差近似斜率
    return np.concatenate([flat, mean, std, slope], axis=1)


def main():
    d = np.load(NPZ, allow_pickle=True)
    Xtr, ytr = d["X_train"], d["y_train"]
    Xva, yva = d["X_val"], d["y_val"]
    Xte, yte = d["X_test"], d["y_test"]
    feat_cols = list(d["feature_cols"])

    log(f"训练 {Xtr.shape} / 验证 {Xva.shape} / 测试 {Xte.shape}")
    Xtr2 = flatten_with_stats(Xtr)
    Xva2 = flatten_with_stats(Xva)
    Xte2 = flatten_with_stats(Xte)
    log(f"压平后特征维度: {Xtr2.shape[1]}")

    # 类别映射: -1,0,1 -> 0,1,2
    def remap(y):
        return y + 1
    ytr_m, yva_m, yte_m = remap(ytr), remap(yva), remap(yte)

    # 类别权重：缓解“上爬坡 recall 过低”的类别不平衡
    cls_cnt = np.bincount(ytr_m, minlength=3).astype(float)
    cls_w = (cls_cnt.sum() / (3 * cls_cnt))   # 均衡权重
    log(f"类别计数 {cls_cnt.astype(int).tolist()} -> 权重 {np.round(cls_w,3).tolist()}")
    wtr = cls_w[ytr_m]

    params = dict(
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
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
        num_threads=-1,
    )
    dtr = lgb.Dataset(Xtr2, ytr_m, weight=wtr)
    dva = lgb.Dataset(Xva2, yva_m, reference=dtr)

    log("训练 LightGBM ...")
    evals = {}
    model = lgb.train(
        params, dtr, num_boost_round=1000,
        valid_sets=[dva], valid_names=["val"],
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(100), lgb.record_evaluation(evals)],
    )
    log(f"best_iteration = {model.best_iteration}")

    # 评估
    label_names = ["-1(下爬坡)", "0(无)", "1(上爬坡)"]
    def evaluate(name, X2, y):
        prob = model.predict(X2, num_iteration=model.best_iteration)
        pred = prob.argmax(axis=1)
        acc = accuracy_score(y + 1, pred)
        f1m = f1_score(y + 1, pred, average="macro")
        cm = confusion_matrix(y + 1, pred, labels=[0, 1, 2])
        rep = classification_report(y + 1, pred, labels=[0, 1, 2], target_names=label_names, zero_division=0)
        log(f"[{name}] acc={acc:.4f} macroF1={f1m:.4f}")
        return dict(acc=acc, f1=f1m, cm=cm, report=rep, prob=prob)

    res_tr = evaluate("train", Xtr2, ytr)
    res_va = evaluate("val", Xva2, yva)
    res_te = evaluate("test", Xte2, yte)

    # 二分类(是否爬坡, 忽略方向)的宏 F1
    def ramp_f1(y, prob):
        pred = prob.argmax(axis=1)
        y_ramp = (y != 0).astype(int)
        p_ramp = (pred != 1).astype(int)
        return f1_score(y_ramp, p_ramp, average="binary")
    te_ramp_f1 = ramp_f1(yte, res_te["prob"])
    va_ramp_f1 = ramp_f1(yva, res_va["prob"])

    # 保存报告
    lines = ["# LightGBM 基线实验报告\n"]
    lines.append("任务：三分类（-1 下爬坡 / 0 无 / +1 上爬坡）；输入窗口 72 步（12h）× 14 特征，压平+统计特征。\n")
    lines.append("## 1. 总体指标\n")
    lines.append("| 集合 | Accuracy | Macro-F1 | 是否爬坡二分类 F1 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| 训练 | {res_tr['acc']:.4f} | {res_tr['f1']:.4f} | - |")
    lines.append(f"| 验证 | {res_va['acc']:.4f} | {res_va['f1']:.4f} | {va_ramp_f1:.4f} |")
    lines.append(f"| 测试 | {res_te['acc']:.4f} | {res_te['f1']:.4f} | {te_ramp_f1:.4f} |")
    lines.append("")
    lines.append("## 2. 测试集分类报告\n")
    lines.append("```\n" + res_te["report"] + "```\n")
    lines.append("## 3. 测试集混淆矩阵\n")
    lines.append("```\n" + str(res_te["cm"]) + "\n```\n")
    lines.append("（行=真实 [-1,0,1]，列=预测 [-1,0,1]）\n")
    lines.append("## 4. 特征重要性 Top-20\n")
    imp = model.feature_importance(importance_type="gain")
    names = [f"{c}_t{-KEEP_LAST+i}" for i in range(KEEP_LAST) for c in feat_cols] + \
            [f"{c}_mean" for c in feat_cols] + [f"{c}_std" for c in feat_cols] + [f"{c}_slope" for c in feat_cols]
    imps = sorted(zip(names, imp), key=lambda x: -x[1])[:20]
    lines.append("| 特征 | 增益 |")
    lines.append("| --- | --- |")
    for nm, v in imps:
        lines.append(f"| {nm} | {v:.1f} |")
    with open(os.path.join(OUT_DIR, "lightgbm_report.md"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))

    # 保存测试集概率供后续决策层用
    np.savez(os.path.join(OUT_DIR, "lgb_pred.npz"),
             prob_test=res_te["prob"], y_test=yte, t_test=d["t_test"],
             prob_val=res_va["prob"], y_val=yva, t_val=d["t_val"])
    log("报告与预测已保存")


if __name__ == "__main__":
    main()