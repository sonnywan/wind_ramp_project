# -*- coding: utf-8 -*-
"""
步骤 3a'：两阶段 LightGBM 分类 —— 修复"上行爬坡塌缩到下行"

背景
----
原 train_lightgbm.py 用单一三分类 softmax（-1/0/1），测试集上"上行"召回率仅 2%、
其中 437/731 个上行被判成下行。根因：三分类让上行/下行在同一 softmax 里抢概率，
而"无事件"占比 ~74% 吸走概率质量，模型退化为"拿不准就说下行"。

本脚本改为两阶段：
  阶段1  爬坡 vs 无      二分类  y1 = (y != 0)
  阶段2  上行 vs 下行     二分类  仅用爬坡点训练  y2 = (y == 1)，再对所有点打分
组合概率（三列之和恒为 1，接口与旧 lgb_pred.npz 完全一致）：
  col0 p_dn   = p_ramp * (1 - p_up_given)
  col1 p_none = 1 - p_ramp
  col2 p_up   = p_ramp * p_up_given

输出：output/lgb_pred.npz（覆盖）+ output/lightgbm_report.md（更新）
"""
import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             accuracy_score, roc_auc_score)

OUT_DIR = "output"
NPZ = os.path.join(OUT_DIR, "dataset.npz")
KEEP_LAST = 24


def log(m):
    print(f"[cls2] {m}")


def flatten_with_stats(X, keep_last=24):
    N, T, F = X.shape
    flat = X[:, -keep_last:, :].reshape(N, keep_last * F)
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    slope = X[:, -1, :] - X[:, 0, :]
    return np.concatenate([flat, mean, std, slope], axis=1)


def balanced_weight(y01):
    cnt = np.bincount(y01, minlength=2).astype(float)
    cnt = np.where(cnt < 1, 1.0, cnt)
    w = cnt.sum() / (2.0 * cnt)
    return w[y01]


def train_binary(Xtr, ytr, Xva, yva, name, min_child=60, lr=0.03, rounds=1000):
    """训练二分类 LightGBM，返回训练好的 model（可对任意点打分）。"""
    wtr = balanced_weight(ytr)
    params = dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=lr,
        num_leaves=31,
        max_depth=6,
        min_child_samples=min_child,
        feature_fraction=0.5,
        bagging_fraction=0.8,
        bagging_freq=5,
        lambda_l1=0.5,
        lambda_l2=2.0,
        verbose=-1,
        num_threads=-1,
    )
    dtr = lgb.Dataset(Xtr, ytr, weight=wtr)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=rounds,
        valid_sets=[dva], valid_names=["val"],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(200)],
    )
    p_va = model.predict(Xva, num_iteration=model.best_iteration)
    auc = roc_auc_score(yva, p_va) if len(np.unique(yva)) > 1 else float("nan")
    log(f"[{name}] best_iter={model.best_iteration}  val_auc={auc:.4f}")
    return model


def combine(p_ramp, p_up_given):
    p_up = np.clip(p_up_given, 0.0, 1.0)
    p_up_final = p_ramp * p_up
    p_dn_final = p_ramp * (1.0 - p_up)
    p_none_final = 1.0 - p_ramp
    return np.stack([p_dn_final, p_none_final, p_up_final], axis=1)  # col0=down, col1=none, col2=up


def evaluate(name, prob, y):
    pred = prob.argmax(axis=1) - 1
    acc = accuracy_score(y, pred)
    f1m = f1_score(y, pred, average="macro")
    cm = confusion_matrix(y, pred, labels=[-1, 0, 1])
    rep = classification_report(y, pred, labels=[-1, 0, 1],
                                target_names=["-1(下爬坡)", "0(无)", "1(上爬坡)"],
                                zero_division=0)
    up_recall = cm[2, 2] / max(1, cm[2].sum())
    dn_recall = cm[0, 0] / max(1, cm[0].sum())
    log(f"[{name}] acc={acc:.4f} macroF1={f1m:.4f} 上行召回={up_recall:.3f} 下行召回={dn_recall:.3f}")
    return dict(acc=acc, f1=f1m, cm=cm, report=rep,
                up_recall=up_recall, dn_recall=dn_recall)


def main():
    d = np.load(NPZ, allow_pickle=True)
    Xtr, ytr = d["X_train"], d["y_train"]
    Xva, yva = d["X_val"], d["y_val"]
    Xte, yte, tte = d["X_test"], d["y_test"], d["t_test"]
    ttr, tva = d["t_train"], d["t_val"]

    log(f"训练 {Xtr.shape} / 验证 {Xva.shape} / 测试 {Xte.shape}")
    Xtr2 = flatten_with_stats(Xtr)
    Xva2 = flatten_with_stats(Xva)
    Xte2 = flatten_with_stats(Xte)
    log(f"压平后特征维度: {Xtr2.shape[1]}")

    # ---- 阶段1：爬坡(±1) vs 无(0) ----
    y1_tr = (ytr != 0).astype(int)
    y1_va = (yva != 0).astype(int)
    m1 = train_binary(Xtr2, y1_tr, Xva2, y1_va, "阶段1 爬坡vs无")
    p_ramp_tr = m1.predict(Xtr2, num_iteration=m1.best_iteration)
    p_ramp_va = m1.predict(Xva2, num_iteration=m1.best_iteration)
    p_ramp_te = m1.predict(Xte2, num_iteration=m1.best_iteration)

    # ---- 阶段2：上行(+1) vs 下行(-1)，仅用爬坡点训练，再对所有点打分 ----
    m_tr = ytr != 0
    m_va = yva != 0
    y2_tr = (ytr[m_tr] == 1).astype(int)
    y2_va = (yva[m_va] == 1).astype(int)
    m2 = train_binary(Xtr2[m_tr], y2_tr, Xva2[m_va], y2_va, "阶段2 上vs下", min_child=30)
    p_up_given_tr = m2.predict(Xtr2, num_iteration=m2.best_iteration)
    p_up_given_va = m2.predict(Xva2, num_iteration=m2.best_iteration)
    p_up_given_te = m2.predict(Xte2, num_iteration=m2.best_iteration)

    # ---- 组合概率 ----
    prob_tr = combine(p_ramp_tr, p_up_given_tr)
    prob_va = combine(p_ramp_va, p_up_given_va)
    prob_te = combine(p_ramp_te, p_up_given_te)

    res_tr = evaluate("train", prob_tr, ytr)
    res_va = evaluate("val", prob_va, yva)
    res_te = evaluate("test", prob_te, yte)

    # ---- 写报告 ----
    lines = ["# LightGBM 两阶段分类报告（修复上行塌缩）\n"]
    lines.append("任务：两阶段（阶段1 爬坡vs无 + 阶段2 上行vs下行）；输入窗口 72 步×14 特征，压平+统计特征。\n")
    lines.append("组合概率：p_up=p_ramp·p_up_given，p_dn=p_ramp·(1-p_up_given)，p_none=1-p_ramp；三列和=1，接口同旧三分类。\n")
    lines.append("## 1. 测试集指标（与旧三分类对比）\n")
    lines.append("| 指标 | 旧三分类 | 本两阶段 |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Accuracy | 0.7351 | {res_te['acc']:.4f} |")
    lines.append(f"| Macro-F1 | 0.4708 | {res_te['f1']:.4f} |")
    lines.append(f"| 上行召回率 | 0.02 | {res_te['up_recall']:.3f} |")
    lines.append(f"| 下行召回率 | 0.86 | {res_te['dn_recall']:.3f} |")
    lines.append("")
    lines.append("## 2. 测试集分类报告\n")
    lines.append("```\n" + res_te["report"] + "```\n")
    lines.append("## 3. 测试集混淆矩阵（行=真实，列=预测 [-1,0,1]）\n")
    lines.append("```\n" + str(res_te["cm"]) + "\n```\n")
    lines.append("## 4. 验证集分类报告\n")
    lines.append("```\n" + res_va["report"] + "```\n")
    lines.append("## 5. 训练集分类报告\n")
    lines.append("```\n" + res_tr["report"] + "```\n")
    with open(os.path.join(OUT_DIR, "lightgbm_report.md"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))

    # ---- 保存（与旧格式一致，覆盖 lgb_pred.npz）----
    np.savez(os.path.join(OUT_DIR, "lgb_pred.npz"),
             prob_test=prob_te, y_test=yte, t_test=tte,
             prob_val=prob_va, y_val=yva, t_val=tva)
    log("已写回 lgb_pred.npz（col0=down, col1=none, col2=up）")
    log(f"测试集上行概率 col2: max={prob_te[:,2].max():.3f}  >=0.6占比={(prob_te[:,2]>=0.6).mean():.3f}")
    log(f"测试集下行概率 col0: max={prob_te[:,0].max():.3f}  >=0.6占比={(prob_te[:,0]>=0.6).mean():.3f}")


if __name__ == "__main__":
    main()
