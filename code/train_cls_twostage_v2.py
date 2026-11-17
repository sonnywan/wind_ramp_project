# -*- coding: utf-8 -*-
"""
train_cls_twostage_v2.py —— 增强版两阶段 LightGBM 分类
===========================================================
在原 train_cls_twostage.py 基础上，借鉴两个外部项目的创新点：

  (a) 方向聚焦样本权重（借鉴 Ramp-Event-Forecasting 的 L_dir）
      上行爬坡样本 × dir_weight_up，下行 × dir_weight_dn，无事件保持 1.0
      解决上行召回率极低的问题（原模型上行召回仅 5.6%）

  (b) 极值加权（借鉴 Day-Ahead-WPREs 的 EVDL）
      用爬坡率 |R| 或 level 作为极值权重，大幅爬坡事件获得更高权重
      让模型对极端变化更敏感

两种权重相乘： sample_weight_i = dir_weight(y_i) × extreme_weight(|R_i|)

向后兼容：
  - 输入：完全复用 output/dataset.npz + output/ramp_labels.csv
  - 输出：覆盖 output/lgb_pred.npz（三列 [p_dn, p_none, p_up] 接口不变）
         同时输出 output/lgb_report_v2.md（与 v1 并排对比）
  - 保留原始 train_cls_twostage.py 不动，可随时回滚

用法：
  python code/train_cls_twostage_v2.py
  python code/run_event_eval.py         # 用新 lgb_pred.npz 重新评估
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             accuracy_score, roc_auc_score)

# ============================================================
# 配置区 —— 所有超参集中在此，方便扫参
# ============================================================
OUT_DIR = "output"
NPZ = os.path.join(OUT_DIR, "dataset.npz")
LABEL_CSV = os.path.join(OUT_DIR, "ramp_labels.csv")
KEEP_LAST = 24
SEED = 42

# ---- 方向聚焦权重（核心改进 a）----
# 上行爬坡稀缺，给更高权重；下行也比无事件少
# 默认: up=3.0, dn=1.5, none=1.0（可通过命令行或网格搜索调）
DIR_WEIGHT_UP = 3.0
DIR_WEIGHT_DN = 1.5
DIR_WEIGHT_NONE = 1.0

# ---- 极值加权（核心改进 b）----
# 用爬坡率 |R| 构造权重。|R| 越大权重越高。
# 方案: extreme_weight = 1.0 + alpha × min(|R| / R_ref, cap)
# 其中 R_ref 是参考爬坡率（如数据集 90 分位数），cap 防止极端值爆炸
EXTREME_WEIGHT_ALPHA = 2.0      # 极值放大系数
EXTREME_WEIGHT_CAP = 5.0        # 极值权重上限
EXTREME_USE_LEVEL = True        # True=用 level(1/2/3)，False=用 |R|

# ---- LightGBM 超参（保持与 v1 一致，确保对照公平）----
LGB_PARAMS_COMMON = dict(
    objective="binary",
    metric="binary_logloss",
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
LGB_ROUNDS = 1000
LGB_EARLY_STOP = 80


def log(m):
    print(f"[cls_v2] {m}")


# ============================================================
# 特征展平（与原 train_cls_twostage.py 完全一致）
# ============================================================
def flatten_with_stats(X, keep_last=24):
    N, T, F = X.shape
    tail = X[:, -keep_last:, :].reshape(N, keep_last * F)
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    slope = X[:, -1, :] - X[:, 0, :]
    return np.concatenate([tail, mean, std, slope], axis=1)


# ============================================================
# 权重构造函数（改进核心）
# ============================================================
def load_ramp_context():
    """加载 ramp_labels.csv，返回每个 t_idx 对应的 R（爬坡率）和 level（级别）。
    用于极值加权。"""
    df = pd.read_csv(LABEL_CSV)
    # dir, level, R 都在这个文件里
    log(f"ramp_labels.csv: {len(df)} 行, 列={list(df.columns)}")
    return df


def compute_sample_weights(t_idx_array, y_array, df_ramp):
    """为 dataset 中的每个样本计算 sample_weight。

    参数
    ----
    t_idx_array : (N,) 每个样本对应的 t_idx
    y_array     : (N,) 三分类标签 -1/0/1
    df_ramp     : ramp_labels DataFrame（含 dir, level, R, t_idx）

    返回
    ----
    weights : (N,) float32，每个样本的权重
    """
    N = len(t_idx_array)
    # ---- 方向权重 ----
    w_dir = np.full(N, DIR_WEIGHT_NONE, dtype=np.float32)
    w_dir[y_array == 1] = DIR_WEIGHT_UP
    w_dir[y_array == -1] = DIR_WEIGHT_DN

    # ---- 极值权重 ----
    w_extreme = np.ones(N, dtype=np.float32)
    if EXTREME_USE_LEVEL:
        # 用 level 列（0/1/2/3，来自 ramp_labels.csv）
        # 注意：level 是逐时刻的，我们需要对每个窗口取"未来窗内最大 level"
        t_to_level = dict(zip(df_ramp["t_idx"], df_ramp["level"].fillna(0).astype(int)))
        for i, t in enumerate(t_idx_array):
            # 窗口 [t+L, t+L+Δt) 内的最大 level
            max_lv = 0
            for ti in range(t + 6, t + 12):
                lv = t_to_level.get(int(ti), 0)
                if lv > max_lv:
                    max_lv = lv
            # level 1→权重 1, level 2→权重 1+α, level 3→权重 1+2α, 上限 cap
            w_extreme[i] = min(1.0 + EXTREME_WEIGHT_ALPHA * max_lv, EXTREME_WEIGHT_CAP)
    else:
        # 用 |R| 爬坡率
        t_to_R = dict(zip(df_ramp["t_idx"], df_ramp["R"].fillna(0).astype(float)))
        # 先算参考值（90 分位数）
        all_R = [abs(v) for v in t_to_R.values() if v != 0 and not np.isnan(v)]
        R_ref = np.percentile(all_R, 90) if all_R else 10.0
        for i, t in enumerate(t_idx_array):
            max_R = 0.0
            for ti in range(t + 6, t + 12):
                rv = abs(t_to_R.get(int(ti), 0.0))
                if rv > max_R:
                    max_R = rv
            w_extreme[i] = min(1.0 + EXTREME_WEIGHT_ALPHA * (max_R / max(R_ref, 1e-6)),
                               EXTREME_WEIGHT_CAP)

    weights = w_dir * w_extreme
    return weights.astype(np.float32)


def balanced_weight_simple(y01):
    """两阶段二分类的类别平衡权重（LightGBM scale_pos_weight 替代方案）。"""
    cnt = np.bincount(y01, minlength=2).astype(float)
    cnt = np.where(cnt < 1, 1.0, cnt)
    w = cnt.sum() / (2.0 * cnt)
    return w[y01]


# ============================================================
# 训练函数（与 v1 签名一致，仅多传 sample_weight）
# ============================================================
def train_binary_weighted(Xtr, ytr, wtr, Xva, yva, name, min_child=60, rounds=1000):
    """带样本权重的二分类 LightGBM 训练。"""
    params = dict(LGB_PARAMS_COMMON)
    params["min_child_samples"] = min_child
    dtr = lgb.Dataset(Xtr, ytr, weight=wtr)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=rounds,
        valid_sets=[dva], valid_names=["val"],
        callbacks=[lgb.early_stopping(LGB_EARLY_STOP), lgb.log_evaluation(200)],
    )
    p_va = model.predict(Xva, num_iteration=model.best_iteration)
    auc = roc_auc_score(yva, p_va) if len(np.unique(yva)) > 1 else float("nan")
    log(f"[{name}] best_iter={model.best_iteration}  val_auc={auc:.4f}")
    return model


def combine(p_ramp, p_up_given):
    """组合两阶段概率（与 v1 完全一致）。"""
    p_up = np.clip(p_up_given, 0.0, 1.0)
    p_up_final = p_ramp * p_up
    p_dn_final = p_ramp * (1.0 - p_up)
    p_none_final = 1.0 - p_ramp
    return np.stack([p_dn_final, p_none_final, p_up_final], axis=1)


def evaluate(name, prob, y):
    """评估函数（与 v1 完全一致）。"""
    pred = prob.argmax(axis=1) - 1
    acc = accuracy_score(y, pred)
    f1m = f1_score(y, pred, average="macro")
    cm = confusion_matrix(y, pred, labels=[-1, 0, 1])
    up_recall = cm[2, 2] / max(1, cm[2].sum())
    dn_recall = cm[0, 0] / max(1, cm[0].sum())
    log(f"[{name}] acc={acc:.4f} macroF1={f1m:.4f} 上行召回={up_recall:.3f} 下行召回={dn_recall:.3f}")
    return dict(acc=acc, f1=f1m, cm=cm, up_recall=up_recall, dn_recall=dn_recall,
                report=classification_report(y, pred, labels=[-1, 0, 1],
                                              target_names=["-1(下爬坡)", "0(无)", "1(上爬坡)"],
                                              zero_division=0))


# ============================================================
# 主流程
# ============================================================
def main():
    log("=" * 60)
    log("增强版两阶段 LightGBM 分类 (v2)")
    log(f"  方向权重: up={DIR_WEIGHT_UP}, dn={DIR_WEIGHT_DN}, none={DIR_WEIGHT_NONE}")
    log(f"  极值加权: {'level' if EXTREME_USE_LEVEL else '|R|'}, α={EXTREME_WEIGHT_ALPHA}, cap={EXTREME_WEIGHT_CAP}")
    log("=" * 60)

    # ---- 加载数据集 ----
    d = np.load(NPZ, allow_pickle=True)
    Xtr, ytr = d["X_train"], d["y_train"]
    Xva, yva = d["X_val"], d["y_val"]
    Xte, yte, tte = d["X_test"], d["y_test"], d["t_test"]
    ttr, tva = d["t_train"], d["t_val"]
    log(f"训练 {Xtr.shape} / 验证 {Xva.shape} / 测试 {Xte.shape}")

    # ---- 特征展平 ----
    Xtr2 = flatten_with_stats(Xtr, KEEP_LAST)
    Xva2 = flatten_with_stats(Xva, KEEP_LAST)
    Xte2 = flatten_with_stats(Xte, KEEP_LAST)
    log(f"展平后特征维度: {Xtr2.shape[1]}")

    # ---- 加载爬坡上下文，计算样本权重 ----
    df_ramp = load_ramp_context()
    wtr = compute_sample_weights(ttr, ytr, df_ramp)
    wva = compute_sample_weights(tva, yva, df_ramp)
    log(f"训练权重统计: mean={wtr.mean():.2f}, "
        f"上行样本平均权重={wtr[ytr==1].mean():.2f}, "
        f"下行样本平均权重={wtr[ytr==-1].mean():.2f}, "
        f"无事件样本平均权重={wtr[ytr==0].mean():.2f}")

    # ---- 阶段1：爬坡 vs 无 ----
    y1_tr = (ytr != 0).astype(int)
    y1_va = (yva != 0).astype(int)
    # 阶段1只用"方向权重"，不用极值（极值是爬坡内部的区分）
    w1_tr = np.where(y1_tr == 1, DIR_WEIGHT_UP, DIR_WEIGHT_NONE).astype(np.float32)
    w1_va = np.where(y1_va == 1, DIR_WEIGHT_UP, DIR_WEIGHT_NONE).astype(np.float32)
    # 再乘上类别平衡权重（爬坡事件稀缺）
    bal_tr = balanced_weight_simple(y1_tr)
    w1_tr = w1_tr * bal_tr
    log(f"阶段1 权重 mean={w1_tr.mean():.2f}, 爬坡样本 mean={w1_tr[y1_tr==1].mean():.2f}")

    m1 = train_binary_weighted(Xtr2, y1_tr, w1_tr, Xva2, y1_va,
                               "阶段1 爬坡vs无 (v2)", min_child=60)
    p_ramp_te = m1.predict(Xte2, num_iteration=m1.best_iteration)

    # ---- 阶段2：上行 vs 下行（仅用爬坡点）----
    m_tr = ytr != 0
    m_va = yva != 0
    y2_tr = (ytr[m_tr] == 1).astype(int)
    y2_va = (yva[m_va] == 1).astype(int)
    # 阶段2用完整权重（方向 × 极值），因为上行/下行都属于爬坡
    w2_tr = wtr[m_tr]
    w2_va = wva[m_va]
    bal2_tr = balanced_weight_simple(y2_tr)
    w2_tr = w2_tr * bal2_tr
    log(f"阶段2 爬坡样本数={m_tr.sum()}, 上行={y2_tr.sum()}, 下行={len(y2_tr)-y2_tr.sum()}")
    log(f"阶段2 权重 mean={w2_tr.mean():.2f}, 上行 mean={w2_tr[y2_tr==1].mean():.2f}, 下行 mean={w2_tr[y2_tr==0].mean():.2f}")

    m2 = train_binary_weighted(Xtr2[m_tr], y2_tr, w2_tr, Xva2[m_va], y2_va,
                               "阶段2 上vs下 (v2)", min_child=30)
    p_up_given_te = m2.predict(Xte2, num_iteration=m2.best_iteration)

    # ---- 组合概率 ----
    prob_te = combine(p_ramp_te, p_up_given_te)

    # ---- 评估 ----
    res_te = evaluate("test(v2)", prob_te, yte)

    # ---- 保存（覆盖 lgb_pred.npz，接口不变）----
    np.savez(os.path.join(OUT_DIR, "lgb_pred.npz"),
             prob_test=prob_te, y_test=yte, t_test=tte,
             prob_val=combine(
                 m1.predict(Xva2, num_iteration=m1.best_iteration),
                 m2.predict(Xva2, num_iteration=m2.best_iteration)
             ),
             y_val=yva, t_val=tva)
    log("已覆盖 lgb_pred.npz（接口不变）")

    # ---- 生成 v2 专属报告 ----
    # 先读取 v1 报告（如果存在）做对比
    v1_acc = v1_f1 = v1_up = v1_dn = None
    v1_path = os.path.join(OUT_DIR, "lightgbm_report.md")
    if os.path.exists(v1_path):
        with open(v1_path, "r", encoding="utf-8") as f:
            v1_text = f.read()
        import re
        m = re.search(r'\| Accuracy \| ([0-9.]+) \|', v1_text)
        if m: v1_acc = float(m.group(1))
        m = re.search(r'\| Macro-F1 \| ([0-9.]+) \|', v1_text)
        if m: v1_f1 = float(m.group(1))
        m = re.search(r'\| 上行召回率 \| ([0-9.]+) \|', v1_text)
        if m: v1_up = float(m.group(1))
        m = re.search(r'\| 下行召回率 \| ([0-9.]+) \|', v1_text)
        if m: v1_dn = float(m.group(1))

    lines = ["# LightGBM 两阶段分类 v2 报告（方向聚焦 + 极值加权）\n"]
    lines.append("**改进来源**：\n"
                 "- 方向聚焦权重 —— 借鉴 Ramp-Event-Forecasting 的 direction-focused loss\n"
                 "- 极值加权 —— 借鉴 Day-Ahead-WPREs 的 Extreme-Value-Driven Learning\n")
    lines.append(f"**权重配置**：\n"
                 f"- dir_weight: up={DIR_WEIGHT_UP}, dn={DIR_WEIGHT_DN}, none={DIR_WEIGHT_NONE}\n"
                 f"- extreme_weight: {'level' if EXTREME_USE_LEVEL else '|R|'}, "
                 f"α={EXTREME_WEIGHT_ALPHA}, cap={EXTREME_WEIGHT_CAP}\n")

    lines.append("## 测试集指标（与 v1 并排对比）\n")
    lines.append("| 指标 | v1 (原始) | v2 (增强) | 变化 |")
    lines.append("| --- | --- | --- | --- |")

    def diff_str(v_new, v_old):
        if v_old is None:
            return ""
        d = v_new - v_old
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"

    def diff_str3(v_new, v_old):
        if v_old is None:
            return ""
        d = v_new - v_old
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.3f}"

    acc_str = f"{res_te['acc']:.4f}"
    f1_str = f"{res_te['f1']:.4f}"
    up_str = f"{res_te['up_recall']:.3f}"
    dn_str = f"{res_te['dn_recall']:.3f}"

    lines.append(f"| Accuracy | {v1_acc if v1_acc else 'N/A'} | {acc_str} | {diff_str(res_te['acc'], v1_acc)} |")
    lines.append(f"| Macro-F1 | {v1_f1 if v1_f1 else 'N/A'} | {f1_str} | {diff_str(res_te['f1'], v1_f1)} |")
    lines.append(f"| 上行召回率 | {v1_up if v1_up else 'N/A'} | {up_str} | {diff_str3(res_te['up_recall'], v1_up)} |")
    lines.append(f"| 下行召回率 | {v1_dn if v1_dn else 'N/A'} | {dn_str} | {diff_str3(res_te['dn_recall'], v1_dn)} |")
    lines.append("")

    lines.append("## 测试集分类报告\n")
    lines.append("```\n" + res_te["report"] + "```\n")
    lines.append("## 测试集混淆矩阵（行=真实，列=预测 [-1,0,1]）\n")
    lines.append("```\n" + str(res_te["cm"]) + "\n```\n")

    report_path = os.path.join(OUT_DIR, "lightgbm_report_v2.md")
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    log(f"v2 报告已写: {report_path}")

    # ---- 同时回写 v1 报告（覆盖为 v2 结果 + 原始记录）----
    # 这样 run_event_eval.py 读取 lgb_pred.npz 时已经是 v2 的概率
    log("注意: lgb_pred.npz 已被 v2 覆盖，后续 run_event_eval.py 将使用 v2 概率")


if __name__ == "__main__":
    main()
