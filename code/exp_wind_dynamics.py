# -*- coding: utf-8 -*-
"""
exp_wind_dynamics.py —— 风速动态特征工程：拯救上爬坡可分辨性
================================================================
动机：现有 14 维特征对风速只有"绝对值 + 全程均值/标准差/首尾差"，
      看不到"最近风速在加速上升"这一上爬坡核心信号。
      测试期上爬坡概率 max 仅 0.57，模型对上行几乎无分辨力。

做法：在 flatten_with_stats 基础上，追加【风速的多尺度动态特征】：
      - 短窗斜率：最近 3/6/12/24 步的 Wspd 线性斜率(首尾差/步数)
      - 加速度：最近 6 步斜率 - 前 6 步斜率
      - 风速滚动极差、破位次数(近 12 步创新高次数)
      - 功率同比动态（同理，P_mw 短窗斜率）
    Wspd 与 P_mw 均来自特征列，无需反归一化（树对单调缩放不敏感，
    且短窗斜率归一化后仍保序保比例）。

输出：output/wind_dynamics_report.md（新旧对比 + 上行 τ 扫描）
      并保存增强模型的 prob 供复现：output/wd_pred.npz
"""
import os
import io
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import event_eval as ee

OUT = "output"
NPZ = os.path.join(OUT, "dataset.npz")
KEEP_LAST = 24
MERGE_GAP = 6
T_MIN, T_MAX = 29952, 35267


def log(m):
    print("[wd] " + m)


def flatten_base(X, keep_last=KEEP_LAST):
    N, T, F = X.shape
    flat = X[:, -keep_last:, :].reshape(N, keep_last * F)
    mean = X.mean(axis=1); std = X.std(axis=1)
    slope = X[:, -1, :] - X[:, 0, :]
    return np.concatenate([flat, mean, std, slope], axis=1)


def wind_dynamics(X, cols):
    """X:(N,T,F) 已归一化。返回风速/功率动态特征 (N, K)。"""
    N, T, F = X.shape
    ci = {c: i for i, c in enumerate(cols)}
    w = X[:, :, ci["Wspd"]]      # (N,T)
    p = X[:, :, ci["P_mw"]]      # (N,T)
    feats = []

    def slope_win(arr, k):
        seg = arr[:, -k:]
        # 线性斜率（最小二乘等价：首尾差/k 已足够，用回归更稳）
        xs = np.arange(k) - (k - 1) / 2.0
        yc = seg - seg.mean(axis=1, keepdims=True)
        s = (yc * xs).sum(axis=1) / (xs ** 2).sum()
        return s

    for k in (3, 6, 12, 24):
        feats.append(slope_win(w, k))
    # 加速度：近6步斜率 - 前6步(6..12)斜率
    s6 = slope_win(w, 6)
    w_prev6 = w[:, -12:-6]
    xs = np.arange(6) - 2.5
    yc = w_prev6 - w_prev6.mean(axis=1, keepdims=True)
    s6_prev = (yc * xs).sum(axis=1) / (xs ** 2).sum()
    feats.append(s6 - s6_prev)                 # 风速加速度
    # 近12步滚动极差
    feats.append(w[:, -12:].max(axis=1) - w[:, -12:].min(axis=1))
    # 近12步创新高次数
    w12 = w[:, -12:]
    new_high = np.array([(np.diff(w12[i]) > 0).sum() for i in range(N)], dtype=float)
    feats.append(new_high)
    # 功率动态：近6步斜率、近24步斜率
    feats.append(slope_win(p, 6))
    feats.append(slope_win(p, 24))
    # 风速与功率斜率的同步性（爬坡时二者同向上行）
    feats.append(slope_win(w, 6) * slope_win(p, 6))

    return np.stack(feats, axis=1)


def build(X, cols):
    return np.concatenate([flatten_base(X), wind_dynamics(X, cols)], axis=1)


def train_eval(Xtr, ytr, Xva, yva, Xte, yte, tag):
    def remap(y):
        return y + 1
    ytr_m, yva_m, yte_m = remap(ytr), remap(yva), remap(yte)
    cnt = np.bincount(ytr_m, minlength=3).astype(float)
    cls_w = cnt.sum() / (3 * cnt)
    wtr = cls_w[ytr_m]
    params = dict(
        objective="multiclass", num_class=3, metric="multi_logloss",
        learning_rate=0.03, num_leaves=31, max_depth=6, min_child_samples=60,
        feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=5,
        lambda_l1=0.5, lambda_l2=2.0, verbose=-1, num_threads=-1,
    )
    dtr = lgb.Dataset(Xtr, ytr_m, weight=wtr)
    dva = lgb.Dataset(Xva, yva_m, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)])
    prob_te = model.predict(Xte, num_iteration=model.best_iteration)
    prob_va = model.predict(Xva, num_iteration=model.best_iteration)
    yte_pred = prob_te.argmax(axis=1)
    # 是否爬坡二分类 AUC：正类=有爬坡(上或下), 分数=1-p(无爬坡)
    y_bin = (yte_m != 1).astype(int)          # 1=有爬坡(上/下), 0=无
    score_ramp = 1.0 - prob_te[:, 1]          # p(无爬坡) 越低越可能是爬坡
    auc = roc_auc_score(y_bin, score_ramp)
    acc = (yte_pred == yte_m).mean()
    macro_f1 = f1_score(yte_m, yte_pred, average="macro")
    # 上行 recall / 下行 recall
    cm = confusion_matrix(yte_m, yte_pred, labels=[0, 1, 2])
    up_recall = cm[2, 2] / cm[2].sum() if cm[2].sum() else 0.0
    dn_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0.0
    return dict(tag=tag, auc=auc, acc=acc, macro_f1=macro_f1,
                up_recall=up_recall, dn_recall=dn_recall,
                prob_te=prob_te, prob_va=prob_va, best_iter=model.best_iteration)


def up_tau_scan(prob_te, t_te, true_events):
    order = np.argsort(t_te)
    p_up = prob_te[order, 2]
    t = t_te[order]
    rows = []
    for tau in [0.60, 0.50, 0.45, 0.40, 0.35, 0.30]:
        ad = ee.build_alarm_from_prob(p_up, t, threshold=tau)
        pe = ee.merge_alarm_events(ad, ramp_dir=1, merge_gap=MERGE_GAP)
        byd = ee.compute_metrics_by_direction(true_events, pe)
        up = byd[1]
        rows.append((tau, up["hit_rate"], up["false_alarm_rate"], up["num_pred"]))
    return rows


def main():
    d = np.load(NPZ, allow_pickle=True)
    Xtr, ytr = d["X_train"], d["y_train"]
    Xva, yva = d["X_val"], d["y_val"]
    Xte, yte = d["X_test"], d["y_test"]
    t_te = d["t_test"].astype(int)
    cols = list(d["feature_cols"])
    log(f"原始 {Xtr.shape} / 特征列 {len(cols)}")

    # 基线（无动态特征）
    Xtr_b = flatten_base(Xtr); Xva_b = flatten_base(Xva); Xte_b = flatten_base(Xte)
    # 增强
    Xtr_e = build(Xtr, cols); Xva_e = build(Xva, cols); Xte_e = build(Xte, cols)
    log(f"基线维度 {Xtr_b.shape[1]} / 增强维度 {Xtr_e.shape[1]}")

    log("训练基线 ...")
    base = train_eval(Xtr_b, ytr, Xva_b, yva, Xte_b, yte, "base")
    log("训练增强(风速动态) ...")
    enh = train_eval(Xtr_e, ytr, Xva_e, yva, Xte_e, yte, "enh")

    for r in (base, enh):
        log(f"{r['tag']}: AUC={r['auc']:.4f} acc={r['acc']:.4f} macroF1={r['macro_f1']:.4f} "
            f"up_recall={r['up_recall']:.4f} dn_recall={r['dn_recall']:.4f} iter={r['best_iter']}")

    # 事件级上行 τ 扫描（用增强模型）
    ev = __import__("pandas").read_csv(os.path.join(OUT, "ramp_events.csv"))
    ev = ev[(ev.is_main == 1) & (ev.start_tidx >= T_MIN) & (ev.start_tidx <= T_MAX)].copy()
    ev["end_t_idx"] = ev.start_tidx + ev.n_trig - 1
    ev["ramp_dir"] = np.where(ev.direction == "up", 1, -1)
    ev = ev.reset_index(drop=True); ev["event_id"] = ev.index
    true_events = ev[["event_id", "start_tidx", "end_t_idx", "ramp_dir"]].rename(
        columns={"start_tidx": "start_t_idx"}).to_dict("records")

    scan_e = up_tau_scan(enh["prob_te"], t_te, true_events)
    scan_b = up_tau_scan(base["prob_te"], t_te, true_events)

    # 保存增强概率供复现
    np.savez(os.path.join(OUT, "wd_pred.npz"),
             prob_test=enh["prob_te"], y_test=yte, t_test=t_te,
             prob_val=enh["prob_va"], y_val=yva,
             prob_base_test=base["prob_te"])

    # 报告
    L = []
    L.append("# 风速动态特征工程实验报告\n")
    L.append("## 1. 动机\n")
    L.append("现有 14 维特征对风速只有绝对值+全程统计，看不到'近端加速上升'，")
    L.append("导致上爬坡概率被压扁(max≈0.57)、上行 recall 极低。")
    L.append("本实验追加速度动态特征重训，**在不降阈值的前提下**提升上爬坡可分辨性。\n")
    L.append("## 2. 新增动态特征\n")
    L.append("风速短窗线性斜率(3/6/12/24步)、风速加速度(近6步−前6步)、")
    L.append("近12步极差、近12步创新高次数、功率近6/24步斜率、风速×功率斜率同步性。\n")
    L.append("## 3. 新旧模型对比（测试集，统一 τ 无关）\n")
    L.append("| 模型 | AUC(是否爬坡) | acc | macro-F1 | 上行 recall | 下行 recall |\n|---|---|---|---|---|---|")
    for r in (base, enh):
        name = "基线(原14维)" if r["tag"] == "base" else "增强(+风速动态)"
        L.append(f"| {name} | {r['auc']:.4f} | {r['acc']:.4f} | {r['macro_f1']:.4f} | "
                 f"{r['up_recall']:.4f} | {r['dn_recall']:.4f} |")
    L.append("")
    L.append("## 4. 上行事件命中率 — 同阈值下新旧对比（不就阈值做妥协）\n")
    L.append("| 上行τ | 基线命中 | 基线虚警 | 增强命中 | 增强虚警 |\n|---|---|---|---|---|")
    for (tb, hb, fb, _), (te, he, fe, _) in zip(scan_b, scan_e):
        L.append(f"| {tb:.2f} | {hb:.3f} | {fb:.3f} | {he:.3f} | {fe:.3f} |")
    L.append("")
    # 结论
    d_auc = enh["auc"] - base["auc"]
    d_up = enh["up_recall"] - base["up_recall"]
    L.append("## 5. 结论\n")
    L.append(f"- 是否爬坡 AUC：{base['auc']:.4f} → {enh['auc']:.4f}（Δ={d_auc:+.4f}）")
    L.append(f"- 上行 recall：{base['up_recall']:.4f} → {enh['up_recall']:.4f}（Δ={d_up:+.4f}）")
    L.append(f"- 同 τ 下上行事件命中率见第 4 节；若增强模型在 τ=0.6 仍抓不到上行，")
    L.append("  说明单靠风速动态仍不足，需引入 NWP 预报或更长历史；本实验如实呈现。")

    txt = "\n".join(L)
    io.open(os.path.join(OUT, "wind_dynamics_report.md"), "w", encoding="utf-8").write(txt)
    print(txt)
    log("写出 output/wind_dynamics_report.md 与 wd_pred.npz")


if __name__ == "__main__":
    main()
