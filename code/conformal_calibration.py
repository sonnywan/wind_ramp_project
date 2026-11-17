# -*- coding: utf-8 -*-
"""
conformal_calibration.py —— 共形推理（Conformal Prediction）校准模块
=======================================================================
借鉴 Day-Ahead-WPREs 的 Confidence-Aware Detection 思路，用共形推理
为回归模型的点预测提供**具有严格统计保证**的置信区间。

与现有经验性残差分位数法的对比：
  - 现有方法 (train_reg_lightgbm.py)：用验证集残差分位数 (P05/P95) 平移 → 区间
    缺点：没有覆盖率保证，区间宽度随数据分布漂移而失真
  - 共形推理：用验证集构造非一致性分数，再用指定覆盖率 (1-α) 取分位数
    保证：在交换性假设下，覆盖率 ≥ 1-α（有限样本精确保证）

向后兼容：
  - 输入：reg_pred.npz（回归点预测 + 验证/测试集残差），reg_lightgbm_model.pkl
  - 输出：覆盖 reg_pred.npz，新增 conformal_P10/P90 等列；生成 conformal_report.md
  - 不影响任何现有调用方（新增字段，原有字段保留）

核心函数（纯函数，可独立调用）：
  conformal_quantile(residuals_calib, alpha=0.10) → float
    用校准集残差（应该是 |e|）计算共形分位点
  conformal_interval(pred_point, q, y_calib_std) → (lo, hi)
    根据点预测和共形分位点构造区间
"""
import os
import json
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUT_DIR = "output"
REG_NPZ = os.path.join(OUT_DIR, "reg_pred.npz")
REG_MODEL = os.path.join(OUT_DIR, "reg_lightgbm_model.pkl")
REPORT_MD = os.path.join(OUT_DIR, "conformal_report.md")

# 共形推理的覆盖率配置
# 常用: 90% (α=0.10), 80% (α=0.20), 95% (α=0.05)
ALPHA_LEVELS = [0.10, 0.20, 0.30]  # 对应 90%, 80%, 70% 置信度


def log(m):
    print(f"[conformal] {m}")


# ============================================================
# 共形推理核心（纯函数）
# ============================================================
def conformal_quantile(nonconformity_scores, alpha=0.10):
    """从校准集非一致性分数中取共形分位点。

    参数
    ----
    nonconformity_scores : (n_calib,) array-like
        校准集非一致性分数。对回归任务通常取 |e_i| = |y_i - ŷ_i|。
    alpha : float
        显著性水平。如 α=0.10 → 90% 置信区间。

    返回
    ----
    q : float
        共形分位点。构造区间用 [ŷ - q, ŷ + q]，保证覆盖率 ≥ 1-α。

    理论保证（交换性假设下）：
      P(|y - ŷ| ≤ q) ≥ 1 - α
    """
    scores = np.asarray(nonconformity_scores).flatten()
    scores = np.sort(scores)
    n = len(scores)
    # 有限样本精确共形分位点公式
    # q = ceil((n+1)(1-α)) / n 的位置
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    rank = min(rank, n)  # 防止越界
    q = scores[rank - 1]  # 0-indexed
    return float(q)


def conformal_interval(pred_point, q):
    """用共形分位点构造预测区间。

    返回
    ----
    (lo, hi) : 两端点
    """
    return (pred_point - q, pred_point + q)


def coverage_check(y_true, pred_point, q):
    """经验覆盖率检查。"""
    covered = np.abs(y_true - pred_point) <= q
    return float(covered.mean())


# ============================================================
# 分位数回归变体（可选）
# ============================================================
def conformal_quantile_asymmetric(residuals_calib, alpha=0.10):
    """非对称共形区间：分别对正/负残差取分位点。

    当残差分布不对称时（如风电功率有零下界），非对称区间更合理。
    """
    e = np.asarray(residuals_calib).flatten()
    # 下界分位点（正残差方向：y > ŷ 多远）
    q_hi = float(np.quantile(e, 1 - alpha / 2))
    # 上界分位点（负残差方向：ŷ - y 多远，即 e 的负端）
    q_lo = float(np.quantile(e, alpha / 2))
    return q_lo, q_hi


def coverage_check_asymmetric(y_true, pred_point, q_lo, q_hi):
    covered = (y_true >= pred_point + q_lo) & (y_true <= pred_point + q_hi)
    return float(covered.mean())


# ============================================================
# 主流程
# ============================================================
def main():
    log("=" * 60)
    log("共形推理校准 (Conformal Prediction)")
    log("=" * 60)

    # ---- 加载已有回归结果 ----
    if not os.path.exists(REG_NPZ):
        log(f"错误: 找不到 {REG_NPZ}，请先运行 train_reg_lightgbm.py")
        return

    dz = np.load(REG_NPZ, allow_pickle=True)
    resid_val = dz["resid_val"].astype(np.float64)  # 验证集残差 (校准集)
    resid_test = dz["resid_test"].astype(np.float64)  # 测试集残差
    pred_val = dz["pred_val"].astype(np.float64)
    pred_test = dz["pred_test"].astype(np.float64)
    y_val = dz["y_reg_val"].astype(np.float64)
    y_test = dz["y_reg_test"].astype(np.float64)
    log(f"校准集(验证): n={len(resid_val)}, 测试集: n={len(resid_test)}")

    # ---- 对称共形区间 ----
    log("\n--- 对称共形区间 ---")
    results = {}
    for alpha in ALPHA_LEVELS:
        q = conformal_quantile(np.abs(resid_val), alpha=alpha)
        cov_val = coverage_check(y_val, pred_val, q)
        cov_test = coverage_check(y_test, pred_test, q)
        results[f"alpha_{alpha:.2f}"] = {
            "q": q,
            "nominal_coverage": 1 - alpha,
            "actual_coverage_val": cov_val,
            "actual_coverage_test": cov_test,
            "interval_width": 2 * q,
        }
        log(f"α={alpha:.2f} (置信度 {1-alpha:.0%}): "
            f"q={q:.2f} MW, 验证集覆盖={cov_val:.4f}, 测试集覆盖={cov_test:.4f}, "
            f"区间宽={2*q:.2f} MW")

    # ---- 非对称共形区间 ----
    log("\n--- 非对称共形区间 ---")
    asym_results = {}
    for alpha in ALPHA_LEVELS:
        q_lo, q_hi = conformal_quantile_asymmetric(resid_val, alpha=alpha)
        cov_val = coverage_check_asymmetric(y_val, pred_val, q_lo, q_hi)
        cov_test = coverage_check_asymmetric(y_test, pred_test, q_lo, q_hi)
        asym_results[f"alpha_{alpha:.2f}"] = {
            "q_lo": q_lo, "q_hi": q_hi,
            "nominal_coverage": 1 - alpha,
            "actual_coverage_val": cov_val,
            "actual_coverage_test": cov_test,
            "interval_width": q_hi - q_lo,
        }
        log(f"α={alpha:.2f}: q_lo={q_lo:.2f}, q_hi={q_hi:.2f}, "
            f"验证集覆盖={cov_val:.4f}, 测试集覆盖={cov_test:.4f}, "
            f"区间宽={q_hi - q_lo:.2f} MW")

    # ---- 与原经验分位数法对比 ----
    log("\n--- 与原经验分位数法对比 (90% 区间) ---")
    e_val = resid_val
    p05 = float(np.percentile(e_val, 5))
    p95 = float(np.percentile(e_val, 95))
    cov_test_emp = float(((resid_test >= p05) & (resid_test <= p95)).mean())
    width_emp = p95 - p05
    q_sym_90 = conformal_quantile(np.abs(resid_val), alpha=0.10)
    width_sym_90 = 2 * q_sym_90
    cov_test_sym = coverage_check(y_test, pred_test, q_sym_90)
    log(f"原经验 P05/P95: 宽度={width_emp:.2f} MW, 测试集覆盖={cov_test_emp:.4f}")
    log(f"共形对称 90%: 宽度={width_sym_90:.2f} MW, 测试集覆盖={cov_test_sym:.4f}")
    log(f"共形 保证: P(覆盖) ≥ 0.90; 经验法 无保证")

    # ---- 写报告 ----
    lines = ["# 共形推理校准报告\n"]
    lines.append("**方法**: Split-Conformal Prediction（验证集做校准集）\n")
    lines.append("## 1. 对称共形区间\n")
    lines.append("| α | 名义置信度 | 共形分位点 q (MW) | 区间宽 (MW) | 验证集覆盖 | 测试集覆盖 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for alpha in ALPHA_LEVELS:
        r = results[f"alpha_{alpha:.2f}"]
        lines.append(f"| {alpha:.2f} | {r['nominal_coverage']:.0%} | "
                     f"{r['q']:.3f} | {r['interval_width']:.3f} | "
                     f"{r['actual_coverage_val']:.4f} | {r['actual_coverage_test']:.4f} |")
    lines.append("")

    lines.append("## 2. 非对称共形区间\n")
    lines.append("| α | 名义置信度 | q_lo (MW) | q_hi (MW) | 区间宽 (MW) | 验证集覆盖 | 测试集覆盖 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for alpha in ALPHA_LEVELS:
        r = asym_results[f"alpha_{alpha:.2f}"]
        lines.append(f"| {alpha:.2f} | {r['nominal_coverage']:.0%} | "
                     f"{r['q_lo']:.3f} | {r['q_hi']:.3f} | {r['interval_width']:.3f} | "
                     f"{r['actual_coverage_val']:.4f} | {r['actual_coverage_test']:.4f} |")
    lines.append("")

    lines.append("## 3. 与原经验分位数法对比（90% 区间）\n")
    lines.append("| 方法 | 区间下界 | 区间上界 | 区间宽 (MW) | 测试集覆盖 | 覆盖率保证 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    lines.append(f"| 经验 P05/P95 | {p05:.3f} | {p95:.3f} | {width_emp:.3f} | "
                 f"{cov_test_emp:.4f} | ❌ 无理论保证 |")
    lines.append(f"| 共形对称 90% | -{q_sym_90:.3f} | +{q_sym_90:.3f} | {width_sym_90:.3f} | "
                 f"{cov_test_sym:.4f} | ✅ ≥90% (交换性假设) |")
    lines.append("")

    lines.append("## 4. 使用方式\n")
    lines.append("```python\n"
                 "from code.conformal_calibration import conformal_quantile, conformal_interval\n"
                 "\n"
                 "# 已有校准集残差\n"
                 "q = conformal_quantile(np.abs(residuals_val), alpha=0.10)  # 90% 区间\n"
                 "lo, hi = conformal_interval(y_pred_point, q)\n"
                 "# 保证: P(y_true ∈ [lo, hi]) ≥ 0.90\n"
                 "```\n")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"报告已写: {REPORT_MD}")

    # ---- 扩展保存: 更新 reg_pred.npz，新增 conformal 列 ----
    # 保持向后兼容: 原有字段不变，新增 conformal 字段
    extras = {}
    for alpha in ALPHA_LEVELS:
        q = conformal_quantile(np.abs(resid_val), alpha=alpha)
        extras[f"conformal_q_{alpha:.2f}"] = np.float64(q)
        extras[f"conformal_lo_test_{alpha:.2f}"] = (pred_test - q).astype(np.float32)
        extras[f"conformal_hi_test_{alpha:.2f}"] = (pred_test + q).astype(np.float32)
    np.savez(REG_NPZ, **{k: dz[k] for k in dz.files}, **extras)
    log(f"已扩展保存 {REG_NPZ} (新增 {len(extras)} 个 conformal 字段)")


if __name__ == "__main__":
    main()
