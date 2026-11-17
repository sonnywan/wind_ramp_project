# -*- coding: utf-8 -*-
"""
步骤 1+2：构造时序数据集
=========================================================
读取 clean_turbine.parquet / field_power.csv / ramp_labels.csv，
构造特征、标签，按时间切分（不打乱），归一化，滑动窗口采样，保存 .npz。

预测任务（与方案 §2 预警口径一致）：
    用过去 N=72 步（12h）历史 -> 预测未来是否发生爬坡。
    预测窗 = [t+L, t+L+Δt]，L=1h=6 步, Δt=1h=6 步 (即 t+6 ~ t+12)。
    窗内任一点 dir=+1 -> 标签 1 (上爬坡)；任一点 dir=-1 -> -1 (下爬坡)；否则 0。
    （上下同时出现时以上爬坡优先，主口径；此处正负不平衡用单任务三分类）

特征（风场级 + 气象）：全部来自 field_power / 机组聚合
    - Wspd_mean  平均风速
    - Wdir_xxx   平均风向(用 sin/cos 编码避免 0/360 断点)
    - Etmp/Itmp  平均环境温度/机舱温度
    - Ndir/Pab1-3/Prtv 平均桨距/机舱方向(已聚合在 clean_turbine, 这里取逐时刻均值)
    - C_mw       可用装机
    - P_mw       当前功率（滞后）
    - R          当前爬坡率（滞后）

时序切分（严禁 shuffle）：
    train Day 1-171 / val 172-208 / test 209-245
"""
import os
import numpy as np
import pandas as pd

OUT_DIR = "output"
DATA_DIR = "data"
TURB_PARQUET = os.path.join(OUT_DIR, "clean_turbine.parquet")
FIELD_CSV = os.path.join(OUT_DIR, "field_power.csv")
LABEL_CSV = os.path.join(OUT_DIR, "ramp_labels.csv")

STEPS_PER_DAY = 144
N_HIST = 72          # 输入历史 72 步 = 12h
L_STEPS = 6          # 提前量 L = 1h
DT_STEPS = 6         # 事件窗 Δt = 1h
CAP_MW = 201.0
SKIP_DAYS = [66, 67]

SPLIT_RANGE = {"train": (1, 171), "val": (172, 208), "test": (209, 245)}


def log(m):
    print(f"[dataset] {m}")


def load_and_build_features():
    log("读取机组级数据 ...")
    df = pd.read_parquet(TURB_PARQUET)

    # ---- 机组级按 (Day, step) 聚合 -> 风场级特征 ----
    # 风向用 sin/cos 编码
    df["Wdir_rad"] = np.deg2rad(df["Wdir"])
    df["Wdir_sin"] = np.sin(df["Wdir_rad"])
    df["Wdir_cos"] = np.cos(df["Wdir_rad"])
    df["Ndir_rad"] = np.deg2rad(df["Ndir"])
    df["Ndir_sin"] = np.sin(df["Ndir_rad"])
    df["Ndir_cos"] = np.cos(df["Ndir_rad"])

    agg = df.groupby(["Day", "step"], as_index=False).agg(
        Wspd=("Wspd", "mean"),
        Wdir_sin=("Wdir_sin", "mean"),
        Wdir_cos=("Wdir_cos", "mean"),
        Etmp=("Etmp", "mean"),
        Itmp=("Itmp", "mean"),
        Ndir_sin=("Ndir_sin", "mean"),
        Ndir_cos=("Ndir_cos", "mean"),
        Pab1=("Pab1", "mean"),
        Pab2=("Pab2", "mean"),
        Pab3=("Pab3", "mean"),
        Prtv=("Prtv", "mean"),
    )

    log("读取风场功率与标签 ...")
    fld = pd.read_csv(FIELD_CSV)[["Day", "step", "P_mw", "C_mw", "R", "is_gap", "is_main"]]
    lab = pd.read_csv(LABEL_CSV)[["Day", "step", "dir"]].rename(columns={"dir": "ramp_dir"})

    # 合并（P_mw/R/is_gap 已在 fld，lab 只补 ramp_dir，避免重复列）
    f = agg.merge(fld, on=["Day", "step"], how="left").merge(lab, on=["Day", "step"], how="left")
    f = f.rename(columns={"is_main": "ramp_main"})
    # 时间索引
    f["t_idx"] = (f["Day"] - 1) * STEPS_PER_DAY + f["step"]
    f = f.sort_values("t_idx").reset_index(drop=True)
    log(f"特征表: {f.shape}")
    return f


def make_labels_for_windows(f):
    """对每个 t，标签取 [t+L, t+L+Δt) 区间内 ramp_dir 的合并结果。
    上爬坡优先(+1 > -1)，无则 0。"""
    n = len(f)
    ramp = f["ramp_dir"].fillna(0).astype(int).to_numpy()
    lab = np.zeros(n, dtype=np.int8)
    valid = np.zeros(n, dtype=bool)  # 该 t 是否能构造有效 label（未来窗完整且非 gap）
    is_gap = f["is_gap"].fillna(1).astype(int).to_numpy()
    for t in range(n):
        a = t + L_STEPS
        b = t + L_STEPS + DT_STEPS
        if b >= n:
            continue
        # 未来窗内不能有 gap（数据缺口）
        if is_gap[a:b].any():
            continue
        seg = ramp[a:b]
        if (seg == 1).any():
            lab[t] = 1
        elif (seg == -1).any():
            lab[t] = -1
        else:
            lab[t] = 0
        valid[t] = True
    return lab, valid


FEATURE_COLS = ["Wspd", "Wdir_sin", "Wdir_cos", "Etmp", "Itmp",
                "Ndir_sin", "Ndir_cos", "Pab1", "Pab2", "Pab3", "Prtv",
                "C_mw", "P_mw", "R"]


def split_masks(f):
    masks = {}
    for name, (lo, hi) in SPLIT_RANGE.items():
        masks[name] = f["Day"].between(lo, hi).to_numpy()
    return masks


def build_windows(f, features, labels, valid, masks, scaler_stats):
    """滑动窗口采样。返回每个 split 的 X (N, N_HIST, F), y (N,).
    窗口终止于 t（含），输入为 [t-N_HIST+1, t] 的特征。"""
    X_mu, X_sd = scaler_stats
    X_all = (features - X_mu) / X_sd
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    out = {}
    for name, m in masks.items():
        xs, ys, ts = [], [], []
        # 只在窗口终点落在该 split 内、且 label 有效时采样
        idxs = np.where(m & valid)[0]
        for t in idxs:
            s = t - N_HIST + 1
            if s < 0:
                continue
            # 若窗口跨越全缺日(Day66/67)的 gap 直接跳过
            if f["is_gap"].iloc[s:t + 1].fillna(1).astype(int).any():
                continue
            xs.append(X_all[s:t + 1])
            ys.append(labels[t])
            ts.append(t)
        X = np.stack(xs) if xs else np.empty((0, N_HIST, features.shape[1]), np.float32)
        y = np.array(ys, dtype=np.int64)
        tarr = np.array(ts, dtype=np.int64)
        out[name] = (X, y, tarr)
        cnt = {v: int((y == v).sum()) for v in (-1, 0, 1)}
        log(f"  {name}: X={X.shape}, label分布 {cnt}")
    return out


def main():
    f = load_and_build_features()
    log("构造窗口标签 ...")
    labels, valid = make_labels_for_windows(f)
    masks = split_masks(f)

    features = f[FEATURE_COLS].to_numpy(dtype=np.float64)
    # 归一化统计量只用训练集（防止泄露）
    train_mask = masks["train"]
    tr = features[train_mask]
    X_mu = np.nanmean(tr, axis=0)
    X_sd = np.nanstd(tr, axis=0)
    X_sd = np.where(X_sd < 1e-8, 1.0, X_sd)
    log(f"归一化统计(仅训练集): mu={np.round(X_mu,2)}")

    out = build_windows(f, features, labels, valid, masks, (X_mu, X_sd))

    save = os.path.join(OUT_DIR, "dataset.npz")
    np.savez_compressed(
        save,
        X_train=out["train"][0], y_train=out["train"][1], t_train=out["train"][2],
        X_val=out["val"][0], y_val=out["val"][1], t_val=out["val"][2],
        X_test=out["test"][0], y_test=out["test"][1], t_test=out["test"][2],
        feat_mu=X_mu, feat_sd=X_sd, feature_cols=np.array(FEATURE_COLS),
    )
    log(f"已保存 {save}")
    log(f"特征维度: {len(FEATURE_COLS)} = {FEATURE_COLS}")


if __name__ == "__main__":
    main()