# -*- coding: utf-8 -*-
"""
步骤 3b：LSTM / TCN 训练（PyTorch + GPU）
=========================================================
用法: python code/train_dl.py --model lstm
git     python code/train_dl.py --model tcn

输入: output/dataset.npz
任务: 三分类 (-1/0/+1 -> 0/1/2)
严格按时间切分，不打乱验证/测试；训练按 epoch 打乱 batch（标准做法）。

输出: output/dl_{model}_report.md + output/dl_{model}_pred.npz
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from models import build_model

OUT_DIR = "output"
NPZ = os.path.join(OUT_DIR, "dataset.npz")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(m):
    print(f"[dl] {m}")


def class_weights(y, device):
    y = y + 1   # 0,1,2
    cnt = np.bincount(y, minlength=3).astype(np.float32)
    w = cnt.sum() / (3 * cnt)
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_prob, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        all_prob.append(prob)
        all_y.append(yb.numpy())
    prob = np.concatenate(all_prob)
    y = np.concatenate(all_y)       # 已由 loader 映射为 0/1/2
    pred = prob.argmax(axis=1)
    return y, pred, prob


def metrics(y, pred):
    acc = accuracy_score(y, pred)
    f1m = f1_score(y, pred, average="macro")
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    y_ramp = (y != 1).astype(int)
    p_ramp = (pred != 1).astype(int)
    ramp_f1 = f1_score(y_ramp, p_ramp, average="binary", zero_division=0)
    rep = classification_report(y, pred, labels=[0, 1, 2],
                                target_names=["-1(下爬坡)", "0(无)", "1(上爬坡)"], zero_division=0)
    return dict(acc=acc, f1=f1m, ramp_f1=ramp_f1, cm=cm, report=rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", choices=["lstm", "tcn"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    d = np.load(NPZ, allow_pickle=True)
    Xtr, ytr = d["X_train"], d["y_train"]
    Xva, yva = d["X_val"], d["y_val"]
    Xte, yte = d["X_test"], d["y_test"]
    n_feat = Xtr.shape[2]
    log(f"{args.model} | 训练 {Xtr.shape} / 验证 {Xva.shape} / 测试 {Xte.shape} | device={DEVICE}")

    def to_loader(X, y, shuffle):
        # 标签 -1/0/+1 -> 0/1/2，供 CrossEntropyLoss 使用
        y_mapped = (y + 1).astype(np.int64)
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_mapped))
        return DataLoader(ds, batch_size=args.bs, shuffle=shuffle, num_workers=0)

    tr_loader = to_loader(Xtr, ytr, True)
    va_loader = to_loader(Xva, yva, False)
    te_loader = to_loader(Xte, yte, False)

    model = build_model(args.model, n_feat).to(DEVICE)
    w = class_weights(ytr, DEVICE)
    log(f"class_weights={w.cpu().numpy().round(3).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=w)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=5)

    best_val_f1 = -1
    best_state = None
    best_epoch = -1
    patience, bad = 10, 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tot_loss += loss.item() * len(xb)
        tot_loss /= len(tr_loader.dataset)

        yv, pv, _ = evaluate(model, va_loader, DEVICE)
        mv = metrics(yv, pv)
        scheduler.step(1 - mv["f1"])
        log(f"ep{ep:02d} loss={tot_loss:.4f} val_acc={mv['acc']:.4f} val_f1={mv['f1']:.4f} rampF1={mv['ramp_f1']:.4f}")

        if mv["f1"] > best_val_f1:
            best_val_f1 = mv["f1"]
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log(f"早停 @ ep{ep}, best ep{best_epoch}")
                break

    # 载入最佳
    model.load_state_dict(best_state)
    ytr_, ptr_, prob_tr = evaluate(model, tr_loader, DEVICE)
    yva_, pva_, prob_va = evaluate(model, va_loader, DEVICE)
    yte_, pte_, prob_te = evaluate(model, te_loader, DEVICE)
    mtr = metrics(ytr_, ptr_)
    mva = metrics(yva_, pva_)
    mte = metrics(yte_, pte_)
    log(f"[train] acc={mtr['acc']:.4f} f1={mtr['f1']:.4f}")
    log(f"[val]   acc={mva['acc']:.4f} f1={mva['f1']:.4f} rampF1={mva['ramp_f1']:.4f}")
    log(f"[test]  acc={mte['acc']:.4f} f1={mte['f1']:.4f} rampF1={mte['ramp_f1']:.4f}")

    # 保存
    np.savez(os.path.join(OUT_DIR, f"dl_{args.model}_pred.npz"),
             prob_test=prob_te, y_test=yte_, t_test=d["t_test"],
             prob_val=prob_va, y_val=yva_, t_val=d["t_val"])
    torch.save(best_state, os.path.join(OUT_DIR, f"dl_{args.model}_best.pt"))

    lines = [f"# 深度学习模型实验报告（{args.model.upper()}）\n"]
    lines.append(f"设备：{DEVICE}；最佳 epoch：{best_epoch}；训练样本：{len(Xtr)}。\n")
    lines.append("## 1. 总体指标\n")
    lines.append("| 集合 | Accuracy | Macro-F1 | 是否爬坡二分类 F1 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| 训练 | {mtr['acc']:.4f} | {mtr['f1']:.4f} | - |")
    lines.append(f"| 验证 | {mva['acc']:.4f} | {mva['f1']:.4f} | {mva['ramp_f1']:.4f} |")
    lines.append(f"| 测试 | {mte['acc']:.4f} | {mte['f1']:.4f} | {mte['ramp_f1']:.4f} |")
    lines.append("\n## 2. 测试集分类报告\n")
    lines.append("```\n" + mte["report"] + "```\n")
    lines.append("## 3. 测试集混淆矩阵\n")
    lines.append("```\n" + str(mte["cm"]) + "\n```\n")
    lines.append("（行=真实 [-1,0,1]，列=预测 [-1,0,1]）")
    with open(os.path.join(OUT_DIR, f"dl_{args.model}_report.md"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    log("报告与预测已保存")


if __name__ == "__main__":
    main()