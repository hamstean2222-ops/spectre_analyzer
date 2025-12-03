import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit

# импортируем из твоего cnn_train.py
from cnn_train import SpectroDenseNet, HDMIset, load_model_from_ckpt

# ──────────────────────────────────────────────────────────
# Вспомогательные функции

def infer_f0_all(meta):
    """
    Восстановить f0 (в Гц) так же, как в cnn_train.py:
    - новый формат: (fname, t_idx, f_idx, f0_MHz, f1_MHz)
    - старый формат: (fname, t0, f0_Hz, f1_Hz)
    """
    m0 = meta[0]
    if len(m0) >= 5:
        f0_mhz = np.array([m[3] for m in meta], dtype=np.float64)
        return f0_mhz * 1e6
    else:
        return np.array([m[2] for m in meta], dtype=np.float64)


def build_split_indices(X, y, meta, seed=42):
    """
    Повторяем тот же GroupShuffleSplit, что в cnn_train.py:
    - группировка по fname
    - test_size=0.30
    - половина остатка -> val, половина -> test
    """
    groups = [str(m[0]) for m in meta]
    gss = GroupShuffleSplit(test_size=0.30, random_state=seed)
    train_idx, tmp_idx = next(gss.split(X, y, groups))
    rng = np.random.default_rng(seed)
    tmp_idx = np.asarray(tmp_idx)
    rng.shuffle(tmp_idx)
    val_size = len(tmp_idx) // 2
    val_idx = tmp_idx[:val_size]
    test_idx = tmp_idx[val_size:]
    return train_idx, val_idx, test_idx


def get_subset_indices(mode, train_idx, val_idx, test_idx):
    if mode == "train":
        return train_idx
    elif mode == "val":
        return val_idx
    elif mode == "test":
        return test_idx
    elif mode == "all":
        return np.arange(len(np.concatenate([train_idx, val_idx, test_idx])))
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ──────────────────────────────────────────────────────────
# Основная логика

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="hdmi_windows_multiclass.npz",
                        help="Датасет .npz, как в обучении")
    parser.add_argument("--ckpt", type=str, default="hdmi_cnn_best_ema.pth",
                        help="Чекпоинт EMA-модели")
    parser.add_argument("--mode", type=str, default="val",
                        choices=["train", "val", "test", "all"],
                        help="На какой части смотреть ошибки")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--out_prefix", type=str, default="errors",
                        help="Префикс выходных файлов: <prefix>_summary.csv, <prefix>_indices.npz")
    args = parser.parse_args()

    # ──────────────────────────────────────────────────────
    # Загрузка данных

    npz = np.load(args.npz, allow_pickle=True)
    X, y, w, meta = npz["X"], npz["y"], npz["w"], npz["meta"]
    ys = npz["ys"] if "ys" in npz.files else (y > 0).astype(np.float32)
    file_stats = npz["file_stats"].item() if "file_stats" in npz.files else {}
    meta_arr = np.array(meta, dtype=object)

    f0_all = infer_f0_all(meta)
    print(f"Loaded dataset: X={X.shape}, y.shape={y.shape}")
    print(f"f0 range: [{f0_all.min():.3e}, {f0_all.max():.3e}] Hz")

    # ──────────────────────────────────────────────────────
    # Сплит train/val/test как в обучении

    train_idx, val_idx, test_idx = build_split_indices(X, y, meta_arr, seed=42)
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    if args.mode == "all":
        idx = np.arange(len(X))
    else:
        if args.mode == "train":
            idx = train_idx
        elif args.mode == "val":
            idx = val_idx
        elif args.mode == "test":
            idx = test_idx
        else:
            raise ValueError("Unknown mode")

    print(f"Evaluating on subset '{args.mode}' with {len(idx)} windows.")

    # ──────────────────────────────────────────────────────
    # Dataset / DataLoader для выбранного подмножества

    subset_ds = HDMIset(
        X[idx],
        y[idx],
        ys[idx],
        w[idx],
        f0_all[idx],
        meta=meta_arr[idx],
        file_stats=file_stats,
        augment=False,
        use_domain_weights=False,
        soften_domain_weights=False,
        split_hz=float(2.5e9)   # будет перезаписан split_hz из чекпоинта
    )

    loader = DataLoader(
        subset_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # ──────────────────────────────────────────────────────
    # Модель и пороги

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SpectroDenseNet().to(device)
    model = load_model_from_ckpt(model, args.ckpt, device)
    model.eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    split_hz = float(ckpt.get("split_hz", 2.5e9))
    thr_lo = float(ckpt.get("thr_lo", 0.5))
    thr_hi = float(ckpt.get("thr_hi", 0.5))

    print(f"Loaded thresholds from ckpt: split_hz={split_hz:.3e}, thr_lo={thr_lo:.4f}, thr_hi={thr_hi:.4f}")

    # ──────────────────────────────────────────────────────
    # Прогоняем и собираем предсказания

    all_prob = []
    all_y = []
    all_f0 = []
    all_global_idx = []

    with torch.no_grad():
        base = 0
        for batch_i, (x, _yS, _wb, f0, y_hard) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            logits = model(x).squeeze(1)
            prob = torch.sigmoid(logits).cpu().numpy()

            all_prob.append(prob)
            all_y.append(y_hard.numpy())
            all_f0.append(f0.numpy())

            # глобальные индексы в исходном X/y
            batch_size = y_hard.shape[0]
            all_global_idx.append(idx[base:base + batch_size])
            base += batch_size

    all_prob = np.concatenate(all_prob)
    all_y = np.concatenate(all_y)
    all_f0 = np.concatenate(all_f0)
    all_global_idx = np.concatenate(all_global_idx)

    # ──────────────────────────────────────────────────────
    # Строим предсказанные метки с частотными порогами

    m_hi = all_f0 >= split_hz
    pred = np.zeros_like(all_prob, dtype=bool)
    pred[~m_hi] = all_prob[~m_hi] > thr_lo
    pred[m_hi] = all_prob[m_hi] > thr_hi

    y_bool = all_y.astype(bool)

    fp_mask = (pred == 1) & (y_bool == 0)
    fn_mask = (pred == 0) & (y_bool == 1)

    n_fp = int(fp_mask.sum())
    n_fn = int(fn_mask.sum())
    print(f"FP={n_fp}, FN={n_fn} on subset '{args.mode}'.")

    idx_fp = all_global_idx[fp_mask]
    idx_fn = all_global_idx[fn_mask]
    prob_fp = all_prob[fp_mask]
    prob_fn = all_prob[fn_mask]
    f0_fp = all_f0[fp_mask]
    f0_fn = all_f0[fn_mask]

    # ──────────────────────────────────────────────────────
    # Сохраняем краткую сводку в CSV (удобно смотреть глазами)

    csv_path = f"{args.out_prefix}_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("subset,err_type,global_idx,prob,f0_Hz,y_true,fname,meta_raw\n")
        # FP
        for gi, p, f0h in zip(idx_fp, prob_fp, f0_fp):
            m = meta_arr[gi]
            fname = str(m[0])
            meta_str = repr(tuple(m))
            y_true = int(y[gi] > 0)
            f.write(f"{args.mode},FP,{gi},{p:.6f},{f0h:.3f},{y_true},{fname},{meta_str}\n")
        # FN
        for gi, p, f0h in zip(idx_fn, prob_fn, f0_fn):
            m = meta_arr[gi]
            fname = str(m[0])
            meta_str = repr(tuple(m))
            y_true = int(y[gi] > 0)
            f.write(f"{args.mode},FN,{gi},{p:.6f},{f0h:.3f},{y_true},{fname},{meta_str}\n")

    print(f"Saved CSV summary to: {csv_path}")

    # ──────────────────────────────────────────────────────
    # Сохраняем индексы и вероятности в NPZ (для дальнейшей визуализации)

    npz_path = f"{args.out_prefix}_indices.npz"
    np.savez(
        npz_path,
        mode=args.mode,
        idx_fp=idx_fp,
        idx_fn=idx_fn,
        prob_fp=prob_fp,
        prob_fn=prob_fn,
        f0_fp=f0_fp,
        f0_fn=f0_fn,
        split_hz=split_hz,
        thr_lo=thr_lo,
        thr_hi=thr_hi,
    )
    print(f"Saved error indices to: {npz_path}")


if __name__ == "__main__":
    main()
