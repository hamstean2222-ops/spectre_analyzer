import os, json, hashlib, numpy as np, torch, pathlib, argparse, random
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
import torch.nn as nn
from typing import Type, List, Union
from sklearn.metrics import precision_recall_curve
import mlflow
from collections import OrderedDict

# ──────────────────────────────────────────────────────────
# Dataset с soft/hard метками

class HDMIset(Dataset):
    def __init__(self, X, y_hard, y_soft, w, f0,
                 meta,                      # <— ДОБАВИЛИ
                 file_stats=None,           # <— ДОБАВИЛИ
                 augment=False,
                 use_domain_weights=True, soften_domain_weights=False,
                 split_hz: float = 2.5e9):

        self.X       = X
        self.y_hard  = (y_hard > 0).astype(np.float32)
        self.y_soft  = y_soft.astype(np.float32)
        self.f0      = f0.astype(np.float32)

        # имена файлов и статистики для глобальной нормализации
        self.fnames  = np.array([m[0] for m in meta], dtype=object)
        # если meta нового формата: (fname, t_idx, f_idx, f0_MHz, f1_MHz)
        # если старого — поставим -1 и уйдём в локальную нормализацию
        self.f_idx   = np.array([ (int(m[2]) if len(m) >= 5 else -1) for m in meta ], dtype=int)
        self.stats   = file_stats or {}

        # базовая нормировка доменных весов
        w = w.astype(np.float32)
        w = w / (w.mean() + 1e-9)
        w = np.clip(w, 0.0, 5.0)
        if soften_domain_weights:
            mn, mx = float(w.min()), float(w.max())
            if mx > mn:
                w = 0.7 + (w - mn) * (1.3 - 0.7) / (mx - mn + 1e-9)
            else:
                w = np.ones_like(w, dtype=np.float32)
        self.w = w
        self.use_domain_weights = bool(use_domain_weights)
        self.split_hz = float(split_hz)
        self.augment = bool(augment)

    def __len__(self):  return len(self.y_hard)

    def _augment_window(self, x2):
        # x2: (2, 16, 128)
        if random.random() < 0.5:
            dt = random.randint(-2, 2)
            if dt != 0: x2 = torch.roll(x2, shifts=dt, dims=-2)  # time
        if random.random() < 0.5:
            df = random.randint(-6, 6)
            if df != 0: x2 = torch.roll(x2, shifts=df, dims=-1)  # freq
        if random.random() < 0.3:
            w_t = random.randint(1, 2)
            t0  = random.randint(0, x2.shape[-2]-w_t)
            x2[:, t0:t0+w_t, :] = 0.0
        if random.random() < 0.3:
            w_f = random.randint(3, 6)
            f0  = random.randint(0, x2.shape[-1]-w_f)
            x2[:, :, f0:f0+w_f] = 0.0
        if random.random() < 0.8:
            x2 = x2 + 0.02 * torch.randn_like(x2)
        # baseline drift по частоте (широкополосный, как «пузырь»)
        if random.random() < 0.5:
            F = x2.shape[-1]
            f = torch.linspace(0, 1, F, device=x2.device)
            drift = (0.15 * (torch.sin(2*3.1415*(random.uniform(0.2,0.6)*f + random.random())))).view(1,1,-1)
            x2 = x2 + drift

        # редкие импульсы (по времени) — мимик шум-тык
        if random.random() < 0.3:
            k = random.randint(1, 3)
            for _ in range(k):
                t = random.randrange(x2.shape[-2])
                x2[:, t:t+1, :] += 0.5*torch.randn(1,1,x2.shape[-1], device=x2.device)

        # «штрихи» по частоте (тонкие вертикальные полоски)
        if random.random() < 0.4:
            k = random.randint(1, 3)
            for _ in range(k):
                f = random.randrange(x2.shape[-1])
                x2[:, :, f:f+1] += torch.randn(1, x2.shape[-2], 1, device=x2.device) * 0.2
        return x2

    def __getitem__(self, i):
        x  = torch.tensor(self.X[i], dtype=torch.float32)            # (1,T,Fwin)
        yS = torch.tensor(self.y_soft[i], dtype=torch.float32)
        yH = torch.tensor(self.y_hard[i], dtype=torch.float32)
        w  = torch.tensor(self.w[i], dtype=torch.float32)
        f0 = torch.tensor(self.f0[i], dtype=torch.float32)

        fname = self.fnames[i]
        stats = self.stats.get(fname, None)
        Fwin  = x.shape[-1]
        f0i   = int(self.f_idx[i])

        if (stats is not None) and (f0i >= 0) and (f0i + Fwin <= len(stats["q20"])):
            # по-файловая стандартизация
            ch0 = (x - stats["mu"]) / (stats["sigma"] + 1e-6)

            q20 = torch.as_tensor(stats["q20"][f0i : f0i + Fwin], dtype=torch.float32, device=x.device).view(1,1,-1)
            rob = torch.as_tensor(stats["robust"][f0i : f0i + Fwin], dtype=torch.float32, device=x.device).view(1,1,-1).clamp_min(1e-6)
            z   = (x - q20) / rob

            ch1 = torch.relu(z)                        # «выше фона»
            ch1 = ch1 / (ch1.std() + 1e-6)             # лёгкое беление
            ch2 = (z - z.mean()) / (z.std() + 1e-6)    # «подписанный» контраст (узкие линии видны)
        else:
            # fallback: локальная по окну
            ch0 = (x - x.mean()) / (x.std() + 1e-6)
            bg  = x.quantile(q=0.20, dim=1, keepdim=True)
            mad = torch.median(torch.abs(x - bg), dim=1, keepdim=True).values
            z   = (x - bg) / (1.4826 * mad + 1e-6)
            ch1 = torch.relu(z); ch1 = ch1 / (ch1.std() + 1e-6)
            ch2 = (z - z.mean()) / (z.std() + 1e-6)

        x2 = torch.cat([ch0, ch1, ch2], dim=0)         # (3,T,Fwin)

        if self.augment:
            x2 = self._augment_window(x2)

        if self.use_domain_weights and yH.item() == 1.0 and f0.item() >= self.split_hz:
            w = w * 1.5

        return x2, yS, w, f0, yH

# ──────────────────────────────────────────────────────────
# Модель: DenseNet с SE-attention

k                  = 16      # growth-rate
compression_factor = 0.5
layers_per_block   = [6, 12, 32, 32]
in_channels        = 3       # два канала (сырой + контраст)

class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.seq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//r, 1, bias=True), nn.SiLU(),
            nn.Conv2d(c//r, c, 1, bias=True), nn.Sigmoid()
        )
    def forward(self, x): return x * self.seq(x)

class DenseLayer(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        inter = 4 * k
        self.seq = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, inter, 1, bias=False),
            nn.BatchNorm2d(inter),
            nn.SiLU(),
            nn.Conv2d(inter, k, 3, padding=1, bias=False),
        )
    def forward(self, x): return torch.cat([x, self.seq(x)], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, n_layers, in_ch):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(n_layers):
            layers.append(DenseLayer(ch))
            ch += k
        self.block = nn.Sequential(*layers)
        self.out_ch = ch
    def forward(self, x): return self.block(x)

class Transition(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        out_ch = int(in_ch * compression_factor)
        self.out_ch = out_ch
        self.seq = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool2d(kernel_size=(1,2), stride=(1,2)),  # ↓ только по частоте (ширине)
        )
    def forward(self, x): return self.seq(x)

class SpectroDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3,7), stride=(1,2), padding=(1,3), bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        ch = 64
        blocks = []
        for idx, n in enumerate(layers_per_block):
            db = DenseBlock(n, ch);  ch = db.out_ch
            blocks.append(db)
            blocks.append(SE(ch))
            if idx != len(layers_per_block) - 1:
                tr = Transition(ch); ch = tr.out_ch
                blocks.append(tr)
        self.features = nn.Sequential(*blocks)

        self.norm = nn.BatchNorm2d(ch)
        self.act  = nn.SiLU()

        # хвост: увеличиваем поле зрения по частоте (dilated depthwise)
        self.tail = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=(1,2), dilation=(1,2), groups=ch, bias=False),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 1, bias=False),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, 1)
        )

        with torch.no_grad():
            self.head[-1].bias.fill_(0.0)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.act(self.norm(x))
        x = self.tail(x)
        return self.head(x)                 # [B,1] логиты

# ──────────────────────────────────────────────────────────
# Чекпоинты: чистое сохранение/загрузка

def get_clean_state_dict(m: nn.Module):
    # если модель скомпилирована/обёрнута DataParallel — достаём «оригинал»
    return m._orig_mod.state_dict() if hasattr(m, "_orig_mod") else m.state_dict()

def load_model_from_ckpt(model: nn.Module, path: str, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        state = ckpt
    else:
        raise RuntimeError(f"Checkpoint at {path} doesn't look like a state_dict.")
    new_state = OrderedDict()
    for k, v in state.items():
        if k.startswith("module."):    k = k[7:]
        if k.startswith("_orig_mod."): k = k[10:]
        new_state[k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[warn] missing keys: {list(missing)[:8]} (+{max(0,len(missing)-8)} more)")
    if unexpected:
        print(f"[warn] unexpected keys: {list(unexpected)[:8]} (+{max(0,len(unexpected)-8)} more)")
    return model

# ──────────────────────────────────────────────────────────

def main():

    # ──────────────────────────────────────────────────────────
    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="hdmi_windows_multiclass.npz")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--split_hz", type=float, default=2.5e9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", type=int, default=1, help="1=use light augmentations on train")
    parser.add_argument("--balance", type=int, default=1, help="1=use WeightedRandomSampler (balanced batches)")
    parser.add_argument("--pos_weight", type=float, default=0.0, help=">0 enables BCE pos_weight (used when --balance 0)")
    parser.add_argument("--use_domain_weights", type=int, default=1, help="1=use per-sample domain weights w")
    parser.add_argument("--soften_domain_weights", type=int, default=1, help="1=map w to [0.7,1.3] when balance=1")
    parser.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    args = parser.parse_args()

    ACCUM = max(1, args.accum)

    # ──────────────────────────────────────────────────────────

    # Глобальные константы / seed
    SEED       = args.seed
    SPLIT_HZ   = float(args.split_hz)
    BETA_ALL   = 2.0
    BETA_LO    = 3.0
    BETA_HI    = 2.0
    DELTA_F    = 0.01

    def seed_everything(seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed_everything()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
        
    def infer_f0_all(meta):
        # meta может быть в двух форматах:
        # (A) старый: (fname, t0, f0_HZ, f1_HZ)  → len=4, m[2] — Гц
        # (B) новый:  (fname, t_idx, f_idx, f0_MHz, f1_MHz) → len=5, m[3] — МГц
        m0 = meta[0]
        if len(m0) >= 5:
            f0_mhz = np.array([m[3] for m in meta], dtype=np.float64)
            return f0_mhz * 1e6  # конвертируем в Гц
        else:
            # уже в Гц
            return np.array([m[2] for m in meta], dtype=np.float64)

    # ──────────────────────────────────────────────────────────
    # MLflow
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "5"
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("hdmi_classifier")

    print("imports ok")

    # ──────────────────────────────────────────────────────────
    # 1) Загрузка
    NPZ_PATH = args.npz
    npz = np.load(NPZ_PATH, allow_pickle=True)          # <— allow_pickle=True ОБЯЗАТЕЛЬНО
    X, y, w, meta = npz["X"], npz["y"], npz["w"], npz["meta"]
    ys = npz["ys"] if "ys" in npz.files else (y > 0).astype(np.float32)

    file_stats = npz["file_stats"].item() if "file_stats" in npz.files else {}   # <— ДОБАВИЛИ
    meta_arr   = np.array(meta, dtype=object)                                     # <— ДОБАВИЛИ

    f0_all = infer_f0_all(meta)  # как у тебя
    print(f"f0 range (Hz): [{f0_all.min():.3e}, {f0_all.max():.3e}], split={SPLIT_HZ:.3e}")

    print("dataset ready", X.shape)

    # ──────────────────────────────────────────────────────────
    # 2) Dataset / Dataloader
    # полный датасет (нужен для BN-recalc без аугментов/весов)
    full = HDMIset(
        X, y, ys, w, f0_all,
        meta=meta_arr,                 # ← ОБЯЗАТЕЛЬНО
        file_stats=file_stats,         # ← чтобы работала глобальная нормализация
        augment=False,
        use_domain_weights=bool(args.use_domain_weights),
        soften_domain_weights=bool(args.balance and args.soften_domain_weights),
        split_hz=SPLIT_HZ
    )

    # групповой train/val/test по файлам
    groups = [str(m[0]) for m in meta]
    gss = GroupShuffleSplit(test_size=0.30, random_state=SEED)
    train_idx, tmp_idx = next(gss.split(X, y, groups))
    rng = np.random.default_rng(SEED)
    tmp_idx = np.asarray(tmp_idx); rng.shuffle(tmp_idx)
    val_size = len(tmp_idx) // 2
    val_idx, test_idx = tmp_idx[:val_size], tmp_idx[val_size:]

    # отдельные сабсеты с разными режимами
    train_ds = HDMIset(
        X[train_idx], y[train_idx], ys[train_idx], w[train_idx], f0_all[train_idx],
        meta=meta_arr[train_idx], file_stats=file_stats,
        augment=bool(args.augment),
        use_domain_weights=bool(args.use_domain_weights),
        soften_domain_weights=bool(args.balance and args.soften_domain_weights),
        split_hz=SPLIT_HZ
    )

    val_ds = HDMIset(
        X[val_idx], y[val_idx], ys[val_idx], w[val_idx], f0_all[val_idx],
        meta=meta_arr[val_idx], file_stats=file_stats,
        augment=False, use_domain_weights=False, soften_domain_weights=False,
        split_hz=SPLIT_HZ
    )

    test_ds = HDMIset(
        X[test_idx], y[test_idx], ys[test_idx], w[test_idx], f0_all[test_idx],
        meta=meta_arr[test_idx], file_stats=file_stats,
        augment=False, use_domain_weights=False, soften_domain_weights=False,
        split_hz=SPLIT_HZ
    )

    num_workers = max(os.cpu_count() // 2, 2)
    pin_mem = True

    # Самплер для балансировки батчей — по жёстким меткам
    if args.balance:
        y_train_bin = (y[train_idx] > 0).astype(int)
        class_counts = np.bincount(y_train_bin, minlength=2)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[y_train_bin]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(train_idx),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch,
                                  sampler=sampler, drop_last=True,
                                  pin_memory=pin_mem, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch,
                                  shuffle=True, drop_last=True,
                                  pin_memory=pin_mem, num_workers=num_workers)

    val_loader   = DataLoader(val_ds,   batch_size=max(512, args.batch), pin_memory=pin_mem, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=max(512, args.batch), pin_memory=pin_mem, num_workers=num_workers)

    # ──────────────────────────────────────────────────────────
    # 4) Обучение (EMA, F2, частотные пороги)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = SpectroDenseNet().to(device)
    ema    = SpectroDenseNet().to(device); ema.load_state_dict(model.state_dict())

    # channels-last ускоряет свёртки
    model.to(memory_format=torch.channels_last)
    ema.to(memory_format=torch.channels_last)

    # bfloat16 на Ada (4090) – без GradScaler
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    USE_BF16 = (device.type == 'cuda') and torch.cuda.is_bf16_supported()
    autocast_kwargs = dict(device_type='cuda', dtype=(torch.bfloat16 if USE_BF16 else torch.float16))
    AMP_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    scaler    = torch.cuda.amp.GradScaler(enabled=(AMP_DTYPE == torch.float16))
    ema_decay = 0.9995

    try:
        model = torch.compile(model, backend="eager")
        ema   = torch.compile(ema,   backend="eager")
        print("torch.compile: backend=eager (no Triton)")
    except Exception:
        print("torch.compile: disabled")

    EPOCHS         = args.epochs
    lr             = args.lr
    weight_decay   = args.weight_decay

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr, weight_decay=weight_decay,
        fused=False,          # ← выключаем fused-ядро
        foreach=False         # ← чтобы не смешивал типы
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    # Лосс: режимы
    use_pos_weight = (not args.balance) and (args.pos_weight > 0.0)
    if use_pos_weight:
        pos_weight = torch.tensor([args.pos_weight], device=device)
        criterion  = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    else:
        criterion  = nn.BCEWithLogitsLoss(reduction='none')  # без pos_weight

    use_domain_weights = bool(args.use_domain_weights)
    scaler = torch.cuda.amp.GradScaler()

    best_f2, best_path = 0.0, "hdmi_cnn_best_ema.pth"

    def file_sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1<<20), b""):
                h.update(chunk)
        return h.hexdigest()

    @torch.no_grad()
    def ema_update(student, teacher, decay=ema_decay):
        for p_s, p_t in zip(student.parameters(), teacher.parameters()):
            p_t.data.mul_(decay).add_(p_s.data * (1.0 - decay))

    @torch.no_grad()
    def copy_bn_buffers(src: nn.Module, dst: nn.Module):
        for m_s, m_e in zip(src.modules(), dst.modules()):
            if isinstance(m_s, nn.BatchNorm2d):
                m_e.running_mean.data.copy_(m_s.running_mean.data)
                m_e.running_var.data.copy_(m_s.running_var.data)
                m_e.num_batches_tracked.data.copy_(m_s.num_batches_tracked.data)

    @torch.no_grad()
    def bn_recalc_on_loader(model, loader, device, max_batches: int = 150):
        model.train()
        seen = 0
        for x, *_ in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            _ = model(x)
            seen += 1
            if seen >= max_batches: break
        model.eval()


    def best_thr_fbeta(y, p, beta=2.0, delta=0.01, default=None):
        y = np.asarray(y); p = np.asarray(p)
        if y.size == 0:
            return (0.5 if default is None else float(default)), float("nan")
        uniq = np.unique(y)
        if uniq.size == 1:
            thr = 1.0 if uniq[0] == 0 else 0.0
            return (float(default) if default is not None else thr), float("nan")
        prec, rec, thr = precision_recall_curve(y, p)
        f = (1+beta**2) * prec * rec / (beta**2 * prec + rec + 1e-9)
        fmax = f.max()
        good = np.where(f >= fmax - delta)[0]
        thr_list = []
        for k in good:
            if k == 0:           thr_list.append(1.0)
            elif k == len(thr):  thr_list.append(thr[-1])
            else:                thr_list.append(thr[k-1])
        return float(np.mean(thr_list)), float(fmax)

    def metrics_with_band_thresholds(y, p, f0, thr_lo, thr_hi, split=SPLIT_HZ):
        y = np.asarray(y); p = np.asarray(p); f0 = np.asarray(f0)
        m_hi = f0 >= split
        pred = np.zeros_like(p, dtype=bool)
        pred[~m_hi] = p[~m_hi] > thr_lo
        pred[ m_hi] = p[ m_hi] > thr_hi
        TP = ((pred == 1) & (y == 1)).sum()
        FP = ((pred == 1) & (y == 0)).sum()
        FN = ((pred == 0) & (y == 1)).sum()
        P  = TP / (TP + FP + 1e-9)
        R  = TP / (TP + FN + 1e-9)
        F1 = 2*P*R/(P+R+1e-9)
        beta=2.0
        F2 = (1+beta**2)*P*R/(beta**2*P+R+1e-9)
        return P, R, F1, F2
        
    # перед эпохами
    def lambda_schedule(epoch, total, start=0.7, end=0.3):
        """Косинусный спад доли soft во времени: от start к end."""
        if total <= 1:
            return end
        t = (epoch - 1) / (total - 1)
        import math
        return end + 0.5 * (start - end) * (1 + math.cos(math.pi * t))


    run_name = f"SpectroDenseNet_aug{args.augment}_bal{args.balance}_pw{args.pos_weight}_split_{int(SPLIT_HZ/1e6)}MHz"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "arch": "SpectroDenseNet+SE+dilated",
            "in_channels": 2,
            "lr": lr,
            "weight-decay": weight_decay,
            "epochs": EPOCHS,
            "batch": args.batch,
            "growth-rate": k,
            "layers": sum(layers_per_block),
            "ema_decay": ema_decay,
            "split_hz": SPLIT_HZ,
            "seed": SEED,
            "npz_path": NPZ_PATH,
            "npz_sha256": file_sha256(NPZ_PATH),
            "augment": int(args.augment),
            "balance": int(args.balance),
            "pos_weight": float(args.pos_weight),
            "use_domain_weights": int(args.use_domain_weights),
            "soften_domain_weights": int(args.soften_domain_weights),
            "accum": ACCUM,
            "uses_soft_labels": int("ys" in npz.files),
        })

        for epoch in range(1, EPOCHS + 1):
            lam_epoch = lambda_schedule(epoch, EPOCHS, start=0.7, end=0.3)

            model.train()
            running = 0.0
            opt.zero_grad(set_to_none=True)
            step_in_epoch = 0

            for x, y_soft, wb, _, y_hard in train_loader:
                step_in_epoch += 1
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                y_soft = y_soft.to(device, non_blocking=True)
                y_hard = y_hard.to(device, non_blocking=True)
                wb     = wb.to(device, non_blocking=True)

                conf = (1.0 - torch.abs(y_soft - y_hard)).clamp(0.0, 1.0)
                lam  = lam_epoch * conf
                y_mix = lam * y_soft + (1.0 - lam) * y_hard

                with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                    logits = model(x).squeeze(1)
                    loss_per = criterion(logits, y_mix)
                    loss = (loss_per * wb).mean() if use_domain_weights else loss_per.mean()

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if step_in_epoch % ACCUM == 0:
                        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); ema_update(model, ema)
                else:
                    loss.backward()
                    if step_in_epoch % ACCUM == 0:
                        opt.step(); opt.zero_grad(set_to_none=True); ema_update(model, ema)


                running += loss.item() * x.size(0)

            # «хвост» градиентов, если батчи не кратны ACCUM
            if step_in_epoch % ACCUM != 0:
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                ema_update(model, ema)

            scheduler.step()
            train_loss = running / len(train_loader.dataset)

            # === BN-калибровка EMA (без аугментов/самплера) ===
            ema.eval()
            copy_bn_buffers(model, ema)
            bn_ds = HDMIset(
                X[train_idx], y[train_idx], ys[train_idx], w[train_idx], f0_all[train_idx],
                meta=meta_arr[train_idx], file_stats=file_stats,
                augment=False, use_domain_weights=False, soften_domain_weights=False,
                split_hz=SPLIT_HZ
            )
            bn_loader = DataLoader(bn_ds, batch_size=512, shuffle=True, num_workers=num_workers, pin_memory=True)
            bn_recalc_on_loader(ema, bn_loader, device, max_batches=100)

            # ─── валидация на EMA ───
            ema.eval()
            all_prob, all_yhard, all_f0 = [], [], []
            with torch.no_grad():
                for x, _yS, _wb, f0, y_hard in val_loader:
                    prob = ema(x.to(device)).squeeze(1).sigmoid().cpu().numpy()
                    all_prob.append(prob)
                    all_yhard.append(y_hard.numpy())
                    all_f0.append(f0.numpy())
            all_prob = np.concatenate(all_prob); all_y = np.concatenate(all_yhard); all_f0 = np.concatenate(all_f0)

            # маски диапазонов
            mask_hi = all_f0 >= SPLIT_HZ

            # общий порог как default
            thr_all, _ = best_thr_fbeta(all_y, all_prob, beta=BETA_ALL, delta=DELTA_F)
            thr_lo, f_lo = best_thr_fbeta(all_y[~mask_hi], all_prob[~mask_hi],
                                          beta=BETA_LO, delta=DELTA_F, default=thr_all)
            thr_hi, f_hi = best_thr_fbeta(all_y[ mask_hi], all_prob[ mask_hi],
                                          beta=BETA_HI, delta=DELTA_F, default=thr_lo)

            P, R, F1, F2 = metrics_with_band_thresholds(all_y, all_prob, all_f0, thr_lo, thr_hi, split=SPLIT_HZ)

            print(f"E{epoch:02d}  loss={train_loss:.4f}  F2={F2:.3f}  P={P:.2f} R={R:.2f}  "
                  f"thr_lo={thr_lo:.4f} thr_hi={thr_hi:.4f}  lam={lam_epoch:.2f}")
            mlflow.log_metric("train_loss", train_loss, epoch)
            mlflow.log_metric("val_F2",     F2,         epoch)
            mlflow.log_metric("val_P",      P,          epoch)
            mlflow.log_metric("val_R",      R,          epoch)
            mlflow.log_metric("thr_lo",     thr_lo,     epoch)
            mlflow.log_metric("thr_hi",     thr_hi,     epoch)
            mlflow.log_metric("lam_epoch",  lam_epoch,  epoch)

            if F2 > best_f2:
                best_f2 = F2
                ckpt = {
                    "state_dict": get_clean_state_dict(ema),   # ← чистые ключи
                    "arch": "SpectroDenseNet",
                    "split_hz": SPLIT_HZ,
                    "thr_lo": float(thr_lo),
                    "thr_hi": float(thr_hi),
                }
                torch.save(ckpt, "hdmi_cnn_best_ema.pth")
                with open("band_thresholds.json", "w") as f:
                    json.dump({"split_hz": SPLIT_HZ, "thr_lo": float(thr_lo), "thr_hi": float(thr_hi)}, f, indent=2)
                mlflow.log_artifact("hdmi_cnn_best_ema.pth", "checkpoints")
                mlflow.log_artifact("band_thresholds.json", "thresholds")
                print("  ✓ new best")

    # ──────────────────────────────────────────────────────────
    # 5) Тест с частотными порогами (взяты с валидации EMA)

    torch.cuda.synchronize()
    del model, opt, scheduler
    try: del scaler
    except: pass
    torch.cuda.empty_cache()
    
    ema = SpectroDenseNet().to(device)
    ema = load_model_from_ckpt(ema, "hdmi_cnn_best_ema.pth", device)
    ema.eval()

    torch.cuda.synchronize()

    all_prob, all_yhard, all_f0 = [], [], []
    with torch.no_grad():
        for x, _yS, _wb, f0, y_hard in val_loader:
            x = x.to(device, non_blocking=True)
            # на всякий: eval без AMP
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=False):
                pr = ema(x).squeeze(1).sigmoid().float().cpu().numpy()
            all_prob.append(pr)
            all_yhard.append(y_hard.numpy())
            all_f0.append(f0.numpy())

    all_prob = np.concatenate(all_prob); all_y = np.concatenate(all_yhard); all_f0 = np.concatenate(all_f0)
    mask_hi = all_f0 >= SPLIT_HZ
    thr_lo, _ = best_thr_fbeta(all_y[~mask_hi], all_prob[~mask_hi], beta=BETA_LO, delta=DELTA_F)
    thr_hi, _ = best_thr_fbeta(all_y[ mask_hi], all_prob[ mask_hi], beta=BETA_HI, delta=DELTA_F)

    # метрики на тесте
    tp=fp=fn=0
    tp_lo=fp_lo=fn_lo=0
    tp_hi=fp_hi=fn_hi=0

    with torch.no_grad():
        for x, _yS, _wb, f0, y_hard in test_loader:
            pr = ema(x.to(device)).squeeze(1).sigmoid().cpu().numpy()
            yb = y_hard.numpy().astype(bool)
            f0n= f0.numpy()
            m_hi = f0n >= SPLIT_HZ
            pred = np.zeros_like(pr, dtype=bool)
            pred[~m_hi] = pr[~m_hi] > thr_lo
            pred[ m_hi] = pr[ m_hi] > thr_hi

            tp += np.logical_and(pred, yb).sum()
            fp += np.logical_and(pred, ~yb).sum()
            fn += np.logical_and(~pred, yb).sum()

            tp_lo += np.logical_and(pred[~m_hi], yb[~m_hi]).sum()
            fp_lo += np.logical_and(pred[~m_hi], ~yb[~m_hi]).sum()
            fn_lo += np.logical_and(~pred[~m_hi], yb[~m_hi]).sum()

            tp_hi += np.logical_and(pred[m_hi], yb[m_hi]).sum()
            fp_hi += np.logical_and(pred[m_hi], ~yb[m_hi]).sum()
            fn_hi += np.logical_and(~pred[m_hi], yb[m_hi]).sum()

    P  = tp / (tp + fp + 1e-9);   R  = tp / (tp + fn + 1e-9)
    F1 = 2*P*R/(P+R+1e-9);        F2 = (1+4)*P*R/(4*P+R+1e-9)
    P_lo = tp_lo/(tp_lo+fp_lo+1e-9); R_lo = tp_lo/(tp_lo+fn_lo+1e-9)
    P_hi = tp_hi/(tp_hi+fp_hi+1e-9); R_hi = tp_hi/(tp_hi+fn_hi+1e-9)

    mlflow.log_metric("test_P", P);       mlflow.log_metric("test_R", R)
    mlflow.log_metric("test_F1", F1);     mlflow.log_metric("test_F2", F2)
    mlflow.log_metric("test_R_lo", R_lo); mlflow.log_metric("test_R_hi", R_hi)
    mlflow.log_param("test_thr_lo", float(thr_lo))
    mlflow.log_param("test_thr_hi", float(thr_hi))
    mlflow.log_param("test_split_hz", SPLIT_HZ)

    print(f"TEST  P={P:.3f} R={R:.3f} F1={F1:.3f} F2={F2:.3f} | "
          f"R_lo={R_lo:.3f} R_hi={R_hi:.3f} | thr_lo={thr_lo:.4f} thr_hi={thr_hi:.4f}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
