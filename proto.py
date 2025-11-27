import torch, numpy as np, pandas as pd, pathlib
import torch.nn as nn
from scipy.ndimage import median_filter
from typing import Type, List, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import json
from collections import defaultdict
import re
from collections import OrderedDict

k                  = 32      # growth-rate
compression_factor = 0.5
layers_per_block   = [6, 12, 24, 16]
in_channels        = 3

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
            nn.AvgPool2d(kernel_size=(1,2), stride=(1,2)),  # ↓ только по частоте
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
            blocks.append(SE(ch))                    # === NEW: SE после блока
            if idx != len(layers_per_block) - 1:
                tr = Transition(ch); ch = tr.out_ch
                blocks.append(tr)
        self.features = nn.Sequential(*blocks)

        self.norm = nn.BatchNorm2d(ch)
        self.act  = nn.SiLU()

        # лёгкий «хвост»: расширяем поле зрения по частоте (dilated depthwise)
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
            self.head[-1].bias.fill_(0.0)  # старт без «слишком уверенных нулей»

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.act(self.norm(x))
        x = self.tail(x)
        return self.head(x)                 # [B,1] логиты


# ГИПЕРПАРАМЕТРЫ
WIN_T, STRIDE_T = 16, 8     # ширина временоого окна; шаг движения временного окна
WIN_F, STRIDE_F = 128, 8     # ширина частотного окна; шаг движения частотного окна
THR_P           = 0.0       # порог вероятности, при котором окно учитывается, как потенциально содержащее сигнал
THR_BIN         = 0.2       # порог средней вероятности учитываемых окон, при котором конкретный бин сигнала считается содержащим сигнал 
MED_K           = 1         # (бин) размер медианного фильтра
MIN_WIN_F       = 0.0       # (МГц) порог ширины диапазона сигнала (если ширина меньше - в ответ не берем)

HIGH_THR        = 0.9
LOW_THR         = 0.6

# These will be initialised in the main block when running the script
device: torch.device | None = None
model: nn.Module | None = None


def init_model(model_path: str = "hdmi_cnn_best_ema.pth",
               device_name: str | None = None) -> None:
    """Initialise global ``device`` and ``model``.

    Parameters
    ----------
    model_path:
        Path to the checkpoint with model weights.
    device_name:
        Optional name of the device. If not provided, CUDA is used when
        available, otherwise CPU.
    """

    global device, model

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SpectroDenseNet().to(device)
    safe_load_state_dict(model, model_path, device)
    model.eval()

def center_freq_key(p: pathlib.Path) -> int:
    """Возвращает центральную частоту из имени center_<freq>MHz_… ."""
    m = re.search(r'center_(\d+)MHz', p.name)
    return int(m.group(1)) if m else 0      # fallback = 0
    
def compute_file_stats(amps: np.ndarray):
    """
    amps: (T, F) в dBm
    Возвращает mu, sigma (скаляры по файлу) и q20, robust (векторы длины F).
    """
    mu = float(amps.mean())
    sigma = float(amps.std()) + 1e-6

    # квантиль и robust std по времени для каждого бина частоты
    q20 = np.quantile(amps, 0.20, axis=0).astype(np.float32)             # (F,)
    mad = np.median(np.abs(amps - q20[None, :]), axis=0).astype(np.float32)
    robust = (1.4826 * mad + 1e-6).astype(np.float32)                    # (F,)

    return {"mu": mu, "sigma": sigma, "q20": q20, "robust": robust}


# функция получения диапазонов сигнала
def detect_ranges(csv_path: pathlib.Path):
    df = pd.read_csv(csv_path, header=0)
    freqs = df.columns[1:].astype(float).to_numpy()
    times = df.iloc[:, 0].astype(float).to_numpy()
    amps  = df.iloc[:, 1:].astype(float).to_numpy()   # (T,F)

    # ← новое: статистики по файлу для согласованной нормализации
    stats = compute_file_stats(amps)

    score = np.zeros(len(freqs), dtype=float)
    count = np.zeros_like(freqs, dtype=int)

    for t0 in range(0, len(times) - WIN_T + 1, STRIDE_T):
        for f0 in range(0, len(freqs) - WIN_F + 1, STRIDE_F):
            x = make_batch(amps, t0, f0, WIN_T, WIN_F, stats=stats, device=device)
            with torch.no_grad():
                p = model(x).sigmoid().item()
            if p > THR_P:
                f1 = f0 + WIN_F
                score[f0:f1] += p ** 3
                count[f0:f1] += 1
                
    # средняя вероятность по каждому бину
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.divide(score, count, out=np.zeros_like(score), where=count>0)

    # медианный фильтр
    avg = median_filter(avg, size=MED_K)
    bg = np.percentile(avg[~np.isnan(avg)], 60)        # фон по файлу
    bg = min(bg, 0.5)                                  # не даём фону уйти слишком высоко
    avg_adj = np.clip((avg - 0.9*bg) / (1.0 - 0.9*bg + 1e-6), 0.0, 1.0)
    
    #ranges = hysteresis_segmentation(avg, freqs)

    marked = avg >= THR_BIN

    # склейка диапазонов
    ranges, i = [], 0
    while i < len(freqs):
        if marked[i]:
            j = i
            while j < len(freqs) and marked[j]:
                j += 1

            f_start = freqs[i]
            f_end   = freqs[j-1]

            # фильтр по минимальной ширине диапазона
            if (f_end - f_start) < MIN_WIN_F:
                i = j
                continue

            ranges.append((round(f_start, 1), round(f_end, 1)))
            i = j
        else:
            i += 1
    return ranges, avg

# запись аннотации в .json файл
def save_predictions_json(pred_by_file: dict, out_path: pathlib.Path):
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(pred_by_file, f, indent=2, ensure_ascii=False)
        
def save_avg_csv(freqs: np.ndarray,
                 avg:   np.ndarray,
                 out_path: pathlib.Path):
    """
    Сохраняет пару (частота, средняя вероятность) в CSV.
    """
    pd.DataFrame({
        "freq_MHz": freqs,
        "p_avg":    avg
    }).to_csv(out_path, index=False, float_format="%.6f")

# проходим по csv файлам, ищем сигналы, пишем аннотацию
def collect_dataset(root: pathlib.Path):
    if model is None:
        init_model()

    preds_json = defaultdict(list)
    
    # подпапка, куда кладём *_probs.csv
    probs_dir = root / "probs"
    probs_dir.mkdir(exist_ok=True)
    
    for csv in sorted(root.glob("*.csv"), key=center_freq_key):
        df = pd.read_csv(csv, header = 0)
        ranges, avg = detect_ranges(csv)
        start_time = float(df.iloc[0, 0])
        end_time   = float(df.iloc[-1, 0])
        preds_json[csv.name] = [{"freq_start": a, "freq_end": b, "time_start": start_time, "time_end": end_time, "type": "SIGNAL"} for a, b in ranges]
        
        prob_path = probs_dir / f"{csv.stem}_probs.csv"
        freqs = df.columns[1:].astype(float).to_numpy()
        save_avg_csv(freqs, avg, prob_path)
        print(f"Saved probs → {prob_path.name}")

    save_predictions_json(preds_json, root / "annotations.json")
    print(f"Saved:  {root/'annotations.json'}")
    
def hysteresis_segmentation(avg: np.ndarray,
                            freqs: np.ndarray,
                            high_thr: float = 0.9,
                            low_thr : float = 0.6,
                            min_gap_hz: float = 0.5,
                            min_width_hz: float = 3.0):
    mask_hi  = avg >= high_thr
    mask_low = avg >= low_thr

    ranges = []
    i, n = 0, len(freqs)
    while i < n:
        if mask_hi[i]:
            # расширяемся, пока находимся хотя бы в low-области
            j = i + 1
            while j < n and mask_low[j]:
                j += 1
            f_start, f_end = freqs[i], freqs[j-1]

            # слишком узкие диапазоны отбрасываем
            if (f_end - f_start) >= min_width_hz:
                ranges.append((round(f_start, 1), round(f_end, 1)))
            i = j
        else:
            i += 1

    # дополнительное разделение: если зазор > min_gap_hz — точно два разных сигнала
    final = []
    for f0, f1 in ranges:
        if final and (f0 - final[-1][1]) < min_gap_hz:
            # слипаем, т.к. зазор мал
            final[-1] = (final[-1][0], f1)
        else:
            final.append((f0, f1))
    return final

def make_batch(amps, t0, f0, WIN_T=16, WIN_F=128, stats=None, device="cpu"):
    t1, f1 = t0 + WIN_T, f0 + WIN_F
    win = torch.tensor(amps[t0:t1, f0:f1], dtype=torch.float32, device=device)  # (T,F)
    x0  = win.unsqueeze(0)  # (1,T,F)

    if stats is not None:
        mu, sigma = stats["mu"], stats["sigma"]
        ch0 = (x0 - mu) / (sigma + 1e-6)

        q20 = torch.tensor(stats["q20"][f0:f1], dtype=torch.float32, device=device).view(1,1,-1)
        rob = torch.tensor(stats["robust"][f0:f1], dtype=torch.float32, device=device).view(1,1,-1).clamp_min(1e-6)

        # легкая защита от экстремально больших/маленьких rob на шумных файлах
        rob_med = rob.median()
        rob = rob.clamp(min=0.5*rob_med, max=2.0*rob_med)

        z = (x0 - q20) / rob

        # ch1: только положительные отклонения, с нормировкой по std
        ch1 = torch.relu(z)
        ch1 = ch1 / (ch1.std() + 1e-6)

        # ch2: симметричный «контраст» (центрируем и белим)
        ch2 = (z - z.mean()) / (z.std() + 1e-6)

    else:
        # fallback локальная нормализация (если stats не передали)
        ch0 = (x0 - x0.mean()) / (x0.std() + 1e-6)
        bg  = x0.quantile(q=0.20, dim=1, keepdim=True)
        mad = torch.median(torch.abs(x0 - bg), dim=1, keepdim=True).values
        z   = (x0 - bg) / (1.4826 * mad + 1e-6)

        ch1 = torch.clamp(z, min=0.0)
        ch1 = ch1 / (ch1.std() + 1e-6)

        ch2 = (z - z.mean()) / (z.std() + 1e-6)

    x = torch.cat([ch0, ch1, ch2], dim=0).unsqueeze(0)  # (1,3,T,F)
    return x



def safe_load_state_dict(model: nn.Module, path: str, device):
    ckpt = torch.load(path, map_location=device)
    # допускаем как чистый state_dict, так и словарь с 'state_dict'
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise RuntimeError(f"{path}: не похоже на state_dict")

    # снимаем префиксы от DataParallel/torch.compile
    new_sd = OrderedDict()
    for k, v in sd.items():
        k = k.replace("module.", "", 1) if k.startswith("module.") else k
        k = k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k
        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:8]}{'...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")
    return model

if __name__ == "__main__":
    init_model()

    result_dir = pathlib.Path(r"C:\Users\user\Desktop\test\case6_fhd120_cab2")

    # sanity
    df = pd.read_csv(next(iter(sorted(result_dir.glob("*.csv")))))
    amps = df.iloc[:, 1:].astype(float).to_numpy()
    stats = compute_file_stats(amps)
    x = make_batch(amps, 0, 0, WIN_T, WIN_F, stats=stats, device=device)
    with torch.no_grad():
        p = model(x).sigmoid().item()
    print("first window prob:", p)

    # классификация
    collect_dataset(result_dir.resolve())
