# prep_windows_soft.py
import json, pathlib, random, numpy as np, pandas as pd
from tqdm import tqdm
from proto import center_freq_key

ROOT   = pathlib.Path("dataset_root")
OUT    = "hdmi_windows_multiclass.npz"

# окно/шаг
WIN_T, STRIDE_T  = 16, 4
WIN_F, STRIDE_F  = 128, 4

# физика/правила
RBW_MHZ      = 3.0    # разрешение прибора
M_SHRINK     = 0.5    # сколько «съесть» у разметки с краёв (МГц)
TAU_WIN      = 0.08   # мин. доля окна, накрытая сигналом, для POS
TAU_SIG      = 0.12   # мин. доля сигнала, накрытая окном, для POS

# soft-label
ALPHA_NOISE  = 1.2    # насколько штрафовать шум
GAMMA        = 1.0    # нелинейность по покрытиям
EPS          = 0.01   # label smoothing

# отбор/веса
KEEP_BG_PROB = 0.4       # вероятность брать фоновые окна
MIN_W, MAX_W = 0.2, 2.0   # мин/макс вес окна
MAIN_W       = 1.0        # множитель веса за основный сигнал
NOISE_W      = 8.0        # множитель веса за шум
HARD_NEG_W   = 2.0        # множитель веса за жёсткий негатив рядом с сигналом

QUALITY = {
    "SIGNAL":         0.50,  # сомнительный/слабый
    "SIGNAL_NORMAL":  0.80,  # виден хорошо
    "SIGNAL_GOOD":    0.90,  # сильный, отлично виден
    "SIGNAL_MAIN":    1.00,  # очень сильный
}

def load_trace(p: pathlib.Path):
    df = pd.read_csv(p, header=0)
    freqs = df.columns[1:].astype(float).to_numpy()
    times = df.iloc[:, 0].astype(float).to_numpy()
    amps  = df.iloc[:, 1:].astype(float).to_numpy()
    return times, freqs, amps

def build_masks(freqs, ranges):
    m_sig   = np.zeros_like(freqs, bool)
    m_main  = np.zeros_like(freqs, bool)
    m_noise = np.zeros_like(freqs, bool)

    qual_ranges, main_ranges, noise_ranges = [], [], []

    for r in ranges:
        lo, hi = float(r["freq_start"]), float(r["freq_end"])
        t = r.get("type", "")

        if t in QUALITY:
            q = QUALITY.get(t, QUALITY["SIGNAL"])
            m_sig |= (freqs >= lo) & (freqs <= hi)
            qual_ranges.append((lo, hi, q))        # <-- теперь тройки (lo, hi, качество)
            if t == "SIGNAL_MAIN":
                m_main |= (freqs >= lo) & (freqs <= hi)
                main_ranges.append((lo, hi))
        elif t == "NOISE_STRONG":
            m_noise |= (freqs >= lo) & (freqs <= hi)
            noise_ranges.append((lo, hi))

    return m_main, m_sig, m_noise, qual_ranges, main_ranges, noise_ranges

def shrink_range(lo, hi, m):
    if hi - lo <= 2*m:
        c = 0.5*(lo+hi); return c, c
    return lo+m, hi-m

def overlap_len(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))

X, y_hard, y_soft, w, meta = [], [], [], [], []

GLOBAL_STATS = {}

for case in sorted(ROOT.iterdir()):
    if not case.is_dir(): continue
    ann = json.load(open(case/"annotations.json", "r", encoding="utf-8")) if (case/"annotations.json").exists() else {}
    file_stats = {}
    case_files = sorted(case.glob("*.csv"), key=center_freq_key)

    # --- загрузка и преподсчёты по файлам ---
    loaded = []
    for csv in tqdm(case_files, desc=case.name):
        t, f, a = load_trace(csv)
        fname_key = f"{case.name}/{csv.name}"   # ← уникальное имя для мета/статистик

        BIN_W_MHZ = float(np.median(np.diff(f)))
        mu_file   = float(a.mean())
        sigma_file= float(a.std() + 1e-6)

        q20_F     = np.quantile(a, 0.20, axis=0).astype("float32")
        mad_F     = np.median(np.abs(a - q20_F), axis=0).astype("float32")
        robust_F  = (1.4826 * mad_F + 1e-6).astype("float32")

        GLOBAL_STATS[fname_key] = {
            "mu":    mu_file,
            "sigma": sigma_file,
            "q20":   q20_F,
            "robust": robust_F,
            "bin_w_mhz": BIN_W_MHZ,
        }

        m_main, m_sub, m_noise, qual_ranges, main_ranges, noise_ranges = \
            build_masks(f, ann.get(csv.name, []))

        loaded.append({
            "path": csv,
            "fname_key": fname_key,
            "times": t,
            "freqs": f,
            "amps": a,
            "m_main": m_main,
            "m_sub": m_sub,
            "m_noise": m_noise,
            "qual_ranges": qual_ranges,
            "main_ranges": main_ranges,
            "noise_ranges": noise_ranges,
        })

    # --- проход по окнам ---
    for idx, data in enumerate(loaded):
        next_data = loaded[idx + 1] if idx + 1 < len(loaded) else None

        t, f, a = data["times"], data["freqs"], data["amps"]
        m_main, m_sub, m_noise = data["m_main"], data["m_sub"], data["m_noise"]
        qual_ranges = data["qual_ranges"]
        main_ranges = data["main_ranges"]
        noise_ranges = data["noise_ranges"]
        fname_key = data["fname_key"]

        for t0 in range(0, len(t)-WIN_T+1, STRIDE_T):
            f0 = 0
            while f0 < len(f):
                f1 = f0 + WIN_F
                use_next = f1 > len(f)

                if not use_next:
                    win_f = f[f0:f1]
                    win_a = a[t0:t0+WIN_T, f0:f1]
                    win_m_main  = m_main [f0:f1]
                    win_m_sub   = m_sub  [f0:f1]
                    win_m_noise = m_noise[f0:f1]
                    win_qual_ranges  = qual_ranges
                    win_noise_ranges = noise_ranges
                else:
                    if next_data is None:
                        break
                    missing = f1 - len(f)
                    if len(next_data["freqs"]) < missing:
                        break
                    if len(next_data["times"]) < t0 + WIN_T:
                        break

                    nf, na = next_data["freqs"], next_data["amps"]
                    nm_main, nm_sub, nm_noise = next_data["m_main"], next_data["m_sub"], next_data["m_noise"]

                    win_f = np.concatenate([f[f0:], nf[:missing]])
                    win_a = np.concatenate([
                        a[t0:t0+WIN_T, f0:],
                        na[t0:t0+WIN_T, :missing]
                    ], axis=1)
                    win_m_main  = np.concatenate([m_main[f0:], nm_main[:missing]])
                    win_m_sub   = np.concatenate([m_sub [f0:], nm_sub [:missing]])
                    win_m_noise = np.concatenate([m_noise[f0:], nm_noise[:missing]])
                    win_qual_ranges  = qual_ranges + next_data["qual_ranges"]
                    win_noise_ranges = noise_ranges + next_data["noise_ranges"]

                win_lo, win_hi = float(win_f[0]), float(win_f[-1])
                Ww = win_hi - win_lo
                min_overlap = TAU_WIN*Ww

                # --- лучший сигнал (макс. перекрытие, при равенстве — выше качество)
                best_Wo, best_Ws, best_q = 0.0, 1.0, QUALITY["SIGNAL"]
                for (s_lo, s_hi, qvis) in win_qual_ranges:
                    Ws = max(0.0, s_hi - s_lo)
                    if Ws <= 0: continue
                    Wo = overlap_len(win_lo, win_hi, s_lo, s_hi)
                    if (Wo > best_Wo) or (Wo == best_Wo and qvis > best_q):
                        best_Wo, best_Ws, best_q = Wo, Ws, qvis

                # шум внутри окна (в МГц) как было
                Wn = 0.0
                for (n_lo, n_hi) in win_noise_ranges:
                    Wn += overlap_len(win_lo, win_hi, n_lo, n_hi)

                # доли покрытия
                s_win = best_Wo / max(Ww, 1e-6)
                s_sig = best_Wo / max(best_Ws, 1e-6)

                # жёсткая метка
                yH = int((best_Wo >= TAU_WIN*Ww) and (s_sig >= TAU_SIG))

                # piecewise soft + модификатор качества
                TAU_DEAD_W  = 0.06
                TAU_DEAD_S  = 0.10
                GAMMA_POS   = 1.6
                GAMMA_NEG   = 2.4
                ALPHA_NOISE = 2.2

                if (s_win < TAU_DEAD_W) or (s_sig < TAU_DEAD_S):
                    base = 0.0
                else:
                    sw = ((s_win - TAU_DEAD_W) / (1 - TAU_DEAD_W)) ** (GAMMA_POS if yH else GAMMA_NEG)
                    ss = ((s_sig - TAU_DEAD_S) / (1 - TAU_DEAD_S)) ** (GAMMA_POS if yH else GAMMA_NEG)
                    base = 0.7*sw + 0.3*ss

                # модификатор видимости (SIGNAL<...<MAIN)
                base *= best_q

                # шум
                n_win = Wn / max(Ww, 1e-6)
                r = base / (base + ALPHA_NOISE*n_win + 1e-3)

                EPS_POS = 0.01
                if yH == 1:
                    yS = EPS_POS + (1 - EPS_POS)*r
                else:
                    yS = max(0.0, r)

                # --- веса
                ov_main = win_m_main.sum()
                ov_sub  = win_m_sub .sum()
                ov_noise= win_m_noise.sum()
                frac_main  = ov_main / WIN_F
                frac_sub   = ov_sub  / WIN_F
                frac_noise = ov_noise/ WIN_F
                frac_sig   = frac_main + frac_sub

                if yH == 1:
                    weight = MIN_W + (MAX_W - MIN_W) * s_win
                    # усилить в зависимости от качества:
                    weight *= (0.7 + 0.6 * best_q)    # MAIN~1.3×, GOOD~1.25×, NORMAL~1.16×, SIGNAL~1.06×
                    if frac_main > 0 and frac_sig > 0:
                        share_main = frac_main / max(frac_sig, 1e-6)
                        weight *= (1.0 + (MAIN_W - 1.0) * share_main)
                else:
                    if best_Wo > 0.0:
                        # hard negative рядом с сильным сигналом особенно ценный
                        base_w = MIN_W + (MAX_W - MIN_W) * (1.0 - min(best_Wo/(TAU_WIN*Ww), 1.0))
                        weight = base_w * (HARD_NEG_W * (1.0 + 0.5*best_q))
                    elif frac_noise > 0.0:
                        weight = MAX_W * (1.0 + (NOISE_W - 1.0) * min(frac_noise / 0.5, 1.0))
                    else:
                        if random.random() > KEEP_BG_PROB:
                            f0 += STRIDE_F
                            continue
                        weight = MIN_W

                # --- накопление ---
                sl = win_a[None].astype("float32")
                X.append(sl)
                y_hard.append(yH)
                y_soft.append(float(yS))
                w.append(float(weight))
                meta.append((fname_key, int(t0), int(f0), float(win_f[0]), float(win_f[-1])))

                f0 += STRIDE_F

X      = np.stack(X).astype("float32")        # (N,1,16,128)
y_hard = np.asarray(y_hard, dtype=np.uint8)
y_soft = np.asarray(y_soft, dtype=np.float32)
w      = np.asarray(w,      dtype=np.float32)

np.savez_compressed(OUT, X=X, y=y_hard, ys=y_soft, w=w, meta=meta, file_stats=GLOBAL_STATS)
print("Saved", X.shape, "class counts:", np.bincount(y_hard))

print("files in meta:", len(set(m[0] for m in meta)))
print("files in stats:", len(GLOBAL_STATS))
miss = set(m[0] for m in meta) - set(GLOBAL_STATS.keys())
if miss: print("missing stats for:", list(sorted(miss))[:5])