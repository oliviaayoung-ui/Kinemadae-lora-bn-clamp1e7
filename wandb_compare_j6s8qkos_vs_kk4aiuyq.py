"""
wandb 두 run 비교 plot:
  j6s8qkos: kinemadae_lora_bn_clamp1e7 (우리 stage1)
  kk4aiuyq: dit_align_20blocks_no_noise_clamp1e5 (이전 baseline)

train/align_loss, val/lpips, val/psnr, val/cknna, val/recon 등 한 figure 에 subplot.
"""
import wandb, os
import matplotlib.pyplot as plt

api = wandb.Api()
RUNS = {
    "w lora, 20 blocks align":   "kplove0503/kinemadae/j6s8qkos",
    "w/o lora, 20 blocks align": "kplove0503/kinemadae/kk4aiuyq",
}
COLORS = {"w lora, 20 blocks align": "tab:blue", "w/o lora, 20 blocks align": "tab:orange"}

# 비교할 metrics + plot 제목 + 작을수록 좋은지
# [NOTE] val/recon = wandb image object (recon video preview) — scalar 아니라 plot 불가, 제외
METRICS = [
    ("train/align_loss",          "Train Align Loss",                "log",  True),
    ("val/lpips",                 "Val LPIPS (↓)",                   "log",  True),
    ("val/psnr",                  "Val PSNR (↑)",                    None,   False),
    ("val/cknna_z_align",         "Val CKNNA z_align (↑)",           None,   False),
    ("val/linear_cka_z_align",    "Val Linear CKA z_align (↑)",      None,   False),
    ("val/cknna_z_drift",         "Val CKNNA z_drift",               None,   False),
    ("val/linear_cka_z_drift",    "Val Linear CKA z_drift",          None,   False),
    ("val/cosine_sim_z_drift",    "Val Cosine Sim z_drift",          None,   False),
    ("train/latents_std",         "Train Latents std",               None,   False),
    ("train/rec_loss",            "Train Recon Loss (↓)",            "log",  True),
    ("train/kl_loss",             "Train KL Loss",                   None,   False),
]


def ema_smooth(values, decay=0.99):
    """EMA smoothing (wandb style). decay 클수록 smoothing 강함."""
    if len(values) == 0: return []
    smoothed = []
    cur = float(values[0])
    for v in values:
        cur = decay * cur + (1.0 - decay) * float(v)
        smoothed.append(cur)
    return smoothed

# 데이터 가져오기
data = {}
for label, path in RUNS.items():
    print(f"loading {path} ...")
    r = api.run(path)
    # [FIX] samples 충분히 (wandb default 500 → 20000) — val 의 sparse 점도 모두 보임
    df = r.history(pandas=True, samples=20000)
    df = df.sort_values("_step")
    data[label] = df
    print(f"  rows={len(df)}, step range=[{df['_step'].min()}, {df['_step'].max()}]")

# 9-panel subplot
n_metrics = len(METRICS)
n_cols = 3
n_rows = (n_metrics + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for ax, (metric, title, yscale, _) in zip(axes, METRICS):
    has_data = False
    for label, df in data.items():
        if metric not in df.columns:
            continue
        sub = df[["_step", metric]].dropna()
        if len(sub) == 0:
            continue
        # numeric value 만 (dict/list 같은 wandb media 객체 제외)
        sub = sub[sub[metric].apply(lambda v: isinstance(v, (int, float)))]
        if len(sub) == 0:
            continue
        sub = sub.sort_values("_step")
        steps = sub["_step"].values
        raw = sub[metric].astype(float).values
        if metric.startswith("val/"):
            # validation: EMA 안 적용 — eval_steps 마다 측정한 raw value (보통 점 적음)
            ax.plot(steps, raw, label=label, color=COLORS[label], linewidth=2.0, alpha=1.0, marker='o', markersize=4)
        else:
            # train: raw 옅게 (variance) + EMA 0.99 진하게
            ax.plot(steps, raw, color=COLORS[label], linewidth=0.8, alpha=0.25)
            smoothed = ema_smooth(raw, decay=0.99)
            ax.plot(steps, smoothed, label=label, color=COLORS[label], linewidth=2.0, alpha=1.0)
        has_data = True
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("step")
    if yscale: ax.set_yscale(yscale)
    ax.grid(True, alpha=0.3)
    if has_data:
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

# 남은 subplot 비우기
for ax in axes[n_metrics:]:
    ax.axis("off")

fig.suptitle("Stage1 Alignment Training Comparison: w/ LoRA vs w/o LoRA (20 blocks align)",
             fontsize=15, fontweight='bold', y=1.005)
plt.tight_layout()

out_path = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-lora-bn-clamp1e7/wandb_compare_j6s8qkos_vs_kk4aiuyq.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nsaved: {out_path}")
