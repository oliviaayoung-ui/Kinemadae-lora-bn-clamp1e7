"""
4 run train metric 비교 plot:
  stage2 DiT 학습:
    ncjxc7ny: w lora+bn, stage1 lora-bn-clamp1e7 transfer
    rm57nmln: w/o lora/patchify, kk align fresh
  stage1 align 학습:
    rw5jh2hf: dit_align 104ch
    66xy0xrz: dit_align 104ch (다른 run)

train metric 위주. EMA 0.99 적용 + raw 옅게.
"""
import wandb, os
import matplotlib.pyplot as plt

api = wandb.Api()
# [NEW] stage1_combined = rw5jh2hf + 66xy0xrz cat (resume continuation)
# 사용자 노트: rw5jh2hf 가 grad_accum 2배 안 했을 거 → step 단위 다를 수 있음. 일단 step 그대로 cat (overlap 자연 처리).
RUNS = {
    "stage 2 (from neurips)": ("kplove0503/kinemadae-dit/xxjx16s4", "tab:green"),
    "stage 2 w/o lora bn":    ("kplove0503/kinemadae-dit/rm57nmln", "tab:orange"),
    "stage 2 w lora bn":      ("kplove0503/kinemadae-dit/ncjxc7ny", "tab:blue"),
}

METRIC_ALIAS = {}   # 모두 stage2 학습 — 같은 train metric, alias 불필요

# 3 stage2 run 의 train metric (모두 DiT noise prediction)
METRICS = [
    ("train/loss",         "Train Loss (total noise pred MSE, ↓)", "log", True),
    ("train/loss_z_main",  "Train Residual Latent Loss (↓)",       "log", True),
    ("train/loss_z_prior", "Train Base Latent Loss (↓)",           "log", True),
]


def ema_smooth(values, decay=0.99):
    if len(values) == 0: return []
    smoothed = []
    cur = float(values[0])
    for v in values:
        cur = decay * cur + (1.0 - decay) * float(v)
        smoothed.append(cur)
    return smoothed


# 데이터 가져오기 — combined run 은 여러 path 의 df cat
data = {}
COLORS = {}
for label, spec in RUNS.items():
    if isinstance(spec, list):
        # combined: 여러 run 의 df cat (step 그대로, duplicate 제거)
        dfs = []
        for path, color in spec:
            print(f"loading {path} (combined into {label}) ...")
            r = api.run(path)
            df_ = r.history(pandas=True, samples=20000).sort_values("_step")
            dfs.append(df_)
            print(f"  rows={len(df_)}, step range=[{df_['_step'].min()}, {df_['_step'].max()}]")
        import pandas as pd
        df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['_step'], keep='last').sort_values("_step")
        data[label] = df
        COLORS[label] = spec[0][1]
        print(f"  → combined rows={len(df)}, step range=[{df['_step'].min()}, {df['_step'].max()}]")
    else:
        path, color = spec
        print(f"loading {path} ...")
        r = api.run(path)
        df = r.history(pandas=True, samples=20000).sort_values("_step")
        data[label] = df
        COLORS[label] = color
        print(f"  rows={len(df)}, step range=[{df['_step'].min()}, {df['_step'].max()}]")

def make_plot(xlim, out_path, suffix="", include_runs=None):
    runs_to_plot = include_runs if include_runs is not None else list(RUNS.keys())
    n = len(METRICS)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for ax, (metric, title, yscale, _) in zip(axes, METRICS):
        _plot_panel(ax, metric, title, yscale, xlim, runs_to_plot)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Stage 2 DiT Training Comparison{suffix}",
                 fontsize=14, fontweight='bold', y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def _plot_panel(ax, metric, title, yscale, xlim, runs_to_plot=None):
    if runs_to_plot is None: runs_to_plot = list(RUNS.keys())
    has_data = False
    for label in runs_to_plot:
        df = data[label]
        color = COLORS[label]
        # [NEW] metric alias 적용 — 특정 RUN 에 다른 metric name 사용 시 변환
        actual_metric = METRIC_ALIAS.get(label, {}).get(metric, metric)
        if actual_metric not in df.columns:
            continue
        sub = df[["_step", actual_metric]].dropna()
        # rename 으로 일관 — 아래 코드가 metric 이름 사용
        if actual_metric != metric:
            sub = sub.rename(columns={actual_metric: metric})
        if len(sub) == 0:
            continue
        sub = sub[sub[metric].apply(lambda v: isinstance(v, (int, float)))]
        if len(sub) == 0:
            continue
        sub = sub.sort_values("_step")
        steps = sub["_step"].values
        raw = sub[metric].astype(float).values
        # train metric: raw 옅게 + EMA 0.99 진하게
        ax.plot(steps, raw, color=color, linewidth=0.8, alpha=0.20)
        smoothed = ema_smooth(raw, decay=0.99)
        ax.plot(steps, smoothed, label=label, color=color, linewidth=2.0, alpha=1.0)
        has_data = True
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("step")
    if yscale: ax.set_yscale(yscale)
    ax.grid(True, alpha=0.3)
    if xlim is not None:
        ax.set_xlim(xlim)
    if has_data:
        ax.legend(fontsize=7, loc="best")
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)


# 3 가지 plot 저장: full range + 15000 step + ncjxc7ny max step
NCJX_MAX = int(data["stage 2 w lora bn"]["_step"].max())
OUT_DIR = "/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-lora-bn-clamp1e7"
make_plot(xlim=None,            out_path=f"{OUT_DIR}/wandb_compare_4runs_train.png",                  suffix=" (full range)")
make_plot(xlim=(0, 15000),      out_path=f"{OUT_DIR}/wandb_compare_4runs_train_step15k.png",          suffix=" (step 0-15000)")
make_plot(xlim=(0, NCJX_MAX),   out_path=f"{OUT_DIR}/wandb_compare_4runs_train_ncjxmax.png",          suffix=f" (step 0-{NCJX_MAX}, ncjxc7ny max)")
# [NEW] from neurips (xxjx16s4) 제외, 2 run 만 비교
make_plot(xlim=(0, NCJX_MAX),
          out_path=f"{OUT_DIR}/wandb_compare_2runs_train_ncjxmax_no_neurips.png",
          suffix=f" (step 0-{NCJX_MAX}, w/o from-neurips)",
          include_runs=["stage 2 w/o lora bn", "stage 2 w lora bn"])
