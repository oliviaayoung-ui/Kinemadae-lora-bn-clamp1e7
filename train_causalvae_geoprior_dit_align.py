# [NEW - oliviaa/dit_align] Geoprior VAE + DiT intermediate feature alignment training.
#
# 기존 geoprior VAE 학습에 DiT feature alignment loss 추가.
# Teacher: pretrained Wan VAE → pretrained DiT (all frozen)
# Student: geoprior z_cat (48ch) → expanded patchify → same DiT blocks (grad checkpoint)
# Alignment: per-layer feature MSE/cosine after parameter-free upsample.
#
# 가져온 것:
#   [COPIED] train_causalvae.py — 전체 training infrastructure
#   [COPIED] train_causalvae_geoprior.py — geoprior args pre-parse + _video_vae patch
#   [COPIED] dit_feature_extractor.py — load_pipeline, prepare_y, prepare_clip_feature, prepare_null_context
#   [COPIED] WanModel.forward() 구조 — blocks 순회 + gradient_checkpoint_forward
#
# 새로 작성한 것:
#   [NEW] GeopriorDiTAlignModel — forward에서 z_cat 노출하는 wrapper
#   [NEW] compute_alignment_loss — per-layer feature alignment (MSE/cosine)
#   [NEW] create_student_patchify — DiT patchify 확장 (36ch→104ch)
#   [NEW] dit_forward_with_features — blocks 순회하면서 features 수집

# ─── Geoprior pre-parse ──────────────────────────────────────
# [COPIED from train_causalvae_geoprior.py:1-55]
# geoprior 전용 args를 먼저 파싱 후 sys.argv에서 제거.
# _video_vae → _video_vae_geoprior로 패치.
import argparse
import sys
from functools import partial
import json as _json

import kinemadae_geoprior as kinemadae

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--subsample_mode', default='avg_pool',
                     choices=['avg_pool', 'stride', 'bilinear'])
_parser.add_argument('--no_dual_branch', action='store_true')
_parser.add_argument('--prior_z_dim', type=int, default=16)
_parser.add_argument('--add_decoder_tail_stages', type=str, default=None)
_parser.add_argument('--add_decoder_before_head_stages', type=str, default=None)
_parser.add_argument('--decoder_conv1_zmain_init', type=str, default='zero',
                     choices=['zero', 'pretrained'])
_parser.add_argument('--no_expand_conv2', action='store_true')
_parser.add_argument('--expand_encoder_head', action='store_true')
_known, _remaining = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

dual_branch = not _known.no_dual_branch
_tail_stages = _json.loads(_known.add_decoder_tail_stages) if _known.add_decoder_tail_stages else None
_before_head_stages = _json.loads(_known.add_decoder_before_head_stages) if _known.add_decoder_before_head_stages else None
kinemadae._video_vae = partial(
    kinemadae._video_vae_geoprior,
    dual_branch=dual_branch,
    subsample_mode=_known.subsample_mode,
    prior_z_dim=_known.prior_z_dim,
    add_decoder_tail_stages=_tail_stages,
    add_decoder_before_head_stages=_before_head_stages,
    decoder_conv1_zmain_init=_known.decoder_conv1_zmain_init,
    expand_conv2=not _known.no_expand_conv2,
    expand_encoder_head=_known.expand_encoder_head,
)
sys.modules['kinemadae'] = kinemadae

# ─── Imports ─────────────────────────────────────────────────
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    HAS_FSDP2 = True
except ImportError:
    HAS_FSDP2 = False


from dit_align import (
    compute_alignment_loss, create_student_patchify,
    load_pipeline, prepare_null_context,
    FlowMatchScheduler,
    run_teacher_forward, run_student_forward, setup_dit_memory,
    setup_text_encoder_memory,
    fused_dit_align_forward,
)

from torch.utils.data import DataLoader, DistributedSampler, Subset
from PIL import Image
import logging
from colorlog import ColoredFormatter
import tqdm
from itertools import chain
import wandb
from typing import Union, Tuple
import random
import numpy as np
from pathlib import Path
from einops import rearrange
import time

try:
    import lpips
except:
    raise Exception("Need lpips to valid.")

# [COPIED from train_causalvae.py] Local imports
from kinemadae import WanVAE_, _video_vae
from perceptual_loss import LPIPSWithDiscriminator3D
from ema_model import EMA
from ddp_sampler import CustomDistributedSampler
from video_dataset import TrainVideoDataset, ValidVideoDataset
from video_utils import tensor_to_video
from distrib_utils import DiagonalGaussianDistribution

# [Modified - oliviaa] External library paths
#   Default: <repo_root>/external/DiffSynth-Studio  and  <repo_root>/external/dit_probing
#   Override via env var: KINEMADAE_DIFFSYNTH_PATH, KINEMADAE_PROBING_PATH
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DIFFSYNTH = os.environ.get(
    "KINEMADAE_DIFFSYNTH_PATH",
    os.path.join(_REPO_ROOT, "external", "DiffSynth-Studio"),
)
if _DIFFSYNTH not in sys.path:
    sys.path.insert(0, _DIFFSYNTH)
_PROBING = os.environ.get(
    "KINEMADAE_PROBING_PATH",
    os.path.join(_REPO_ROOT, "external", "dit_probing"),
)
if _PROBING not in sys.path:
    sys.path.insert(0, _PROBING)


# ─── NEW: DiT alignment 전용 클래스/함수 ─────────────────────

class GeopriorDiTAlignModel(nn.Module):
    """[NEW - oliviaa/dit_align] Geoprior VAE wrapper.
    kinemadae_geoprior.py forward() 가 z_cat 을 반환하지 않아서 wrapper 필요.
    encode → reparameterize → _encode_prior → cat → decode 를 직접 호출하여 z_cat 노출.
    student_patchify 는 DDP 밖에서 별도 관리 (forward 밖에서 사용되므로 DDP 충돌 방지).
    """
    # pretrained Wan2.1 VAE z_prior 통계 (frozen prior encoder 출력용)
    _prior_mean = torch.tensor([-0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
                                 0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921])
    _prior_inv_std = torch.tensor([1.0/2.8184, 1.0/1.4541, 1.0/2.3275, 1.0/2.6558, 1.0/1.2196, 1.0/1.7708,
                                    1.0/2.6052, 1.0/2.0743, 1.0/3.2687, 1.0/2.1526, 1.0/2.8652, 1.0/1.5579,
                                    1.0/1.6382, 1.0/1.1253, 1.0/2.8251, 1.0/1.9160])

    def __init__(self, vae, normalize_zprior=False, zmain_stats=None,
                 decoder_noise_tau_main=0.0, decoder_noise_tau_prior=0.0,
                 decoder_noise_random_mode=True,
                 decoder_noise_warmup_steps=0,
                 decoder_noise_warmup_power=1.0,
                 align_weight=0.0, align_adaptive_weight=False,
                 use_2backward_adaptive=False, adaptive_max_weight=1e4,
                 normalize_zmain_bn=False, bn_momentum=0.1, zmain_bn_init='zprior',
                 z_dim=16):
        super().__init__()
        self.vae = vae
        self.normalize_zprior = normalize_zprior
        if normalize_zprior:
            self.register_buffer('prior_mean', self._prior_mean.clone())
            self.register_buffer('prior_inv_std', self._prior_inv_std.clone())
        # [NEW - oliviaa] 변종 B: z_main도 precomputed stats로 정규화 후 alignment에 흘림.
        # 정규화된 z_main을 student_patchify에 입력 → patchify가 normed 분포에 calibrate
        # → Stage 2에서 같은 정규화 적용 시 patchify 직접 reuse 가능 (수학 변환 불필요).
        self.normalize_zmain = zmain_stats is not None
        if self.normalize_zmain:
            zm_mean = torch.tensor(zmain_stats['mean'], dtype=torch.float32)
            zm_std  = torch.tensor(zmain_stats['std'],  dtype=torch.float32)
            self.register_buffer('zmain_mean', zm_mean)
            self.register_buffer('zmain_inv_std', 1.0 / zm_std)
        # [NEW] REPA-E style: BN3d for z_main online stats learning.
        # train mode: batch stats normalize + running_stats EMA update.
        # eval mode: running stats normalize. ckpt 에 running stats 저장 → stage2 호환.
        self.normalize_zmain_bn = normalize_zmain_bn
        if normalize_zmain_bn:
            self.zmain_bn = nn.BatchNorm3d(
                z_dim, eps=1e-4, momentum=bn_momentum,
                affine=False,  # gamma/beta 없음 (z_prior 와 일관)
                track_running_stats=True,
            )
            if zmain_bn_init == 'zprior' and normalize_zprior:
                # REPA-E init_bn 과 동등: 사전 측정 stats 로 running_mean/var init
                with torch.no_grad():
                    self.zmain_bn.running_mean.copy_(self.prior_mean)
                    self.zmain_bn.running_var.copy_((1.0 / self.prior_inv_std).pow(2))
            elif zmain_bn_init == 'cold':
                with torch.no_grad():
                    self.zmain_bn.running_mean.zero_()
                    self.zmain_bn.running_var.fill_(1.0)
            # 'pytorch_default' = nn.BatchNorm3d 기본 init 그대로 (running_mean=0, var=1)
        # [NEW - oliviaa] RAE-style decoder noise augmentation
        # Per-sample random sigma in [0, tau], applied in normalized z space.
        # See: RAE paper (Diffusion Transformers with Representation Autoencoders).
        # Random mode per-sample: 0=main_only, 1=prior_only, 2=both
        # Curriculum: linearly ramp tau from 0 to final value over warmup_steps
        # (to avoid cold-start shock when resuming from clean-trained decoder).
        self.decoder_noise_tau_main = decoder_noise_tau_main
        self.decoder_noise_tau_prior = decoder_noise_tau_prior
        self.decoder_noise_random_mode = decoder_noise_random_mode
        self.decoder_noise_warmup_steps = decoder_noise_warmup_steps
        # power > 1: slow start, fast end (quadratic/cubic)
        # power = 1: linear
        # power < 1: fast start, slow end
        self.decoder_noise_warmup_power = decoder_noise_warmup_power
        # Counter for curriculum (per training forward call; auto-increments in self.training mode)
        self.register_buffer('_decoder_noise_step', torch.tensor(0, dtype=torch.long))
        self.align_weight = align_weight
        self.align_adaptive_weight = align_adaptive_weight
        # [NEW] 2-backward adaptive mode — mutually exclusive with _AdaptiveWeightingFn.
        # When True, forward() must NOT split z_cat; train loop calls
        # compute_adaptive_weight_2bwd() and scales align_loss explicitly.
        self.use_2backward_adaptive = use_2backward_adaptive
        self.adaptive_max_weight = adaptive_max_weight

    def _norm_zprior(self, z_prior):
        if self.normalize_zprior:
            return (z_prior - self.prior_mean.view(1, -1, 1, 1, 1).to(z_prior)) * self.prior_inv_std.view(1, -1, 1, 1, 1).to(z_prior)
        return z_prior

    def _denorm_zprior(self, z_prior):
        if self.normalize_zprior:
            return z_prior / self.prior_inv_std.view(1, -1, 1, 1, 1).to(z_prior) + self.prior_mean.view(1, -1, 1, 1, 1).to(z_prior)
        return z_prior

    # [NEW - oliviaa] z_main normalize/denormalize (변종 B)
    def _norm_zmain(self, z_main):
        if self.normalize_zmain_bn:
            # REPA-E style BN: train mode 면 batch stats, eval mode 면 running stats
            return self.zmain_bn(z_main)
        if self.normalize_zmain:
            return (z_main - self.zmain_mean.view(1, -1, 1, 1, 1).to(z_main)) * self.zmain_inv_std.view(1, -1, 1, 1, 1).to(z_main)
        return z_main

    def _denorm_zmain(self, z_main):
        if self.normalize_zmain_bn:
            # BN 의 역변환: running stats 사용 (eval-style)
            mean = self.zmain_bn.running_mean.view(1, -1, 1, 1, 1).to(z_main)
            std = (self.zmain_bn.running_var + self.zmain_bn.eps).sqrt().view(1, -1, 1, 1, 1).to(z_main)
            return z_main * std + mean
        if self.normalize_zmain:
            return z_main / self.zmain_inv_std.view(1, -1, 1, 1, 1).to(z_main) + self.zmain_mean.view(1, -1, 1, 1, 1).to(z_main)
        return z_main

    def forward(self, x):
        mu, log_var = self.vae.encode(x, scale=None)
        z_main = self.vae.reparameterize(mu, log_var)
        with torch.no_grad():
            z_prior = self.vae._encode_prior(x)
            z_prior = self._norm_zprior(z_prior)
        # [NEW - oliviaa] alignment 경로: z_main을 정규화 (변종 B)
        # decode 경로는 여전히 raw z_main 사용 (VAE 사전학습 호환)
        z_main_align = self._norm_zmain(z_main)
        z_cat = torch.cat([z_main_align, z_prior], dim=1)

        # Adaptive gradient weighting: split z_cat so that g_loss and align_loss
        # both flow back through the same representation level.  z_cat_rec feeds
        # the decoder (→ recon → disc); z_cat is returned for the alignment path.
        # Only active during training and when align_adaptive_weight is enabled
        # AND we are NOT in legacy 2-backward mode (use_2backward_adaptive=True
        # disables the autograd.Function split — adaptive weight comes from
        # compute_adaptive_weight_2bwd() in the train loop instead).
        if (self.training and self.align_adaptive_weight and self.align_weight > 0
                and not self.use_2backward_adaptive):
            z_cat_rec, z_cat = _AdaptiveWeightingFn.apply(
                z_cat, z_cat.clone(), self.align_weight, 1e-6, self.adaptive_max_weight)
        else:
            z_cat_rec = z_cat

        # [NEW - oliviaa] RAE-style decoder noise augmentation
        # Per-sample uniform sigma ∈ [0, tau_eff], applied in NORMALIZED space.
        # tau_eff = tau * warmup_factor (curriculum: linear ramp 0 → tau over warmup_steps).
        # Random mode per-sample: 0=main_only, 1=prior_only, 2=both
        # Noise is applied to z_cat_rec (the decoder branch after the AW split).
        z_main_rec = z_cat_rec[:, :z_main.shape[1]]
        z_prior_rec = z_cat_rec[:, z_main.shape[1]:]
        if self.training and (self.decoder_noise_tau_main > 0 or self.decoder_noise_tau_prior > 0):
            B = z_main_rec.shape[0]
            device = z_main_rec.device
            dtype = z_main_rec.dtype

            # Curriculum warmup factor (power schedule: (t/T)^power)
            # power=1: linear, power=2: quadratic (slow start, fast end)
            if self.decoder_noise_warmup_steps > 0:
                cur = int(self._decoder_noise_step.item())
                progress = min(cur / float(self.decoder_noise_warmup_steps), 1.0)
                warmup_factor = progress ** self.decoder_noise_warmup_power
            else:
                warmup_factor = 1.0
            tau_main_eff = self.decoder_noise_tau_main * warmup_factor
            tau_prior_eff = self.decoder_noise_tau_prior * warmup_factor

            # Increment counter (only in training)
            self._decoder_noise_step += 1

            if self.decoder_noise_random_mode:
                # Per-sample random mode (1/4 each):
                #   0: clean (★no noise either, preserves clean recon ability★)
                #   1: main_only (z_main noise, z_prior clean)
                #   2: prior_only (z_prior noise, z_main clean)
                #   3: both noisy
                mode = torch.randint(0, 4, (B,), device=device)
                apply_main = ((mode == 1) | (mode == 3)).view(B, 1, 1, 1, 1).to(dtype)
                apply_prior = ((mode == 2) | (mode == 3)).view(B, 1, 1, 1, 1).to(dtype)
            else:
                apply_main = torch.ones((B, 1, 1, 1, 1), device=device, dtype=dtype)
                apply_prior = torch.ones((B, 1, 1, 1, 1), device=device, dtype=dtype)

            # z_main
            if tau_main_eff > 0 and self.normalize_zmain:
                sigma_m = tau_main_eff * torch.rand(
                    (B, 1, 1, 1, 1), device=device, dtype=dtype) * apply_main
                z_main_dec = self._denorm_zmain(z_main_rec + sigma_m * torch.randn_like(z_main_rec))
            else:
                z_main_dec = self._denorm_zmain(z_main_rec)

            # z_prior
            if tau_prior_eff > 0 and self.normalize_zprior:
                sigma_p = tau_prior_eff * torch.rand(
                    (B, 1, 1, 1, 1), device=device, dtype=dtype) * apply_prior
                z_prior_dec = self._denorm_zprior(z_prior_rec + sigma_p * torch.randn_like(z_prior_rec))
            else:
                z_prior_dec = self._denorm_zprior(z_prior_rec)

            z_cat_raw = torch.cat([z_main_dec, z_prior_dec], dim=1)
        else:
            z_cat_raw = torch.cat([self._denorm_zmain(z_main_rec), self._denorm_zprior(z_prior_rec)], dim=1)
        recon = self.vae.decode(z_cat_raw, scale=None)
        return recon, mu, log_var, z_cat




class _AdaptiveWeightingFn(torch.autograd.Function):
    """Adaptive loss weighting via gradient norm equalization.

    Splits a shared tensor into two branches (x → loss_a, y → loss_b).
    In backward, scales grad_y so that its norm equals alpha * ||grad_x||,
    making loss_b contribute alpha times the gradient magnitude of loss_a.
    All-reduces norms across DDP ranks so every rank applies the same coefficient.
    """

    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                y: torch.Tensor,
                alpha: Union[float, torch.Tensor],
                eps: float,
                max_weight: float = 1e4) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx.alpha = alpha
        ctx.eps = eps
        ctx.max_weight = max_weight
        return x, y

    @staticmethod
    def backward(ctx, grad_x, grad_y):
        if grad_x is None or grad_y is None:
            return grad_x, grad_y, None, None, None

        work_dtype = torch.promote_types(grad_y.dtype, torch.float32)
        nx_sq = grad_x.detach().to(work_dtype).pow(2).sum()
        ny_sq = grad_y.detach().to(work_dtype).pow(2).sum()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(nx_sq, op=dist.ReduceOp.SUM)
            dist.all_reduce(ny_sq, op=dist.ReduceOp.SUM)

        alpha = ctx.alpha
        if torch.is_tensor(alpha):
            alpha = alpha.detach().to(nx_sq.dtype)
        ratio = (nx_sq / ny_sq.clamp_min(ctx.eps)).sqrt()
        if ctx.max_weight > 0:
            ratio = ratio.clamp(0.0, ctx.max_weight)
        c = (alpha * ratio).to(grad_y.dtype)
        _AdaptiveWeightingFn._last_c = c.detach()

        return grad_x, c * grad_y, None, None, None


# ─── [LEGACY 2-backward adaptive] ───────────────────────────────
# Activated only by --use_2backward_adaptive. Mutually exclusive with the
# single-backward _AdaptiveWeightingFn path above: when 2-backward is on,
# GeopriorDiTAlignModel.forward must NOT call _AdaptiveWeightingFn.apply,
# and the train loop must call compute_adaptive_weight_2bwd here instead.
def compute_adaptive_weight_2bwd(rec_loss, align_loss, last_layer, max_weight=1e4, eps=1e-6):
    """Adaptive weight via two autograd.grad calls with retain_graph=True.

    w = ||d rec_loss / d last_layer|| / ||d align_loss / d last_layer||
    This is the original 2-backward path (pre _AdaptiveWeightingFn). Adds ~20-30%
    step time and FSDP2 reshard pressure due to retain_graph, but kept available
    for parity comparisons against the single-backward path.

    Args:
        rec_loss:    scalar loss whose grad is the reference scale
        align_loss:  scalar loss whose grad is scaled to match rec_loss
        last_layer:  parameter tensor (typically encoder.head[-1].weight)
        max_weight:  upper clamp for w (0 or negative disables clamp)
        eps:         numerical floor for align grad norm

    Returns:
        (w_clamped, w_raw): both detached scalar tensors
    """
    rec_grads   = torch.autograd.grad(rec_loss,   last_layer, retain_graph=True)[0]
    align_grads = torch.autograd.grad(align_loss, last_layer, retain_graph=True)[0]
    w = torch.norm(rec_grads) / (torch.norm(align_grads) + eps)
    w_raw = w.detach()
    if max_weight > 0:
        w = w.clamp(0.0, max_weight)
    return w.detach(), w_raw




# ─── Utilities (from train_causalvae.py) ─────────────────────

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def setup_logger(rank):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = ColoredFormatter(
        f"[rank{rank}] %(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        reset=True,
        style="%",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(stream_handler)

    return logger

def check_unused_params(model):
    unused_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused_params.append(name)
    return unused_params

def set_requires_grad_optimizer(optimizer, requires_grad):
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            param.requires_grad = requires_grad

def total_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_millions = total_params / 1e6
    return int(total_params_in_millions)


def get_exp_name(args):
    return f"{args.exp_name}-lr{args.lr:.2e}-bs{args.batch_size}-rs{args.resolution}-sr{args.sample_rate}-fr{args.num_frames}"


def set_train(modules):
    for module in modules:
        module.train()


def set_eval(modules):
    for module in modules:
        module.eval()


def set_modules_requires_grad(modules, requires_grad):
    for module in modules:
        module.requires_grad_(requires_grad)


def save_checkpoint(
    epoch,
    current_step,
    optimizer_state,
    state_dict,
    scaler_state,
    sampler_state,
    checkpoint_dir,
    filename="checkpoint.ckpt",
    ema_state_dict={},
    lora_state_dict={},
):
    filepath = checkpoint_dir / Path(filename)
    torch.save(
        {
            "epoch": epoch,
            "current_step": current_step,
            "optimizer_state": optimizer_state,
            "state_dict": state_dict,
            "ema_state_dict": ema_state_dict,
            "lora_state_dict": lora_state_dict,
            "scaler_state": scaler_state,
            "sampler_state": sampler_state,
        },
        filepath,
    )
    return filepath


def valid(global_rank, rank, model, val_dataloader, precision, args, lpips_model=None):
    # [Modified - oliviaa] lpips_model을 외부에서 받아 재사용.
    # 기존에는 매 valid() 호출마다 AlexNet을 새로 할당했는데,
    # EMA 활성화 시 eval step당 2번 호출 → 두 번째 호출에서 OOM 발생.
    if args.eval_lpips and lpips_model is None:
        lpips_model = lpips.LPIPS(net="alex", spatial=True)
        lpips_model.to(rank)
        lpips_model = DDP(lpips_model, device_ids=[rank])
        lpips_model.requires_grad_(False)
        lpips_model.eval()

    bar = None
    if global_rank == 0:
        bar = tqdm.tqdm(total=len(val_dataloader), desc="Validation...")

    psnr_list = []
    lpips_list = []
    video_log = []
    num_video_log = args.eval_num_video_log

    # [NEW - oliviaa] CKNNA용 latent 수집 (dual_branch 모델에서만)
    # [MODIFIED - oliviaa/dit_align] GeopriorDiTAlignModel wrapper 경유
    raw_model = model.module if hasattr(model, 'module') else model
    if hasattr(raw_model, 'vae'):
        raw_model = raw_model.vae
    is_dual_branch = getattr(raw_model, 'dual_branch', False)
    z_main_vecs = []   # list of (B, C) CPU tensors
    z_prior_vecs = []  # list of (B, C) CPU tensors
    z_cat_vecs = []    # [NEW - oliviaa/dit_align] z_cat (student latent) for alignment CKA
    z_ref_vecs = []    # [NEW - oliviaa/dit_align] z_ref (teacher latent) for alignment CKA

    # [NEW - oliviaa/dit_align] dit_pipe 접근 — valid() 밖에서 주입
    _dit_pipe = getattr(valid, '_dit_pipe', None)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs = batch["video"].to(rank)
            with torch.cuda.amp.autocast(dtype=precision):
                outputs = model(inputs)
                video_recon = outputs[0]

            # Upload videos
            if global_rank == 0:
                for i in range(len(video_recon)):
                    if num_video_log <= 0:
                        break
                    gt_video = tensor_to_video(inputs[i])
                    rec_video = tensor_to_video(video_recon[i])
                    concat_video = np.concatenate([gt_video, rec_video], axis=3)
                    video_log.append(concat_video)
                    num_video_log -= 1
            # [NEW - oliviaa] CKNNA용 latent 수집
            if is_dual_branch:
                with torch.cuda.amp.autocast(dtype=precision):
                    z_main = outputs[1]  # mu: (B, z_dim, T', H', W')
                    z_prior = raw_model._encode_prior(inputs)
                    # [NEW - oliviaa/dit_align] z_cat = concat, z_ref = teacher VAE encode
                    if len(outputs) > 3:
                        z_cat = outputs[3]  # GeopriorDiTAlignModel returns 4-tuple
                    else:
                        z_cat = torch.cat([raw_model.reparameterize(outputs[1], outputs[2]), z_prior], dim=1)
                z_main_vecs.append(z_main.mean(dim=(2, 3, 4)).detach().cpu())
                z_prior_vecs.append(z_prior.mean(dim=(2, 3, 4)).detach().cpu())
                z_cat_vecs.append(z_cat.mean(dim=(2, 3, 4)).detach().cpu())
                # [NEW - oliviaa/dit_align] teacher VAE encode → z_ref
                if _dit_pipe is not None:
                    with torch.cuda.amp.autocast(dtype=precision):
                        z_ref_batch = []
                        for i in range(inputs.shape[0]):
                            z_i = _dit_pipe.vae.encode(
                                [inputs[i].to(dtype=precision)], device=rank, tiled=True
                            )[0]
                            z_ref_batch.append(z_i.mean(dim=(1, 2, 3)))  # global avg pool → (C,)
                        z_ref_vecs.append(torch.stack(z_ref_batch).detach().cpu())

            B, C, T, H, W = inputs.shape
            inputs = rearrange(inputs, "b c t h w -> (b t) c h w").contiguous()
            video_recon = rearrange(
                video_recon, "b c t h w -> (b t) c h w"
            ).contiguous()

            # Calculate per-video PSNR (one value per video, not per batch)
            # to avoid partial-batch bias when DDP gather averages across ranks
            mse = torch.mean(torch.square(inputs - video_recon), dim=(1, 2, 3))  # (B*T,)
            psnr_frames = 20 * torch.log10(1 / torch.sqrt(mse))  # (B*T,)
            psnr_per_video = psnr_frames.view(B, T).mean(dim=1)   # (B,) mean over frames
            psnr_list.extend(psnr_per_video.detach().cpu().tolist())

            # Calculate per-video LPIPS
            if args.eval_lpips:
                lpips_frames = (
                    lpips_model.forward(inputs, video_recon)
                    .mean(dim=(1, 2, 3))  # (B*T,)
                )
                lpips_per_video = lpips_frames.view(B, T).mean(dim=1)  # (B,)
                lpips_list.extend(lpips_per_video.detach().cpu().tolist())

            if global_rank == 0:
                bar.update()
            # Release gpus memory
            torch.cuda.empty_cache()
    return psnr_list, lpips_list, video_log, z_main_vecs, z_prior_vecs, z_cat_vecs, z_ref_vecs


def gather_valid_result(psnr_list, lpips_list, video_log_list, rank, world_size,
                        z_main_vecs=None, z_prior_vecs=None):
    gathered_psnr_list = [None for _ in range(world_size)]
    gathered_lpips_list = [None for _ in range(world_size)]
    gathered_video_logs = [None for _ in range(world_size)]

    dist.all_gather_object(gathered_psnr_list, psnr_list)
    dist.all_gather_object(gathered_lpips_list, lpips_list)
    dist.all_gather_object(gathered_video_logs, video_log_list)

    # [NEW - oliviaa] drift metrics (CKNNA, CKA, cosine sim): 각 rank의 z 벡터를 gather하여 rank 0에서 계산
    drift_metrics = None
    if z_main_vecs is not None and len(z_main_vecs) > 0:
        gathered_z_main = [None for _ in range(world_size)]
        gathered_z_prior = [None for _ in range(world_size)]
        z_main_cat = torch.cat(z_main_vecs, dim=0)   # (N_local, D)
        z_prior_cat = torch.cat(z_prior_vecs, dim=0)
        dist.all_gather_object(gathered_z_main, z_main_cat)
        dist.all_gather_object(gathered_z_prior, z_prior_cat)
        if rank == 0:
            all_z_main = torch.cat(gathered_z_main, dim=0)   # (N_total, D)
            all_z_prior = torch.cat(gathered_z_prior, dim=0)
            drift_metrics = {
                "cknna":      compute_cknna(all_z_main, all_z_prior, topk=10),
                "linear_cka": compute_linear_cka(all_z_main, all_z_prior),
                "cosine_sim": compute_mean_cosine_sim(all_z_main, all_z_prior),
            }

    return (
        np.mean(list(chain(*gathered_psnr_list))),
        np.mean(list(chain(*gathered_lpips_list))) if any(gathered_lpips_list) else 0.0,
        list(chain(*gathered_video_logs)),
        drift_metrics,
    )


# [NEW - oliviaa] alignment metrics — platonic-rep/metrics.py 기반으로 구현
# Source: /data/karlo-research_715/workspace/kinemadae/projects/oliviaa/platonic-rep/metrics.py

def _hsic_unbiased(K, L):
    """Unbiased HSIC estimator (Song et al. 2012, Eq.5)"""
    m = K.shape[0]
    K_t = K.clone().fill_diagonal_(0)
    L_t = L.clone().fill_diagonal_(0)
    hsic = (
        torch.sum(K_t * L_t.T)
        + torch.sum(K_t) * torch.sum(L_t) / ((m - 1) * (m - 2))
        - 2 * torch.sum(torch.mm(K_t, L_t)) / (m - 2)
    )
    return hsic / (m * (m - 3))


def _hsic_biased(K, L):
    """Biased HSIC via centering matrix H"""
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype) - 1.0 / n
    return torch.trace(K @ H @ L @ H)


def compute_cknna(z1, z2, topk=10):
    """
    CKNNA: kNN neighborhood에 HSIC 적용 (platonic-rep 기반)
    z1, z2: (N, D) CPU float tensors
    Returns score in [0, 1].
    """
    n = z1.shape[0]
    topk = min(topk, n - 1)
    if topk < 2:
        return 0.0
    # L2 normalize (platonic-rep test code 참고)
    z1 = F.normalize(z1.float(), dim=-1)
    z2 = F.normalize(z2.float(), dim=-1)
    K = z1 @ z1.T  # cosine similarity matrix
    L = z2 @ z2.T
    # unbiased: 대각 -inf로 마스킹 후 topk
    K_hat = K.clone().fill_diagonal_(float('-inf'))
    L_hat = L.clone().fill_diagonal_(float('-inf'))
    _, idx_K = torch.topk(K_hat, topk, dim=1)
    _, idx_L = torch.topk(L_hat, topk, dim=1)
    mask_K = torch.zeros(n, n).scatter_(1, idx_K, 1.0)
    mask_L = torch.zeros(n, n).scatter_(1, idx_L, 1.0)
    # kNN intersection에 HSIC 적용
    sim_kl = _hsic_unbiased(mask_K * K, mask_L * L)
    sim_kk = _hsic_unbiased(mask_K * K, mask_K * K)
    sim_ll = _hsic_unbiased(mask_L * L, mask_L * L)
    return (sim_kl / (torch.sqrt(sim_kk * sim_ll) + 1e-6)).item()


def compute_linear_cka(z1, z2):
    """
    Linear CKA (platonic-rep 기반, biased HSIC)
    z1, z2: (N, D) CPU float tensors — D 달라도 됨
    Returns score in [0, 1].
    """
    z1 = z1.float()
    z2 = z2.float()
    K = z1 @ z1.T
    L = z2 @ z2.T
    hsic_kl = _hsic_biased(K, L)
    hsic_kk = _hsic_biased(K, K)
    hsic_ll = _hsic_biased(L, L)
    return (hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)).item()


# [NEW - oliviaa] Mean cosine similarity: per-sample 방향 유사도
# z_main(z_dim)과 z_prior(prior_z_dim)의 차원이 같을 때만 유효
def compute_mean_cosine_sim(z1, z2):
    """
    z1, z2: (N, D) CPU float tensors — D가 같아야 함
    Returns mean cosine similarity in [-1, 1], 또는 차원 불일치 시 None.
    """
    if z1.shape[1] != z2.shape[1]:
        return None  # z_dim != prior_z_dim인 경우 스킵
    z1 = F.normalize(z1.float(), dim=-1)
    z2 = F.normalize(z2.float(), dim=-1)
    return (z1 * z2).sum(-1).mean().item()


def train(args):
    # Setup logger
    ddp_setup()
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    logger = setup_logger(rank)

    # Init
    ckpt_dir = Path(args.ckpt_dir) / Path(get_exp_name(args))
    if global_rank == 0:
        try:
            ckpt_dir.mkdir(exist_ok=False, parents=True)
        except:
            logger.warning(f"`{ckpt_dir}` exists!")
            time.sleep(5)
    dist.barrier()

    # [Modified - oliviaa] Load generator model
    # 원본: ModelRegistry.get_model() + from_pretrained/from_config로 OSP VAE 로드
    # 변경: _video_vae()로 Wan VAE 로드. ModelRegistry는 OSP 전용 레지스트리라 불필요.
    # add_stages는 JSON 문자열로 받아 파싱 (예: '[{"mode":"downsample3d","num_res_blocks":2}]')
    import json
    add_encoder_stages = json.loads(args.add_encoder_stages) if args.add_encoder_stages else None
    add_decoder_stages = json.loads(args.add_decoder_stages) if args.add_decoder_stages else None

    model = _video_vae(
        pretrained_path=args.pretrained_model_name_or_path,
        z_dim=args.z_dim,  # [Modified - oliviaa] 하드코딩 16 → argparse에서 받음
        device='cpu',
        add_encoder_stages=add_encoder_stages,
        add_decoder_stages=add_decoder_stages,
    )

    # [NEW - oliviaa] Freeze pretrained weights, keep added stages trainable
    # Wan pretrained에서 시작하므로 기존 weight는 고정하고 새 stage만 학습
    if args.freeze_pretrained:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.encoder.add_downsamples.parameters():
            param.requires_grad = True
        # [NEW - oliviaa] --unfreeze_encoder: encoder 전체 풀기 (pretrained 포함)
        if getattr(args, 'unfreeze_encoder', False):
            model.encoder.requires_grad_(True)
            if global_rank == 0:
                logger.warning("Encoder fully unfrozen (all pretrained encoder weights trainable).")
        # [NEW - oliviaa] --unfreeze_decoder: decoder 전체 풀기 (pretrained 포함)
        if getattr(args, 'unfreeze_decoder', False):
            model.decoder.requires_grad_(True)
            if global_rank == 0:
                logger.warning("Decoder fully unfrozen (all pretrained decoder weights trainable).")
        else:
            for param in model.decoder.add_upsamples.parameters():
                param.requires_grad = True
        # [NEW - oliviaa] z_dim이 pretrained(16)와 다르면 z_dim 관련 layer도 열어야 함
        # encoder head, conv1, conv2, decoder conv1이 z_dim에 의존
        if args.z_dim != 16:
            model.encoder.head[-1].requires_grad_(True)   # CausalConv3d(384→z_dim*2)
            model.conv1.requires_grad_(True)              # CausalConv3d(z_dim*2→z_dim*2)
            model.conv2.requires_grad_(True)              # CausalConv3d(z_dim→z_dim)
            model.decoder.conv1.requires_grad_(True)      # CausalConv3d(z_dim→384)
            if global_rank == 0:
                logger.warning(f"z_dim={args.z_dim} != pretrained(16). z_dim-related layers unfrozen.")
        # [NEW - oliviaa] --freeze_encoder: encoder 전체 freeze (decoder-only 학습용)
        if getattr(args, 'freeze_encoder', False):
            model.encoder.requires_grad_(False)
            model.conv1.requires_grad_(False)
            model.conv2.requires_grad_(False)
        # [NEW - oliviaa] --freeze_decoder: decoder 전체 freeze (Stage 1.5용)
        # student_patchify만 학습할 때 사용. VAE 전체가 frozen이라 rec_loss는 계산만 되고
        # gradient가 흐르지 않음 (불필요 compute지만 무시 가능).
        if getattr(args, 'freeze_decoder', False):
            model.decoder.requires_grad_(False)
            if global_rank == 0:
                logger.info("Decoder fully frozen (--freeze_decoder).")
            if global_rank == 0:
                logger.warning("Encoder fully frozen (decoder-only training mode).")
        if global_rank == 0:
            logger.warning("Pretrained weights frozen. Only added stages are trainable.")

    # [Modified - oliviaa] wandb 로깅 (TensorBoard 대체). working repo 와 동일 패턴.
    # --wandb_run_id 지정 시 resume, 없으면 새 run. WANDB_PROJECT env 로 project 설정.
    if global_rank == 0:
        logger.warning("Connecting to WANDB...")
        wandb_kwargs = dict(
            project=os.environ.get("WANDB_PROJECT", "kinemadae"),
            config=vars(args),
            name=get_exp_name(args),
        )
        if getattr(args, 'wandb_run_id', None):
            wandb_kwargs["id"] = args.wandb_run_id
            wandb_kwargs["resume"] = "must"
        wandb.init(**wandb_kwargs)

    dist.barrier()

    # [Modified - oliviaa] Load discriminator model
    # 원본: resolve_str_to_obj()로 문자열에서 클래스 찾기 — OSP 유틸 의존
    # 변경: LPIPSWithDiscriminator3D 직접 호출. resolve_str_to_obj 불필요.
    disc = LPIPSWithDiscriminator3D(
        disc_start=args.disc_start,
        disc_weight=args.disc_weight,
        kl_weight=args.kl_weight,
        logvar_init=args.logvar_init,
        perceptual_weight=args.perceptual_weight,
        loss_type=args.loss_type,
        wavelet_weight=args.wavelet_weight
    )
    if getattr(args, 'lpips_chunk_size', 0) > 0:
        disc.lpips_chunk_size = args.lpips_chunk_size
        if global_rank == 0:
            logger.info(f"LPIPS chunk size: {args.lpips_chunk_size} (sequential computation for memory saving)")

    # ─── [NEW - oliviaa/dit_align] DiT pipeline + student patchify ───
    # DiT blocks 1 세트만 로드 (teacher/student 공유). patchify 만 별도.
    dit_pipe = None
    dit = None
    student_patchify = None
    if args.align_weight > 0:
        dit_pipe = load_pipeline(
            args.dit_ckpt_dir, f"cuda:{rank}",
            lora_checkpoint=args.lora_checkpoint if getattr(args, 'use_lora', False) else None,
            lora_target_modules=getattr(args, 'lora_target_modules', 'q,k,v,o,k_img,v_img,ffn.0,ffn.2'),
            lora_rank=getattr(args, 'lora_rank', 512),
        )
        dit = dit_pipe.dit  # WanModel
        dit.eval()
        # Freeze everything first; selectively unfreeze LoRA below.
        for p in dit.parameters():
            p.requires_grad = False

        # [NEW] LoRA: random-init inject when no ckpt + unfreeze LoRA params (student weights).
        if getattr(args, 'use_lora', False):
            if not args.lora_checkpoint:
                # load_pipeline only injects when lora_checkpoint is set → manually inject random init.
                from peft import LoraConfig, inject_adapter_in_model
                _target_modules = args.lora_target_modules.split(',')
                _lora_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank,
                                          target_modules=_target_modules)
                dit = inject_adapter_in_model(_lora_config, dit)
                dit_pipe.dit = dit
                dit = dit.to(device=f"cuda:{rank}", dtype=torch.bfloat16)
                dit_pipe.dit = dit
            # Unfreeze only LoRA params (student-side trainable; pretrained DiT body frozen).
            # [FIX] keep LoRA params in bf16 (autocast/grad flow). bf16 학습에선 GradScaler 비활성.
            # 이전엔 fp32 cast 했지만, autocast(bf16) 안에서 fp32 weight 와 bf16 input mismatch 로
            # backward 시 LoRA params 의 .grad 가 None 이 됨 (확인된 버그).
            _n_lora_params = 0
            for n, p in dit.named_parameters():
                if 'lora_' in n:
                    p.requires_grad = True
                    _n_lora_params += p.numel()
            if global_rank == 0:
                logger.info(f"[LoRA] inject={'ckpt' if args.lora_checkpoint else 'random'} "
                            f"rank={args.lora_rank} targets=[{args.lora_target_modules}] "
                            f"trainable params={_n_lora_params:,}")
            # [NEW] resume LoRA weight from ckpt (lora_checkpoint 와 별개로 자동 처리)
            if args.resume_from_checkpoint:
                _ckpt_for_lora = torch.load(args.resume_from_checkpoint, map_location='cpu')
                _lora_sd = _ckpt_for_lora.get('lora_state_dict', {})
                if _lora_sd:
                    _loaded = 0
                    for n, p in dit.named_parameters():
                        if n in _lora_sd:
                            p.data.copy_(_lora_sd[n].to(p.device, p.dtype))
                            _loaded += 1
                    if global_rank == 0:
                        logger.info(f"[LoRA] resumed from ckpt: {_loaded} tensors loaded")
                del _ckpt_for_lora

    # Student patchify (FSDP 전에 생성 — pretrained weight 복사 필요)
    if args.align_weight > 0:
        _mask_ch = 12 if args.mask_mode == 'dual12' else 8
        _patchify_in_ch = (args.z_dim + _known.prior_z_dim) * 2 + _mask_ch  # noisy(48) + mask + image(48)
        student_patchify = create_student_patchify(dit, in_channels=_patchify_in_ch,
                                                    init_mode=args.patchify_init,
                                                    mask_init=args.patchify_mask_init,
                                                    mask_mode=args.mask_mode,
                                                    z_dim=args.z_dim,
                                                    prior_z_dim=_known.prior_z_dim)
        student_patchify = student_patchify.to(rank)

        # z_prior patchify weight 고정 (pretrained copy 유지, z_main만 학습)
        if getattr(args, 'freeze_patchify_zprior', False):
            _z_dim = args.z_dim
            _prior_z_dim = _known.prior_z_dim
            def _zero_zprior_grad(grad):
                # z_prior: noisy(32:48), image_z_prior(마지막 16ch)
                grad[:, _z_dim:_z_dim+_prior_z_dim] = 0
                grad[:, -_prior_z_dim:] = 0
                return grad
            student_patchify.weight.register_hook(_zero_zprior_grad)
            if global_rank == 0:
                logger.info(f"Patchify z_prior channels frozen (weight[:, {_z_dim}:{_z_dim+_prior_z_dim}] + weight[:, -{_prior_z_dim}:])")
    else:
        if global_rank == 0:
            logger.info("align_weight=0: DiT and student_patchify not loaded (pure VAE training)")

    # [NEW] DiT 메모리 최적화
    _dit_offload, _dit_fsdp2, _align_block_set = setup_dit_memory(
        dit, args, rank, global_rank,
        has_fsdp2=HAS_FSDP2, fully_shard=fully_shard, MixedPrecisionPolicy=MixedPrecisionPolicy,
        logger=logger if global_rank == 0 else None,
    )

    # FlowMatchScheduler — noise 추가용
    scheduler = FlowMatchScheduler(template="Wan")
    scheduler.set_timesteps(args.dit_num_inference_steps)

    # Text context: null prompt / per-batch caption / T5 cache
    _use_caption = getattr(args, 'caption_metadata', None) is not None
    _use_t5_cache = getattr(args, 't5_cache_dir', None) is not None
    caption_map = {}
    t5_cache = {}
    if _use_t5_cache:
        import glob as _glob_cache
        cache_files = sorted(_glob_cache.glob(os.path.join(args.t5_cache_dir, "shard_*.pt")))
        for cf in cache_files:
            shard = torch.load(cf, map_location='cpu', weights_only=False)
            t5_cache.update(shard)
            del shard
        if global_rank == 0:
            logger.info(f"Loaded {len(t5_cache)} T5 cached embeddings from {args.t5_cache_dir}")
            logger.info("T5 can be offloaded (using cache)")
    elif _use_caption:
        import json as _json_cap
        with open(args.caption_metadata) as f:
            for line in f:
                entry = _json_cap.loads(line)
                caption_map[entry['video']] = entry['prompt']
        if global_rank == 0:
            logger.info(f"Loaded {len(caption_map)} captions from {args.caption_metadata}")
            logger.info("T5 stays on GPU (per-batch encoding)")

    null_context = None
    if dit_pipe is not None:
        with torch.no_grad():
            null_context = prepare_null_context(dit_pipe)
        # T5 offload — caption mode에서는 T5가 매 배치 필요하므로 offload 불가
        # text_fsdp2와 상호 배타적: FSDP2 sharding이 활성화된 경우 offload 불필요
        _t5_offload_active = (getattr(args, 't5_offload', False) and not _use_caption or _use_t5_cache)
        if _t5_offload_active and getattr(args, 'text_fsdp2', False):
            if global_rank == 0:
                logger.warning("--t5_offload and --text_fsdp2 both set; text_fsdp2 takes precedence, skipping offload")
            _t5_offload_active = False
        if _t5_offload_active:
            if dit_pipe.text_encoder is not None:
                dit_pipe.text_encoder.to('cpu')
            torch.cuda.empty_cache()
            if global_rank == 0:
                logger.info("T5 offloaded to CPU. freed ~10GB VRAM")
        setup_text_encoder_memory(
            dit_pipe, args, rank, global_rank,
            has_fsdp2=HAS_FSDP2, fully_shard=fully_shard, MixedPrecisionPolicy=MixedPrecisionPolicy,
            logger=logger if global_rank == 0 else None,
        )
        if global_rank == 0:
            logger.info(f"DiT loaded. student_patchify in_channels={student_patchify.in_channels}")
            logger.info(f"DiT blocks: {len(dit.blocks)}, dim={dit.dim}")

    # [NEW - oliviaa] Stage 1.5: zmain_stats 로드 (--normalize_zmain 시에만 사용)
    _zmain_stats = None
    if getattr(args, 'normalize_zmain', False):
        if not getattr(args, 'zmain_stats_path', None):
            raise ValueError("--normalize_zmain requires --zmain_stats_path")
        import json as _json_zm
        with open(args.zmain_stats_path) as _f_zm:
            _zmain_stats = _json_zm.load(_f_zm)
        if global_rank == 0:
            logger.info(f"[Stage 1.5] Loaded zmain_stats from {args.zmain_stats_path} "
                        f"(z_dim={len(_zmain_stats.get('mean', []))})")

    # GeopriorDiTAlignModel wrapper (VAE only — student_patchify 는 DDP 밖)
    # [FIX - oliviaa] student_patchify 를 DDP wrapper 안에 넣으면 forward() 밖에서
    # 사용하는 param 이 DDP gradient hook 에서 "marked ready twice" 에러 발생.
    # student_patchify 는 별도로 관리하고 backward 후 수동 all_reduce.
    model = GeopriorDiTAlignModel(
        model,
        normalize_zprior=getattr(args, 'normalize_zprior', False),
        zmain_stats=_zmain_stats,
        decoder_noise_tau_main=getattr(args, 'decoder_noise_tau_main', 0.0),
        decoder_noise_tau_prior=getattr(args, 'decoder_noise_tau_prior', 0.0),
        decoder_noise_random_mode=getattr(args, 'decoder_noise_random_mode', True),
        decoder_noise_warmup_steps=getattr(args, 'decoder_noise_warmup_steps', 0),
        decoder_noise_warmup_power=getattr(args, 'decoder_noise_warmup_power', 1.0),
        align_weight=args.align_weight,
        align_adaptive_weight=args.align_adaptive_weight,
        use_2backward_adaptive=getattr(args, 'use_2backward_adaptive', False),
        adaptive_max_weight=getattr(args, 'adaptive_max_weight', 1e4),
        normalize_zmain_bn=getattr(args, 'normalize_zmain_bn', False),
        bn_momentum=getattr(args, 'bn_momentum', 0.1),
        zmain_bn_init=getattr(args, 'zmain_bn_init', 'zprior'),
        z_dim=args.z_dim,
    )
    # [NEW] SyncBN convert (multi-GPU 면) — REPA-E 따라
    if getattr(args, 'normalize_zmain_bn', False) and dist.get_world_size() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if global_rank == 0:
            logger.info(f"[zmain_bn] SyncBatchNorm applied (world_size={dist.get_world_size()})")

    model = model.to(rank)
    # [NEW - oliviaa] Stage 1.5: VAE 전체가 frozen이면 DDP wrap이 실패함
    # ("DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient")
    # → trainable param 없으면 DDP 건너뛰고 plain model 사용. .module 인터페이스는 wrapper로 유지.
    _has_trainable_vae = any(p.requires_grad for p in model.parameters())
    if _has_trainable_vae:
        model = DDP(
            model, device_ids=[rank], find_unused_parameters=args.find_unused_parameters
        )
    else:
        if global_rank == 0:
            logger.info("VAE has no trainable params → skipping DDP wrap (Stage 1.5 patchify-only mode)")
        # Wrapper that mimics DDP interface (.module attribute + forward delegation)
        class _NoDDPWrapper:
            def __init__(self, m): object.__setattr__(self, 'module', m)
            def __call__(self, *args, **kwargs): return self.module(*args, **kwargs)
            def __getattr__(self, name):
                # fallback: delegate to underlying module
                return getattr(self.module, name)
            def no_sync(self): import contextlib; return contextlib.nullcontext()
            def train(self, mode=True): self.module.train(mode); return self
            def eval(self): self.module.eval(); return self
            def to(self, *a, **k): self.module.to(*a, **k); return self
        model = _NoDDPWrapper(model)
    disc = disc.to(rank)
    disc = DDP(
        disc, device_ids=[rank], find_unused_parameters=args.find_unused_parameters
    )

    # Load dataset
    dataset = TrainVideoDataset(
        args.video_path,
        sequence_length=args.num_frames,
        resolution=args.resolution,
        sample_rate=args.sample_rate,
        dynamic_sample=args.dynamic_sample,
        cache_file="idx.pkl",
        is_main_process=global_rank == 0,
    )
    ddp_sampler = CustomDistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=ddp_sampler,
        pin_memory=True,
        num_workers=args.dataset_num_worker,
    )
    val_dataloader = None
    if args.eval_video_path is not None:
        val_dataset = ValidVideoDataset(
            real_video_dir=args.eval_video_path,
            num_frames=args.eval_num_frames,
            sample_rate=args.eval_sample_rate,
            crop_size=args.eval_resolution,
            resolution=args.eval_resolution,
        )
        indices = range(args.eval_subset_size)
        val_dataset = Subset(val_dataset, indices=indices)
        val_sampler = CustomDistributedSampler(val_dataset)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            sampler=val_sampler,
            pin_memory=True,
        )

    # [NEW - oliviaa] Additional eval dataloaders for higher resolutions
    # --eval_resolutions_hd: comma-separated list e.g. "512x512,480x832"
    hd_val_dataloaders = []  # list of (name_tag, dataloader)
    if getattr(args, 'eval_resolutions_hd', None):
        for res_str in args.eval_resolutions_hd.split(','):
            res_str = res_str.strip()
            if 'x' in res_str:
                h, w = map(int, res_str.split('x'))
                res = (h, w)
                name_tag = f"{h}x{w}"
            else:
                res = int(res_str)
                name_tag = str(res)
            hd_bs = max(1, args.eval_batch_size // 4)  # 고해상도는 batch 줄임
            hd_dataset = ValidVideoDataset(
                real_video_dir=args.eval_video_path,
                num_frames=args.eval_num_frames,
                sample_rate=args.eval_sample_rate,
                crop_size=res,
                resolution=res,
            )
            hd_subset = Subset(hd_dataset, indices=range(args.eval_subset_size))
            hd_sampler = CustomDistributedSampler(hd_subset)
            hd_loader = DataLoader(hd_subset, batch_size=hd_bs, sampler=hd_sampler, pin_memory=True)
            hd_val_dataloaders.append((name_tag, hd_loader))

    # [MODIFIED - oliviaa/dit_align] Optimizer — VAE + patchify 별도 param group.
    # model.module = GeopriorDiTAlignModel(vae, student_patchify)
    vae_module = model.module.vae
    patchify_module = student_patchify  # DDP 밖에서 별도 관리 (None if no align)

    vae_params = [p for p in vae_module.parameters() if p.requires_grad]
    if patchify_module is not None and getattr(args, 'freeze_patchify_full', False):
        for _p in patchify_module.parameters():
            _p.requires_grad = False
        patchify_params = []
        if global_rank == 0:
            logger.info("student_patchify fully frozen (--freeze_patchify_full).")
    else:
        patchify_params = list(patchify_module.parameters()) if patchify_module is not None else []

    # modules_to_train: set_train/set_eval 에서 사용
    if getattr(args, 'unfreeze_encoder', False):
        enc_module = vae_module.encoder
    else:
        enc_module = vae_module.encoder.add_downsamples
    if getattr(args, 'unfreeze_decoder', False):
        modules_to_train = [enc_module, vae_module.decoder]
    else:
        modules_to_train = [enc_module, vae_module.decoder.add_upsamples]
    if patchify_module is not None and not getattr(args, 'freeze_patchify_full', False):
        modules_to_train.append(patchify_module)
    if getattr(args, 'expand_encoder_head', False):
        modules_to_train += [vae_module.encoder.head, vae_module.conv1]

    param_groups = [{'params': vae_params, 'lr': args.lr}]
    if patchify_params:
        param_groups.append({'params': patchify_params, 'lr': args.patchify_lr})
    # [NEW] LoRA params on student DiT (trainable). Same lr as patchify_lr.
    if getattr(args, 'use_lora', False) and dit is not None:
        lora_params = [p for n, p in dit.named_parameters() if 'lora_' in n and p.requires_grad]
        if lora_params:
            param_groups.append({'params': lora_params, 'lr': args.patchify_lr})
            if global_rank == 0:
                logger.info(f"[LoRA] optimizer received {len(lora_params)} param tensors "
                            f"({sum(p.numel() for p in lora_params):,} elements) at lr={args.patchify_lr}")
    gen_optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    disc_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, disc.module.discriminator.parameters()), lr=args.lr, weight_decay=0.01
    )

    # AMP scaler — disabled when LoRA + bf16 (mixed Tensor/DTensor unscale_ fails).
    # bf16 has its own numerical range so GradScaler is unnecessary anyway.
    _scaler_enabled = not (getattr(args, 'use_lora', False) and args.mix_precision == 'bf16')
    scaler = torch.cuda.amp.GradScaler(enabled=_scaler_enabled)
    disc_scaler = torch.cuda.amp.GradScaler(enabled=_scaler_enabled)  # [NEW] disc 용 별도 scaler (accum과 충돌 방지)
    precision = torch.bfloat16
    if args.mix_precision == "fp16":
        precision = torch.float16
    elif args.mix_precision == "fp32":
        precision = torch.float32
    print(precision)
    
    # Load from checkpoint
    start_epoch = 0
    current_step = 0
    if args.resume_from_checkpoint:
        if not os.path.isfile(args.resume_from_checkpoint):
            raise Exception(
                f"Make sure `{args.resume_from_checkpoint}` is a ckpt file."
            )
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model.module.load_state_dict(checkpoint["state_dict"]["gen_model"], strict=False)
        # [FIX - oliviaa] decoder_only ckpt는 student_patchify가 빈 dict {}로 저장돼 있음 (align_weight=0).
        # 빈 dict로 strict load 시 "Missing keys" 에러 → 비어있으면 skip하고 fresh init 사용.
        _stu_pe_state = checkpoint["state_dict"].get("student_patchify")
        if _stu_pe_state and student_patchify is not None:
            student_patchify.load_state_dict(_stu_pe_state)
            logger.info(f"Loaded student_patchify from resume ckpt ({len(_stu_pe_state)} keys)")
        elif student_patchify is not None:
            logger.info("student_patchify state empty in resume ckpt → using fresh init")
        disc.module.load_state_dict(checkpoint["state_dict"]["dics_model"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        # [NEW - oliviaa] trainable params가 바뀐 경우 optimizer state 크기 불일치 → skip
        try:
            gen_optimizer.load_state_dict(checkpoint["optimizer_state"]["gen_optimizer"])
            disc_optimizer.load_state_dict(checkpoint["optimizer_state"]["disc_optimizer"])
        except (ValueError, KeyError) as e:
            logger.warning(f"Optimizer state 로드 스킵 (trainable params 변경으로 인한 불일치): {e}")
        ddp_sampler.load_state_dict(checkpoint["sampler_state"])
        start_epoch = checkpoint["sampler_state"]["epoch"]
        current_step = checkpoint["current_step"]
        logger.info(
            f"Checkpoint loaded from {args.resume_from_checkpoint}, starting from epoch {start_epoch} step {current_step}"
        )

    if args.ema:
        logger.warning(f"Start with EMA. EMA decay = {args.ema_decay}.")
        ema = EMA(model, args.ema_decay)
        ema.register()
        if args.resume_from_checkpoint and checkpoint.get("ema_state_dict"):
            _ema_sd = checkpoint["ema_state_dict"]
            # [NEW] new format: {'shadow': ..., 'shadow_buffers': ...} / legacy: dict 자체가 shadow
            if isinstance(_ema_sd, dict) and 'shadow' in _ema_sd and 'shadow_buffers' in _ema_sd:
                _shadow_sd = _ema_sd['shadow']
                _shadow_buf_sd = _ema_sd['shadow_buffers']
            else:
                _shadow_sd = _ema_sd
                _shadow_buf_sd = {}
            # parameter shadow
            for name, param in model.named_parameters():
                if name in _shadow_sd:
                    ema.shadow[name] = _shadow_sd[name].to(dtype=param.dtype, device=param.device)
            # [NEW] buffer shadow (BN running stats EMA 등) — REPA-E style
            for name, buf in model.named_buffers():
                if name in _shadow_buf_sd:
                    ema.shadow_buffers[name] = _shadow_buf_sd[name].to(dtype=buf.dtype, device=buf.device)
            logger.info(f"EMA state loaded from checkpoint ({len(ema.shadow)} params, {len(ema.shadow_buffers)} buffers)")

    # [NEW - oliviaa] GAN adaptive weight용 last_layer 결정
    # [MODIFIED - oliviaa/dit_align] model.module = GeopriorDiTAlignModel → .vae 경유
    if args.gan_last_layer == "add_upsamples":
        gan_last_layer = model.module.vae.decoder.add_upsamples[-1][-1].resample[-1].weight
        logger.info(f"GAN last_layer: decoder.add_upsamples[-1][-1].resample[-1] (trainable)")
    else:
        gan_last_layer = model.module.vae.decoder.head[-1].weight
        logger.info(f"GAN last_layer: decoder.head[-1] (default)")


    # Training loop
    logger.info("Prepared!")
    dist.barrier()
    if global_rank == 0:
        logger.info(f"=== Model Params ===")
        logger.info(f"Generator:\t\t{total_params(model.module)}M")
        logger.info(f"\t- Encoder:\t{total_params(model.module.vae.encoder):d}M")
        logger.info(f"\t- Decoder:\t{total_params(model.module.vae.decoder):d}M")
        if student_patchify is not None:
            logger.info(f"\t- Patchify:\t{total_params(student_patchify):d}M")
        logger.info(f"Discriminator:\t{total_params(disc.module):d}M")
        logger.info(f"===========")
        logger.info(f"Precision is set to: {args.mix_precision}!")
        logger.info("Start training!")

    # Training Bar
    bar_desc = ""
    bar = None
    if global_rank == 0:
        max_steps = (
            args.epochs * len(dataloader) if args.max_steps is None else args.max_steps
        )
        bar = tqdm.tqdm(total=max_steps, desc=bar_desc.format(current_epoch=0, loss=0))
        bar.update(current_step)
        bar_desc = "E{current_epoch} gen:{gen_loss} disc:{disc_loss} rec:{rec_loss} nll:{nll_loss} kl:{kl_loss} std:{latents_std}"
        logger.warning("Training Details: ")
        logger.warning(f" Max steps: {max_steps}")
        logger.warning(f" Dataset Samples: {len(dataloader)}")
        logger.warning(
            f" Total Batch Size: {args.batch_size} * {os.environ['WORLD_SIZE']}"
        )
    dist.barrier()

    # Training Loop
    num_epochs = args.epochs

    # [Modified - oliviaa] progress bar에 주요 메트릭 표시
    last_metrics = {"gen_loss": "-", "disc_loss": "-", "rec_loss": "-", "nll_loss": "-", "kl_loss": "-", "latents_std": "-"}

    def update_bar(bar):
        if global_rank == 0:
            bar.desc = bar_desc.format(current_epoch=epoch, **last_metrics)
            bar.update()

    # [NEW - oliviaa] LPIPS 모델을 학습 루프 전에 한 번만 생성.
    # 기존에는 valid() 내부에서 매번 생성 → EMA 있을 때 eval step당 2회 할당 → OOM.
    shared_lpips_model = None
    if args.eval_lpips:
        shared_lpips_model = lpips.LPIPS(net="alex", spatial=True)
        shared_lpips_model.to(rank)
        shared_lpips_model = DDP(shared_lpips_model, device_ids=[rank])
        shared_lpips_model.requires_grad_(False)
        shared_lpips_model.eval()

    # [NEW] grad accumulation 용 변수 초기화
    _accum = getattr(args, 'grad_accum_steps', 1)
    _loss_accum = {"g_loss": 0.0, "rec_loss": 0.0, "kl_loss": 0.0, "nll_loss": 0.0,
                   "align_loss": 0.0, "total_loss": 0.0}
    optimizer_step = 0

    if global_rank == 0:
        torch.cuda.empty_cache()
        _mem_train_start = torch.cuda.memory_allocated(rank) / 1e9
        _mem_train_reserved = torch.cuda.memory_reserved(rank) / 1e9
        logger.info(f"GPU {rank} memory at training start: allocated={_mem_train_start:.2f}GB, reserved={_mem_train_reserved:.2f}GB")

    for epoch in range(num_epochs):
        gen_optimizer.zero_grad()
        set_train(modules_to_train)
        ddp_sampler.set_epoch(epoch)  # Shuffle data at every epoch
        for batch_idx, batch in enumerate(dataloader):
            # [FIX - oliviaa] max_steps 도달 시 조기 종료
            if args.max_steps is not None and current_step >= args.max_steps:
                break
            inputs = batch["video"].to(rank)

            # [DEBUG] per-step memory logging
            if global_rank == 0:
                torch.cuda.reset_peak_memory_stats(rank)
                _mem_step_start = torch.cuda.memory_allocated(rank) / 1e9
                logger.info(f"[mem] step {current_step} start: allocated={_mem_step_start:.2f}GB")

            if (
                current_step % 2 == 1
                and current_step >= disc.module.discriminator_iter_start
            ):
                set_modules_requires_grad(modules_to_train, False)
                step_gen = False
                step_dis = True
            else:
                set_modules_requires_grad(modules_to_train, True)
                step_gen = True
                step_dis = False

            assert (
                step_gen or step_dis
            ), "You should backward either Gen or Dis in a step."

            with torch.cuda.amp.autocast(dtype=precision):
                # [MODIFIED - oliviaa/dit_align] GeopriorDiTAlignModel.forward() → 4-tuple
                recon, mu, log_var, z_cat = model(inputs)
                posterior = DiagonalGaussianDistribution(torch.cat([mu, log_var], dim=1))
                wavelet_coeffs = None

            # Generator Step
            if step_gen:
                with torch.cuda.amp.autocast(dtype=precision):
                    g_loss, g_log = disc(
                        inputs,
                        recon,
                        posterior,
                        optimizer_idx=0,
                        global_step=optimizer_step,
                        last_layer=gan_last_layer,
                        wavelet_coeffs=wavelet_coeffs,
                        split="train",
                    )

                # ─── [NEW - oliviaa/dit_align] DiT dual-branch alignment ───
                align_loss = torch.tensor(0.0, device=rank)
                align_per_layer = {}
                if args.align_weight > 0:
                    _align_bs = args.align_batch_size if args.align_batch_size > 0 else inputs.shape[0]
                    _align_bs = min(_align_bs, inputs.shape[0])
                    inputs_align = inputs[:_align_bs]
                    z_cat_align = z_cat[:_align_bs]
                    with torch.cuda.amp.autocast(dtype=precision):
                        # Timestep 샘플링
                        if args.dit_timestep_mode == 'random':
                            tidx = random.randint(0, len(scheduler.timesteps) - 1)
                        else:
                            # fixed: 미리 정한 timestep 순환
                            _fixed_ts = [int(x) for x in args.dit_fixed_timesteps.split(",")]
                            tidx = _fixed_ts[current_step % len(_fixed_ts)]
                        timestep = scheduler.timesteps[tidx]
                        t_tensor = torch.tensor([timestep], device=rank, dtype=precision)

                        # Decide the align_weight passed into the fused forward.
                        # - 2-backward mode: align_loss must be RAW (no weighting); the train
                        #   loop multiplies by w * align_weight after compute_adaptive_weight_2bwd.
                        # - single-backward adaptive: pass 1.0; _AdaptiveWeightingFn handles scaling.
                        # - no adaptive: pass args.align_weight directly.
                        if getattr(args, 'use_2backward_adaptive', False):
                            _aw_for_fused = 1.0
                        elif args.align_adaptive_weight:
                            _aw_for_fused = 1.0
                        else:
                            _aw_for_fused = args.align_weight

                        if getattr(args, 'no_fused_align', False):
                            # [NEW] Origin (kk4aiuyq) 식: run_teacher → run_student → compute_alignment_loss.
                            # fused 의 per-block detach + AlignGradInjector 대신 sum(grad-tracked losses).backward().
                            features_ref, grid_ref, context, t_mod, freqs_ref = run_teacher_forward(
                                dit=dit, dit_pipe=dit_pipe,
                                inputs_align=inputs_align,
                                scheduler=scheduler, timestep=timestep, t_tensor=t_tensor,
                                null_context=null_context,
                                _align_block_set=_align_block_set,
                                _dit_offload=_dit_offload, _dit_fsdp2=_dit_fsdp2,
                                align_after_patchify=args.align_after_patchify,
                                rank=rank, precision=precision,
                                _use_t5_cache=_use_t5_cache, t5_cache=t5_cache,
                                _use_caption=_use_caption, caption_map=caption_map,
                                batch=batch, _align_bs=_align_bs,
                                logger=logger if global_rank == 0 else None,
                            )
                            features_stu, grid_stu, noisy_cat, _patchify_output_ref = run_student_forward(
                                student_patchify=student_patchify, dit=dit,
                                inputs_align=inputs_align, z_cat_align=z_cat_align,
                                scheduler=scheduler, timestep=timestep,
                                context=context, t_mod=t_mod,
                                model_module=model.module, mask_mode=args.mask_mode,
                                _align_block_set=_align_block_set,
                                _dit_offload=_dit_offload,
                                _use_gc=getattr(args, 'use_grad_checkpoint', False),
                                grad_checkpoint_num_blocks=getattr(args, 'grad_checkpoint_num_blocks', 0),
                                align_after_patchify=args.align_after_patchify,
                                rank=rank, precision=precision,
                                retain_grads=(not args.no_log_grad and global_rank == 0 and current_step % args.log_steps == 0),
                                logger=logger if global_rank == 0 else None,
                            )
                            align_loss, align_per_layer = compute_alignment_loss(
                                features_stu, features_ref,
                                grid_stu=grid_stu, grid_ref=grid_ref,
                                loss_type=args.align_loss_type,
                                selected_layers=[int(x) for x in args.align_layers.split(",")] if args.align_layers != "all" else None,
                                agg=args.align_agg,
                            )
                            # align_weight 적용 (compute_alignment_loss 는 weight 안 받음 — train loop 에서 scale)
                            align_loss = _aw_for_fused * align_loss
                        else:
                            align_loss, align_per_layer, noisy_cat, _patchify_output_ref = fused_dit_align_forward(
                                dit=dit, student_patchify=student_patchify, dit_pipe=dit_pipe,
                                inputs_align=inputs_align, z_cat_align=z_cat_align,
                                scheduler=scheduler, timestep=timestep, t_tensor=t_tensor,
                                null_context=null_context, model_module=model.module, mask_mode=args.mask_mode,
                                _align_block_set=_align_block_set,
                                _dit_offload=_dit_offload, _dit_fsdp2=_dit_fsdp2,
                                _use_gc=getattr(args, 'use_grad_checkpoint', False),
                                grad_checkpoint_num_blocks=getattr(args, 'grad_checkpoint_num_blocks', 0),
                                align_after_patchify=args.align_after_patchify,
                                rank=rank, precision=precision,
                                align_weight=_aw_for_fused,
                                loss_type=args.align_loss_type,
                                selected_layers=[int(x) for x in args.align_layers.split(",")] if args.align_layers != "all" else None,
                                agg=args.align_agg,
                                _use_t5_cache=_use_t5_cache, t5_cache=t5_cache,
                                _use_caption=_use_caption, caption_map=caption_map,
                                batch=batch, _align_bs=_align_bs,
                                retain_grads=(not args.no_log_grad and global_rank == 0 and current_step % args.log_steps == 0),
                                logger=logger if global_rank == 0 else None,
                            )

                # [NEW] 2-backward adaptive (legacy path) — mutually exclusive with _AdaptiveWeightingFn.
                # Computes w via two autograd.grad(retain_graph=True) calls on encoder.head[-1].weight,
                # then assembles total_loss = g_loss + align_weight * w * align_loss.
                # Single-backward path leaves total_loss = g_loss + align_loss (scaling already in graph).
                _w_adaptive_2bwd = None
                _w_raw_2bwd = None
                if (getattr(args, 'use_2backward_adaptive', False)
                        and args.align_weight > 0
                        and align_loss.item() > 0):
                    _align_last_layer = model.module.vae.encoder.head[-1].weight
                    _w_adaptive_2bwd, _w_raw_2bwd = compute_adaptive_weight_2bwd(
                        g_loss, align_loss, _align_last_layer,
                        max_weight=getattr(args, 'adaptive_max_weight', 1e4),
                    )
                    total_loss = g_loss + args.align_weight * _w_adaptive_2bwd * align_loss
                else:
                    total_loss = g_loss + align_loss

                # [NEW - oliviaa/dit_align] gradient accumulation 지원
                scaled_loss = scaler.scale(total_loss / _accum)
                # [NEW] DDP no_sync: 중간 accum step 에서는 gradient sync 생략 → 통신 overhead 절감
                # current_step 은 0 부터 시작, backward 후 증가 → (current_step + 1) 패턴으로 off-by-one 방지
                _is_accum_step = (current_step + 1) % _accum == 0
                # [DEBUG] memory before backward
                if global_rank == 0:
                    _mem_pre_bwd = torch.cuda.memory_allocated(rank) / 1e9
                    logger.info(f"[mem] step {current_step} pre-backward: allocated={_mem_pre_bwd:.2f}GB")

                if _accum > 1 and not _is_accum_step:
                    with model.no_sync():
                        scaled_loss.backward()
                else:
                    scaled_loss.backward()

                # [DEBUG] memory after backward
                if global_rank == 0:
                    _mem_post_bwd = torch.cuda.memory_allocated(rank) / 1e9
                    _mem_peak = torch.cuda.max_memory_allocated(rank) / 1e9
                    logger.info(f"[mem] step {current_step} post-backward: allocated={_mem_post_bwd:.2f}GB, peak={_mem_peak:.2f}GB")

                # accumulation 완료 시에만 optimizer step
                if _is_accum_step:
                    # [FIX - oliviaa] student_patchify 는 DDP 밖이라 gradient 수동 sync
                    if dist.get_world_size() > 1 and student_patchify is not None:
                        for p in student_patchify.parameters():
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

                    # [NEW - oliviaa] gradient clipping + norm 로깅
                    scaler.unscale_(gen_optimizer)
                    _max_grad_norm = getattr(args, 'max_grad_norm', 0.0)
                    if _max_grad_norm > 0:
                        _all_params = list(model.parameters())
                        if student_patchify is not None:
                            _all_params += list(student_patchify.parameters())
                        torch.nn.utils.clip_grad_norm_(_all_params, _max_grad_norm)
                    if not args.no_log_grad and global_rank == 0 and current_step % args.log_steps == 0:
                        def _grad_norm(params):
                            grads = [p.grad for p in params if p.grad is not None]
                            if not grads:
                                return 0.0
                            return torch.cat([g.flatten() for g in grads]).norm().item()

                        _grad_log = {
                            "grad/encoder": _grad_norm(vae_module.encoder.parameters()),
                            "grad/decoder": _grad_norm(vae_module.decoder.parameters()),
                            "grad/conv1": _grad_norm(vae_module.conv1.parameters()),
                        }
                        if getattr(args, 'use_lora', False) and dit is not None:
                            _grad_log["grad/lora"] = _grad_norm(
                                p for n, p in dit.named_parameters() if 'lora_' in n
                            )
                        # [FIX] z_dim/prior_z_dim 동적 — 이전 32/48 hardcoded 였음.
                        _zd = args.z_dim
                        _pzd = _known.prior_z_dim
                        _np = _zd + _pzd  # noisy_z_prior end (= z_dim + prior_z_dim)
                        if student_patchify is not None and student_patchify.weight.grad is not None:
                            _w = student_patchify.weight.grad
                            # pretrained 부분 = noisy_z_prior + image_z_prior (둘 다 prior 채널)
                            _pre = torch.cat([_w[:, _zd:_np], _w[:, -_pzd:]], dim=1)
                            # added 부분 = noisy_z_main + mask + image_z_main
                            _add = torch.cat([_w[:, :_zd], _w[:, _np:-_pzd]], dim=1)
                            _grad_log["grad/patchify_pretrained"] = _pre.norm().item()
                            _grad_log["grad/patchify_added"] = _add.norm().item()
                            _grad_log["grad/patchify_ratio"] = (_add.norm() / (_pre.norm() + 1e-10)).item()
                        if 'noisy_cat' in dir() and hasattr(noisy_cat, 'grad') and noisy_cat.grad is not None:
                            _grad_log["grad/input_z_main"] = noisy_cat.grad[:, :_zd].norm().item()
                            _grad_log["grad/input_z_prior"] = noisy_cat.grad[:, _zd:_np].norm().item()
                            _grad_log["grad/input_ratio"] = (noisy_cat.grad[:, :_zd].norm() / (noisy_cat.grad[:, _zd:_np].norm() + 1e-10)).item()
                        if 'z_cat_align' in dir() and hasattr(z_cat_align, 'grad') and z_cat_align.grad is not None:
                            _grad_log["grad/zcat_z_main"] = z_cat_align.grad[:, :_zd].norm().item()
                            _grad_log["grad/zcat_z_prior"] = z_cat_align.grad[:, _zd:_np].norm().item()
                        elif hasattr(z_cat, 'grad') and z_cat.grad is not None:
                            _grad_log["grad/zcat_z_main"] = z_cat.grad[:, :_zd].norm().item()
                            _grad_log["grad/zcat_z_prior"] = z_cat.grad[:, _zd:_np].norm().item()
                        if '_patchify_output_ref' in dir() and hasattr(_patchify_output_ref, 'grad') and _patchify_output_ref.grad is not None:
                            _grad_log["grad/patchify_output"] = _patchify_output_ref.grad.norm().item()
                        wandb.log(_grad_log, step=optimizer_step)

                    scaler.step(gen_optimizer)
                    scaler.update()
                    gen_optimizer.zero_grad()
                    if args.ema:
                        ema.update()
                # [NEW] grad accumulation: loss 누적 (accum>1 일 때)
                if _accum > 1:
                    _loss_accum["g_loss"] += g_loss.item() / _accum
                    _loss_accum["rec_loss"] += g_log['train/rec_loss'] / _accum
                    _loss_accum["kl_loss"] += g_log['train/kl_loss'] / _accum
                    _loss_accum["nll_loss"] += g_log['train/nll_loss'] / _accum
                    _loss_accum["align_loss"] += align_loss.item() / _accum
                    _loss_accum["total_loss"] += total_loss.item() / _accum

                # accum=1 이면 매 step 로깅, accum>1 이면 optimizer step 시에만 로깅
                _should_log = global_rank == 0 and current_step % args.log_steps == 0
                if _accum > 1:
                    _should_log = _should_log and _is_accum_step

                if _should_log:
                    if _accum > 1:
                        _gl = _loss_accum["g_loss"]
                        _rl = _loss_accum["rec_loss"]
                        _kl = _loss_accum["kl_loss"]
                        _nl = _loss_accum["nll_loss"]
                        _al = _loss_accum["align_loss"]
                        _tl = _loss_accum["total_loss"]
                    else:
                        _gl = g_loss.item()
                        _rl = g_log['train/rec_loss']
                        _kl = g_log['train/kl_loss']
                        _nl = g_log['train/nll_loss']
                        _al = align_loss.item()
                        _tl = total_loss.item()

                    latents_std = posterior.sample().std().item()
                    last_metrics["gen_loss"] = f"{_gl:.4f}"
                    last_metrics["rec_loss"] = f"{_rl:.4f}"
                    last_metrics["nll_loss"] = f"{_nl:.4f}"
                    last_metrics["kl_loss"] = f"{_kl:.6f}"
                    last_metrics["latents_std"] = f"{latents_std:.4f}"
                    wandb.log({"train/generator_loss": _gl}, step=optimizer_step)
                    wandb.log({"train/rec_loss": _rl}, step=optimizer_step)
                    wandb.log({"train/kl_loss": _kl}, step=optimizer_step)
                    wandb.log({"train/nll_loss": _nl}, step=optimizer_step)
                    wandb.log({"train/latents_std": latents_std}, step=optimizer_step)
                    wandb.log({"train/g_loss": g_log.get('train/g_loss', 0)}, step=optimizer_step)
                    wandb.log({"train/d_weight": g_log.get('train/d_weight', 0)}, step=optimizer_step)
                    if 'train/sb_loss' in g_log:
                        wandb.log({"train/sb_loss": g_log['train/sb_loss']}, step=optimizer_step)
                    if 'train/wl_loss' in g_log:
                        wandb.log({"train/wl_loss": g_log['train/wl_loss']}, step=optimizer_step)
                    wandb.log({"train/align_loss": _al}, step=optimizer_step)
                    wandb.log({"train/total_loss": _tl}, step=optimizer_step)
                    # [REVIVED] align_loss_weighted — actual magnitude flowing through backward.
                    # adaptive paths: w sourced from active mode (2-backward vs single-backward).
                    if getattr(args, 'log_adaptive_weight', False):
                        if getattr(args, 'use_2backward_adaptive', False) and _w_adaptive_2bwd is not None:
                            _w_log = float(_w_adaptive_2bwd.item() if hasattr(_w_adaptive_2bwd, 'item') else _w_adaptive_2bwd)
                            if _w_raw_2bwd is not None:
                                wandb.log({"train/adaptive_weight_w_raw": float(_w_raw_2bwd.item())}, step=optimizer_step)
                        elif args.align_adaptive_weight and getattr(_AdaptiveWeightingFn, '_last_c', None) is not None:
                            _w_log = float(_AdaptiveWeightingFn._last_c.item())
                        else:
                            _w_log = 1.0
                        wandb.log({"train/adaptive_weight_w": _w_log}, step=optimizer_step)
                        wandb.log({"train/align_loss_weighted": args.align_weight * _w_log * _al}, step=optimizer_step)
                    for layer_name, layer_loss in align_per_layer.items():
                        wandb.log({f"train/align_layer/{layer_name}": layer_loss.item()}, step=optimizer_step)

                    # [NEW] z_main BN running stats 로깅 (normalize_zmain_bn 활성 시)
                    if getattr(args, 'normalize_zmain_bn', False):
                        # GeopriorDiTAlignModel 는 DDP 안: model.module.zmain_bn (no DDP 면 model.zmain_bn)
                        _wrapper = model.module if hasattr(model, 'module') else model
                        _bn = _wrapper.zmain_bn if hasattr(_wrapper, 'zmain_bn') else None
                        if _bn is not None:
                            _rm = _bn.running_mean.detach()
                            _rv = _bn.running_var.detach()
                            # summary: per-channel mean/var 의 통계
                            wandb.log({
                                "bn_zmain/rm_mean": _rm.mean().item(),
                                "bn_zmain/rm_std": _rm.std().item(),
                                "bn_zmain/rm_absmax": _rm.abs().max().item(),
                                "bn_zmain/rv_mean": _rv.mean().item(),
                                "bn_zmain/rv_std": _rv.std().item(),
                                "bn_zmain/rv_min": _rv.min().item(),
                                "bn_zmain/rv_max": _rv.max().item(),
                            }, step=optimizer_step)
                            # per-channel detail
                            for c in range(_rm.shape[0]):
                                wandb.log({
                                    f"bn_zmain/rm_c{c:02d}": _rm[c].item(),
                                    f"bn_zmain/rv_c{c:02d}": _rv[c].item(),
                                }, step=optimizer_step)

                    # 누적 초기화
                    if _accum > 1:
                        for k in _loss_accum:
                            _loss_accum[k] = 0.0

            # Discriminator Step
            if step_dis:
                with torch.cuda.amp.autocast(dtype=precision):
                    d_loss, d_log = disc(
                        inputs,
                        recon,
                        posterior,
                        optimizer_idx=1,
                        global_step=optimizer_step,
                        last_layer=None,
                        split="train",
                    )
                # [NEW] disc 에도 accumulation 적용 (gen 과 동일 빈도로 update)
                disc_scaler.scale(d_loss / _accum).backward()
                if _is_accum_step:
                    disc_scaler.unscale_(disc_optimizer)
                    torch.nn.utils.clip_grad_norm_(disc.module.discriminator.parameters(), 1.0)
                    disc_scaler.step(disc_optimizer)
                    disc_scaler.update()
                    disc_optimizer.zero_grad()
                    if global_rank == 0 and current_step % args.log_steps == 0:
                        last_metrics["disc_loss"] = f"{d_loss.item():.4f}"
                        wandb.log({"train/discriminator_loss": d_loss.item()}, step=optimizer_step)

            update_bar(bar)
            current_step += 1
            # optimizer_step: accum=1 이면 current_step 과 동일, accum>1 이면 current_step // accum
            optimizer_step = current_step // _accum

            def valid_model(model, name="", dataloader=None):
                set_eval(modules_to_train)
                _loader = dataloader if dataloader is not None else val_dataloader
                # [NEW - oliviaa/dit_align] dit_pipe 를 valid() 에 주입
                valid._dit_pipe = dit_pipe
                psnr_list, lpips_list, video_log, z_main_vecs, z_prior_vecs, z_cat_vecs, z_ref_vecs = valid(
                    global_rank, rank, model, _loader, precision, args,
                    lpips_model=shared_lpips_model,
                )
                valid_psnr, valid_lpips, valid_video_log, drift_metrics = gather_valid_result(
                    psnr_list, lpips_list, video_log, rank, dist.get_world_size(),
                    z_main_vecs=z_main_vecs, z_prior_vecs=z_prior_vecs,
                )
                # [NEW - oliviaa/dit_align] z_cat vs z_ref alignment CKA
                align_metrics = None
                if z_cat_vecs and z_ref_vecs:
                    gathered_z_cat = [None for _ in range(dist.get_world_size())]
                    gathered_z_ref = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(gathered_z_cat, torch.cat(z_cat_vecs, dim=0))
                    dist.all_gather_object(gathered_z_ref, torch.cat(z_ref_vecs, dim=0))
                    if rank == 0:
                        all_z_cat = torch.cat(gathered_z_cat, dim=0)
                        all_z_ref = torch.cat(gathered_z_ref, dim=0)
                        align_metrics = {
                            "cknna": compute_cknna(all_z_cat, all_z_ref, topk=10),
                            "linear_cka": compute_linear_cka(all_z_cat, all_z_ref),
                        }

                if global_rank == 0:
                    name = "_" + name if name != "" else name
                    # video: wandb.Video accepts (N, T, C, H, W) or (T, C, H, W).
                    # tensor_to_video already returns (T, C, H, W); stacked → (N, T, C, H, W). No transpose needed.
                    _vid_arr = np.array(valid_video_log)
                    wandb.log({f"val{name}/recon": wandb.Video(_vid_arr, fps=10)}, step=optimizer_step)
                    wandb.log({f"val{name}/psnr": valid_psnr}, step=optimizer_step)
                    wandb.log({f"val{name}/lpips": valid_lpips}, step=optimizer_step)
                    # z_main↔z_prior drift
                    if drift_metrics is not None:
                        wandb.log({f"val{name}/cknna_z_drift":      drift_metrics["cknna"]},      step=optimizer_step)
                        wandb.log({f"val{name}/linear_cka_z_drift": drift_metrics["linear_cka"]}, step=optimizer_step)
                        if drift_metrics["cosine_sim"] is not None:
                            wandb.log({f"val{name}/cosine_sim_z_drift": drift_metrics["cosine_sim"]}, step=optimizer_step)
                        cos_str = f"{drift_metrics['cosine_sim']:.4f}" if drift_metrics['cosine_sim'] is not None else "N/A (dim mismatch)"
                        logger.info(
                            f"val{name} drift — CKNNA: {drift_metrics['cknna']:.4f} | "
                            f"CKA: {drift_metrics['linear_cka']:.4f} | "
                            f"cos: {cos_str}"
                        )
                    # [NEW - oliviaa/dit_align] z_cat↔z_ref alignment metrics
                    if align_metrics is not None:
                        wandb.log({f"val{name}/cknna_z_align":      align_metrics["cknna"]},      step=optimizer_step)
                        wandb.log({f"val{name}/linear_cka_z_align": align_metrics["linear_cka"]}, step=optimizer_step)
                        logger.info(
                            f"val{name} align — CKNNA: {align_metrics['cknna']:.4f} | "
                            f"CKA: {align_metrics['linear_cka']:.4f}"
                        )
                    logger.info(f"{name} Validation done.")

            if args.eval_video_path is not None and (optimizer_step % args.eval_steps == 0 or optimizer_step == 1):
                if global_rank == 0:
                    logger.info("Starting validation...")
                valid_model(model)
                if args.ema:
                    ema.apply_shadow()
                    valid_model(model, "ema")
                    ema.restore()
                # [NEW - oliviaa] HD resolution evals
                for _res_tag, _hd_loader in hd_val_dataloaders:
                    valid_model(model, _res_tag, dataloader=_hd_loader)
                    if args.ema:
                        ema.apply_shadow()
                        valid_model(model, f"ema_{_res_tag}", dataloader=_hd_loader)
                        ema.restore()

            # Checkpoint
            # [FIX - oliviaa] _is_accum_step 게이팅 추가
            # optimizer_step 은 _accum 개의 연속된 current_step 동안 동일 값 유지 → 게이팅 없으면
            # save_ckpt_step 경계에서 _accum 번 연속 저장됨 (예: accum=4 → checkpoint-N0,N1,N2,N3 4개 동일 파일).
            if _is_accum_step and optimizer_step % args.save_ckpt_step == 0 and global_rank == 0:
                file_path = save_checkpoint(
                    epoch,
                    current_step,
                    {
                        "gen_optimizer": gen_optimizer.state_dict(),
                        "disc_optimizer": disc_optimizer.state_dict(),
                    },
                    {
                        "gen_model": model.module.state_dict(),
                        "student_patchify": student_patchify.state_dict() if student_patchify is not None else {},
                        "dics_model": disc.module.state_dict(),
                    },
                    scaler.state_dict(),
                    ddp_sampler.state_dict(),
                    ckpt_dir,
                    f"checkpoint-{current_step}.ckpt",
                    ema_state_dict=ema.state_dict() if args.ema else {},
                    # [NEW] LoRA weight save (dit_pipe.dit 의 lora_ param 만)
                    lora_state_dict=(
                        {n: p.data.cpu().clone()
                         for n, p in dit.named_parameters() if 'lora_' in n}
                        if getattr(args, 'use_lora', False) else {}
                    ),
                )
                logger.info(f"Checkpoint has been saved to `{file_path}`.")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Distributed Training")
    # Exp setting
    parser.add_argument(
        "--exp_name", type=str, default="test", help="number of epochs to train"
    )
    parser.add_argument("--seed", type=int, default=1234, help="seed")
    # Training setting
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="number of epochs to train"
    )
    parser.add_argument("--save_ckpt_step", type=int, default=1000, help="")
    parser.add_argument("--ckpt_dir", type=str, default="./results/", help="")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--log_steps", type=int, default=5, help="log steps")
    parser.add_argument("--no_log_grad", action="store_true",
                        help="disable gradient norm logging and retain_grad (saves GPU memory)")
    # [Modified - oliviaa] 원본: OSP에서 encoder만 freeze하는 옵션. 현재는 --freeze_pretrained으로 대체.
    # 필요하면 나중에 재활용 가능하므로 남겨둠.
    parser.add_argument("--freeze_encoder", action="store_true", help="")
    parser.add_argument("--freeze_decoder", action="store_true",
                        help="[Stage 1.5] decoder 전체 freeze. student_patchify만 학습할 때 사용.")
    parser.add_argument("--clip_grad_norm", type=float, default=1e5, help="")

    # Data
    parser.add_argument("--video_path", type=str, default=None, help="")
    parser.add_argument("--num_frames", type=int, default=17, help="")
    parser.add_argument("--resolution", type=int, default=256, help="")
    parser.add_argument("--sample_rate", type=int, default=2, help="")
    parser.add_argument("--dynamic_sample", action="store_true", help="")
    # Generator model
    # [Removed - oliviaa] --ignore_mismatched_sizes: diffusers from_pretrained 전용 옵션, _video_vae에서 불필요
    # [Removed - oliviaa] --model_name: OSP ModelRegistry 전용, Wan VAE는 _video_vae()로 직접 로드
    # [Removed - oliviaa] --not_resume_training_process: 사용처 없음
    # [Removed - oliviaa] --model_config: OSP from_config 전용
    parser.add_argument("--find_unused_parameters", action="store_true", help="")
    # [NEW - oliviaa] latent 채널 수. pretrained=16. 변경 시 z_dim 관련 layer가 재초기화됨
    parser.add_argument("--z_dim", type=int, default=16, help="latent channel dim. pretrained=16")
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default=None, help="path to Wan2.1_VAE.pth"
    )
    # [NEW - oliviaa] Added stages config — JSON string
    # 예: '[{"mode":"downsample3d","num_res_blocks":2}]'
    parser.add_argument("--add_encoder_stages", type=str, default=None,
                        help='JSON list of encoder stages, e.g. \'[{"mode":"downsample3d","num_res_blocks":2}]\'')
    parser.add_argument("--add_decoder_stages", type=str, default=None,
                        help='JSON list of decoder stages, e.g. \'[{"mode":"upsample3d","num_res_blocks":2}]\'')
    # [NEW - oliviaa] Freeze pretrained Wan VAE weights, train only added stages
    parser.add_argument("--freeze_pretrained", action="store_true", help="")
    # [NEW - oliviaa] freeze_pretrained 시 encoder 전체를 풀기 (add_downsamples만 푸는 기본 동작 대신)
    parser.add_argument("--unfreeze_encoder", action="store_true", help="")
    # [NEW - oliviaa] freeze_pretrained 시 decoder 전체를 풀기 (add_upsamples만 푸는 기본 동작 대신)
    parser.add_argument("--unfreeze_decoder", action="store_true", help="")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="")
    parser.add_argument(
        "--mix_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="precision for training",
    )
    parser.add_argument("--wavelet_loss", action="store_true", help="")
    parser.add_argument("--wavelet_weight", type=float, default=0.1, help="")
    # Discriminator Model
    # [Removed - oliviaa] --load_disc_from_checkpoint: 현재 미사용
    # [Removed - oliviaa] --disc_cls: resolve_str_to_obj 전용, LPIPSWithDiscriminator3D 직접 호출로 변경
    parser.add_argument("--disc_start", type=int, default=5, help="")
    parser.add_argument("--disc_weight", type=float, default=0.5, help="")
    parser.add_argument("--kl_weight", type=float, default=1e-06, help="")
    parser.add_argument("--perceptual_weight", type=float, default=1.0, help="")
    parser.add_argument("--loss_type", type=str, default="l1", help="")
    parser.add_argument("--logvar_init", type=float, default=0.0, help="")
    # [NEW - oliviaa] GAN adaptive weight 계산에 사용할 last_layer 선택
    # decoder_head: 원본 방식 (decoder.head[-1].weight). freeze_pretrained 시 사용 불가
    # add_upsamples: 추가된 decoder stage의 마지막 conv weight. freeze_pretrained 시 사용
    parser.add_argument("--gan_last_layer", type=str, default="decoder_head",
                        choices=["decoder_head", "add_upsamples"],
                        help="layer for GAN adaptive weight calculation")

    # Validation
    parser.add_argument("--eval_steps", type=int, default=1000, help="")
    parser.add_argument("--eval_video_path", type=str, default=None, help="")
    parser.add_argument("--eval_num_frames", type=int, default=17, help="")
    parser.add_argument("--eval_resolution", type=int, default=256, help="")
    parser.add_argument("--eval_sample_rate", type=int, default=1, help="")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="")
    parser.add_argument("--eval_subset_size", type=int, default=100, help="")
    # [NEW - oliviaa] Additional HD eval resolutions, comma-separated e.g. "512x512,480x832"
    parser.add_argument("--eval_resolutions_hd", type=str, default=None, help="")
    parser.add_argument("--eval_num_video_log", type=int, default=2, help="")
    parser.add_argument("--eval_lpips", action="store_true", help="")

    # Dataset
    parser.add_argument("--dataset_num_worker", type=int, default=4, help="")

    # Wandb (replaces TensorBoard)
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="wandb run id to resume (e.g. '3rtlyr9o'). Empty = new run.")

    # EMA
    parser.add_argument("--ema", action="store_true", help="")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="")

    # [NEW - oliviaa/dit_align] DiT alignment args
    parser.add_argument("--dit_ckpt_dir", type=str,
                        default="checkpoints/Wan2.1-I2V-14B-480P",
                        help="DiT checkpoint directory (Wan2.1-I2V-14B). "
                             "Relative to repo root by default; pass absolute path to override.")
    parser.add_argument("--align_weight", type=float, default=0.5,
                        help="alignment loss weight (0 = disable)")
    parser.add_argument("--align_loss_type", type=str, default="l2_mean",
                        choices=["mse", "cosine", "l2_mean"],
                        help="alignment loss type. l2_mean: per-token L2 norm mean (g_loss와 스케일 유사)")
    parser.add_argument("--align_layers", type=str, default="all",
                        help="DiT layers for alignment. 'all' or comma-separated indices e.g. '0,10,20,30,39'")
    parser.add_argument("--align_agg", type=str, default="sum",
                        choices=["sum", "mean"], help="layer별 loss aggregation. sum: layer 합, mean: layer 평균")
    parser.add_argument("--patchify_lr", type=float, default=1e-4,
                        help="student patchify learning rate (separate from VAE lr)")
    parser.add_argument("--patchify_init", type=str, default="zero",
                        choices=["zero", "normal", "kaiming"],
                        help="추가 채널 초기화. zero: pretrained 동작 유지, normal: N(0,0.02), kaiming: Conv default")
    parser.add_argument("--patchify_mask_init", type=str, default="zero",
                        choices=["zero", "copy4_zero4", "copy4_half4"],
                        help="mask_main 8ch 초기화. zero: 전부 zero, copy4_zero4: 앞 4ch pretrained + 뒤 4ch zero, copy4_half4: 복제 × 0.5")
    parser.add_argument("--mask_mode", type=str, default="single8",
                        choices=["single8", "dual12"],
                        help="mask 채널 구성. single8: 8ch(tf=8), dual12: 8ch(tf=8)+4ch(tf=4, pretrained 호환)")
    parser.add_argument("--dit_dit_offload", action="store_true",
                        help="DiT blocks를 CPU에 offload. forward/backward 시에만 GPU로 이동. ~27GB VRAM 절약")
    parser.add_argument("--dit_fsdp2", action="store_true",
                        help="FSDP v2 (fully_shard)로 DiT blocks를 GPU 간 shard. CPU offload보다 빠름")
    parser.add_argument("--text_fsdp2", action="store_true",
                        help="FSDP v2로 UMT5 blocks+token_embedding 및 CLIP transformer를 shard. "
                             "UMT5 ~11.3GB→~2.8GB/GPU, CLIP ~1.3GB→~0.3GB/GPU. t5_offload와 상호 배타적.")
    parser.add_argument("--dit_num_inference_steps", type=int, default=50,
                        help="FlowMatch scheduler num steps (for noise schedule)")
    parser.add_argument("--dit_timestep_mode", type=str, default="random",
                        choices=["random", "fixed"], help="timestep sampling mode")
    parser.add_argument("--dit_fixed_timesteps", type=str, default="0,12,25,37,49",
                        help="fixed timestep indices (used when dit_timestep_mode=fixed)")
    parser.add_argument("--align_after_patchify", action="store_true", default=True,
                        help="patchify 직후 (block 0 입력)에도 alignment loss 적용")
    parser.add_argument("--no_align_after_patchify", dest="align_after_patchify", action="store_false",
                        help="patchify 직후 alignment 비활성화")
    parser.add_argument("--align_adaptive_weight", action="store_true", default=True,
                        help="adaptive weight 사용 (gradient ratio 로 스케일 자동 조정)")
    parser.add_argument("--no_align_adaptive_weight", dest="align_adaptive_weight", action="store_false",
                        help="adaptive weight 비활성화 (align_weight 를 직접 스케일로 사용)")
    # [NEW] Legacy 2-backward adaptive (autograd.grad x 2 + retain_graph). Mutually
    # exclusive with the default single-backward _AdaptiveWeightingFn path.
    # When True: forward() skips _AdaptiveWeightingFn.apply, train loop computes
    # w from gradients on encoder.head[-1].weight via compute_adaptive_weight_2bwd().
    # Requires --align_adaptive_weight (the master switch).
    parser.add_argument("--use_2backward_adaptive", action="store_true", default=False,
                        help="legacy 2-backward adaptive path (autograd.grad x2 + retain_graph). "
                             "Mutually exclusive with single-backward _AdaptiveWeightingFn. "
                             "Requires --align_adaptive_weight. Slower (~+20-30%% step time) but "
                             "matches encoder.head[-1].weight measurement of original production.")
    parser.add_argument("--adaptive_max_weight", type=float, default=1e4,
                        help="upper clamp for adaptive weight ratio (applies to both single-bwd and 2-bwd modes). "
                             "0 or negative = no clamp.")
    parser.add_argument("--log_adaptive_weight", action="store_true",
                        help="wandb log adaptive weight w (raw ratio) + properly-weighted train/align_loss_weighted. "
                             "Default off (avoids per-step .item() cpu sync).")
    parser.add_argument("--no_fused_align", action="store_true",
                        help="origin (kk4aiuyq) 식 alignment 사용: "
                             "run_teacher → run_student → compute_alignment_loss (sum(grad-tracked losses)). "
                             "Default off = fused_dit_align_forward (per-block detach + AlignGradInjector inject, memory 효율). "
                             "True = origin 식 (teacher 의 모든 features 보존, peak memory ↑).")
    # [NEW] LoRA on student DiT (dit_training-compatible setting).
    parser.add_argument("--use_lora", action="store_true",
                        help="Inject LoRA into DiT body (student-side trainable). DiT body weights stay frozen; "
                             "only LoRA params trainable. teacher path disables adapter at runtime.")
    parser.add_argument("--lora_rank", type=int, default=512,
                        help="LoRA rank (and alpha, equal). dit_training default = 512.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,k_img,v_img,ffn.0,ffn.2",
                        help="comma-separated module names to inject LoRA into. dit_training default.")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="optional safetensors LoRA ckpt to load on inject (resume). None = random init.")
    parser.add_argument("--t5_offload", action="store_true",
                        help="null text 계산 후 T5 를 CPU 로 offload. ~10GB VRAM 절약 (caption mode에서는 무시)")
    parser.add_argument("--normalize_zprior", action="store_true",
                        help="z_prior를 pretrained 통계로 normalize. teacher z_ref와 scale 일치.")
    # [NEW] REPA-E style: z_main 도 BN3d 로 online 정규화
    parser.add_argument("--normalize_zmain_bn", action="store_true",
                        help="z_main 을 BatchNorm3d 로 online 정규화 (REPA-E style). running stats EMA update.")
    parser.add_argument("--bn_momentum", type=float, default=0.1,
                        help="BN 의 EMA momentum (PyTorch / REPA-E default 0.1)")
    parser.add_argument("--zmain_bn_init", type=str, default="zprior",
                        choices=["zprior", "cold", "pytorch_default"],
                        help="zprior=z_prior pre_stats 로 init (REPA-E init_bn 동등), cold=0/1, pytorch_default=BN 기본")
    parser.add_argument("--freeze_patchify_zprior", action="store_true",
                        help="student patchify의 z_prior weight 고정 (pretrained copy 유지, z_main만 학습)")
    parser.add_argument("--freeze_patchify_full", action="store_true",
                        help="student patchify 전체 freeze (LoRA-only pure isolation 실험용)")
    # [NEW - oliviaa] RAE-style decoder noise augmentation
    parser.add_argument("--decoder_noise_tau_main", type=float, default=0.0,
                        help="z_main decoder noise: per-sample sigma ~ Uniform[0, tau]. "
                             "Applied in normalized space. Default 0 = no noise. "
                             "Measured DiT inference noise std max ≈ 1.54 → recommend tau ≈ 1.8")
    parser.add_argument("--decoder_noise_tau_prior", type=float, default=0.0,
                        help="z_prior decoder noise: per-sample sigma ~ Uniform[0, tau]. "
                             "Measured DiT inference noise std max ≈ 0.95 → recommend tau ≈ 1.1")
    parser.add_argument("--decoder_noise_random_mode", action="store_true", default=True,
                        help="Per-sample random mode: main_only / prior_only / both (1/3 each). "
                             "Default True. Disable with --no_decoder_noise_random_mode")
    parser.add_argument("--no_decoder_noise_random_mode", dest="decoder_noise_random_mode",
                        action="store_false",
                        help="Always apply noise on both channels (no random mode selection)")
    parser.add_argument("--decoder_noise_warmup_steps", type=int, default=0,
                        help="Curriculum: ramp tau from 0 to final value over this many "
                             "forward steps. Use to avoid cold-start shock when resuming from "
                             "clean-trained decoder. 0 = no warmup (apply full tau immediately).")
    parser.add_argument("--decoder_noise_warmup_power", type=float, default=1.0,
                        help="Power for curriculum schedule: warmup_factor = (step/total)^power. "
                             "1.0 = linear (default). 2.0 = quadratic (slow start, fast end). "
                             "3.0 = cubic (very slow start). 0.5 = sqrt (fast start, slow end).")
    # [NEW - oliviaa] Stage 1.5 (변종 B) — z_main을 precomputed stats로 정규화 후 alignment 학습.
    # 기본값 OFF → Stage 1 (기존) 동작과 동일.
    # ON 시 student_patchify가 normalized z_main에 calibration → Stage 2에서 같은 정규화 적용 시 직접 reuse 가능.
    parser.add_argument("--normalize_zmain", action="store_true",
                        help="[Stage 1.5 변종 B] z_main을 zmain_stats로 정규화해서 alignment에 흘림")
    parser.add_argument("--zmain_stats_path", type=str, default=None,
                        help="[Stage 1.5] z_main 통계 JSON 파일 경로 (--normalize_zmain와 함께 사용)")
    parser.add_argument("--caption_metadata", type=str, default=None,
                        help="caption jsonl 경로 (video→prompt). 지정 시 per-batch text encoding, T5 offload 불가")
    parser.add_argument("--t5_cache_dir", type=str, default=None,
                        help="pre-cached T5 embeddings 디렉토리. 지정 시 T5 없이 학습, offload 가능")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="gradient accumulation steps. effective_batch = batch_size * num_gpu * accum_steps")
    parser.add_argument("--use_grad_checkpoint", action="store_true",
                        help="student DiT branch 에 gradient checkpointing 적용")
    parser.add_argument("--grad_checkpoint_num_blocks", type=int, default=40,
                        help="gradient checkpoint 적용할 block 수 (앞에서부터). 40=전부, 15=앞쪽 15개만")
    parser.add_argument("--align_num_blocks", type=int, default=40,
                        help="alignment에 사용할 DiT block 수. 40=전부, 20=절반. 줄이면 activation 메모리 절약")
    parser.add_argument("--align_block_stride", type=int, default=1,
                        help="block 선택 간격. 1=연속(앞쪽 N개), 2=짝수번째(0,2,4,...). stride>1이면 전체 depth 커버")
    parser.add_argument("--align_batch_size", type=int, default=0,
                        help="alignment에 사용할 batch 크기. 0=전체 batch 사용, 1=첫 샘플만. rec은 전체 batch 유지")
    parser.add_argument("--lpips_chunk_size", type=int, default=0,
                        help="LPIPS를 chunk 단위로 순차 계산. 0=전체 한번에, >0=chunk size. batch_size>1일 때 메모리 절약")
    parser.add_argument("--max_grad_norm", type=float, default=0.0,
                        help="gradient clipping max norm. 0=비활성화, >0=clipping 적용. 7.0 권장")

    args = parser.parse_args()

    # [NEW] Mutual exclusion for adaptive weighting paths.
    if getattr(args, 'use_2backward_adaptive', False):
        assert args.align_adaptive_weight, (
            "--use_2backward_adaptive requires --align_adaptive_weight (master switch). "
            "Use --no_align_adaptive_weight to disable adaptive entirely instead.")
        assert args.align_weight > 0, (
            "--use_2backward_adaptive is meaningless when --align_weight 0.")

    set_random_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
