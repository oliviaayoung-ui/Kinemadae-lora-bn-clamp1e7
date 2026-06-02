"""dit_align.py — DiT forward utilities for geoprior alignment training.

Extracted from train_causalvae_geoprior_dit_align.py.  Contains the pieces
that are specific to running DiT blocks (as opposed to VAE training infra):

  CPUOffloadBlock          block-level CPU offload with checkpoint-style recompute
  create_student_patchify  expand pretrained 36-ch patchify to 104/108 ch
  compute_alignment_loss   per-layer feature alignment loss (sequential)
  AlignGradInjector        identity forward; injects precomputed gradient in backward
  fused_dit_align_forward  teacher+student block loop with inline gradient injection
"""

from __future__ import annotations

import contextlib
import os
import sys


def _maybe_disable_lora(dit):
    """LoRA inject 된 DiT 의 teacher forward 시 adapter off → pretrained behavior.
    inject 안 된 경우 nullcontext 반환."""
    return dit.disable_adapter() if hasattr(dit, 'disable_adapter') else contextlib.nullcontext()

# Ensure external libraries are on the path (same defaults as the training file;
# override via KINEMADAE_DIFFSYNTH_PATH / KINEMADAE_PROBING_PATH env vars).
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for _env, _default in (
    ("KINEMADAE_DIFFSYNTH_PATH", os.path.join(_REPO_ROOT, "external", "DiffSynth-Studio")),
    ("KINEMADAE_PROBING_PATH",   os.path.join(_REPO_ROOT, "external", "dit_probing")),
):
    _p = os.environ.get(_env, _default)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.distributed.fsdp import FSDPModule
except ImportError:
    FSDPModule = None
from einops import rearrange
from PIL import Image
from dit_feature_extractor import (
    load_pipeline, prepare_y, prepare_clip_feature, prepare_null_context,
)
from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d, gradient_checkpoint_forward


# ---------------------------------------------------------------------------
# Block-level CPU offload
# ---------------------------------------------------------------------------

class CPUOffloadBlock(torch.autograd.Function):
    """Block을 CPU에 두고, forward/backward 시에만 GPU로 올리는 autograd function.
    Gradient checkpointing과 동일하게 forward 시 activation 저장하지 않고 backward 시 recompute."""

    @staticmethod
    def forward(ctx, x, block, device, context, t_mod, freqs):
        ctx.block = block
        ctx.device = device
        ctx.save_for_backward(x.detach(), context.detach(), t_mod.detach(), freqs.detach())
        block.to(device)
        with torch.no_grad():
            output = block(x, context, t_mod, freqs)
        block.to('cpu')
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, context, t_mod, freqs = ctx.saved_tensors
        block = ctx.block
        block.to(ctx.device)
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            output = block(x, context, t_mod, freqs)
        torch.autograd.backward(output, grad_output)
        block.to('cpu')
        return x.grad, None, None, None, None, None


# ---------------------------------------------------------------------------
# Student patchify initialisation
# ---------------------------------------------------------------------------

def create_student_patchify(dit, in_channels=None, init_mode='zero', mask_init='zero',
                            mask_mode='single8', z_dim=32, prior_z_dim=16):
    """Pretrained patchify(36→5120) 에서 student patchify 생성.

    입력 순서 (z_cat = [z_main | z_prior]):
    mask_mode='single8':
      [noisy_z_main(z_dim) | noisy_z_prior(prior_z_dim) | mask(8) | image_z_main(z_dim) | image_z_prior(prior_z_dim)]
      → in_channels = (z_dim + prior_z_dim) * 2 + 8  (e.g. 32+16: 104, 16+16: 72)

    mask_mode='dual12':
      [noisy_z_main(z_dim) | noisy_z_prior(prior_z_dim) | mask_main(8) | mask_prior(4) | image_z_main(z_dim) | image_z_prior(prior_z_dim)]
      → in_channels = (z_dim + prior_z_dim) * 2 + 12  (e.g. 32+16: 108, 16+16: 76)

    Pretrained 36ch 구조: [noisy(16) | mask(4) | image(16)]
      - z_prior(prior_z_dim ch)에 pretrained weight copy → step 0에서 pretrained 동작 보존
      - z_main(z_dim ch)은 zero/normal/kaiming init

    init_mode: 'zero' | 'normal' | 'kaiming' — z_main 채널 초기화 방식
    mask_init: 'zero' | 'copy4_zero4' | 'copy4_half4' — mask_main 8ch 초기화
    mask_mode: 'single8' | 'dual12' — mask 채널 구성
    z_dim, prior_z_dim: encoder 의 z_main / z_prior dim (caller 가 args.z_dim, _known.prior_z_dim 전달)
    """
    # 동적 boundary 계산
    mask_ch = 12 if mask_mode == 'dual12' else 8
    if in_channels is None:
        in_channels = (z_dim + prior_z_dim) * 2 + mask_ch
    # noisy: [0, z_dim) | [z_dim, z_dim+prior_z_dim)
    nm_e = z_dim                          # noisy_z_main end
    np_e = nm_e + prior_z_dim             # noisy_z_prior end
    # mask: [np_e, np_e+mask_ch)
    mk_e = np_e + mask_ch                 # mask end (single8: +8, dual12: +12)
    # image: [mk_e, mk_e+z_dim) | [im_e, im_e+prior_z_dim)
    im_e = mk_e + z_dim                   # image_z_main end
    ip_e = im_e + prior_z_dim             # image_z_prior end (= in_channels)
    assert ip_e == in_channels, f"channel layout mismatch: {ip_e} != {in_channels}"

    src = dit.patch_embedding  # Conv3d(36, 5120, kernel=(1,2,2), stride=(1,2,2))
    dst = nn.Conv3d(in_channels, src.out_channels,
                    kernel_size=src.kernel_size, stride=src.stride)
    # src weight layout: [:, 0:16] = noisy, [:, 16:20] = mask, [:, 20:36] = image
    # (pretrained DiT 의 36ch 구조 — z_prior=16 가정. prior_z_dim<16 시 일부만 copy)
    _src_noisy_end = min(prior_z_dim, 16)
    _src_image_end = min(prior_z_dim, 16)
    with torch.no_grad():
        dst.weight.zero_()
        dst.bias.copy_(src.bias)
        # noisy_z_main [0:nm_e) ← init_mode
        if init_mode == 'normal':
            nn.init.normal_(dst.weight[:, 0:nm_e], std=0.02)
        elif init_mode == 'kaiming':
            nn.init.kaiming_uniform_(dst.weight[:, 0:nm_e])
        # noisy_z_prior [nm_e:np_e) ← pretrained noisy [0:_src_noisy_end)
        dst.weight[:, nm_e:nm_e+_src_noisy_end] = src.weight[:, 0:_src_noisy_end]

        if mask_mode == 'dual12':
            # mask_main [np_e:np_e+8) ← mask_init
            mm_s = np_e
            if mask_init == 'copy4_zero4':
                dst.weight[:, mm_s:mm_s+4] = src.weight[:, 16:20]
            elif mask_init == 'copy4_half4':
                dst.weight[:, mm_s:mm_s+4] = src.weight[:, 16:20] * 0.5
                dst.weight[:, mm_s+4:mm_s+8] = src.weight[:, 16:20] * 0.5
            # mask_prior [np_e+8:np_e+12) ← pretrained mask copy (항상)
            dst.weight[:, mm_s+8:mm_s+12] = src.weight[:, 16:20]
            # image_z_main [mk_e:im_e) ← init_mode
            if init_mode == 'normal':
                nn.init.normal_(dst.weight[:, mk_e:im_e], std=0.02)
            elif init_mode == 'kaiming':
                nn.init.kaiming_uniform_(dst.weight[:, mk_e:im_e])
            # image_z_prior [im_e:ip_e) ← pretrained image (20:20+_src_image_end)
            dst.weight[:, im_e:im_e+_src_image_end] = src.weight[:, 20:20+_src_image_end]
        else:  # single8
            # mask [np_e:np_e+8) ← mask_init
            mm_s = np_e
            if mask_init == 'copy4_zero4':
                dst.weight[:, mm_s:mm_s+4] = src.weight[:, 16:20]
            elif mask_init == 'copy4_half4':
                dst.weight[:, mm_s:mm_s+4] = src.weight[:, 16:20] * 0.5
                dst.weight[:, mm_s+4:mm_s+8] = src.weight[:, 16:20] * 0.5
            # image_z_main [mk_e:im_e) ← init_mode
            if init_mode == 'normal':
                nn.init.normal_(dst.weight[:, mk_e:im_e], std=0.02)
            elif init_mode == 'kaiming':
                nn.init.kaiming_uniform_(dst.weight[:, mk_e:im_e])
            # image_z_prior [im_e:ip_e) ← pretrained image (20:20+_src_image_end)
            dst.weight[:, im_e:im_e+_src_image_end] = src.weight[:, 20:20+_src_image_end]
    return dst


# ---------------------------------------------------------------------------
# Alignment loss
# ---------------------------------------------------------------------------

def compute_alignment_loss(features_stu, features_ref, grid_stu, grid_ref,
                            loss_type='mse', selected_layers=None, agg='sum'):
    """[NEW - oliviaa/dit_align] Per-layer DiT feature alignment loss.
    features_stu[l] 을 spatial reshape → trilinear upsample → features_ref[l] 과 비교.

    Args:
        features_stu: list[(B, seq_stu, D)] — student DiT block outputs (in graph)
        features_ref: list[(B, seq_ref, D)] — teacher DiT block outputs (detached)
        grid_stu: (f, h, w) — student 의 patchified spatial grid
        grid_ref: (f, h, w) — teacher 의 patchified spatial grid
        loss_type: 'mse' | 'cosine' | 'l2_mean'
        selected_layers: list[int] 또는 None (전체)
        agg: 'sum' | 'mean' — layer 별 loss aggregation

    Returns:
        (total_loss, per_layer_losses dict)
    """
    per_layer = {}
    f_s, h_s, w_s = grid_stu
    f_r, h_r, w_r = grid_ref
    for l, (feat_s, feat_r) in enumerate(zip(features_stu, features_ref)):
        if selected_layers is not None and l not in selected_layers:
            continue
        B, _, D = feat_s.shape
        feat_s_3d = feat_s.reshape(B, f_s, h_s, w_s, D).permute(0, 4, 1, 2, 3)
        feat_s_up = F.interpolate(feat_s_3d.float(), size=(f_r, h_r, w_r),
                                   mode='trilinear', align_corners=False)
        feat_s_up = feat_s_up.permute(0, 2, 3, 4, 1).reshape(B, -1, D)
        # features_ref가 CPU에 있을 수 있으므로 GPU로 이동
        feat_r_gpu = feat_r.to(feat_s.device) if feat_r.device != feat_s.device else feat_r
        if loss_type == 'mse':
            loss_l = F.mse_loss(feat_s_up, feat_r_gpu.float())
        elif loss_type == 'cosine':
            loss_l = 1.0 - (F.normalize(feat_s_up, dim=-1) *
                            F.normalize(feat_r_gpu.float(), dim=-1)).sum(-1).mean()
        elif loss_type == 'l2_mean':
            loss_l = (feat_s_up - feat_r_gpu.float()).pow(2).sum(dim=-1).sqrt().mean()
        name = 'patch' if l == 0 else f'b{l-1}'
        per_layer[name] = loss_l
    losses = list(per_layer.values())
    if not losses:
        return torch.tensor(0.0), {}
    if agg == 'sum':
        total = sum(losses)
    else:
        total = sum(losses) / len(losses)
    return total, per_layer


# ---------------------------------------------------------------------------
# Teacher forward pass
# ---------------------------------------------------------------------------

def run_teacher_forward(
    dit, dit_pipe, inputs_align, scheduler, timestep, t_tensor,
    null_context, _align_block_set, _dit_offload, _dit_fsdp2,
    align_after_patchify, rank, precision,
    _use_t5_cache, t5_cache, _use_caption, caption_map, batch, _align_bs,
    logger=None,
):
    """Branch 1: pretrained Wan VAE encode → noisy → teacher patchify → blocks.

    Runs entirely under torch.no_grad(). Returns the collected teacher features,
    the patchified grid shape, and the shared conditioning tensors (context,
    t_mod, freqs_ref) that are reused by the student branch.

    Returns:
        features_ref : list of (B, seq_ref, D) CPU tensors
        grid_ref     : (f_r, h_r, w_r)
        context      : (B, seq_ctx, D) on rank device
        t_mod        : (B, 6, D) on rank device
        freqs_ref    : (seq_ref, 1, rope_dim) on rank device
    """
    with torch.no_grad():
        # [FIX - oliviaa] VAE encode list API: sample 별로 encode → cat.
        z_ref_list = []
        for i in range(_align_bs):
            z_i = dit_pipe.vae.encode(
                [inputs_align[i].to(dtype=precision)], device=rank, tiled=True
            )[0].unsqueeze(0)
            z_ref_list.append(z_i)
        z_ref = torch.cat(z_ref_list, dim=0).to(device=rank, dtype=precision)
        noise_ref = torch.randn_like(z_ref)
        noisy_ref = scheduler.add_noise(z_ref, noise_ref, timestep)

        # I2V conditioning at z_ref resolution
        h_ref, w_ref = z_ref.shape[3] * 8, z_ref.shape[4] * 8
        ys_ref = []
        clips_ref = []
        for i in range(_align_bs):
            first_frame = inputs_align[i, :, 0]  # (3, H, W)
            first_np = ((first_frame * 0.5 + 0.5).clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
            pil_img = Image.fromarray(first_np)
            ys_ref.append(prepare_y(dit_pipe, pil_img, inputs_align.shape[2], h_ref, w_ref))
            clips_ref.append(prepare_clip_feature(dit_pipe, pil_img, h_ref, w_ref))
        y_ref = torch.cat(ys_ref, dim=0).to(device=rank, dtype=precision)
        clip_ref = torch.cat(clips_ref, dim=0).to(device=rank, dtype=precision)

        # Teacher forward: patchify → blocks
        x_ref = torch.cat([noisy_ref, y_ref], dim=1)
        x_ref = dit.patch_embedding(x_ref)
        f_r, h_r, w_r = x_ref.shape[2:]
        x_ref = rearrange(x_ref, 'b c f h w -> b (f h w) c').contiguous()

        # Time + text embeddings
        t_emb = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, t_tensor).to(precision))
        t_mod = dit.time_projection(t_emb).unflatten(1, (6, dit.dim))
        if _use_t5_cache:
            video_paths = batch.get("video_path", [""] * _align_bs)[:_align_bs]
            text_contexts = []
            for p in video_paths:
                cached = t5_cache.get(p, None)
                if cached is not None:
                    pad_len = null_context.shape[1] - cached.shape[0]
                    if pad_len > 0:
                        cached = torch.cat([cached, torch.zeros(pad_len, cached.shape[1])], dim=0)
                    text_contexts.append(cached.unsqueeze(0))
                else:
                    text_contexts.append(null_context.cpu())
            text_ctx = torch.cat(text_contexts, dim=0).to(device=rank, dtype=precision)
        elif _use_caption:
            captions = [caption_map.get(p, "") for p in batch.get("video_path", [""] * _align_bs)[:_align_bs]]
            text_contexts = []
            for cap in captions:
                ids, mask = dit_pipe.tokenizer(cap, return_mask=True, add_special_tokens=True)
                ids, mask = ids.to(rank), mask.to(rank)
                seq_lens = mask.gt(0).sum(dim=1).long()
                tc = dit_pipe.text_encoder(ids, mask)
                for i, v in enumerate(seq_lens):
                    tc[:, v:] = 0
                text_contexts.append(tc)
            text_ctx = torch.cat(text_contexts, dim=0).to(dtype=precision)
        else:
            text_ctx = null_context.expand(_align_bs, -1, -1)
        context = dit.text_embedding(text_ctx)
        clip_emb = dit.img_emb(clip_ref)
        context = torch.cat([clip_emb, context], dim=1)

        freqs_ref = torch.cat([
            dit.freqs[0][:f_r].view(f_r, 1, 1, -1).expand(f_r, h_r, w_r, -1),
            dit.freqs[1][:h_r].view(1, h_r, 1, -1).expand(f_r, h_r, w_r, -1),
            dit.freqs[2][:w_r].view(1, 1, w_r, -1).expand(f_r, h_r, w_r, -1),
        ], dim=-1).reshape(f_r * h_r * w_r, 1, -1).to(rank)

        features_ref = []
        if align_after_patchify:
            features_ref.append(x_ref.detach().cpu())
        for _bi, block in enumerate(dit.blocks):
            if _bi not in _align_block_set:
                continue
            if _dit_offload:
                block.to(rank)
            with _maybe_disable_lora(dit):
                x_ref = block(x_ref, context, t_mod, freqs_ref)
            features_ref.append(x_ref.detach().cpu())
            if _dit_offload:
                block.to('cpu')
            if FSDPModule is not None and isinstance(block, FSDPModule):
                block.reshard()

        if logger is not None:
            _mem = torch.cuda.memory_allocated(rank) / 1e9
            logger.info(f"[mem] after teacher forward: allocated={_mem:.2f}GB")

    return features_ref, (f_r, h_r, w_r), context, t_mod, freqs_ref


# ---------------------------------------------------------------------------
# Student forward pass
# ---------------------------------------------------------------------------

def run_student_forward(
    student_patchify, dit, inputs_align, z_cat_align, scheduler, timestep,
    context, t_mod, model_module, mask_mode,
    _align_block_set, _dit_offload, _use_gc, grad_checkpoint_num_blocks,
    align_after_patchify, rank, precision,
    retain_grads=False, logger=None,
):
    """Branch 2: geoprior z_cat → noisy → student patchify → blocks.

    Grad flows through student patchify back to z_cat_align (and through it
    to the VAE encoder).  context and t_mod come from run_teacher_forward.

    Args:
        model_module          : unwrapped model (model.module) for VAE encode
        retain_grads          : call .retain_grad() on noisy_cat and patchify
                                output (for gradient logging)

    Returns:
        features_stu          : list of (B, seq_stu, D) grad-tracked tensors
        grid_stu              : (f_s, h_s, w_s)
        noisy_cat             : noisy z_cat (retained grad if retain_grads)
        x_stu_post_patchify   : patchify output (retained grad if retain_grads)
    """
    noise_cat = torch.randn_like(z_cat_align)
    noisy_cat = scheduler.add_noise(z_cat_align, noise_cat, timestep)
    if retain_grads:
        noisy_cat.retain_grad()
        z_cat_align.retain_grad()

    # I2V conditioning: geoprior VAE encode(48ch) + mask(8ch)
    B_align, _, T_lat, H_lat, W_lat = z_cat_align.shape
    tf_stu = 8  # geoprior VAE temporal factor
    num_frames = inputs_align.shape[2]
    with torch.no_grad():
        # image conditioning: encode first frame with geoprior VAE → 48ch
        img_input = torch.zeros_like(inputs_align)
        img_input[:, :, 0:1] = inputs_align[:, :, 0:1]
        mu_img, _ = model_module.vae.encode(img_input, scale=None)
        # [NEW - oliviaa] 변종 B: image_z_main도 정규화 (alignment 경로 일관성)
        mu_img = model_module._norm_zmain(mu_img)
        z_prior_img = model_module.vae._encode_prior(img_input)
        z_prior_img = model_module._norm_zprior(z_prior_img)
        image_z_cat = torch.cat([mu_img, z_prior_img], dim=1)  # (B_align, 48, T_lat, H_lat, W_lat)

        # mask_main: tf=8, Wan causal 1+tf*k 구조
        msk_main = torch.ones(B_align, num_frames, H_lat, W_lat, device=rank)
        msk_main[:, 1:] = 0
        msk_main = torch.cat([msk_main[:, 0:1].repeat(1, tf_stu, 1, 1), msk_main[:, 1:]], dim=1)
        msk_main = msk_main.view(B_align, msk_main.shape[1] // tf_stu, tf_stu, H_lat, W_lat)
        msk_main = msk_main.transpose(1, 2)  # (B_align, 8, T_lat, H_lat, W_lat)
        msk_main = msk_main.to(dtype=precision)

        if mask_mode == 'dual12':
            tf_prior = 4
            num_frames_prior = (num_frames + 1) // 2  # avg_pool3d 후 프레임 수
            msk_prior = torch.ones(B_align, num_frames_prior, H_lat, W_lat, device=rank)
            msk_prior[:, 1:] = 0
            msk_prior = torch.cat([msk_prior[:, 0:1].repeat(1, tf_prior, 1, 1), msk_prior[:, 1:]], dim=1)
            msk_prior = msk_prior.view(B_align, msk_prior.shape[1] // tf_prior, tf_prior, H_lat, W_lat)
            msk_prior = msk_prior.transpose(1, 2)  # (B_align, 4, T_lat, H_lat, W_lat)
            msk_prior = msk_prior.to(dtype=precision)

    if mask_mode == 'dual12':
        # [noisy_z_cat(48) | mask_main(8) | mask_prior(4) | image_z_cat(48)] = 108ch
        x_stu = torch.cat([noisy_cat, msk_main, msk_prior, image_z_cat], dim=1)
    else:
        # [noisy_z_cat(48) | mask(8) | image_z_cat(48)] = 104ch
        x_stu = torch.cat([noisy_cat, msk_main, image_z_cat], dim=1)

    if retain_grads:
        x_stu.retain_grad()
    x_stu = student_patchify(x_stu)
    x_stu_post_patchify = x_stu
    if retain_grads:
        x_stu.retain_grad()
    f_s, h_s, w_s = x_stu.shape[2:]
    x_stu = rearrange(x_stu, 'b c f h w -> b (f h w) c').contiguous()

    freqs_stu = torch.cat([
        dit.freqs[0][:f_s].view(f_s, 1, 1, -1).expand(f_s, h_s, w_s, -1),
        dit.freqs[1][:h_s].view(1, h_s, 1, -1).expand(f_s, h_s, w_s, -1),
        dit.freqs[2][:w_s].view(1, 1, w_s, -1).expand(f_s, h_s, w_s, -1),
    ], dim=-1).reshape(f_s * h_s * w_s, 1, -1).to(rank)

    features_stu = []
    if align_after_patchify:
        features_stu.append(x_stu)
    for bi, block in enumerate(dit.blocks):
        if bi not in _align_block_set:
            continue
        if _dit_offload:
            x_stu = CPUOffloadBlock.apply(x_stu, block, rank, context, t_mod, freqs_stu)
        elif _use_gc and bi < grad_checkpoint_num_blocks:
            x_stu = gradient_checkpoint_forward(block, True, False, x_stu, context, t_mod, freqs_stu)
        else:
            x_stu = block(x_stu, context, t_mod, freqs_stu)
        features_stu.append(x_stu)

        if FSDPModule is not None and isinstance(block, FSDPModule):
            block.reshard()

    if logger is not None:
        _mem = torch.cuda.memory_allocated(rank) / 1e9
        logger.info(f"[mem] after student forward: allocated={_mem:.2f}GB")

    return features_stu, (f_s, h_s, w_s), noisy_cat, x_stu_post_patchify


# ---------------------------------------------------------------------------
# DiT memory / FSDP setup
# ---------------------------------------------------------------------------

def setup_dit_memory(
    dit, args, rank, global_rank,
    has_fsdp2, fully_shard, MixedPrecisionPolicy,
    logger=None,
):
    """Configure DiT block memory layout (FSDP v2 / CPU offload / partial offload).

    Returns:
        _dit_offload    : bool — blocks live on CPU, moved per-forward
        _dit_fsdp2      : bool — blocks wrapped with FSDP v2
        _align_block_set: set[int] — block indices kept on GPU for alignment
    """
    _dit_offload = False
    _dit_fsdp2 = False
    _align_block_set = set()

    if dit is None:
        return _dit_offload, _dit_fsdp2, _align_block_set

    _dit_offload = getattr(args, 'dit_cpu_offload', False)
    _dit_fsdp2 = getattr(args, 'dit_fsdp2', False)

    # Compute align_block_set unconditionally so all branches can use it.
    _align_n_blocks = getattr(args, 'align_num_blocks', 40)
    _align_stride = getattr(args, 'align_block_stride', 1)
    _total_blocks = len(dit.blocks)
    if _align_stride > 1:
        _align_block_indices = list(range(0, _total_blocks, _align_stride))[:_align_n_blocks]
    else:
        _align_block_indices = list(range(min(_align_n_blocks, _total_blocks)))
    _align_block_set = set(_align_block_indices)

    if global_rank == 0 and logger is not None:
        _mem = torch.cuda.memory_allocated(rank) / 1e9
        logger.info(f"GPU {rank} memory BEFORE FSDP/offload: {_mem:.2f}GB")

    if _dit_fsdp2:
        assert has_fsdp2, "FSDP v2 requires PyTorch 2.0+ with torch.distributed._composable.fsdp"
        _fsdp2_mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        _fsdp2_kw = dict(mp_policy=_fsdp2_mp, reshard_after_forward=False)
        for bi, block in enumerate(dit.blocks):
            if bi not in _align_block_set:
                block.to('cpu')
                continue
            fully_shard(block, **_fsdp2_kw)
        torch.cuda.empty_cache()
        if global_rank == 0 and logger is not None:
            _mem_a = torch.cuda.memory_allocated(rank) / 1e9
            _mem_r = torch.cuda.memory_reserved(rank) / 1e9
            logger.info(f"DiT wrapped with FSDP v2 (fully_shard, {len(dit.blocks)} blocks)")
            logger.info(f"  GPU {rank} memory after FSDP: allocated={_mem_a:.2f}GB, reserved={_mem_r:.2f}GB")
            b0 = list(dit.blocks[0].parameters())[0]
            logger.info(f"  block[0] param: type={type(b0).__name__}, shape={b0.shape}, device={b0.device}")
            if hasattr(b0, '_local_tensor'):
                logger.info(f"  block[0] local_tensor shape: {b0._local_tensor.shape}")
            pe = list(dit.patch_embedding.parameters())[0]
            logger.info(f"  patch_embedding param: type={type(pe).__name__}, shape={pe.shape}")

    if _dit_offload:
        for block in dit.blocks:
            block.to('cpu')
        torch.cuda.empty_cache()
        if global_rank == 0 and logger is not None:
            logger.info(f"DiT blocks offloaded to CPU ({len(dit.blocks)} blocks)")

    else:
        if len(_align_block_set) < _total_blocks:
            for bi, block in enumerate(dit.blocks):
                if bi not in _align_block_set:
                    block.to('cpu')
            torch.cuda.empty_cache()
            if global_rank == 0 and logger is not None:
                _mem = torch.cuda.memory_allocated(rank) / 1e9
                logger.info(f"DiT align blocks: {sorted(_align_block_indices)} "
                            f"({len(_align_block_indices)}/{_total_blocks} on GPU). "
                            f"GPU memory: {_mem:.2f}GB")

    return _dit_offload, _dit_fsdp2, _align_block_set


# ---------------------------------------------------------------------------
# Text encoder (UMT5 + CLIP) memory / FSDP setup
# ---------------------------------------------------------------------------

def setup_text_encoder_memory(
    dit_pipe, args, rank, global_rank,
    has_fsdp2, fully_shard, MixedPrecisionPolicy,
    logger=None,
):
    """FSDP v2-shard UMT5 (text_encoder) and CLIP (image_encoder) blocks.

    Shards:
      - text_encoder.blocks        (24 × T5SelfAttention, ~4631M params)
      - text_encoder.token_embedding (Embedding, ~1050M params)
      - image_encoder.model.visual.transformer (32 × AttentionBlock, ~627M params)

    With 4 GPUs this reduces UMT5 from ~11.3 GB to ~2.8 GB/GPU and CLIP visual
    from ~1.3 GB to ~0.3 GB/GPU.  Mutually exclusive with --t5_offload.

    Returns:
        _text_fsdp2: bool — True if sharding was applied
    """
    _text_fsdp2 = getattr(args, 'text_fsdp2', False)
    if not _text_fsdp2 or dit_pipe is None:
        return False

    assert has_fsdp2, "FSDP v2 requires PyTorch 2.0+ with torch.distributed._composable.fsdp"
    _fsdp2_mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    _fsdp2_kw = dict(mp_policy=_fsdp2_mp, reshard_after_forward=True)

    n_t5, n_clip = 0, 0

    if hasattr(dit_pipe, 'text_encoder') and dit_pipe.text_encoder is not None:
        te = dit_pipe.text_encoder
        for block in te.blocks:
            fully_shard(block, **_fsdp2_kw)
            n_t5 += 1
        fully_shard(te.token_embedding, **_fsdp2_kw)

    if hasattr(dit_pipe, 'image_encoder') and dit_pipe.image_encoder is not None:
        vit = dit_pipe.image_encoder.model.visual
        for block in vit.transformer:
            fully_shard(block, **_fsdp2_kw)
            n_clip += 1

    torch.cuda.empty_cache()

    if global_rank == 0 and logger is not None:
        _mem_a = torch.cuda.memory_allocated(rank) / 1e9
        _mem_r = torch.cuda.memory_reserved(rank) / 1e9
        logger.info(
            f"T5+CLIP wrapped with FSDP v2 "
            f"({n_t5} T5 blocks + token_embedding, {n_clip} CLIP blocks)"
        )
        logger.info(f"  GPU {rank} after text FSDP: allocated={_mem_a:.2f}GB, reserved={_mem_r:.2f}GB")

    return True


# ---------------------------------------------------------------------------
# Fused teacher+student forward with inline gradient injection
# ---------------------------------------------------------------------------

class AlignGradInjector(torch.autograd.Function):
    """Identity forward; injects precomputed alignment gradient in backward.

    Keeps teacher memory O(1): g_align is computed on a detached student copy
    immediately after each block, then the teacher activation is freed on the
    next loop iteration when x_ref is reassigned.  Only g_align (same shape as
    x_stu) persists in ctx until backward.

    backward(grad_out) returns grad_out + g_align, so upstream gradients from
    both the reconstruction loss and the alignment signal are summed correctly
    in a single backward pass.
    """

    @staticmethod
    def forward(ctx, x_stu, g_align):
        ctx.save_for_backward(g_align)
        return x_stu

    @staticmethod
    def backward(ctx, grad_out):
        g_align, = ctx.saved_tensors
        return grad_out + g_align, None


def _align_loss_single(feat_s, feat_r, grid_stu, grid_ref, loss_type):
    """Single-layer alignment loss with optional trilinear upsample."""
    f_s, h_s, w_s = grid_stu
    f_r, h_r, w_r = grid_ref
    B, _, D = feat_s.shape
    feat_s_3d = feat_s.reshape(B, f_s, h_s, w_s, D).permute(0, 4, 1, 2, 3).float()
    if (f_s, h_s, w_s) != (f_r, h_r, w_r):
        feat_s_3d = F.interpolate(feat_s_3d, size=(f_r, h_r, w_r),
                                   mode='trilinear', align_corners=False)
    feat_s_up = feat_s_3d.permute(0, 2, 3, 4, 1).reshape(B, -1, D)
    feat_r = (feat_r.to(feat_s.device) if feat_r.device != feat_s.device else feat_r).float()
    if loss_type == 'mse':
        return F.mse_loss(feat_s_up, feat_r)
    elif loss_type == 'cosine':
        return 1.0 - (F.normalize(feat_s_up, dim=-1) * F.normalize(feat_r, dim=-1)).sum(-1).mean()
    else:  # l2_mean
        return (feat_s_up - feat_r).pow(2).sum(dim=-1).sqrt().mean()


def fused_dit_align_forward(
    dit, student_patchify, dit_pipe,
    inputs_align, z_cat_align,
    scheduler, timestep, t_tensor,
    null_context, model_module, mask_mode,
    _align_block_set, _dit_offload, _dit_fsdp2,
    _use_gc, grad_checkpoint_num_blocks,
    align_after_patchify, rank, precision,
    align_weight=1.0,
    loss_type='mse', selected_layers=None, agg='sum',
    _use_t5_cache=False, t5_cache=None,
    _use_caption=False, caption_map=None,
    batch=None, _align_bs=1,
    retain_grads=False, logger=None,
):
    """Fused teacher+student DiT forward with per-block alignment gradient injection.

    For each aligned block:
      1. Teacher block runs under no_grad.
      2. Student block runs with grad (supports gradient_checkpoint_forward and
         CPUOffloadBlock; FSDP v2 pre/post-forward hooks fire normally for both).
      3. g_align = autograd.grad(align_weight * loss_l, x_stu.detach()) is computed.
      4. AlignGradInjector injects g_align into the live student graph.
      5. The old x_ref is freed on the next iteration when the variable is reassigned.

    FSDP2 note: the teacher no_grad pass triggers unshard/reshard via the usual
    hooks; an explicit reshard() call follows as a safety measure (same as
    run_teacher_forward).  The student pass then unshards again; FSDP2 backward
    hooks handle resharding during the backward pass.

    GC note: gradient_checkpoint_forward saves only inputs and recomputes
    activations in backward.  AlignGradInjector sits between block outputs and
    does not interfere with the recompute — g_align is stored in ctx, not
    recomputed.

    Args:
        align_weight : multiplied into g_align before injection.
                       Pass args.align_weight for the non-adaptive case;
                       pass 1.0 when _AdaptiveWeightingFn handles scaling.

    Returns:
        align_loss          : grad-tracked scalar (= injected grad path + detached total);
                              add directly to total_loss.
        per_layer           : dict[str, Tensor] of per-layer losses (detached).
        noisy_cat           : noisy z_cat (grad retained when retain_grads).
        x_stu_post_patchify : student patchify output (grad retained when retain_grads).
    """
    # ── Teacher pre-block setup (no_grad) ───────────────────────────────────
    with torch.no_grad():
        z_ref = dit_pipe.vae.single_encode(inputs_align[:_align_bs].to(dtype=precision), device=rank)
        noise_ref = torch.randn_like(z_ref)
        noisy_ref = scheduler.add_noise(z_ref, noise_ref, timestep)

        h_ref, w_ref = z_ref.shape[3] * 8, z_ref.shape[4] * 8
        ys_ref, clips_ref = [], []
        for i in range(_align_bs):
            first_frame = inputs_align[i, :, 0]
            first_np = ((first_frame * 0.5 + 0.5).clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
            pil_img = Image.fromarray(first_np)
            ys_ref.append(prepare_y(dit_pipe, pil_img, inputs_align.shape[2], h_ref, w_ref, device=rank))
            clips_ref.append(prepare_clip_feature(dit_pipe, pil_img, h_ref, w_ref))
        y_ref = torch.cat(ys_ref, dim=0).to(device=rank, dtype=precision)
        clip_ref = torch.cat(clips_ref, dim=0).to(device=rank, dtype=precision)

        x_ref = torch.cat([noisy_ref, y_ref], dim=1)
        x_ref = dit.patch_embedding(x_ref)
        f_r, h_r, w_r = x_ref.shape[2:]
        x_ref = rearrange(x_ref, 'b c f h w -> b (f h w) c').contiguous()

        t_emb = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, t_tensor).to(precision))
        t_mod = dit.time_projection(t_emb).unflatten(1, (6, dit.dim))

        if _use_t5_cache:
            video_paths = batch.get("video_path", [""] * _align_bs)[:_align_bs]
            text_contexts = []
            for p in video_paths:
                cached = t5_cache.get(p, None)
                if cached is not None:
                    pad_len = null_context.shape[1] - cached.shape[0]
                    if pad_len > 0:
                        cached = torch.cat([cached, torch.zeros(pad_len, cached.shape[1])], dim=0)
                    text_contexts.append(cached.unsqueeze(0))
                else:
                    text_contexts.append(null_context.cpu())
            text_ctx = torch.cat(text_contexts, dim=0).to(device=rank, dtype=precision)
        elif _use_caption:
            captions = [caption_map.get(p, "") for p in batch.get("video_path", [""] * _align_bs)[:_align_bs]]
            text_contexts = []
            for cap in captions:
                ids, mask = dit_pipe.tokenizer(cap, return_mask=True, add_special_tokens=True)
                ids, mask = ids.to(rank), mask.to(rank)
                seq_lens = mask.gt(0).sum(dim=1).long()
                tc = dit_pipe.text_encoder(ids, mask)
                for i, v in enumerate(seq_lens):
                    tc[:, v:] = 0
                text_contexts.append(tc)
            text_ctx = torch.cat(text_contexts, dim=0).to(dtype=precision)
        else:
            text_ctx = null_context.expand(_align_bs, -1, -1)
        context = dit.text_embedding(text_ctx)
        clip_emb = dit.img_emb(clip_ref)
        context = torch.cat([clip_emb, context], dim=1)

        freqs_ref = torch.cat([
            dit.freqs[0][:f_r].view(f_r, 1, 1, -1).expand(f_r, h_r, w_r, -1),
            dit.freqs[1][:h_r].view(1, h_r, 1, -1).expand(f_r, h_r, w_r, -1),
            dit.freqs[2][:w_r].view(1, 1, w_r, -1).expand(f_r, h_r, w_r, -1),
        ], dim=-1).reshape(f_r * h_r * w_r, 1, -1).to(rank)

    # ── Student pre-block setup ──────────────────────────────────────────────
    noise_cat = torch.randn_like(z_cat_align)
    noisy_cat = scheduler.add_noise(z_cat_align, noise_cat, timestep)
    if retain_grads:
        noisy_cat.retain_grad()
        z_cat_align.retain_grad()

    B_align, _, T_lat, H_lat, W_lat = z_cat_align.shape
    tf_stu = 8
    num_frames = inputs_align.shape[2]
    with torch.no_grad():
        img_input = torch.zeros_like(inputs_align)
        img_input[:, :, 0:1] = inputs_align[:, :, 0:1]
        mu_img, _ = model_module.vae.encode(img_input, scale=None)
        mu_img = model_module._norm_zmain(mu_img)
        z_prior_img = model_module.vae._encode_prior(img_input)
        z_prior_img = model_module._norm_zprior(z_prior_img)
        image_z_cat = torch.cat([mu_img, z_prior_img], dim=1)

        msk_main = torch.ones(B_align, num_frames, H_lat, W_lat, device=rank)
        msk_main[:, 1:] = 0
        msk_main = torch.cat([msk_main[:, 0:1].repeat(1, tf_stu, 1, 1), msk_main[:, 1:]], dim=1)
        msk_main = msk_main.view(B_align, msk_main.shape[1] // tf_stu, tf_stu, H_lat, W_lat)
        msk_main = msk_main.transpose(1, 2).to(dtype=precision)

        if mask_mode == 'dual12':
            tf_prior = 4
            num_frames_prior = (num_frames + 1) // 2
            msk_prior = torch.ones(B_align, num_frames_prior, H_lat, W_lat, device=rank)
            msk_prior[:, 1:] = 0
            msk_prior = torch.cat([msk_prior[:, 0:1].repeat(1, tf_prior, 1, 1), msk_prior[:, 1:]], dim=1)
            msk_prior = msk_prior.view(B_align, msk_prior.shape[1] // tf_prior, tf_prior, H_lat, W_lat)
            msk_prior = msk_prior.transpose(1, 2).to(dtype=precision)

    if mask_mode == 'dual12':
        x_stu_in = torch.cat([noisy_cat, msk_main, msk_prior, image_z_cat], dim=1)
    else:
        x_stu_in = torch.cat([noisy_cat, msk_main, image_z_cat], dim=1)

    if retain_grads:
        x_stu_in.retain_grad()
    x_stu = student_patchify(x_stu_in)
    x_stu_post_patchify = x_stu
    if retain_grads:
        x_stu.retain_grad()
    f_s, h_s, w_s = x_stu.shape[2:]
    x_stu = rearrange(x_stu, 'b c f h w -> b (f h w) c').contiguous()

    freqs_stu = torch.cat([
        dit.freqs[0][:f_s].view(f_s, 1, 1, -1).expand(f_s, h_s, w_s, -1),
        dit.freqs[1][:h_s].view(1, h_s, 1, -1).expand(f_s, h_s, w_s, -1),
        dit.freqs[2][:w_s].view(1, 1, w_s, -1).expand(f_s, h_s, w_s, -1),
    ], dim=-1).reshape(f_s * h_s * w_s, 1, -1).to(rank)

    # ── Fused block loop ─────────────────────────────────────────────────────
    grid_stu = (f_s, h_s, w_s)
    grid_ref = (f_r, h_r, w_r)
    per_layer: dict = {}
    l = 0

    if align_after_patchify:
        if selected_layers is None or l in selected_layers:
            feat_s_det = x_stu.detach().requires_grad_(True)
            loss_l = _align_loss_single(feat_s_det, x_ref, grid_stu, grid_ref, loss_type)
            g = torch.autograd.grad(align_weight * loss_l, feat_s_det)[0]
            x_stu = AlignGradInjector.apply(x_stu, g)
            per_layer['patch'] = loss_l.detach()
        l += 1

    for bi, block in enumerate(dit.blocks):
        if bi not in _align_block_set:
            continue

        # Teacher block (no_grad + LoRA disabled) — teacher uses pretrained DiT.
        with torch.no_grad(), _maybe_disable_lora(dit):
            if _dit_offload:
                block.to(rank)
            x_ref = block(x_ref, context, t_mod, freqs_ref)
            if _dit_offload:
                block.to('cpu')

        # Student block (grad tracked) — same dispatch as run_student_forward
        if _dit_offload:
            x_stu = CPUOffloadBlock.apply(x_stu, block, rank, context, t_mod, freqs_stu)
        elif _use_gc and bi < grad_checkpoint_num_blocks:
            x_stu = gradient_checkpoint_forward(block, True, False, x_stu, context, t_mod, freqs_stu)
        else:
            x_stu = block(x_stu, context, t_mod, freqs_stu)

        if FSDPModule is not None and isinstance(block, FSDPModule):
            block.reshard()

        # Compute g_align on a detached copy; inject into live student graph.
        # autograd.grad frees loss_l's graph, releasing the reference to x_ref.
        # x_ref itself is freed on the next iteration when the variable is reassigned.
        if selected_layers is None or l in selected_layers:
            feat_s_det = x_stu.detach().requires_grad_(True)
            loss_l = _align_loss_single(feat_s_det, x_ref, grid_stu, grid_ref, loss_type)
            g = torch.autograd.grad(align_weight * loss_l, feat_s_det)[0]
            x_stu = AlignGradInjector.apply(x_stu, g)
            per_layer[f'b{bi}'] = loss_l.detach()
        l += 1

    # Zero-valued trigger: traces backward through the entire student chain so
    # that AlignGradInjector.backward fires and alignment gradients flow back
    # through student_patchify into z_cat_align.
    align_trigger = (x_stu * 0).sum()

    losses = list(per_layer.values())
    if not losses:
        total_align_loss = torch.tensor(0.0, device=rank)
    else:
        total = sum(v.item() for v in losses)
        total_align_loss = torch.tensor(total / len(losses) if agg != 'sum' else total, device=rank)

    if logger is not None:
        _mem = torch.cuda.memory_allocated(rank) / 1e9
        logger.info(f"[mem] after fused_dit_align_forward: allocated={_mem:.2f}GB")

    return align_trigger + total_align_loss, per_layer, noisy_cat, x_stu_post_patchify
