# [NEW - oliviaa] DiT intermediate feature extractor.
#
# Wan 2.1 I2V-14B DiT 에서 intermediate transformer block output 을 hook 으로 뽑아
# mean pool → (B, 5120) feature 를 반환.
#
# 가져온 것:
#   [COPIED] DiffSynth sys.path 주입 — inference_pretrained_abl.py:24-29
#   [COPIED] build_model_paths() — inference_pretrained_abl.py:165-173
#   [COPIED] load_lora_checkpoint() — inference_pretrained_abl.py:138-162
#   [COPIED] FlowMatchScheduler — DiffSynth/diffsynth/diffusion/flow_match.py
#   [COPIED] WanVideoPipeline.from_pretrained — inference_pretrained_abl.py:231-236
#   [COPIED] I2V y 조립 (mask + vae encode) — wan_video.py:485-508
#   [COPIED] CLIP image encoding — wan_video.py:462-473
#   [COPIED] scheduler.add_noise — flow_match.py:221-227
#
# 새로 작성한 것:
#   [NEW] _register_hooks() / _remove_hooks() — DiTBlock forward hook
#   [NEW] extract() — hooked forward + mean pool
#
# 유일한 변경점 vs I2V inference:
#   - text prompt = null (empty string). Class label inject 방지.
#   - 나머지 (CLIP, mask, y, scheduler, timestep) 전부 I2V inference 와 동일.

import logging
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# [COPIED from inference_pretrained_abl.py:24-29]
_DIFFSYNTH = "/data/karlo-research_715/workspace/kinemadae/projects/oliviaa/DiffSynth-Studio"
if _DIFFSYNTH not in sys.path:
    sys.path.insert(0, _DIFFSYNTH)

_KINEMADAE = "/data/karlo-research_715/workspace/kinemadae/projects/oliviaa/KinemaDAE"
if _KINEMADAE not in sys.path:
    sys.path.insert(0, _KINEMADAE)

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig, model_fn_wan_video
from diffsynth.diffusion.flow_match import FlowMatchScheduler

# [COPIED from inference_pretrained_abl.py:138-162]
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file as safetensors_load_file

logger = logging.getLogger(__name__)


# ─── Pipeline loading ─────────────────────────────────────────────────

def build_model_paths(ckpt_dir):
    """[COPIED from inference_pretrained_abl.py:165-173]"""
    return [
        [os.path.join(ckpt_dir, f"diffusion_pytorch_model-0000{i}-of-00007.safetensors")
         for i in range(1, 8)],
        os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(ckpt_dir, "Wan2.1_VAE.pth"),
        os.path.join(ckpt_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ]


def load_lora_checkpoint(pipe, checkpoint_path, lora_target_modules, lora_rank, torch_dtype):
    """[COPIED verbatim from inference_pretrained_abl.py:138-162]"""
    target_modules = lora_target_modules.split(",")
    if len(target_modules) == 1:
        target_modules = target_modules[0]
    lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_rank, target_modules=target_modules)
    dit_with_lora = inject_adapter_in_model(lora_config, pipe.dit)
    pipe.dit = dit_with_lora
    state_dict = safetensors_load_file(checkpoint_path)
    new_state_dict = {}
    for key, value in state_dict.items():
        if "lora_A.weight" in key or "lora_B.weight" in key:
            new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    missing, unexpected = pipe.dit.load_state_dict(new_state_dict, strict=False)
    logger.info(f"LoRA loaded: {len(new_state_dict)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    pipe.dit.to(device=pipe.device, dtype=torch_dtype)


def load_pipeline(ckpt_dir, device, lora_checkpoint=None,
                  lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=512):
    """Pipeline 로드 + optional LoRA.

    [ADAPTED from inference_pretrained_abl.py:227-244]
    """
    tokenizer_path = os.path.join(ckpt_dir, "google/umt5-xxl")
    model_paths = build_model_paths(ckpt_dir)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[ModelConfig(path) for path in model_paths],
        tokenizer_config=ModelConfig(tokenizer_path),
    )
    if lora_checkpoint:
        logger.info(f"Loading LoRA from {lora_checkpoint}")
        load_lora_checkpoint(pipe, lora_checkpoint, lora_target_modules, lora_rank, torch.bfloat16)
    else:
        logger.info("No LoRA — pretrained DiT only")

    pipe.dit.eval()
    for p in pipe.dit.parameters():
        p.requires_grad = False
    return pipe


# ─── I2V input preparation ───────────────────────────────────────────

def prepare_y(pipe, input_image, num_frames, height, width):
    """I2V conditioning: mask (4ch) + image VAE latent (16ch) → y (1, 20, T_lat, H/8, W/8).

    [COPIED from wan_video.py:485-508 — WanVideoUnit_ImageEmbedderVAE.process()]
    유일한 차이: end_image 지원 안 함 (probing 에서 불필요).
    """
    device = pipe.device
    dtype = pipe.torch_dtype

    image = pipe.preprocess_image(input_image.resize((width, height))).to(device)
    # [COPIED] 첫 프레임 = image, 나머지 = zero
    vae_input = torch.cat([
        image.transpose(0, 1),
        torch.zeros(3, num_frames - 1, height, width, device=device)
    ], dim=1)

    # [COPIED] Mask: 첫 프레임=1, 나머지=0 + temporal reshape (Wan 1+4k 구조)
    msk = torch.ones(1, num_frames, height // 8, width // 8, device=device)
    msk[:, 1:] = 0
    msk = torch.cat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
    msk = msk.transpose(1, 2)[0]  # (4, T_lat, H/8, W/8)

    # [COPIED] VAE encode
    # [FIX - oliviaa] tile_size=None, tile_stride=None 제거.
    # tiled=True + tile_size=None → VAE 내부에서 tile_size[0] 접근 시 NoneType crash.
    # tile_size 생략하면 VAE default 사용.
    y = pipe.vae.encode(
        [vae_input.to(dtype=dtype, device=device)],
        device=device, tiled=True
    )[0]
    y = y.to(dtype=dtype, device=device)
    y = torch.cat([msk, y])  # (20, T_lat, H/8, W/8)
    y = y.unsqueeze(0)       # (1, 20, T_lat, H/8, W/8)
    return y


def prepare_clip_feature(pipe, input_image, height, width):
    """이미지별 CLIP feature 추출.

    [COPIED from wan_video.py:462-473 — WanVideoUnit_CLIPImageEncoder.process()]
    """
    image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
    clip_context = pipe.image_encoder.encode_image([image])
    clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
    return clip_context


def prepare_null_context(pipe):
    """Null text embedding (empty prompt).

    [ADAPTED from wan_video.py:438-442 — WanVideoUnit_TextEncoder.process()]
    유일한 변경점: 실제 prompt 대신 empty string 사용 (class label inject 방지).
    """
    ids, mask = pipe.tokenizer("", return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    # [COPIED from wan_video.py:441-443] padding masking
    seq_lens = mask.gt(0).sum(dim=1).long()
    context = pipe.text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        context[:, v:] = 0
    return context.to(dtype=pipe.torch_dtype, device=pipe.device)


# ─── Hook-based feature extraction ───────────────────────────────────

class DiTFeatureExtractor:
    """DiT intermediate feature 를 hook 으로 추출.

    [NEW - oliviaa] DDAE model/DDPM.py:get_feature() 패턴을 Wan DiT 에 적응.

    DDAE 원본 (get_feature, DDPM.py:113-143):
        x_noised = perturb(x, t)                       # noise 추가
        _, acts = unet(x_noised, t, ret_activation=True)  # activation 만 사용, output 버림
        features = {name: gap_and_norm(acts[name])}     # spatial avg pool

    우리:
        noisy = scheduler.add_noise(z, noise, t)        # noise 추가 (flow matching)
        _ = dit(noisy, t, context, clip, y)             # output 버림
        features = {key: hook_output.mean(dim=1)}       # token mean pool

    사용법:
        extractor = DiTFeatureExtractor(pipe, block_indices=[0, 5, 10, 20, 30, 39])
        features = extractor.extract(clean_latent, timestep_indices=[0, 25, 49],
                                     y=y, clip_feature=clip)
        # features: dict["s{sigma}/b{block}" → Tensor(B, 5120)]
    """

    def __init__(self, pipe, block_indices, num_inference_steps=50):
        self.pipe = pipe
        self.dit = pipe.dit
        self.block_indices = block_indices
        self.device = pipe.device
        self.dtype = pipe.torch_dtype

        # [COPIED from wan_video.py:39 + flow_match.py:32-41]
        # Inference 와 동일한 scheduler + timestep schedule
        self.scheduler = FlowMatchScheduler(template="Wan")
        self.scheduler.set_timesteps(num_inference_steps)

        # Null text context — 1 회 계산 후 재사용
        self._null_context = None

    def _ensure_null_context(self):
        if self._null_context is None:
            logger.info("Computing null text context (1 time)...")
            with torch.no_grad():
                self._null_context = prepare_null_context(self.pipe)
            logger.info(f"  null_context: {self._null_context.shape}")

    def _register_hooks(self):
        """[NEW - oliviaa] DiTBlock 에 forward hook 등록.

        DDAE model/unet.py:193-213 의 hook 패턴을 DiTBlock 에 적응.
        DDAE: UNet 내부에서 ret_activation=True 로 ResAttBlock output 을 dict 에 수집.
        우리: 외부 hook 으로 DiTBlock output 을 캡처 (DiT 코드 수정 없이).
        """
        self._hook_features = {}
        self._hooks = []
        for idx in self.block_indices:
            def hook_fn(module, input, output, block_idx=idx):
                # output: (B, seq_len, 5120) — DiTBlock output (누적 representation)
                self._hook_features[block_idx] = output.detach().float()
            handle = self.dit.blocks[idx].register_forward_hook(hook_fn)
            self._hooks.append(handle)

    def _remove_hooks(self):
        """[COPIED pattern from DDAE model/unet.py:211-212]"""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._hook_features = {}

    @torch.no_grad()
    def extract(self, clean_latent, timestep_indices, y, clip_feature):
        """Clean latent 에서 여러 timestep 에 대해 intermediate feature 추출.

        [NEW - oliviaa] DDAE get_feature() 의 multi-timestep 버전.
        DDAE 는 ClassifierDict.train() (linear.py:65-66) 에서 timestep 별로
        feat_func(x, time) 을 순회 → 우리도 동일하게 timestep_indices 순회.

        Args:
            clean_latent: (B, 16, T, H, W) — VAE encoded clean latent
            timestep_indices: list[int] — scheduler.timesteps 의 index (0~49)
            y: (B, 20, T_lat, H_lat, W_lat) — I2V conditioning (mask + image latent)
            clip_feature: (B, 1, 1280) — CLIP image feature

        Returns:
            dict["s{sigma:.3f}/b{block}" → Tensor(B, 5120)] — pooled features
        """
        self._ensure_null_context()

        B = clean_latent.shape[0]
        context = self._null_context.expand(B, -1, -1)
        noise = torch.randn_like(clean_latent)

        features = {}
        self._register_hooks()
        try:
            for tidx in timestep_indices:
                timestep = self.scheduler.timesteps[tidx]

                noisy_latent = self.scheduler.add_noise(clean_latent, noise, timestep)

                # [FIX - oliviaa] timestep 을 (1,) 로 넘겨서 Head batch 호환.
                # probing 에서 batch 내 timestep 은 전부 동일하므로 expand 불필요.
                # (1,) → time_embedding → t (1, dim) → Head.modulation (1,2,dim) 과 broadcast OK.
                # DiTBlock 은 t_mod (1,6,dim) + block.modulation (1,6,dim) → broadcast OK.
                # latents 등은 (B,...) 그대로 — patchify 이후 x (B,seq,dim) 으로 block 순회.
                timestep_tensor = torch.tensor(
                    [timestep], device=self.device, dtype=self.dtype
                )

                _ = model_fn_wan_video(
                    dit=self.dit,
                    latents=noisy_latent.to(device=self.device, dtype=self.dtype),
                    timestep=timestep_tensor,
                    context=context.to(device=self.device, dtype=self.dtype),
                    clip_feature=clip_feature.to(device=self.device, dtype=self.dtype),
                    y=y.to(device=self.device, dtype=self.dtype),
                )

                for block_idx in self.block_indices:
                    block_feat = self._hook_features[block_idx]  # (B, seq_len, 5120)
                    pooled = block_feat.mean(dim=1)
                    key = f"t{tidx}/b{block_idx}"
                    # [FIX - oliviaa] 40 blocks × 5 timesteps hook feature GPU 유지 시 OOM.
                    # CPU 로 옮겨서 메모리 절약. probe forward 전 .to(device) 필요.
                    features[key] = pooled.cpu()

                self._hook_features = {}

        finally:
            self._remove_hooks()

        return features
