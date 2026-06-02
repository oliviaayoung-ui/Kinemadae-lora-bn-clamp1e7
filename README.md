# KinemaDAE — LoRA + BN (Stage 1, clamp1e7)

KinemaDAE stage 1 alignment 학습. Wan2.1 I2V-14B DiT 의 feature 와 VAE encoder 의 z_main alignment 학습.

메인 학습 세팅: **clamp 1e7 = j6s8qkos** (`launch_train_lora_bn_clamp1e7.sh`).

## 핵심 변경 사항 (vs 이전 push)

- **LoRA fp32 cast 제거 (가장 중요)** — LoRA params 를 bf16 그대로 유지.
  fp32 cast 했던 게 `grad/lora = 0`, `lora_param = 0` 의 원인이었음. 제거 후 LoRA 정상 작동.
- **`_NoDDPWrapper.__getattr__` fallback** — VAE 전체 freeze 시 `_NoDDPWrapper`
  가 underlying module 의 모든 메서드 (named_buffers, no_sync 등) 를 자동
  위임. 필요할 때마다 wrapper 에 메서드 추가하지 않아도 됨.
- **`--freeze_patchify_full` 옵션** — patchify 전체 freeze (LoRA-only isolation
  실험). exp3 lora pure 에서 사용.
- **LoRA grad norm wandb 로깅** (`grad/lora`) — LoRA 가 실제로 학습되는지
  step 단위 모니터.
- **추가 launch shell**:
  - `launch_train_lora_bn_clamp1e7.sh` — 메인 (j6s8qkos)
  - `launch_train_lora_bn_clamp1e8.sh` — adaptive weight clamp 1e8
  - `launch_train_lora_bn_fresh.sh` — fresh start
  - `launch_train_exp2_lora_only.sh` — LoRA-only ablation
  - `launch_train_exp3_lora_pure.sh` / `_1gpu.sh` — LoRA pure (patchify
    전체 freeze)

## Setup (다른 서버에서 처음 사용 시)

### 1. 의존성

```bash
# Python 3.11 권장
conda create -n kinemadae python=3.11 -y
conda activate kinemadae

# PyTorch (CUDA 12.x 기준, 환경에 맞게 조정)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 나머지 pip 의존
pip install -r requirements.txt
```

### 2. DiffSynth-Studio 설치 (외부 의존)

```bash
mkdir -p external
cd external
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
# (특정 commit 필요 시 git checkout <hash>)
cd ../..
```

또는 환경변수 override:
```bash
export KINEMADAE_DIFFSYNTH_PATH=/path/to/DiffSynth-Studio
export KINEMADAE_PROBING_PATH=/path/to/this_repo/external/dit_probing
```

### 3. Pretrained checkpoints 다운로드

```bash
mkdir -p checkpoints
cd checkpoints
# Wan2.1 I2V-14B-480P 모델 (HuggingFace: Wan-AI/Wan2.1-I2V-14B-480P)
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir Wan2.1-I2V-14B-480P
# VAE checkpoint 도 같은 dir 안에 있음 (Wan2.1_VAE.pth)
ln -s Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth Wan2.1_VAE.pth
cd ..
```

### 4. Panda-70M 영상 데이터

```bash
# panda70m_train.txt / panda70m_eval.txt 직접 준비 (영상 절대 경로 list)
# (repo 에 포함 안 됨 — 357MB 라 .gitignore 처리)
# eval_subset_size=100 (default) 라 panda70m_eval.txt 의 첫 100 영상이 평가에 사용됨
```

`data/panda70m_metadata_captioned.jsonl` — caption metadata 파일도 별도 준비.

## 학습 실행

```bash
# 메인 (clamp 1e7, j6s8qkos)
bash launch_train_lora_bn_clamp1e7.sh

# 또는 base launch
bash launch_train.sh
```

기본 setup:
- batch_size 2, grad_accum 2 (effective 16, 4 GPU)
- LoRA (rank 512) on DiT q,k,v,o,k_img,v_img,ffn.0,ffn.2 — **bf16 유지 (fp32 cast 안 함)**
- z_main BatchNorm3d 정규화 (REPA-E style, SyncBN, EMA running stats)
- z_prior 정규화 + freeze_patchify_zprior
- VAE GC (encoder/decoder ResidualBlock checkpoint)
- align_num_blocks 20 (DiT 의 처음 20 block)
- bf16 mixed precision
- EMA model (param + buffer 둘 다 EMA)
- alignment loss: cosine + adaptive weight (clamp 1e7)

## 주요 옵션

| flag | 의미 |
|------|------|
| `--normalize_zmain_bn` | z_main 을 BatchNorm3d 로 online 정규화 (REPA-E style) |
| `--bn_momentum 0.1` | BN running stats EMA momentum |
| `--zmain_bn_init zprior` | BN running_mean/var 를 z_prior pre_stats 로 init |
| `--use_lora --lora_rank 512` | DiT 의 LoRA fine-tune |
| `--text_fsdp2` | T5/CLIP FSDP2 샤딩 (메모리 절감 ~8GB/GPU) |
| `--use_2backward_adaptive` | Adaptive alignment weight (2-backward 방식) |
| `--no_fused_align` | fused align 비활성, origin compute_alignment_loss 사용 |
| `--align_num_blocks 20` | DiT 의 처음 N block 만 alignment |
| `--adaptive_max_weight 1e7` | Adaptive weight clamp 상한 (메인) |
| `--normalize_zprior` | z_prior 사전 stats 정규화 |
| `--freeze_patchify_zprior` | student patchify 의 z_prior weight 고정 |
| `--freeze_patchify_full` | patchify 전체 freeze (LoRA-only isolation) |

## 메모리 (4×B200 183GB 기준)

- batch 2 + accum 2 + LoRA + text_fsdp2 + VAE GC: peak ~65 GB / GPU
- BN 추가 부하: ~1 MB (무시)

## Stage 2 호환

- BN 의 running_mean/var 는 ckpt 의 state_dict 에 자동 저장
- Stage 2 학습 시 같은 `--normalize_zmain_bn` flag + ckpt load 로 정규화 일관 유지
- z_main / z_prior 가 같은 scale (~zero mean, ~unit var) → stage 2 diffusion loss balance
