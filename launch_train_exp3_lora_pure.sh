#!/bin/bash
# Experiment 3: Pure LoRA-only (patchify 도 freeze).
# VAE 전체 freeze + recon/perceptual/kl weight=0 + patchify freeze → 진짜 LoRA 만 학습.
# 목표: align loss 가 떨어지면 → LoRA gradient pipeline 정상 + 효과 확정.
#       안 떨어지면 → 이전 실험 2 의 align loss 감소는 patchify 가 driver.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_GPUS=4
PY=/opt/conda/bin/python

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
$PY -m torch.distributed.run --nproc_per_node=$NUM_GPUS --standalone train_causalvae_geoprior_dit_align.py \
    --exp_name kinemadae_exp3_lora_pure \
    --pretrained_model_name_or_path checkpoints/Wan2.1_VAE.pth \
    --add_encoder_stages '[{"mode":"downsample3d","num_res_blocks":2,"init":"zero"}]' \
    --add_decoder_before_head_stages '[{"mode":"upsample3d","num_res_blocks":2}]' \
    --z_dim 16 \
    --no_expand_conv2 \
    --freeze_pretrained \
    --freeze_encoder \
    --freeze_decoder \
    --subsample_mode bilinear \
    --video_path panda70m_train.txt \
    --eval_video_path panda70m_eval.txt \
    --num_frames 17 \
    --resolution 256 \
    --batch_size 2 \
    --lr 8e-5 \
    --patchify_lr 1e-4 \
    --epochs 50 \
    --kl_weight 0 \
    --perceptual_weight 0 \
    --disc_weight 0 \
    --disc_start 9999999 \
    --gan_last_layer decoder_head \
    --save_ckpt_step 100 \
    --eval_steps 50 \
    --log_steps 1 \
    --mix_precision bf16 \
    --ema \
    --ema_decay 0.999 \
    --eval_lpips \
    --find_unused_parameters \
    --dit_ckpt_dir checkpoints/Wan2.1-I2V-14B-480P \
    --align_weight 1.0 \
    --align_loss_type cosine \
    --align_layers all \
    --align_agg sum \
    --dit_timestep_mode random \
    --grad_accum_steps 2 \
    --patchify_init zero \
    --patchify_mask_init copy4_zero4 \
    --normalize_zprior \
    --freeze_patchify_zprior \
    --freeze_patchify_full \
    --caption_metadata data/panda70m_metadata_captioned.jsonl \
    --align_num_blocks 20 \
    --seed 1234 \
    --text_fsdp2 \
    --use_lora \
    --lora_rank 512 \
    --lora_target_modules q,k,v,o,k_img,v_img,ffn.0,ffn.2 \
    --no_fused_align \
