#!/bin/bash
# LoRA + BN 학습, fresh start, adaptive clamp 1e7 (adaptive 메커니즘 살리기).
# 이전 1e5 는 96.7% step 에서 hit 라 fixed weight 처럼 작동했음.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_GPUS=4
PY=/opt/conda/bin/python

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
$PY -m torch.distributed.run --nproc_per_node=$NUM_GPUS --standalone train_causalvae_geoprior_dit_align.py \
    --exp_name kinemadae_lora_bn_clamp1e7 \
    --pretrained_model_name_or_path checkpoints/Wan2.1_VAE.pth \
    --add_encoder_stages '[{"mode":"downsample3d","num_res_blocks":2,"init":"zero"}]' \
    --add_decoder_before_head_stages '[{"mode":"upsample3d","num_res_blocks":2}]' \
    --z_dim 16 \
    --no_expand_conv2 \
    --unfreeze_decoder \
    --subsample_mode bilinear \
    --video_path panda70m_train.txt \
    --eval_video_path panda70m_eval.txt \
    --num_frames 17 \
    --resolution 256 \
    --batch_size 2 \
    --lr 8e-5 \
    --patchify_lr 1e-4 \
    --epochs 50 \
    --kl_weight 3e-6 \
    --perceptual_weight 3.0 \
    --disc_weight 0.5 \
    --disc_start 9999999 \
    --gan_last_layer decoder_head \
    --save_ckpt_step 500 \
    --eval_steps 500 \
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
    --align_adaptive_weight \
    --adaptive_max_weight 10000000 \
    --grad_accum_steps 2 \
    --patchify_init zero \
    --patchify_mask_init copy4_zero4 \
    --normalize_zprior \
    --freeze_patchify_zprior \
    --caption_metadata data/panda70m_metadata_captioned.jsonl \
    --align_num_blocks 20 \
    --use_2backward_adaptive \
    --log_adaptive_weight \
    --seed 1234 \
    --text_fsdp2 \
    --use_lora \
    --lora_rank 512 \
    --lora_target_modules q,k,v,o,k_img,v_img,ffn.0,ffn.2 \
    --normalize_zmain_bn \
    --bn_momentum 0.1 \
    --zmain_bn_init zprior \
    --no_fused_align \
