#!/bin/bash
# LoRA + BN 학습, fresh start, adaptive clamp 1e7
# [CHANGED] align_num_blocks 20 → 40 (모든 block 학습)
# [CHANGED] 모든 path 절대경로 (우리 환경 = /NHNHOME/.../jeeyoung/)
# [CHANGED] conda path = /home/kaist_peta/miniconda3/envs/kinemadae
# [CHANGED] 8 GPU (4 → 8)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_GPUS=8
PY=/home/kaist_peta/miniconda3/envs/kinemadae/bin/python

# [NEW] batch_size 환경변수 override (max batch throughput test 용)
BS=${BS:-2}
EXP_SUFFIX=${EXP_SUFFIX:-}

# 절대경로 (우리 환경)
WAN_CKPT=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/checkpoints_persistent/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth
DIT_CKPT_DIR=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/checkpoints_persistent/Wan2.1-I2V-14B-480P
VIDEO_TRAIN=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/panda70m_train.txt
VIDEO_EVAL=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/panda70m_eval.txt
CAPTION_META=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq-to-lora/data/panda70m_metadata_captioned.jsonl

# [FIX] lora-bn-clamp1e7 안 external 없음 → KK 의 external 사용 (KINEMADAE_DIFFSYNTH_PATH + KINEMADAE_PROBING_PATH env var)
export KINEMADAE_DIFFSYNTH_PATH=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq/external/DiffSynth-Studio
export KINEMADAE_PROBING_PATH=/NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/KinemaDAE-kk4aiq/external/dit_probing

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
$PY -m torch.distributed.run --nproc_per_node=$NUM_GPUS --standalone train_causalvae_geoprior_dit_align.py \
    --exp_name "kinemadae_lora_bn_clamp1e7_align40_bs${BS}${EXP_SUFFIX}" \
    --pretrained_model_name_or_path "$WAN_CKPT" \
    --add_encoder_stages '[{"mode":"downsample3d","num_res_blocks":2,"init":"zero"}]' \
    --add_decoder_before_head_stages '[{"mode":"upsample3d","num_res_blocks":2}]' \
    --z_dim 16 \
    --no_expand_conv2 \
    --unfreeze_decoder \
    --subsample_mode bilinear \
    --video_path "$VIDEO_TRAIN" \
    --eval_video_path "$VIDEO_EVAL" \
    --num_frames 17 \
    --resolution 256 \
    --batch_size $BS \
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
    --dit_ckpt_dir "$DIT_CKPT_DIR" \
    --align_weight 1.0 \
    --align_loss_type cosine \
    --align_layers all \
    --align_agg sum \
    --dit_timestep_mode random \
    --align_adaptive_weight \
    --adaptive_max_weight 10000000 \
    --grad_accum_steps ${GRAD_ACCUM:-2} \
    --patchify_init zero \
    --patchify_mask_init copy4_zero4 \
    --normalize_zprior \
    --freeze_patchify_zprior \
    --caption_metadata "$CAPTION_META" \
    --align_num_blocks 40 \
    --resume_from_checkpoint /NHNHOME/WORKSPACE/0226010404_A/CVLAB/CVLAB2/jeeyoung/Kinemadae-lora-bn-clamp1e7/results/kinemadae_lora_bn_clamp1e7_align40_bs8-lr8.00e-05-bs8-rs256-sr2-fr17/checkpoint-1000.ckpt \
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
