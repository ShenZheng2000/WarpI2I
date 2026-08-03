# NOTE: --max_train_steps is doubled (10k->20k) with fewer gpus (8->4)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE: adjust config if needed: /home/shenzhen/.cache/huggingface/accelerate/default_config.yaml
# NOTE: --resolution here only for clean-fid calculation, not for training 
train_pix2pix() {
    DATASET_NAME=$1   # e.g. exp_10_16/candlelight_1; exp_10_16_warped/candlelight_1
    MAX_STEPS=${2:-20000}    # optional arg; default = 20000

    # NOTE: use /ssd0 (or others) instead of /home
    DATASET="/ssd0/shenzhen/Datasets/relighting/${DATASET_NAME}"
    OUTPUT="/ssd0/shenzhen/pix2pix_turbo/${DATASET_NAME}"
    PROJECT="${DATASET_NAME//\//_}"  # for wandb
    accelerate launch src/train_pix2pix_turbo.py \
        --pretrained_model_name_or_path="stabilityai/sd-turbo" \
        --dataset_folder="$DATASET" \
        --output_dir="$OUTPUT" \
        --tracker_project_name="$PROJECT" \
        --resolution=512 \
        --train_batch_size=1 \
        --enable_xformers_memory_efficient_attention \
        --viz_freq 25 \
        --track_val_fid \
        --report_to "wandb" \
        --train_image_prep "no_resize" \
        --test_image_prep "no_resize" \
        --max_train_steps $MAX_STEPS
}

# train_pix2pix "2_24_drive_v2/golden_sunlight_1"
# train_pix2pix "2_24_drive_v2/foggy_1"

# train_pix2pix "2_24_drive_v2_warped_128/golden_sunlight_1"
# train_pix2pix "2_24_drive_v2_warped_128/foggy_1"

# train_pix2pix "2_24_drive_v2_warped_64/golden_sunlight_1"
# train_pix2pix "2_24_drive_v2_warped_64/foggy_1"

# train_pix2pix "exp_1_10_1_exp_1_10_1_v2_merged_warped_128_eyes/moonlight_1" 40000
# train_pix2pix "exp_1_10_1_exp_1_10_1_v2_merged_warped_128_eyes/golden_sunlight_1" 40000
# train_pix2pix "exp_1_10_1_exp_1_10_1_v2_merged_warped_128_eyes/noon_sunlight_1" 40000
# train_pix2pix "exp_1_10_1_exp_1_10_1_v2_merged_warped_128_eyes/foggy_1" 40000


# train_pix2pix "exp_1_10_1/golden_sunlight_1"
# train_pix2pix "exp_1_10_1/noon_sunlight_1"
# train_pix2pix "exp_1_10_1/moonlight_1"
# train_pix2pix "exp_1_10_1/foggy_1"

# train_pix2pix "exp_1_10_1_warped_128_eyes/golden_sunlight_1"
# train_pix2pix "exp_1_10_1_warped_128_eyes/noon_sunlight_1"
# train_pix2pix "exp_1_10_1_warped_128_eyes/moonlight_1"
# train_pix2pix "exp_1_10_1_warped_128_eyes/foggy_1"


# train_pix2pix "exp_1_1/golden_sunlight_1"
# train_pix2pix "exp_1_1/noon_sunlight_1"
# train_pix2pix "exp_1_1/moonlight_1"
# train_pix2pix "exp_1_1/foggy_1"

# train_pix2pix "exp_1_1_warped_128_eyes/golden_sunlight_1"
# train_pix2pix "exp_1_1_warped_128_eyes/noon_sunlight_1"
# train_pix2pix "exp_1_1_warped_128_eyes/moonlight_1"
# train_pix2pix "exp_1_1_warped_128_eyes/foggy_1"