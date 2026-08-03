train_cyclegan() {
    DATASET_NAME="$1"
    TRAIN_PREP="$2"
    RESTRICTED="${3:-0}"   # default 0

    ACC_CONFIG=/home/shenzhen/.cache/huggingface/accelerate/default_config8gpus.yaml
    DATASET="/ssd0/shenzhen/Datasets/driving/${DATASET_NAME}"
    # OUTPUT="output/cyclegan_turbo/${DATASET_NAME}_${TRAIN_PREP}" # TODO: move ckpt to output/ or data3 later! 
    OUTPUT="/ssd0/shenzhen/cyclegan_turbo/${DATASET_NAME}_${TRAIN_PREP}"
    PROJECT="${DATASET_NAME//\//_}_${TRAIN_PREP}"  # for wandb tracking
    
    EXTRA_ARGS=""
    if [ "$RESTRICTED" = "1" ]; then
        EXTRA_ARGS="--restricted_pairs"
    fi

    # NOTE: --max_train_steps assumes 8 GPUs!!!!!
    # NOTE: adjust config: /home/shenzhen/.cache/huggingface/accelerate/default_config.yaml
    # NOTE: use validation_steps > max_train_steps to save time! 
    accelerate launch \
        --config_file "$ACC_CONFIG" \
        src/train_cyclegan_turbo.py \
        --pretrained_model_name_or_path="stabilityai/sd-turbo" \
        --dataset_folder "$DATASET" \
        --output_dir="$OUTPUT" \
        --tracker_project_name="$PROJECT" \
        --learning_rate="1e-5" \
        --train_batch_size=1 \
        --enable_xformers_memory_efficient_attention \
        --report_to "wandb" \
        --train_img_prep "$TRAIN_PREP" \
        --val_img_prep "no_resize" \
        --validation_steps 50000 \
        --max_train_steps 25000 \
        $EXTRA_ARGS
}


# train_cyclegan "BDD100K_day2night_warped_128" "resize_286_randomcrop_256x256_hflip"
# train_cyclegan "BDD100K_clear2rainy_warped_128" "resize_286_randomcrop_256x256_hflip"
# train_cyclegan "cityscapes_to_dark_zurich" "resize_286_randomcrop_256x256_hflip"
# train_cyclegan "cityscapes_to_dark_zurich_warped_128" "resize_286_randomcrop_256x256_hflip"
# train_cyclegan "cityscapes_to_acdc_fog" "resize_286_randomcrop_256x256_hflip"
# train_cyclegan "cityscapes_to_acdc_fog_warped_128" "resize_286_randomcrop_256x256_hflip"