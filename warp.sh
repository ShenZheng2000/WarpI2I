warp_relight() {
    local exp_name="$1"     # must match yaml name (without .yaml)
    local relight_type="$2"
    local gpu="${3:-1}"

    local input_root
    case "$exp_name" in
        exp_*|2_24_*)
            input_root="/home/shenzhen/Datasets/relighting"
            ;;
        cityscapes_*|BDD100K_*)
            input_root="/ssd0/shenzhen/Datasets/driving"
            ;;
        *)
            echo "Cannot infer input_root from exp_name: $exp_name"
            return 1
            ;;
    esac

    CUDA_VISIBLE_DEVICES=$gpu python warp_dataset.py \
        --exp_config "configs/${exp_name}.yaml" \
        --input_root "$input_root" \
        --target_prefix "${exp_name%%_warped*}" \
        --relight_type "$relight_type"
}


# Driving Relighting
# warp_relight  2_24_drive_v2_warped_128 golden_sunlight_1
# warp_relight  2_24_drive_v2_warped_128 foggy_1


# Human Relighting
# warp_relight  exp_1_10_1_warped_128_eyes                       golden_sunlight_1
# warp_relight  exp_1_10_1_warped_128_eyes                       foggy_1
# warp_relight  exp_1_10_1_warped_128_eyes                       moonlight_1
# warp_relight  exp_1_10_1_exp_1_10_1_v2_merged_warped_128_eyes  noon_sunlight_1


# Driving I2I
# warp_relight BDD100K_day2night_warped_128
# warp_relight BDD100K_clear2rainy_warped_128
# warp_relight cityscapes_to_acdc_fog_warped_128
# warp_relight cityscapes_to_dark_zurich_warped_128