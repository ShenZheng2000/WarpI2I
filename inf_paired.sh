run_infer() {
    exp_name="$1"
    relight_type="$2"
    data_name="$3"
    gpu_id="${4:-0}"
    ckpt_exp="${5:-$exp_name}"   # override ckpt dir, default to exp_name

    input_root=/data3/shenzhen/Datasets
    output_root=output/pix2pix_turbo

    case "$relight_type" in
      noon_sunlight_1)
        prompt="Relit with bright noon sunlight in a clear outdoor setting, casting soft natural shadows and surrounding the subject in crisp white light to create a clean, vibrant daytime mood."
        ;;
      golden_sunlight_1)
        prompt="Relit with warm golden sunlight during the late afternoon, casting gentle directional shadows and surrounding the subject in soft amber tones to create a calm, radiant mood."
        ;;
      foggy_1)
        prompt="Relit with dense fog in a muted outdoor setting, casting soft diffused shadows and surrounding the subject in pale gray light to create a quiet, atmospheric mood."
        ;;
      moonlight_1)
        prompt="Relit with cold moonlight in a minimalist nighttime scene, casting crisp soft shadows and bathing the subject in icy blue highlights to create a tranquil, distant mood."
        ;;
      *)
        echo "Unknown relight_type: $relight_type"
        return 1
        ;;
    esac

    ckpt_dir=${output_root}/${ckpt_exp}/${relight_type}/checkpoints
    model_path=$(ls -1 ${ckpt_dir}/model_*.pkl | sort -V | tail -n 1)

    CUDA_VISIBLE_DEVICES=$gpu_id python src/inference_paired_folder.py \
        --exp_config configs/${exp_name}.yaml \
        --input_dir ${input_root}/${data_name} \
        --output_dir ${output_root}/${exp_name}/${relight_type}/${data_name} \
        --prompt "$prompt" \
        --model_path "$model_path"
}



# Driving Relighting
# run_infer   2_24_drive_v2_warped_128                          golden_sunlight_1   relighting/workzone_segm/boston
# run_infer   2_24_drive_v2_warped_128                          foggy_1             relighting/workzone_segm/boston


# Human Relighting
# run_infer   exp_1_10_1_warped_128_eyes                        golden_sunlight_1   VITON/test
# run_infer   exp_1_10_1_warped_128_eyes                        foggy_1             VITON/test
# run_infer   exp_1_10_1_warped_128_eyes                        moonlight_1         VITON/test
# run_infer   exp_1_10_1_exp_1_10_1_v2_merged_warped_128_eyes   noon_sunlight_1     VITON/test