data_root=$1
cuda_id=$2
current_time=$(date +"%Y-%m-%d-%T")

mkdir -p output/

CUDA_VISIBLE_DEVICES=$cuda_id  sgn-train splatfacto-scene-graph \
    --experiment_name street-gaussians-ns \
    --output_dir output/$current_time \
    --vis viewer+wandb \
    --viewer.quit_on_train_completion True \
    colmap-data-parser-config \
    --data $data_root \
    --colmap_path colmap/sparse/0 \
    --load_3D_points True \
    --max_2D_matches_per_3D_point 0 \
    --undistort True \
    --segments-path segs \
    --filter_camera_id 1 \
