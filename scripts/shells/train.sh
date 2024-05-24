data_root=$1
cuda_id=$2

mkdir -p output/

CUDA_VISIBLE_DEVICES=$cuda_id  sgn-train street-gaussians-ns \
    --experiment_name street-gaussians-ns \
    --output_dir output/ \
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
    --init_points_filename points3D_withlidar.txt
