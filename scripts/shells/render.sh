config_path=$1
cuda_id=$2

CUDA_VISIBLE_DEVICES=$cuda_id sgn-render dataset \
    --load-config $config_path \