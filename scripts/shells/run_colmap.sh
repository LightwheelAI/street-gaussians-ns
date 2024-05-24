DATASET_PATH=$1

python scripts/pythons/transform2colmap.py \
    --input_path $DATASET_PATH \

mkdir $DATASET_PATH/colmap

colmap feature_extractor \
   --database_path $DATASET_PATH/colmap/database.db \
   --image_path $DATASET_PATH/images \
   --ImageReader.mask_path $DATASET_PATH/masks \

colmap exhaustive_matcher \
    --database_path $DATASET_PATH/colmap/database.db

mkdir $DATASET_PATH/colmap/sparse
mkdir $DATASET_PATH/colmap/sparse/not_align

colmap mapper \
    --database_path $DATASET_PATH/colmap/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/colmap/sparse/not_align \
    --Mapper.init_max_forward_motion 1.0 \
    --Mapper.init_min_tri_angle 0.5 \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_extra_params 0 \
    --Mapper.multiple_models 1 \
    --Mapper.ba_global_max_num_iterations 30 \
    --Mapper.ba_global_images_ratio 1.3 \
    --Mapper.ba_global_points_ratio 1.3 \
    --Mapper.ba_global_images_freq 2000 \
    --Mapper.ba_global_points_freq 35000 \
    --Mapper.filter_min_tri_angle 0.1 \

colmap model_comparer \
    --input_path1 $DATASET_PATH/colmap/sparse/not_align/0 \
    --input_path2 $DATASET_PATH/colmap/sparse/origin \
    --output_path $DATASET_PATH/colmap/sparse/0 \
    --alignment_error proj_center
    
# rm -rf $DATASET_PATH/colmap/sparse/not_align

colmap point_triangulator \
    --database_path $DATASET_PATH/colmap/database.db \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/colmap/sparse/origin \
    --output_path $DATASET_PATH/colmap/sparse/origin \