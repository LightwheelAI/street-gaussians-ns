root=$1

mkdir $root/colmap/sparse/lidar

python scripts/pythons/pcd2colmap_points3D.py \
    --root_path $root \
    --output_path $root/colmap/sparse/lidar/points3D.bin \
    --main_lidar_in_transforms lidar_FRONT \

python scripts/pythons/colmap_pts_combine.py \
    --src1 $root/colmap/sparse/lidar/points3D.bin \
    --src2 $root/colmap/sparse/0/points3D.bin \
    --dst $root/colmap/sparse/0/points3D_withlidar.bin