root=$1

python scripts/pythons/pcd2colmap_points3D.py \
    --root_path $root \
    --output_path $root/colmap/lidar/points3D.txt \
    --main_lidar_in_transforms lidar_FRONT \

python scripts/pythons/colmap_pts_combine.py \
    --src1 $root/colmap/lidar/points3D.txt \
    --src2 $root/colmap/sparse/origin/points3D.txt \
    --dst $root/colmap/points3D.txt