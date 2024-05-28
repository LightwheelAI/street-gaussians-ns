root=$1

sh scripts/shells/segs_generate.sh $root

sh scripts/shells/masks_generate.sh $root

sh scripts/shells/run_colmap.sh $root

sh scripts/shells/points_cloud_generate.sh $root

sh scripts/shells/object_pts_generate.sh $root
