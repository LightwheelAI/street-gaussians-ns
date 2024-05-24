root=$1

python dependencies/Mask2Former/segs_generate.py \
    --root_path $root \
    --config-file dependencies/Mask2Former/configs/mapillary-vistas/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml \
    --opts MODEL.WEIGHTS dependencies/Mask2Former/models/model_final_90ee2d.pkl