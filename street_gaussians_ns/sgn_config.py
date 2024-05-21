"""
Street Gaussians configuration file.
"""
from pathlib import Path

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from street_gaussians_ns.data.sgn_datamanager import FullImageDatamanagerConfig
from street_gaussians_ns.data.sgn_dataparser import ColmapDataParserConfig
from street_gaussians_ns.data.utils.bbox_optimizers import BBoxOptimizerConfig
from street_gaussians_ns.sgn_splatfacto import SplatfactoModelConfig
from street_gaussians_ns.sgn_splatfacto_scene_graph import SplatfactoSceneGraphModelConfig


street_gaussians_ns_method = MethodSpecification(
    config=TrainerConfig(
        method_name="street-gaussians-ns",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=30000, 
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100,'semantic':10},
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=ColmapDataParserConfig(
                    load_3D_points=True,
                    max_2D_matches_per_3D_point=0,
                    undistort=True,
                    colmap_path=Path("colmap/sparse/0"),
                    segments_path=Path("segs"),
                    load_dynamic_annotations=True,
                ),
            ),
            model=SplatfactoSceneGraphModelConfig(
                # TODO simplify this, warper model use background_model directly
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                bbox_optimizer=BBoxOptimizerConfig(mode="simple"),
                use_sky_sphere=True,
                sh_degree=3,
                background_model=SplatfactoModelConfig(
                    cull_alpha_thresh=0.02,
                    cull_scale_thresh=0.2,
                    # densify_grad_thresh=0.0002,
                    warmup_length=500,
                    refine_every=100,
                    reset_alpha_every=30,
                    stop_split_at=25000,
                    fourier_features_dim=1,
                ),
                object_model_template=SplatfactoModelConfig(
                    cull_alpha_thresh=0.005,
                    cull_scale_thresh=0.2,
                    densify_grad_thresh=0.0002,
                    warmup_length=500,
                    refine_every=100,
                    reset_alpha_every=30,
                    stop_split_at=25000,
                    fourier_features_dim=5,
                    num_random=10000,
                )
            ),
        ),
        optimizers={
            "sky_sphere": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=70000),
            },
            "bbox_opt":{
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=70000),
            },
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=70000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer_legacy+tensorboard",
    ),
    description="Base config for Street Gaussians",
)