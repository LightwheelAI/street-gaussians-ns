"""
render.py
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import mediapy as media
import numpy as np
import torch
import tyro
import viser.transforms as tf
from rich import box, style
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from typing_extensions import Annotated

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.scripts.render import BaseRender

from street_gaussians_ns.data.sgn_datamanager import FullImageDatamanagerConfig
from street_gaussians_ns.data.sgn_dataset import Dataset


@contextmanager
def _disable_datamanager_setup(cls):
    """
    Disables setup_train or setup_eval for faster initialization.
    """
    old_setup_train = getattr(cls, "setup_train")
    old_setup_eval = getattr(cls, "setup_eval")
    old_cache_images = getattr(cls, "cache_images", None)
    setattr(cls, "setup_train", lambda *args, **kwargs: None)
    setattr(cls, "setup_eval", lambda *args, **kwargs: None)
    setattr(cls, "cache_images", lambda *args, **kwargs: (None,None))
    yield cls
    setattr(cls, "setup_train", old_setup_train)
    setattr(cls, "setup_eval", old_setup_eval)
    setattr(cls, "cache_images", old_cache_images)


@dataclass
class DatasetRender(BaseRender):
    """Render all images in the dataset."""

    output_path: Optional[Path] = None  # Path("renders")
    """Path to output video file."""
    data: Optional[Path] = None
    """Override path to the dataset."""
    downscale_factor: Optional[float] = None
    """Scaling factor to apply to the camera image resolution."""
    split: Literal["train", "val", "test", "train+test", "all"] = "all"
    """Split to render."""
    rendered_output_names: Optional[List[str]] = field(default_factory=lambda: None)
    """Name of the renderer outputs to use. rgb, depth, raw-depth, gt-rgb etc. By default all outputs are rendered."""
    output_format: Literal["images", "video", "images+video"] = "video"
    """How to save output data."""
    vehicle_config: Optional[Path] = None
    """Camera pose transform config on the new vehicle."""
    depth_near_plane: Optional[float] = 0.
    """Closest depth to consider when using the colormap for depth. If None, use min value."""
    depth_far_plane: Optional[float] = 3.
    """Furthest depth to consider when using the colormap for depth. If None, use max value."""
    load_sky_sphere_image_file: Optional[str] = None # "ldr_sky_sphere_inpainted_0.png"
    """Load sky sphere image"""
    export_sky_sphere_mask: bool = False
    """Export sky sphere sphere mask image, only support splatfacto model"""

    def __post_init__(self):
        if self.output_path is None:
            self.output_path = self.load_config.parent/"renders"

    def main(self):
        config: TrainerConfig

        import torch
        if torch.__version__.split("+")[0] >= "2.1.0":
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True

        def update_config(config: TrainerConfig) -> TrainerConfig:
            data_manager_config = config.pipeline.datamanager
            assert isinstance(data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig))
            data_manager_config.eval_num_images_to_sample_from = -1
            data_manager_config.eval_num_times_to_repeat_images = -1
            if isinstance(data_manager_config, VanillaDataManagerConfig):
                data_manager_config.train_num_images_to_sample_from = -1
                data_manager_config.train_num_times_to_repeat_images = -1
            if self.data is not None:
                data_manager_config.data = self.data
            if self.downscale_factor is not None:
                assert hasattr(data_manager_config.dataparser, "downscale_factor")
                setattr(data_manager_config.dataparser, "downscale_factor", self.downscale_factor)
            return config

        config, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=update_config,
        )
        data_manager_config = config.pipeline.datamanager
        assert isinstance(data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig))

        for split in self.split.split("+"):
            datamanager: VanillaDataManager
            dataset: Dataset
            if split == "train":
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode="test", device=pipeline.device)

                dataset = datamanager.train_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", datamanager.train_dataparser_outputs)
            else:
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode=split, device=pipeline.device)

                dataset = datamanager.eval_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
                if dataparser_outputs is None:
                    dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(split=datamanager.test_split)
            image_names_idx = [id for id, _ in sorted(enumerate(dataparser_outputs.image_filenames), key=lambda x: x[1],reverse=False)]
            if self.vehicle_config is not None:
                self._transform_cameras_to_new_vehicle(dataset, dataparser_outputs)
            
            dataloader = FixedIndicesEvalDataloader(
                input_dataset=dataset,
                image_indices=image_names_idx,
                device=datamanager.device,
                num_workers=datamanager.world_size * 4,
            )
            images_root = Path(os.path.commonpath(dataparser_outputs.image_filenames))
            with Progress(
                TextColumn(f":movie_camera: Rendering split {split} :movie_camera:"),
                BarColumn(),
                TaskProgressColumn(
                    text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                    show_speed=True,
                ),
                ItersPerSecColumn(suffix="fps"),
                TimeRemainingColumn(elapsed_when_finished=False, compact=False),
                TimeElapsedColumn(),
            ) as progress, ExitStack() as stack:
                writer = {}
                for camera_idx, (camera, batch) in enumerate(progress.track(dataloader, total=len(dataset))):
                    camera_idx = batch["image_idx"]
                    with torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera(camera)

                    gt_batch = batch.copy()
                    gt_batch["rgb"] = gt_batch.pop("image")
                    all_outputs = (
                        list(outputs.keys())
                        + [f"raw-{x}" for x in outputs.keys()]
                        + [f"gt-{x}" for x in gt_batch.keys()]
                        + [f"raw-gt-{x}" for x in gt_batch.keys()]
                    )
                    rendered_output_names = self.rendered_output_names
                    if rendered_output_names is None:
                        rendered_output_names = ["gt-rgb"] + list(outputs.keys())
                    for rendered_output_name in rendered_output_names:
                        if rendered_output_name not in all_outputs:
                            CONSOLE.rule("Error", style="red")
                            CONSOLE.print(
                                f"Could not find {rendered_output_name} in the model outputs", justify="center"
                            )
                            CONSOLE.print(
                                f"Please set --rendered-output-name to one of: {all_outputs}", justify="center"
                            )
                            sys.exit(1)

                        is_raw = False
                        is_depth = rendered_output_name.find("depth") != -1
                        is_semantic = rendered_output_name.find("semantic") != -1
                        image_name = f"{camera_idx:05d}"

                        # Try to get the original filename
                        image_name = dataparser_outputs.image_filenames[camera_idx].relative_to(images_root)

                        output_path = self.output_path / split / rendered_output_name / image_name
                        output_path.parent.mkdir(exist_ok=True, parents=True)

                        output_name = rendered_output_name
                        if output_name.startswith("raw-"):
                            output_name = output_name[4:]
                            is_raw = True
                            if output_name.startswith("gt-"):
                                output_name = output_name[3:]
                                output_image = gt_batch[output_name]
                            else:
                                output_image = outputs[output_name]
                                if is_depth:
                                    # Divide by the dataparser scale factor
                                    output_image.div_(dataparser_outputs.dataparser_scale)
                        else:
                            if output_name.startswith("gt-"):
                                output_name = output_name[3:]
                                output_image = gt_batch[output_name]
                            else:
                                output_image = outputs[output_name]
                        del output_name

                        # Map to color spaces / numpy
                        if is_raw:
                            output_image = output_image.cpu().numpy()
                        elif is_depth:
                            output_image = (
                                colormaps.apply_depth_colormap(
                                    output_image,
                                    near_plane=self.depth_near_plane,
                                    far_plane=self.depth_far_plane,
                                    colormap_options=self.colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )
                        elif is_semantic:
                            if output_name.startswith("gt-"):
                                output_image = (output_image.squeeze().cpu().numpy() * 100).astype(np.uint8)
                            else:
                                # Output image is logits
                                output_image = (output_image.argmax(dim=-1).cpu().numpy() * 100).astype(np.uint8)
                        else:
                            output_image = (
                                colormaps.apply_colormap(
                                    image=output_image,
                                    colormap_options=self.colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )

                        # Save to file
                        if "video" in self.output_format.split("+"):
                            output_filename = str(output_path.parent.with_suffix(".mp4"))
                            # output_image = output_image[:299,:539]
                            if output_filename not in writer:
                                render_width = int(output_image.shape[1])
                                render_height = int(output_image.shape[0])
                                writer[output_filename] = stack.enter_context(
                                    media.VideoWriter(
                                        path=output_filename,
                                        shape=(render_height, render_width),
                                        fps=10,
                                    )
                                )
                            writer[output_filename].add_image(output_image)
                        if "images" in self.output_format.split("+"):
                            if is_raw:
                                with gzip.open(output_path.with_suffix(".npy.gz"), "wb") as f:
                                    np.save(f, output_image)
                            elif self.image_format == "png":
                                media.write_image(output_path.with_suffix(".png"), output_image, fmt="png")
                            elif self.image_format == "jpeg":
                                media.write_image(
                                    output_path.with_suffix(".jpg"), output_image, fmt="jpeg", quality=self.jpeg_quality
                                )
                            else:
                                raise ValueError(f"Unknown image format {self.image_format}")

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        for split in self.split.split("+"):
            table.add_row(f"Outputs {split}", str(self.output_path / split))
        CONSOLE.print(Panel(table, title=f"[bold][green]:tada: Render on split {split} Complete :tada:[/bold]", expand=False))


    def _transform_cameras_to_new_vehicle(self, dataset, dataparser_outputs):
        dataparser_scale = dataparser_outputs.dataparser_scale
        new_vehicle_sensors = json.load(self.vehicle_config.open())

        cameras = dataset.cameras
        for camera_config in new_vehicle_sensors:
            image_path_patten = camera_config['image_path_patten']
            ca2cb = torch.tensor(camera_config['transform'], dtype=dataset.cameras.camera_to_worlds.dtype,
                                 device=dataset.cameras.camera_to_worlds.device)
            ca2cb[:3, 3] *= dataparser_scale

            p = re.compile(image_path_patten)
            for i, image_path in enumerate(dataparser_outputs.image_filenames):
                if not p.match(image_path.as_posix()):
                    continue

                ca2w = cameras.camera_to_worlds[i]
                row = torch.tensor([0,0,0,1], dtype=ca2w.dtype, device=ca2w.device).reshape((1,4))
                ca2w = torch.cat([ca2w, row])
                cb2w = torch.linalg.inv(ca2cb @ torch.linalg.inv(ca2w))
                cameras.camera_to_worlds[i] = cb2w[:3]

        dataset.cameras = cameras
        dataparser_outputs.cameras = cameras


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[DatasetRender, tyro.conf.subcommand(name="dataset")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa