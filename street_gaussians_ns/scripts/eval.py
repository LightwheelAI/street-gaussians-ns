"""
eval.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from time import time

import tyro
import torch
import torchvision.utils as vutils
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import profiler

from street_gaussians_ns.sgn_splatfacto import SplatfactoModelConfig
from street_gaussians_ns.data.sgn_datamanager import FullImageDatamanager

@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Optional[Path] = None  # Path("output.json")
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = None
    # Path to aggregate lidar.
    aggregate_lidar_path: Optional[Path] = Path("aggregate_lidar/output.ply")
    load_sky_sphere_image_file: Optional[str] = None # "ldr_sky_sphere_inpainted_0.png"
    """Load sky sphere image"""

    def _update_config_callback(self, config: TrainerConfig) -> TrainerConfig:
        if self.load_sky_sphere_image_file and isinstance(config.pipeline.model, SplatfactoModelConfig):
            config.pipeline.model.load_sky_sphere_image_path = config.get_base_dir() / self.load_sky_sphere_image_file
        return config

    def main(self) -> None:
        """Main function."""
        config, self.pipeline, checkpoint_path, _ = eval_setup(self.load_config, update_config_callback=self._update_config_callback)
        if self.output_path is None:
            self.output_path = self.load_config.parent / "eval_output.json"
        assert self.output_path.suffix == ".json"
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True, exist_ok=True)
        metrics_dict = self.get_average_eval_image_metrics(output_path=self.render_output_path, get_std=True)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}\n{benchmark_info}")

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.pipeline.eval()
        metrics_dict_list = []
        assert isinstance(self.pipeline.datamanager, FullImageDatamanager)
        num_images = len(self.pipeline.datamanager.fixed_indices_eval_dataloader)
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            idx = 0
            for camera, batch in self.pipeline.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                outputs = self.pipeline.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, image_dict = self.pipeline.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    for key in image_dict.keys():
                        image = image_dict[key]  # [H, W, C] order
                        vutils.save_image(image.permute(2, 0, 1).cpu(), output_path / f"eval_{key}_{idx:04d}.png")

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                idx = idx + 1
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        self.pipeline.train()
        return metrics_dict


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa