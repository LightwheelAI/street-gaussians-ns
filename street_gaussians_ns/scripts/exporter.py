"""
Script for exporting NeRF into other formats.
"""


from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import open3d as o3d
import torch
import tyro
from typing_extensions import Annotated, Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, DataManager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.exporter import texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import collect_camera_poses, generate_point_cloud, get_mesh_from_filename
from nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes
from nerfstudio.fields.sdf_field import SDFField  # noqa
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.scripts.exporter import Exporter

from street_gaussians_ns.sgn_splatfacto import SplatfactoModel
from street_gaussians_ns.sgn_splatfacto_scene_graph import SplatfactoSceneGraphModel


@dataclass
class ExportGaussianSplat(Exporter):
    """
    Export 3D Gaussian Splatting model to a .ply
    """

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config, test_mode="inference")

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model

        filename = self.output_dir / "splat.ply"

        map_to_tensors = {}

        # model: GaussianSplattingModel = pipeline.model
        
        def save_gs_model(model,filename):
            map_to_tensors = {}

            with torch.no_grad():
                positions = model.means.cpu().numpy()
                n = positions.shape[0]
                map_to_tensors["x"] = positions[:,0, None]
                map_to_tensors["y"] = positions[:,1, None]
                map_to_tensors["z"] = positions[:,2, None]
                normals = np.zeros_like(positions)
                map_to_tensors["nx"] = normals[:,0, None]
                map_to_tensors["ny"] = normals[:,1, None]
                map_to_tensors["nz"] = normals[:,2, None]

                if model.config.sh_degree > 0:
                    shs_0 = model.shs_0.contiguous().cpu().numpy()
                    for i in range(shs_0.shape[1]):
                        map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

                    # transpose(1, 2) was needed to match the sh order in Inria version
                    shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                    shs_rest = shs_rest.reshape((n, -1))
                    for i in range(shs_rest.shape[-1]):
                        map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
                else:
                    colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                    map_to_tensors["colors"] = (colors * 255).astype(np.uint8)

                map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

                scales = model.scales.data.cpu().numpy()
                if scales.shape[-1] == 2:
                    map_to_tensors["scale_0"] = np.full_like(scales[:, 0, None], np.log(1e-8))
                    for i in range(1, 3):
                        map_to_tensors[f"scale_{i}"] = scales[:, i - 1, None]
                else:
                    for i in range(3):
                        map_to_tensors[f"scale_{i}"] = scales[:, i, None]

                quats = model.quats.data.cpu().numpy()
                for i in range(4):
                    map_to_tensors[f"rot_{i}"] = quats[:, i, None]


            # post optimization, it is possible have NaN/Inf values in some attributes
            # to ensure the exported ply file has finite values, we enforce finite filters.
            select = np.ones(n, dtype=bool)
            for k, t in map_to_tensors.items():
                n_before = np.sum(select)
                select = np.logical_and(select, np.isfinite(t).all(axis=1))
                n_after = np.sum(select)
                if n_after < n_before:
                    CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")

            if np.sum(select) < n:
                CONSOLE.print(f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
                for k, t in map_to_tensors.items():
                    map_to_tensors[k] = map_to_tensors[k][select, :]

            # pcd = o3d.t.geometry.PointCloud(map_to_tensors)

            # o3d.t.io.write_point_cloud(str(filename), pcd)
            from plyfile import PlyData, PlyElement
            dtype_full = [(attribute, 'f4') for attribute in map_to_tensors.keys()]
            elements = np.empty(np.sum(select), dtype=dtype_full)
            attributes = np.concatenate(list(map_to_tensors.values()), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(str(filename))
            CONSOLE.print(f"[bold green]:white_check_mark: Saved PLY to {filename}")

        if isinstance(model, SplatfactoSceneGraphModel):
            for k, sub_model in model.all_models.items():
                save_gs_model(sub_model, self.output_dir / f"point_cloud_{k}.ply")
        elif isinstance(model, SplatfactoModel):
            save_gs_model(model, self.output_dir / "point_cloud.ply")
        else:
            raise ValueError(f"Unsupported model type {type(model)}")



Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportGaussianSplat, tyro.conf.subcommand(name="gaussian-splat")],
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