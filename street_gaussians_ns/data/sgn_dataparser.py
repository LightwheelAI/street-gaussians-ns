""" Data parser for nerfstudio datasets. """

from __future__ import annotations

import json
import math
import tqdm
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Dict, Literal, Optional, Type, Tuple
from collections import defaultdict

import cv2
import imagesize
import numpy as np
import torch
from PIL import Image
from nerfstudio.cameras import camera_utils
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
from nerfstudio.process_data.colmap_utils import parse_colmap_camera_params
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.scripts import run_command
from nerfstudio.utils.rich_utils import CONSOLE, status


from nerfstudio.process_data.colmap_utils import parse_colmap_camera_params
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, CameraType, Cameras

from street_gaussians_ns.data.utils.dynamic_annotation import InterpolatedAnnotation
from street_gaussians_ns.data.utils.geometric_metric import gl2cv

import joblib

MAX_AUTO_RESOLUTION = 2000


@dataclass
class ColmapDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: ColmapDataParser)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    assume_colmap_world_coordinate_convention: bool = False
    """Colmap optimized world often have y direction of the first camera pointing towards down direction,
    while nerfstudio world set z direction to be up direction for viewer. Therefore, we usually need to apply an extra
    transform when orientation_method=none. This parameter has no effects if orientation_method is set other than none.
    When this parameter is set to False, no extra transform is applied when reading data from colmap.
    """
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "interval"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval (used by most academic papers, e.g. MipNerf360, GSplat).
    All uses all the images for any split.
    """
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    filter_camera_id: Optional[List[int]] = None
    """Filter images by camera id, None or empty list means no need to filter."""
    images_path: Path = Path("images")
    """Path to images directory relative to the data path."""
    masks_path: Optional[Path] = None
    """Path to masks directory. If not set, masks are not loaded."""
    segments_path: Optional[Path] = None
    """Path to segments directory. If not set, segments are not loaded."""
    colmap_path: Path = Path("colmap/sparse/0")
    """Path to the colmap reconstruction directory relative to the data path."""
    init_points_filename: str = "points3D.bin"
    """Specify the init points filename."""
    meta_file: Path = Path("transform.json")
    """meta file name."""
    load_3D_points: bool = True
    """Whether to load the 3D points from the colmap reconstruction. This is helpful for Gaussian splatting and
    generally unused otherwise, but it's typically harmless so we default to True."""
    max_2D_matches_per_3D_point: int = 0
    """Maximum number of 2D matches per 3D point. If set to -1, all 2D matches are loaded. If set to 0, no 2D matches are loaded."""
    undistort: bool = False
    """If true, undistort data in advance."""
    force_save_undistort_data: Tuple[Literal["all", "image", "mask", "depth", "semantic", "normal"], ...] = ()
    """The specified data will be forcely saved or updated after undistorting."""
    load_dynamic_annotations: bool = False
    """Whether to load dynamic annotations."""
    frame_select=[100,190]
    """Frame selection for dynamic annotations."""


class ColmapDataParser(DataParser):
    """COLMAP DatasetParser.
    Expects a folder with the following structure:
        images/ # folder containing images used to create the COLMAP model
        sparse/0 # folder containing the COLMAP reconstruction (either TEXT or BINARY format)
        masks/ # (OPTIONAL) folder containing masks for each image
        depths/ # (OPTIONAL) folder containing depth maps for each image
    The paths can be different and can be specified in the config. (e.g., sparse/0 -> sparse)
    Currently, most COLMAP camera models are supported except for the FULL_OPENCV and THIN_PRISM_FISHEYE models.

    The dataparser loads the downscaled images from folders with `_{downscale_factor}` suffix.
    If these folders do not exist, the user can choose to automatically downscale the images and
    create these folders.

    The loader is compatible with the datasets processed using the ns-process-data script and
    can be used as a drop-in replacement. It further supports datasets like Mip-NeRF 360 (although
    in the case of Mip-NeRF 360 the downsampled images may have a different resolution because they
    use different rounding when computing the image resolution).
    """

    config: ColmapDataParserConfig
    includes_time: bool = True
    channel_names: List = ["image", "mask", "segment"]

    def __init__(self, config: ColmapDataParserConfig):
        super().__init__(config)
        self.config = config
        self._downscale_factor = None
        self.channel_path = {}
        for channel in self.channel_names:
            self.channel_path[channel] = getattr(config, f"{channel}s_path")

    def _get_all_images_and_cameras(self, recon_dir: Path):
        if (recon_dir / "cameras.txt").exists():
            cam_id_to_camera = colmap_utils.read_cameras_text(recon_dir / "cameras.txt")
            im_id_to_image = colmap_utils.read_images_text(recon_dir / "images.txt")
        elif (recon_dir / "cameras.bin").exists():
            cam_id_to_camera = colmap_utils.read_cameras_binary(recon_dir / "cameras.bin")
            im_id_to_image = colmap_utils.read_images_binary(recon_dir / "images.bin")
        else:
            raise ValueError(f"Could not find cameras.txt or cameras.bin in {recon_dir}")

        file2time = None
        meta_file = self.config.data / self.config.meta_file
        meta = None
        if not meta_file.exists():
            CONSOLE.log(f"Fail to retrieval time because meta file {meta_file} doesn't exist.")
            self.includes_time = False
        else:
            with open(meta_file, "r") as f:
                meta = json.load(f)
            file2time = {frame["file_path"]: float(frame["timestamp"]) for frame in meta["frames"]}

        cameras = {}
        frames = []
        camera_model = []

        # Parse cameras
        for cam_id, cam_data in cam_id_to_camera.items():
            cameras[cam_id] = parse_colmap_camera_params(cam_data)

        # Parse frames
        # we want to sort all images based on im_id
        ordered_im_id = sorted(im_id_to_image.keys())
        for im_id in ordered_im_id:
            im_data = im_id_to_image[im_id]
            # NB: COLMAP uses Eigen / scalar-first quaternions
            # * https://colmap.github.io/format.html
            # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
            # the `rotation_matrix()` handles that format for us.
            rotation = colmap_utils.qvec2rotmat(im_data.qvec)
            translation = im_data.tvec.reshape(3, 1)
            w2c = np.concatenate([rotation, translation], 1)
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
            c2w = np.linalg.inv(w2c)
            # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
            c2w[0:3, 1:3] *= -1
            if self.config.assume_colmap_world_coordinate_convention:
                # world coordinate transform: map colmap gravity guess (-y) to nerfstudio convention (+z)
                c2w = c2w[np.array([0, 2, 1, 3]), :]
                c2w[2, :] *= -1

            frame = {
                "file_path": (self.config.data / self.config.images_path / im_data.name).as_posix(),
                "transform_matrix": c2w,
                "colmap_im_id": im_id,
                "camera_id": im_data.camera_id,
            }
            if file2time:
                frame["time"] = file2time[(self.config.images_path / im_data.name).as_posix()]
            frame.update(cameras[im_data.camera_id])

            if self.config.masks_path is not None:
                frame["mask_path"] = (
                    (self.config.data / self.config.masks_path / im_data.name).with_suffix(".png").as_posix()
                )
            if self.config.segments_path is not None:
                frame["segment_path"] = (
                    (self.config.data / self.config.segments_path / im_data.name).with_suffix(".png").as_posix()
                )
            frames.append(frame)
            camera_model.append(frame["camera_model"])

        out = {}
        frames.sort(key=lambda x: (x["camera_id"], x["time"] if "time" in x else 0, x["file_path"]))
        out["frames"] = frames
        if self.config.assume_colmap_world_coordinate_convention:
            # world coordinate transform: map colmap gravity guess (-y) to nerfstudio convention (+z)
            applied_transform = np.eye(4)[:3, :]
            applied_transform = applied_transform[np.array([0, 2, 1]), :]
            applied_transform[2, :] *= -1
            out["applied_transform"] = applied_transform.tolist()
        out["camera_model"] = camera_model
        if meta:
            # Colmap is run after this translation to all poses. 
            first_frame_pose = np.array(meta["frames"][0]["transform_matrix"])[:3,3]
            out["applied_translation_in_colmap"] = (-first_frame_pose * 0.98).tolist()
        assert len(frames) > 0, "No images found in the colmap model"
        return out

    def _get_image_indices(self, image_filenames, camera_ids, split):
        has_split_files_spec = (
            (self.config.data / "train_list.txt").exists()
            or (self.config.data / "test_list.txt").exists()
            or (self.config.data / "validation_list.txt").exists()
        )
        if (self.config.data / f"{split}_list.txt").exists():
            CONSOLE.log(f"Using {split}_list.txt to get indices for split {split}.")
            with (self.config.data / f"{split}_list.txt").open("r", encoding="utf8") as f:
                filenames = f.read().splitlines()
            # Validate split first
            split_filenames = set(self.config.data / self.config.images_path / x for x in filenames)
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(
                    f"Some filenames for split {split} were not found: {set(map(str, unmatched_filenames))}."
                )

            indices = []
            for i, (path, camera_id) in enumerate(zip(image_filenames, camera_ids)):
                if path not in split_filenames:
                    continue
                if self.config.filter_camera_id and camera_id not in self.config.filter_camera_id:
                    continue
                indices.append(i)

            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
        else:
            # filter image_filenames and poses based on train/eval split percentage
            if self.config.frame_select is not None:
                _, counts = torch.unique(camera_ids, return_counts=True)
                frame_len = counts[0]
                all_idx_list = []
                for i in range(len(self.config.filter_camera_id)):
                    start_frame = self.config.frame_select[0] + i * frame_len
                    end_frame = self.config.frame_select[1] + i * frame_len
                    all_idx_list.extend(range(start_frame, end_frame))
                all_idx = np.array(all_idx_list, dtype=np.int32)
            else:
                all_idx = np.arange(len(image_filenames), dtype=np.int32)
            if self.config.filter_camera_id:
                all_idx = np.array(
                    [i for i in all_idx if camera_ids[i] in self.config.filter_camera_id], dtype=np.int32
                )
            num_images = len(all_idx)
            num_train_images = math.ceil(num_images * self.config.train_split_fraction)
            num_eval_images = num_images - num_train_images
            i_all = np.arange(num_images)
            i_train = np.linspace(
                0, num_images - 1, num_train_images, dtype=int
            )  # equally spaced training images starting and ending at 0 and num_images-1
            i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
            assert len(i_eval) == num_eval_images
            if split == "train":
                indices = all_idx[i_train]
            elif split in ["val", "test"]:
                indices = all_idx[i_eval]
            elif split == "all":
                indices = all_idx
            else:
                raise ValueError(f"Unknown dataparser split {split}")
        return indices

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        colmap_path = self.config.data / self.config.colmap_path
        assert colmap_path.exists(), f"Colmap path {colmap_path} does not exist."

        dataparser_outputs_path = self.config.data/"dataparser_transforms.json"
        meta = self._get_all_images_and_cameras(colmap_path)
        camera_type = torch.tensor([CAMERA_MODEL_TO_TYPE[model].value for model in meta["camera_model"]])
        
        
        filenames = defaultdict(list) 

        camera_ids = []
        poses = []
        times = []

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
            fx.append(float(frame["fl_x"]))
            fy.append(float(frame["fl_y"]))
            cx.append(float(frame["cx"]))
            cy.append(float(frame["cy"]))
            height.append(int(frame["h"]))
            width.append(int(frame["w"]))
            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )

            filenames["image"].append(Path(frame["file_path"]))
            poses.append(frame["transform_matrix"])
            camera_ids.append(frame["camera_id"])
            if "time" in frame:
                times.append(frame["time"])
            for channel in self.channel_names:
                if f"{channel}_path" in frame:
                    filenames[channel].append(Path(frame[f"{channel}_path"]))

        # check that all filenames
        assert len(eval('times')) == 0 or (len(eval('times')) == len(filenames["image"])), """
        Different number of image and times.
        You should check that times is specified for every frame (or zero frames) in transforms.json.
        """

        for key, value in filenames.items():
            assert len(value) == 0 or (len(value) == len(filenames["image"])), f"""
            Different number of image and {value}.
            You should check that {value} is specified for every frame (or zero frames) in transforms.json.
            """

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        if(dataparser_outputs_path.exists()):
            dataparser_outputs = load_from_json(dataparser_outputs_path)
            transform_matrix = torch.from_numpy(np.array(dataparser_outputs["transform"]).astype(np.float32))
            if "applied_transform" in meta:
                applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
                transform_matrix = transform_matrix @  torch.inverse(torch.cat(
                    [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
                ))
            poses = transform_matrix @ poses
            scale_factor = float(dataparser_outputs["scale"])
        else:
            poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
                poses,
                method=self.config.orientation_method,
                center_method=self.config.center_method,
            )

            # Scale poses
            scale_factor = 1.0
            if self.config.auto_scale_poses:
                scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))

        scale_factor *= self.config.scale_factor
        poses[:, :3, 3] *= scale_factor

        fx = torch.tensor(fx, dtype=torch.float32)
        fy = torch.tensor(fy, dtype=torch.float32)
        cx = torch.tensor(cx, dtype=torch.float32)
        cy = torch.tensor(cy, dtype=torch.float32)
        height = torch.tensor(height, dtype=torch.int32)
        width = torch.tensor(width, dtype=torch.int32)
        distortion_params = torch.stack(distort, dim=0)
        camera_ids = torch.tensor(camera_ids, dtype=torch.uint8)
        camera_type = camera_type
        if times:
            times = torch.tensor(times, dtype=torch.float64)
            # times = (times - times.min()) / (times.max() - times.min())

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            times=times if self.includes_time else None
        )
        (
            cameras,
            filenames,
            downscale_factor,
        ) = self._setup_downscale_and_undistort(cameras, filenames)

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        indices = self._get_image_indices(filenames["image"], camera_ids, split)
        for channel, filelist in filenames.items():
            filenames[channel] = [filelist[i] for i in indices]

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        cameras = cameras[idx_tensor]
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
            transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        metadata = {}
        if self.config.load_3D_points:
            # Load 3D points
            metadata.update(self._load_3D_points(colmap_path, transform_matrix, scale_factor))
        if self.config.load_dynamic_annotations:
            translation = meta["applied_translation_in_colmap"]
            translation = gl2cv(np.append(translation, 1))
            # get 4x4 matrix
            transform_matrix_colmap = np.eye(4)
            transform_matrix_colmap[:3,3] = translation[:3]
            # Load dynamic annotations
            transform_matrix_anno = transform_matrix.numpy() @ transform_matrix_colmap
            metadata.update({"object_annos":InterpolatedAnnotation(
                anno_json_path=self.config.data / 'annotation.json',
                lidar_path=self.config.data / 'aggregate_lidar' / 'dynamic_objects',
                transform_matrix=transform_matrix_anno,
                scale_factor=scale_factor,)})
            # TOOD anno pointcloud
        if "applied_translation_in_colmap" in meta:
            metadata["applied_translation_in_colmap"] = meta["applied_translation_in_colmap"]
        dataparser_outputs = DataparserOutputs(
            image_filenames=filenames["image"],
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=filenames.get("mask", None),
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "segment_filenames": filenames.get("segment", None),
                "cameras": cameras,
                **metadata,
            },
        )
        return dataparser_outputs

    def _load_3D_points(self, colmap_path: Path, transform_matrix: torch.Tensor, scale_factor: float):
        points_filepath = colmap_path / self.config.init_points_filename
        assert points_filepath.exists()
        if points_filepath.suffix == ".bin":
            colmap_points = colmap_utils.read_points3D_binary(points_filepath)
        elif points_filepath.suffix == ".txt":
            colmap_points = colmap_utils.read_points3D_text(points_filepath)
        else:
            raise ValueError(f"Could not find points3D.txt or points3D.bin in {colmap_path}")
        points3D = torch.from_numpy(np.array([p.xyz for p in colmap_points.values()], dtype=np.float32))
        points3D = (
            torch.cat(
                (
                    points3D,
                    torch.ones_like(points3D[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T
        )
        points3D *= scale_factor

        # Load point colours
        points3D_rgb = torch.from_numpy(np.array([p.rgb for p in colmap_points.values()], dtype=np.uint8))
        points3D_num_points = torch.tensor([len(p.image_ids) for p in colmap_points.values()], dtype=torch.int64)
        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
            "points3D_error": torch.from_numpy(np.array([p.error for p in colmap_points.values()], dtype=np.float32)),
            "points3D_num_points2D": points3D_num_points,
        }
        if self.config.max_2D_matches_per_3D_point != 0:
            if (colmap_path / "images.txt").exists():
                im_id_to_image = colmap_utils.read_images_text(colmap_path / "images.txt")
            elif (colmap_path / "images.bin").exists():
                im_id_to_image = colmap_utils.read_images_binary(colmap_path / "images.bin")
            else:
                raise ValueError(f"Could not find images.txt or images.bin in {colmap_path}")
            downscale_factor = self._downscale_factor
            max_num_points = int(torch.max(points3D_num_points).item())
            if self.config.max_2D_matches_per_3D_point > 0:
                max_num_points = min(max_num_points, self.config.max_2D_matches_per_3D_point)
            points3D_image_ids = []
            points3D_image_xy = []
            for p in colmap_points.values():
                nids = np.array(p.image_ids, dtype=np.int64)
                nxy_ids = np.array(p.point2D_idxs, dtype=np.int32)
                if self.config.max_2D_matches_per_3D_point != -1:
                    # Randomly sample 2D matches
                    idxs = np.argsort(p.error)[: self.config.max_2D_matches_per_3D_point]
                    nids = nids[idxs]
                    nxy_ids = nxy_ids[idxs]
                nxy = [im_id_to_image[im_id].xys[pt_idx] for im_id, pt_idx in zip(nids, nxy_ids)]
                nxy = torch.from_numpy(np.stack(nxy).astype(np.float32))
                nids = torch.from_numpy(nids)
                assert len(nids.shape) == 1
                assert len(nxy.shape) == 2
                points3D_image_ids.append(
                    torch.cat((nids, torch.full((max_num_points - len(nids),), -1, dtype=torch.int64)))
                )
                points3D_image_xy.append(
                    torch.cat((nxy, torch.full((max_num_points - len(nxy), nxy.shape[-1]), 0, dtype=torch.float32)))
                    / downscale_factor
                )
            out["points3D_image_ids"] = torch.stack(points3D_image_ids, dim=0)
            out["points3D_points2D_xy"] = torch.stack(points3D_image_xy, dim=0)
        return out

    def _downscale_images(
        self,
        paths,
        get_fname,
        downscale_factor: int,
        nearest_neighbor: bool = False,
    ):
        with status(msg="[bold yellow]Downscaling images...", spinner="growVertical"):
            assert downscale_factor > 1
            assert isinstance(downscale_factor, int)

            def _downscale(path):
                nn_flag = "" if not nearest_neighbor else ":flags=neighbor"
                path_out = get_fname(path)
                path_out.parent.mkdir(parents=True, exist_ok=True)

                if path_out.suffix in (".jpg", ".png", ".jpeg"):
                    ffmpeg_cmd = [
                        f'ffmpeg -y -noautorotate -i "{path}" ',
                        f"-q:v 2 -vf scale=iw/{downscale_factor}:ih/{downscale_factor}{nn_flag} ",
                        f'"{path_out}"',
                    ]
                    ffmpeg_cmd = " ".join(ffmpeg_cmd)
                    run_command(ffmpeg_cmd)
                elif path_out.suffix == ".npy":
                    data = np.load(path)
                    h, w = data.shape
                    scaled = cv2.resize(data, (w // downscale_factor, h // downscale_factor), interpolation=cv2.INTER_NEAREST)
                    np.save(path_out, scaled)
                else:
                    raise NotImplementedError

            joblib.Parallel(n_jobs=-1, backend="threading")(
                joblib.delayed(_downscale)(path) for path in tqdm.tqdm(paths)
            )
        CONSOLE.log("[bold green]:tada: Done downscaling images.")

    def _downscale_and_undistort_one(self, camera: Cameras, filename):
        assert len(camera.shape) == 0
        
        width, height = imagesize.get(filename["image"])
        width = width // self._downscale_factor
        height = height // self._downscale_factor

        contents = [(channel, self.channel_path[channel], filename[channel]) for channel in filename.keys()] 
        data = {}
        for key, path, filename in contents:
            if filename is None:
                continue
            out = self._get_fname(self.config.data / path, filename)
            if "all" not in self.config.force_save_undistort_data and key not in self.config.force_save_undistort_data and out.exists():
                continue
            assert filename.suffix in (".png", ".jpg")
            value = cv2.imread(filename.as_posix(), cv2.IMREAD_UNCHANGED)
            if self._downscale_factor > 1:
                value = cv2.resize(
                    value, (width, height), interpolation=cv2.INTER_LINEAR if key == "image" else cv2.INTER_NEAREST
                )
            data[key] = value

        if self.config.undistort:
            if camera.distortion_params is not None and camera.distortion_params.abs().sum() > 0:
                K = camera.get_intrinsics_matrices().numpy()
                distortion_params = camera.distortion_params.numpy()

                if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
                    distortion_params = np.array(
                        [
                            distortion_params[0],
                            distortion_params[1],
                            distortion_params[4],
                            distortion_params[5],
                            distortion_params[2],
                            distortion_params[3],
                            0,
                            0,
                        ]
                    )
                    newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (width, height), 0)
                    map1, map2 = cv2.initUndistortRectifyMap(
                        K, distortion_params, None, newK, (width, height), cv2.CV_32FC1
                    )
                    x, y, w, h = roi
                    # update the width, height
                    camera.width[:] = w + 1
                    camera.height[:] = h + 1

                    for key in data.keys():
                        value = data[key]
                        dtype = value.dtype

                        # Some fields must use nearst interpolation.
                        if key in ("mask", "semantic"):
                            undistort_value = cv2.remap(
                                value.astype(np.float32), map1, map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
                            )
                        else:
                            undistort_value = cv2.undistort(value.astype(np.float32), K, distortion_params, None, newK)
                        undistort_value = undistort_value[y : y + h + 1, x : x + w + 1].astype(dtype)
                        data[key] = undistort_value

                elif camera.camera_type.item() == CameraType.FISHEYE.value:
                    distortion_params = np.array(
                        [distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3]]
                    )
                    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                        K, distortion_params, (width, height), np.eye(3), balance=0
                    )
                    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                        K, distortion_params, np.eye(3), newK, (width, height), cv2.CV_32FC1
                    )

                    for key in data.keys():
                        value = data[key]
                        dtype = value.dtype

                        # Some fields must use nearst interpolation.
                        if key in ("mask", "semantic"):
                            undistort_value = cv2.remap(
                                value.astype(np.float32), map1, map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
                            )
                        else:
                            undistort_value = cv2.fisheye.undistortImage(value.astype(np.float32), K, distortion_params, None, newK)
                        undistort_value = undistort_value.astype(dtype)
                        data[key] = undistort_value
                else:
                    raise NotImplementedError("Only perspective and fisheye cameras are supported")

                camera.fx[:] = float(newK[0, 0])
                camera.fy[:] = float(newK[1, 1])
                camera.cx[:] = float(newK[0, 2])
                camera.cy[:] = float(newK[1, 2])
                camera.distortion_params[:] = 0.

        for key, path, filename in contents:
            if key not in data:
                continue
            out = self._get_fname(self.config.data / path, filename)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(out.as_posix(), data[key])

        return camera

    def _setup_downscale_and_undistort(
        self,
        cameras: Cameras,
        filenames: Dict[List[Path]],
    ):
        need_undistort = self.config.undistort and cameras.distortion_params is not None and cameras.distortion_params.abs().sum() > 1e-3
        if self.config.undistort and not need_undistort:
            CONSOLE.log(f"No need to undistort images.")
            self.config.undistort = need_undistort

        if self._downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(filenames["image"][0])
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) <= MAX_AUTO_RESOLUTION:
                        break
                    df += 1

                self._downscale_factor = 2**df
                CONSOLE.log(f"Using image downscale factor of {self._downscale_factor}")
            else:
                self._downscale_factor = self.config.downscale_factor

        if self._downscale_factor > 1:
            cameras.rescale_output_resolution(scaling_factor=1.0 / self._downscale_factor)

        filename_pers = [{key: filenames[key][i] for key in filenames.keys()} for i in range(len(filenames["image"]))]

        with status(msg="[bold yellow]Downscaling and undistorting images...", spinner="growVertical"):
            camera_list = joblib.Parallel(n_jobs=8, backend="multiprocessing")(
                joblib.delayed(self._downscale_and_undistort_one)(
                    cameras[i],
                    filename_pers[i],
                )
                for i in tqdm.tqdm(range(len(filenames["image"])))
            )
        cameras = Cameras(
            fx=torch.stack([c.fx for c in camera_list]),
            fy=torch.stack([c.fy for c in camera_list]),
            cx=torch.stack([c.cx for c in camera_list]),
            cy=torch.stack([c.cy for c in camera_list]),
            distortion_params=torch.stack([c.distortion_params for c in camera_list]),
            height=torch.stack([c.height for c in camera_list]),
            width=torch.stack([c.width for c in camera_list]),
            camera_to_worlds=torch.stack([c.camera_to_worlds for c in camera_list]),
            camera_type=torch.stack([c.camera_type for c in camera_list]),
            times=torch.stack([c.times for c in camera_list]) if self.includes_time else None
        )

        for channel, filelist in filenames.items():
            assert self.channel_path[channel] is not None
            filenames[channel] = [self._get_fname(self.config.data / self.channel_path[channel], fp) for fp in filelist]
        
        return cameras, filenames, self._downscale_factor

    def _get_fname(self, parent: Path, filepath: Path, downscale_factor=None, undistort=None) -> Path:
        """Returns transformed file name when downscale factor is applied"""
        if downscale_factor is None:
            downscale_factor = self._downscale_factor
        if undistort is None:
            undistort = self.config.undistort
        rel_part = filepath.relative_to(parent)
        base_part = parent.parent / f"{parent.name}{'_ud' if undistort else ''}{'_' + str(downscale_factor) if downscale_factor > 1 else ''}"
        return base_part / rel_part
