"""
BBox Pose and Rotation Optimizers
"""

import functools
from dataclasses import dataclass, field
from typing import Literal, Optional, Type, Union, List

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from typing_extensions import assert_never
import numpy as np

from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.utils import poses as pose_utils
from nerfstudio.engine.optimizers import OptimizerConfig
from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.cameras.camera_utils import quaternion_from_matrix, quaternion_matrix
from pytorch3d.transforms import quaternion_multiply

from street_gaussians_ns.data.utils.dynamic_annotation import InterpolatedAnnotation, Box


@dataclass
class BBoxOptimizerConfig(InstantiateConfig):
    """Configuration of optimization for camera poses."""

    _target: Type = field(default_factory=lambda: BBoxOptimizer)

    mode: Literal["off", "SO3xR3", "SE3", "simple"] = "off"
    """Pose optimization strategy to use. If enabled, we recommend SO3xR3."""

    center_l2_penalty: float = 1e-2
    """L2 penalty on center parameters."""

    rot_l2_penalty: float = 1e-3
    """L2 penalty on rotation parameters."""

    center_noise: float = 0.0
    """Noise added to center parameters."""

    rot_noise: bool = False
    """Noise added to rotation parameters."""

    optimizer: Optional[OptimizerConfig] = field(default=None)
    """Deprecated, now specified inside the optimizers dict"""

    scheduler: Optional[SchedulerConfig] = field(default=None)
    """Deprecated, now specified inside the optimizers dict"""


class BBoxOptimizer(nn.Module):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    config: BBoxOptimizerConfig

    def __init__(
        self,
        config: BBoxOptimizerConfig,
        num_frames: int,
        num_bboxes: int,
        frame_idx_map: dict,
        device: Union[torch.device, str],
        bbox_list: List=None,
        non_trainable_bbox_indices: Optional[Int[Tensor, "num_non_trainable_bboxes"]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_frames = num_frames
        self.num_bboxes = num_bboxes
        self.frame_idx_map = frame_idx_map
        self.device = device
        self.non_trainable_bbox_indices = non_trainable_bbox_indices
        self.bbox_list = list(bbox_list)

        # Initialize learnable parameters.
        if self.config.mode == "off":
            return
        elif self.config.mode in ("SO3xR3", "SE3", "simple"):
            if self.config.mode in ("SO3xR3", "SE3"):
                self.pose_adjustment = torch.nn.Parameter(torch.zeros((num_frames, num_bboxes, 6), device=device))
            elif self.config.mode == "simple":
                self.delta_center = torch.nn.Parameter(torch.zeros((num_frames, num_bboxes, 3), device=device)).requires_grad_(True)
                self.delta_yaw = torch.nn.Parameter(torch.zeros((num_frames, num_bboxes), device=device)).requires_grad_(True)

            # Initialize noise parameters.
            self.center_noise = torch.zeros((3,)).to(device)
            self.rot_noise = torch.eye(3).to(device)
            # add noise if set in config
            if config.center_noise != 0.0:
                self.center_noise = torch.randn_like(self.center_noise)
                self.center_noise /= self.center_noise.norm(dim=-1, keepdim=True)
                self.center_noise *= config.center_noise
            if config.rot_noise:
                random_matrix = np.random.rand(num_bboxes, 3, 3)
                rot_noise, _ = np.linalg.qr(random_matrix)
                self.rot_noise = rot_noise
        else:
            assert_never(self.config.mode)

    def forward(
        self,
        frame_idx: Int[Tensor, "bbox_frame_idx"],
        bbox_idx: Int[Tensor, "bbox_idx"],
    ) -> Float[Tensor, "bbox_indices 3 4"]:
        """Indexing into camera adjustments.
        Args:
            frame_id: frame id of BBox to optimize.
            indices: indices of BBox to optimize.
        Returns:
            Transformation matrices from optimized bboxes coordinates
            to given bboxes coordinates.
        """
        outputs = []

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        elif self.config.mode == "SO3xR3":
            outputs.append(exp_map_SO3xR3(self.pose_adjustment[frame_idx, bbox_idx, :]))
        elif self.config.mode == "SE3":
            outputs.append(exp_map_SE3(self.pose_adjustment[frame_idx, bbox_idx, :]))
        else:
            assert_never(self.config.mode)
        # Detach non-trainable indices by setting to identity transform
        if self.non_trainable_bbox_indices is not None:
            if self.non_trainable_bbox_indices.device != self.pose_adjustment.device:
                self.non_trainable_bbox_indices = self.non_trainable_bbox_indices.to(self.pose_adjustment.device)
            outputs[0][self.non_trainable_bbox_indices] = torch.eye(4, device=self.pose_adjustment.device)[:3, :4]

        # Return: identity if no transforms are needed, otherwise multiply transforms together.
        if len(outputs) == 0:
            # Note that using repeat() instead of tile() here would result in unnecessary copies.
            return torch.eye(4, device=self.device)[None, :3, :4].tile(frame_idx.shape[0], 1, 1)
        return functools.reduce(pose_utils.multiply, outputs)

    def apply_to_bbox(self, bbox: Box) -> None:
        """Apply the pose correction to the bbox"""
        if self.config.mode != "off":
            origin_centers = torch.from_numpy(bbox.center).to(self.device)
            origin_rots = torch.from_numpy(bbox.rot).to(self.device)
            frame_idx = self.frame_idx_map[bbox.frame_id]
            bbox_idx = self.bbox_list.index(bbox.trackId)
            if self.config.mode in ("SO3xR3", "SE3"):
                correction_matrix = self(frame_idx, bbox_idx).cpu()  # type: ignore
                # Apply correction to bboxes
                optimized_center  = (origin_centers + correction_matrix[bbox_idx, :3, 3] + self.center_noise).detach().numpy()
                optimized_rot = torch.bmm(correction_matrix[bbox_idx, :3, :3].double(), origin_rots.double())
                # Add noise to rotation
                rot_noise = self.rot_noise
                optimized_rot = torch.bmm(rot_noise.double(), optimized_rot).detach().numpy()
                bbox.center = optimized_center
                bbox.rot = optimized_rot
            elif self.config.mode == "simple":
                optimized_center = (origin_centers + self.delta_center[frame_idx, bbox_idx].cpu() + self.center_noise).detach().numpy() # + self.center_noise
                # delta_rot is computed from delta_yaw
                bbox_rot_quat = torch.from_numpy((quaternion_from_matrix(bbox.rot)))
                delta_rot = torch.zeros_like(bbox_rot_quat)
                delta_rot[0] = torch.cos(self.delta_yaw[frame_idx, bbox_idx])
                delta_rot[3] = torch.sin(self.delta_yaw[frame_idx, bbox_idx])
                optimized_rot = quaternion_multiply(bbox_rot_quat, delta_rot).detach().numpy()
                bbox.center = optimized_center
                bbox.rot = quaternion_matrix(optimized_rot)[:3, :3]

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        if self.config.mode != "off":
            loss_dict["bbox_opt_regularizer"] = (
                self.pose_adjustment[:, :3].norm(dim=-1).mean() * self.config.center_l2_penalty
                + self.pose_adjustment[:, 3:].norm(dim=-1).mean() * self.config.rot_l2_penalty
            )

    def get_correction_matrices(self):
        """Get optimized pose correction matrices"""
        return self(torch.arange(0, self.num_frames).long())

    def get_metrics_dict(self, metrics_dict: dict) -> None:
        """Get bbox optimizer metrics"""
        if self.config.mode != "off":
            metrics_dict["bbox_opt_center"] = self.pose_adjustment[:, :3].norm()
            metrics_dict["bbox_opt_rot"] = self.pose_adjustment[:, 3:].norm()

    def get_param_groups(self, param_groups: dict) -> None:
        """Get bbox optimizer parameters"""
        bbox_opt_params = list(self.parameters())
        if self.config.mode != "off":
            assert len(bbox_opt_params) > 0
            param_groups["bbox_opt"] = bbox_opt_params
        else:
            assert len(bbox_opt_params) == 0
