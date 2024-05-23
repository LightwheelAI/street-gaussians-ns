from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type, Union
import copy
import math

from gsplat.sh import spherical_harmonics
from pytorch3d.transforms import quaternion_multiply
from torch.nn import Parameter
import mediapy as media
import torch
import torchvision.transforms.functional as TF

from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.cameras.camera_utils import quaternion_from_matrix
from nerfstudio.cameras.cameras import Cameras

from street_gaussians_ns.sgn_splatfacto import SplatfactoModel, SplatfactoModelConfig
from street_gaussians_ns.data.utils.bbox_optimizers import BBoxOptimizerConfig, BBoxOptimizer
from street_gaussians_ns.data.utils.dynamic_annotation import InterpolatedAnnotation, Box, parse_timestamp



@dataclass
class SplatfactoSceneGraphModelConfig(SplatfactoModelConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: SplatfactoSceneGraphModel)

    background_model: SplatfactoModelConfig = field(default_factory=SplatfactoModelConfig)
    """Background model config"""
    object_model_template: SplatfactoModelConfig = field(default_factory=SplatfactoModelConfig)
    """Object model config"""
    bbox_optimizer: BBoxOptimizerConfig = field(default_factory=BBoxOptimizerConfig)
    """Bounding box optimizer config"""
    object_acc_entropy_loss_mult: float = 0.001
    """loss weight of object-background accumulation cross entropy loss"""


class SplatfactoSceneGraphModel(SplatfactoModel):
    """Gaussian Splatting model

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """

    config: SplatfactoSceneGraphModelConfig
    def populate_modules(self):
        self.seed_points = None
        self.config.random_init = True
        super().populate_modules()
        for gs_param in list(self.gauss_params.keys()):
            # remove parameter from nn.Moudle, or it can cause bug when assign new value
            delattr(self, gs_param)
            # set the attribute to None
            setattr(self, gs_param, None)

        self._xys = None
        self._radii = None
        self._depths = None
        self._conics = None
        self._num_tiles_hit = None
        self._last_size = None

        self.all_models = torch.nn.ModuleDict()
        self.config.background_model.use_sky_sphere = False
        self.config.background_model.sh_degree = self.config.sh_degree
        self.all_models["background"] = self.config.background_model.setup(
            scene_box=self.scene_box,
            num_train_data=self.num_train_data,
            model_idx_in_scene_graph=0,
            **self.kwargs
        )

        self.object_annos: InterpolatedAnnotation = self.kwargs["metadata"].get("object_annos", InterpolatedAnnotation(anno_json_path=None))
        for idx, (obj_id, obj_meta) in enumerate(self.object_annos.objects_meta.items()):
            object_model_config = copy.deepcopy(self.config.object_model_template)
            object_model_config.use_sky_sphere = False
            object_model_config.sh_degree = self.config.sh_degree
            object_model_config.extent = torch.from_numpy(obj_meta.size).float() / 2
            self.all_models[self.get_object_model_name(obj_id)] = object_model_config.setup(
                scene_box=self.scene_box,
                num_train_data=self.num_train_data,
                seed_points=self.object_annos.get_seed_pts(obj_id),
                metadata = self.kwargs["metadata"],
                model_idx_in_scene_graph = idx + 1
            )
        self.visible_model_names: List[str] = list(self.all_models.keys())
        self.means = torch.cat([self.background_model.means], dim=0)
        self.bbox_optimizer: BBoxOptimizer = self.config.bbox_optimizer.setup(
            num_frames=len(self.object_annos.annos.keys()), 
            num_bboxes=len(self.object_annos.objects_meta), 
            frame_idx_map=self.build_frame_idx_map(), device="cpu",
            bbox_list=self.object_annos.objects_meta.keys(),
        )

    def get_object_model_name(self, object_id):
        return f"object_{object_id}"

    def build_frame_idx_map(self):
        frame_idx_map = {}
        for frame_idx, timestamp in enumerate(self.object_annos.annos.keys()):
            frame_idx_map[int(timestamp)] = torch.tensor(frame_idx,dtype=torch.int)
        return frame_idx_map

    @property
    def background_model(self) -> SplatfactoModel:
        return self.all_models["background"]

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = {}
        for model_name, model in self.all_models.items():
            for group_name, params in model.get_gaussian_param_groups().items():
                if group_name in groups:
                    groups[group_name] += params
                else:
                    groups[group_name] = params
        assert len(set(len(v) for v in groups.values())) == 1, "Submodules contain different gaussian param groups"
        return groups

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = super().get_param_groups()
        # add bbox optimizer param groups
        self.bbox_optimizer.get_param_groups(groups)
        return groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback(
            [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # for scene graph model, no longer need to call `after_train`` and `refinement_after` of itself
        for model in self.all_models.values():
            cbs += model.get_training_callbacks(training_callback_attributes)
        return cbs

    def get_aggreated_variable(self, var: str):
        """
        return a concated tensor of the variable from visable models
        eg. means_{all}=torch.cat([means_{bg},means_{obj1},means_{obj2},...],dim=0)
        """
        vars = [getattr(self.all_models[name], var) for name in self.visible_model_names]
        for a in vars:
            assert isinstance(a, torch.Tensor)
        return torch.concat(vars, dim=0)

    def set_variable_for_all_models(self, var_name: str, vals):
        for model_name in self.visible_model_names:
            setattr(self.all_models[model_name], var_name, vals)

    def set_split_tensor_variable(self, var_name: str, vals, retain_grad=False):
        """
        set the split tensor variable back to each model
        eg. means_{bg}=means_{all}[:len(means_{bg})]
            means_{obj1}=means_{all}[len(means_{bg}):len(means_{bg})+len(means_{obj1})]
            means_{obj2}=means_{all}[len(means_{bg})+len(means_{obj1}):]
        """
        assert isinstance(vals, torch.Tensor)
        vars_n = [self.all_models[model_name].num_points for model_name in self.visible_model_names]
        vals_split = torch.split(vals,vars_n)
        for model_name, vals_t in zip(self.visible_model_names, vals_split):
            model = self.all_models[model_name]
            setattr(model, var_name, vals_t)
            if retain_grad and self.training:
                getattr(model, var_name).retain_grad()

    @property
    def xys(self):
        return self._xys

    @xys.setter
    def xys(self, xys):
        self.set_split_tensor_variable("xys", xys, retain_grad=True)
        for model_name, model in self.all_models.items():
            if model_name not in self.visible_model_names:
                setattr(model, "xys", None)
        self._xys = self.get_aggreated_variable("xys")

    @property
    def radii(self):
        return self._radii

    @radii.setter
    def radii(self, radii):
        self.set_split_tensor_variable("radii", radii)
        self._radii = self.get_aggreated_variable("radii")

    @property
    def depths(self):
        return self._depths

    @depths.setter
    def depths(self, depths):
        self.set_split_tensor_variable("depths", depths)
        self._depths = self.get_aggreated_variable("depths")

    @property
    def conics(self):
        return self._conics

    @conics.setter
    def conics(self, conics):
        self.set_split_tensor_variable("conics", conics)
        self._conics = self.get_aggreated_variable("conics")

    @property
    def num_tiles_hit(self):
        return self._num_tiles_hit

    @num_tiles_hit.setter
    def num_tiles_hit(self, num_tiles_hit):
        self.set_split_tensor_variable("num_tiles_hit", num_tiles_hit)
        self._num_tiles_hit = self.get_aggreated_variable("num_tiles_hit")

    @property
    def last_size(self):
        return self._last_size

    @last_size.setter
    def last_size(self, last_size):
        self.set_variable_for_all_models("last_size", last_size)
        self._last_size = last_size

    # TODO remove this shit after make self.means a property
    @property
    def colors(self):
        if self.features_dc is None:
            return None
        return super().colors

    @property
    def num_points(self):
        if self.means is None:
            return 0
        return super().num_points

    def get_fourier_features(self, frame, trackId, obj_model: SplatfactoModel):
        frame_list = self.object_annos.objects_frames[trackId]
        if len(frame_list) == 1:
            normalized_frame = 1.0
        else:
            normalized_frame = (frame - frame_list[0]) / (frame_list[-1] - frame_list[0])
        t = normalized_frame * obj_model.config.fourier_features_scale
        idft_base = IDFT(t, obj_model.config.fourier_features_dim).to(self.device)
        return torch.sum(obj_model.features_dc*idft_base[..., None], dim=1, keepdim=True)

    def aggregate_submodel_var(self, var_name: str, submodel_names: List[str])->torch.Tensor:
        vars = [getattr(self.all_models[name], var_name) for name in submodel_names]
        for a in vars:
            assert isinstance(a, torch.Tensor)
        return torch.cat(vars, dim=0)

    def get_submodel_output(self, camera: Cameras,  submodel_names: List[str], sky_capture=None, object_means=None, object_features_dc=None, output_names=[]) -> torch.Tensor:
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        if object_means is None:
            submodel_means = self.aggregate_submodel_var("means", submodel_names)
            submodel_features_dc = self.aggregate_submodel_var("features_dc", submodel_names)
        else:
            assert object_features_dc is not None, "object_features_dc should not be None when object_means is not None"
            empty = torch.zeros(camera.height.item(), camera.width.item(), 1, device=self.device)
            if len(object_means) == 0:
                if 'accumulation' in output_names and len(output_names)==1:
                    return {'accumulation':empty}
                return {'rgb': empty if sky_capture is None else sky_capture, 'depth': empty}
            submodel_means = torch.cat(object_means, dim=0)
            submodel_features_dc = torch.cat(object_features_dc, dim=0)
        submodel_opacities = self.aggregate_submodel_var("opacities", submodel_names)
        submodel_features_rest = self.aggregate_submodel_var("features_rest", submodel_names)
        submodel_xys = self.aggregate_submodel_var("xys", submodel_names)
        submodel_depths = self.aggregate_submodel_var("depths", submodel_names)
        submodel_radii = self.aggregate_submodel_var("radii", submodel_names)
        submodel_conics = self.aggregate_submodel_var("conics", submodel_names)
        submodel_num_tiles_hit = self.aggregate_submodel_var("num_tiles_hit", submodel_names)
        # render submodel
        colors = torch.cat((submodel_features_dc, submodel_features_rest), dim=1)
        if self.config.sh_degree > 0:
            viewdirs = submodel_means.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            if not self.training:
                n = self.config.sh_degree
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
        gaussian_attrs = {
            "means": submodel_means,
            "colors": colors,
            "opacities": submodel_opacities,
            "xys": submodel_xys,
            "depths": submodel_depths,
            "radii": submodel_radii,
            "conics": submodel_conics,
            "num_tiles_hit": submodel_num_tiles_hit,
        }
        if sky_capture is not None:
            gaussian_attrs["sky_capture"] = sky_capture
        outputs = self.render_gaussian_attrs(camera, gaussian_attrs, output_names)
        camera.rescale_output_resolution(camera_downscale)
        return outputs

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Camera and returns a dictionary of outputs.

        Args:
            camera: Input camera of viewpoint. This camera should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        # TODO background only
        # TODO move aggregating means into a function or @property
        # prepare parameters
        # self.set_concated_gaussian_param_groups()
        #TODO get extra output for submodels debug and instance loss etc
        object_means = []
        object_quats = []
        object_features_dc = []
        assert camera.times is not None

        self.visible_model_names = ["background"]
        annos_t: List[Box] = self.object_annos[camera.times.item()] #type: ignore
        timestamp = parse_timestamp(camera.times.item())
        exist_frame = False
        if timestamp in self.object_annos.all_names:
            exist_frame = True
        if annos_t is not None and len(annos_t):
            for anno in annos_t:
                trackId = anno.trackId
                model_name = self.get_object_model_name(trackId)
                assert model_name not in self.visible_model_names
                obj_model = self.all_models[model_name]
                # prevent empty object
                if obj_model.num_points == 0:
                    continue
                if exist_frame:
                    self.bbox_optimizer.apply_to_bbox(anno)
                # add fourier features of time
                if self.config.fourier_features_dim > 1:
                    object_features_dc.append(self.get_fourier_features(anno.frame, trackId, obj_model))
                else:
                    object_features_dc.append(obj_model.features_dc)
                # aggregate all models properties for splatting
                self.visible_model_names.append(model_name)
                obj_means, obj_quats = object2world_gs(
                    obj_model.means, obj_model.quats, anno.center, anno.rot)
                object_means.append(obj_means)
                object_quats.append(obj_quats)

        # render all models
        self.means = torch.cat([self.background_model.means, *object_means], dim=0)
        self.quats = torch.cat([self.background_model.quats, *object_quats], dim=0)
        self.features_dc = torch.cat([self.background_model.features_dc, *object_features_dc], dim=0)
        self.opacities = self.get_aggreated_variable("opacities")
        self.features_rest = self.get_aggreated_variable("features_rest")
        self.scales = self.get_aggreated_variable("scales")
        assert self.crop_box is None or self.training, "crop_box is not supported for scene graph model now"
        # forward like the original model
        out = super().get_outputs(camera)
        out['object_acc'] = (self.get_submodel_output(camera, [submodel_name for submodel_name in self.visible_model_names if submodel_name.startswith("object")],
                                                       object_means=object_means, object_features_dc=object_features_dc, output_names=['accumulation']))['accumulation']
        out['background_acc'] = (self.get_submodel_output(camera, ["background"], output_names=['accumulation']))['accumulation']
        if not self.training:
            with torch.no_grad():
                background_output = self.get_submodel_output(camera, ["background"], sky_capture=out.get("sky", None), output_names=['rgb'])
                out.update({f"background_{k}":v for k, v in background_output.items()})
                object_output = self.get_submodel_output(camera, [submodel_name for submodel_name in self.visible_model_names if submodel_name.startswith("object")], object_means=object_means, object_features_dc=object_features_dc, output_names=['rgb'])
                out.update({f"object_{k}":v for k, v in object_output.items()})

        return out

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        losses = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.config.object_acc_entropy_loss_mult > 0. and self.step > self.config.background_model.stop_split_at:
            object_acc = torch.clamp(outputs['object_acc'], min=1e-5, max=1-1e-5)
            losses['object_acc_entropy_loss'] = self.config.object_acc_entropy_loss_mult * \
            -(object_acc*torch.log(object_acc) + (1. - object_acc)*torch.log(1. - object_acc)).mean()

        return losses

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        for model_name, model in self.all_models.items():
            sub_dict = {}
            for key in list(dict.keys()):
                if model_name in key:
                    sub_dict[".".join(key.split(".")[2:])] = dict.pop(key)
            model.load_state_dict(sub_dict, **kwargs)
        torch.nn.Module.load_state_dict(self, dict, strict=False)



def object2world_gs(means, quats, pose, rot):
    """Transform the object GS from object to world coordinate system
    """

    assert means.dim() == 2 and means.shape[1] == 3
    assert quats.dim() == 2 and quats.shape[1] == 4
    pose_o2w = torch.tensor(pose).to(means.device, dtype=means.dtype)  # c2w,3
    rot_o2w = torch.tensor(rot).to(means.device, dtype=means.dtype)  # c2w,3*3
    # convert rot_o2w to quat_o2w
    quat_o2w = torch.from_numpy(quaternion_from_matrix(rot))
    # transform the object GS from object to world coordinate system
    means_w = torch.matmul(means, rot_o2w.T) + pose_o2w[None, :] #[N_pts,3]
    quat_w = quaternion_multiply(quat_o2w, quats)
    return [means_w.squeeze(), quat_w.squeeze()]


def IDFT(time, dim):
    """
    Computes the inverse discrete Fourier transform of a given time signal.
    """
    if isinstance(time, float):
        time = torch.tensor(time)
    t = time.view(-1, 1)
    idft = torch.zeros(t.shape[0], dim, dtype=t.dtype, device=t.device)
    indices = torch.arange(dim, dtype=torch.int, device=t.device)
    even_indices = indices[::2]
    odd_indices = indices[1::2]
    idft[:, even_indices] = torch.cos(t * even_indices * 2 * math.pi / dim)
    idft[:, odd_indices] = torch.sin(t * (odd_indices + 1) * 2 * math.pi / dim)
    return idft