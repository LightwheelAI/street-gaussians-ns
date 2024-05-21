import json
from typing import List
from scipy.spatial.transform import Rotation as R
import numpy as np
import open3d as o3d
from pathlib import Path
import bisect

import argparse
import pickle
import torch
from functools import reduce

from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.cameras.camera_utils import quaternion_from_matrix, quaternion_slerp, quaternion_matrix

# only keep the box with label in FILTER_LABEL, if is None, select all
# FILTER_LABEL = None
FILTER_LABEL = ['car']

# the ratio used to expend 3D bbox
EXP_RATE = np.array([1.3, 1.3, 1.1])

COLORMAPS = [
    [1.0, 0.0, 0.0],  # Red
    [0.0, 1.0, 0.0],  # Green
    [0.0, 0.0, 1.0],  # Blue
    [1.0, 1.0, 0.0],  # Yellow
    [1.0, 0.0, 1.0],  # Magenta
    [0.0, 1.0, 1.0],  # Cyan
    [0.5, 0.5, 0.5],  # Gray
    [1.0, 0.5, 0.0],  # Orange
    [0.5, 0.0, 1.0],  # Purple
    [0.0, 0.5, 0.5]   # Teal
]

def parse_args():
    parser = argparse.ArgumentParser(
        description='prepapre annos.pkl for vdbfusion')
    parser.add_argument('--pcds_path', type=str, help='dataset path to pcd files',
                        default=None,
                        )
    parser.add_argument('--annos_path', type=str, help='dataset path to pcd files',
                        default=''
                        )
    parser.add_argument('--dataset_type', type=str, default='WaymoDataset',
                        choices=['WaymoDataset'])
    parser.add_argument('--anno_dst', type=str, default='./output/annos.pkl')
    parser.add_argument('--transform_json', type=str,
                        help='dataset path to pcd files', default=None)

    args = parser.parse_args()
    return args


def merge_mesh(meshes):
    return reduce(lambda x, y: x+y, meshes)


def upper_bound(nums, target):
    low, high = 0, len(nums)-1
    pos = len(nums)
    while low < high:
        mid = (low+high)//2
        if nums[mid] <= target:
            low = mid+1
        else:  # >
            high = mid
            pos = high
    if nums[low] > target:
        pos = low
    return pos


def frame_interpolation(frame_i1, frame_i2, frame_id):
    # get intersection  box
    trackId_i1 = {i.trackId: i for i in frame_i1}
    trackId_i2 = {i.trackId: i for i in frame_i2}
    intersection_trackId = list(
        set(trackId_i1.keys()) & set(trackId_i2.keys()))
    i_frame = []
    for trackId in intersection_trackId:
        box_i1 = trackId_i1[trackId]
        box_i2 = trackId_i2[trackId]
        iterpolation_box = Box.interploate(box_i1, box_i2, frame_id=frame_id)
        i_frame.append(iterpolation_box)
    return i_frame


def parse_timestamp(timestamp, l=16):
    if isinstance(timestamp, str):
        timestamp = float(timestamp)
    timestamp_str = str(int(timestamp))
    timestamp *= np.power(10, l-len(timestamp_str))
    timestamp = str(int(timestamp))
    return timestamp


class Box:
    def __init__(self, center, yaw=None, trackId=None, size=None, label=None, frame_id=-1, frame=-1, rot=None, quat=None) -> None:
        self.trackId = trackId
        self.center = np.array(center)
        # TODO assert

        self.croped_pc = None
        # self.yaw=yaw
        if rot is not None:
            assert rot.shape == (3, 3)
            self.rot = rot
        else:
            self.rot = R.from_euler('xyz', [0, yaw, 0]).as_matrix()
        # self.verticles = self.get_verticles(self.center, size, self.rot)

        self.size = size
        self.label = label
        self.frame_id = int(frame_id)
        self.frame = int(frame)
        self.quat = quat

    def get_inliers_outliers(self, pcd):
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(self.verticles))
        inliers_indices = obb.get_point_indices_within_bounding_box(pcd.points)
        # select inside points = cropped
        inliers_pcd = pcd.select_by_index(inliers_indices, invert=False)
        outliers_pcd = pcd.select_by_index(
            inliers_indices, invert=True)  # select outside points
        # print('@len(inliers_pcd.points)',len(inliers_pcd.points))
        return inliers_pcd, outliers_pcd

    def set_croped_pc(self, pcd: o3d.geometry.PointCloud):
        self.croped_pc = np.asarray(pcd.points)
        # transform to local coordinate
        self.croped_pc = self.croped_pc-self.center
        # rot = R.from_euler('xyz', [0,-self.yaw,0]).as_matrix()

        # rotate to local coordinate
        self.croped_pc = self.croped_pc@self.rot

    @staticmethod
    def get_verticles(center, size, rot):
        dx = 1/2*np.array([-1, -1, -1, -1, 1, 1, 1, 1])*size[0]
        dy = 1/2*np.array([-1, 1, 1, -1, -1, 1, 1, -1])*size[1]
        dz = 1/2*np.array([-1, -1, 1, 1, -1, -1, 1, 1])*size[2]
        verticles = np.array([dx, dy, dz]).T
        # verticles+center
        # rot=R.from_euler('xyz', [0,-yaw,0]).as_matrix() #c2w
        verticles = verticles@rot.T
        verticles = verticles+center
        return verticles

    @property
    def verticles(self):
        return self.get_verticles(self.center, self.size, self.rot)

    @staticmethod
    def interploate(box1, box2, frame_id, c2w=None):
        frame_id = int(frame_id)
        t = (frame_id-box1.frame_id)/(box2.frame_id-box1.frame_id)

        i_center = box1.center*(1-t)+box2.center*t
        # i_yaw=box1.yaw*(1-t)+box2.yaw*t
        rot1 = quaternion_from_matrix(box1.rot)
        rot2 = quaternion_from_matrix(box2.rot)
        i_quat = quaternion_slerp(rot1, rot2, t)
        i_rot = quaternion_matrix(i_quat)[:3, :3]

        # TODO to world
        box = Box(i_center, rot=i_rot, trackId=box1.trackId,
                  size=box1.size, label=box1.label, frame_id=frame_id)
        return box

    def to_mesh(self,c=[0.9, 0.1, 0.1]):
        w, h, l = self.size
        x, y, z = self.center
        mesh_box = o3d.geometry.TriangleMesh.create_box(
            width=w, height=h, depth=l)  # x y z
        # set anchor to center
        mesh_box.compute_vertex_normals()
        mesh_box.paint_uniform_color(c)
        mesh_box.rotate(self.rot)
        mesh_box.translate([x, y, z])
        mesh_box.translate([-w/2, -h/2, -l/2], relative=True)
        return mesh_box

    def transform(self, translation, rotation):
        self.center = np.dot(rotation, self.center)+translation
        # rot=R.from_euler('xyz', [0,-self.yaw,0]).as_matrix()
        self.rot = np.dot(rotation, self.rot)
        # self.yaw=R.from_matrix(rot).as_euler('xyz')[1]
        # self.verticles = self.get_verticles(self.center, self.size, self.rot)

    def scale(self, scale_factor):
        self.center = self.center*scale_factor
        self.size = self.size*scale_factor
        # self.verticles = self.get_verticles(self.center, self.size, self.rot)


def aggregate_crop_pcd(boxes: List[Box]):
    res = []
    for box in boxes:
        if box.croped_pc is None:
            print('box.croped_pc is None')
            continue
        res.append(box.croped_pc)
    if len(res) == 0:
        return None
    res = np.concatenate(res, axis=0)
    return res


class InterpolatedAnnotation:
    def __init__(self, anno_json_path, self_car_label=None, lidar_path=None, transform_matrix: np.ndarray = None, scale_factor: float = 1) -> None:
        # anno_json_path = Path(root)/'annotation.json'
        annos = []
        if anno_json_path is not None:
            assert anno_json_path.exists()
            # dynamic obj
            annos = json.load(open(anno_json_path))["frames"]
            annos = sorted(annos, key=lambda x: x['timestamp'])
        self.self_car_label = self_car_label
        self.lidar_path = lidar_path
        # self.transforms = json.load(open(transform_json_path))
        self.transform_matrix = np.eye(
            4) if transform_matrix is None else transform_matrix
        self.scale_factor = scale_factor
        # mapping timestamp to list of Box object
        self.annos = {}
        # mapping trackID to each object's seed_pts
        self.seed_pts = {}
        # mapping trackID to Box object
        self.objects_meta = {}
        self.objects_frames = {}
        for i, item in enumerate(annos):
            item['timestamp'] = parse_timestamp(item['timestamp'])
            self.annos[str(item['timestamp'])] = self.load_anno_json_one_frame(
                item['objects'], item['timestamp'], i, filter_label=FILTER_LABEL, ignore_static=True)

        self.all_names = list(self.annos.keys())
        self.unique_track_ids = list(self.objects_meta.keys())
        self.not_objects = []

    def __len__(self):
        return len(self.all_names)

    def get_by_id(self, index):
        return self.annos[self.all_names[index]]

    def get_seed_pts(self, trackId):
        return self.seed_pts[trackId]

    def __getitem__(self, frame_id):
        # get nearest frame_id in all_names
        if not len(self):
            return []
        # frame_id = str(frame_id)
        if isinstance(frame_id, (int, float)):
            # check within 0-1, assume it is a portion of the whole sequence rather than a timestamp
            if isinstance(frame_id,float) and frame_id>=0 and frame_id<=1 and len(self.all_names):
                frame_id = min(round(frame_id*len(self.all_names)),len(self.all_names)-1)
                frame_id = self.all_names[frame_id]
                print('@'*50,'frame_id is a portion of the whole sequence, use frame_id',frame_id)
            else:
                frame_id =  parse_timestamp(frame_id)
        elif isinstance(frame_id, str):
            frame_id = parse_timestamp(frame_id)
        else:
            raise ValueError('frame_id should be int or str')
        if frame_id in self.all_names:
            return self.annos[frame_id]
        # find nearest frame_id
        if frame_id < self.all_names[0] or frame_id > self.all_names[-1]:
            # print('@'*50,'frame_id',frame_id,'out of range')
            return []

        nearest_frame_id = bisect.bisect(
            [int(i) for i in self.all_names], int(frame_id))

        # print('@'*50,'nearest_frame_id-1',nearest_frame_id-1,'nearest_frame_id',nearest_frame_id,'frame_id',frame_id,self.all_names[nearest_frame_id-1])
        frame_i1, frame_i2 = self.all_names[nearest_frame_id -
                                            1], self.all_names[nearest_frame_id]
        # print('@'*50,"use interpolation, query frame_id",frame_id,'nearest_frame_id',nearest_frame_id,'frame_i1',frame_i1,'frame_i2',frame_i2)
        frame_i1, frame_i2 = self.annos[frame_i1], self.annos[frame_i2]
        i_frame = frame_interpolation(frame_i1, frame_i2, frame_id)
        # self.update(frame_id,i_frame)
        return i_frame

    def update(self, frame_id, boxes):
        frame_id = str(frame_id)
        self.all_names.append(frame_id)
        self.all_names.sort()
        self.annos.update({frame_id: boxes})
        self.annos = dict(sorted(self.annos.items(), key=lambda x: int(x[0])))

    def to_mesh(self):
        mesh = []
        for i in self.annos.values():
            # if True:##DEBUG
            # t=list(self.annos.keys())
            # i=self.annos[t[len(t)//2]]
            for j in i:
                mesh.append(j.to_mesh())
        return merge_mesh(mesh)

    def load_anno_json_one_frame(self, obj_list, timestamp, frame, filter_label=None, ignore_static=False):
        """
        filter_label: only select the box with label in filter_label, if is None, select all
        """
        boxes = []
        for idx, obj in enumerate(obj_list):
            if filter_label is not None:
                if obj['type'] not in filter_label and not obj['type'].endswith('Car'):
                    continue
            if ignore_static and not obj['is_moving']:
                continue
            if self.self_car_label is not None:
                if obj['gid'] == self.self_car_label:
                    continue
            center = obj['translation']
            quat = obj['rotation']
            trackId = obj['gid']
            ply_path = self.lidar_path / f"{trackId}.ply"
            if not ply_path.exists():
                continue
            pts = self.load_object_3D_points(trackId)
            if pts is None:
                continue
            rot = quaternion_matrix(quat)[:3, :3]
            size = EXP_RATE*np.array(obj['size'])
            box = Box(center, trackId=trackId, size=size,
                      label=obj['type'], frame_id=timestamp, frame=frame, rot=rot, quat=quat)
            box.transform(
                self.transform_matrix[:3, 3], self.transform_matrix[:3, :3])
            box.scale(self.scale_factor)
            boxes.append(box)
            # use first box as meta
            if trackId not in self.objects_meta:
                if self.lidar_path is not None:
                    pts = self.load_object_3D_points(trackId)
                    self.seed_pts[trackId] = pts
                    CONSOLE.log(f"Load object_{trackId} lidar points.")
                self.objects_meta[trackId] = box
                self.objects_frames[trackId] = []
            self.objects_frames[trackId].append(frame)
        # return np.array(pts)
        return boxes

    def load_object_3D_points(self, trackId: str):
        ply_path = self.lidar_path / f"{trackId}.ply"
        if not ply_path.exists():
            return None
        # assert ply_path.exists(), f"{ply_path} not exists"
        pcd = o3d.io.read_point_cloud(str(ply_path))
        # read points_xyz
        points3D = torch.from_numpy(np.array(pcd.points, dtype=np.float32))
        if points3D.shape[0] < 10000:
            return None
        points3D *= self.scale_factor
        # Load point colours
        if pcd.has_colors():
            points3D_rgb = torch.from_numpy(np.array(pcd.colors, dtype=np.float32)).float() * 255.
        else:
            points3D_rgb = torch.rand(points3D.shape[0], 3, dtype=torch.float32) * 255.
            
        return (points3D, points3D_rgb)

    def save_pickle(self, anno_dst, pcds_path, transform_json=None):
        # load
        annos_i = {}
        if transform_json is not None and Path(transform_json).exists():
            meta = json.load(open(transform_json))
            all_frame_id = [parse_timestamp(i["timestamp"])
                            for i in meta["lidar_frames"]]
            for frame_id in all_frame_id:
                t_ = self[frame_id]
                if t_ is None:
                    # frame before first anno or after last anno
                    continue
                annos_i[frame_id] = t_
            # for k,v in self.annos.items():
            #     assert len(v)==len(annos_i[k])
        annos_i.update(self.annos)
        # for waymo dataset ,we assume the timestamp of pcd is the same as the timestamp of annotation
        print('@'*50, 'before', len(self), 'after', len(self))
        Path(anno_dst).parent.mkdir(parents=True, exist_ok=True)
        with open(anno_dst, 'wb') as f:
            pickle.dump(annos_i, f)


if __name__ == '__main__':

    args = parse_args()
    if not (Path(args.annos_path)/'annotation.json').exists():
        exit()
    annos = InterpolatedAnnotation(
        anno_json_path=Path(args.annos_path)/'annotation.json',
        lidar_path=Path(args.annos_path)/'aggregate_lidar/dynamic_objects')
    # annos.save_pickle(args.anno_dst, args.pcds_path,
    #                   transform_json=args.transform_json)
    dst=Path(f'/home/qinglin.yang/qinglin/debug_anno')
    dst.mkdir(parents=True,exist_ok=True)
    for i,(t,anno) in enumerate(annos.annos.items()):
        if i%10:
            continue
        mesh = []
        for j in anno:
            c=COLORMAPS[annos.unique_track_ids.index(j.trackId)%len(COLORMAPS)]
            mesh.append(j.to_mesh(c=c))
        o3d.io.write_triangle_mesh((dst/f'anno_mesh_{t}.ply').as_posix(),merge_mesh(mesh))