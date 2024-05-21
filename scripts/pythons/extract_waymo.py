from multiprocessing import Pool
from pathlib import Path
from typing import List, Dict, Any
import argparse
import json
import traceback

from scipy.spatial.transform import Rotation as R
from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.utils import frame_utils
import numpy as np
import open3d as o3d
import tensorflow as tf


if int(tf.__version__.split(".")[0]) < 2:
    tf.enable_eager_execution()


class WaymoDataExtractor:
    RETURN_OK = 0
    RETURN_SKIP = 1

    MIN_MOVING_SPEED = 0.2

    def __init__(self, waymo_root: Path | str, num_workers: int) -> None:
        self.waymo_root = Path(waymo_root)
        self.num_workers = num_workers

        self._box_type_to_str = {
            label_pb2.Label.Type.TYPE_UNKNOWN: "unknown",
            label_pb2.Label.Type.TYPE_VEHICLE: "car",
            label_pb2.Label.Type.TYPE_PEDESTRIAN: "pedestrian",
            label_pb2.Label.Type.TYPE_SIGN: "sign",
            label_pb2.Label.Type.TYPE_CYCLIST: "cyclist",
        }

    def extract_all(self, split: str, specify_segments: List[str], out_root: Path | str):
        all_segments = self.list_segments()

        def find_segement(partial_segment_name: str, segments: List[Path]):
            for seg in segments:
                if partial_segment_name in seg.as_posix():
                    return seg
            return None

        inexist_segs, task_segs = [], []
        if specify_segments:
            for specify_segment in specify_segments:
                seg = find_segement(specify_segment, all_segments)
                if seg is None:
                    inexist_segs.append(specify_segment)
                else:
                    task_segs.append(seg)
        else:
            task_segs = all_segments

        if inexist_segs:
            print(f"{len(inexist_segs)} segments not found:")
            for seg in inexist_segs:
                print(seg)

        def print_error(e):
            # print("ERROR:", e)
            traceback.print_exception(e)

        fail_tasks, skip_tasks, succ_tasks = [], [], []
        with Pool(processes=self.num_workers) as pool:
            results = [
                pool.apply_async(func=self.extract_one, args=(seg, Path(out_root) / split), error_callback=print_error)
                for seg in task_segs
            ]

            for result in results:
                result.wait()

            for segment, result in zip(task_segs, results):
                if not result.successful():
                    fail_tasks.append(segment)
                elif result.get() == WaymoDataExtractor.RETURN_SKIP:
                    skip_tasks.append(segment)
                elif result.get() == WaymoDataExtractor.RETURN_OK:
                    succ_tasks.append(segment)

        print(
            f"""{len(task_segs)} tasks total, {len(fail_tasks)} tasks failed, """
            f"""{len(skip_tasks)} tasks skipped, {len(succ_tasks)} tasks success"""
        )
        print("Failed tasks:")
        for seg in fail_tasks:
            print(seg.as_posix())
        print("Skipped tasks:")
        for seg in skip_tasks:
            print(seg.as_posix())

    def list_segments(self, split=None) -> List[str]:
        if split is None:
            return list(self.waymo_root.glob("*/*.tfrecord"))
        else:
            return list((self.waymo_root / split).glob("*.tfrecord"))

    def extract_one(self, segment_tfrecord: Path, out_dir: Path) -> int:
        dataset = tf.data.TFRecordDataset(segment_tfrecord.as_posix(), compression_type="")
        segment_name = None
        segment_out_dir = None
        sensor_params = None
        camera_frames = []
        lidar_frames = []
        annotations = []

        for frame_idx, data in enumerate(dataset):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            if frame_idx == 0:
                segment_name = frame.context.name
                segment_out_dir = out_dir / segment_name
                if (segment_out_dir / "transform.json").exists() and (segment_out_dir / "annotation.json").exists():
                    return WaymoDataExtractor.RETURN_SKIP
                sensor_params = self.extract_sensor_params(frame)
            else:
                assert segment_name == frame.context.name

            camera_frames.extend(self.extact_frame_images(frame, segment_out_dir, sensor_params))
            lidar_frames.extend(self.extract_frame_lidars(frame, segment_out_dir, sensor_params))
            annotations.append(self.extract_frame_annotation(frame))

        camera_frames.sort(key=lambda frame: f"{frame['file_path']}")
        lidar_frames.sort(key=lambda frame: f"{frame['file_path']}")
        meta = {"sensor_params": sensor_params, "frames": camera_frames, "lidar_frames": lidar_frames}

        with open(segment_out_dir / "transform.json", "w") as fout:
            json.dump(meta, fout, indent=4)
        with open(segment_out_dir / "annotation.json", "w") as fout:
            json.dump({"frames": annotations}, fout, indent=4)

        return WaymoDataExtractor.RETURN_OK

    def extract_sensor_params(self, frame: dataset_pb2.Frame) -> Dict[str, Any]:
        out = {"camera_order": [self.get_camera_name(i) for i in [1, 2, 4, 5, 3]]}
        for camera_calib in frame.context.camera_calibrations:
            camera_name = self.get_camera_name(camera_calib.name)

            intrinsic = camera_calib.intrinsic
            fx, fy, cx, cy = intrinsic[:4]
            distortion = intrinsic[4:]

            extrinsic = np.array(camera_calib.extrinsic.transform).reshape((4, 4))
            # convert waymo camera coord to opencv camera coord.
            opencv2waymo = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            extrinsic[:3, :3] = extrinsic[:3, :3] @ opencv2waymo

            out[camera_name] = {
                "type": "camera",
                "camera_model": "OPENCV",
                "camera_intrinsic": [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                "camera_D": distortion,
                "extrinsic": extrinsic.tolist(),
                "width": camera_calib.width,
                "height": camera_calib.height,
            }

        for lidar_calib in frame.context.laser_calibrations:
            lidar_name = self.get_lidar_name(lidar_calib.name)
            extrinsic = np.array(lidar_calib.extrinsic.transform).reshape((4, 4))
            out[lidar_name] = {"type": "lidar", "extrinsic": extrinsic.tolist()}

        return out

    def extact_frame_images(
        self, frame: dataset_pb2.Frame, segment_out_dir: Path, sensor_params
    ) -> List[Dict[str, Any]]:
        lidar_timestamp: int = frame.timestamp_micros
        frame_images = []

        for image_data in frame.images:
            camera_name = self.get_camera_name(image_data.name)

            save_path = segment_out_dir / "images" / camera_name / f"{lidar_timestamp}.jpg"
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True)
            with open(save_path, "wb") as fp:
                fp.write(image_data.image)

            ego_pose = np.array(image_data.pose.transform).reshape((4, 4))
            camera_params = sensor_params[camera_name]
            intrinsic = np.array(camera_params["camera_intrinsic"])
            extrinsic = np.array(camera_params["extrinsic"])  # opencv camera coord
            distortion = camera_params["camera_D"]
            camera2world = ego_pose @ extrinsic  # opencv camera coord
            # Convert to nerfstudio/blender camera coord
            camera2world[0:3, 1:3] *= -1
            camera2world = camera2world[np.array([1, 0, 2, 3]), :]
            camera2world[2, :] *= -1

            frame_images.append(
                {
                    "file_path": save_path.relative_to(segment_out_dir).as_posix(),
                    "fl_x": intrinsic[0, 0],
                    "fl_y": intrinsic[1, 1],
                    "cx": intrinsic[0, 2],
                    "cy": intrinsic[1, 2],
                    "w": camera_params["width"],
                    "h": camera_params["height"],
                    "camera_model": "OPENCV",
                    "camera": camera_name,
                    "timestamp": lidar_timestamp / 1.0e6,
                    "k1": distortion[0],
                    "k2": distortion[1],
                    "k3": distortion[4],
                    "k4": 0.0,
                    "p1": distortion[2],
                    "p2": distortion[3],
                    "transform_matrix": camera2world.tolist(),
                }
            )

        return frame_images

    def extract_frame_lidars(
        self, frame: dataset_pb2.Frame, segment_out_dir: Path, sensor_params
    ) -> List[Dict[str, Any]]:
        lidar_timestamp: int = frame.timestamp_micros
        frame_lidars = []
        pose = np.array(frame.pose.transform).reshape((4, 4))

        (
            range_images,
            camera_projections,
            _,
            range_image_top_pose,
        ) = frame_utils.parse_range_image_and_camera_projection(frame)
        points, _ = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose
        )
        points_ri2, _ = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose, ri_index=1
        )
        points = [np.concatenate([p1, p2]) for p1, p2 in zip(points, points_ri2)]
        lidar_ids = [calib.name for calib in frame.context.laser_calibrations]
        lidar_ids.sort()
        for lidar_id, lidar_points in zip(lidar_ids, points):
            lidar_name = self.get_lidar_name(lidar_id)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(lidar_points)
            save_path = segment_out_dir / "lidars" / lidar_name / f"{lidar_timestamp}.pcd"
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True)
            o3d.io.write_point_cloud(save_path.as_posix(), pcd)

            frame_lidars.append(
                {
                    "file_path": save_path.relative_to(segment_out_dir).as_posix(),
                    "lidar": lidar_name,
                    "timestamp": lidar_timestamp / 1.0e6,
                    "transform_matrix": pose.tolist(),
                }
            )

        return frame_lidars

    def extract_frame_annotation(self, frame: dataset_pb2.Frame) -> Dict[str, Any]:
        pose = np.array(frame.pose.transform).reshape((4, 4))
        objects = []
        for label in frame.laser_labels:
            center_vcs = np.array([label.box.center_x, label.box.center_y, label.box.center_z, 1])
            center_wcs = pose @ center_vcs
            heading = label.box.heading
            rotation_vcs = R.from_euler("xyz", [0, 0, heading], degrees=False).as_matrix()
            rotation_wcs = pose[:3, :3] @ rotation_vcs
            rotation_wcs = R.from_matrix(rotation_wcs).as_quat()

            speed = np.sqrt(label.metadata.speed_x**2 + label.metadata.speed_y**2 + label.metadata.speed_z**2)

            objects.append(
                {
                    "type": self._box_type_to_str[label.type],
                    "gid": label.id,
                    "translation": center_wcs[:3].tolist(),
                    "size": [label.box.length, label.box.width, label.box.height],
                    "rotation": [rotation_wcs[3], rotation_wcs[0], rotation_wcs[1], rotation_wcs[2]],
                    "is_moving": bool(speed > self.MIN_MOVING_SPEED)
                }
            )

        return {"timestamp": frame.timestamp_micros / 1.0e6, "objects": objects}

    def get_camera_name(self, name_int) -> str:
        return dataset_pb2.CameraName.Name.Name(name_int)

    def get_lidar_name(self, name_int) -> str:
        # Avoid using same names with cameras
        return "lidar_" + dataset_pb2.LaserName.Name.Name(name_int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--waymo_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--split", default="training", const="training", nargs="?", choices=["training", "testing", "validation"])
    parser.add_argument("--specify_segments", default=[], nargs="+")
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    extractor = WaymoDataExtractor(args.waymo_root, args.num_workers)
    extractor.extract_all(args.split, args.specify_segments, args.out_root)
