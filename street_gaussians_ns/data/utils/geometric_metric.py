import numpy as np
import open3d as o3d
from street_gaussians_ns.sgn_splatfacto import SplatfactoModel

CD_UNIT = 1e-4


def cv2gl(c2w: np.array):
    applied_transform = np.eye(4)
    applied_transform = applied_transform[np.array([1, 0, 2, 3]), :]
    applied_transform[2, :] *= -1
    return np.matmul(applied_transform, c2w)


def gl2cv(c2w: np.array):
    return cv2gl(c2w)


def write_points_pcd(points, filename):
    """Writes points to a PCD file."""
    import open3d as o3d

    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        pcd = points
    o3d.io.write_point_cloud(str(filename), pcd)


def read_pcd_file(pcd_path, ignore_nan=True, filter_ego=True, return_pcd=False):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    # Convert to numpy array
    points = np.asarray(pcd.points)
    if ignore_nan:
        # print('@filter Nan')
        points = points[~np.isnan(points).any(axis=1)]
    if filter_ego:
        self_mask = lambda points: ~(
            ((x := points[:, 0]) < 3)
            & (x > -1)
            & ((y := np.abs(points[:, 1])) < 1)
            & ((z := points[:, 2]) < 2)
            & (z > -1)
        )

        points = points[self_mask(points)]
    if return_pcd:
        return np2pcd(points)
    return points

def np2pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def calc_chamfer_distance(pred, gt):
    assert isinstance(gt, np.ndarray) and isinstance(pred, np.ndarray)
    assert gt.shape[1] == 3 and pred.shape[1] == 3
    gt_pcd = np2pcd(gt)
    pred_pcd = np2pcd(pred)
    dists1 = pred_pcd.compute_point_cloud_distance(gt_pcd)
    dists1 = np.asarray(dists1).mean()
    dists2 = gt_pcd.compute_point_cloud_distance(pred_pcd)
    dists2 = np.asarray(dists2).mean()
    # dists = 0.5 * (dists1 + dists2) / CD_UNIT
    return dists1 / CD_UNIT, dists2 / CD_UNIT


def evaluate_lidar_geometric(aggregate_lidar_path, pipeline):
    # TODO
    ## check lidar path
    # if self.config.aggregate_lidar_path is None:
    #     aggregate_lidar_path = self.config.data / "aggregate_lidar"/ "output.ply"
    # else:
    if not isinstance(pipeline.model, SplatfactoModel):
        return {}
    assert aggregate_lidar_path is not None and aggregate_lidar_path.exists()
    train_dataparser_outputs = pipeline.datamanager.train_dataparser_outputs
    assert "applied_translation_in_colmap" in train_dataparser_outputs.metadata
    # transform lidar to camera space
    translation = train_dataparser_outputs.metadata["applied_translation_in_colmap"]
    # TODO(zz): fix transform due to colmap dataparser assume_colmap_world_coordinate_convention change.
    translation = gl2cv(np.append(translation, 1))[:3]
    dataparser_scale = train_dataparser_outputs.dataparser_scale
    dataparser_transform = train_dataparser_outputs.dataparser_transform.numpy()
    lidar_pcd = read_pcd_file(aggregate_lidar_path.as_posix(), return_pcd=False)
    lidar_pcd += translation  # w2colmap
    lidar_pcd = lidar_pcd @ dataparser_transform[:3, :3].T + dataparser_transform[:3, 3]  # colmap2gl
    lidar_pcd *= dataparser_scale

    ## check is gs model
    gs_pts = pipeline.model.means.detach().cpu().numpy()
    # calc chamfer distance
    # import time;tic=time.time()
    dist1, dist2 = calc_chamfer_distance(gs_pts, lidar_pcd)
    # print(f'calc chamfer distance time:{time.time()-tic}')
    return {"lidar_chamfer_distance_1": dist1, "lidar_chamfer_distance_2": dist2, 'lidar_chamfer_distance_avg':(dist1+dist2)/2}
