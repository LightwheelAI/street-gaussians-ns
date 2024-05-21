import argparse
import random
import numpy as np
import json
import json  
import open3d as o3d
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import os

def get_box_corners(center, dimensions, orientation):
    
    # 解包中心坐标、维度和四元数
    cx, cy, cz = center
    length, width, height = dimensions
    q = orientation

    # 生成正交的包围盒顶点集
    dx  = length / 2.0
    dy = width / 2.0
    dz = height / 2.0

    corners = np.array(
        [
            [dx, dy, dz],
            [-dx, dy, dz],
            [-dx, -dy, dz],
            [dx, -dy, dz],
            [dx, dy, -dz],
            [-dx, dy, -dz],
            [-dx, -dy, -dz],
            [dx, -dy, -dz],
        ]
    )

    # 使用四元数创建旋转并应用到顶点集
    rotation = R.from_quat([q[1], q[2], q[3], q[0]])  # 注意quaternion顺序为[x, y, z, w]
    rotated_corners = rotation.apply(corners)

    # 将局部坐标添加到中心点坐标上得到世界坐标
    world_corners = rotated_corners + center

    return world_corners

def undistort_nearest(cv_image, k, d,fisheye = True):

    if fisheye:
       mapx, mapy = cv2.fisheye.initUndistortRectifyMap(k, d, None, k, (cv_image.shape[1], cv_image.shape[0]), cv2.CV_32FC1)
    else:
       mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (cv_image.shape[1], cv_image.shape[0]), cv2.CV_32FC1)

    cv_image_undistorted = cv2.remap(cv_image, mapx, mapy, cv2.INTER_NEAREST)

    return cv_image_undistorted

def extract_value_between(string, start_char, end_char):
    start_index = string.find(start_char) + len(start_char)
    end_index = string.find(end_char, start_index)
    if(end_char ==""):
        end_index = len(string)
    if start_index != -1 :
        return string[start_index:end_index]
    else:
        return None
    

if __name__ == "__main__":
    import argparse    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="/home/ubuntu/project/nerf/nerfstudio/data/waymo")
    parser.add_argument("--meta_file", default="transform.json")
    parser.add_argument("--main_lidar_in_transforms", default="lidar_TOP")
    parser.add_argument("--world_coordinate", action="store_true")
    parser.add_argument("--annotation_file", type=str, default="annotation.json")

    args = parser.parse_args()
    root_path = args.root_path+"/"
    #nuscenes to opencv
    transform1 = np.array(
        [
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    lidars_folder = root_path+"lidars/"
    sensor_path = root_path+args.meta_file
    annotation_path = root_path + args.annotation_file
    moving_gids = []
    with open(sensor_path, "r") as f:
        data = json.load(f)
    with open(annotation_path, "r") as f:
        annotation_data = json.load(f)
    annotation_frames = annotation_data["frames"]
    c2w0 = np.array(data["frames"][0]["transform_matrix"])

    T0 = (c2w0[:3,3])*0.98
    T0_OPENCV = np.array([T0[1], T0[0],-1*T0[2]])
        
    i = 1
    #前向第一帧id，后向第一帧id
    offsets= [0,594,792]
    #lidar 与图像的offset
    lidar_offset = 0
    pic_num =198
    frames = data["frames"]
    lidar_frames = data["lidar_frames"]
    
    obj_pcd = {}

    for lidar_frame in tqdm(lidar_frames):
        # if (lidar_frame["lidar"] == args.main_lidar_in_transforms ):
        found_dict =  [d for d in frames if str(d.get('timestamp')) in str(lidar_frame["timestamp"])]
        annotation_found_dict = next((d for d in annotation_frames if str(d.get("timestamp")) in str(lidar_frame["timestamp"])), None)
        
        if annotation_found_dict is not None:
            obbs = []
            for object in annotation_found_dict["objects"]:
                if (object["is_moving"] or object["gid"] in moving_gids) and object["type"] == 'car':
                    # if args.track_moving:
                    #     moving_gids.append(object["gid"])
                    if object["gid"] not in obj_pcd:
                        obj_pcd[object["gid"]] = {
                            'xyz': [],
                            'rgb': [],
                        }
                    q = object["rotation"]
                    rotation_matrix = R.from_quat([q[1], q[2], q[3], q[0]])
                    obj = {
                        'gid': object["gid"],
                        'translation': object["translation"],
                        'size': object["size"],
                        'rotation': rotation_matrix.as_matrix(),
                    }
                    translation = object["translation"]
                    lwh = object["size"]
                    rotation = object["rotation"]
                    world_corners = get_box_corners(translation, lwh, rotation)
                    world_corners[:] -= T0_OPENCV
                    obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(world_corners))
                    scale_x = 1.1
                    scale_y = 1.1
                    scale_z = 1.1
                    extents = np.array(obb.extent) * np.array([scale_x, scale_y, scale_z]) # 更新边界长度
                    obb = o3d.geometry.OrientedBoundingBox(obb.center, obb.R, extents)
                    obj['obb'] = obb
                    obbs.append(obj)
                    # print("OBB的中心点:", obb.center)
                    # print("OBB的范围:", obb.extent)
                    # print("OBB的旋转矩阵:", obb.R)
        else:
            continue

        l2w = np.array(lidar_frame["transform_matrix"]) 
        # opencv2waymo = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        #  extrinsic[:3, :3] = extrinsic[:3, :3] @ opencv2waymo
        l2w[0:3, 1:3] *= -1       
        l2w = l2w[np.array([1, 0, 2, 3]), :]
        l2w[2, :] *= -1                   
        l2w[:3,3] -= T0
        l2w[2, :] *= -1
        l2w = l2w[np.array([1, 0, 2, 3]), :]
        l2w[0:3, 1:3] *= -1
        
        #读取pcd
        file_path = root_path + lidar_frame["file_path"]
        # points = np.fromfile(file=file_path, dtype=np.float32, count=-1).reshape([-1,5])[:,0:3]
        pcd_data = o3d.io.read_point_cloud(file_path)
        points = np.array(pcd_data.points)
        indices = points[:, 2] > -2
        points = points[indices]
        nan_rows = np.isnan(points).any(axis=1)

        # 使用布尔索引删除包含NaN值的行
        points = points[~nan_rows]
        # 将每个点表示为齐次坐标 (x, y, z, 1)
        homogeneous_positions = np.hstack([points , np.ones((points.shape[0], 1))])
        transformed_positions = np.dot(l2w, homogeneous_positions.T).T[:,:3]
        for frame in found_dict:
            obj_pcds = {}
            if (len(obbs) > 0):
                pcds = point_cloud = o3d.geometry.PointCloud()
                pcds.points = o3d.utility.Vector3dVector(transformed_positions)
                for obj in obbs:
                    obb = obj['obb']
                    inliers_indices = obb.get_point_indices_within_bounding_box(pcds.points)
                    inliers_pcd =  pcds.select_by_index(inliers_indices, invert=False) # select inside points = cropped
                    # __import__('ipdb').set_trace()
                    # if obj['gid'] not in obj_pcds:
                    obj_pcds[obj['gid']] = np.array(inliers_pcd.points)
                    # else:
                    #     obj_pcds[obj['gid']] = np.concatenate((obj_pcds[obj['gid']], np.array(inliers_pcd.points)), axis=0)
                    # outliers_pcd =  pcds.select_by_index(inliers_indices, invert=True) #select outside points
                    # pcds = outliers_pcd
                # transformed_positions = np.array(inliers_pcd.points)
                
            rgb=cv2.imread(root_path + frame["file_path"])
            c2w = np.array(frame["transform_matrix"])
            c2w[:3,3] -= T0
            c2w[2, :] *= -1
            c2w = c2w[np.array([1, 0, 2, 3]), :]
            c2w[0:3, 1:3] *= -1

            w2c = np.linalg.inv(c2w)
            h=frame['h']
            w=frame['w']
            fl_x=frame['fl_x']
            fl_y=frame['fl_y']
            cx=frame['cx']
            cy=frame['cy']
            ##去畸变
            if (frame['camera_model']=="OPENCV"):
                k = np.asarray([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])
                d = np.asarray([frame['k1'],frame['k2'],frame['p1'],frame['p2']])
                rgb = undistort_nearest(rgb, k, d,False)
            elif (frame['camera_model']=="OPENCV_FISHEYE"):
                k = np.asarray([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])
                d = np.asarray([frame['k1'],frame['k2'],frame['k3'],frame['k4']])
                rgb = undistort_nearest(rgb, k, d, True)                       
            
            intrinsic_matrix=np.array([[fl_x,0,cx,0],
                                    [0,fl_y,cy,0],
                                    [0,0,1,0],
                                    [0,0,0,1]])  
            
            # 提取结果中的前三列，去除齐次坐标
            # final_positions = transformed_positions[:, :3]-np.array([l2w[0,3],l2w[1,3],l2w[2,3]])
            # print("final_max_X:" , np.max(final_positions[:,2]))
            obj_ixd = 0
            for gid, pts in obj_pcds.items():
                assert obbs[obj_ixd]['gid'] == gid
                obj = obbs[obj_ixd]
                t = obj['translation']
                rot = obj['rotation']
                o2w = np.eye(4)
                o2w[:3,:3] = rot
                o2w[:3,3] = t
                w2o = np.linalg.inv(o2w)
                for pt in pts:
                    if abs(pt[0]) >100000:
                        continue
                    m_1= np.array([pt[0],pt[1],pt[2],1])
                    #print(m_1)
                    # 射影并只有在最后一步转换为整数
                    uv_homogeneous = intrinsic_matrix @ w2c @ m_1
                    u, v = (uv_homogeneous[:2] / uv_homogeneous[2]).astype(int)
                    #print (u,v)
                    # 检查坐标是否在图像的有效范围内
                    if 0 <= u < w and 0 <= v < h and uv_homogeneous[2]>0:
                        #rgb_point = rgb[v, u]
                        rgb_point = rgb[v, u] / 255.
                        error = random.uniform(0,1)
                        # 输出点信息和一些随机的值
                        m_1[:3] += T0_OPENCV
                        pt_obj = w2o@m_1
                        pt_obj = pt_obj[:3] / pt_obj[3]
                        obj_pcd[gid]['xyz'].append(pt_obj)
                        obj_pcd[gid]['rgb'].append(rgb_point)
                        i += 1
                obj_ixd += 1
    save_path = root_path + f"aggregate_lidar/dynamic_objects/"
    if not os.path.exists(root_path + f"aggregate_lidar/"):
        os.mkdir(root_path + f"aggregate_lidar/")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for gid, pcd in obj_pcd.items():
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(pcd['xyz']).astype(np.float32))
        point_cloud.colors = o3d.utility.Vector3dVector(np.array(pcd['rgb']).astype(np.float32))
        o3d.io.write_point_cloud(str(save_path + f"{gid}.ply"), point_cloud)