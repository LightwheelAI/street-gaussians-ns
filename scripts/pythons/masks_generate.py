"""

Installation:
    pip install git+https://github.com/facebookresearch/segment-anything.git
    pip install opencv-python pycocotools matplotlib onnxruntime onnx
    wget https://github.com/facebookresearch/segment-anything#model-checkpoints
"""
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from tqdm import tqdm
import json
import os


def get_box_corners(center, dimensions, orientation):
    # 解包中心坐标、维度和四元数
    cx, cy, cz = center
    length, width, height = dimensions
    q = orientation

    # 生成正交的包围盒顶点集
    dx = length / 2.0
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="")

    parser.add_argument("--meta_file", type=str, default="transform.json")
    parser.add_argument("--annotation_file", type=str, default="annotation.json")
    parser.add_argument("--dilation_radius", type=int, default=15, help=">3 odd number")
    parser.add_argument("--nuscenes", action="store_true")
    parser.add_argument("--seen_cameras", action="store_true")
    parser.add_argument("--draw_boxes", action="store_true")
    parser.add_argument("--track_moving", action="store_true")

    args = parser.parse_args()
    root_path = args.root_path + "/"
    transform_path = root_path + args.meta_file
    annotation_path = root_path + args.annotation_file
    dilation_radius = args.dilation_radius
    device = "cuda"
    model_type = "default"

    ## nuscenes
    transform1 = np.array(
        [
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    # 读取 transforms.json 文件
    with open(transform_path, "r") as f:
        transform_data = json.load(f)

    with open(annotation_path, "r") as f:
        annotation_data = json.load(f)

    frames = transform_data["frames"]
    annotation_frames = annotation_data["frames"]
    moving_gids = []
    for frame in tqdm(frames):
        # 获取 file_path 值
        file_path = frame["file_path"]
        c2w = np.array(frame["transform_matrix"])
        ##nerfstudio to opencv
        c2w[2, :] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[0:3, 1:3] *= -1
        if args.nuscenes:
            c2w = np.linalg.inv(transform1) @ c2w

        h = frame["h"]
        w = frame["w"]
        fl_x = frame["fl_x"]
        fl_y = frame["fl_y"]
        cx = frame["cx"]
        cy = frame["cy"]
        camera_matrix = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype="float32")
        intrinsic_matrix = np.array([[fl_x, 0, cx, 0], [0, fl_y, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        camera = frame["camera"]
        camera_model = frame["camera_model"]
        if camera_model == "OPENCV_FISHEYE":
            dist_coeffs = np.array([frame["k1"], frame["k2"], frame["k3"], frame["k4"]], dtype="float32")
        elif camera_model == "OPENCV":
            dist_coeffs = np.array([frame["k1"], frame["k2"], frame["p1"], frame["p2"]], dtype="float32")
        timestamp = frame["timestamp"]
        image_path = root_path + file_path
        mask_path = image_path.replace("images", "masks")
        mask_box_path = image_path.replace("images", "masks_box2d")
        mask_path = os.path.splitext(mask_path)[0]
        mask_path = mask_path + ".png"
        mask_box_path = os.path.splitext(mask_box_path)[0]
        mask_box_path = mask_box_path + ".png"
        directory = os.path.dirname(mask_path)
        # 判断文件夹是否存在
        if not os.path.exists(directory):
            # 如果文件夹不存在，则创建文件夹
            os.makedirs(directory)
        directory = os.path.dirname(mask_box_path)
        # 判断文件夹是否存在
        if not os.path.exists(directory):
            # 如果文件夹不存在，则创建文件夹
            os.makedirs(directory)
        # 设置默认图像大小和颜色（255表示白色的灰度值）
        width, height = w, h
        color = 255

        # 创建一个新的白色灰度图像
        image_white = Image.new("L", (width, height), color)
        found_dict = next((d for d in annotation_frames if str(timestamp) in str(d.get("timestamp"))), None)

        if found_dict is not None:
            boxes = []

            for object in found_dict["objects"]:
                if args.seen_cameras and camera not in object["seen_cameras"]:
                    continue
                if object["is_moving"] or object["gid"] in moving_gids:
                    if args.track_moving:
                        moving_gids.append(object["gid"])
                    translation = object["translation"]
                    lwh = object["size"]
                    if args.nuscenes:
                        lwh[0], lwh[1] = lwh[1], lwh[0]
                    rotation = object["rotation"]
                    world_corners = get_box_corners(translation, lwh, rotation)
                    w2c = np.linalg.inv(c2w)
                    rvec = w2c[:3, :3]
                    tvec = w2c[:3, 3]
                    umin = w
                    vmin = h
                    umax = 0
                    vmax = 0
                    for m in world_corners:
                        m_1 = np.array([m[0], m[1], m[2], 1])
                        # 射影并只有在最后一步转换为整数
                        points_3D = np.array([[m[0], m[1], m[2]]], dtype="float32").reshape(-1, 1, 3)
                        uv_homogeneous = intrinsic_matrix @ w2c @ m_1
                        if uv_homogeneous[2] > 0:
                            if camera_model == "OPENCV_FISHEYE":
                                points_2D, _ = cv2.fisheye.projectPoints(
                                    points_3D,
                                    cv2.Rodrigues(rvec)[0],
                                    np.ascontiguousarray(tvec),
                                    camera_matrix,
                                    dist_coeffs,
                                )
                                u = int(points_2D[0][0][0])
                                v = int(points_2D[0][0][1])

                            # elif(camera_model=="OPENCV"):
                            #     points_2D, _ = cv2.projectPoints(points_3D,cv2.Rodrigues(rvec)[0], np.ascontiguousarray(tvec), camera_matrix, dist_coeffs)
                            #     u = points_2D[0][0][0]
                            #     v = points_2D[0][0][1]
                            else:
                                u, v = (uv_homogeneous[:2] / uv_homogeneous[2]).astype(int)

                            umax = max(umax, u)
                            vmax = max(vmax, v)
                            umin = min(umin, u)
                            vmin = min(vmin, v)
                    if umin == w or vmin == h or umax == 0 or vmax == 0:
                        continue
                    # box = [umin-30,vmin-30,umax+30,vmax+30]
                    umin = max (umin ,0)
                    vmin = max (vmin ,0)
                    umax = min (umax ,w - 1)
                    vmax = min (vmax ,h - 1)
                    
                    box = [
                        max(umin - int((umax - umin) / 10.0), 0),
                        max(vmin - int((vmax - vmin) / 10.0), 0),
                        min(umax + int((umax - umin) / 10.0), w - 1),
                        min(vmax + int((vmax - vmin) / 10.0), h - 1),
                    ]
                    boxes.append(box)

            if len(boxes) > 0:
                if args.draw_boxes:
                    image = Image.open(image_path)
                    # 创建ImageDraw对象
                    draw = ImageDraw.Draw(image)
                    # 使用ImageDraw在图片上画出每个box框
                    for box in boxes:
                        if 0 <= box[0] < w and 0 <= box[1] < h and 0 <= box[2] < w and 0 <= box[3] < h:
                            draw.rectangle(box, outline="blue", width=2)  # 你可以根据需要修改颜色和线宽
                    image.save(mask_box_path)

                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                A = np.ones((h, w)) * 255
                # print(masks)
                for box in boxes:
                    x_min, y_min, x_max, y_max = box
                    A[y_min:y_max, x_min:x_max] = 0
                    y_min += int(0.5 * (y_max - y_min))
                    # 截取image中的box区域
                    roi = image[y_min:y_max, x_min:x_max]

                    # 通过比较数组进行操作，创建掩模
                    mask = (roi < [96, 96, 96]).all(axis=2)

                    # 更新到纯白图像A的对应位置，将符合条件的位置设为黑色
                    A[y_min:y_max, x_min:x_max][mask] = 1

                images = []
                # masks = torch.any(masks, dim=0, keepdim=True)

                # image_data = np.where(masks[0], 0, 255)
                # image_data[A==0] = 0
                # 将数据类型转换为uint8，以满足Pillow库的图像要求
                image_data = A.astype(np.uint8)
                image = Image.fromarray(image_data)
                image.save(mask_path)
            else:
                image_white.save(mask_path)
        else:
            image_white.save(mask_path)