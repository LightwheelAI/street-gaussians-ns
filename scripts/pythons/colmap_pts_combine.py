from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
from pathlib import Path
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src1", type=Path, required=True, help="path to first points3D.bin")
    parser.add_argument("--src2", type=Path, required=True, help="path to second points3D.bin")
    parser.add_argument("--dst", type=Path, required=True)
    return parser.parse_args()


def read(path):
    if path.suffix == ".bin":
        return colmap_utils.read_points3D_binary(path)
    elif path.suffix == ".txt":
        return colmap_utils.read_points3D_text(path)
    else:
        raise ValueError(f"Unknown file extension {path.suffix}")


if __name__ == "__main__":
    args = parse_args()
    colmap_points1 = read(args.src1)
    colmap_points2 = read(args.src2)
    offset = max([i for i in colmap_points1.keys()]) + 1
    # combine 2 dicts
    for k, v in tqdm(colmap_points2.items()):
        assert k + offset not in colmap_points1
        colmap_points1[k + offset] = v._replace(id=k + offset)

    colmap_utils.write_points3D_binary(colmap_points1, args.dst)