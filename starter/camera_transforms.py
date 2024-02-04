"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np

from starter.utils import get_device, get_mesh_renderer


def render_cow(
    cow_path="data/cow_with_axis.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)

    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    T = R_relative @ torch.tensor([0.0, 0.0, 3.0]) + T_relative
    print(T)
    # since the pytorch3d internal uses Point= point@R+t instead of using Point=R @ point+t,
    # we need to add R.t() to compensate that.
    renderer = get_mesh_renderer(image_size=image_size)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.t().unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow_with_axis.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="results/transform_cow.jpg")
    args = parser.parse_args()

    # rotation about z by 90 degrees
    R1 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    # translation along z by +2
    T2 = [0, 0, 3.0]
    # translation along x by +0.5, y by -0.4, z by -0.1
    T3 = [0.5, -0.4, -0.1]
    # rotation about y by -90 degrees
    R4 = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    T4 = [3, 0, 3]

    plt.imsave("results/transform_cow_r1.jpg", render_cow(cow_path=args.cow_path, image_size=args.image_size, R_relative=R1))
    plt.imsave("results/transform_cow_t2.jpg", render_cow(cow_path=args.cow_path, image_size=args.image_size, T_relative=T2))
    plt.imsave("results/transform_cow_t3.jpg", render_cow(cow_path=args.cow_path, image_size=args.image_size, T_relative=T3))
    plt.imsave("results/transform_cow_t4.jpg", render_cow(cow_path=args.cow_path, image_size=args.image_size, R_relative=R4, T_relative=T4))
