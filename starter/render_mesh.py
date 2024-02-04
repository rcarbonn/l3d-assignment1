"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
import imageio

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh, load_obj_mesh


def get_color_interpolation(color1, color2, mesh_pts):
    zmin = mesh_pts[:, 2].min()
    zmax = mesh_pts[:, 2].max()
    color1 = np.array(color1)
    color2 = np.array(color2)
    alpha = (mesh_pts[:, 2] - zmin) / (zmax - zmin)
    alpha = alpha.reshape(-1, 1)
    color = alpha * color2 + (1 - alpha) * color1
    return color

def render_n_views(
    obj_path="data/cow.obj", n=1,
    image_size=256, color=[0.7, 0.7, 1], device=None,
    **kwargs
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(obj_path)
    vertices = vertices.unsqueeze(0).repeat(n,1,1)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0).repeat(n,1,1)  # (N_f, 3) -> (1, N_f, 3)

    if kwargs.get("color_texture", False):
        color = get_color_interpolation([1.0,0.0,0.0], [0.0,0.0,1.0], vertices[0].cpu().numpy())

    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color, dtype=torch.float)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Prepare the cameras:
    Rs, Ts = pytorch3d.renderer.look_at_view_transform(dist=kwargs.get("dist", [-3.0]*n),
                                                        elev=kwargs.get("elev", [0.0]*n),
                                                        azim=kwargs.get("azim",[0.0]*n),
                                                        up=((0,1,0),), device=device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=Rs, T=Ts, fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = list(rend.cpu().numpy()[..., :3])  # (B, H, W, 4) -> (H, W, 3)
    rend = [(im*255).astype(np.uint8) for im in rend]
    # The .cpu moves the tensor to GPU (if needed).
    return rend


def render_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend


def make_gif(rendered_images, gif_path, fps=15):
    print("Creating GIF...")
    imageio.mimsave(gif_path, rendered_images, duration=1000/fps, loop=0)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--obj_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/cow_render.jpg")
    parser.add_argument("--save_path", type=str, default="results/cow_mesh.gif")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=5)

    args = parser.parse_args()
    # image = render_cow(cow_path=args.cow_path, image_size=args.image_size)
    # plt.imsave(args.output_path, image)
    n_views = 25
    n_view_config = {
        "dist": [-3.0]*n_views,
        "azim": list(np.linspace(0,360,n_views)),
        "color_texture": True
    }
    images = render_n_views(obj_path=args.obj_path, n=n_views, image_size=args.image_size, **n_view_config)
    make_gif(images, args.save_path, args.fps)
