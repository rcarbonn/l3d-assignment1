"""
Usage:
    python -m starter.dolly_zoom --num_frames 10
"""

import argparse

import imageio
import numpy as np
import pytorch3d
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from starter.utils import get_device, get_mesh_renderer


def dolly_zoom(
    image_size=256,
    num_frames=10,
    duration=3,
    device=None,
    output_file="output/dolly.gif",
):
    if device is None:
        device = get_device()

    mesh = pytorch3d.io.load_objs_as_meshes(["data/cow_on_plane.obj"])
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    fovs = torch.linspace(5, 120, num_frames)
    # f_ = 1/np.tan(120*np.pi/360)

    renders = []
    K = 1.5*np.tan(60*np.pi/180)
    for fov in tqdm(fovs[:-10]):
        f = 1/np.tan(fov*np.pi/360)
        d = K/(np.tan(fov*np.pi/360))
        print(d)
        distance=3
        T = [[0, 0, d]]  # TODO: Change this.
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"fov: {fovs[i]:.2f}", fill=(255, 0, 0))
        images.append(np.array(image))
    fps = (num_frames/duration)
    imageio.mimsave(output_file, images, duration=1000/fps, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--output_file", type=str, default="results/dolly.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    dolly_zoom(
        image_size=args.image_size,
        num_frames=args.num_frames,
        duration=args.duration,
        output_file=args.output_file,
    )
