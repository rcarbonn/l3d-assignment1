"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
import imageio

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image, make_gif


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def render_rgbd_point_cloud(n_views=1, **kwargs):
    d = load_rgbd_data()
    points1, rgb1 = unproject_depth_image(torch.from_numpy(d["rgb1"]), torch.from_numpy(d["mask1"]), torch.from_numpy(d["depth1"]), d["cameras1"])
    points2, rgb2 = unproject_depth_image(torch.from_numpy(d["rgb2"]), torch.from_numpy(d["mask2"]), torch.from_numpy(d["depth2"]), d["cameras2"])
    points3 = torch.vstack([points1, points2])
    rgb3 = torch.vstack([rgb1, rgb2])
    point_cloud1 = {"verts": points1, "rgb": rgb1}
    point_cloud2 = {"verts": points2, "rgb": rgb2}
    point_cloud3 = {"verts": points3, "rgb": rgb3}
    # n_view_config = {
    #     "dist": [-8.0]*n_views,
    #     "azim": list(np.linspace(0,360,n_views)),
    #     "color_texture": True,
    #     "up": ((0,-1,0),)
    # }
    imgs1 = render_point_cloud(point_cloud1, n=n_views, **kwargs)
    imgs2 = render_point_cloud(point_cloud2, n=n_views, **kwargs)
    imgs3 = render_point_cloud(point_cloud3, n=n_views, **kwargs)
    return imgs1, imgs2, imgs3


def render_point_cloud(
    point_cloud,
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
    n=1,
    **kwargs
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    # point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::]).to(device).unsqueeze(0).repeat(n,1,1)
    rgb = torch.Tensor(point_cloud["rgb"][::]).to(device).unsqueeze(0).repeat(n,1,1)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)

    # Prepare the cameras:
    Rs, Ts = pytorch3d.renderer.look_at_view_transform(dist=kwargs.get("dist", [-8.0]*n),
                                                        elev=kwargs.get("elev", [0.0]*n),
                                                        azim=kwargs.get("azim",[0.0]*n),
                                                        up=kwargs.get("up",((0,-1,0),)), device=device)
    
    # R, T = pytorch3d.renderer.look_at_view_transform(-8, 0, 0, up=((0, -1, 0),))

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=Rs, T=Ts, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = list(rend.cpu().numpy()[..., :3])  # (B, H, W, 4) -> (H, W, 3)
    rend = [(im*255).astype(np.uint8) for im in rend]
    return rend


def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


def render_sphere(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sin(Theta) * torch.cos(Phi)
    y = torch.cos(Theta)
    z = torch.sin(Theta) * torch.sin(Phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()


def render_parametric_surface(surface="torus", image_size=256, num_samples=200, device=None, n_views=1, **kwargs):

    if surface=="torus":
        phi = torch.linspace(0, 2 * np.pi, num_samples*10)
        theta = torch.linspace(0, 2* np.pi, num_samples)
        # Densely sample phi and theta on a grid
        Phi, Theta = torch.meshgrid(phi, theta)
        R = 2
        r = 0.5

        x = (R+r*torch.cos(Theta)) * torch.cos(Phi)
        y = (R+r*torch.cos(Theta)) * torch.sin(Phi)
        z = r*torch.sin(Theta)
    
    elif surface=="hyperboloid":
        u = torch.linspace(0, 2*np.pi, num_samples)
        v = torch.linspace(-3, 3, num_samples)
        u, v = torch.meshgrid(u, v)

        x = torch.cosh(v)*torch.cos(u)
        y = torch.sinh(v)*torch.sin(u)
        z = torch.sinh(v)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())
    print(points.shape)
    # n_view_config = {
    #     "dist": [-30.0]*n_views,
    #     "azim": list(np.linspace(0,360,n_views)),
    #     "color_texture": True,
    #     "up": ((0,1,0),)
    # }
    imgs = render_point_cloud({"verts": points, "rgb": color}, image_size=image_size, n=n_views, **kwargs)
    return imgs



# def render_torus(image_size=256, num_samples=200, device=None, n_views=1):
#     """
#     Renders a torus using parametric sampling. Samples num_samples ** 2 points.
#     """

#     if device is None:
#         device = get_device()

#     phi = torch.linspace(0, 2 * np.pi, num_samples*10)
#     theta = torch.linspace(0, 2* np.pi, num_samples)
#     # Densely sample phi and theta on a grid
#     Phi, Theta = torch.meshgrid(phi, theta)
#     R = 2
#     r = 0.5

#     x = (R+r*torch.cos(Theta)) * torch.cos(Phi)
#     y = (R+r*torch.cos(Theta)) * torch.sin(Phi)
#     z = r*torch.sin(Theta)

#     # Parametric form of hyperboloid
#     # u = torch.linspace(0, 2*np.pi, num_samples)
#     # v = torch.linspace(-3, 3, num_samples)
#     # u, v = torch.meshgrid(u, v)

#     # x = torch.cosh(v)*torch.cos(u)
#     # y = torch.sinh(v)*torch.sin(u)
#     # z = torch.sinh(v)

#     points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
#     color = (points - points.min()) / (points.max() - points.min())
#     print(points.shape)
#     n_view_config = {
#         "dist": [-30.0]*n_views,
#         "azim": list(np.linspace(0,360,n_views)),
#         "color_texture": True,
#         "up": ((0,1,0),)
#     }
#     imgs = render_point_cloud({"verts": points, "rgb": color}, n=n_views, **n_view_config)
#     return imgs

def render_implicit_mesh(obj="torus", image_size=256, voxel_size=64, device=None, n_views=1, **kwargs):

    if device is None:
        device = get_device()
    min_value = -3.0
    max_value = 3.0
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    if obj=="torus":
        R = 2
        r = 0.5
        voxels = (X ** 2 + Y ** 2 + Z ** 2 + R**2 - r**2)**2 - 4*R**2*(X**2 + Y**2)
    elif obj=="hyperboloid":
        voxels = X ** 2 + Y ** 2 - Z ** 2 -1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    vertices = vertices.unsqueeze(0).repeat(n_views,1,1)
    faces = faces.unsqueeze(0).repeat(n_views,1,1)
    textures = pytorch3d.renderer.TexturesVertex(vertices)

    mesh = pytorch3d.structures.Meshes(vertices, faces, textures=textures).to(
        device
    )
    # n_view_config = {
    #     "dist": [10.0]*n_views,
    #     "elev": [-60.0]*n_views,
    #     "azim": list(np.linspace(0,360,n_views)),
    #     "color_texture": True
    # }

    Rs, Ts = pytorch3d.renderer.look_at_view_transform(dist=kwargs.get("dist", [10.0]*n_views),
                                                        elev=kwargs.get("elev", [0.0]*n_views),
                                                        azim=kwargs.get("azim",[0.0]*n_views),
                                                        up=((0,1,0),), device=device)

    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    # R, T = pytorch3d.renderer.look_at_view_transform(dist=10, elev=-60, azim=0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=Rs, T=Ts, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = list(rend[..., :3].detach().cpu().numpy().clip(0, 1))
    rend = [(im*255).astype(np.uint8) for im in rend]
    return rend


# def render_torus_mesh(image_size=256, voxel_size=64, device=None, n_views=1, **kwargs):
#     if device is None:
#         device = get_device()
#     min_value = -3.0
#     max_value = 3.0
#     R = 2
#     r = 0.5
#     X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
#     voxels = (X ** 2 + Y ** 2 + Z ** 2 + R**2 - r**2)**2 - 4*R**2*(X**2 + Y**2)
#     # voxels for hyperboloid
#     # voxels = X ** 2 + Y ** 2 - Z ** 2 -1
#     vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
#     vertices = torch.tensor(vertices).float()
#     faces = torch.tensor(faces.astype(int))
#     # Vertex coordinates are indexed by array position, so we need to
#     # renormalize the coordinate system.
#     vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
#     textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
#     vertices = vertices.unsqueeze(0).repeat(n_views,1,1)
#     faces = faces.unsqueeze(0).repeat(n_views,1,1)
#     textures = pytorch3d.renderer.TexturesVertex(vertices)

#     mesh = pytorch3d.structures.Meshes(vertices, faces, textures=textures).to(
#         device
#     )
#     n_view_config = {
#         "dist": [10.0]*n_views,
#         "elev": [-60.0]*n_views,
#         "azim": list(np.linspace(0,360,n_views)),
#         "color_texture": True
#     }

#     Rs, Ts = pytorch3d.renderer.look_at_view_transform(dist=n_view_config.get("dist", [10.0]*n_views),
#                                                         elev=n_view_config.get("elev", [0.0]*n_views),
#                                                         azim=n_view_config.get("azim",[0.0]*n_views),
#                                                         up=((0,1,0),), device=device)

#     lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
#     renderer = get_mesh_renderer(image_size=image_size, device=device)
#     # R, T = pytorch3d.renderer.look_at_view_transform(dist=10, elev=-60, azim=0)
#     cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=Rs, T=Ts, device=device)
#     rend = renderer(mesh, cameras=cameras, lights=lights)
#     rend = list(rend[..., :3].detach().cpu().numpy().clip(0, 1))
#     rend = [(im*255).astype(np.uint8) for im in rend]
#     return rend


def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric", "implicit", "rgbd"],
    )
    parser.add_argument("--output_path", type=str, default="results/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()
    if args.render == "point_cloud":
        image = render_bridge(image_size=args.image_size)
    elif args.render == "parametric":
        # image = render_sphere(image_size=args.image_size, num_samples=args.num_samples)
        image = render_torus(image_size=args.image_size, num_samples=args.num_samples, n_views=10)
        make_gif(image, args.output_path, args.fps)
    elif args.render == "implicit":
        # image = render_sphere_mesh(image_size=args.image_size)
        image = render_torus_mesh(image_size=args.image_size, n_views=10)
        make_gif(image, args.output_path, args.fps)
    elif args.render == "rgbd":
        image1,image2,image3 = render_rgbd_point_cloud(n_views=10)
        fname = args.output_path.split(".")[0]
        make_gif(image1, fname+"1.gif", args.fps)
        make_gif(image2, fname+"2.gif", args.fps)
        make_gif(image3, fname+"3.gif", args.fps)
        # plt.imsave(fname+"1.jpg", image1)
        # plt.imsave(fname+"2.jpg", image2)
    else:
        raise Exception("Did not understand {}".format(args.render))
    # plt.imsave(args.output_path, image)

