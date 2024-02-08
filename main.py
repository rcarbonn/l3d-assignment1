import numpy as np
import argparse 
import matplotlib.pyplot as plt

from starter.utils import make_gif
from starter.render_mesh import render_n_views
from starter.render_mesh import render_cow, render_textured_mesh
from starter.dolly_zoom import dolly_zoom
from starter.camera_transforms import render_cow as render_cow_transformed
from starter.render_generic import render_bridge, render_parametric_surface, render_implicit_mesh, render_rgbd_point_cloud
from starter.render_generic import sample_points_from_mesh

QUESTION_PARAMS = {
    "q1" : {
        "cow_path" : "data/cow.obj",
        "obj_path" : "data/cow.obj",
        "save_path" : "results/",
        "image_size" : 256,
        "fps" : 5,
        "duration" : 5,
        "n_view_config" : {
            "dist": [-3.0]*36,
            "elev": [0.0]*36,
            "azim": list(np.linspace(0,360,36)),
            "color_texture": False
        }
    },
    "q2" : {
        "obj_path1" : "data/tetrahedron.obj",
        "obj_path2" : "data/cube.obj",
        "save_path" : "results/",
        "image_size" : 256,
        "fps" : 5,
        "duration" : 5,
        "n_view_config" : {
            "dist": [-3.0]*36,
            "elev": [0.0]*36,
            "azim": list(np.linspace(0,360,36)),
            "color_texture": False
        }
    },
    "q3" : {
        "obj_path" : "data/cow.obj",
        "save_path" : "results/",
        "image_size" : 256,
        "fps" : 5,
        "duration" : 5,
        "n_view_config" : {
            "dist": [-3.0]*36,
            "elev": [0.0]*36,
            "azim": list(np.linspace(0,360,36)),
            "color_texture": True
        }
    },
    "q4" : {
        "obj_path" : "data/cow_with_axis.obj",
        "save_path" : "results/",
        "image_size" : 256
    },
    "q5" : {
        "render" : ["rgbd", "parametric", "implicit"],
        "save_path" : "results/",
        "image_size" : 256,
        "num_samples" : 500,
        "fps" : 5,
        "n_views" : 36,
        "rgbd" : {
            "n_view_config" : {
                "dist": [-8.0]*36,
                "azim": list(np.linspace(0,360,36)),
                "color_texture": True,
                "up": ((0,-1,0),)
            }
        },
        "parametric" : {
            "n_views" : 10,
            "torus" : {
                "n_view_config" : {
                    "dist": [-8.0]*10,
                    "azim": list(np.linspace(0,360,10)),
                    "color_texture": True,
                    "up": ((0,1,0),)
                }
            },

            "hyperboloid" : {
                "n_view_config" : {
                    "dist": [-30.0]*10,
                    "azim": list(np.linspace(0,360,10)),
                    "color_texture": True,
                    "up": ((0,1,0),)
                }
            }

        },
        "implicit" : {
            "torus" : {
                "voxel_size" : 64,
                "n_view_config" : {
                    "dist": [10.0]*36,
                    "elev": [-60.0]*36,
                    "azim": list(np.linspace(0,360,36)),
                    "color_texture": True
                }
            },
            "hyperboloid" : {
                "voxel_size" : 64,
                "n_view_config" : {
                    "dist": [10.0]*36,
                    "elev": [-60.0]*36,
                    "azim": list(np.linspace(0,360,36)),
                    "color_texture": True
                }
            }
        }
    },
    "q6" : {
        "obj_path" : "data/Tennis_Ball.obj",
        "save_path" : "results/",
        "image_size" : 256,
        "fps" : 5,
        "duration" : 5,
        "n_views" : 36,
        "n_view_config" : {
            "dist": [-0.5]*36,
            "elev": [0.0]*36,
            "azim": list(np.linspace(0,360,36)),
            "color_texture": True
        },
        "morph" : "scaled",
    },
    "q7" : {
        "cow_path" : "data/cow.obj",
        "obj_path" : "data/cow.obj",
        "save_path" : "results/",
        "image_size" : 256,
        "fps" : 5,
        "duration" : 5,
        "n_views" : 36,
        "num_samples" : 10,
        "n_view_config" : {
            "dist": [-3.0]*36,
            "elev": [0.0]*36,
            "azim": list(np.linspace(0,360,36)),
            "color_texture": False,
            "up": ((0,1,0),)
        }
    },
}


def main(args):

    config = QUESTION_PARAMS[args.question]

    if args.question == "q1":
        q1 = args.question
        # default image
        # image = render_cow(cow_path=config["cow_path"], image_size=config["image_size"])
        # 360 view gif
        images = render_n_views(obj_path=config["obj_path"], n=36, image_size=config["image_size"], **config["n_view_config"])
        save_path = config["save_path"] + q1 + ".1.gif"
        make_gif(images, save_path, config["fps"])

        # dolly zoom gif
        dolly_zoom(image_size = config["image_size"], num_frames=args.num_frames, duration=config["duration"], output_file=config["save_path"] + q1 + ".2.gif")
    
    elif args.question == "q2":
        q2 = args.question
        images1 = render_n_views(obj_path=config["obj_path1"], n=36, image_size=config["image_size"], **config["n_view_config"])
        images2 = render_n_views(obj_path=config["obj_path2"], n=36, image_size=config["image_size"], **config["n_view_config"])
        save_path1 = config["save_path"] + q2 + ".1.gif"
        save_path2 = config["save_path"] + q2 + ".2.gif"
        make_gif(images1, save_path1, config["fps"])
        make_gif(images2, save_path2, config["fps"])
    
    elif args.question == "q3":
        q3 = args.question
        images = render_n_views(obj_path=config["obj_path"], n=36, image_size=config["image_size"], **config["n_view_config"])
        save_path = config["save_path"] + q3 + ".gif"
        make_gif(images, save_path, config["fps"])
    
    elif args.question == "q4":
        q4 = args.question
        # rotation about z by 90 degrees
        R1 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        save_path1 = config["save_path"] + q4 + ".1.jpg"
        plt.imsave(save_path1, render_cow_transformed(cow_path=config["obj_path"], image_size=config["image_size"], R_relative=R1))
        # translation along z by +2
        T2 = [0, 0, 3.0]
        save_path2 = config["save_path"] + q4 + ".2.jpg"
        plt.imsave(save_path2, render_cow_transformed(cow_path=config["obj_path"], image_size=config["image_size"], T_relative=T2))
        # translation along x by +0.5, y by -0.4, z by -0.1
        T3 = [0.5, -0.4, -0.1]
        save_path3 = config["save_path"] + q4 + ".3.jpg"
        plt.imsave(save_path3, render_cow_transformed(cow_path=config["obj_path"], image_size=config["image_size"], T_relative=T3))
        # rotation about y by -90 degrees
        R4 = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
        T4 = [3, 0, 3]
        save_path4 = config["save_path"] + q4 + ".4.jpg"
        plt.imsave(save_path4, render_cow_transformed(cow_path=config["obj_path"], image_size=config["image_size"], R_relative=R4, T_relative=T4))
    
    elif args.question == "q5":
        q5 = args.question
        for rtype in config["render"]:
            if rtype == "point_cloud" and args.render == "point_cloud":
                image = render_bridge(image_size=config["image_size"])
                plt.imsave(config["save_path"] + q5 + "_bridge.jpg", image)
            elif rtype == "rgbd" and args.render == "rgbd":
                image1, image2, image3 = render_rgbd_point_cloud(n_views=config["n_views"], **config["rgbd"]["n_view_config"])
                save_path1 = config["save_path"] + q5 + ".1.1.gif"
                save_path2 = config["save_path"] + q5 + ".1.2.gif"
                save_path3 = config["save_path"] + q5 + ".1.3.gif"
                make_gif(image1, save_path1, config["fps"])
                make_gif(image2, save_path2, config["fps"])
                make_gif(image3, save_path3, config["fps"])
            elif rtype == "parametric" and args.render == "parametric":
                image = render_parametric_surface(surface="torus", image_size=config["image_size"], num_samples=config["num_samples"], n_views=config["parametric"]["n_views"], **config["parametric"]["torus"]["n_view_config"])
                make_gif(image, config["save_path"] + q5 + ".2_torus.gif", config["fps"])
                image = render_parametric_surface(surface="hyperboloid", image_size=config["image_size"], num_samples=config["num_samples"], n_views=config["parametric"]["n_views"], **config["parametric"]["hyperboloid"]["n_view_config"])
                make_gif(image, config["save_path"] + q5 + ".2_hyperboloid.gif", config["fps"])
            elif rtype == "implicit" and args.render == "implicit":
                image = render_implicit_mesh(obj="torus", image_size=config["image_size"], voxel_size=config["implicit"]["torus"]["voxel_size"], n_views=config["n_views"], **config["implicit"]["torus"]["n_view_config"])
                make_gif(image, config["save_path"] + q5 + ".3_torus.gif", config["fps"])
                image = render_implicit_mesh(obj="hyperboloid", image_size=config["image_size"], voxel_size=config["implicit"]["hyperboloid"]["voxel_size"], n_views=config["n_views"], **config["implicit"]["hyperboloid"]["n_view_config"])
                make_gif(image, config["save_path"] + q5 + ".3_hyperboloid.gif", config["fps"])
            else:
                continue
                raise Exception("Did not understand {}".format(rtype))
    
    elif args.question == "q6":
        q6 = args.question
        images = render_textured_mesh(obj_path=config["obj_path"], image_size=config["image_size"], n=config["n_views"], morph=config["morph"], **config["n_view_config"])
        save_path = config["save_path"] + q6 + ".1b.gif"
        make_gif(images, save_path, config["fps"])
        # config["morph"] = "color"
        # images = render_textured_mesh(obj_path=config["obj_path"], image_size=config["image_size"], n=config["n_views"], morph=config["morph"], **config["n_view_config"])
        # save_path = config["save_path"] + q6 + ".2b.gif"
        # make_gif(images, save_path, config["fps"])
    
    elif args.question == "q7":
        q7 = args.question
        imgs = sample_points_from_mesh(obj_path=config["obj_path"],n=config["n_views"], num_samples=config["num_samples"], image_size=config["image_size"], **config["n_view_config"])
        save_path = config["save_path"] + q7 + "_" + str(config["num_samples"]) + ".gif"
        make_gif(imgs, save_path, config["fps"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="q1", choices=["q1", "q2", "q3", "q4", "q5", "q6", "q7"])
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric", "implicit", "rgbd"],
    )
    args = parser.parse_args()
    main(args)