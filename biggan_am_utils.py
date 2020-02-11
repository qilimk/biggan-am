import argparse
import datetime
import json
import os
import torch
import torch.nn as nn
import torchvision.models as models

from robustness import datasets, model_utils
from user_constants import DATA_PATH_DICT

dim_z_dict = {128: 120, 256: 140, 512: 128}
attn_dict = {128: "64", 256: "128", 512: "64"}
max_clamp_dict = {128: 0.83, 256: 0.61}
min_clamp_dict = {128: -0.88, 256: -0.59}


def get_config(resolution):
    return {
        "G_param": "SN",
        "D_param": "SN",
        "G_ch": 96,
        "D_ch": 96,
        "D_wide": True,
        "G_shared": True,
        "shared_dim": 128,
        "dim_z": dim_z_dict[resolution],
        "hier": True,
        "cross_replica": False,
        "mybn": False,
        "G_activation": nn.ReLU(inplace=True),
        "G_attn": attn_dict[resolution],
        "norm_style": "bn",
        "G_init": "ortho",
        "skip_init": True,
        "no_optim": True,
        "G_fp16": False,
        "G_mixed_precision": False,
        "accumulate_stats": False,
        "num_standing_accumulations": 16,
        "G_eval_mode": True,
        "BN_eps": 1e-04,
        "SN_eps": 1e-04,
        "num_G_SVs": 1,
        "num_G_SV_itrs": 1,
        "resolution": resolution,
        "n_classes": 1000,
    }


def load_mit(model_name):
    model_file = f"{model_name}_places365.pth.tar"
    if not os.access(model_file, os.W_OK):
        weight_url = f"http://places2.csail.mit.edu/models_places365/{model_file}"
        os.system(f"wget {weight_url}")

    model = models.__dict__[model_name](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {
        str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)

    return model


def load_madrylab_imagenet(arch):
    data = "ImageNet"
    arch = arch
    dataset_function = getattr(datasets, data)
    dataset = dataset_function(DATA_PATH_DICT[data])
    model_kwargs = {
        "arch": arch,
        "dataset": dataset,
        "resume_path": f"madrylab_models/{data}.pt",
        "state_dict_path": "model",
    }
    (model, _) = model_utils.make_and_restore_model(**model_kwargs)

    return model


def load_net(model_name):
    print(f"Loading {model_name} classifier...")
    if model_name == "resnet50":
        return models.resnet50(pretrained=True)

    elif model_name == "alexnet":
        return models.alexnet(pretrained=True)

    elif model_name == "alexnet_conv5":
        return models.alexnet(pretrained=True).features

    elif model_name == "inception_v3":
        # Modified the original file in torchvision/models/inception.py!!!
        return models.inception_v3(pretrained=True)

    elif model_name == "mit_alexnet":
        return load_mit("alexnet")

    elif model_name == "mit_resnet18":
        return load_mit("resnet18")

    elif model_name == "madrylab_resnet50":
        return load_madrylab_imagenet("resnet50")

    else:
        raise ValueError(f"{model_name} is not a supported classifier...")


def parse_options():
    parser = argparse.ArgumentParser(description="Optimizing with seed.")

    parser.add_argument(
        "--seed_z", type=int, default=0, help="Random seed for z to use."
    )

    parser.add_argument(
        "--ini_y_num", type=int, default=3, help="The number of initial ys."
    )

    parser.add_argument(
        "--ini_y_method",
        type=str,
        default="mean_random",
        help="The method to initialize the y: random/one_hot/mean_random.",
    )

    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")

    parser.add_argument(
        "--dr",
        type=float,
        default=0.9,
        help="Weight decay rate used by the Adam optimizer",
    )

    parser.add_argument(
        "--n_iters", type=int, default=10, help="The number of iterations."
    )

    parser.add_argument("--z_num", type=int, default=10, help="The number of zs.")

    parser.add_argument(
        "--steps_per_z", type=int, default=20, help="The number of update steps per z."
    )

    parser.add_argument("--cuda", type=int, default=0, help="The index of the GPU.")

    parser.add_argument(
        "--model", type=str, default="alexnet", help="The classifier model."
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="The resolution of the BigGAN output.",
    )

    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.1,
        help="The standard deviation of the Gaussian used for one-hot origin initialization.",
    )

    parser.add_argument(
        "--with_dloss",
        type=bool,
        default=False,
        help="Add diversity loss to total loss function.",
    )

    parser.add_argument(
        "--dloss_function",
        type=str,
        default="softmax",
        help="The diversity loss function: softmax/pixelwise/features.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="The coefficient for the diversity loss term.",
    )

    parser.add_argument(
        "--weight_path",
        type=str,
        default="pretrained_weights/138k/G_ema.pth",
        help="The path for the pre-trained BigGAN weights.",
    )

    parser.add_argument(
        "--class_list",
        type=str,
        default="class_list.txt",
        help="List of classes to optimize.",
    )

    return parser.parse_args()
