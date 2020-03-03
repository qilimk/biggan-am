import BigGAN
import numpy as np
import sys

from torchvision.utils import save_image
from utils import *


def main():
    # Set up cudnn.benchmark for free speed.
    torch.backends.cudnn.benchmark = True
    device = "cuda:0"

    print("Loading the BigGAN generator model...", flush=True)
    resolution = 256
    config = get_config(resolution)
    G = BigGAN.Generator(**config)
    G.load_state_dict(
        torch.load("pretrained_weights/biggan_256_weights.pth"), strict=False
    )
    G = nn.DataParallel(G).to(device)
    G.eval()

    data_source = sys.argv[1]  # "imagenet" or "places".
    target = sys.argv[2]  # Filename found in "imagenet" or "places" directory.
    class_embedding = np.load(f"{data_source}/{target}.npy")
    class_embedding = torch.tensor(class_embedding)

    z_num = 16
    repeat_class_embedding = class_embedding.repeat(z_num, 1).to(device)
    zs = torch.randn((z_num, dim_z_dict[resolution]), requires_grad=False).to(device)

    gan_images_tensor = G(zs, repeat_class_embedding)

    save_dir = "samples"
    print(f"Saving class embedding samples in {save_dir}.", flush=True)
    os.makedirs(save_dir, exist_ok=True)
    final_image_path = f"{save_dir}/{data_source}_{target}.jpg"
    save_image(gan_images_tensor, final_image_path, normalize=True, nrow=4)


if __name__ == "__main__":
    main()
