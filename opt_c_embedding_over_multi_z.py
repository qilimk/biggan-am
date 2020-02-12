import BigGAN
import numpy as np
import random
import time
import torch.nn.functional as F
import yaml

from torch import optim
from torchvision.utils import save_image
from utils import *


def get_initial_embeddings():
    index_list = []
    embedding_dim = 128

    if resolution == 128:
        embedding_name = "other_embedding.npy"
        class_embeddings = np.load(embedding_name)
    else:
        class_embeddings = np.load("1000_embedding_array.npy")

    class_embeddings = torch.from_numpy(class_embeddings)

    if opts["init_method"] == "mean":

        mean_class_embedding = torch.mean(class_embeddings, dim=0)
        init_embeddings = mean_class_embedding.repeat(init_num, 1)
        init_embeddings += torch.randn((init_num, embedding_dim)) * 0.1

    if opts["init_method"] == "top":

        class_embeddings_clamped = torch.clamp(class_embeddings, min_clamp, max_clamp)

        num_samples = 10
        avg_list = []
        for i in range(1000):

            class_embedding = class_embeddings_clamped[i]
            repeat_class_embedding = class_embedding.repeat(num_samples, 1)
            final_z = torch.randn((num_samples, dim_z), requires_grad=False)

            with torch.no_grad():
                gan_image_tensor = G(final_z, repeat_class_embedding)
                final_image_tensor = nn.functional.interpolate(
                    gan_image_tensor, size=224
                )
                final_out = eval_net(final_image_tensor)

            final_probs = nn.functional.softmax(final_out, dim=1)
            avg_target_prob = final_probs[:, target_class].mean().item()
            avg_list.append(avg_target_prob)

        avg_array = np.array(avg_list)
        sort_index = np.argsort(avg_array)

        print(f"The top {init_num} classes: {sort_index[-init_num:]}")

        init_embeddings = class_embeddings[sort_index[-init_num:]]
        index_list = sort_index[-init_num:]

    elif opts["init_method"] == "random":

        random_list = random.sample(range(1000), init_num)
        init_embeddings = class_embeddings[random_list]
        index_list = random_list

    elif opts["init_method"] == "target":

        init_embeddings = (
            class_embeddings[target_class].unsqueeze(0).repeat(init_num, 1)
        )
        init_embeddings += torch.randn((init_num, embedding_dim)) * noise_std
        index_list = [target_class] * init_num

    return (init_embeddings, index_list)


def optimize_embedding():

    global_step_id = 0

    optim_embedding = init_embedding.unsqueeze(0).to(device)
    optim_embedding.requires_grad_()
    optimizer = optim.Adam([optim_embedding], lr=opts["lr"], weight_decay=opts["dr"])

    torch.set_rng_state(state_z)

    for epoch in range(opts["n_iters"]):

        zs = torch.randn((z_num, dim_z), requires_grad=False).to(device)

        for n in range(opts["steps_per_z"]):
            global_step_id += 1

            optimizer.zero_grad()

            clamped_embedding = torch.clamp(optim_embedding, min_clamp, max_clamp)
            repeat_clamped_embedding = clamped_embedding.repeat(z_num, 1).to(device)
            gan_images_tensor = G(zs, repeat_clamped_embedding)
            resized_images_tensor = nn.functional.interpolate(
                gan_images_tensor, size=224
            )
            pred_logits = net(resized_images_tensor)
            loss = criterion(pred_logits, labels)
            pred_probs = nn.functional.softmax(pred_logits, dim=1)

            if dloss_function:
                denom = F.pairwise_distance(zs[odd_list, :], zs[even_list, :])

                if dloss_function == "softmax":

                    diversity_loss = torch.sum(
                        F.pairwise_distance(
                            pred_probs[odd_list, :], pred_probs[even_list, :]
                        )
                    )

                elif dloss_function == "features":

                    features_out = alexnet_conv5(resized_images_tensor)
                    diversity_loss = torch.sum(
                        F.pairwise_distance(
                            features_out[odd_list, :].view(half_z_num, -1),
                            features_out[even_list, :].view(half_z_num, -1),
                        )
                    )

                else:

                    diversity_loss = torch.sum(
                        F.pairwise_distance(
                            resized_images_tensor[odd_list, :].view(half_z_num, -1),
                            resized_images_tensor[even_list, :].view(half_z_num, -1),
                        )
                    )

                loss += -opts["alpha"] * diversity_loss / denom

            loss.backward()
            optimizer.step()

            avg_target_prob = pred_probs[:, target_class].mean().item()
            print(
                f"Epoch: {epoch:0=5d}\tStep: {n:0=5d}\tavg_prob:{avg_target_prob:.4f}"
            )

            if intermediate_dir:
                img_f = f"{embedding_idx}_{global_step_id:0=7d}.jpg"
                output_image_path = f"{intermediate_dir}/{img_f}"
                save_image(gan_images_tensor, output_image_path, normalize=True, nrow=10)

            torch.cuda.empty_cache()

    return optim_embedding


def save_final_samples():
    optim_embedding_clamped = torch.clamp(optim_embedding, min_clamp, max_clamp)
    repeat_optim_embedding = optim_embedding_clamped.repeat(10, 1).to(device)
    save_all = []
    torch.set_rng_state(state_z)
    for show_id in range(3):
        final_z = torch.randn((10, dim_z), device=device, requires_grad=False)
        with torch.no_grad():
            gan_images_tensor = G(final_z, repeat_optim_embedding)

        save_all.append(gan_images_tensor)

    final_image_path = f"{final_dir}/{embedding_idx}.jpg"
    save_all = torch.cat(save_all, dim=0)
    save_image(save_all, final_image_path, normalize=True, nrow=10)

    optim_embedding_clamped.detach().cpu().numpy().save(f"{final_dir}/{embedding_idx}.npy")


if __name__ == "__main__":
    opts = yaml.safe_load("opts.yaml")

    # Set random seed.
    seed_z = opts["seed_z"]
    torch.manual_seed(seed_z)
    torch.cuda.manual_seed(seed_z)
    np.random.seed(seed_z)
    random.seed(seed_z)

    print(f"Initialization method: " + opts["init_method"])
    if opts["init_method"] == "target":
        noise_std = opts["noise_std"]
        print(f"The noise std is: {noise_std}")

    z_num = opts["z_num"]
    list_1 = list(range(0, z_num - 1, 2)) + [random.randint(0, 9) for p in range(10)]
    list_2 = list((np.array(range(1, z_num, 2)) + 4) % 20) + [
        random.randint(10, 19) for p in range(10)
    ]

    dloss_function = opts["dloss_function"]
    if dloss_function:
        print(f"Using diversity loss: {dloss_function}")
        half_z_num = int(z_num / 2)
        odd_list = list(range(0, z_num - 1, 2)) + list_1
        even_list = list(range(1, z_num, 2)) + list_2

    # Load the models.
    start_time = time.time()

    # Set up cudnn.benchmark for free speed.
    torch.backends.cudnn.benchmark = True
    device = "cuda:0"

    print("Loading the BigGAN generator model...")
    resolution = opts["resolution"]
    config = get_config(resolution)
    G = BigGAN.Generator(**config)
    if resolution == 128:
        biggan_weights = "pretrained_weights/138k/G_ema.pth"
    else:
        biggan_weights = None

    G.load_state_dict(torch.load(f"{biggan_weights}"), strict=False)
    G = G.to(device)
    G.eval()

    model = opts["model"]
    net = load_net(model).to(device)
    net.eval()

    if model in {"mit_alexnet", "mit_resnet18"}:
        eval_net = net
    else:
        eval_net = load_net("alexnet")
        eval_net.eval()

    if dloss_function == "features":
        alexnet_conv5 = load_net("alexnet_conv5")
        alexnet_conv5.eval()

    # Set up the optimization.
    criterion = nn.CrossEntropyLoss()
    state_z = torch.get_rng_state()

    init_num = opts["init_num"]
    dim_z = dim_z_dict[resolution]
    max_clamp = max_clamp_dict[resolution]
    min_clamp = min_clamp_dict[resolution]

    print(f"BigGAN initialization time: {time.time() - start_time}")

    target_class = opts["target_class"]
    (init_embeddings, index_list) = get_initial_embeddings()
    labels = torch.LongTensor([target_class] * z_num).to(device)

    intermediate_dir = opts["intermediate_dir"]
    if intermediate_dir:
        os.makedirs(intermediate_dir, exist_ok=True)

    final_dir = opts["final"]
    if final_dir:
        os.makedirs(final_dir, exist_ok=True)

    for (embedding_idx, init_embedding) in enumerate(init_embeddings):
        embedding_idx = str(embedding_idx).zfill(2)
        optim_embedding = optimize_embedding()
        if final_dir:
            save_final_samples()
