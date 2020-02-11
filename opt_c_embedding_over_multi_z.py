import BigGAN
import numpy as np
import random
import time
import torch.nn.functional as F

from biggan_am_utils import *
from torch import optim, nn
from torchvision.utils import save_image


def slt_ini_method():
    index_list = []
    if ini_y_method == "random":
        y_total = torch.randn((ini_y_num, 128)) * gaussian_var

    elif ini_y_method == "mean_random":
        y_total = y_mean_torch.repeat(ini_y_num, 1)
        y_total += torch.randn((ini_y_num, 128)) * 0.1

    elif ini_y_method == "one_hot":
        (y_total, index_list) = slt_one_hot(ini_y_num, 10)

    return (y_total, index_list)


def slt_one_hot(num_y, num_samples):
    y_embedding_torch = torch.from_numpy(y_embedding)

    if ini_onehot_method == "top":

        avg_list = []
        for i in range(1000):

            final_y = torch.clamp(y_embedding_torch[i], min_clamp, max_clamp)
            repeat_final_y = final_y.repeat(num_samples, 1)
            final_z = torch.randn((num_samples, dim_z), requires_grad=False)

            with torch.no_grad():
                gan_image_tensor = G(final_z, repeat_final_y)
                if model == "inception_v3":
                    final_image_tensor = nn.functional.interpolate(
                        gan_image_tensor, size=299
                    )
                    (final_out, _) = eval_net(final_image_tensor)
                else:
                    final_image_tensor = nn.functional.interpolate(
                        gan_image_tensor, size=224
                    )
                    final_out = eval_net(final_image_tensor)

            final_probs = nn.functional.softmax(final_out, dim=1)
            avg_prob_y = final_probs[:, target_class].mean().item()
            avg_list.append(avg_prob_y)

        avg_array = np.array(avg_list)
        sort_index = np.argsort(avg_array)

        print(f"The top {num_y} guesses: {sort_index[-num_y:]}")

        y_slt = y_embedding_torch[sort_index[-num_y:]]
        index_list = sort_index[-num_y:]

    elif ini_onehot_method == "random":
        random_list = random.sample(range(0, 1000), num_y)
        y_slt = y_embedding_torch[random_list]
        index_list = random_list

    elif ini_onehot_method == "origin":
        print(f"The noise std is: {noise_std}")
        y_slt = y_embedding_torch[target_class].unsqueeze(0).repeat(num_y, 1)
        y_slt += torch.randn((num_y, 128)) * noise_std
        index_list = [target_class] * num_y

    else:
        raise ValueError("Please choose a method to generate the one-hot class.")

    return (y_slt, index_list)


def get_diversity_loss():
    denom = F.pairwise_distance(z_total[odd_list, :], z_total[even_list, :])

    if dloss_funtion == "softmax":

        return -alpha * torch.sum(
            F.pairwise_distance(total_probs[odd_list, :], total_probs[even_list, :])
            / denom
        )

    elif dloss_funtion == "features":

        features_out = alexnet_conv5(total_image_tensor)
        return -alpha * torch.sum(
            F.pairwise_distance(
                features_out[odd_list, :].view(half_z_num, -1),
                features_out[even_list, :].view(half_z_num, -1),
            )
            / denom
        )

    else:

        return -alpha * torch.sum(
            F.pairwise_distance(
                total_image_tensor[odd_list, :].view(half_z_num, -1),
                total_image_tensor[even_list, :].view(half_z_num, -1),
            )
            / denom
        )


def final_samples():
    dt = datetime.datetime.now()
    final_y = torch.clamp(ys, min_clamp, max_clamp)
    repeat_final_y = final_y.repeat(10, 1).to(device)
    save_all = []
    sum_final_probs = 0
    torch.set_rng_state(state_z)
    for show_id in range(3):

        final_z = torch.randn((10, dim_z), device=device, requires_grad=False)
        with torch.no_grad():
            gan_image_tensor = G(final_z, repeat_final_y)
            final_image_tensor = nn.functional.interpolate(gan_image_tensor, size=224)
            final_out = eval_net(final_image_tensor)

        final_probs = nn.functional.softmax(final_out, dim=1)
        avg_prob_y = final_probs[:, target_class].mean().item()

        save_all.append(gan_image_tensor)
        sum_final_probs += avg_prob_y

    final_image_path = f"{dir_name}/final_{str(model)}_y_over_z_target_"
    final_image_path += f"{target_class:0=3d}_iter_{n_iters:0=3d}_znum_"
    final_image_path += f"{z_num:0=3d}_optnum_{steps_per_z:0=3d}_lr_"
    final_image_path += f"{lr:.6f}_avgprob_{steps_per_z:0=3d}_lr_"
    final_image_path += f"{sum_final_probs / 3.0:.3f}_{dt:%m%d%H%M%S}.jpg"

    save_all = torch.cat(save_all, dim=0)
    save_image(save_all, final_image_path, normalize=True, nrow=10)


if __name__ == "__main__":
    args = parse_options()
    ini_y_num = args.ini_y_num
    ini_y_method = args.ini_y_method
    seed_z = args.seed_z
    lr = args.lr
    dr = args.dr
    n_iters = args.n_iters
    z_num = args.z_num
    steps_per_z = args.steps_per_z
    model = args.model
    resolution = args.resolution
    gaussian_var = args.gaussian_var
    experiment_name = args.experiment_name
    ini_onehot_method = args.ini_onehot_method
    with_dloss = args.with_dloss
    alpha = args.alpha
    dloss_funtion = args.dloss_funtion
    noise_std = args.noise_std
    weight_path = args.weight_path
    weight_name = weight_path.split("/")[-1].split(".")[0]
    class_list = args.class_list

    if ini_y_method == "random":
        print("Using random initialization of y.")
    elif ini_y_method == "one_hot":
        print("Using one hot initialization of y.")
    elif ini_y_method == "mean_random":

        # Load mean as the initial value of y.
        print("Using mean embedding vector to initialize y.")
        # Load mean as the initial value of y.
        if resolution == 128:
            embedding_name = (
                weight_path.split("/")[-1].split(".")[0] + "_embedding_mean.npy"
            )
            y_embedding = np.load(embedding_name)
        else:
            y_embedding = np.load("./mean_1000_embedding.npy")

        y_embedding_torch = torch.from_numpy(y_embedding)
        y_mean_torch = torch.mean(y_embedding_torch, dim=0)

    else:
        raise ValueError("Please choose a method to initialize the y!!!")

    # Set random seed.
    torch.manual_seed(seed_z)
    torch.cuda.manual_seed(seed_z)
    np.random.seed(seed_z)
    random.seed(seed_z)

    # Set up cudnn.benchmark for free speed.
    torch.backends.cudnn.benchmark = True
    device = "cuda:0"

    # Read the target classes file.
    target_list = []
    with open(class_list, "r") as t_list:
        for f in t_list.readlines():
            target_list.append(int(f))

    dim_z_dict = {128: 120, 256: 140, 512: 128}
    max_clamp_dict = {128: 0.83, 256: 0.61}
    min_clamp_dict = {128: -0.88, 256: -0.59}
    dim_z = dim_z_dict[resolution]
    max_clamp = max_clamp_dict[resolution]
    min_clamp = min_clamp_dict[resolution]

    # Load the models.
    start_time = time.time()

    print("Loading the BigGAN generator model...")
    config = get_config(resolution)
    G = BigGAN.Generator(**config)
    G.load_state_dict(torch.load(weight_path), strict=False)
    G = nn.DataParallel(G).to(device)
    G.eval()

    net = load_net(model)
    net = nn.DataParallel(net).to(device)
    net.eval()

    if (model in {"mit_alexnet", "mit_resnet18"}) or (model == "alexnet"):
        eval_net = net
    else:
        eval_net = load_net("alexnet")
        eval_net.eval()

    if with_dloss and (dloss_funtion == "features"):
        alexnet_conv5 = load_net("alexnet_conv5")

    print(f"BigGAN initialization time: {time.time() - start_time}")

    # Set up the optimization.
    criterion = nn.CrossEntropyLoss()

    state_z = torch.get_rng_state()
    list_1 = list(range(0, z_num - 1, 2)) + [random.randint(0, 9) for p in range(0, 10)]
    list_2 = list((np.array(range(1, z_num, 2)) + 4) % 20) + [
        random.randint(10, 19) for p in range(0, 10)
    ]

    if with_dloss:
        print("Using the diversity loss.")
        half_z_num = int(z_num / 2)
        odd_list = list(range(0, z_num - 1, 2)) + list_1
        even_list = list(range(1, z_num, 2)) + list_2
        if dloss_funtion == "features":
            print(f"Diversity loss in feature space.")

    save_metadata = {
        "experiment_name": experiment_name,
        "model": model,
        "index_class": -1,
        "ini_y_method": ini_y_method,
        "ini_onehot_method": ini_onehot_method,
        "n_iters": n_iters,
        "z_num": z_num,
        "steps_per_z": steps_per_z,
        "lr": lr,
        "alpha": alpha,
        "dloss_function": dloss_funtion,
        "seed_z": seed_z,
    }
    if ini_y_method == "one_hot":
        save_metadata["one_hot"] = True
        if resolution == 128:
            embedding_name = weight_name + "_embedding.npy"
            y_embedding = np.load(embedding_name)
        else:
            y_embedding = np.load("./1000_embedding_array.npy")

    for target_class in target_list:

        save_metadata["target_class"] = target_class
        (y_total, index_list) = slt_ini_method()
        labels = torch.LongTensor([target_class] * z_num).to(device)

        for y_n in range(ini_y_num):

            # Initialize optimization and create output folder.

            global_step_id = 0
            (y_save, z_save) = ([], [])
            (target_prob_list, target_index_list) = ([], [])
            (top1_prob_list, top1_index_list) = ([], [])
            iters_no_list = []
            step_index_list = []
            z_index_list = []

            ys = y_total[y_n].unsqueeze(0).to(device)
            ys.requires_grad_()

            if ini_y_method == "one_hot":
                save_metadata["index_class"] = index_list[y_n]

            (
                dir_name,
                filename_y_save,
                filename_z_save,
                intermediate_data_save,
            ) = save_files(**save_metadata)

            optimizer = optim.Adam([ys], lr=lr, weight_decay=dr)

            torch.set_rng_state(state_z)

            # Optimize y.
            for epoch in range(n_iters):

                # Sample a new batch of zs every loop.
                z_total = torch.randn(
                    (z_num, dim_z), device=device, requires_grad=False
                )
                z_save.append(z_total.cpu().numpy())

                for n in range(steps_per_z):
                    global_step_id += 1

                    optimizer.zero_grad()

                    clamped_y = torch.clamp(ys, min_clamp, max_clamp)
                    repeat_clamped_y = clamped_y.repeat(z_num, 1).to(device)
                    gan_image_tensor = G(z_total, repeat_clamped_y)

                    if model == "inception_v3":
                        total_image_tensor = nn.functional.interpolate(
                            gan_image_tensor, size=299
                        )
                        (total_out, aux_out) = net(total_image_tensor)
                        total_loss = criterion(total_out, labels) + criterion(
                            aux_out, labels
                        )

                    else:
                        total_image_tensor = nn.functional.interpolate(
                            gan_image_tensor, size=224
                        )
                        total_out = net(total_image_tensor)
                        total_loss = criterion(total_out, labels)

                    total_probs = nn.functional.softmax(total_out, dim=1)

                    # Add diversity loss.
                    if with_dloss:
                        diversity_loss = get_diversity_loss()
                        total_loss += diversity_loss

                    total_loss.backward()
                    optimizer.step()

                    (top1_prob, top1_index) = torch.max(total_probs, 1)
                    target_prob = total_probs[:, target_class]
                    for z_index in range(z_num):
                        z_index_list.append(z_index)
                        iters_no_list.append(epoch)
                        step_index_list.append(n)
                        target_prob_list.append(
                            float(target_prob[z_index].cpu().detach().numpy())
                        )
                        target_index_list.append(target_class)
                        top1_prob_list.append(
                            float(top1_prob[z_index].cpu().detach().numpy())
                        )
                        top1_index_list.append(
                            int(top1_index[z_index].cpu().detach().numpy())
                        )

                    y_save.append(clamped_y.detach().cpu().numpy())

                    avg_prob_y = total_probs[:, target_class].mean().item()
                    print(
                        f"epoch: {epoch:0=5d}\tstep: {n:0=5d}\tavg_prob:{avg_prob_y:.4f}"
                    )

                    output_image_path = f"{dir_name}/opt_{model}_y_over_z_iter_"
                    output_image_path += f"{global_step_id:0=7d}_ylr_{lr}_target_"
                    output_image_path += f"{target_class:0=3d}_epoch_{epoch:0=5d}_"
                    output_image_path += f"zidx_{0:0=2d}_yiters_{n:0=2d}_avgprob_"
                    output_image_path += f"{avg_prob_y:.3f}.jpg"

                    # Only show 10 images per row.
                    save_image(
                        gan_image_tensor, output_image_path, normalize=True, nrow=10
                    )

                    torch.cuda.empty_cache()
                    print(output_image_path)

            plot_data = {
                "run_no": iters_no_list,
                "z_index": z_index_list,
                "step_index": step_index_list,
                "target_prob": target_prob_list,
                "top1_prob": top1_prob_list,
                "top1_index": top1_index_list,
                "target_index": target_index_list,
            }

            save_intermediate_data(plot_data, intermediate_data_save)
            final_samples()
            np.save(filename_y_save, y_save)
            np.save(filename_z_save, z_save)
