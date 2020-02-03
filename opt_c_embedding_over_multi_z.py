import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torch import optim, nn
import torch.nn.functional as F
from torchvision.utils import save_image
import argparse
import os
import seaborn as sns
import pandas as pd
import datetime
import json
import pdb
import BigGAN
import random
# import MadryLab classifier
from robustness import model_utils, datasets
from user_constants import DATA_PATH_DICT
import time


class AlexNetConv5(nn.Module):

    def __init__(self, original_model):
        super(AlexNetConv5, self).__init__()
        self.features = nn.Sequential(
            # stop at conv5
            *list(original_model.features.children())[:-2]
        )

    def forward(self, x):
        x = self.features(x)
        return x


class BigGanAM:

    def __init__(self, resolution, ini_y_num, ini_y_method, seed_z,
                 lr, dr, n_iters, z_num, steps_per_z,
                 model, device, experiment_name,
                 alpha, dloss_funtion, weight_path):

        self.resolution = resolution
        self.ini_y_num = ini_y_num
        self.ini_y_method = ini_y_method
        self.seed_z = seed_z
        self.lr = lr
        self.dr = dr
        self.n_iters = n_iters
        self.z_num = z_num
        self.steps_per_z = steps_per_z
        self.model = model
        self.device = device
        self.config = self.get_config()
        self.experiment_name = experiment_name
        self.alpha = alpha
        self.dloss_funtion = dloss_funtion
        self.weight_path = weight_path
        self.weight_name = weight_path.split('/')[-1].split('.')[0]

        self.G = self.load_biggan()

        self.net = self.load_net(self.model)
        if self.model == 'mit_alexnet':
            self.eval_net = self.net
        elif self.model == 'mit_resnet18':
            self.eval_net = self.net
        elif self.model != 'alexnet':
            self.eval_net = self.load_net('alexnet')
        else:
            self.eval_net = self.net

    def get_config(self):
        attn_dict = {128: '64', 256: '128', 512: '64'}
        dim_z_dict = {128: 120, 256: 140, 512: 128}
        config = {'G_param': 'SN', 'D_param': 'SN',
                  'G_ch': 96, 'D_ch': 96,
                  'D_wide': True, 'G_shared': True,
                  'shared_dim': 128, 'dim_z': dim_z_dict[self.resolution],
                  'hier': True, 'cross_replica': False,
                  'mybn': False, 'G_activation': nn.ReLU(inplace=True),
                  'G_attn': attn_dict[self.resolution],
                  'norm_style': 'bn',
                  'G_init': 'ortho', 'skip_init': True, 'no_optim': True,
                  'G_fp16': False, 'G_mixed_precision': False,
                  'accumulate_stats': False, 'num_standing_accumulations': 16,
                  'G_eval_mode': True,
                  'BN_eps': 1e-04, 'SN_eps': 1e-04,
                  'num_G_SVs': 1, 'num_G_SV_itrs': 1, 'resolution': self.resolution,
                  'n_classes': 1000}

        return config

    def load_biggan(self):

        # Import the model.
        print ("Loading the BigGAN generator model...")
        G = BigGAN.Generator(**self.config)

        if self.resolution == 128:
            weights_path = self.weight_path
        elif self.resolution == 512:
            weights_path = "/home/qi/biggan_optimization/pretrained_weights/biggan-512.pth"
        else:
            weights_path = "/home/qi/biggan_optimization/pretrained_weights/biggan-256.pth"

        G.load_state_dict(torch.load(weights_path), strict=False)
        G = nn.DataParallel(G).to(self.device)
        # G = G.to(self.device)
        return G

    # generate 30 random images based on optimized y
    def final_samples(self, ys, dir_name, state_z, dim_z, max_clamp, min_clamp, target_class):

        dt = datetime.datetime.now()
        final_y = torch.clamp(ys, min_clamp, max_clamp)
        repeat_final_y = final_y.repeat(10, 1).to(self.device)
        save_all = torch.Tensor().to(self.device)
        sum_final_probs = 0

        torch.set_rng_state(state_z)

        for show_id in range(3):

            final_z = torch.randn((10, dim_z), device=self.device, requires_grad=False)
            gan_image_tensor = self.G(final_z, repeat_final_y)

            '''
            if self.model == 'inception_v3':
                final_image_tensor = nn.functional.interpolate(gan_image_tensor, size=299)
                final_out, aux_out = self.net(final_image_tensor)
            else:
                final_image_tensor = nn.functional.interpolate(gan_image_tensor, size=224)
                final_out = self.net(final_image_tensor)
            '''
            final_image_tensor = nn.functional.interpolate(gan_image_tensor, size=224)
            final_out = self.eval_net(final_image_tensor)
            final_probs = nn.functional.softmax(final_out, dim=1)
            avg_prob_y = final_probs[:, target_class].mean().item()

            save_all = torch.cat((save_all, gan_image_tensor), dim=0)
            sum_final_probs += avg_prob_y

        final_image_path = str(dir_name) + "/final_" + str(self.model) + "_y_over_z" + \
                           "_target_" + str("{0:0=3d}".format(target_class)) + \
                           "_iter_" + str("{0:0=3d}".format(self.n_iters)) + \
                           "_znum_" + str("{0:0=3d}".format(self.z_num)) + \
                           "_optnum_" + str("{0:0=3d}".format(self.steps_per_z)) + \
                           "_lr_" + str("{:.6f}".format(self.lr)) + \
                           "_avgprob_" + str("{:.3f}".format(sum_final_probs / 3.0)) + \
                           "_" + "{:%m%d%H%M%S}".format(dt) + \
                           ".jpg"
        save_image(save_all, final_image_path, normalize=True, nrow=10)

    def load_net(self, model_name):
        # Load the classifier
        print("Loading {} classifier...".format(model_name))
        if model_name == 'resnet50':
            net = models.resnet50(pretrained=True)

        elif model_name == 'alexnet':
            net = models.alexnet(pretrained=True)
            # features = nn.Sequential(
            #     # stop at conv5
            #     *list(net.features.children())[:-2]
            # )

            self.alexnet_conv5 = net.features

        elif model_name == 'inception_v3':
            # modified the original file in torchvision/models/inception.py !!!
            net = models.inception_v3(pretrained=True)

        elif model_name == 'mit_alexnet':
            net = self.load_mit('alexnet')

        elif model_name == 'mit_resnet18':
            net = self.load_mit('resnet18')

        # elif model_name == 'restriced_resnet50':
        #     net = self.load_madrylab_imagenet('resnet50')

        elif model_name == 'madrylab_resnet50':
            net = self.load_madrylab_imagenet('resnet50')

        else:
            print("Please specify the classifier...")

        net = nn.DataParallel(net).to(self.device)

        return net

    def load_features(self):

        alexnet_con5 = nn.Sequential(
            # stop at conv5
            *list(self.eval_net.features.children())[:-2]
        )
        nn.DataParallel(nalexnet_con5).to(self.device)

        return alexnet_con5

    def load_madrylab_imagenet(self,arch):
        DATA = 'ImageNet'
        arch = arch
        dataset_function = getattr(datasets, DATA)
        dataset = dataset_function(DATA_PATH_DICT[DATA])

        # Load model
        model_kwargs = { 'arch': arch, 'dataset': dataset, 'resume_path': f'./madrylab_models/{DATA}.pt'}

        model_kwargs['state_dict_path'] = 'model'
        model, _ = model_utils.make_and_restore_model(**model_kwargs)

        return model

    def load_mit(self, model_name):
        # the architecture to use
        # arch = 'alexnet'
        # arch = 'resnet18'
        arch = model_name

        # load the pre-trained weights
        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)

        return model

    def save_files(self, target_class, one_hot=False, index_class=0, ini_onehot_method='top'):
        # Create the output folder
        dt = datetime.datetime.now()

        if one_hot:
            # pdb.set_trace()
            dir_name = "./" + self.experiment_name + "_" + str(self.model) + "_opt_y_over_z/opt_y_multi_z" + \
                       "_target_" + str("{0:0=3d}".format(target_class)) + \
                       "_optmethod_" + self.ini_y_method + '_' + ini_onehot_method + \
                       "_iter_" + str("{0:0=3d}".format(self.n_iters)) + \
                       "_znum_" + str("{0:0=3d}".format(self.z_num)) + \
                       "_optnum_" + str("{0:0=3d}".format(self.steps_per_z)) + \
                       "_lr_" + str("{:.6f}".format(self.lr)) + \
                       "_alpha_" + str("{:.3f}".format(self.alpha)) + \
                       "_dloss_funtion_" + self.dloss_funtion + \
                       "_" + "{:%m%d%H%M%S}".format(dt)

            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            file_extent = "_target_" + str("{0:0=3d}".format(target_class)) + \
                              "_onehot_" + str("{0:0=3d}".format(index_class)) + \
                              "_iter_" + str("{0:0=3d}".format(self.n_iters)) + \
                              "_znum_" + str("{0:0=3d}".format(self.z_num)) + \
                              "_optnum_" + str("{0:0=3d}".format(self.steps_per_z)) + \
                              "_lr_" + str("{:.6f}".format(self.lr)) + \
                              "_seed_" + str("{0:0=2d}".format(self.seed_z)) + \
                              "_" + "{:%m%d%H%M%S}".format(dt)

            filename_y_save = str(dir_name) + "/save_" + str(self.model) + "_opt_y" + str(file_extent)

            filename_z_save = str(dir_name) + "/save_" + str(self.model) + "_opt_z" + str(file_extent)

            intermediate_data_save = str(dir_name) + "/intermediate_data_" + str(self.model) + "_opt_y" + \
                                     str(file_extent) + ".json"

        else:

            dir_name = "./" + self.experiment_name + "_" + str(self.model) + "_opt_y_over_z/opt_y_multi_z" + \
                       "_target_" + str("{0:0=3d}".format(target_class)) + \
                       "_optmethod_" + self.ini_y_method + \
                       "_iter_" + str("{0:0=3d}".format(self.n_iters)) + \
                       "_znum_" + str("{0:0=3d}".format(self.z_num)) + \
                       "_optnum_" + str("{0:0=3d}".format(self.steps_per_z)) + \
                       "_lr_" + str("{:.6f}".format(self.lr)) + \
                       "_alpha_" + str("{:.3f}".format(self.alpha)) + \
                       "_dloss_funtion_" + self.dloss_funtion + \
                       "_" + "{:%m%d%H%M%S}".format(dt)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            file_extent = "_target_" + str("{0:0=3d}".format(target_class)) + \
                              "_iter_" + str("{0:0=3d}".format(self.n_iters)) + \
                              "_znum_" + str("{0:0=3d}".format(self.z_num)) + \
                              "_optnum_" + str("{0:0=3d}".format(self.steps_per_z)) + \
                              "_lr_" + str("{:.6f}".format(self.lr)) + \
                              "_seed_" + str("{0:0=2d}".format(self.seed_z)) + \
                              "_" + "{:%m%d%H%M%S}".format(dt)

            filename_y_save = str(dir_name) + "/save_" + str(self.model) + "_opt_y" + str(file_extent)

            filename_z_save = str(dir_name) + "/save_" + str(self.model) + "_opt_z" + str(file_extent)

            intermediate_data_save = str(dir_name) + "/intermediate_data_" + str(self.model) + "_opt_y" + \
                                     str(file_extent) + ".json"

        return dir_name, filename_y_save, filename_z_save, intermediate_data_save

    def criterion_(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def labels_(self, target_class):
        labels = torch.LongTensor([target_class] * self.z_num).to(self.device)
        return labels

    def optimizer_(self, ys):
        optimizer = optim.Adam([ys], lr=self.lr, weight_decay=self.dr)
        return optimizer

    def slt_one_hot(self, num_y, num_samples, method, dim_z, max_clamp, min_clamp, target_class, std=.0):

        avg_list = []

        if self.resolution == 128:
            embedding_name = self.weight_name + '_embedding.npy'
            y_embedding = np.load(embedding_name)
        else:
            y_embedding = np.load("./1000_embedding_array.npy")

        y_embedding_torch = torch.from_numpy(y_embedding)
        if method == 'top':
            for i in range(0, 1000):

                final_y = torch.clamp(y_embedding_torch[i], min_clamp, max_clamp)
                repeat_final_y = final_y.repeat(num_samples, 1)
                final_z = torch.randn((num_samples, dim_z), requires_grad=False)

                gan_image_tensor = self.G(final_z, repeat_final_y)
                final_image_tensor = nn.functional.interpolate(gan_image_tensor, size=224)
                final_out = self.eval_net(final_image_tensor)

                final_probs = nn.functional.softmax(final_out, dim=1)
                avg_prob_y = final_probs[:, target_class].mean().item()
                avg_list.append(avg_prob_y)

            avg_array = np.array(avg_list)
            sort_index = np.argsort(avg_array)

            print(f'The top {num_y} guess:{sort_index[-num_y:]}')

            y_slt = y_embedding_torch[sort_index[-num_y:]]
            index_list = sort_index[-num_y:]

        elif method == 'random':
            random_list = random.sample(range(0, 999), num_y)
            y_slt = y_embedding_torch[random_list]
            index_list = random_list

        elif method == 'origin':
            print(f'The noise std is: {std}.')
            y_slt = y_embedding_torch[target_class].unsqueeze(0).repeat(num_y, 1)
            y_slt += torch.randn((num_y, 128)) * std
            index_list = [target_class] * num_y

        else:
            print(f'please choose one method to generate the one hot class.')
            y_slt = 0
            index_list = 0

        # pdb.set_trace()
        return y_slt, index_list

    def save_intermediate_data(self, plot_data, path):
        json_save = json.dumps(plot_data)
        f = open(path, "w")
        f.write(json_save)
        f.close()


def slt_ini_method(opt_biggan, ini_y_method, ini_y_num, gaussian_var,
                   resolution, dim_z, max_clamp, min_clamp, noise_std,
                   ini_onehot_method, weight_path, target_class):


    index_list = []
    if ini_y_method == 'random':
        print('Using random initialization of y.')
        y_total = torch.randn((ini_y_num, 128)) * gaussian_var

    elif ini_y_method == 'one_hot':
        print('Using one hot initialization of y.')
        if ini_onehot_method == 'top':
            y_total, index_list = opt_biggan.slt_one_hot(ini_y_num, 10,
                                                         'top', dim_z, max_clamp, min_clamp, target_class)
        elif ini_onehot_method == 'random':
            y_total, index_list = opt_biggan.slt_one_hot(ini_y_num, 10,
                                                         'random', dim_z, max_clamp, min_clamp, target_class)
        else:
            y_total, index_list = opt_biggan.slt_one_hot(ini_y_num, 10,
                                                         'origin', dim_z, max_clamp, min_clamp, target_class, noise_std)

    elif ini_y_method == 'mean_random':
        # load mean as the initial value of y
        print('Using mean of embedding vector to initialize y.')
        if resolution == 128:
            embedding_name = weight_path.split('/')[-1].split('.')[0] + '_embedding_mean.npy'
            y_embedding = np.load(embedding_name)
        else:
            y_embedding = np.load("./mean_1000_embedding.npy")

        y_embedding_torch = torch.from_numpy(y_embedding)
        y_mean_torch = torch.mean(y_embedding_torch,dim=0)
        y_total = y_mean_torch.repeat(ini_y_num, 1)
        y_total += torch.randn((ini_y_num, 128)) * 0.1

    else:
        print('Please choose a method to initialize the y!!!')
        y_total = torch.randn((ini_y_num, 128))

    return y_total, index_list


def main():

    parser = argparse.ArgumentParser(description='Optimizing the process with seed')

    parser.add_argument(
        '--seed_z', type=int, default=0,
        help='Random seed for z to use')

    parser.add_argument(
        '--ini_y_num', type=int, default=5,
        help='the number of initial y')

    parser.add_argument(
        '--ini_y_method', type=str, default='random',
        help='the methods to intialize the y: random/one_hot/mean_random')

    parser.add_argument(
        '--lr', type=float, default=0.1,
        help='Initial learning rate')

    parser.add_argument(
        '--dr', type=float, default=0.9,
        help='Weight decay rate by using adam optimizer')

    parser.add_argument(
        '--n_iters', type=int, default=100,
        help='the number of iterations')

    parser.add_argument(
        '--z_num', type=int, default=10,
        help='the number of z')

    parser.add_argument(
        '--steps_per_z', type=int, default=50,
        help='the number of iterations')

    # parser.add_argument(
    #     '--idx', type=int, default=13,
    #     help='the index of ImageNet classes')

    parser.add_argument(
        '--cuda', type=int, default=0,
        help='the index of gpu3')

    parser.add_argument(
        '--model', type=str, default='alexnet',
        help='the classifier model')

    parser.add_argument(
        '--resolution', type=int, default=256,
        help='the resolution of BigGAN output')

    parser.add_argument(
        '--gaussian_var', type=float, default=1.0,
        help='The variance of gaussian distribution for random initialization ')

    parser.add_argument(
        '--experiment_name', type=str, default='E1',
        help='program will produce the output folder based on experiment name.')

    parser.add_argument(
        '--ini_onehot_method', type=str, default='top',
        help='the methods to generate one hot classes: top/random')

    parser.add_argument(
        '--with_dloss', type=bool, default=False,
        help='add diversity loss to total loss function if True')

    parser.add_argument(
        '--dloss_funtion', type=str, default='softmax',
        help='use different diversity loss functions which are softmax/pixelwise/features')

    parser.add_argument(
        '--alpha', type=float, default=0.1,
        help='The coefficient of diversity loss term')

    parser.add_argument(
        '--noise_std', type=float, default=0,
        help='The std of gaussian add to one hot origin initialization.')

    parser.add_argument(
        '--weight_path', type=str, default='pretrained_weights/E7_G_ema_138000_original.pth',
        help='The std of gaussian add to one hot origin initialization.')

    parser.add_argument(
        '--class_list', type=str, default='../head_50_random_class.txt',
        help='class list to optimize')

    args = parser.parse_args()
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
    cuda_id = args.cuda
    experiment_name = args.experiment_name
    ini_onehot_method = args.ini_onehot_method
    with_dloss = args.with_dloss
    alpha = args.alpha
    dloss_funtion = args.dloss_funtion
    noise_std = args.noise_std
    weight_path = args.weight_path
    class_list = args.class_list


    # set random seed
    torch.manual_seed(seed_z)
    torch.cuda.manual_seed(seed_z)
    np.random.seed(seed_z)
    random.seed(seed_z)

    # Set up cudnn.benchmark for free speed.
    torch.backends.cudnn.benchmark = True
    device = 'cuda'
    start_time = time.time()
    opt_biggan = BigGanAM(resolution, ini_y_num, ini_y_method,seed_z,
                          lr, dr, n_iters, z_num, steps_per_z,
                          model, device, experiment_name, alpha,
                          dloss_funtion, weight_path)
    print(f'initial BigGAN time:{time.time() - start_time}')
    # read the target classes file
    target_list = []
    with open(class_list, 'r') as t_list:
        for f in t_list.readlines():
            target_list.append(int(f))

    dim_z_dict = {128: 120, 256: 140, 512: 128}
    max_clamp_dict = {128: 0.83, 256: 0.61}
    min_clamp_dict = {128: -0.88, 256: -0.59}
    dim_z = dim_z_dict[resolution]
    max_clamp = max_clamp_dict[resolution]
    min_clamp = min_clamp_dict[resolution]

    # Load the models
    G = opt_biggan.G
    G.eval()
    net = opt_biggan.net
    if model == 'alexnet':
        alexnet_con5 = opt_biggan.alexnet_conv5
    net.eval()

    # set up the optimization

    criterion = opt_biggan.criterion_()

    state_z = torch.get_rng_state()
    list_1 = list(range(0, z_num-1, 2)) + [random.randint(0, 9) for p in range(0, 10)]
    list_2 = list((np.array(range(1, z_num, 2)) + 4 ) % 20) + [random.randint(10, 19) for p in range(0, 10)]
    for target_class in target_list:

        y_total, index_list = slt_ini_method(opt_biggan, ini_y_method, ini_y_num, gaussian_var,
                                             resolution, dim_z, max_clamp, min_clamp, noise_std,
                                             ini_onehot_method, weight_path, target_class)


        labels = opt_biggan.labels_(target_class)

        for y_n in range(ini_y_num):

            # initial optimization and create output folder

            global_step_id = 0
            y_save, z_save = [], []
            target_prob_list, target_index_list, top1_prob_list, top1_index_list= [], [], [], []
            iters_no_list = []
            step_index_list = []
            z_index_list = []

            ys = y_total[y_n].unsqueeze(0).to(device)
            ys.requires_grad_()

            if ini_y_method == 'one_hot':
                dir_name, filename_y_save, filename_z_save, intermediate_data_save = \
                    opt_biggan.save_files(target_class, True, index_list[y_n], ini_onehot_method)
            else:
                dir_name, filename_y_save, filename_z_save, intermediate_data_save = opt_biggan.save_files(target_class)

            optimizer = opt_biggan.optimizer_(ys)

            torch.set_rng_state(state_z)
            # start to optimize y
            for epoch in range(n_iters):

                # Sample a new batch of z every loop
                z_total = torch.randn((z_num, dim_z), device=device, requires_grad=False)
                z_save.append(z_total.cpu().numpy())

                for n in range(steps_per_z):
                    global_step_id += 1

                    optimizer.zero_grad()
                    clamped_y = torch.clamp(ys, min_clamp, max_clamp)

                    repeat_clamped_y = clamped_y.repeat(z_num, 1).to(opt_biggan.device)

                    gan_image_tensor = G(z_total, repeat_clamped_y)


                    # pdb.set_trace()
                    if opt_biggan.model == 'inception_v3':
                        # net.train()
                        total_image_tensor = nn.functional.interpolate(gan_image_tensor, size=299)
                        total_out, aux_out = net(total_image_tensor)
                        total_loss = criterion(total_out, labels) + criterion(aux_out, labels)
                    else:
                        total_image_tensor = nn.functional.interpolate(gan_image_tensor, size=224)
                        total_out = net(total_image_tensor)

                        total_loss = criterion(total_out, labels)

                    total_probs = nn.functional.softmax(total_out, dim=1)

                    # add diversity loss
                    if with_dloss:
                        features_out = alexnet_con5(total_image_tensor)
                        print(f'using the diversity loss')
                        half_z_num = int(z_num / 2)
                        odd_list = list(range(0, z_num-1, 2)) + list_1
                        even_list = list(range(1, z_num, 2)) + list_2
                        # pdb.set_trace()
                        # second_list_1 = list(range())

                        if dloss_funtion == 'softmax':

                            diversity_loss = -alpha * torch.sum(
                                F.pairwise_distance(total_probs[odd_list, :], total_probs[even_list, :]) / \
                                F.pairwise_distance(z_total[odd_list, :], z_total[even_list, :]))
                        elif dloss_funtion == 'features':
                            print(f'compute the features space diversity loss!')
                            diversity_loss = -alpha * torch.sum(
                                F.pairwise_distance(features_out[odd_list, :].view(half_z_num,-1), features_out[even_list, :].view(half_z_num,-1)) / \
                                F.pairwise_distance(z_total[odd_list, :], z_total[even_list, :]))
                        else:
                            diversity_loss = -alpha * torch.sum(
                                F.pairwise_distance(total_image_tensor[odd_list, :].view(half_z_num,-1), total_image_tensor[even_list, :].view(half_z_num,-1)) / \
                                F.pairwise_distance(z_total[odd_list, :], z_total[even_list, :]))
                            # diversity_loss = -alpha * torch.sum(
                            #     torch.dist(total_image_tensor[odd_list, :], total_image_tensor[even_list, :]) / \
                            #     torch.dist(z_total[odd_list, :], z_total[even_list, :]))

                        total_loss += diversity_loss

                    avg_prob_y = total_probs[:, target_class].mean().item()

                    top1_prob, top1_index = torch.max(total_probs, 1)
                    target_prob = total_probs[:, target_class]

                    for z_index in range(z_num):
                        z_index_list.append(z_index)
                        iters_no_list.append(epoch)
                        step_index_list.append(n)
                        target_prob_list.append(float(target_prob[z_index].cpu().detach().numpy()))
                        target_index_list.append(target_class)
                        top1_prob_list.append(float(top1_prob[z_index].cpu().detach().numpy()))
                        top1_index_list.append(int(top1_index[z_index].cpu().detach().numpy()))

                    total_loss.backward()
                    optimizer.step()

                    y_save.append(clamped_y.detach().cpu().numpy())

                    print("epoch: {:0=5d}  step: {:0=5d}  avg_prob:{:.4f}".format(epoch, n, avg_prob_y))
                    output_image_path = str(dir_name) + "/opt_"+str(model) + \
                                        "_y_over_z_iter__" + str("{:0=7d}__".format(global_step_id)) + \
                                        "_ylr" + str(lr) + \
                                        "_target_" + str("{0:0=3d}".format(target_class)) + \
                                        "_epoch_" + str("{0:0=5d}".format(epoch)) + \
                                        "_zidx_" + str("{0:0=2d}".format(0)) + \
                                        "_yiters_" + str("{0:0=2d}".format(n)) + \
                                        "_avgprob_" + str("{:.3f}".format(avg_prob_y)) + \
                                        ".jpg"

                    # Just show 10 images per row
                    save_image(gan_image_tensor, output_image_path, normalize=True, nrow=10)

                    torch.cuda.empty_cache()
                    print(output_image_path)

            plot_data = {"run_no": iters_no_list,
                         "z_index": z_index_list,
                         "step_index": step_index_list,
                         "target_prob": target_prob_list,
                         "top1_prob": top1_prob_list,
                         "top1_index": top1_index_list,
                         "target_index": target_index_list,
                         }

            opt_biggan.save_intermediate_data(plot_data, intermediate_data_save)
            opt_biggan.final_samples(ys, dir_name,state_z, dim_z, max_clamp, min_clamp, target_class)
            np.save(filename_y_save, y_save)
            np.save(filename_z_save, z_save)

if __name__ == '__main__':
    main()

