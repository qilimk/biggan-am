# BigGAN-AM

This code release accompanies the following paper:

## A cost-effective method for improving and re-purposing large, pre-trained GANs by fine-tuning their class-embeddings \[[Video](https://youtu.be/y5bDc-dbNjg)\]  \[[arXiv](https://arxiv.org/abs/1910.04760)\]

Qi Li, Long Mai, Michael A. Alcorn, and [Anh Nguyen](http://anhnguyen.me/). Asian Conference on Computer Vision (ACCV). 2020. **Oral presentation**. :star: Huawei Best Application Paper Honorable Mention at ACCV 2020. ⭐

### Abstract

<sup>Large, pre-trained generative models have been increasingly popular and useful to both the research and wider communities. Specifically, BigGANs a class-conditional Generative Adversarial Networks trained on ImageNet—achieved excellent, state-of-the-art capability in generating realistic photos. However, fine-tuning or training BigGANs from scratch is practically impossible for most researchers and engineers because (1) GAN training is often unstable and suffering from mode-collapse; and (2) the training requires a significant amount of computation, 256 Google TPUs for 2 days or 8xV100 GPUs for 15 days. Importantly, many pre-trained generative models both in NLP and image domains were found to contain biases that are harmful to society. Thus, we need computationally-feasible methods for modifying and re-purposing these huge, pre-trained models for downstream tasks. In this paper, we propose a cost-effective optimization method for improving and re-purposing BigGANs by fine-tuning only the class-embedding layer. We show the effectiveness of our model-editing approach in three tasks: (1) significantly improving the realism and diversity of samples of complete mode-collapse classes; (2) re-purposing ImageNet BigGANs for generating images for Places365; and (3) de-biasing or improving the sample diversity for selected ImageNet classes.</sup>

![ceaser](/doc/ceaser_daisy.png)

![framework](/doc/framework.png)

![synthesize_Places365](/doc/synthesize_new_dataset_images.png)

## Setup

1) Download the pre-trained BigGAN model weights (see [here](https://stackoverflow.com/a/48133859/1316276) for an explanation of the commands):

```bash
cd biggan-am/pretrained_weights

# 128x128 weights.
fileid="1nAle7FCVFZdix2--ks0r5JBkFnKw8ctW"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o biggan_weights.zip
unzip biggan_weights.zip
rm biggan_weights.zip

# 256x256 weights.
fileid="1FEAXaUjRcV8mb0sHIwuS2EFyAcAITDTw"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o biggan_256_weights.pth
```

2) Install the Python packages (we used Python 3.6):

```bash
cd biggan-am
pip3 install -r requirements.txt
```

Alternatively, you can use our provided [`Dockerfile`](Dockerfile) to run the model in a container:

```bash
cd biggan-am
docker build -t biggan-am:v1 .
nvidia-docker run \
    -a stdin \
    -a stdout \
    -it \
    --name biggan-am \
    --mount type=bind,source=$(pwd),target=/biggan-am \
    biggan-am:v1 \
    /bin/bash
```

## Running BigGAN-AM

1) Edit [`opts.yaml`](opts.yaml):

```bash
cd biggan-am
nano opts.yaml
```

2) Run BigGAN-AM:

```bash
python3 biggan_am.py
```

## Generating samples from a BigGAN-AM optimized class embedding

```bash
cd biggan-am
python3 sample_biggan_am_embedding.py [directory] [target]
```

e.g.:

```bash
python3 sample_biggan_am_embedding.py "places" "266_pier"
```

## Citation
If you find this work useful for your research, please consider citing:
```
@article{li2020improving,
  title={A cost-effective method for improving and re-purposing large, pre-trained GANs by fine-tuning their class-embeddings},
  author={Li, Qi and Mai, Long and Alcorn, Michael A. and Nguyen, Anh},
  journal={Asian Conference on Computer Vision},
  year={2020}
}
```

## Acknowledgments
This work is supported by the National Science Foundation under Grant No. 1850117 and a donation from Adobe Inc.

Thanks to [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch) by [Andy Brock](https://github.com/ajbrock)

[robustness package](https://github.com/MadryLab/robustness) by MadryLab

[Places365-CNNs](https://github.com/CSAILVision/places365) by MIT CSAIL


