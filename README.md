# BigGAN-AM

This code release accompanies the following paper:

## Improving sample diversity of a pre-trained, class-conditional GAN by changing its class embeddings \[[Video](https://youtu.be/y5bDc-dbNjg)\]  \[[arXiv](https://arxiv.org/abs/1910.04760)\]

Qi Li, Long Mai, Michael A. Alcorn, and Anh Nguyen

### Abstract

Mode collapse is a well-known issue with Generative Adversarial Networks (GANs) and is a byproduct of unstable GAN training. We propose to improve the sample diversity of a pre-trained class-conditional generator by modifying its class embeddings in the direction of maximizing the log probability outputs of a classifier pre-trained on the same dataset. We improved the sample diversity of state-of-the-art ImageNet BigGANs at both 128 x 128 and 256 x 256 resolutions. By replacing the embeddings, we can also synthesize plausible images for Places365 using a BigGAN pre-trained on ImageNet.

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
@article{li2019improving,
  title={Improving sample diversity of a pre-trained, class-conditional GAN by changing its class embeddings},
  author={Li, Qi and Mai, Long and Alcorn, Michael A. and Nguyen, Anh},
  journal={arXiv preprint arXiv:1910.04760},
  year={2019}
}
```

## Acknowledgments
This work is supported by the National Science Foundation under Grant No. 1850117 and a donation from Adobe Inc.

Thanks to [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch) by [Andy Brock](https://github.com/ajbrock)

[robustness package](https://github.com/MadryLab/robustness) by MadryLab

[Places365-CNNs](https://github.com/CSAILVision/places365) by MIT CSAIL


