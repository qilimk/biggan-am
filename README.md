# BigGAN-AM

This code release accompanies the following paper:

## Improving sample diversity of a pre-trained, class-conditional GAN by changing its class embeddings \[[Video](https://youtu.be/y5bDc-dbNjg)\]  \[[arXiv](https://arxiv.org/abs/1910.04760)\]

Qi Li, Long Mai, Anh Nguyen

### Abstract
Mode collapse is a well-known issue with Generative Adversarial Networks (GANs) and is a byproduct of unstable GAN training. We propose to improve the sample diversity of a pre-trained class-conditional generator by modifying its class embeddings in the direction of maximizing the log probability outputs of a classifier pre-trained on the same dataset. We improved the sample diversity of state-of-the-art ImageNet BigGANs at both 128 x 128 and 256 x 256 resolutions. By replacing the embeddings, we can also synthesize plausible images for Places365 using a BigGAN pre-trained on ImageNet.

![ceaser](/doc/ceaser_daisy.png)

![framework](/doc/framework.png)

![synthesize_Places365](/doc/synthesize_new_dataset_images.png)

## Requirements

Please download the pre-trained BigGAN model from [BigGAN-PyTorch](https://drive.google.com/file/d/1nAle7FCVFZdix2--ks0r5JBkFnKw8ctW/view).

* Python 3.6
* PyTorch
* NumPy
* `torchvision`
* Pillow
* [`robustness`](https://github.com/MadryLab/robustness)

## Run the optimization

```bash
python3 opt_c_embedding_over_multi_z.py
```

## Citation
If you find this work useful for your research, please consider citing:
```
@article{li2019improving,
  title={Improving sample diversity of a pre-trained, class-conditional GAN by changing its class embeddings},
  author={Li, Qi and Mai, Long and Nguyen, Anh},
  journal={arXiv preprint arXiv:1910.04760},
  year={2019}
}
```

## Acknowledgments
This work is supported by the National Science Foundation under Grant No. 1850117 and a donation from Adobe Inc.
