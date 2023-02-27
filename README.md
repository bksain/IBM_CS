# IBM_CS
The implementation of "Information Bottleneck Measurement for Compressed Sensing Image Reconstruction", IEEE SIGNAL PROCESSING LETTERS, 2022

# Abstract
Image Compressed Sensing (CS) has achieved a lot of performance improvement thanks to advances in deep networks. The CS method is generally composed of a sensing and a decoder. The sensing and decoder networks have a significant impact on the reconstruction performance, and it is obvious that both two networks must be in harmony. However, previous studies have focused on designing the loss function considering only the decoder network. In this paper, we propose a novel training process that can learn sensing and decoder networks simultaneously using Information Bottleneck (IB) theory. By maximizing importance through proposed importance generator, the sensing network is trained to compress important information for image reconstruction of the decoder network. The representative experimental results demonstrate that the proposed method is applied in recently proposed CS algorithms and increases the reconstruction performance with large margin in all CS ratios.

Thanks to the work of Karl Schulz, the code of this repository borrow from his IBA repository. https://github.com/BioroboticsLab/IBA

# Requirements
Python 3.7

pytorch 1.7.1

CUDA 11.0

# Training

1. Download BSD500 Dataset

2. make "BSD500" folder and put in images

Ex: 

test_data - barbara.tif, boats.tif, ...

BSD500 - 2018.jpg, 2092jpg, ...

3. Run train_patch3_IFBN_n3.py at least 400 epoch

# Citation
Please cite our paper if the code is helpful to your research.

@article{lee2022information,
  title={Information Bottleneck Measurement for Compressed Sensing Image Reconstruction},
  author={Lee, Bokyeung and Ko, Kyungdeuk and Hong, Jonghwan and Ku, Bonhwa and Ko, Hanseok},
  journal={IEEE Signal Processing Letters},
  volume={29},
  pages={1943--1947},
  year={2022},
  publisher={IEEE}
}

# Reference
[1] Schulz, K., Sixt, L., Tombari, F., & Landgraf, T. (2020). Restricting the flow: Information bottlenecks for attribution. arXiv preprint arXiv:2001.00396.
