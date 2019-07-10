# Unsupervised Inpainting for Occluded Sea Surface Temperature Sequences
Repository for Unsupervised Inpainting for Occluded Sea Surface Temperature Sequences (Yin et al., 2019)

The code will be released soon.

## Abstract

Sea Surface Temperature (SST) data remotely detected from satellites are inevitably occluded by clouds or rains. To learn the prior of SST data and complete the missing part, inpainting approaches in machine learning often require ground truth in the occluded area, which is not applicable for remote SST. In this paper, we propose an inpainting approach based on generative adversarial networks (GANs) for occluded SST. It is capable to complete the occluded area without any supervision on ground truth data of the occluded area. Our framework is evaluated on real SST data with simulated cloud masks.

## Samples

We shreshold Liquid Water Path (LWP, in unit of g/m<sup>2</sup>) to generate clouds at different occlusion rates. Here are some samples from test sets. (Each sample from top to bottom: unknown ground truth, observation, recovered sequence) 

| LWP Threshold | Sample | LWP Threshold | Sample | 
| :-----------: | ------ | :-----------: | ------ |
| 55 g/m<sup>2</sup> | ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_55gm2_1.gif) ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_55gm2_2.gif) | 70 g/m<sup>2</sup> | ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_70gm2_1.gif) ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_70gm2_2.gif) | 
| 60 g/m<sup>2</sup> | ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_60gm2_1.gif) ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_60gm2_2.gif) | 75 g/m<sup>2</sup> | ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_75gm2_1.gif) ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_75gm2_2.gif) | 
| 65 g/m<sup>2</sup> | ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_65gm2_1.gif) ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_65gm2_2.gif) | 80 g/m<sup>2</sup> | ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_80gm2_1.gif) ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_80gm2_2.gif) | 
