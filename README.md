# Unsupervised Inpainting for Occluded Sea Surface Temperature Sequences
Repository for Unsupervised Inpainting for Occluded Sea Surface Temperature Sequences (Yin et al., 2019)

The code will be released soon.

## Abstract

Sea Surface Temperature (SST) data remotely detected from satellites are inevitably occluded by clouds or rains. To learn the prior of SST data and complete the missing part, inpainting approaches in machine learning often require ground truth in the occluded area, which is not applicable for remote SST. In this paper, we propose an inpainting approach based on generative adversarial networks (GANs) for occluded SST. It is capable to complete the occluded area without any supervision on ground truth data of the occluded area. Our framework is evaluated on real SST data with simulated cloud masks.

## Download Instruction for Datasets

To download the cloud dataset, please check https://doi.org/10.6084/m9.figshare.9145016.

To download the SST dataset, please sign in to Copernicus Marine environment monitoring service, check the product at http://marine.copernicus.eu/services-portfolio/access-to-products/?option=com_csw&view=details&product_id=GLOBAL_ANALYSIS_FORECAST_PHY_001_024, and select corresponding regions. For training and validation sets, select the period from 2018-01-01 00:30 to 2018-12-31 23:30 between latitude 20°~25.25°N (+20~+25.25) and longitude 34.75°~40W (-40~-34.75). For test set, select the period from 2019-01-01 00:30 to 2019-06-30 23:30 between latitude 14.75°~20°S (-20~-14.75) and longitude 14.75°-20W (-20~-14.75).

## Installation of Requirements

This code requires Python 3, and we recommend a version higher than 3.6.

To install all requirements, run 
```
pip3 install -r requirements.txt
```

## Experiments

To run experiment, set cloud dataset path and SST dataset paths in `src/unisst.yaml` and experiment storage path in `run.py`, then run
```
bash train.sh unisst 0
```
on GPU `cuda:0`, you can use multiple GPU for larger batch sizes, for example
```
bash train.sh unisst 0,1,2,3
```

## Samples

We shreshold Liquid Water Path (LWP, in unit of g/m<sup>2</sup>) to generate clouds at different occlusion rates. Here are some samples from test sets. (Each sample from top to bottom: unknown ground truth, observation, recovered sequence) 

| LWP Threshold | Sample | LWP Threshold | Sample | 
| :-----------: | ------ | :-----------: | ------ |
| 55 g/m<sup>2</sup> | ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_55gm2_1.gif) ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_55gm2_2.gif) | 70 g/m<sup>2</sup> | ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_70gm2_1.gif) ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_70gm2_2.gif) | 
| 60 g/m<sup>2</sup> | ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_60gm2_1.gif) ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_60gm2_2.gif) | 75 g/m<sup>2</sup> | ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_75gm2_1.gif) ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_75gm2_2.gif) | 
| 65 g/m<sup>2</sup> | ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_65gm2_1.gif) ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_65gm2_2.gif) | 80 g/m<sup>2</sup> | ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_80gm2_1.gif) ![alt text](https://github.com/yuan-yin/UNISST/blob/master/samples/test_80gm2_2.gif) | 
