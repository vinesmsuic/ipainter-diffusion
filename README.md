# Intelligent Painter with Resampling Diffusion Models
[![arXiv](https://img.shields.io/badge/arXiv-2210.17106-b31b1b.svg)](https://arxiv.org/abs/2210.17106)

![](https://user-images.githubusercontent.com/34955859/200509260-4c56d4f8-0cd6-4e7d-b32d-efe398629cbf.png)

This repository contains the official PyTorch implementation of the following paper:
> **Intelligent Painter: Picture Composition With Resampling Diffusion Model**<br>
> [Wing-Fung Ku](https://kuwingfung.github.io/), [Wan-Chi Siu](https://scholar.google.com/citations?user=ouQRncoAAAAJ), [Xi Cheng](https://scholar.google.com/citations?user=kpcwnZkAAAAJ), H. Anthony Chan<br>
> https://arxiv.org/abs/2210.17106
> 
> **Abstract:** *Have you ever thought that you can be an intelligent painter? This means that you can paint a picture with a few expected objects in mind, or with a desirable scene. This is different from normal inpainting approaches for which the location of specific objects cannot be determined. In this paper, we present an intelligent painter that generate a person's imaginary scene in one go, given explicit hints. We propose a resampling strategy for Denoising Diffusion Probabilistic Model (DDPM) to intelligently compose harmonized scenery images by injecting explicit landmark inputs at specific locations. By exploiting the diffusion property, we resample efficiently to produce realistic images. Experimental results show that our resampling method favors the semantic meaning of the generated output efficiently and generate less blurry output. Quantitative analysis of image quality assessment shows that our method produces higher perceptual quality images compared with the state-of-the-art methods.*

## Demo Setup

We created a simple GUI demo for users to interact with intelligent painter.

A suitable conda environment named `ipainter` can be created and activated with:
```shell
conda env create -f environment.yaml
conda activate ipainter
```

Clone the repo:

```shell
git clone https://github.com/vinesmsuic/intelligent-painter-diffusion.git
cd intelligent-painter-diffusion
```

We use weights pretrained on [Place2](http://places2.csail.mit.edu/download.html) dataset from the [RePaint](https://github.com/andreas128/RePaint) repo. The `places256_300000.pt` weight can be obtained by the commends:

```shell
cd weights
gdown https://drive.google.com/uc?id=1QEl-btGbzQz6IwkXiFGd49uQNTUtTHsk
cd ..
```
We provided some images to play with. The image folder can be obtained by the commends:

```shell
gdown https://drive.google.com/uc?id=1RBj72Cv82cEj417zeDAzRg5HQheMqknQ
unzip inject_photo.zip
```

To add extra images into your image pool, please add the 4-channel ARGB pngs into the folder `inject_photo`. 

Once you are ready, run the GUI program.

```shell
python main.py
```

The output might look in low-res because we are showing a 256x256 output in a 512x512 window. Check `out_inpaint` folder for the result.



### Direct Inference

Alternatively, users can directly run `repainter.py` with prepared input and mask image.

```shell
python inference.py -h
usage: inference.py [-h] [-s SOURCE] [-d DEST] [--num_samples NUM_SAMPLES]

optional arguments:
  -h, --help            show this help message and exit
  -s SOURCE, --source SOURCE
                        filepath to input image folder. default=out_inpaint
  -d DEST, --dest DEST  Destination folder for saving results. default=out_inpaint_batch
  --num_samples NUM_SAMPLES
                        produce N different samples per image. default=5
```

### Example

we provide a small portion of inputs to test the program. You need to select some of them to test with.

```shell
gdown https://drive.google.com/uc?id=1H9gKH8E_OkgX2eh156yLvJW3OknjvtNo
unzip test_data_small.zip
python inference.py -s test_data_small -d test_data_small_out --num_samples 5
```



## Paint Clearly with Resampling Diffusion models

![](https://user-images.githubusercontent.com/34955859/200509579-4cf1b90f-ce88-459a-b3a9-1341ff0ec233.png)

We proposed a simple yet efficient resampling method to produce clear result in uncondtional image. Our method is 50% faster than RePaint while achieving less blurry results.

> One serious downside is that resampling increases the inference time significantly, as it increases the operations. Moreover, the resampling approach proposed by RePaint often produces image with unclear details. To mitigate this issue, we stop the resampling at timestep t = 100.
>
> We perform Fourier analysis on the forward process and found that the high-frequency components of the image such as fine details are corrupted at lower timesteps, while at larger timesteps the low-frequency components of the image such as coarse structures are corrupted. Therefore, we can assume that the learned reverse process first generates the coarse structure at higher timesteps and then makes fine details at lower timesteps. 
>
> ![](https://user-images.githubusercontent.com/34955859/200742081-4148f124-6a43-4dde-970f-d2391f5d8bbf.png)
>
> The resampling at lower timesteps could potentially blur the details because the preserved low-level information in the forward steps has intervened the denoising process at lower timestep. Since the coarse structure harmonized at bigger timesteps has provided enough information to generate the fine details, we can exploit this property to stop resampling at a smaller timestep, therefore improving the image perceptual quality and the inference time.

![](https://user-images.githubusercontent.com/34955859/214084133-d1443a55-3365-493f-809d-77069e8e7bbe.png)

* While No Resampling inference fastest and produce least blurry image, it failed to harmonize the picture
* Full Resampling approach (RePaint) in unconditional state often produce blurry artifacts, and it is due to the resampling at lower timesteps.
* Our approach of Resampling stops at t=100 solves both problem, providing fast inference and produce less blurry image.



## State-of-the-Art Comparison

![](https://user-images.githubusercontent.com/34955859/200741607-3ccf4d48-7e97-4c9c-9d3c-2821c7a360a4.png)

|  Method  	|   FID (⬇️)  	| NIQE (⬇️) 	| HyperIQA (⬆️) 	| NIMA (⬆️) 	|
|:--------:	|:----------:	|:--------:	|:------------:	|:--------:	|
|  AOT-GAN 	|   277.83   	|   5.96   	|     44.75    	|   5.22   	|
| CoModGAN 	|   225.62   	|   5.30   	|     46.42    	|   5.28   	|
| Big-LaMa 	|   243.55   	|   4.95   	|     45.41    	|   5.10   	|
|    MAT   	|   231.15   	|   5.65   	|     44.66    	|   5.24   	|
|  RePaint 	|   238.88   	|   5.56   	|   **47.86**  	| **5.38** 	|
|   Ours   	| **221.20** 	| **4.86** 	|   **47.64**  	| **5.33** 	|



## Citation

If you found IntelligentPainter useful in your research, please consider starring us on GitHub and citing us in your research.
```bibtex
@misc{ku2022painter,
Author = {Wing-Fung Ku and Wan-Chi Siu and Xi Cheng and H. Anthony Chan},
Title = {Intelligent Painter: Picture Composition With Resampling Diffusion Model},
Year = {2022},
Eprint = {arXiv:2210.17106},
```

## Acknowledgement
This work is partly supported by the Caritas Institute of Higher Education (ISG200206) and UGC Grant (UGC/IDS(C)11/E01/20) of the Hong Kong Special Administrative Region. 

This repository is based on [guided-diffuion](https://github.com/openai/guided-diffusion) and [RePaint](https://github.com/andreas128/RePaint).

