# Multilingual-StyleCLIP

* Global direction playground: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/esoyeon/Multilingual-StyleCLIP/blob/main/ColabSharing/github_styleclip_global.ipynb)
* Latent optimization playground: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/esoyeon/Multilingual-StyleCLIP/blob/main/ColabSharing/github_styleclip_optimization.ipynb)
* Latent mapper playground: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/esoyeon/Multilingual-StyleCLIP/blob/main/ColabSharing/github_styleclip_mapper.ipynb)

## Overview
 Since the release of CLIP by OpenAPI, multiple applications of this multi-modal model have been made, including StyleCLIP. StyleCLIP is a combination of high-resolution image generator - StyleGAN and text-image connecter - CLIP. By measuring cosine similarities of text vector generated by CLIP and image vector generated by StyleGAN, StyleCLIP makes it possible to conveniently manipulate an image with a text prompt. 

 We further extended the benefits of StyleCLIP by implementing Multilingual-CLIP to this model. Multilingual-CLIP consists of two encoders: an image encoder and a fine-tuned text encoder that is capable of encoding any language. Thus, our version of StyleCLIP manipulates an image not only with an English text prompt, but also with a text prompt in any other language, for example in Korean. 
 
 Accuracy of image encoding task also has increased. Official image encoder in StyleCLIP is Encoder4Encoding(e4e) which plays its role when training and testing. However empirically we found out that the result of e4e is quite different from the original input image. To overcome this issue, we encoded data sets for training a mapper and images for inference with Restyle Encoder. Restlye Encoder which was introduced in the paper “ReStyle: A Residual-Based StyleGAN Encoder via Iterative Refinement (ICCV 2021)” iteratively self-corrects the inverted latent code, resulting in increased accuracy. 

This repository contains:
-	Pytorch training code for Multilingual Latent Optimizatizer, Latent Mapper, Global Direction
-	Pytorch inference code Multilingual Latent Optimizatizer, Latent Mapper, Global Direction
-	latent mapper, global direction weights
-	CelebA-HQ Dataset latents (encoded via Restlye)
-	Restlye encoder applied over pSp pretrained on the FFHQ dataset
-	Huggingface available transformer M-BERT Base ViT-B
-	CLIP
-	StyleGAN2

## Setup
The experiment was done in following conditions:
- Python 3.7.12
-	Torch 1.10.0+cu11
-	Google Colab

# Usage
## Latent optimization
The code relies on Rosinality pytorch implementation of StyleGAN2. Facial Recognition weights and pretrained restyle encoder are to be downloaded here.
- --description is for the driving text (can be in any language).
-	To control the manipulation effect, adjust l2 lambda and ID lambda parameters

## Latent mapper
The code relies on Rosinality pytorch implementation of StyleGAN2. Facial Recognition weights and pretrained restyle encoder are to be downloaded here.
### training
-	To resume a training, provide --checkpoint_path.
-	--description is for the driving text (can be in any language).
-	To control the manipulation effect, adjust l2 lambda and ID lambda parameters
-	Takes up 10 hours for proper training

### Inference
-	For inference, we provide several pretrained mappers (text prompt in Korean language)

## Global Direction 
The code relies on the official TensorFlow implementation of StyleGAN2. Facial Recognition weights and pretrained restyle encoder are to be downloaded here.

# Editing Examples
### encoder results comparison
![encoder 성능 비교](https://user-images.githubusercontent.com/78332579/145041059-835cba4b-604e-4f93-8799-223e0f53e55e.jpg)

Images below are from celebA-HQ, and were inverted into latent space via Restyle Encoder.
### Latent optimization
- text prompt "a person with purple hair" in Russian and in Korean.
![optimization purple hair](https://user-images.githubusercontent.com/78332579/145050220-dbc2cfb8-4492-4792-b8fe-8d1c9f2fe8a4.jpg)

### Latent mapper
- text prompt "a person with purple hair" in Korean and "a face with makeup" in Russian.
![mapper results](https://user-images.githubusercontent.com/78332579/145152212-567282d7-c640-48ad-8a82-0f896f3c1636.jpg)

### Global Direction 
- text prompt "a face with blue eyes" and "man's face" in Korean.
![global direction results](https://user-images.githubusercontent.com/78332579/145153053-27dfb696-12b6-491d-a97b-edae8675ca97.jpg)

