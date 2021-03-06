# Multilingual-StyleCLIP

* Global direction Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aa0mXXHiniuLkScEKNYv1URBalNuKmhS#scrollTo=2dalTroRccEr)
* Latent optimization Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c0h4GVihhUwuT57DW8KI5bi15DSKYAZC#scrollTo=spLeV1kX0EQu)
* Latent mapper Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dIxgJscjF4_cFXGGVG33NE9NK7Rsw31F)

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


## Latent optimization
The code relies on Rosinality pytorch implementation of StyleGAN2. Facial Recognition weights and pretrained restyle encoder are to be downloaded [here](https://github.com/orpatashnik/StyleCLIP).
- --description is for the driving text (can be in any language).
-	To control the manipulation effect, adjust l2 lambda and ID lambda parameters

### Usage
Given a textual description, one can both edit a given image, or generate a random image that best fits to the description. Both operations can be done through the main.py script, or the optimization_playground.ipynb notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c0h4GVihhUwuT57DW8KI5bi15DSKYAZC#scrollTo=spLeV1kX0EQu).

## Latent mapper
The code relies on Rosinality pytorch implementation of StyleGAN2. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dIxgJscjF4_cFXGGVG33NE9NK7Rsw31F)

### training
- This repository trains the mapper with dataset that was inverted by e4e encoder instead of restyle encoder. 
- Inferencing on restyle encoder-inverted images works just fine.
- e4e encoder-inverted dataset is located in the original StyleClip repository.
-	To resume a training, provide --checkpoint_path.
-	--description is for the driving text (can be in any language).
-	To control the manipulation effect, adjust l2 lambda and ID lambda parameters
-	Takes up 10 hours for proper training

```python
!python models/mapper/scripts/train.py --exp_dir exp_dir --no_fine_mapper --description "보라색 머리카락을 가진 사람" \
--latents_train_path data/celebA/train_faces.pt --latents_test_path data/celebA/test_faces.pt \
```

### Inference
-	For inference, we provide several pretrained mappers (text prompt in Korean language)
-	- google drive links for pretrained weights:
     * ["a person with earings" in french](https://drive.google.com/file/d/1f3pwKzarydCM6gGbF47RlBNIXOaf9zuY/view?usp=sharing)

## Global Direction 
The code relies on the official TensorFlow implementation of StyleGAN2. Facial Recognition weights and pretrained restyle encoder are to be downloaded here.

### Usage
Open the notebook in colab and run all the cells. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aa0mXXHiniuLkScEKNYv1URBalNuKmhS#scrollTo=2dalTroRccEr)

In the last cell you can play with the image.
beta corresponds to the disentanglement threshold, and alpha to the manipulation strength.
After you set the desired set of parameters, please run again the last cell to generate the image.


## Editing Examples
### encoder results comparison
![encoder 성능 비교](https://user-images.githubusercontent.com/78332579/145041059-835cba4b-604e-4f93-8799-223e0f53e55e.jpg)

Images below are from celebA-HQ, and were inverted into latent space via Restyle Encoder.
### Latent optimization
- Compare results in other languages : English, Korean, Chinese, Russian
- Original 
   <p align="center">
   <img width="200" alt="original" src="https://user-images.githubusercontent.com/78332579/146683774-45ba513b-9ecf-4613-8572-be62c8e68409.jpg" >
 </p>

   - Text prompt "a person with purple hair"
   ![global_img_3](https://user-images.githubusercontent.com/78332579/147036304-e5589083-d2cf-4b87-92ae-d79a863214cc.jpg)


### Latent mapper
- Compare results in other languages : English, Korean, Russian, Japanese
- Original 
   <p align="center">
   <img width="200" alt="original" src="https://user-images.githubusercontent.com/78332579/147733646-28e8a50f-f0e1-496c-b95b-6ab527e9301f.jpg" >
 </p>

   - Text prompt "a child"
   ![global_img_3](https://user-images.githubusercontent.com/78332579/147733681-a68032b8-15bb-4f3b-a00b-2d824f9ec4be.jpg)

### Global Direction 
- text prompt "a smiling face" and "man's face" in Korean.
![global_img_1](https://user-images.githubusercontent.com/67999107/145788905-59f83085-3751-41c4-b045-6f36977085c3.png)

- Compare results in other languages : English, Korean, Chinese, Spanish
- Original 
   <p align="center">
   <img width="200" alt="original" src="https://user-images.githubusercontent.com/67999107/145793027-836a9bdd-77c3-407c-bb5b-f514dc6736b4.png" >
 </p>

   - Text prompt "a smiling face"
   ![global_img_2](https://user-images.githubusercontent.com/67999107/145790847-c0d625ed-17f4-4aeb-991c-c6f491ed5807.png)

   - Text prompt "a male face"
   ![global_img_3](https://user-images.githubusercontent.com/67999107/145790871-4b872bfe-13d8-4da3-a055-7d17347e89b2.png)


## Acknowledgement
- [CLIP](https://openai.com/blog/clip/)
- [Multilingual-CLIP](https://github.com/FreddeFrallan/Multilingual-CLIP)
- [StyleCLIP](https://github.com/orpatashnik/StyleCLIP)
- [restyle-encoder](https://github.com/yuval-alaluf/restyle-encoder)
- [HuggingFace](https://huggingface.co/)
