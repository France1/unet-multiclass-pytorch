# unet-multiclass-pytorch

Multiclass semantic segmentation using U-Net architecture combined with strong image augmentation (i.e. patch training and inference) tested on sythetic images

## Background 
This project combines (i) the U-Net archicture [1], as implemented in PyTorch by Milesial [2], with (ii) the patch training and inference technique implemented by Orobix for retina blood vessel segmentation [3], and extend them to a broad class of multi-class semantic segmentation tasks with small numbers of images and labels

## Project structure
```
unet-multiclass-pytorch/
    - checkpoints/
    - dataloader/
    - .gitignore
    - lib/
    - model/
    - notebooks/
    - parameters.json
    - runs/
    - README.md
    - requirements.txt
    - train.py
```
in which:
- `checkpoints/` contains PyTorch U-Net model parameters
- `dataloader/` contains functions for loading raw data
- `lib/` contains functions for generating and processing training data, and for model visualization
- `model/` contains model parts and model related functions
- `notebooks/` contains jupyter notebooks for preparing training data, and for model inference and evaluation
- `parameters.json` define all the parameters of the analysis
- `runs/` contains Tensorboard summary files
- `train.py` is the main script for model training

## Synthetic image generation
Syntethic images and masks of size 400 by 200 are generated through the [`image_generation.ipynb`](./notebooks/Image_generation.ipynb) notebook as shown below. Randomly oriented lines and ellipses with variable gray scale intensity are placed onto a Gaussian noise background. The images are stored into the `data` folder as:
```
.data/
    - images_training   # (100 images)
    - images_test       # (10 images)
```
<div align="center">
    <img src="pictures/Input_data.png" alt="drawing" width="400"/>
</div>

## Patches preparation
Training images and masks are prepared using the [`Image_preparation`](./notebooks/Image_preparation.ipynb) notebook. Square patches are extracted at random locations following [Orobix](https://github.com/orobix/retina-unet) approach, as demonstrated below. In this project 20 patches of size 96 by 96 are extracted per image. Additional augmentation for each pactch consists of all combinations of up/down and right/left flipping.  
<div align="center">
    <img src="pictures/Image_patches.png" alt="drawing" width="1000"/>
</div>

The notebook generates folders and images into the `data` folder as
```
./data/patches_s96/:
    - images/:
        - p0.png
        - ...
        - pN.png
    - masks/:
        - p0.png
        - ...
        - pN.png
```

## Model training and evaluation
The following command is an example to train the model for 100 epochs, with batch size 16, and learning rate 0.01, using `patches_s96` dataset:
```
python train.py -e 100 -l 0.01 -b 16 -f patches_s96
```

Multiclass cross entropy loss function is used with SGD optimizer. The learning rate is decreased towards the second half of the epochs in order to stabilize the model training. Model performance is measured using mean Intersection Over Union (mIoU) across all the classes following [Keras](https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/metrics.py) approach. During training the model is evaluated on 10% of the patches dataset. The mean IoU obtained on the patches evaluation set is 0.98, as shown below
<p align="center">
    <img src="pictures/Model_training.png" alt="drawing" width="500"/>
</p>

## Model inference
At inference stage patches are slided across the image to segment with a 50% overlapping as a stride, and the average probability is calculated for each class, similarly to [Orobix](https://github.com/orobix/retina-unet) approach. The mean IoU obtained on the full image evaluation set is 0.97.

Below is an example of predicted segmentation mask for a full image. The final prediction is obtained as the argmax probability between {background, lines, ellipses}. The dark gray regions in the probabily maps result from patch overlap averaging.
<p align="center">
    <img src="pictures/Model_prediction.png" alt="drawing" width="1000"/>
</p>

## References
[1] Ronneberger O., et al. U-Net: Convolutional Networks for Biomedical Image Segmentation, (2015) <br/>
[2] https://github.com/milesial/Pytorch-UNet <br/>
[3] https://github.com/orobix/retina-unet