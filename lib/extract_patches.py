import os
import random
import cv2
from pathlib import Path
import numpy as np
from PIL import Image

from lib.utils import mask2rgb, make_image_dir


def random_patches(image, mask, n=1000, patch_h=48, patch_w=48):
    '''
    Extract randomly cropped images and masks. Adapted from:
    https://github.com/orobix/retina-unet/blob/master/lib/extract_patches.py
    
    Inputs:
        image : array 
            grayscale or RGB image
        mask : array 
            RGB image
        n : int
            number of patches to extract from image
        patch_h : int
            patch height
        patch_w : int
            patch width
    
    Outputs:
        patches : list[array]
            extracted patches
        patch_masks : list[array]
            mask of extracted patches
    '''
    
    img_h, img_w = image.shape[:2]

    patches = []
    patch_masks = []

    for _ in range(n):

        x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
        y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))

        patch = image[y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
        patch_mask = mask[y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            
        patches.append(patch)
        patch_masks.append(patch_mask)
            
    return patches, patch_masks
            
def input_filled_mirroring(x, e = 10):      
    '''Fill missing data by mirroring the input image contours (see Figure 2 from Ronneberger et al.). 
    Adapted from https://github.com/hansbu/CSE527_FinalProject/blob/master/Utils.py

    Inputs:
        x : array 
            grayscale or RGB image patch
        
    Outputs:
        y : array 
            expanded grayscale or RGB image patch

    '''
    h, w = np.shape(x)[0], np.shape(x)[1]
    y = np.zeros((h + e * 2, w + e * 2))
    y[e:h + e, e:w + e] = x
    y[e:e + h, 0:e] = np.flip(y[e:e + h, e:2 * e], 1)  # flip vertically
    y[e:e + h, e + w:2 * e + w] = np.flip(y[e:e + h, w:e + w], 1)  # flip vertically
    y[0:e, 0:2 * e + w] = np.flip(y[e:2 * e, 0:2 * e + w], 0)  # flip horizontally
    y[e + h:2 * e + h, 0:2 * e + w] = np.flip(y[h:e + h, 0:2 * e + w], 0)  # flip horizontally
    return y

def augment_rectangular(data):
    '''agument annotation masks with all combinations of flipping up&down and left&right

    Inputs:
        data : tuple[list]
            list of patch images and masks
    
    Outputs:
        data_aug : tuple[list]
            list of augmented patch images and masks

    '''
    
    data_aug  = []
    for patch,mask in data:
        patch_ud = np.flipud(patch)
        mask_ud = np.flipud(mask)
        patch_lr = np.fliplr(patch)
        mask_lr = np.fliplr(mask)
        patch_lr_ud = np.flipud(patch_lr)
        mask_lr_ud = np.flipud(mask_lr)
    
        data_aug.extend([(patch,mask), (patch_lr,mask_lr), (patch_ud,mask_ud), (patch_lr_ud,mask_lr_ud)])
    
    return data_aug

def save_patches(export_dir, data):
    '''Export patches and masks for model training
        
    Inputs:
        export_dir : str
            path of directory in which images will be saved
        data : tuple[list]
            list of patch images and masks
    
    Outputs:
        None
    '''
    
    save_dir_images = os.path.join(export_dir, 'images')
    save_dir_masks = os.path.join(export_dir, 'masks')

    make_image_dir(save_dir_images)
    make_image_dir(save_dir_masks)

    for i, (patch, patch_mask) in enumerate(data):

        file_name = f'p{i}'
        save_image_path = os.path.join(save_dir_images, file_name + '.png')
        save_mask_path = os.path.join(save_dir_masks, file_name + '.png')

        patch = Image.fromarray(np.uint8(patch))
        patch_mask = Image.fromarray(mask2rgb(patch_mask))
        
        patch.save(save_image_path)
        patch_mask.save(save_mask_path)