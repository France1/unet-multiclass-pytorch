import shutil
import os
from pathlib import Path
import numpy as np
from PIL import Image

LABEL_TO_COLOR = {0:[0,0,0], 1:[255,0,0], 2:[0,255,0], 3:[0,0,255]}

def make_image_dir(path_dir):
    path = Path(path_dir)
    # remove folder if it exists
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)

def mask2rgb(mask):
    
    rgb = np.zeros(mask.shape+(3,), dtype=np.uint8)
    
    for i in np.unique(mask):
            rgb[mask==i] = LABEL_TO_COLOR[i]
            
    return rgb

def rgb2mask(rgb):
    
    mask = np.zeros((rgb.shape[0], rgb.shape[1]))

    for k,v in LABEL_TO_COLOR.items():
        mask[np.all(rgb==v, axis=2)] = k
        
    return mask

def save_images(export_dir, data):
    
    save_dir_images = os.path.join(export_dir, 'images')
    save_dir_masks = os.path.join(export_dir, 'masks')

    make_image_dir(save_dir_images)
    make_image_dir(save_dir_masks)

    for i, (img, mask) in enumerate(data):

        save_image_path = os.path.join(save_dir_images, f'Img_{i}.png')
        save_mask_path = os.path.join(save_dir_masks, f'Mask_{i}.png')

        img = Image.fromarray(img)
        mask = Image.fromarray(mask2rgb(mask))
        
        img.save(save_image_path)
        mask.save(save_mask_path)