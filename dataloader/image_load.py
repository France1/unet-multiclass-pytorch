import os
from glob import glob
import natsort
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from PIL import Image
import re
import json

from dataloader.utils import *
from lib.utils import *

def get_image_names(mode='train'):

    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    params = params['models_settings']

    if mode=='train':
        data_dir = params['train_dir']
    if mode=='test':
        data_dir = params['test_dir']

    file_names = []
    for filename in natsort.natsorted(glob(os.path.join(data_dir, params['image_folder'], '*'))): 
            file_names.append(filename.split('/')[-1])
            
    return file_names

def list_labels(file_names, mode='train'):

    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    params = params['models_settings']

    masks = load_masks(file_names,mode=mode)
    shape_to_label = params['label_to_value']
    label_to_shape = {v:k for k,v in shape_to_label.items()}
    
    labels = set()
    for mask in masks:  
        mask = rgb2mask(mask)
        labels = labels.union(set([label_to_shape[label] for label in np.unique(mask)]))
        
    return labels

def get_sizes(image_names, mode='train'):

    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    params = params['models_settings']

    if mode=='train':
        data_dir = params['train_dir']
    if mode=='test':
        data_dir = params['test_dir']

    h = []
    w = []
    
    for image_name in image_names:

        file_name = os.path.join(data_dir, params['image_folder'],  image_name)
        image = np.array(Image.open(file_name))
        
        h.append(image.shape[0])
        w.append(image.shape[1])
        
    d = {'h-range': [min(h), max(h)],
         'w-range': [min(w), max(w)]}
    
    return d 

def load_images(image_names, mode='train'):

    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    params = params['models_settings']

    if mode=='train':
        data_dir = params['train_dir']
    if mode=='test':
        data_dir = params['test_dir']

    resize_w = params['resize_width']
    equalize = bool(params['equalize'])
      
    images = []
    for image_name in image_names:
        
        file_name = os.path.join(data_dir, params['image_folder'],  image_name)
        image = Image.open(file_name)
        
        if resize_w is not None: 
            orig_w, orig_h = image.size[:2]
            resize_h = int(resize_w/orig_w*orig_h)
            image = np.array(image.resize((resize_w,resize_h), Image.BILINEAR))
            
        images.append(image)
        
    return images

def load_masks(image_names, mode='train'):
    
    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    params = params['models_settings']

    if mode=='train':
        data_dir = params['train_dir']
    if mode=='test':
        data_dir = params['test_dir']

    resize_w = params['resize_width']

    masks = []
    for image_name in image_names:  
        image_name = re.sub("Img", "Mask", image_name)
        file_name = os.path.join(data_dir, params['mask_folder'],  image_name)
        mask = Image.open(file_name)
        if resize_w is not None:
            orig_w, orig_h = mask.size[:2]
            resize_h = int(resize_w/orig_w*orig_h)
            mask = mask.resize((resize_w,resize_h), Image.NEAREST)

        masks.append(np.array(mask))
        
    return masks
