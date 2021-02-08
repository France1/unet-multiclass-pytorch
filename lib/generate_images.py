import numpy as np
from skimage.morphology import dilation, disk
from skimage.draw import line,ellipse
from scipy.stats import truncnorm
import os
import json

from .utils import mask2rgb

PATH_PARAMETERS = '../params.json'


def make_background(img_w, img_h, mean=0.8, std=0.05, clip_min=0.6, clip_max=1.0):
    '''
    Generate Truncated Gaussian noise image background
    
    Inputs:
        img_w : int 
            image width
        img_wh : int 
            image height
        mean : float
            mean of Gaussian 
        std : float
            std of Gaussian
        clip_min : float
            min of Truncated Gaussian
        clip_max : float
            max of Truncated Gaussian
    
    Outputs:
        background : array
            background image
    '''
    
    a, b = (clip_min - mean) / std, (clip_max - mean) / std
    background = truncnorm.rvs(a, b, loc=mean, scale=std, size=(img_h,img_w))
    
    return background

def make_ellipse(img_w, img_h, centre_r, centre_c, radius_r, radius_c, angle_deg):
    '''
    Generate row, col coordinates of an elliptic region onto an image
    
    Inputs:
        img_w : int 
            image width
        img_h : int 
            image height
        centre_r : int
            row center coordinate 
        centre_c : int
            column center coordinate 
        radius_c : int
            radius along colum coordinate
        radius_r : int
            radius along row coordinate
        angle_deg : int
            orientation of ellipse in degree wrt column coordinate
    
    Outputs:
        (rows, columns) : tuple[list]
            lists of coordinate of the ellipse region in for rows and columns
    '''
    
    image_ellipse = np.zeros((img_h,img_w),dtype=bool)
    rr, cc = ellipse(centre_r, centre_c, radius_r, radius_c, rotation=np.deg2rad(angle_deg))
    mask_rr = rr>=img_h
    mask_cc = cc>=img_w
    rr[mask_rr] = 0
    cc[mask_cc] = cc[mask_cc]-img_w
    image_ellipse[rr, cc] = True
    
    return np.nonzero(image_ellipse)

def make_line(width,height,xc,yc,theta,l,thickness):
    '''
    Generate row, col coordinates of a line onto an image
    
    Inputs:
        width : int 
            image width
        height : int 
            image height
        xc : int
            center of the line in column coordinates 
        yc : int
            center of the line in row coordinates 
        theta : int
            orientation of the line in degree wrt column coordinates
        thickness : int
            thickness of the line

    Outputs:
        (rows, columns) : tuple[list]
            lists of coordinate of the line region in for rows and columns
    '''
    
    theta = np.deg2rad(-theta)
    # line end points
    xa = int(xc+l/2*np.cos(theta))
    ya = int(yc+l/2*np.sin(theta))
    xb = int(xc-l/2*np.cos(theta))
    yb = int(yc-l/2*np.sin(theta))
    # draw line mask
    img_line = np.zeros((height,width),dtype=bool)
    rr, cc = line(ya, xa, yb, xb)
    mask = (cc>=0) & (cc<width) & (rr>=0) & (rr<height) 
    rr = rr[mask]
    cc = cc[mask]

    img_line[rr, cc] = True
    img_line = dilation(img_line, selem=disk(thickness))
    
    return np.nonzero(img_line)

def generate_params(width, height, shape):
    '''
    Generate random parameter of a gemoteric shape
    
    Inputs:
        width : int 
            image width
        height : int 
            image height
        shape : string
            type of shape: {'line', 'ellipse'}

    Outputs:
        params : dict
            dictionary of shape parameters
    '''
    
    params = {'x_centre': np.random.randint(low=0, high=width),
              'y_centre': np.random.randint(low=0, high=height),
              'phi': np.random.randint(low=0, high=180),
              'intensity': np.random.uniform(low=0.0, high=0.7)
             }  
    if shape == 'ellipse':
        params['shape'] = 'ellipse'
        params['x_radius'] = np.random.randint(low=5, high=20)
        params['y_radius'] = params['x_radius']*np.random.uniform(low=0.5, high=2)   
    if shape == 'line':
        params['shape'] = 'line'
        params['length'] = np.random.randint(low=20, high=width)
        params['y_radius'] = np.random.randint(low=5, high=30)
        params['thickness'] = np.random.randint(low=1, high=3)
            
    return params

def generate_shapes(width, height, max_ellipses=10, max_lines=10):
    '''
    Generate parameters of a random number of shapes
    
    Inputs:
        width : int 
            image width
        height : int 
            image height
        max_ellipses : int
            max number of generated ellipses
        max_lines : int
            max number of generated lines

    Outputs:
        shapes : dict
            dictionary of shape parameters for lines and ellipses
    '''
    
    shapes = {}
    
    n_e = np.random.randint(low=1, high=max_ellipses)
    p_e = [generate_params(width, height, 'ellipse') for _ in range(n_e)]
    shapes['ellipses'] = p_e
    n_l = np.random.randint(low=1, high=max_ellipses)
    p_l = [generate_params(width, height, 'line') for _ in range(n_e)]
    shapes['lines'] = p_l
    
    return shapes

def make_image(width,height):
    '''
    Generate full image - rely on json file specified by PATH_PARAMETERS for additional settings
    
    Inputs:
        width : int 
            image width
        height : int 
            image height

    Outputs:
        image : array
            simulated image
        mask_label : arrry
            mask of simulated image with labes specified in PATH_PARAMETERS
    '''

    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    shape_to_label = params['models_settings']['label_to_value']
    
    mean_background = np.random.uniform(low=0.7, high=0.9)
    
    shapes = generate_shapes(width, height, max_ellipses=10, max_lines=10)

    img = make_background(width, height, mean_background, 0.1)
    mask_label = np.zeros_like(img, dtype=int)

    for p in shapes['ellipses']:
        rr,cc = make_ellipse(width, height, p['y_centre'], p['x_centre'], p['x_radius'], 
                             p['y_radius'], p['phi'])
        img[rr,cc] = p['intensity']
        mask_label[rr,cc] = shape_to_label['ellipse']

    for p in shapes['lines']:
        rr,cc = make_line(width, height, p['x_centre'], p['y_centre'], p['phi'], 
                          p['length'], p['thickness'])
        img[rr,cc] = p['intensity']
        mask_label[rr,cc] = shape_to_label['line']

    # RGB image
    img = np.stack([np.uint8(255*img)]*3, axis=2)
    
    return img, mask_label