import os
import numpy as np
import torch
import torch.nn as nn
from .unet_model import UNet

class UnetFracture:
    
    def __init__(self, params, model_dir='..', e=5):
        
        self.model_path = os.path.join(model_dir,params['path'])
        self.patch_w = params['patch_width']
        self.patch_h = params['patch_height']
        self.stride_w = params['stride_horizontal']
        self.stride_h = params['stride_vertical']
        self.n_channels = params['n_channels']
        self.n_classes = params['n_classes']
        self.e = e
        
    def initialize(self):
        
        self.device = ("cuda" if torch.cuda.is_available() else "cpu" )
        self.net = UNet(n_channels=self.n_channels, n_classes=self.n_classes)
        try:
            self.net.load_state_dict(torch.load(self.model_path))
        except RuntimeError:
            self.net = nn.DataParallel(self.net)
            self.net.load_state_dict(torch.load(self.model_path))
        self.net.to(self.device)
        
        self.net.eval()
      
    def expand_image(self, image, mask=None):
        
        self.img_h, self.img_w = image.shape[:2]
        
        self.n_w = (self.img_w-self.patch_w)//self.stride_w+1
        self.n_h = (self.img_h-self.patch_h)//self.stride_h+1

        leftover_w = (self.img_w-self.patch_w)%self.stride_w
        leftover_h = (self.img_h-self.patch_h)%self.stride_h

        img_expand = np.zeros((self.img_h+self.stride_h-leftover_h, self.img_w+self.stride_w-leftover_w))
        if len(image.shape)==3:
            img_expand = np.repeat(img_expand[:, :, np.newaxis], 3, axis=2)

        img_expand[:self.img_h,:self.img_w] = image
            
        return img_expand
        
    def predict_proba(self, image):
        
        assert image.dtype.type == np.uint8, 'image format should be uint8 (PIL Image)' 
        
        image = image/255   # normalize between 0-1 - consired standard scaling instead
        img_expand = self.expand_image(image)
        
        img_prob = np.zeros(img_expand.shape[:2]+(self.n_classes,))
        # img_sum is used to count pixel oberlapping during for sliding patch prediction
        img_sum = np.zeros(img_expand.shape[:2])


        for i_h in range(self.n_h+1):

            start_h = i_h*self.stride_h
            end_h = start_h+self.patch_h

            for i_w in range(self.n_w+1):

                start_w = i_w*self.stride_w
                end_w = start_w+self.patch_w
                
                patch = img_expand[start_h:end_h,start_w:end_w]
                # mirror patch contours before prediction - U-Net overlap-tile strategy
                patch = input_filled_mirroring(patch, e=self.e)

                if len(patch.shape)==2:
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float()
                if len(patch.shape)==3:
                    patch_tensor = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).float()
                
                pred = self.net(patch_tensor.to(self.device))
                
                prob_predict = torch.sigmoid(pred).squeeze().detach().cpu().numpy()
                prob_predict = np.transpose(prob_predict, (1,2,0))
                
                # crop patch contours after prediction
                img_prob[start_h:end_h,start_w:end_w,:] += prob_predict[self.e:-self.e,self.e:-self.e,:]
                img_sum[start_h:end_h,start_w:end_w] += 1
                
        img_sum = np.dstack([img_sum]*self.n_classes)
        avg_prob = img_prob/img_sum
        avg_prob = avg_prob[:self.img_h,:self.img_w,:] # crop to original image size
        
        return avg_prob
    
    def predict_image(self, image):
        
        prob_predict = self.predict_proba(image)
        mask = np.argmax(prob_predict, axis=2)
        
        return mask
    

def input_filled_mirroring(x, e = 10):      
    '''fill missing data by mirroring the input image contours
    from https://github.com/hansbu/CSE527_FinalProject/blob/master/Utils.py
    '''
    h, w = np.shape(x)[0], np.shape(x)[1]
    y = np.zeros((h + e * 2, w + e * 2))
    if len(x.shape)==3:
        y = np.repeat(y[:, :, np.newaxis], 3, axis=2)
    y[e:h + e, e:w + e] = x
    y[e:e + h, 0:e] = np.flip(y[e:e + h, e:2 * e], 1)  # flip vertically
    y[e:e + h, e + w:2 * e + w] = np.flip(y[e:e + h, w:e + w], 1)  # flip vertically
    y[0:e, 0:2 * e + w] = np.flip(y[e:2 * e, 0:2 * e + w], 0)  # flip horizontally
    y[e + h:2 * e + h, 0:2 * e + w] = np.flip(y[h:e + h, 0:2 * e + w], 0)  # flip horizontally
    return y