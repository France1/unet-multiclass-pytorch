import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def compute_IoU(cm):
    '''
    Adapted from:
        https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py
        https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/metrics.py#L2716-L2844
    '''
    
    sum_over_row = cm.sum(axis=0)
    sum_over_col = cm.sum(axis=1)
    true_positives = np.diag(cm)

    # sum_over_row + sum_over_col = 2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives
    
    iou = true_positives / denominator
    
    return iou, np.nanmean(iou) 

def eval_net_loader(net, val_loader, n_classes, device='cpu'):
    
    net.eval()
    labels = np.arange(n_classes)
    cm = np.zeros((n_classes,n_classes))
      
    for i, sample_batch in enumerate(val_loader):
            imgs = sample_batch['image']
            true_masks = sample_batch['mask']
            
            imgs = imgs.to(device)
            true_masks = true_masks.to(device)

            outputs = net(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            for j in range(len(true_masks)): 
                true = true_masks[j].cpu().detach().numpy().flatten()
                pred = preds[j].cpu().detach().numpy().flatten()
                cm += confusion_matrix(true, pred, labels=labels)
    
    class_iou, mean_iou = compute_IoU(cm)
    
    return class_iou, mean_iou

def IoU(mask_true, mask_pred, n_classes=2):
        
        labels = np.arange(n_classes)
        cm = confusion_matrix(mask_true.flatten(), mask_pred.flatten(), labels=labels)
        
        return compute_IoU(cm)