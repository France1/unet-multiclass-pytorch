import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter

from model import UNet, make_dataloaders, eval_net_loader, make_checkpoint_dir
from lib import plot_net_predictions


def train_epoch(epoch,train_loader,criterion,optimizer,batch_size,scheduler):
    
    net.train()
    epoch_loss = 0
    
    for i, sample_batch in enumerate(train_loader):

        imgs = sample_batch['image']
        true_masks = sample_batch['mask']

        imgs = imgs.to(device)
        true_masks = true_masks.to(device)

        outputs = net(imgs)
        probs = torch.softmax(outputs, dim=1)
        masks_pred = torch.argmax(probs, dim=1)

        loss = criterion(outputs, true_masks)
        epoch_loss += loss.item()

        print(f'epoch = {epoch+1:d}, iteration = {i:d}/{len(train_loader):d}, loss = {loss.item():.5f}')
        # save to summary
        if i%100==0:
            writer.add_scalar('train_loss_iter', 
                                  loss.item(), 
                                  i + len(train_loader) * epoch)
            writer.add_figure('predictions vs. actuals',   
                                  plot_net_predictions(imgs, true_masks, masks_pred, batch_size),    
                                  global_step = i + len(train_loader) * epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch finished ! Loss: {epoch_loss/i:.2f}, lr:{scheduler.get_lr()}')

        
def validate_epoch(epoch,train_loader,val_loader,device):
    
    class_iou, mean_iou = eval_net_loader(net, val_loader, 3, device)
    print('Class IoU:', ' '.join(f'{x:.3f}' for x in class_iou), f'  |  Mean IoU: {mean_iou:.3f}') 
    # save to summary
    writer.add_scalar('mean_iou', mean_iou, len(train_loader) * (epoch+1))
    
    return mean_iou
 

def train_net(train_loader, val_loader, net, device, epochs=5, batch_size=1, lr=0.1, save_cp=True):
    
#     params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4}
#     train_loader, val_loader =  make_dataloaders(dir_data, val_ratio, params)

    print(f'''
    Starting training:
        Epochs: {epochs}
        Batch size: {batch_size}
        Learning rate: {lr}
        Training size: {len(train_loader.dataset)}
        Validation size: {len(val_loader.dataset)}
        Checkpoints: {str(save_cp)}
        Device: {str(device)}
    ''')
          
    optimizer = optim.SGD(net. parameters(),lr=lr, momentum=0.9, weight_decay=0.0005)
    # multiply learning rate by 0.1 after 30% of epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(0.3*epochs), gamma=0.1)
    
    criterion = nn.CrossEntropyLoss()
    # weighted cross entropy loss
#     criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 8.2, 1.0]).cuda()) 
    
    best_precision = 0
    for epoch in range(epochs):
          
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        train_epoch(epoch,train_loader,criterion,optimizer,batch_size,scheduler)
        precision = validate_epoch(epoch,train_loader,val_loader,device)
        scheduler.step()

        if save_cp and (precision>best_precision):
            state_dict = net.state_dict()
            if device=="cuda":
                state_dict = net.module.state_dict()
            torch.save(state_dict, dir_checkpoint+f'CP{epoch + 1}.pth')
            print('Checkpoint {} saved !'.format(epoch + 1))
            best_precision = precision
    
    writer.close()

    
def get_args():
          
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=8,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-f', '--folder', dest='folder', 
                      default='', help='folder name')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    args = get_args()
    
    dir_data = f'./data/{args.folder}'
    dir_checkpoint = f'./checkpoints/{args.folder}_b{args.batchsize}/'
    dir_summary = f'./runs/{args.folder}_b{args.batchsize}'
    params = {'batch_size': args.batchsize, 'shuffle': True, 'num_workers': 4}

    make_checkpoint_dir(dir_checkpoint)
    writer = SummaryWriter(dir_summary)
    
    val_ratio=0.1
    train_loader, val_loader =  make_dataloaders(dir_data, val_ratio, params)
    
    net = UNet(n_channels=3, n_classes=3)
    net.to(device)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))
    
    # train model in parallel on multiple-GPUs
    if torch.cuda.device_count() > 1:
        print("Model training on", torch.cuda.device_count(), "GPUs")
        net = nn.DataParallel(net) 

    try:
        train_net(train_loader, val_loader, net, device, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr)
        
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
