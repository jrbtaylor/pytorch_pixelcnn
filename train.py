"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2017-2018
"""

from PIL import Image

import imageio
import json
import os
import time

import numpy as np
from progressbar import ProgressBar
import torch
from torch import nn
from torch.autograd import Variable

from plot import plot_stats


def _clearline():
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE+ERASE_LINE+CURSOR_UP_ONE)


def generate_images(model,seeds,fill=1):
    model.eval()
    img_size = seeds.shape[2:]
    gen = Variable(torch.from_numpy(seeds).cuda())
    bar = ProgressBar()
    print('Generating images...')
    for r in bar(range(int(img_size[0]*(1-fill)),img_size[0])):
        for c in range(img_size[1]):
            out = model(gen)
            p = torch.exp(out)[:,:,r,c]
            sample = p.multinomial(1)
            gen[:,:,r,c] = sample.float()/(out.shape[1]-1)
    _clearline()
    _clearline()
    # print(np.mean(gen.data.cpu().numpy()))
    return (255*gen.data.cpu().numpy()).astype('uint8')


def tile_images(imgs):
    # imgs = list(imgs)
    n = len(imgs)
    h = imgs[0].shape[1]
    w = imgs[0].shape[2]
    r = int(np.floor(np.sqrt(n)))
    while n%r!=0:
        r -= 1
    c = int(n/r)
    imgs = np.squeeze(np.array(imgs),axis=1)
    imgs = np.transpose(imgs,(1,2,0))
    imgs = np.reshape(imgs,[h,w,r,c])
    imgs = np.transpose(imgs,(2,3,0,1))
    imgs = np.concatenate(imgs,1)
    imgs = np.concatenate(imgs,1)
    return imgs


def fit(train_loader,val_loader,model,exp_path,label_preprocess,loss_fcn,
        optimizer='adam',learnrate=1e-4,cuda=True,patience=20,max_epochs=200,
        resume=False):

    if cuda:
        model = model.cuda()

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
    statsfile = os.path.join(exp_path,'stats.json')

    optimizer = {'adam':torch.optim.Adam(model.parameters(),lr=learnrate),
                 'sgd':torch.optim.SGD(
                     model.parameters(),lr=learnrate,momentum=0.9),
                 'adamax':torch.optim.Adamax(model.parameters(),lr=learnrate)
                 }[optimizer.lower()]

    if not resume:
        stats = {'loss':{'train':[],'val':[]},
                 'mean_output':{'train':[],'val':[]}}
        best_val = np.inf
        stall = 0
        start_epoch = 0
    else:
        with open(statsfile,'r') as js:
            stats = json.load(js)
        best_val = np.min(stats['loss']['val'])
        stall = len(stats['loss']['val'])-np.argmin(stats['loss']['val'])-1
        start_epoch = len(stats['loss']['val'])-1
        print('Resuming from epoch %i'%start_epoch)

    def save_img(x,filename):
        Image.fromarray((255*x).astype('uint8')).save(filename)

    def epoch(dataloader,training):
        bar = ProgressBar()
        losses = []
        mean_outs = []
        for x,_ in bar(dataloader):
            y = label_preprocess(x)
            if cuda:
                x,y = x.cuda(),y.cuda()
            x,y = Variable(x),Variable(y)
            if training:
                optimizer.zero_grad()
                model.train()
            else:
                model.eval()
            output = model(x)
            loss = loss_fcn(output,y)
            # track mean output
            output = output.data.cpu().numpy()
            mean_outs.append(np.mean(np.argmax(output,axis=1))/output.shape[1])
            if training:
                loss.backward()
                optimizer.step()
            losses.append(loss.data.cpu().numpy())
        _clearline()
        return float(np.mean(losses)), np.mean(mean_outs)

    # Get seed images for the generation to fill in: ten per class
    seeds = {i:[] for i in range(10)}
    for xb,yb in val_loader:
        xb = list(xb.numpy())
        yb = list(yb.numpy())
        for x,y in zip(xb,yb):
            if len(seeds[y])<10:
                seeds[y].append(x)
        if all([len(s)==10 for s in seeds.values()]):
            break
    seeds = np.array([s for v in seeds.values() for s in v]).astype('float32')
    zero_seeds = np.random.uniform(size=seeds.shape).astype('float32')

    generated = []
    filled_in = []
    for e in range(start_epoch,max_epochs):
        # Training
        t0 = time.time()
        loss,mean_out = epoch(train_loader,training=True)
        time_per_example = (time.time()-t0)/len(train_loader.dataset)
        stats['loss']['train'].append(loss)
        stats['mean_output']['train'].append(mean_out)
        print(('Epoch %3i:    Training loss = %6.4f    mean output = %1.2f    '
               '%4.2f msec/example')%(e,loss,mean_out,time_per_example*1000))

        # Validation
        t0 = time.time()
        loss,mean_out = epoch(val_loader,training=False)
        time_per_example = (time.time()-t0)/len(val_loader.dataset)
        stats['loss']['val'].append(loss)
        stats['mean_output']['val'].append(mean_out)
        print(('            Validation loss = %6.4f    mean output = %1.2f    '
               '%4.2f msec/example')%(loss,mean_out,time_per_example*1000))

        # Generate images and save gif
        # filled_in.append(tile_images(generate_images(model,seeds,0.8)))
        # imageio.mimsave(os.path.join(exp_path, 'filled_in.gif'),
        #                 np.array(filled_in), format='gif', loop=0, fps=2)
        generated.append(tile_images(generate_images(model,zero_seeds)))
        imageio.mimsave(os.path.join(exp_path, 'generated.gif'),
                        np.array(generated), format='gif', loop=0, fps=2)

        # Save results and update plots
        with open(statsfile,'w') as sf:
            json.dump(stats,sf)
        plot_stats(stats,exp_path)

        # Early stopping
        torch.save(model,os.path.join(exp_path,'last_checkpoint'))
        if loss<best_val:
            best_val = loss
            stall = 0
            torch.save(model,os.path.join(exp_path,'best_checkpoint'))
        else:
            stall += 1
        if stall>=patience:
            break






