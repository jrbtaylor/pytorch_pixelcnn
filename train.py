"""
Written by Jason Taylor <jasonrbtaylor@gmail.com> 2017-2018
"""

import json
import os
import time

import numpy as np
import progressbar
import torch
from torch import nn
from torch.autograd import Variable

from plot import plot_stats


def _clearline():
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE+ERASE_LINE+CURSOR_UP_ONE)


def fit(train_loader,val_loader,model,exp_path,label_preprocess,loss_fcn,
        optimizer='adam',learnrate=1e-3,cuda=True,patience=20,max_epochs=200,
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
        stats = {'loss':{'train':[],'val':[]}}
        best_val = np.inf
        stall = 0
        start_epoch = 0
    else:
        with open(statsfile,'r') as js:
            stats = json.load(js)
        best_val = np.min(stats['loss']['val'])
        stall = len(stats['loss']['val'])-np.argmin(stats['loss']['val'])-1
        start_epoch = len(stats['loss']['val'])

    def epoch(dataloader,training):
        bar = progressbar.ProgressBar()
        losses = []
        for x,_ in bar(dataloader):
            y = label_preprocess(x)
            if cuda:
                x,y = x.cuda(),y.cuda()
            x,y = Variable(x),Variable(y)
            if training:
                optimizer.zero_grad()
            output = model(x)
            loss = loss_fcn(output,y)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(loss.data.cpu().numpy())
        _clearline()
        return float(np.mean(losses))

    for e in range(start_epoch,max_epochs):
        # Training
        t0 = time.time()
        loss = epoch(train_loader,training=True)
        time_per_example = (time.time()-t0)/len(train_loader.dataset)
        stats['loss']['train'].append(loss)
        print(('Epoch %3i:    Training loss = %6.4f    %4.2f msec/example')
              %(e,loss,time_per_example*1000))

        # Validation
        t0 = time.time()
        loss = epoch(val_loader,training=False)
        time_per_example = (time.time()-t0)/len(val_loader.dataset)
        stats['loss']['val'].append(loss)
        print(('            Validation loss = %6.4f    %4.2f msec/example')
              %(loss,time_per_example*1000))

        # Save results and update plots
        with open(statsfile,'w') as sf:
            json.dump(stats,sf)
        plot_stats(stats,exp_path)

        # TODO: generate an image every epoch and output to a gif at the end

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






