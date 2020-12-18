# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:57:57 2020
@author: Xuandi Fu(Carnegie Mellon University)
""
"""

import configparser
from model import ArticleNet
from torch.utils.data import DataLoader, TensorDataset
import torch
from dataLoader import get_dataloader
import os

# config = configparser.ConfigParser()
# config.read('config.ini')
model_root = 'checkpoints'

def get_config():
    config = {}
    config['num_epoch'] = 50
    config['batch_size'] = 64
    config['lr'] = 0.001
    config['q_len']= 50
    config['q_hidden'] = 128
    config['q_layer'] = 1
    config['tit_len'] = 50
    config['tit_hidden'] = 128
    config['t_layer'] = 1
    config['sent_len'] = 50
    config['sent_hidden'] = 128
    config['s_layer'] = 1
    config['art_hidden'] = 128
    config['num_class'] = 1

    config['val_size'] = 15

    return config

def save_checkpoint(model_dict, epoch):
    torch.save(model_dict, os.path.join(model_root, 'snapshot_ep'+epoch+'.pt'))
    return

def train(config):

    model = ArticleNet(config).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    bceloss = torch.nn.BCELoss()
    train_loader = get_dataloader('train', shuffle=True, batch_size=config['batch_size'])
    val_loader = get_dataloader('val', shuffle=False, batch_size=config['val_size'])

    for epoch in range(config['num_epoch']):
        #=========================== train =====================
        model.train()
        for qbatch, tbatch, sbatch, ybatch, len_batch in train_loader:
            ypred = model.forward(qbatch, tbatch, sbatch, len_batch)
            optimizer.zero_grad()
            loss = bceloss(ypred, ybatch)

            loss.backward()
            optimizer.step()
        #=========================== eval =====================
        model.eval()

        with torch.no_grad():
            for qbatch, tbatch, sbatch, ybatch in val_loader:
                a_title, a_sent, a_art = model.forward(qbatch, tbatch, sbatch)
                #print test score

        # =========================== save model =====================
        save_checkpoint(model.state_dict(), epoch)

if __name__=='__main__':
    config = get_config()
    model_dict = train(config)