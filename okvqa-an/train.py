# -*- coding: utf-8 -*-
"""
Created on Nov  6 15:57:57 2020
"
@author: Xuandi FU
"
"""
import configparser
from model import ArticleNet
from torch.utils.data import DataLoader, TensorDataset
import torch

config = configparser.ConfigParser()
config.read('config.ini')

def train(dataloader):
    model = ArticleNet(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    bceloss = torch.nn.functional.binary_cross_entropy()

    for epoch in range(config['num_epoch']):
        #train
        model.train()
        for qbatch, vbatch, tbatch, sbatch, ybatch in train_loader:

            ypred = model.forward(qbatch, vbatch, tbatch, sbatch)

            optimizer.zero_grad()
            loss = bceloss(ypred, ybatch)

            loss.backward()
            optimizer.step()

        #evaluate
        model.eval()

        with torch.no_grad():
            for qbatch, vbatch, tbatch, sbatch, ybatch in val_loader:
                a_title, a_sent, a_art = model.forward(qbatch, vbatch, tbatch, sbatch)
                #print test score

if __name__=='__main__':

    model_dict = train(config)
