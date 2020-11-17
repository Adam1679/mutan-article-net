# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:57:57 2020
@author: Xuandi Fu(Carnegie Mellon University)
"""
import torch
import torch.nn as nn

def myLinear(input_len, output_len):
    return nn.Sequential(
            nn.Linear(input_len, output_len),
            nn.BatchNorm1d(output_len),
            nn.ReLU()
        )

def outputLinear(input_len, output_len):
    return nn.Sequential(
            nn.Linear(input_len, output_len),
            nn.Sigmoid()
    )

class ArticleNet(nn.Module):
    def __init__(self,config):
        super(ArticleNet, self).__init__()
        self.gru_q = nn.GRU(config['q_len'], config['q_hidden'], config['q_layer'])
        self.gru_title = nn.GRU(config['tit_len'], config['tit_hidden'], config['t_layer'])
        self.gru_sent = nn.GRU(config['sent_len'], config['sent_hidden'], config['s_layer'])
        self.fc_title = myLinear(config['tit_hidden'], config['art_hidden'])
        self.fc_sent = myLinear(config['sent_hidden'], config['art_hidden'])

        self.title_class = outputLinear(config['art_hidden'], config['num_class']) # score num_class = 1
        self.sent_class = outputLinear(config['art_hidden'], config['num_class'])
        self.art_class = outputLinear(config['art_hidden'], config['num_class'])

    def forward(self, question, title, sents, length):

        #image: the  visual features Vtaken from an ImageNet trained ResNet152
        #question: question feature
        question = question.unsqueeze(0)
        title = title.unsqueeze(0)
        sents= sents.transpose(0,1)
        _, hq = self.gru_q(question)
        #hqv = hq + image

        atitle, h_title = self.gru_title(title, hq)
        #hq = hq.repeat(100, 1, 1)
        asents, h_sent = self.gru_sent(sents, hq)
        title_emb = self.fc_title(h_title)
        sent_emb = self.fc_sent(h_sent)
        hart = title_emb + sent_emb
        atitle = self.title_class(h_title + hart)
        aart = self.art_class(hart)
        asents = self.sent_class(h_sent + hart)

        return atitle, aart, asents


