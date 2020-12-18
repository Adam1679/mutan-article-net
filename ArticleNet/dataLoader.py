# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:57:57 2020
@author: Xuandi Fu(Carnegie Mellon University)
"""

import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import sys
from tqdm import trange

root_path = 'data'

class MyDataset(Dataset):
    def __init__(self, question, title, sentences, labels, length):
        self.question = question
        self.title = title
        self.sentences = sentences
        self.labels = labels
        self.length = length

    def __len__(self):
        self.len = len(self.labels)
        return self.len

    def __getitem__(self, index):
        # title_score = self.labels[index][0]
        # sent_score = self.labels[index][1:-2]
        # doc_score = self.labels[index][-1]
        # print(len(self.sentences[index]))
        # print(len(self.labels[index]))
        sent = np.asarray(self.sentences[index])
        return self.question[index], self.title[index], sent,self.labels[index], self.length[index]

def load_data(setpath):

    with open(os.path.join(root_path, setpath, 'question.npy'), 'rb') as f:
        question = np.load(f, allow_pickle=True)
    with open(os.path.join(root_path, setpath, 'title.npy'), 'rb') as f:
        title = np.load(f, allow_pickle=True)

    with open(os.path.join(root_path, setpath, 'sentence_pad.npy'), 'rb') as f:
        sentences = np.load(f, allow_pickle=True)

    with open(os.path.join(root_path, setpath, 'label_pad.npy'), 'rb') as f:
        label = np.load(f, allow_pickle=True)

    with open(os.path.join(root_path, setpath, 'length.npy'), 'rb') as f:
        length = np.load(f, allow_pickle=True)

    print(sentences.shape)
    #title = np.loadtxt('data/title_train.npy', delimiter=',')
    return question, title, sentences, label, length


def get_dataloader(setpath, shuffle, batch_size):
    question, title, sentences, label, length = load_data(setpath)
    dataset = MyDataset( question, title, sentences, label, length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def pad_sentence(setpath):
    sent_length = []
    with open(os.path.join(root_path, setpath, 'sentence.npy'), 'rb') as f:
        sentences = np.load(f, allow_pickle=True)
        for i, sent in enumerate(sentences):
            sent_length.append(len(sent))
            print(sent_length)
            sentences[i] = np.pad(sentences[i], (0, 1669 - len(sent)), 'constant', constant_values=0)
            print(sentences[i].shape)
            break

def reduce_data(setpath):

    max_length = 100
    sent_length = []
    with open(os.path.join(root_path, setpath, 'sentence.npy'), 'rb') as f:
        sentences = np.load(f, allow_pickle=True)

    with open(os.path.join(root_path, setpath, 'label.npy'), 'rb') as f:
        labels = np.load(f, allow_pickle=True)

    length = []
    for i in trange(len(labels), file=sys.stdout, desc ='outerloop'):
        label = labels[i]
        length.append(len(label)-2)
        if len(label) > max_length+2:

            if label[-1] == 0:
                sentences[i] = sentences[i][:max_length]
                labels[i] = label[:max_length+2]
                if len(sentences[i]) != 100:
                    break
                if len(labels[i]) != 102:
                    break
            else:
                index = np.where(np.asarray(label[1:-1])==1)
                one_length = len(index[0])
                replace = np.take(sentences[i], index[0], axis=0).tolist()
                if one_length >= max_length:
                    sentences[i] = sentences[i][:max_length]
                    labels[i] = label[:max_length+1] + [label[-1]]
                else:
                    sentences[i] = replace+sentences[i][one_length:max_length]
                    replace = np.ones(one_length).astype(int).tolist()
                    labels[i] = [label[0]] + replace + label[one_length+1:max_length+1]+[1]
                if len(sentences[i]) != 100:
                    break
                if len(labels[i]) != 102:
                    break
        else:
            pad_length = max_length-len(sentences[i])
            labels[i] = label[:-1] + np.zeros(pad_length).tolist() + [label[-1]]
            sentences[i] = sentences[i] + np.zeros((pad_length,50)).tolist()
            if len(sentences[i]) != 100:
                break
            if len(labels[i]) != 102:
                break

    sentences = np.asarray(sentences)
    with open(os.path.join(root_path, setpath, 'sentence_pad.npy'), 'wb') as f:
        np.save(f, sentences)

    with open(os.path.join(root_path, setpath, 'label_pad.npy'), 'wb') as f:
        np.save(f, labels)

    length = np.asarray(length)
    with open(os.path.join(root_path, setpath, 'length.npy'), 'wb') as f:
        np.save(f, length)

# reduce_data('val')
# reduce_data('train')