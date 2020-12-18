# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:57:57 2020
@author: Xuandi Fu(Carnegie Mellon University)
"""

import json
import numpy as np
from language.glove import Glove
import nltk
from string import punctuation
import numpy
from tqdm import trange
import sys

glove_path = 'data/glove.6B.50d.txt'
train_path = 'data/okvqa_an_input_train.txt'
val_path = 'data/okvqa_an_input_val.txt'
glove_obj = Glove(glove_path)

translate_table = dict((ord(char), None) for char in punctuation)

def __process_sent(sent):
    return sent.lower().translate(translate_table)

def get_sent_embedding(sentence, keep_pos=None):
    sentence = __process_sent(sentence)
    word_tokens = nltk.word_tokenize (sentence)
    tokens = nltk.pos_tag(word_tokens)
    tokens = [item[0] for item in tokens if keep_pos is None or item[1] in keep_pos]
    dim = glove_obj.dim
    sent_emb = np.zeros(dim)
    cnt = 0
    for word in tokens:
        emb = glove_obj.embedding.get(word, None)
        if emb is not None:
            sent_emb += emb
    return sent_emb / cnt if cnt > 0 else sent_emb

question_emb = []
title_emb = []
sentence_emb = []

labels = []

def loadData(path):
    train_file = open(path, 'r')
    line_num = 0
    pos_label = 0
    lines = train_file.readlines()
    length = []
    for i in trange(len(lines), file=sys.stdout, desc ='outerloop'):
        train_data = json.loads(lines[i])
        line_num += 1
        #
        # if train_data['docHasAns'] == 1:
        #     pos_label += 1
        # print(train_data['question'])
        # print(train_data.keys())

        #===================load embedding==================
        question_emb.append(get_sent_embedding(train_data['question']))
        #title_emb.append(get_sent_embedding(train_data['title']))


        sentence = []
        #sentence_label = []
        length.append(len(train_data['sentences_pairs']))
        for pair in train_data['sentences_pairs']:
            sentence.append(get_sent_embedding(pair[0]))
            #sentence_label.append(pair[1])
        sentence_emb.append(sentence)


        #=================load label======================
        #labels.append([train_data['titleHasAns']]+sentence_label+[train_data['docHasAns']])
        #print(get_sent_embedding(train_data['question']))

def saveData():
    global question_emb, title_emb, sentence_emb

    question_emb = numpy.array(question_emb)
    sentence_emb = numpy.array(sentence_emb)

    with open('data/train/question.npy', 'wb') as f:
        np.save(f, question_emb)
    with open('data/train/sentence.npy', 'wb') as f:
        np.save(f, sentence_emb)

def saveLabel():
    global labels
    global title_emb
    labels = numpy.array(labels)
    title_emb = numpy.array(title_emb)

    with open('data/train/label.npy', 'wb') as f:
        np.save(f, labels)
    with open('data/train/title.npy', 'wb') as f:
        np.save(f, title_emb)

loadData(train_path)
saveData()
