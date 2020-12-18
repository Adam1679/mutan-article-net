# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:57:57 2020
@author: Xuandi Fu(Carnegie Mellon University)
"""

import json
from language.glove import QuestionAnswerPair
import sys
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.manifold import TSNE
from matplotlib import cm

val_path = 'data/query_train.json'
answer_path = "./data/mscoco_train2014_annotations.json"
question_path = "./data/OpenEnded_mscoco_train2014_questions.json"
article_path = './data/okvqa_an_input_train.txt'
obj = QuestionAnswerPair (answer_path=answer_path, question_path=question_path)

article2query = {}
query2question = {}
question2answer = {}
#article -> question_id -> answer
val_question_id = []
import numpy as np

acc = {}

acc['Vehicles and Transportation'] =4.48
acc['Brands, Companies and Product']=0.93
acc['Objects, Material and Clothing'] =5.09
acc['Sports and Recreation'] =5.11
acc['Cooking and Food'] =5.69
acc['Geography, History, Language and Culture']=6.24
acc['People and Everyday life'] =3.13
acc['Plants and Animals'] =6.95
acc['Science and Technology'] =5.00
acc['Weather and Climate']=9.92
acc['Other'] =5.33

def load_dict():
    val_data = json.load(open(val_path,'r'))
    print(len(val_data))
    for line in val_data:
        #query2question[val_data['query']] = val_data['question_id']
        val_question_id.append(line['question_id'])

def plot(x, y, type, xlabel, ylabel):
    df_subset = pd.DataFrame()

    df_subset[xlabel] = x
    df_subset[ylabel] = y
    df_subset['question category'] = type
    df_subset['acc'] = df_subset['question category'].map(acc)

    print(type)
    mean_set = df_subset.groupby(['question category']).mean()
    # mean_set['acc'] = []
    # mean_set['acc'] = mean_set['question category'].map(acc)

    num_type = len(set(type))

    cmap = cm.get_cmap('Blues')

    colors = cmap(np.asarray(list(acc.values())) / float(max(acc.values())))
    fig, ax = plt.subplots()

    plt.figure(figsize=(16, 6))

    people = list(mean_set[ylabel].axes[0])
    y_pos = np.arange(len(people))
    performance = mean_set[ylabel].array
    #error = np.random.rand(len(people))

    ax.barh(y_pos, performance, color=colors, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(people, fontsize=12)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Rate of positive sentence in retrieve article', fontsize=12)
    plt.subplots_adjust(left=0.4, right=0.9)
    ax.set_title('Rate of positive sentence in retrieve article vs. Category')

    plt.show()
    plt.clf()
    performance = mean_set[xlabel].array
    ax.barh(y_pos, performance, color=colors, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(people, fontsize=12)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Rate of positive sentence in retrieve article', fontsize=12)
    plt.subplots_adjust(left=0.4, right=0.9)
    ax.set_title('Rate of positive sentence in retrieve article vs. Category')
    plt.show()
    # p = sns.scatterplot(
    #     x=xlabel, y='acc',
    #     hue="question category",
    #     palette=sns.color_palette("hls", num_type),
    #     data=mean_set,
    #     legend="full",
    #     alpha=1,
    #     s=100,
    # )
    # p.set_xlabel(xlabel, fontsize=20)
    # p.set_ylabel('accuracy(percentage)', fontsize=20)
    # p.set_title(xlabel+' vs. '+'accuracy', fontsize=20)
    # plt.savefig(xlabel+'_'+'acc.png')
    # plt.cla()
    #
    # sns.scatterplot(
    #     x=ylabel, y='acc',
    #     hue="question category",
    #     palette=sns.color_palette("hls", num_type),
    #     data=mean_set,
    #     legend="full",
    #     alpha=1,
    #     s=100
    # )
    # p.set_xlabel(ylabel, fontsize=20)
    # p.set_ylabel('accuracy(percentage)', fontsize=20)
    # p.set_title(ylabel + ' vs. ' + 'accuracy', fontsize=20)
    # plt.savefig(ylabel + '_' + 'acc.png')
    # plt.cla()
    #
    # sns.scatterplot(
    #     x=xlabel, y=ylabel,
    #     hue="question category",
    #     palette=sns.color_palette("hls", num_type),
    #     data=mean_set,
    #     legend="full",
    #     alpha=1,
    #     s=100
    # )
    # p.set_xlabel(xlabel, fontsize=20)
    # p.set_ylabel(ylabel, fontsize=20)
    # p.set_title(ylabel + ' vs. ' + ylabel, fontsize=20)
    # plt.savefig(xlabel+'_'+ylabel+'.png')
    # plt.cla()
    return

answer_type = []
def load_article():
    global article_path
    train_file = open(article_path, 'r')
    lines = train_file.readlines()
    length = []
    query_length = []
    doc_label = []
    pos_sent = []
    article_length = []
    pos_rate = []

    article_positive_rate = []

    for i in trange(10000, file=sys.stdout, desc='outerloop'):
        train_data = json.loads(lines[i])
        query_length.append(len(train_data['question']))
        doc_label.append(train_data['docHasAns'])
        pos = 0
        senten_length = 0

        for pair in train_data['sentences_pairs']:
            pos += pair[1]
            senten_length += len(pair[0])

        pos_sent.append(pos)
        pos_rate.append(pos/len(train_data['sentences_pairs']))
        article_length.append(senten_length/len(train_data['sentences_pairs']))
        answer_type.append(obj.questionid2type[val_question_id[int(i/15)]])

    #plot(query_length, pos_sent, answer_type, 'number of query entities', 'number of positive sentence retrieved')
    plot(query_length, pos_rate, answer_type, 'number of query entities', 'retrieved rate of positive sentence')
    #plot(query_length, article_length, answer_type, 'number of query entities', 'retrieved article length')

load_dict()
load_article()

def tsne_plot():
    tsne = TSNE(n_components=2)
    with open(os.path.join('data', 'val', 'sentence_pad.npy'), 'rb') as f:
        sentences = np.load(f, allow_pickle=True)

    sentences = sentences[:10000]
    for i, sent in enumerate(sentences):
        sentences[i] = np.sum(np.stack(sent, axis=0),axis=0)
    sentences = np.stack(sentences)
    tsne_results = tsne.fit_transform(sentences)
    df_subset = pd.DataFrame(tsne_results)

    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    df_subset['type'] = answer_type

    num_type = len(set(answer_type))
    plt.figure(figsize=(16, 10))

    p = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="type",
        palette=sns.color_palette("hls", num_type),
        data=df_subset,
        legend="full",
        alpha=1
    )
    p.set_xlabel("tsne-2d-one", fontsize=20)
    p.set_ylabel("tsne-2d-two", fontsize=20)
    p.set_title('Retrieved Article Hidden states', fontsize=20)
    plt.savefig('article_question_category.png')
    return

tsne_plot()
