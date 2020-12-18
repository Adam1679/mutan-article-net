import argparse
import os
import shutil
import yaml
import json
import click
from pprint import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import vqa.lib.engine as engine
import vqa.lib.utils as utils
import vqa.lib.logger as logger
import vqa.lib.criterions as criterions
import vqa.datasets as datasets
import vqa.models as models


if __name__ == "__main__":



    # n_gram_analysis
    import nltk
    from nltk import word_tokenize
    from nltk.tag import pos_tag
    nltk.download ('averaged_perceptron_tagger')
    import json
    path = "./logs/okvqa/mutan_att_train/epoch_499/OpenEnded_mscoco_val2014_model_results.json"
    data = json.load(open(path))

    def preprocess(sent) :
        sent = nltk.word_tokenize (sent)
        sent = nltk.pos_tag (sent)
        return sent

    import pandas as pd
    def get_ans(s):
        return s.split("__")[0], int(s.split("__")[1])/2
    clean_json = []
    # data clean
    for item in data:
        ans, fr = get_ans (item['answers_occurence'])
        item['raw answer'] = ans.lower()
        item['integrity'] = fr
        if fr >= 4.0:
            clean_json.append(item)

    with open("./logs/okvqa/mutan_att_train/epoch_499/OpenEnded_mscoco_val2014_model_results_clean.json", "w") as f:
        f.write(json.dumps(clean_json))

    # integrity
    print ("integrity counts")
    ints = [(item['integrity'], int (data[i]['raw answer'] == data[i]['answer'].lower ())) for i, item in
            enumerate (data)]
    print (pd.DataFrame (ints).groupby (by=0).count ())
    print ("integrity relation with accuracy")
    print (pd.DataFrame (ints).groupby (by=0).mean ())

    data = clean_json
    sents = [s['question_raw'] for s in data]
    ans = [s['true_answer'] for s in data]
    correct = {}
    wrong = {}
    freq = {}
    print("accuracy analysis based (only answers with high confidence)")
    for i, sent in enumerate(sents):
        words = word_tokenize (sent)[:2]
        s = " ".join (words)
        freq[s] = freq.get(s, 0) + 1
        ans = data[i]['raw answer']
        pred = data[i]['answer'].lower()
        if ans == pred:
            correct[s] = correct.get (s, 0) + 1
        else:
            wrong[s] = wrong.get (s, 0) + 1

    freq = {i: v for i, v in freq.items() if v > 10}
    rate = {i: (correct.get(i, 0) / (correct.get(i, 0) + wrong.get(i, 0)+1), v) for i, v in freq.items()}

    print(pd.DataFrame(rate).T.sort_values(by=1, ascending=False)[:20])
    print(pd.DataFrame(rate).T.sort_values(by=0, ascending=True)[:20])

    # get answers
    print("answer diversity")
    all_answers = {}
    for item in data:
        all_answers[item['raw answer']] = all_answers.get(item['raw answer'], 0) + 1
    all_answers = pd.Series(all_answers)
    all_answers = all_answers.sort_values(ascending=False)
    print(all_answers.iloc[:2000].sum() / all_answers.sum())
    print(all_answers.head(10))

    # verbs v.s entities
    for item in data:
        verb_pos = {'VB', 'VBD','VBG', 'VBN', 'VBP', 'VBZ'}
        nuan_pos = {'NNS', 'NN', 'NNP', 'NNPS'}
        tokens = nltk.pos_tag(word_tokenize(item['question_raw']))
        verbs = [item[0] for item in tokens if item[1] in verb_pos]
        nuans = [item[0] for item in tokens if item[1] in nuan_pos]
        n_verb = len(verbs)
        n_nuans = len(nuans)
        n = len(tokens)
        item['n_verb'] = n_verb
        item['n_nuans'] = n_nuans
        item['n'] = n
        # if n_nuans == 5:
        #     print(item['question_raw'], item['img_name'], item['answers_occurence'])
        if n_verb == 5:
            print(item['question_raw'], item['img_name'], item['answers_occurence'],item['answer'])

    ints = [(item['n_nuans'], item['n_verb'], item['n'], int (data[i]['raw answer'] == data[i]['answer'].lower ())) for i, item in
            enumerate (data)]

    print ("question difficulty")
    print (pd.DataFrame (ints).groupby (by=0).agg (['mean', 'count'])[3])
    print (pd.DataFrame (ints).groupby (by=1).agg (['mean', 'count'])[3])
    print (pd.DataFrame (ints).groupby (by=2).agg (['mean', 'count'])[3])

    # or question
    a = 0
    b = 0
    for item in data:
        sent = item['question_raw']
        tokens = word_tokenize(item['question_raw'])
        if "or" in tokens:
            if int (data[i]['raw answer'] == data[i]['answer'].lower ()) == 1:
                a += 1
            else:
                b += 1
            # print(item)
    print("Or question {}/{}".format(a, b))

    # gender
    a1 = 0
    b1 = 0
    a2 = 0
    b2 = 0
    for item in data:
        sent = item['question_raw']
        tokens = word_tokenize(item['question_raw'])
        if "he" in tokens:
            if int (data[i]['raw answer'] == data[i]['answer'].lower ()) == 1:
                a1 += 1
            else:
                b1 += 1

        if "she" in tokens:
            if int (data[i]['raw answer'] == data[i]['answer'].lower ()) == 1:
                a2 += 1
            else:
                b2 += 1
    print ("He question {}/{}".format (a1, b1))
    print ("She question {}/{}".format (a2, b2))


