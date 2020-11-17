# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:57:57 2020
@author: Xuandi Fu(Carnegie Mellon University)
"""

import drqa
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#python scripts/pipeline/predict.py data/datasets/testsample.txt
#python scripts/retriever/predict.py data/datasets/testsample.txt
#export CLASSPATH=$CLASSPATH:/home/xuandif/DrQA/data/corenlp/*
image2text = {}
image2id = {} #image2question map
question2ans = {} #questionid 2 answer

q2ent = {}
q2query = {}

data_path = 'data/OpenEnded_mscoco_train2014_questions.json'
image_path = 'data/img_id_entities_train.json'
query_path = 'data/query_train.json'

stop_words = set(stopwords.words('english'))
print(stop_words)
def loadMap():
    data = json.load(open(data_path))
    pairs = data['questions']
    for pair in pairs:
        print(str(pair['image_id']))
        image2text[str(pair['image_id'])] = pair['question'].lower().replace('?', '')
        image2id[str(pair['image_id'])] = pair['question_id']

def nonstop(question):
    tokens = word_tokenize(question)
    entity = []
    for word in tokens:
        if word not in stop_words:
            entity.append(word)
    return entity

def query(entity):
    length = len(entity)

    for i in range(length - 1):
        for j in range(i+1, length):
            entity.append(' '.join([entity[i], entity[j]]))
    return entity


# ent = filter('we are the family')
# print(ent)
#[[question_id, query1, query2], [question_id, query1, query2], ...]
print(query(['m1', 'm2', 'm3']))
loadMap()

def generateQuery():
    output = []
    image_data = json.load(open(image_path, 'r'))
    i= 0
    j = 0
    length = 0
    for image_id, image_entities in image_data.items() :
        j+=1
        # print(image_id)
        # print(image2text.keys())
        # print(image_entities)
        # print(image2text.keys())
        if image_id in image2text.keys():
            i+=1
            entity = nonstop(image2text[image_id])

            question_id = image2id[image_id]
            #entity.extend(image_entities[:])
            q2ent[question_id] = entity
            #concate question and image entities
            #q2query[question_id] = query(entity)
            queries = query(entity)
            length += len(queries)
            # print(queries)
            # print(entity)
            output.extend([{"question_id": question_id, "query": " ".join(image_entities)}])
            output.extend([{"question_id": question_id, "query": " ".join(entity)}])
            output.extend([{"question_id": question_id, "query": " ".join(entity + image_entities)}])

    print(j)
    print(i)
    print(length/i)
    return output

if __name__=='__main__':

    output = generateQuery()
    f_query = open(query_path, 'w+')
    json.dump(output, f_query)