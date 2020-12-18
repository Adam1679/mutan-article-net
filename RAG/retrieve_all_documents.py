# from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration, BartForConditionalGeneration, RagConfig, DPRQuestionEncoder
import torch

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from stemming.porter2 import stem as st
# ['question', ['title', 0], [(sent1, 1), (sent2, 0), (sent3, 1),...], 0]
# [{"question_id": integer, "query": ""}, {}]
def normalize(text):
    text = text.lower().strip()
    return text

def stem_text(text):
    if " " in text:
        text = " ".join([st(w) for w in text.split(" ")])
    else:
        text = st(text)
    return text

def process_one_document(title, doc, answers, stem=True):
    # return
    func = stem_text if stem else lambda x: x
    doc = normalize(doc)
    title = normalize(title)
    answers = {func(normalize(i)) for i in answers}
    sentences = [sent for sent in sent_tokenize(doc)]
    sentences_pairs = []
    dosHasAns = 0
    for sent in sentences:
        words = set([func(w) for w in word_tokenize(sent)])
        hasAns = 0
        for ans in answers:
            if ans in words:
                hasAns = 1
                break
        sentences_pairs.append((sent, hasAns))
        dosHasAns = max(dosHasAns, hasAns)
    titleHasAns = 0
    for word in word_tokenize(title):
        if func(word) in answers:
            titleHasAns = 1
            break

    return titleHasAns, sentences_pairs, dosHasAns

# tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
# retriever = RagRetriever.from_pretrained ("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
# model = RagTokenForGeneration.from_pretrained ("facebook/rag-token-nq", retriever=retriever)

# def extract(input_q):
#     input_dict = tokenizer.prepare_seq2seq_batch (input_q + "?", return_tensors="pt")
#     generated = model.generate(input_ids=input_dict["input_ids"])
#     ans = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
#     return ans
#
# if __name__ == "__main__":
#     import json
#     import sys
#     train_data_path = "../data/okvqa/interim/train_questions_annotations.json"
#     val_data_path = "../data/okvqa/interim/val_questions_annotations.json"
#
#     val_given_data = "../okvqa-an/query_val.json"
#     train_given_data = "../okvqa-an/query_train.json"
#     tmp_train_query_path = "./train_tmp.txt"
#     tmp_val_query_path = "./val_tmp.txt"
#
#     def process(data_path, given_data):
#         with open (data_path, 'r') as f :
#             data = json.loads (f.read ())
#         with open (given_data, "r") as f :
#             given_data = json.loads (f.read ())
#
#         question2answer = {}
#         question2all = {}
#         for pair in data :
#             question2answer[pair['question_id']] = [item[0] for item in pair["answers_occurence"]]
#             question2all[pair['question_id']] = pair
#
#         if sys.argv[1] == '1':
#             queries = []
#             for query in given_data:
#                 question_id = query['question_id']
#                 # get answer
#                 query_text = query['query']
#                 queries.append({'question': query_text, "answer": question2answer[question_id], "question_id": question_id})
#
#             cnt = 0
#             # import pdb; pdb.set_trace()
#             with open(tmp_query_path, "w") as f:
#                 for q in queries:
#                     if len(q['question'].strip()) == 0:
#                         cnt += 1
#                     else:
#                         f.write (json.dumps (q) + "\n")
#
#             print("None queries: ", cnt)
#
#     process(train_data_path, train_given_data, tmp_train_query_path)
#     process(val_data_path, val_given_data, tmp_val_query_path)

if __name__ == "__main__":
    val_set_path = "/Users/adam/Desktop/mutan-article-net/data/okvqa/interim/val_questions_annotations.json"
    val_set_extracted_path = "/Users/adam/Desktop/mutan-article-net/data/okvqa/extracted_text/query_val.json"
    import json

    data1 = json.load(open(val_set_path))
    data2 = json.load(open(val_set_extracted_path))
    all_ids1 = dict()
    all_ids2 = dict()
    for d in data1:
        all_ids1[d['question_id']] = all_ids1.get(d['question_id'], 0) + 1
    for d in data2:
        all_ids2[d['question_id']] = all_ids2.get(d['question_id'], 0) + 1
    print("val set", len(all_ids1))
    print("query set", len(all_ids2))
    print(all_ids2)
