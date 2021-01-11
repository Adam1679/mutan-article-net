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


if __name__ == "__main__":
    import os
    import json
    import sys
    train_data_path = "../data/okvqa/interim/train_questions_annotations.json"
    val_data_path = "../data/okvqa/interim/val_questions_annotations.json"

    val_given_data = "./query_val.json"
    train_given_data = "./query_train.json"
    tmp_train_query_path = "./drqa/DrQA/train_tmp.txt"
    tmp_val_query_path = "./drqa/DrQA/val_tmp.txt"

    if sys.argv[1] == '1':
        def process(data_path, given_data, tmp_query_path):
            with open (data_path, 'r') as f :
                data = json.loads (f.read ())
            with open (given_data, "r") as f :
                given_data = json.loads (f.read ())

            question2answer = {}
            question2all = {}
            for pair in data :
                question2answer[pair['question_id']] = [item[0] for item in pair["answers_occurence"]]
                question2all[pair['question_id']] = pair

            if sys.argv[1] == '1':
                queries = []
                for query in given_data:
                    question_id = query['question_id']
                    # get answer
                    query_text = query['query']
                    queries.append({'question': query_text, "answer": question2answer[question_id], "question_id": question_id})

                cnt = 0
                # import pdb; pdb.set_trace()
                with open(tmp_query_path, "w") as f:
                    for q in queries:
                        if len(q['question'].strip()) == 0:
                            cnt += 1
                        else:
                            f.write (json.dumps (q) + "\n")
                print("None queries: ", cnt)
        process(train_data_path, train_given_data, tmp_train_query_path)
        process(val_data_path, val_given_data, tmp_val_query_path)
    else:
        import re, tqdm
        pat = re.compile("(\d+).(\d+).json")
        train_root = "./drqa/DrQA/train_okvqa_queries/"
        val_root = "./drqa/DrQA/val_okvqa_queries/"
        train_given_data = "okvqa_an_input_train.json"
        val_given_data = "okvqa_an_input_val.json"

        def process(data_path, root, given_data):
            with open (data_path, 'r') as f :
                data = json.loads (f.read ())

            n_docs = 0
            n_pos_docs = 0
            n_pos_sentences = 0
            n_sentences = 0
            question2answer = {}
            question2all = {}
            for pair in data :
                question2answer[pair['question_id']] = [item[0] for item in pair["answers_occurence"]]
                question2all[pair['question_id']] = pair

            with open(given_data, "w") as ff:
                for file in tqdm.tqdm(os.listdir(root)):
                    if file.endswith(".json"):
                        # import pdb; pdb.set_trace()
                        # print(file)
                        question_id, document_order = re.findall(pat, file)[0]
                        question_id = int(question_id)
                        with open(os.path.join(root, file)) as f:
                            content = json.loads(f.read())

                        title = content['title']
                        doc = content['doc']
                        question = content['question']
                        answers = question2answer[question_id]
                        titleHasAns, sentences_pairs, docHasAns = process_one_document(title, doc, answers)
                        on_sample = {}
                        on_sample['question'] = question
                        on_sample['title'] = title
                        on_sample['titleHasAns'] = titleHasAns
                        on_sample['sentences_pairs'] = sentences_pairs
                        on_sample['docHasAns'] = docHasAns
                        ff.write("{}\n".format(json.dumps(on_sample)))
                        n_docs += 1
                        n_sentences += len(sentences_pairs)
                        n_pos_sentences += sum([i[1] for i in sentences_pairs])
                        n_pos_docs += docHasAns

            print("stats")
            print(f"{n_pos_docs}/{n_docs}")
            print(f"{n_pos_sentences}/{n_sentences}")


        process(train_data_path, train_root, train_given_data)
        process(val_data_path, val_root, val_given_data)

    # retrieve doc

