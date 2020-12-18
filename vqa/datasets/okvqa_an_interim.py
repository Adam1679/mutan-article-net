import json
import os
import argparse
from collections import Counter

def get_subtype(split='train'):
    if split in ['train', 'val']:
        return split + '2014'
    else:
        return 'test2015'

def get_image_name_old(subtype='train2014', image_id='1', format='%s/COCO_%s_%012d.jpg'):
    return format%(subtype, subtype, image_id)

def get_image_name(subtype='train2014', image_id='1', format='COCO_%s_%012d.jpg'):
    return format%(subtype, image_id)

def interim(questions, split='train', annotations=[], sentences={}):
    print('Interim', split)
    data = []
    for i in range(len(questions)):
        row = {}
        row['question_id'] = questions[i]['question_id']
        assert questions[i]['question_id'] == annotations[i]['question_id']
        row['image_name'] = get_image_name(get_subtype(split), questions[i]['image_id'])
        row['sentences'] = sentences[row['question_id']]['sentences']
        row['has_answer'] = sentences[row['question_id']]['has_answer']
        if split in ['train', 'val']:
            answers = []
            for ans in annotations[i]['answers']:
                answers.append(ans['answer'])
            row['answers_occurence'] = Counter(answers).most_common()  # TODO: what is this
            row['answer'] = row['answers_occurence'][0][0]  # TODO: multiple answers? do they align with each other?
        else:
            raise NotImplementedError
        data.append(row)
    return data

def get_doc(doc_path):
    import json
    with open(doc_path, 'r') as f:
        list_of_docs = json.load(f)
    id_to_sentences = {}
    for item in list_of_docs:
        question_id = item['question_id']
        sentences, had_answers = zip(*item['sentences_pairs'])
        if question_id not in id_to_sentences:
            id_to_sentences[question_id] = {'sentences': [], 'has_answer': []}
        id_to_sentences[question_id]['sentences'].extend(sentences)
        id_to_sentences[question_id]['has_answer'].extend(had_answers)
    return id_to_sentences

def vqa_interim(dir_vqa):
    '''
    Put the VQA data into single json file in data/interim
    or train, val, trainval : [[question_id, image_name, question, MC_answer, answer] ... ]
    or test, test-dev :       [[question_id, image_name, question, MC_answer] ... ]
    '''

    path_train_qa    = os.path.join(dir_vqa, 'interim', 'train_questions_annotations.json')
    path_val_qa      = os.path.join(dir_vqa, 'interim', 'val_questions_annotations.json')
    path_test_q      = os.path.join(dir_vqa, 'interim', 'test_questions.json')
    path_testdev_q   = os.path.join(dir_vqa, 'interim', 'testdev_questions.json')

    os.system('mkdir -p ' + os.path.join(dir_vqa, 'interim'))

    print('Loading annotations and questions...')
    annotations_train = json.load(open(os.path.join(dir_vqa, 'raw', 'annotations', 'mscoco_train2014_annotations.json'), 'r'))
    annotations_val   = json.load(open(os.path.join(dir_vqa, 'raw', 'annotations', 'mscoco_val2014_annotations.json'), 'r'))
    questions_train   = json.load(open(os.path.join(dir_vqa, 'raw', 'annotations', 'OpenEnded_mscoco_train2014_questions.json'), 'r'))
    questions_val   = json.load(open(os.path.join(dir_vqa, 'raw', 'annotations', 'OpenEnded_mscoco_val2014_questions.json'), 'r'))

    doc_val     = json.load(open(os.path.join(dir_vqa, 'raw', 'documents', 'okvqa_val.json'), 'r'))
    doc_train     = json.load(open(os.path.join(dir_vqa, 'raw', 'documents', 'okvqa_train.json'), 'r'))
    val_sentences = get_doc(doc_val)
    train_sentences = get_doc(doc_train)
    data_train = interim(questions_train['questions'], 'train', annotations_train['annotations'], train_sentences)
    print('Train size %d'%len(data_train))
    print('Write', path_train_qa)
    json.dump(data_train, open(path_train_qa, 'w'))
    # for item in questions_val['questions']:
    #     item['question'] = item['question'].replace("What kind", "What type")

    data_val = interim(questions_val['questions'], 'val', annotations_val['annotations'])
    data_train = interim(questions_train['questions'], 'train', annotations_train['annotations'], val_sentences)
    print('Val size %d'%len(data_val))
    print('Write', path_val_qa)
    json.dump(data_val, open(path_val_qa, 'w'))

    print('Concat. train and val')
    data_trainval = data_train + data_val
    print('Trainval size %d'%len(data_trainval))

    print('For compatibility concerns, we copy the dev data for test data here')
    print('Testdev size %d'%len(data_val))
    print('Write', path_testdev_q)
    json.dump(data_val, open(path_testdev_q, 'w'))

    print('Test size %d'%len(data_val))
    print('Write', path_test_q)
    json.dump(data_val, open(path_test_q, 'w'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_vqa', default='data/okvqa', type=str, help='Path to vqa data directory')
    args = parser.parse_args()
    vqa_interim(args.dir_vqa)
