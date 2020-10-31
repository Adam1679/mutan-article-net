import numpy as np
import json
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class MetaAnswer :
    answer_id: int  # no unique cross questions
    answer: str
    raw_answer: str
    answer_confidence: str  # yes/no


@dataclass
class MetaAnnotation :
    image_id: str
    answers: list
    confidence: int
    answer_type: str
    question_id: int
    question_type: str

    def __post_init(self) :
        answers = []
        for answer in self.answers :
            answers.append (MetaAnswer (**answer))
        self.answers = answers


@dataclass
class MetaQuestion :
    image_id: int
    question: str
    question_id: int


class QuestionAnswerPair (object) :
    def __init__(self, question_path, answer_path) :
        self.question_meta = self.parse_question (question_path)
        self.answer_meta = self.parse_answer (answer_path)
        self.question2id = {}
        self.id2question = {}
        self.questionid2type = {}
        self.id2questionmeta = {}
        self.id2answermeta = {}
        self.question_type = {}  # Dict[type_id, type_name]
        self.answer2id = {}
        self.id2answer = {}
        self.questionid2answers = defaultdict (list)
        self.prepare ()

    def parse_answer(self, path) :
        with open (path, 'r', encoding="utf-8") as f :
            content = f.read ()
        meta_json = json.loads (content)
        return meta_json

    def parse_question(self, path) :
        with open (path, 'r', encoding="utf-8") as f :
            content = f.read ()
        meta_json = json.loads (content)
        return meta_json['questions']

    def prepare(self) :
        """
        one image -> one question (with unique id)
        (one image, one question) -> 10 anaswers?
        """
        # imageid2questionid = {}
        for part in self.question_meta :
            meta_obj = MetaQuestion (**part)
            self.question2id[meta_obj.question] = meta_obj.question_id
            self.id2question[meta_obj.question_id] = meta_obj.question
            # imageid2questionid[meta_obj.image_id] = meta_obj.question_id
            self.id2questionmeta[meta_obj.question_id] = meta_obj

        self.question_type = self.answer_meta['question_types']
        for part in self.answer_meta['annotations'] :
            meta_obj = MetaAnnotation (**part)
            self.questionid2type[meta_obj.question_id] = self.question_type[meta_obj.question_type]
            for answer in meta_obj.answers :
                obj = MetaAnswer (**answer)

                self.questionid2answers[meta_obj.question_id].append (obj.answer)
                if obj.answer not in self.answer2id :
                    tmp = len (self.answer2id)
                    self.answer2id[answer['answer']] = tmp
                    self.id2answer[tmp] = answer['answer']
                    self.id2answermeta[tmp] = obj

    def iter_question_answer_type_triplets(self) :
        for question_id, answers in self.questionid2answers.items () :
            yield self.id2question[question_id], answers, self.questionid2type[question_id]

    def iter_question_type_pairs(self) :
        for question, question_id in self.question2id.items () :
            yield question, self.questionid2type[question_id]

if __name__ == "__main__":
    """ Summarize the Data Set """
    question_path = "../data/OpenEnded_mscoco_train2014_questions.json"
    answer_path = "../data/mscoco_train2014_annotations.json"
    obj = QuestionAnswerPair(question_path, answer_path)
    all_questions = set()
    for question, type_name in obj.iter_question_type_pairs():
        print(f"{question}: {type_name}")
        all_questions.add(question)
    # Glove hit/miss in questions: 73192/150
    # Glove hit/miss in answers: 35234/1021
    print(f"#question in train/all-set {len(all_questions)}/14055\n #unique question: 12591 \n #unique_words: 7178")