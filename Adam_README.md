### Download the pre-train model for vqa1.0. This is trained on train and eval dataset.
```
mkdir -p logs/vqa
cd logs/vqa
wget
http://webia.lip6.fr/~cadene/Downloads/vqa.pytorch/logs/vqa/mutan_att_trainval.zip
```

### extract features for COCO dataset using the default fbresnet152

```
# extract train data, final feature size = 2048 * 14 * 14
# this would create a hdf5 file with one data field hdf5_file['att']
python extract.py --dir_data data/coco --dataset coco --data_split train --batch_size 8 --mode att
python extract.py --dir_data data/coco --dataset coco --data_split val --batch_size 8 --mode att 

```

### evaluate on the test dataset.
```
# it would automatically download the skip-thoughts model, which is used to
# extract the feature of the language
python train.py -e --path_opt options/okvqa/mutan_att_train.yaml --resume ckpt
```

### evaluate OKVQA data on the train/val dataset.
```
# it would automatically download the skip-thoughts model, which is used to
# extract the feature of the language
python vqa/datasets/okvqa_interim.py --dir_vqa=data/okvqa
python vqa/datasets/vqa_processed.py --dir=data/okvqa
```


### faeture tuning method:





 ### the following comments are for recording the processed data format.
 
 - valset.pickle prototype:
 valset[0] = {'question_id': 3506232, 
 'image_name': 
 'COCO_val2014_000000350623.jpg', 
 'question': 'What is the table made of?', 
 'MC_answer': ['4', 'green', 'no', 'metal', '2', 'blue', 'plastic', 'marble', 'wood', 'white', 'red', 'concrete bricks', 'robe', '3', '1', 'yes', 'siam', 'white and black'], 
 'answer': 'wood', 
 'answers_occurence': [['wood', 10]], 
 'question_words': ['what', 'is', 'the', 'table', 'made', 'of'], 
 'question_words_UNK': ['what', 'is', 'the', 'table', 'made', 'of'], 
 'question_length': 6, 
 'question_wids': [1, 3, 4, 230, 340, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'seq_length': 6, 
 'answer_aid': 21, 
 'answers': ['wood'], 
 'answers_aid': [21], 
 'answers_count': [10]}

 - wid_to_word v.s word_to_wid lengths = 12434
 - ans_to_aid  v.s aid_to_aid length = 2000

raw data is:

{'question_type': 'what', 
'multiple_choice_answer': 'curved', 
'answers': [{'answer': 'oval', 'answer_confidence': 'yes', 'answer_id': 1}, {'answer': 'semi circle', 'answer_confidence': 'yes', 'answer_id': 2}, {'answer': 'curved', 'answer_confidence': 'yes', 'answer_id': 3}, {'answer': 'curved', 'answer_confidence': 'yes', 'answer_id': 4}, {'answer': 'double curve', 'answer_confidence': 'yes', 'answer_id': 5}, {'answer': 'banana', 'answer_confidence': 'maybe', 'answer_id': 6}, {'answer': 'curved', 'answer_confidence': 'yes', 'answer_id': 7}, {'answer': 'wavy', 'answer_confidence': 'yes', 'answer_id': 8}, {'answer': 'twisting', 'answer_confidence': 'no', 'answer_id': 9}, {'answer': 'curved', 'answer_confidence': 'maybe', 'answer_id': 10}], 
'image_id': 487025, 
'answer_type': 'other', 
'question_id': 4870250}

 interm folder contain the compatible processed data for the "vqa_processed"
 
 test_questions.json
 
 testdev_questions.json
 
 train_questions_annotations.json
 
 trainval_questions_annotations.json
 
 val_questions_annotations.json
 
 val_questions_annotations[0] = \
 {'question_id': 3506232, 
 'image_name': 'COCO_val2014_000000350623.jpg', 
 'question': 'What is the table made of?', 
 'MC_answer': ['4', 'green', 'no', 'metal', '2', 'blue', 'plastic', 'marble', 'wood', 'white', 'red', 'concrete bricks', 'robe', '3', '1', 'yes', 'siam', 'white and black'], 
 'answer': 'wood', 
 'answers_occurence': [['wood', 10]]}

 testdev_questions.json[0] = \
 {'question_id': 4195880, 
 'image_name': 'COCO_test2015_000000419588.jpg', 
 'question': 'Are the dogs tied?', 
 'MC_answer': ['1', 'bare', 'bacon hot dog beans', 'stumbling', '4', 'no', '3', 'black', 'yes', '2', 'ringling bros and barnum & bailey', 'quilted northern', 'junk', 'white', 'blue', 'hopefully', 'red', 'grass']}



