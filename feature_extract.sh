# download the pre-train model for vqa1.0. This is trained on train and eval dataset.
# mkdir -p logs/vqa
# cd logs/vqa
# wget
# http://webia.lip6.fr/~cadene/Downloads/vqa.pytorch/logs/vqa/mutan_att_trainval.zip


# extract features for COCO dataset using the default fbresnet152
# extract train data, final feature size = 2048 * 14 * 14
# this would create a hdf5 file with one data field hdf5_file['att']
python extract.py --dir_data data/coco --dataset coco --data_split train --batch_size 8 --mode att
python extract.py --dir_data data/coco --dataset coco --data_split val --batch_size 8 --mode att 


# evaluate on the test dataset.
# it would automatically download the skip-thoughts model, which is used to
# extract the feature of the language
python train.py -e --path_opt options/vqa/mutan_att_trainval.yaml --resume ckpt




# the following comments are for recording the processed dataformat.
# valset.pickle prototype:
# valset[0] = {'question_id': 3506232, 
# 'image_name': 
# 'COCO_val2014_000000350623.jpg', 
# 'question': 'What is the table made of?', 
# 'MC_answer': ['4', 'green', 'no', 'metal', '2', 'blue', 'plastic', 'marble', 'wood', 'white', 'red', 'concrete bricks', 'robe', '3', '1', 'yes', 'siam', 'white and black'], 
# 'answer': 'wood', 
# 'answers_occurence': [['wood', 10]], 
# 'question_words': ['what', 'is', 'the', 'table', 'made', 'of'], 
# 'question_words_UNK': ['what', 'is', 'the', 'table', 'made', 'of'], 
# 'question_length': 6, 
# 'question_wids': [1, 3, 4, 230, 340, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
# 'seq_length': 6, 
# 'answer_aid': 21, 
# 'answers': ['wood'], 
# 'answers_aid': [21], 
# 'answers_count': [10]}

# wid_to_word v.s word_to_wid lengths = 12434
# ans_to_aid  v.s aid_to_aid length = 2000


# interm folder contain the compatible processed data for the "vqa_processed"
# -rw-rw-r-- 1 ubuntu ubuntu  70M Oct 27 15:33 test_questions.json
# -rw-rw-r-- 1 ubuntu ubuntu  18M Oct 27 15:33 testdev_questions.json
# -rw-rw-r-- 1 ubuntu ubuntu  92M Oct 27 15:33 train_questions_annotations.json
# -rw-rw-r-- 1 ubuntu ubuntu 136M Oct 27 15:33 trainval_questions_annotations.json
# -rw-rw-r-- 1 ubuntu ubuntu  45M Oct 27 15:33 val_questions_annotations.json
# val_questions_annotations[0] = \
# {'question_id': 3506232, 
# 'image_name': 'COCO_val2014_000000350623.jpg', 
# 'question': 'What is the table made of?', 
# 'MC_answer': ['4', 'green', 'no', 'metal', '2', 'blue', 'plastic', 'marble', 'wood', 'white', 'red', 'concrete bricks', 'robe', '3', '1', 'yes', 'siam', 'white and black'], 
# 'answer': 'wood', 
# 'answers_occurence': [['wood', 10]]}

# testdev_questions.json[0] = \
# {'question_id': 4195880, 
# 'image_name': 'COCO_test2015_000000419588.jpg', 
# 'question': 'Are the dogs tied?', 
# 'MC_answer': ['1', 'bare', 'bacon hot dog beans', 'stumbling', '4', 'no', '3', 'black', 'yes', '2', 'ringling bros and barnum & bailey', 'quilted northern', 'junk', 'white', 'blue', 'hopefully', 'red', 'grass']}

