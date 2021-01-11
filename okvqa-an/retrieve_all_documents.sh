#!/bin/bash
python process_documents.py 1
cd ./drqa/DrQA
python ./scripts/retriever/eval.py ./train_tmp.txt --num-workers 2
python ./scripts/retriever/eval.py ./val_tmp.txt --num-workers 2
mv
cd ../../
python process_documents.py 2