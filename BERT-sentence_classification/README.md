Utilizing the Google pre-trained language model BERT
https://github.com/google-research/bert

# Pre-trained Model
Here we use the BERT-Base, Multilingual Cased model: 
https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip


(download and set the corresponding directory path)

# Data Preprocessing
Format: <label> <sentence_1_ID> <sentence_2_ID> <sentence_1> <sentence_2>


(sentence_1 & sentence_2: question/comment corresponding to task A~C)

# Run Classifier
Run with run_bert.sh
Two stages:
1. Fine-tuning model, task: sentence classification
2. Model prediction: with trained classifier (from the previous stage)

* Note: run_classifier.py  comes directly from  https://github.com/google-research/bert/blob/master/run_classifier.py
