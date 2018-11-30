export BERT_BASE_DIR='....../multi_cased_L-12_H-768_A-12'
export DATA_DIR=''
export TRAINED_CLASSIFIER=''
export OUTPUT_DIR=''

# Fine-tune model, task: sentence classification
python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$TRAINED_CLASSIFIER

# Model prediction: with trained classifier
python run_classifier.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=$DATA_DIR/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=$OUTPUT_DIR

# MAP Score Calculation
# specify task A/B/C after the filename
python create_output.py A
