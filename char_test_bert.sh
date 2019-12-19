#!/bin/bash

export BERT_BASE_DIR=./bert_base
export BERT_MODEL_OUTPUT=/media/darg1/Data/ozan_bert_models
export BERT_SLICED_DATA=/media/darg1/Data/ozan_bert_data/sliced_bert_data
export BERT_PROCESSED_DATA=/media/darg1/Data/ozan_bert_data/processed_bert_data
EACH_TRAIN_STEP=30000

#rm -rf $BERT_MODEL_OUTPUT/*
#rm -rf $BERT_DATA_OUTPUT/*


for INDEX in {0..5}
do
	python3 create_pretraining_data.py   --input_file=$BERT_SLICED_DATA/wiki_data_$INDEX.txt --output_file=$BERT_PROCESSED_DATA/tf_examples_$INDEX.tfrecord  --vocab_file=$BERT_BASE_DIR/char_vocab.txt --do_lower_case=True   --max_seq_length=96   --max_predictions_per_seq=15 --masked_lm_prob=0.15   --random_seed=$INDEX   --dupe_factor=5
done


for INDEX in {0..5}
do
	python3 run_pretraining.py   --input_file=$BERT_PROCESSED_DATA/tf_examples_$INDEX.tfrecord   --output_dir=$BERT_MODEL_OUTPUT   --do_train=True   --do_eval=True   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --train_batch_size=8   --max_seq_length=96   --max_predictions_per_seq=15   --num_train_steps=$((EACH_TRAIN_STEP*INDEX))   --num_warmup_steps=3000   --learning_rate=2e-5
done
