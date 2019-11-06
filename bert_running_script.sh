#!/bin/bash

rm -rf /tmp/pretraining_output
rm /tmp/tf_examples.tfrecord
export BERT_BASE_DIR=/media/darg2/hdd/yl_tez/bert_project

python3 create_pretraining_data.py   --input_file=/home/darg2/Desktop/test.txt   --output_file=/tmp/tf_examples.tfrecord   --vocab_file=$BERT_BASE_DIR/vocabulary.txt   --do_lower_case=True   --max_seq_length=96   --max_predictions_per_seq=30   --masked_lm_prob=0.31   --random_seed=12345   --dupe_factor=5

python3 run_pretraining.py   --input_file=/tmp/tf_examples.tfrecord   --output_dir=/tmp/pretraining_output   --do_train=True   --do_eval=True   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --train_batch_size=8   --max_seq_length=96   --max_predictions_per_seq=30   --num_train_steps=80   --num_warmup_steps=20   --learning_rate=2e-5

