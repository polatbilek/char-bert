#!/bin/bash

rm -rf /tmp/pretraining_output
rm /tmp/tf_examples.tfrecord
export BERT_BASE_DIR=/home/darg1/Desktop/ozan/bert_config

#python3 create_pretraining_data.py   --input_file=$BERT_BASE_DIR/doc1.txt --output_file=/tmp/tf_examples_1.tfrecord  --vocab_file=$BERT_BASE_DIR/vocabulary.txt --do_lower_case=True   --max_seq_length=128   --max_predictions_per_seq=20 --masked_lm_prob=0.15   --random_seed=12345   --dupe_factor=5

#python3 create_pretraining_data.py   --input_file=$BERT_BASE_DIR/doc2.txt --output_file=/tmp/tf_examples_2.tfrecord --vocab_file=$BERT_BASE_DIR/vocabulary.txt --do_lower_case=True --max_seq_length=128   --max_predictions_per_seq=20 --masked_lm_prob=0.15 --random_seed=12345   --dupe_factor=5

#python3 create_pretraining_data.py   --input_file=$BERT_BASE_DIR/concatdocs.txt   --output_file=/tmp/tf_examples_3.tfrecord  --vocab_file=$BERT_BASE_DIR/vocabulary.txt   --do_lower_case=True   --max_seq_length=128   --max_predictions_per_seq=20   --masked_lm_prob=0.15   --random_seed=12345   --dupe_factor=5

echo "???????????????? first run ???????????????????"
#python3 run_pretraining.py  --input_file=/tmp/tf_examples_1.tfrecord   --output_dir=/media/darg1/Data/ozan_bert_models   --do_train=True   --do_eval=True   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --train_batch_size=8   --max_seq_length=128   --max_predictions_per_seq=20   --num_train_steps=2000   --num_warmup_steps=200   --learning_rate=2e-5

echo "???????????????? second run ???????????????????"
#python3 run_pretraining.py  --init_checkpoint=/media/darg1/Data/ozan_bert_models/model.ckpt-2000  --input_file=/tmp/tf_examples_2.tfrecord   --output_dir=/media/darg1/Data/ozan_bert_models   --do_train=True   --do_eval=True   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --train_batch_size=8   --max_seq_length=128   --max_predictions_per_seq=20   --num_train_steps=4000   --num_warmup_steps=400   --learning_rate=2e-5

#rm -rf /media/darg1/Data/ozan_bert_models/*

echo "???????????????? third run ???????????????????"
python3 run_pretraining.py  --input_file=/tmp/tf_examples_3.tfrecord   --output_dir=/media/darg1/Data/ozan_bert_models   --do_train=True   --do_eval=True   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --train_batch_size=8   --max_seq_length=128   --max_predictions_per_seq=20   --num_train_steps=4000   --num_warmup_steps=400   --learning_rate=2e-5



