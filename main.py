
import tensorflow as tf
import os
from parameters import FLAGS
from preprocess import *
from model import network
from train import *
from eval import *
import time
import sys

##combines the train and eval into a single script
if __name__ == "__main__":

	print("---PREPROCESSING STARTED---")

	print("\treading word embeddings...")
	old = time.time()
	embeddings = readFastTextEmbeddings(FLAGS.word_embed_path)
	print("reading embeddings took " + str(time.time() - old) + " sec")

	print("\treading tweets...")
	old = time.time()
	ground_truth = readData(FLAGS.data_path)
	print("reading data took "+ str(time.time()-old) + " sec")

	sys.exit()
	print("\tconstructing datasets and network...")
	training_tweets, training_users, training_seq_lengths, valid_tweets, valid_users, valid_seq_lengths, test_tweets, test_users, test_seq_lengths = partite_dataset(
		tweets, users, seq_lengths)

	# hyperparameter optimization if it is set
	if FLAGS.optimize == False:
		# print specs
		print("---TRAINING STARTED---")
		model_specs = "with parameters: Learning Rate:" + str(
			FLAGS.learning_rate) + ", Regularization parameter:" + str(FLAGS.l2_reg_lambda) + ", cell size:"
		model_specs += str(FLAGS.rnn_cell_size) + ", embedding size:" + str(
			FLAGS.word_embedding_size) + ", language:" + FLAGS.lang
		print(model_specs)

		# run the network
		tf.reset_default_graph()
		net = network(embeddings)
		train(net, training_tweets, training_users, training_seq_lengths, valid_tweets, valid_users, valid_seq_lengths,
			  target_values, vocabulary, embeddings)

	else:
		for rnn_cell_size in FLAGS.rnn_cell_sizes:
			for learning_rate in FLAGS.l_rate:
				for regularization_param in FLAGS.reg_param:
					# prep the network
					tf.reset_default_graph()
					FLAGS.learning_rate = learning_rate
					FLAGS.l2_reg_lambda = regularization_param
					FLAGS.rnn_cell_size = rnn_cell_size
					net = network(embeddings)

					# print specs
					print("---TRAINING STARTED---")
					model_specs = "with parameters: Learning Rate:" + str(
						FLAGS.learning_rate) + ", Regularization parameter:" + str(FLAGS.l2_reg_lambda) + ", cell size:"
					model_specs += str(FLAGS.rnn_cell_size) + ", embedding size:" + str(
						FLAGS.word_embedding_size) + ", language:" + FLAGS.lang
					print(model_specs)

					# take the logs
					f = open(FLAGS.log_path, "a")
					f.write("---TRAINING STARTED---\n")
					model_specs += "\n"
					f.write(model_specs)
					f.close()

					# start training
					train(net, training_tweets, training_users, training_seq_lengths, valid_tweets, valid_users,
						  valid_seq_lengths, target_values, vocabulary, embeddings)

	print("---TESTING STARTED---")
	print("\treading tweets for test...")
	tweets, users, target_values, seq_lengths = readData(FLAGS.test_data_path)
	print("\ttest set size: " + str(len(tweets)))

	# finds every model in FLAGS.model_path and runs every single one
	if FLAGS.optimize == True:
		models = os.listdir(FLAGS.model_path)
		for model in models:
			if model.endswith(".ckpt.index"):
				FLAGS.model_name = model[:-6]
				tf.reset_default_graph()

				if "150" in FLAGS.model_name:
					FLAGS.rnn_cell_size = 150
				elif "100" in FLAGS.model_name:
					FLAGS.rnn_cell_size = 100
				elif "50" in FLAGS.model_name:
					FLAGS.rnn_cell_size = 50

				net = network(embeddings)
				test(net, tweets, users, seq_lengths, target_values, vocabulary, embeddings)
	# just runs  single model specified in FLAGS.model_path and FLAGS.model_name
	else:
		tf.reset_default_graph()
		net = network(embeddings)
		test(net, tweets, users, seq_lengths, target_values, vocabulary, embeddings)

