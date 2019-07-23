from model import network
from train import *
from test import *
import sys

##combines the train and eval into a single script
if __name__ == "__main__":

	print("---PREPROCESSING STARTED---")

	print("\treading word embeddings...")
	embeddings, vocabulary = readFastTextEmbeddings(FLAGS.word_embed_path)

	print("\treading texts...")
	ground_truth, data = readData(FLAGS.data_path)

	print("\tconstructing datasets and network...")
	training_data, training_gt, valid_data, valid_users, test_data, test_users = partite_dataset(data, ground_truth)


	# hyperparameter optimization if it is set
	if FLAGS.optimize == False:
		# print specs
		print("---TRAINING STARTED---")
		model_specs = "With parameters: Learning Rate:" + str(FLAGS.learning_rate) + \
					  ", Regularization parameter:" + str(FLAGS.l2_reg_lambda) +\
					  ", Global cell size:" + str(FLAGS.global_rnn_cell_size) + \
					  ", Semantic cell size: " + str(FLAGS.semantic_rnn_cell_size) + \
					  ", Fully connected size: " + str(FLAGS.fc_size)
		print(model_specs)

		# run the network
		tf.reset_default_graph()
		net = network(embeddings)
		train(net, training_data, training_gt, valid_data, valid_users, vocabulary, embeddings, ground_truth)

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
					model_specs = "With parameters: Learning Rate:" + str(FLAGS.learning_rate) + \
								  ", Regularization parameter:" + str(FLAGS.l2_reg_lambda) + \
								  ", Global cell size:" + str(FLAGS.global_rnn_cell_size) + \
								  ", Semantic cell size: " + str(FLAGS.semantic_rnn_cell_size) + \
								  ", Fully connected size: " + str(FLAGS.fc_size)
					print(model_specs)

					# take the logs
					f = open(FLAGS.log_path, "a")
					f.write("---TRAINING STARTED---\n")
					model_specs += "\n"
					f.write(model_specs)
					f.close()

					# start training
					train(net, training_data, training_gt, valid_data, valid_users, vocabulary, embeddings, ground_truth)

	print("---TESTING STARTED---")
	print("\ttest set size: " + str(len(test_data)))

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
				test(network, test_data, test_users, vocabulary, embeddings, ground_truth)
	# just runs  single model specified in FLAGS.model_path and FLAGS.model_name
	else:
		tf.reset_default_graph()
		net = network(embeddings)
		test(network, test_data, test_users, vocabulary, embeddings, ground_truth)

