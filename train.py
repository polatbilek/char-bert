
import tensorflow as tf
from preprocess import *
import sys


###########################################################################################################################
##trains and validates the model
###########################################################################################################################
def train(network, training_data, training_users, valid_data, valid_users, vocabulary, embeddings, ground_truth):
	saver = tf.train.Saver(max_to_keep=None)

	with tf.device('/device:GPU:0'):
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

			# init variables
			init = tf.global_variables_initializer()
			sess.run(init)
			sess.run(network.embedding_init, feed_dict={network.embedding_placeholder: embeddings})

			# load the model from checkpoint file if it is required
			if FLAGS.use_pretrained_model == True:
				load_as = os.path.join(FLAGS.model_path, FLAGS.model_name)
				saver.restore(sess, load_as)
				print("Loading the pretrained model from: " + str(load_as))

			# For each epoch
			for epoch in range(FLAGS.num_epochs):
				epoch_loss = 0.0
				epoch_loss_age = 0.0
				epoch_loss_job = 0.0
				epoch_loss_gender = 0.0
				epoch_accuracy_age = 0.0
				epoch_accuracy_job = 0.0
				epoch_accuracy_gender = 0.0

				training_batch_count = int(len(training_data) / FLAGS.batch_size)
				valid_batch_count = int(len(valid_data) / FLAGS.batch_size)

				batch_accuracy_age = 0.0
				batch_accuracy_job = 0.0
				batch_accuracy_gender = 0.0
				batch_loss = 0.0
				batch_loss_age = 0.0
				batch_loss_job = 0.0
				batch_loss_gender = 0.0

				# TRAINING
				for batch in range(training_batch_count):
					# Prepare the batch
					training_batch_x, \
					training_batch_y_age, \
					training_batch_y_job, \
					training_batch_y_gender, \
					training_batch_seqlen = prepWordBatchData(training_data,
																training_users,
																ground_truth,
																vocabulary,
																batch)

					# Prepare feed values
					feed_dict = {network.X: training_batch_x,
								 network.Y_age: training_batch_y_age,
								 network.Y_job: training_batch_y_job,
								 network.Y_gender: training_batch_y_gender,
								 network.sequence_length: training_batch_seqlen,
								 network.reg_param: FLAGS.l2_reg_lambda}


					# Run the computational graph
					_, loss_age, loss_job, loss_gender, loss, \
					prediction_age, prediction_job, prediction_gender, \
					accuracy_age, accuracy_job, accuracy_gender = sess.run(
						[network.train, network.loss_age, network.loss_job, network.loss_gender, network.loss,
						 network.prediction_age, network.prediction_job, network.prediction_gender,
						 network.accuracy_age, network.accuracy_job, network.accuracy_gender], feed_dict=feed_dict)

					# Calculate the metrics
					batch_loss += loss
					epoch_loss += loss

					batch_loss_age += loss_age
					epoch_loss_age += loss_age

					batch_loss_job += loss_job
					epoch_loss_job += loss_job

					batch_loss_gender += loss_gender
					epoch_loss_gender += loss_gender

					batch_accuracy_age += accuracy_age
					epoch_accuracy_age += accuracy_age

					batch_accuracy_job += accuracy_job
					epoch_accuracy_job += accuracy_job

					batch_accuracy_gender += accuracy_gender
					epoch_accuracy_gender += accuracy_gender

					# print the accuracy and progress of the training
					if batch % FLAGS.evaluate_every == 0 and batch != 0:
						batch_accuracy_age /= FLAGS.evaluate_every
						batch_accuracy_job /= FLAGS.evaluate_every
						batch_accuracy_gender /= FLAGS.evaluate_every

						print("Epoch " + "{:2d}".format(epoch) +
							  " , Batch " + "{0:5d}".format(batch) + "/" + str(training_batch_count) +
							  " , Loss= " + "{0:5.4f}".format(batch_loss) +
							  " , Age accuracy= " + "{0:0.5f}".format(batch_accuracy_age) +
							  " , Job accuracy= " + "{0:0.5f}".format(batch_accuracy_job) +
							  " , Gender accuracy= " + "{0:0.5f}".format(batch_accuracy_gender) +
							  " , Progress= " + "{0:2.2f}".format((float(batch) / training_batch_count) * 100) + "%")

						batch_loss = 0.0
						batch_loss_age = 0.0
						batch_loss_job = 0.0
						batch_loss_gender = 0.0
						batch_accuracy_job = 0.0
						batch_accuracy_age = 0.0
						batch_accuracy_gender = 0.0

				# VALIDATION
				batch_accuracy_age = 0.0
				batch_accuracy_job = 0.0
				batch_accuracy_gender = 0.0
				batch_loss = 0.0
				batch_loss_age = 0.0
				batch_loss_job = 0.0
				batch_loss_gender = 0.0

				for batch in range(valid_batch_count):
					# prepare the batch
					valid_batch_x, \
					valid_batch_y_age, \
					valid_batch_y_job, \
					valid_batch_y_gender, \
					valid_batch_seqlen = prepWordBatchData(valid_data,
															valid_users,
															ground_truth,
															vocabulary,
															batch)


					# Prepare feed values
					feed_dict = {network.X: valid_batch_x,
								 network.Y_age: valid_batch_y_age,
								 network.Y_job: valid_batch_y_job,
								 network.Y_gender: valid_batch_y_gender,
								 network.sequence_length: valid_batch_seqlen,
								 network.reg_param: FLAGS.l2_reg_lambda}

					# Run the computational graph
					_, loss_age, loss_job, loss_gender, loss, \
					prediction_age, prediction_job, prediction_gender, \
					accuracy_age, accuracy_job, accuracy_gender = sess.run(
						[network.train, network.loss_age, network.loss_job, network.loss_gender, network.loss,
						 network.prediction_age, network.prediction_job, network.prediction_gender,
						 network.accuracy_age, network.accuracy_job, network.accuracy_gender], feed_dict=feed_dict)

					# Calculate the metrics
					batch_loss += loss
					epoch_loss += loss

					batch_loss_age += loss_age
					epoch_loss_age += loss_age

					batch_loss_job += loss_job
					epoch_loss_job += loss_job

					batch_loss_gender += loss_gender
					epoch_loss_gender += loss_gender

					batch_accuracy_age += accuracy_age
					epoch_accuracy_age += accuracy_age

					batch_accuracy_job += accuracy_job
					epoch_accuracy_job += accuracy_job

					batch_accuracy_gender += accuracy_gender
					epoch_accuracy_gender += accuracy_gender


				# print the accuracy and progress of the validation
				batch_accuracy_age /= valid_batch_count
				batch_accuracy_job /= valid_batch_count
				batch_accuracy_gender /= valid_batch_count

				epoch_accuracy_age /= training_batch_count
				epoch_accuracy_job /= training_batch_count
				epoch_accuracy_gender /= training_batch_count

				print("Epoch " + str(epoch) + " Training loss: " + "{0:5.4f}".format(epoch_loss))
				print("Epoch " + str(epoch) + " Training age accuracy: " + "{0:0.5f}".format(epoch_accuracy_age))
				print("Epoch " + str(epoch) + " Training job accuracy: " + "{0:0.5f}".format(epoch_accuracy_job))
				print("Epoch " + str(epoch) + " Training gender accuracy: " + "{0:0.5f}".format(epoch_accuracy_gender))
				print("Epoch " + str(epoch) + " Validation loss: " + "{0:5.4f}".format(batch_loss))
				print("Epoch " + str(epoch) + " Validation age accuracy: " + "{0:0.5f}".format(batch_accuracy_age))
				print("Epoch " + str(epoch) + " Validation job accuracy: " + "{0:0.5f}".format(batch_accuracy_job))
				print("Epoch " + str(epoch) + " Validation gender accuracy: " + "{0:0.5f}".format(batch_accuracy_gender))

				'''
				if FLAGS.optimize:
					f = open(FLAGS.log_path, "a")

					training_loss_line = "Epoch " + str(epoch) + " training loss: " + str(epoch_loss) + "\n"
					training_accuracy_line = "Epoch " + str(epoch) + " training accuracy: " + str(epoch_accuracy) + "\n"
					validation_loss_line = "Epoch " + str(epoch) + " validation loss: " + str(batch_loss) + "\n"
					validation_accuracy_line = "Epoch " + str(epoch) + " validation accuracy: " + str(
						batch_accuracy) + "\n"

					f.write(training_loss_line)
					f.write(training_accuracy_line)
					f.write(validation_loss_line)
					f.write(validation_accuracy_line)

					f.close()

				# save the model if it performs above the threshold
				# naming convention for the model : {"language"}-model-{"learning rate"}-{"reg. param."}-{"epoch number"}
				if batch_accuracy >= FLAGS.model_save_threshold:
					model_name = str(FLAGS.lang) + "-model-" + str(FLAGS.rnn_cell_size) + "-" + str(
						FLAGS.learning_rate) + "-" + str(FLAGS.l2_reg_lambda) + "-" + str(epoch) + ".ckpt"
					save_as = os.path.join(FLAGS.model_path, model_name)
					save_path = saver.save(sess, save_as)
					print("Model saved in path: %s" % save_path)
				'''
