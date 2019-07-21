class flags(object):

	def __init__(self):

		#set sizes
		self.test_set_size = 0.0
		self.validation_set_size = 0.2
		self.training_set_size = 0.8

		#input file paths
		self.word_embed_path = "C:\\Users\\polat\\Desktop\\fasttext\\crawl-300d-2M.vec" #"/home/darg2/Desktop/crawl-300d-2M.vec"
		self.training_data_path = "C:\\Users\\polat\\Desktop\\new_blogs" #"/home/darg2/Desktop/blogs"
		self.test_data_path = "/home/darg2/Downloads/turkish-twitter-dataset-gender-labeled/turk-tweet-test"
		self.char_embed_path = "./char_embeddings.27B.25d.txt"


		#output file paths
		self.model_path = "/media/cvrg/HDD/darg/models/ar"
		self.model_name = "en-model-0.001-0.0001-0.ckpt"
		self.log_path = "/home/cvrg/darg/logs/logs_CNN_RNN_ar.txt"


		#optimization parameters
		self.lang = "tr"
		self.model_save_threshold = 0.75
		self.optimize = False #if true below values will be used for hyper parameter optimization, or if testing is run: all the models in model_path will be tested
							 #if false hyperparameters specified in "model hyperparameters" will be used, and for testing model with model_name and model_path will be used
		self.l_rate = [0.01]
		self.reg_param = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
		self.rnn_cell_sizes = [90, 120, 150]
		self.cnn_filter_counts = [60, 80, 100]




		#########################################################################################################################
		# Model Hyperparameters
		self.l2_reg_lambda = 0.00005
		self.learning_rate = 0.001
		self.num_classes = 2
			#CNN
		self.num_filters = 100
		self.sequence_length = 190
		self.char_embedding_size = 25
		self.filter_sizes = "3,6,9"
			#RNN
		self.word_embedding_size = 200
		self.rnn_cell_size = 150
		self.fc_size = 2*self.rnn_cell_size





		##########################################################################################################################
		# Training parameters
		self.use_pretrained_model = False
		self.tweet_per_user = 100
		self.batch_size = 10
		self.num_epochs = 18
		self.evaluate_every = 5



FLAGS = flags()