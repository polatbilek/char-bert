from flags import FLAGS
from tqdm import tqdm
import numpy as np
from nltk import casual_tokenize
from scipy.spatial import distance
from nltk.corpus import stopwords
import sys

#TODO: Implement the model's training and test, make it as smooth as state change
class Embed_Average:
	def __init__(self, training_data, test_data, embeddings):
		self.training_data = training_data
		self.test_data = test_data
		self.embeddings = embeddings
		self.threshold = 0.2
		self.stop_words = set(stopwords.words('english'))

	def train(self):
		self.vectors = []

		for data in tqdm(self.training_data):
			averaged = []

			for text in data:
				if data.index(text) == 0:
					averaged.append(text)

				else:
					text_vector = np.zeros(FLAGS.embedding_size)
					total = 1

					for word in casual_tokenize(text):
						try:
							if word not in self.stop_words:
								text_vector = np.add(self.embeddings[word] , text_vector.tolist())
								total += 1
						except:
							continue

					text_vector = text_vector / total
					averaged.append(text_vector.tolist())

			self.vectors.append(averaged)

		#With stop words Ratio:    0.0214383949518647
		#Without stop words Ratio: 0.1898713696302888

		return self.vectors




	def test(self):
		pass

