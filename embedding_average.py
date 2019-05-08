from flags import FLAGS
from tqdm import tqdm
import numpy as np
from nltk import casual_tokenize
import sys

#TODO: Implement the model's training and test, make it as smooth as state change
class Embed_Average:
	def __init__(self, training_data, test_data, embeddings):
		self.training_data = training_data
		self.test_data = test_data
		self.embeddings = embeddings

	def train(self):
		self.vectors = []

		for data in tqdm(self.training_data):
			for text in data:
				if data.index(text) == 0:
					self.vectors.append(text)

				else:
					text_vector = np.zeros(FLAGS.embedding_size)
					total = 1

					for word in casual_tokenize(text):
						try:
							text_vector = np.add(np.ndarray(self.embeddings[word]) + text_vector)
							total += 1
						except:
							continue

					text_vector = text_vector / total
					self.vectors.append(text_vector.tolist())

		return self.vectors




	def test(self):
		pass

