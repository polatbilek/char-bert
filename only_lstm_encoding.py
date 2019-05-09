from flags import FLAGS
from tqdm import tqdm
import numpy as np
from nltk import casual_tokenize
from scipy.spatial import distance
import tensorflow as tf
import sys

#TODO: Implement the model's training and test, make it as smooth as state change
class lstm_encoding:
	def __init__(self, training_data, test_data, embeddings):
		self.training_data = training_data
		self.test_data = test_data
		self.embeddings = embeddings


	def train(self):
		pass



	def test(self):
		pass