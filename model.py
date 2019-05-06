from flags import FLAGS

#TODO: Implement the model's training and test, make it as smooth as state change
class Model:
	def __init__(self, training_data, test_data, embeddings, mode):
		self.mode = mode
		self.training_data = training_data
		self.test_data = test_data
		self.embeddings = embeddings

	def train(self):
		if self.mode == "Embedding average":
			self.embedding_average()

	def test(self):
		pass

	def embedding_average(self):
		pass

