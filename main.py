import preprocess
from flags import FLAGS
from model import Model


training_data, test_data = preprocess.get_data(FLAGS.dataset_path)
embeddings = preprocess.read_embeddings(FLAGS.embedding_path)

model = Model(training_data, test_data, embeddings, "Embedding average")
model.train()
model.test()
