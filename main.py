import preprocess
from flags import FLAGS
from model import Model


data = preprocess.get_data(FLAGS.dataset_path)
embeddings = preprocess.read_embeddings(FLAGS.embedding_path)

model = Model()
model.train()
model.test()
