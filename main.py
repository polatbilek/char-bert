import preprocess
from flags import FLAGS
from embedding_average import Embed_Average
from tqdm import tqdm

training_data, test_data = preprocess.get_data(FLAGS.dataset_path)
embeddings = preprocess.read_embeddings(FLAGS.embedding_path)

model = Embed_Average(training_data, test_data, embeddings)

vecs = model.train()
model.test()
