import preprocess

dataset_path = "/mnt/671728fd-b9e2-46ed-b18b-9f45f387f63e/yl_tez/dataset"
embeddings_path = "/mnt/671728fd-b9e2-46ed-b18b-9f45f387f63e/yl_tez/glove.840B.300d.txt"
data = preprocess.read_data(dataset_path)
embeddings = preprocess.read_embeddings(embeddings_path)