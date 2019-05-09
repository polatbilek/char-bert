import os
import xml.etree.cElementTree as ET
import numpy as np
from tqdm import tqdm
import sys
from flags import FLAGS


#####################################################################
#   Inputs: path(String)
#	Outputs: data(list)
#
#	Reads data in following structure:
#
#	N ambigous NE texts
#	Each element in N starts with the name of the NE
#	Following lists are documents of that NE in one meaning
#
#	(N,1+M) N = # of ambigous NE, M = # of meanings of that NE
#
def read_data(path):
	data = []

	print("Reading dataset...")
	entity_texts = os.listdir(path)

	for text in tqdm(entity_texts):
		data_path = os.path.join(path, text)
		data_point = []

		root = ET.parse(data_path).getroot()
		data_point.append(text.split(".xml")[0])

		for entity_paragraph in root.findall("doc"):
			data_point.append(entity_paragraph.text)

		data.append(data_point)

	return data


#########################################################################################################################
# Read GloVe embeddings
#
# input: path (String)        - Path of embeddings to read
#
# output: embeddings (Dict)   - Dictionary of the embeddings
def read_embeddings(path):
	embeddings = {}

	print("Reading embeddings...")
	with open(path, "r", encoding="utf-8") as f:
		for line in tqdm(f):
			values = line.strip().split(" ")
			word = values.pop(0)
			embeddings[word] = list(map(float, values))

	FLAGS.embedding_size = len(embeddings['the'])
	embeddings["UNK"] = np.random.randn(FLAGS.embedding_size)
	embeddings["PAD"] = np.zeros(FLAGS.embedding_size)

	return embeddings


#########################################################################################################################
# Partition whole dataset into train and test
#
# input: data (list)        - list of string as data (N, 1+M)
#
# output: training_data, test_data (list)   - List of string as partitioned data
def partition_data(data):

	training_data = data[0:int(len(data)*FLAGS.training_size)]
	test_data = data[int(len(data)*FLAGS.training_size):-1]

	return training_data, test_data


#########################################################################################################################
# Lowercasing data - needed as function because data type is so custom to do it inside another function
#
# input: data (list)     - list of string as data (N, 1+M)
#
# output: new_data (list)   - List of string as data-lowercased
def lowercase_data(data):
	new_data = []

	for point in data:
		new_point = []

		for entity in point:
			if point.index(entity) != 0:
				new_point.append(entity.lower())
			else:
				new_point.append(entity)

		new_data.append(new_point)

	return new_data


#########################################################################################################################
# Pipeline of the processes that will occur on the data itself
#
# input: path (string)     - Path of the dataset
#
# output: training_data, test_data (list)   - training and test data partition
def get_data(path):

	data = read_data(path)
	data = lowercase_data(data)
	training_data, test_data = partition_data(data)

	return training_data, test_data