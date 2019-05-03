import os
import xml.etree.cElementTree as ET
import numpy as np
from tqdm import tqdm
import itertools
import sys


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
		root = ET.parse(data_path).getroot()
		data_point = []
		data_point.append(text.split(".xml")[0])

		for entity_paragraph in root.findall("doc"):
			data_point.append(entity_paragraph.text)

		data.append(data_point)

	return data


#Todo: Memory issue for embeddings in RAM, control with 32 RAM otherwise use smaller embedding set
#########################################################################################################################
# Read GloVe embeddings
#
# input: path (String)        - Path of embeddings to read
#
# output: embeddings (Dict)   - Dictionary of the embeddings
def read_embeddings(path):
	embeddings =  {}

	with open(path, "r", encoding="utf-8") as f:
		for line in tqdm(f):
			values = line.strip().split(" ")
			word = values.pop(0)
			embeddings[word] = list(map(float, values))

	return embeddings