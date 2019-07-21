# This code is created to change "&" character into "&amp;" so that xml file can be read

import os
from tqdm import tqdm
import sys
from nltk.tokenize import TweetTokenizer

path = 'C:\\Users\\polat\\Desktop\\blogs'
target_path = 'C:\\Users\\polat\\Desktop\\new_blogs'
tokenizer = TweetTokenizer()


for blog in tqdm(os.listdir(path)):
	with open(path+"\\"+blog,"r", encoding="ansi") as f:
		new_f = open(target_path+"\\"+blog, "w", encoding="utf8")

		for line in f:
			new_line = ""
			for word in tokenizer.tokenize(line):
				if word not in ["<Blog>", "</Blog>", "<date>", "</date>", "<post>", "</post>"]:
					w = ""
					for char in word:
						if char == "&":
							w += "&amp;"
						elif char == "<":
							w += "&lt;"
						elif char == ">":
							w += "&gt;"
						elif char == "\'":
							w += "&apos;"
						elif char == "\"":
							w += "&quot;"
						else:
							w += char
					word = str(w)
				new_line += " " + word
			new_f.write(new_line)
		new_f.close()



