import tokenization

fWrite = open("./ooc.txt", "a+" , encoding="utf8")

tokenizer = tokenization.FullTokenizer(vocab_file="/home/darg1/Desktop/ozan/bert_config/char_vocab.txt", do_lower_case=True)


with open("/media/darg1/Data/ozan_bert_data/wikipedi_dump/wiki_dump_organised.txt", "r",encoding="utf8") as fRead:
	for line in fRead:
		for token in tokenizer.tokenize(line):
			for char in token:
				if char not in list(tokenizer.vocab.keys()):
					fWrite.write(char+"\n")
					tokenizer.vocab[char] = len(list(tokenizer.vocab.keys()))

fWrite.close()

