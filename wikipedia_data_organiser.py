from wiki_dump_reader import Cleaner, iterate
from tqdm import tqdm
import re
from nltk import tokenize
import sys

PATH_WIKI_XML = "/home/darg2/Downloads/enwiki-latest-pages-articles.xml"
path_to_organised_data = "/home/darg2/Desktop/organised.txt"

def process_text(sample):
	partitioned_text = ""
	formatted_text = ""

	sample = sample.replace("i.e.", "i.e")
	sample = sample.replace("e.g.", "e.g")

	splitted = re.split("===*", sample) #this regex looks for "==" or any string where "==" is followed by multiple "="

	# Here there is a trick. I split the text with "==" parts but the content titles are written in "== title ==" format
	# So when you split with "==" the string "title" also comes, to get rid of that, i only get even indexed splitted
	# Example: "passage1 == title1 == passage2 == title2 ==" becomes ["passage1", "title1", "passage2", "title2"]
	# When you get only even indexeds, final_form = ["passage1", "passage2"] so only passages
	i = 0
	while i < len(splitted):
		# removes texts in curly brackets or paranthesis which are equations or CSS tags, or meaningless text
		splitted[i] = re.sub("\{[^)]*\}", "", splitted[i])

		# additional care for css tags
		if "wikitable" in splitted[i]:
			splitted[i] = re.sub("\n*","",splitted[i])
			splitted[i] = re.sub("\{.*\}", "", splitted[i])

		partitioned_text += splitted[i]

		# if we came until "see also" section then there is no content after this point, so we break the loop.
		if i+1 < len(splitted) and "See also" in splitted[i+1]:
			partitioned_text = partitioned_text[:-1] #in see also case, one extra new line char stays for no reason
			break

		i += 2 #to get even indexes

	#postprocessing
	partitioned_text = re.sub("\n\n\n\n*", "\n\n", partitioned_text) #get rids of more than 2 new lines,
																	#makes them all 2 new lines

	for passage in partitioned_text.split("\n\n"): #we tokenize each passage one by one to be able to seperate them
		for sentence in tokenize.sent_tokenize(passage): #now we are ready to split sentences
			formatted_text += sentence + "\n"
		formatted_text += "\n"

	formatted_text = formatted_text.replace(".\n", "\n") #delete the dots at the end of the sentences

	# somehow, multiple new line problem occurs, so we must solve it
	formatted_text = re.sub("\n\n\n\n*", "\n\n", formatted_text)

	return formatted_text

if __name__ == "__main__":
	cleaner = Cleaner()
	f = open(path_to_organised_data, "w+", encoding="utf8")

	for title, text in tqdm(iterate(PATH_WIKI_XML)):
		text = cleaner.clean_text(text)
		cleaned_text, links = cleaner.build_links(text)

		if "REDIRECT" not in cleaned_text:
			text_in_format = process_text(cleaned_text)
			f.write(text_in_format)

	f.close()
	print("Everything is processed :)")