import os

whole_data = "organised.txt"
target_folder = "./sliced_data"
LIMIT = 40000000

with open(whole_data, "r", encoding="utf8") as f:
	data_point = 0
	doc = ""
	index = 0
	found_flag = False

	for line in f:
		if index == LIMIT or found_flag:
			if line.strip() == "":
				f_write = open(os.path.join(target_folder,"wiki_data_"+str(data_point)+".txt"), "w+", encoding="utf8")
				f_write.write(doc)
				f_write.close()
				print("Processed " + str(data_point) + "th data point")
				doc = ""
				index = 0
				data_point += 1
				found_flag= False
			else:
				found_flag = True

		doc = doc + line
		index += 1

	
	f_write = open(os.path.join(target_folder,"wiki_data_"+str(data_point)+".txt"), "w+", encoding="utf8")
	f_write.write(doc)
	f_write.close()
	print("Processed " + str(data_point) + "th data point")
