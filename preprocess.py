import numpy as np
from parameters import FLAGS
import xml.etree.ElementTree as xmlParser
from nltk.tokenize import TweetTokenizer
import os
import sys
import random
import io
from tqdm import tqdm


#########################################################################################################################
# Read FastText Embeddings
#
# input: String (path)        - Path of embeddings to read
#
# output: dict (vocab)        - Dictionary of embeddings as value words as key
def readFastTextEmbeddings(path):
	fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')

	data = {}
	for line in tqdm(fin):
		tokens = line.rstrip().split(' ')
		data[tokens[0]] = np.asarray(list(map(float, tokens[1:])))

	data["<PAD>"] = np.random.randn(FLAGS.embedding_size) #Padding vector
	data["<UNK>"] = np.random.randn(FLAGS.embedding_size) #Unknown word vector

	return data



#########################################################################################################################
# Reads training dataset
# one-hot vectors:
#
# Gender: female = [0,1], male   = [1,0]
#
# Age: There are 8 age group, 10-15,15-20,20-25 ..etc
#	   Length of each onehot is 8 and the index where the age falls into the group is 1 (e.g. 16-> [0,1,0,0,0,0,0,0])
#
# Job: There are 40 jobs, the index where the jobs name has seen is 1 others 0 (if it is 3rd unique job, 2nd index is 1)
#
# input:  string = path  to the training data
# output: ground_truth (dict): key=user_id, value = list [age, job, gender] each feature is a one-hot vector
#         data (list): each element has a list= [sequence_length, user_id, text(as a list of words)]
def readData(path):
	tokenizer = TweetTokenizer()
	job_index = 0

	ground_truth = {}
	jobs = {}
	data = []

	for user in tqdm(os.listdir(path)):
		infos = user.split(".")
		features = []

		# preparing age info of author
		one_hot_age = np.zeros(8)
		one_hot_age[int((int(infos[2])-10)/5)] = 1

		features.append(one_hot_age)

		# preparing job info of author
		if infos[3] not in jobs:
			jobs[infos[3]] = job_index
			job_index += 1

		one_hot_job = np.zeros(40)
		one_hot_job[jobs[infos[3]]] = 1

		features.append(one_hot_job)

		# preparing gender info of author
		if infos[1] == "male":
			features.append([1, 0])
		else:
			features.append([0, 1])


		ground_truth[infos[0]] = features # saving the author info to ground truth dict

		xml_file_name = os.path.join(path, user)
		xmlFile = open(xml_file_name, "r", encoding="utf8")

		# Here we read the xml files, however there are 670 corrupted files, we just ignore them and don't read.
		try:
			for post in xmlParser.parse(xmlFile).findall("post"):
				words = tokenizer.tokenize(post.text)
				data.append([len(words), infos[0], words])
		except:
			pass

	return ground_truth, data


#########################################################################################################################
# Prepares test data
#
# input: List (tweets)  - List of tweets of a user, each tweet has words as list
#        List (user)    - List of usernames
#        dict (target)  - Dictionary for one-hot gender vectors of users
#
# output: List (test_input)  - List of tweets which are padded up to max_tweet_length
#         List (test_output) - List of one-hot gender vector corresponding to tweets in index order
def prepTestData(tweets, user, target):
    # prepare output
    test_output = user2target(user, target)

    # prepare input by adding padding
    tweet_lengths = [len(tweet) for tweet in tweets]
    max_tweet_length = max(tweet_lengths)

    test_input = []
    for i in range(len(tweets)):
        tweet = tweets[i]
        padded_tweet = []
        for j in range(max_tweet_length):
            if len(tweet) > j:
                padded_tweet.append(tweet[j])
            else:
                padded_tweet.append("PAD")
        test_input.append(padded_tweet)

    return test_input, test_output


#########################################################################################################################
# Changes tokenized words to their corresponding ids in vocabulary
#
# input: list (tweets) - List of tweets
#        dict (vocab)  - Dictionary of the vocabulary of GloVe
#
# output: list (batch_tweet_ids) - List of corresponding ids of words in the tweet w.r.t. vocabulary
def word2id(data, vocab):
    batch = []

    for i in range(FLAGS.batch_size): #loop of users
		data_as_wordids = []

		for word in data: #loop in words of tweet
			if word != "PAD":
				word = word.lower()

			try:
				data_as_wordids.append(vocab[word])
			except:
				data_as_wordids.append(vocab["UNK"])

		batch.append(data_as_wordids)

    return batch


#########################################################################################################################
# Prepares batch data, also adds padding to tweets
#
# input: list (tweets)  - List of tweets corresponding to the authors in:
#	     list (users)   - Owner of the tweets
#	     dict (targets) - Ground-truth gender vector of each owner
#	     list (seq_len) - Sequence length for tweets
#	     int  (iter_no) - Current # of iteration we are on
#
# output: list (batch_input)       - Ids of each words to be used in tf_embedding_lookup
# 	      list (batch_output)      - Target values to be fed to the rnn
#	      list (batch_sequencelen) - Number of words in each tweet(gives us the # of time unrolls)
def prepWordBatchData(tweets, users, targets, seq_len, iter_no):
	numof_total_tweet = FLAGS.batch_size * FLAGS.tweet_per_user

	start = iter_no * numof_total_tweet
	end = iter_no * numof_total_tweet + numof_total_tweet

	if end > len(tweets):
		end = len(tweets)

	batch_tweets = tweets[start:end]
	batch_users = users[start:end]
	batch_sequencelen = seq_len[start:end]

	batch_targets = 2#user2target(batch_users, targets)

	# prepare input by adding padding
	tweet_lengths = [len(tweet) for tweet in batch_tweets]
	max_tweet_length = max(tweet_lengths)

	batch_input = []
	for i in range(numof_total_tweet):
		tweet = batch_tweets[i]
		padded_tweet = []
		for j in range(max_tweet_length):
			if len(tweet) > j:
				padded_tweet.append(tweet[j])
			else:
				padded_tweet.append("PAD")
		batch_input.append(padded_tweet)


	#reshape the input for shuffling operation
	tweet_batches = np.reshape(np.asarray(batch_input), (FLAGS.batch_size, FLAGS.tweet_per_user, max_tweet_length)).tolist()
	target_batches = np.reshape(np.asarray(batch_targets), (FLAGS.batch_size, FLAGS.tweet_per_user, 2)).tolist()
	seqlen_batches = np.reshape(np.asarray(batch_sequencelen), (FLAGS.batch_size, FLAGS.tweet_per_user)).tolist()

	#prepare the target values
	target_values = []
	for i in range(len(target_batches)):
		target_values.append(target_batches[i][0]) 
	target_batches = np.reshape(np.asarray(target_values), (FLAGS.batch_size, 2)).tolist()

	'''
	#user level shuffling
	c = list(zip(tweet_batches, target_batches, seqlen_batches))
	random.shuffle(c)
	tweet_batches, target_batches, seqlen_batches = zip(*c)
	'''

	tweet_batches = list(tweet_batches)
	target_values = list(target_values)
	seqlen_batches = list(seqlen_batches)

	#tweet level shuffling
	for i in range(FLAGS.batch_size):
		c = list(zip(tweet_batches[i], seqlen_batches[i]))
		random.shuffle(c)
		tweet_batches[i], seqlen_batches[i] = zip(*c)

	tweet_batches = list(tweet_batches)
	seqlen_batches = list(seqlen_batches)

	return tweet_batches, target_batches, seqlen_batches




#########################################################################################################################
# partites the data into 3 part training, validation, test
#
# input: list (tweets)  - List of tweets corresponding to the authors in:
#	     list (users)   - Owner of the tweets
#	     list (seq_len) - Sequence length for tweets
#
# output: output_format : usagetype_datatype
#         list ("usagetype"_tweets)       - Group of tweets partitioned according to the FLAGS."usagetype"_set_size
# 	      list ("usagetype"_users)        - Group of users partitioned according to the FLAGS."usagetype"_set_size
#	      list ("usagetype"_seqlengths)   - Group of seqlengths partitioned according to the FLAGS."usagetype"_set_size
def partite_dataset(tweets, users, seq_lengths):

    training_set_size = int(len(tweets) * FLAGS.training_set_size)
    valid_set_size = int(len(tweets) * FLAGS.validation_set_size) + training_set_size

    training_tweets = tweets[:training_set_size]
    valid_tweets = tweets[training_set_size:valid_set_size]
    test_tweets = tweets[valid_set_size:]

    training_users = users[:training_set_size]
    valid_users = users[training_set_size:valid_set_size]
    test_users = users[valid_set_size:]

    training_seq_lengths = seq_lengths[:training_set_size]
    valid_seq_lengths = seq_lengths[training_set_size:valid_set_size]
    test_seq_lengths = seq_lengths[valid_set_size:]

    print("\ttraining set size=" + str(len(training_tweets)) + " validation set size=" + str(len(valid_tweets)) + " test set size=" + str(len(test_tweets)))

    return training_tweets, training_users, training_seq_lengths, valid_tweets, valid_users, valid_seq_lengths, test_tweets, test_users, test_seq_lengths


