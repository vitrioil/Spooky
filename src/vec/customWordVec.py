#Stemming
import os
import gc
import sys
import time
import nltk
import keras
import gensim
import pickle
import psutil
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

def prepare_logger():
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)

	handler = logging.FileHandler("train.log")
	handler.setLevel(logging.INFO)

	logger.addHandler(handler)
	return logger

logger = prepare_logger()
stopwords = set(nltk.corpus.stopwords.words("english"))
lemmatizer = nltk.stem.WordNetLemmatizer()
def memory_above(critical_val = 95):
	return (psutil.virtual_memory().percent > critical_val)

def imdb_dataset():
	NUM_WORDS=1000 # only use top 1000 words
	INDEX_FROM=3   # word index offset
	
	train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)
	train_x,train_y = train
	test_x,test_y = test
	
	word_to_id = keras.datasets.imdb.get_word_index()
	word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
	word_to_id["<PAD>"] = 0
	word_to_id["<START>"] = 1
	word_to_id["<UNK>"] = 2
	
	id_to_word = {value:key for key,value in word_to_id.items()}
	document = []
	for sent_id in train_x:
		sentence = []
		for word_id in sent_id:
			sentence.append(id_to_word[word_id])
		document.append(sentence)
	return np.array(document), train_x 

def shuffle(f):    
	perm = np.random.permutation(len(f))
	f = f[perm]
	for item in f:
		yield item

def tokenize(document):
	for line in document:
		yield nltk.word_tokenize(line)

def remove_stopwords(tokenized_list):
	filtered_list = []
	for word in tokenized_list:
		if word not in stopwords:
			filtered_list.append(word)
	return filtered_list

def lemmatize_words(tokenized_list):
	filtered_list = []
	word_pos = nltk.pos_tag(tokenized_list)
	for word, pos in word_pos:
		word = lemmatizer.lemmatize(word, pos)
		filtered_list.append(word)
	return filtered_list

def preprocess(location="../input/train.csv", document=None, tokenized=False):
	word_set = {}
	if document is None:
		df = pd.read_csv(location)
		document = df["text"]
	processed_doc = []
	for sent in document:
		processed_sent = []
		if not tokenized:
			sent = tokenize(sent)
		for word in lemmatize_words(remove_stopwords(sent)):
			processed_sent.append(word)
			if word_set.get(word) is None:
				word_indx = len(word_set) + 1
				word_set[word] = word_indx

		processed_doc.append(processed_sent)
	print(document[0])
	print(processed_doc[0])

	def generate_pairs(word_distance, batch_size):
		context,target = [],[]
		total_sentences = len(processed_doc) 
		for sent_indx, tokenized_list in enumerate(shuffle(processed_doc)):
			if not tokenized:
				tokenized_list = tokenize(tokenized_list)
			tokenized_list = list(remove_stopwords(tokenized_list))
			for indx, context_word in enumerate(tokenized_list):
				#Sampling is suggested
				context_row = [] 
				for distance in range(indx - word_distance, indx + word_distance):
					if distance == indx or distance < 0 or distance >= len(tokenized_list):
						context_row.append(0)
						continue
					context_row.append(word_set[tokenized_list[distance]])
				context.append(context_row)
				target.append(word_set[context_word])
				#Tensorflow gives not a matrix error for only one example ???
				if (len(context) == batch_size or sent_indx == total_sentences - 1) and len(context) != 1:
					yield np.array(context), np.array(target)[..., np.newaxis]
					context, target = [], []

	return generate_pairs, len(word_set), word_set

def initialize_parameters(vocab_size, embedding_size):
	vocab_size += 1 # Adjust for padding
	embeddings = tf.Variable(
			tf.random_uniform(
				[vocab_size, embedding_size],
				-1,
				 1
				)
			)
	
	nce_mat = tf.Variable(
			tf.random_uniform(
				[vocab_size, embedding_size],
				-1,
				 1
			)
		)
	
	nce_bias = tf.Variable(
			tf.zeros(
				[vocab_size]
			)
		)

	return (embeddings, nce_mat, nce_bias)

def model(vocab_size, embedding_size, pair, lr, word_distance):
	embeddings, nce_mat, nce_bias = initialize_parameters(vocab_size, embedding_size)
	(inputs, labels) = pair	

	embed = tf.nn.embedding_lookup(embeddings, inputs)
	sum_embed = tf.squeeze(tf.reduce_sum(embed, axis=1, keep_dims=True))
	loss = tf.reduce_mean(
			tf.nn.nce_loss(
				weights = nce_mat,
				biases = nce_bias,
				labels = labels,
				inputs = sum_embed,
				num_sampled = 10,
				num_classes = vocab_size
			)
		)
	optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

	output = sum_embed @ tf.transpose(nce_mat) + nce_bias

	correct_prediction = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,1), tf.argmax(labels, 1)), tf.float32),name="accuracy")
	return optimizer, loss, correct_prediction


def train(embedding_size = 100, word_distance = 5, batch_size = 64, lr = 1e-3, epochs = 10):
	tf.reset_default_graph()
	timestamp = str(int(time.time()))
	logdir = "model/"+timestamp+"/"

	inputs = tf.placeholder(tf.int32, [None, 2*word_distance])
	labels = tf.placeholder(tf.int32, [None, 1])
	
	document, train_x = imdb_dataset()
	generate_pairs, vocab_size, word_set = preprocess(itr = document, tokenized = True)
	
	optimizer, loss, correct_prediction = model(vocab_size, embedding_size, (inputs, labels), lr, word_distance)
	
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		try:
			sess.run(init)
			saver = tf.train.Saver()
			for e in range(epochs):	
				epoch_loss = 0
				epoch_acc = 0
				for indx,pair in enumerate(generate_pairs(word_distance, batch_size)):
					if memory_above(95):
						log_str = f"epoch:{e} indx:{indx}, terminating! high memory usage"
						logger.info(log_str) 				
						print(log_str)
						del pair
						gc.collect()
						break
					context, target = pair
					_, c  = sess.run(
							[optimizer, loss],
							feed_dict = {inputs:context, labels:target}
						)
					epoch_loss += c
					epoch_acc += 0
					if indx % (50) == 0:
						logger.info(f"{timestamp} epoch:{e} indx:{indx}, loss:{epoch_loss} ") 
						print(f"epoch:{e} indx:{indx}, loss:{epoch_loss} ", end="\r")
					if indx % (1_00_000) == 0:
						saver.save(sess, os.path.join(logdir, "model.ckpt"), indx)
				print(f"\nLoss: {epoch_loss}")
			saver.save(sess, os.path.join(logdir, "model.ckpt"))
		except KeyboardInterrupt as e:
			print(str(e))
			saver.save(sess, os.path.join(logdir, "model.ckpt"))
		finally:
			save_dictionary = input("Save the dictionary?")
			if save_dictionary.lower() != "no":
				with open(logdir+"word_set.pickle","wb") as f:
					pickle.dump(word_set, f)
def cosine_distance(v1, v2):
	"""returns the cosine_distance"""
	return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))



def load_file(path):
	with open(path+"word_set.pickle","rb") as f:
		word_set = pickle.load(f)
	
	#Tensor weight
	embeddings, nce_mat, nce_bias = initialize_parameters(len(word_set), 100)
	with tf.Session() as sess:
		saver = tf.train.Saver()
		saver.restore(sess, path+"model.ckpt")

		embedding = embeddings.eval()
	return word_set,embedding

def test(path):
	word_set, embedding = load_file(path)
	idx_to_word = {idx:word for word,idx in word_set.items()}
	def most_similar(word,topn=5):
		word_indx = word_set.get(word)
		if word_indx is None:
			print("Word not found")
			return None
		vec = embedding[word_indx]
		top_matches = [(None,-np.inf) for i in range(5)]
		for indx,emb_vec in enumerate(embedding, 1):
			distance = cosine_distance(vec, emb_vec)
			if distance > top_matches[-1][1] and word_indx != indx:
				top_matches[-1] = (idx_to_word[indx], distance)
				top_matches.sort(key = lambda x:x[1], reverse = True)
		print(top_matches)
	return most_similar


if __name__ == "__main__":
	print(sys.argv)
	if sys.argv[1] == "train":
		train(embedding_size=100, word_distance=3, batch_size=32, lr=1e-2)
	if sys.argv[1] == "test":
		if len(sys.argv) != 3:
			print("Enter directory")
			sys.exit()
		most_similar = test(f"model/{sys.argv[2]}/")
		while True:
			word = input("=>")
			most_similar(word)
