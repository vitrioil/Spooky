import os
import time
import nltk
import gensim
import pickle
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

def read_input(f):    
	perm = np.random.permutation(len(f))
	f = f[perm]
	for i, line in enumerate(f): 
		yield nltk.word_tokenize(line)

def preprocess(location):
	df = pd.read_csv(location)
	word_set = {}
	for sent in df["text"]:
		for word in nltk.word_tokenize(sent):
			if word_set.get(word) is None:
				word_indx = len(word_set) 
				word_set[word] = word_indx

	def generate_pairs(word_distance, batch_size):
		context,target = [],[]
		for sentence in read_input(df["text"]):
			for indx,words in enumerate(sentence):
				for distance in range(indx - word_distance,indx + word_distance):
					if distance == indx or distance < 0 or distance >= len(sentence):
						continue
					context.append(word_set[sentence[distance]])
					target.append(word_set[words])
					if len(context) == batch_size:
						yield np.array(context),np.array(target)[...,np.newaxis]
						context,target = [],[]

	return generate_pairs, len(word_set)		

def initialize_parameters(vocab_size, embedding_size):
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

def model(vocab_size, embedding_size, pair, lr):
	embeddings, nce_mat, nce_bias = initialize_parameters(vocab_size, embedding_size)
	(inputs, labels) = pair	

	embed = tf.nn.embedding_lookup(embeddings, inputs)
	
	loss = tf.reduce_mean(
			tf.nn.nce_loss(
				weights = nce_mat,
				biases = nce_bias,
				labels = labels,
				inputs = embed,
				num_sampled = 10,
				num_classes = vocab_size
			)
		)
	optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

	output = embed @ tf.transpose(nce_mat) + nce_bias

	correct_prediction = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,1), tf.argmax(labels, 1)), tf.float32),name="accuracy")
	return optimizer, loss, correct_prediction


def train(embedding_size = 100, word_distance = 10, batch_size = 64, lr = 1e-3, epochs = 10):
	tf.reset_default_graph()
	logdir = "model/"+str(int(time.time()))+"/"

	labels = tf.placeholder(tf.int32, [None, 1])
	inputs = tf.placeholder(tf.int32, [None])
	
	generate_pairs, vocab_size = preprocess("../input/train.csv")
	optimizer, loss, correct_prediction = model(vocab_size, embedding_size, (inputs, labels), lr)
	

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		try:
			sess.run(init)
			saver = tf.train.Saver()
			for e in range(epochs):	
				epoch_loss = 0
				epoch_acc = 0
				for indx,pair in enumerate(generate_pairs(word_distance, batch_size)):
					context, target = pair
					_, c, a  = sess.run(
							[optimizer, loss, correct_prediction],
							feed_dict = {inputs:context, labels:target}
						)
					epoch_loss += c
					epoch_acc += a
					if indx % (500) == 0:
						logger.info(f"epoch:{e} indx:{indx}, loss:{epoch_loss} ") 
						print(f"epoch:{e} indx:{indx}, loss:{epoch_loss} ")
				print(f"Accuracy {epoch_acc/indx}")
			saver.save(sess, os.path.join(logdir, "model.ckpt"))
		except KeyboardInterrupt as e:
			print(str(e))
			saver.save(sess, os.path.join(logdir, "model.ckpt"))

def cosine_distance(v1, v2):
	return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))



def load_file(path):
	with open(path+"words/word_set.pickle","rb") as f:
		word_set = pickle.load(f)
	
	#Tensor weight
	embeddings, nce_mat, nce_bias = initialize_parameters(len(word_set), 100)
	with tf.Session() as sess:
		saver = tf.train.Saver()
		saver.restore(sess, "model/1536497133/model.ckpt")

		embedding = embeddings.eval()
	return word_set,embedding

def test():
	word_set, embedding = load_file("model/1536497133/")
	idx_to_word = {idx:word for word,idx in word_set.items()}
	def most_similar(word,topn=5):
		word_indx = word_set.get(word)
		if word_indx is None:
			print("Word not found")
			return None
		vec = embedding[word_indx]
		top_matches = [(None,-np.inf) for i in range(5)]
		for indx,emb_vec in enumerate(embedding):
			distance = cosine_distance(vec, emb_vec)
			if distance > top_matches[-1][1] and word_indx != indx:
				top_matches[-1] = (idx_to_word[indx], distance)
				top_matches.sort(key = lambda x:x[1], reverse = True)
		print(top_matches)
	return most_similar


if __name__ == "__main__":
	train()
	most_similar = test()
	while True:
		word = input(":")
		most_similar(word)
