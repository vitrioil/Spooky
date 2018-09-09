import os
import nltk
import gensim
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

print(os.listdir("../input/"))
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
				word_indx = len(word_set) + 1
				word_set[word] = word_indx

	def generate_pairs(word_distance):
		for sentence in read_input(df["text"]):
			for indx,words in enumerate(sentence):
				for distance in range(indx - word_distance,indx + word_distance):
					if distance == indx or distance < 0 or distance >= len(sentence):
						continue
					yield (np.array([word_set[sentence[distance]]]), np.array([word_set[words]])[...,np.newaxis])

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


def train(embedding_size = 200, word_distance = 10, batch_size = 32, lr = 1e-5, epochs = 10):
	tf.reset_default_graph()

	labels = tf.placeholder(tf.int32, [None, 1])
	inputs = tf.placeholder(tf.int32, [None])
	
	generate_pairs, vocab_size = preprocess("../input/train.csv")
	optimizer, loss, correct_prediction = model(vocab_size, embedding_size, (inputs, labels), lr)
	

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		saver = tf.train.Saver()
		for e in range(epochs):	
			epoch_loss = 0
			epoch_acc = 0
			for indx,pair in enumerate(generate_pairs(word_distance)):
				context, target = pair
				_, c, a  = sess.run(
						[optimizer, loss, correct_prediction],
						feed_dict = {inputs:context, labels:target}
					)
				epoch_loss += c
				epoch_acc += a
				if indx % (100) == 0:
					logger.info(f"epoch:{e} indx:{indx}, loss:{epoch_loss} accuracy:{epoch_acc}") 
					print(f"epoch:{e} indx:{indx}, loss:{epoch_loss} accuracy:{epoch_acc}")
		saver.save(sess, os.path.join(logdir, "model.ckpt"))

def test():
	pass


if __name__ == "__main__":
	train()
