import nltk
import pyemd
import scipy
import pandas
import pickle
import sklearn
import numpy as np

class WMD:
	def __init__(self, path):
		self.embedding, self.word_set = self.load_embedding(path)
		self.rev_set = dict(zip(self.word_set.values(), self.word_set.keys()))
		self.vocab_len, self.embedding_size = self.embedding.shape
		self.stopwords = set(nltk.corpus.stopwords.words("english")) 

	def load_embedding(self, path, vocab_path="vocab_pruned.pickle"):
		embedding = np.load(path, mmap_mode="r")
		with open(vocab_path, "rb") as f:
			vocab = pickle.load(f)
		word_set = {word:indx for indx,word in enumerate(vocab)}
	
		return embedding, word_set
	
	def process(self, file_df, word_set):
		doc = []
		for chunk in file_df:
			for sentence in chunk:
				sample_sent = []
				for word in sentence:
					sample_sent.append(word_set[word])
	def tokenize(self, sent):
		return nltk.word_tokenize(sent)

	def remove_stopwords(self, sent):
		return [i for i in sent if i not in self.stopwords]

	def _sent_to_sparse(self, sent, remove_stop = True):
		if isinstance(sent, str):
			sent = self.tokenize(sent)
		if remove_stop:
			sent = self.remove_stopwords(sent)
		col = []
		for word in sent:
			if self.word_set.get(word) is None:
				print(f"Word {word} not in vocab")
				return
			indx = self.word_set[word]
			if indx not in col:
				col.append(indx)

		row = [0]*len(col)
		words, count = np.unique(sent, return_counts=True)
		count_map = dict(zip(map(self.word_set.get,words), count))
		data = [count_map[indx] for indx in col]
		
		sparse = scipy.sparse.csr_matrix((data, (row, col)), shape=(1, self.vocab_len), dtype=np.float64)
		sparse = sklearn.preprocessing.normalize(sparse, norm="l1", copy=False)
		return sparse

	def wmd(self, sent1="this is sentence", sent2="this is sentence"):
		sp1 = self._sent_to_sparse(sent1)
		sp2 = self._sent_to_sparse(sent2)
		if sp1 is None or sp2 is None:
			return 
		union_idx = np.union1d(sp1.indices, sp2.indices)

		W = sklearn.metrics.euclidean_distances(self.embedding[union_idx])
		W = W.astype("float64")	

		sp1 = sp1[:, union_idx].A.ravel()
		sp2 = sp2[:, union_idx].A.ravel()
		print(sp1,sp2)
		return pyemd.emd(sp1, sp2, W)

if __name__ == "__main__":
	wmd = WMD("word_embedding.npy")
	while True:
		sent1 = input("Sentence 1: ")
		sent2 = input("Sentence 2: ")
		print("Distance is", wmd.wmd(sent1, sent2))
