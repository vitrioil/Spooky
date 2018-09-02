import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#train = pd.read_csv("../../input/train.csv")
#test = pd.read_csv("../../input/test.csv")
#text = list(train["text"])[:20]+list(test["text"])[:20]

#training data
sent = ["nlp is part of ml","dl is part of ml","ml is powerful","dl is more powerful than ml","nlp is language analysis","ml, dl and nlp is data analysis","cnn is powerful","ml and dl are not equivalent","nlp uses rnn","cnn is used in dl a lot"]

#tokenize and accumulate
word_set = set()
for s in sent:
	for w in nltk.word_tokenize(s): 
		word_set.add(w)
if len(word_set) >= 10_000:
	assert False,"OHHH MY GOOOOOOOD!!!!!!!!!!!!!!!!!!!!!!!!!!" 
word_to_id = {w:idx for idx,w in enumerate(word_set)}
co_mat = np.zeros((len(word_set),len(word_set)),dtype=np.int)

#distance between words
distance = 5

#calculate how often two words appear within the word_distance defined above
for l in sent:
	words = nltk.word_tokenize(l)
	for idx,w in enumerate(words):
		next_words,prev_words = [],[]
		# Maybe add distance
		if idx != 0:
			i = idx - 1
			while i>=0 and (idx - i) < distance:
				prev_words.append(words[i])
				i -= 1
		if idx != len(words) - 1:
			i = idx + 1
			while i<=len(words) - 1 and (i - idx) < distance:
				next_words.append(words[i])
				i += 1
		nearby_words = prev_words + next_words
		idx2 = word_to_id[w]
		for  near in nearby_words:
			idx1 = word_to_id[near]
			co_mat[idx1][idx2] += 1
			co_mat[idx2][idx1] += 1

#Calculate the singular value decomposition
U,s,V_ = np.linalg.svd(co_mat)
U = V_.T.copy()
print(U.shape)
word_list = list(word_set)
'''
for i in range(U.shape[0]):
	plt.text(U[i,0],U[i,1],word_list[i]) 
plt.show()
'''
#Test the data
sent1 = "cnn is part of dl"
sent2 = "cnn is part of ml"
sent3 = "data analysis is used a lot"
def find_diff(sent1,sent2,dim=2):
	vec1 = np.zeros((2,1),dtype=np.float)
	vec2 = np.zeros((2,1),dtype=np.float)
	print(vec1.shape,np.array([1,2])[...,np.newaxis].shape)
	for i in nltk.word_tokenize(sent1):
		indx = word_to_id[i]
		vec1 += np.array([U[indx,0],U[indx,1]])[...,np.newaxis]
	for i in nltk.word_tokenize(sent2):
		indx = word_to_id[i]
		vec2 += np.array([U[indx,0],U[indx,1]])[...,np.newaxis]
	return np.sum(np.abs(vec1-vec2))
print(find_diff(sent1,sent2))
print(find_diff(sent1,sent3))
print(find_diff(sent2,sent3))