import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
text = train["text"][:20]+test["text"][:20]

sent = ["I like python","python is easier to learn","python is loved by a lot of people"]

word_set = set()
for s in sent:
	for w in nltk.word_tokenize(s): 
		word_set.add(w)
if len(word_set) >= 10_000:
	assert False,"OHHH MY GOOOOOOOD!!!!!!!!!!!!!!!!!!!!!!!!!!" 
word_to_id = {w:idx for idx,w in enumerate(word_set)}
co_mat = np.zeros((len(word_set),len(word_set)),dtype=np.int)

distance = 3

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
U,s,V_ = np.linalg.svd(co_mat)

print(U.shape)
word_list = list(word_set)
for i in range(U.shape[0]):
	plt.text(U[i,0],U[i,1],word_list[i]) 
plt.show()
