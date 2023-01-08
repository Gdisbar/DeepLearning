

# #Get one hot representation for any string based on this vocabulary. 
# #If the word exists in the vocabulary, its representation is returned. 
# #If not, a list of zeroes is returned for that word. 
# def get_onehot_vector(somestring):
#     onehot_encoded = []
#     for word in somestring.split():
#         temp = [0]*len(vocab)
#         if word in vocab:
#             temp[vocab[word]-1] = 1 # -1 is to take care of the fact indexing in array starts from 0 and not 1
#         onehot_encoded.append(temp)
#     return onehot_encoded


#vocab = {'dog': 1, 'bites': 2, 'man': 4, 'eats': 2, 'meat': 5, 'food': 3}
#vocab = {'dog': 1, 'bites': 0, 'man': 4, 'eats': 2, 'meat': 5, 'food': 3}
# vocab = {'dog': 3, 'bites': 2, 'man': 5, 'and': 0, 'are': 1, 'friends': 4}
# one_hot_encoded = get_onehot_vector("dog bites man") 
# print(one_hot_encoded)
# # [[1, 0, 0, 0, 0, 0], 
# # [0, 0, 0, 0, 0, 1], 
# # [0, 0, 0, 1, 0, 0]]

# # [[1, 0, 0, 0, 0, 0], 
# # [0, 1, 0, 0, 0, 0], 
# # [0, 0, 0, 1, 0, 0]]


# [[0, 0, 1, 0, 0, 0], 
# [0, 1, 0, 0, 0, 0], 
# [0, 0, 0, 0, 1, 0]]
# using BoW - if a word appear in text nothing to do with frequency

# Label encoder assigns nos to keys in vocab

# processed_docs = ["dog bites man","man bites dog","dog and dog are friends"]

# Our vocabulary:  {'and': 0,'are': 1,'bites': 2,'dog': 3,'friends': 4,'man': 5}
# BoW representation for 'dog bites man':  [[0 0 1 1 0 1]]
# BoW representation for 'man bites dog:  [[0 0 1 1 0 1]]
# Bow representation for 'dog and dog are friends': [[1 1 0 2 1 0]]



# CountVectorizer(ngram_range=(1,3))
#Our vocabulary:  [('bites', 0), ('bites dog', 1), ('bites man', 2), 
# ('dog', 3), ('dog bites', 4), ('dog bites man', 5), ('dog eats', 6), 
# ('dog eats , 7), ('eats', 8), ('eats food', 9), ('eats meat', 10), 
# ('food', 11), ('man', 12), ('man bites', 13), meat'('man bites dog', 14), 
# ('man eats', 15), ('man eats food', 16), ('meat', 17)]


# BoW representation for 'dog bites man':  [[1 0 1 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0]]
# BoW representation for 'man bites dog:  [[1 1 0 1 0 0 0 0 0 0 0 0 1 1 1 0 0 0]]
# Bow representation for 'dog and dog are friends': [[0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]

# Note that the number of features (and hence the size of the feature vector) 
# increased a lot for the same data, compared to the ther single word based 
# representations!!





# Clipping each gradient matrix individually changes their relative scale but 
# is also possible,Despite what seems to be popular, you probably want to clip 
# the whole gradient by its global norm:

# optimizer = tf.train.AdamOptimizer(1e-3)
# gradients, variables = zip(*optimizer.compute_gradients(loss))
# # with tf.GradientTape() as tape:
# #   loss = ...
# # variables = ...
# # gradients = tape.gradient(loss, variables)
# gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
# # gradients = [
# #     None if gradient is None else tf.clip_by_norm(gradient, 5.0)
# #     for gradient in gradients]
# optimize = optimizer.apply_gradients(zip(gradients, variables))



# model = tf.keras.models.Sequential([...])
# model.compile(optimizer=tf.keras.optimizers.SGD(clipvalue=0.5),loss,metrics)
# model.fit(train_data,steps_per_epoch,epochs)

### Tensorflow

### PyTorch

Tf-Idf 
-------
The meaning increases proportionally to the number of times in the 
text a word appears but is compensated by the word frequency in 
the corpus (data-set)

tf(t,d) = count of t in d / number of words in d
df(t) = occurrence of t in documents

df(t) = N(t)
where
df(t) = Document frequency of a term t
N(t) = Number of documents containing the term t
idf(t) = N/ df(t) = N/N(t)


#for strange word embedding representation for full text in spacy gives null vector
temp = nlp('practicalnlp is a newword')
temp[0].vector

#skip-gram else its 0 for CBOW. Default is CBOW.

from gensim.models import Word2Vec, KeyedVectors 

from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize