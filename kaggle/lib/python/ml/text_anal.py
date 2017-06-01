# -*- coding: utf-8 -*-
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.stem
from sklearn.feature_extraction.text import CountVectorizer
import sys
import scipy.spatial.distance as sp_dist
import scipy.linalg as la

english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
		
class StemmedTfidfVectorizer(TfidfVectorizer):
	def build_analyzer(self):
		analyzer = super(TfidfVectorizer, self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

def create_stemmed_vectorizer():
	vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')
	return vectorizer

def create_tfidf_vectorizer():
	vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english', decode_error='ignore')
	return vectorizer

def find_bestfit_index(vectorized, find_vec, dist_name='norm'):
	best_dist = sys.float_info.max
	bast_idx = -1
	idx = 0
	dist_method = None
	if dist_name == 'cosine':
		print('using cosine method')
		dist_method = sp_dist.cosine;
	elif dist_name == 'norm':
		print('using norm method')
		dist_method = la.norm
	else: 
		print('using default norm method')
		dist_method = la.norm

	for item_vec in vectorized:
		dist = la.norm(find_vec.toarray()[0] - item_vec.toarray()[0])
		if dist < best_dist:
			best_dist = dist
			best_idx = idx
			idx += 1
	return best_idx
			
# SciKit에서 제공하므로 아래 함수 필요 없음
# def tfidf(term, doc, corpus):
#     tf = doc.count(term) / len(doc)
#     num_docs_with_term = len([d for d in corpus if term in d])
#     idf = sp.log(len(corpus) / num_docs_with_term)
#     return tf * idf