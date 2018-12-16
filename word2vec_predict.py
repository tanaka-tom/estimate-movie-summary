import gensim
import numpy as np
import csv
from scipy import spatial
import os

class SearchSimilarWords():
    def __init__(self):
        self.num_features = 300

        self.words_array = self.build_words_array()

        self.model = gensim.models.KeyedVectors.load_word2vec_format("model.vec", binary=False)

    def build_words_array(self):
        array = []
        with open("words.csv", 'r') as words_csv_file:
            reader = csv.reader(words_csv_file)
            for row in reader:
                array.append(row)
        return array

    def cal(self, target_index):
        target_words = self.words_array[target_index]
        target_words_avg_vector = self.avg_feature_vector(target_words)
        max_similarity = 0.0
        similar_words = []
        for words in self.words_array:
            if words != target_words:
                words_avg_vector = self.avg_feature_vector(words)
                similarity = self.cal_similarity(
                    target_words_avg_vector, words_avg_vector)
                if max_similarity < similarity:
                    similar_words = words
                    max_similarity = similarity
        return {
            'similarity': max_similarity,
            'target_words': target_words,
            'the_most_similar_words': similar_words
        }

    def cal_similarity(self, target_words_avg_vector, words_avg_vector):
        return 1 - spatial.distance.cosine(target_words_avg_vector, words_avg_vector)

    def avg_feature_vector(self, words):
        feature_vec = np.zeros((self.num_features,), dtype="float32")
        for word in words:
            try:
                feature_vec = np.add(feature_vec, self.model[word])
            except KeyError:
                words.remove(word)
        if len(words) > 0:
            feature_vec = np.divide(feature_vec, len(words))
        return feature_vec
