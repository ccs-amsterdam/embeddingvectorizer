from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class BaseEmbeddingVectorizerMixin:
    """
    Base class for Embedding vectorizer
    Note that we implement this as vectorizer rather than transformer because we need to access the vocabulary.
    """

    def fit_transform(self, X: Iterable[str], y=None):
        x = super().fit_transform(X, y).tocsr()
        return np.array(list(self._transform(x)))

    def transform(self, X: Iterable[str], y=None):
        x = super().transform(X, y).tocsr()
        return np.array(list(self._transform(x)))

    def _transform(self, x):
        dim = len(next(iter(self.word2vec.values())))
        voca = self._get_words()
        for doc in range(len(x.indptr) - 1):
            weights = x.data[x.indptr[doc]:x.indptr[doc + 1]]
            words = x.indices[x.indptr[doc]:x.indptr[doc + 1]]

            vec = [self.word2vec[voca[w]] * np.array(weights[i])
                   for i, w in enumerate(words) if voca[w] in self.word2vec]
            yield np.mean(vec, axis=0) if vec else np.zeros(dim)

    def _get_words(self):
        # pretty ugly!
        result = [None] * len(self.vocabulary_)
        for w, i in self.vocabulary_.items():
            result[i] = w
        return result


class EmbeddingCountVectorizer(BaseEmbeddingVectorizerMixin, CountVectorizer):
    def __init__(self, word2vec, dim=320, **kargs):
        # WvA: apparently we should enumerate all arguments rather than use **kargs,
        # e.g. https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/feature_extraction/text.py#L1493-L1509
        super().__init__(**kargs)
        self.word2vec = word2vec


class EmbeddingTfidfVectorizer(BaseEmbeddingVectorizerMixin, TfidfVectorizer):
    def __init__(self, word2vec, dim=320, **kargs):
        # WvA: apparently we should enumerate all arguments rather than use **kargs,
        # e.g. https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/feature_extraction/text.py#L1493-L1509
        super().__init__(**kargs)
        self.word2vec = word2vec


if __name__ == '__main__':
    texts = ["dit is een text", "en dit is een kat", "en dit is een langere zin met heel veel woorden erin"]
    model = {"dit": [3, 0, 0], "kat": [0, 1, 0], "woorden": [0, 0, 1]}
    print(EmbeddingTfidfVectorizer(model, 3).fit_transform(texts))