from nose.tools import assert_equal
from functools import partial
from embeddingvectorizer import EmbeddingCountVectorizer
from embeddingvectorizer.vectorizer import Operator


model = {"test": [1, 0], "document": [0, 1], "mixed": [.5, .5]}


def _vectorize(vectorizer, text):
    return vectorizer.fit_transform([text])[0].tolist()


def test_countvectorizer_mean():
    _t = partial(_vectorize, EmbeddingCountVectorizer(model, operator=Operator.mean))
    assert_equal(_t("test document"), [.5, .5])
    assert_equal(_t("another test"), [1, 0])   # Should unrecognized words count in mean? probably not
    assert_equal(_t("duplicate document duplicate document"), [0, 1])
    assert_equal(_t("test to test a test document"), [.75, .25])
    assert_equal(_t("unknown"), [0, 0])


def test_countvectorizer_max():
    _t = partial(_vectorize, EmbeddingCountVectorizer(model, operator=Operator.max))
    assert_equal(_t("test document"), [1, 1])
    assert_equal(_t("another test"), [1, 0])  # Should unrecognized words count in mean? probably not
    assert_equal(_t("duplicate document duplicate document"), [0, 1])
    assert_equal(_t("test to test a test document"), [1, 1])
    assert_equal(_t("a document and a mixed document"), [0.5, 1])
    assert_equal(_t("a document to test a mixed document (document?)"), [1, 1])
    assert_equal(_t("unknown"), [0, 0])


def test_countvectorizer_sum():
    _t = partial(_vectorize, EmbeddingCountVectorizer(model, operator=Operator.sum))
    assert_equal(_t("test document"), [1, 1])
    assert_equal(_t("another test"), [1, 0])  # Should unrecognized words count in mean? probably not
    assert_equal(_t("duplicate document duplicate document"), [0, 2])
    assert_equal(_t("test to test a test document"), [3, 1])
    assert_equal(_t("a document and a mixed document"), [0.5, 2.5])
    assert_equal(_t("a document to test a mixed document (document?)"), [1.5, 3.5])
    assert_equal(_t("unknown"), [0, 0])

