# EmbeddingVectorizer
Scikit-learn Vectorizer using word2vec embedding models.

Can do mean, max, and sum weighting by word frequency or tfidf (untested).

On a whole, this modules is almost completely untested, so use at your own risk!

# Installing

```{sh}
$ pip install embeddingvectorizer
```

For development (with local env):

```{sh}
git clone https://github.com/ccs-amsterdam/embeddingvectorizer
cd embeddingvectorizer
python3 -m venv env
env/bin/pip -e .[dev]
env/bin/nosetests
```

# Usage

```{python}
import embeddingvectorizer
model =  {"word": [1, .5, 0]} 
v = embeddingvectorizer.EmbeddingCountVectorizer(model, operator='mean')
v.fit_transform(["My lord, a word!"])
## ==> array([[1. , 0.5, 0. ]])
```
