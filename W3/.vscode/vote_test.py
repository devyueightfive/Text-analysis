# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sklearn
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# %%
class MyTransformer(sklearn.base.TransformerMixin):
    def __init__(self, k_first=1):
        self.k = k_first
        self.cols = None

    def fit(self, X, y):
        n = min(self.k, X.shape[1])
        self.cols = [True] * n + [False] * max(X.shape[1] - self.k, 0)
        return self

    def transform(self, X):
        return X[:, self.cols]


# %%
X = np.array([[1,2,3],[4,5,6], [7,8,9], [10,11,12]])
y = np.array([1,0,1, 1])


# %%
y


# %%
t = MyTransformer(k_first=2)


# %%
t.fit(X,y)


# %%
X


# %%
Z = t.transform(X)
Z


# %%
from sklearn.linear_model import LogisticRegression


# %%
logit = LogisticRegression(C = 3)
logit.fit(Z,y)


# %%
accuracy_score(y, logit.predict(Z))


# %%
from sklearn.ensemble import VotingClassifier


# %%
log_t = Pipeline(
        steps=[
            ('transform', t),
            ('classify', logit)
        ])


# %%
log_t.fit(X,y)


# %%
log_t.predict(X)


# %%
hard = VotingClassifier(estimators=[log_t], weights=[1.0])
hard.weights

#%%
hard.fit(X, y, sample_weight=[1])


# %%

hard.predict(X)


# %%


